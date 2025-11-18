"""
Unit tests for Tool Governance Layer.

Tests validation, constraint checking, transactional execution,
and audit logging for Control Plane tools.
"""

import pytest
import json
from pathlib import Path

from tool_governance import (
    ToolGovernance, get_tool_governance, reset_tool_governance,
    ToolValidationError, ToolExecutionError
)
from memory_handler import MemoryHandler
from config import ARCSettings
from schemas import (
    Constraints, ForbiddenRange, SystemState, OperatingMode,
    ActiveExperiment
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_governance(tmp_path):
    """Create tool governance with temporary directory."""
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    memory = MemoryHandler(settings)
    memory.initialize_memory(force=True)

    governance = ToolGovernance(settings=settings, memory=memory)

    yield governance

    reset_tool_governance()


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestToolGovernanceInit:
    """Test tool governance initialization."""

    def test_create_governance(self, temp_governance):
        """Test creating tool governance."""
        assert temp_governance is not None
        assert temp_governance.settings is not None
        assert temp_governance.memory is not None

    def test_get_global_governance(self):
        """Test getting global governance instance."""
        governance1 = get_tool_governance()
        governance2 = get_tool_governance()

        assert governance1 is governance2  # Singleton


# ========================================================================
# Train Request Validation Tests
# ============================================================================

@pytest.mark.unit
class TestTrainRequestValidation:
    """Test training request validation."""

    def test_validate_train_request_valid(self, temp_governance):
        """Test validating a valid training request."""
        train_args = {
            "experiment_id": "exp_001",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        }

        is_valid, error = temp_governance.validate_tool_request("train", train_args)

        assert is_valid
        assert error is None

    def test_validate_train_request_missing_experiment_id(self, temp_governance):
        """Test validation fails when experiment_id missing."""
        train_args = {
            "config": {
                "learning_rate": 0.001
            }
        }

        is_valid, error = temp_governance.validate_tool_request("train", train_args)

        assert not is_valid
        assert "experiment_id" in error

    def test_validate_train_request_violates_constraint(self, temp_governance):
        """Test validation fails when config violates constraints."""
        # Add constraint
        constraints = temp_governance.memory.load_constraints()
        constraints.forbidden_ranges.append(
            ForbiddenRange(
                param="learning_rate",
                min=0.0001,
                max=0.01,
                reason="Unsafe range"
            )
        )
        temp_governance.memory.save_constraints(constraints)

        # Try to train with lr outside range
        train_args = {
            "experiment_id": "exp_002",
            "config": {
                "learning_rate": 0.1  # Above max=0.01
            }
        }

        is_valid, error = temp_governance.validate_tool_request("train", train_args)

        assert not is_valid
        assert "learning_rate" in error
        assert "above safe range" in error


# ============================================================================
# Exec Request Validation Tests
# ============================================================================

@pytest.mark.unit
class TestExecRequestValidation:
    """Test exec request validation."""

    def test_validate_exec_request_valid(self, temp_governance):
        """Test validating a valid exec request."""
        exec_args = {
            "command": "ls -la",
            "role": "test",
            "cycle_id": 1
        }

        is_valid, error = temp_governance.validate_tool_request("exec", exec_args)

        assert is_valid
        assert error is None

    def test_validate_exec_request_blocked_command(self, temp_governance):
        """Test validation fails for blocked command."""
        exec_args = {
            "command": "rm -rf /",
            "role": "test",
            "cycle_id": 1
        }

        is_valid, error = temp_governance.validate_tool_request("exec", exec_args)

        assert not is_valid
        assert "not in allowlist" in error

    def test_validate_exec_request_missing_command(self, temp_governance):
        """Test validation fails when command missing."""
        exec_args = {
            "role": "test",
            "cycle_id": 1
        }

        is_valid, error = temp_governance.validate_tool_request("exec", exec_args)

        assert not is_valid
        assert "command" in error


# ============================================================================
# Eval Request Validation Tests
# ============================================================================

@pytest.mark.unit
class TestEvalRequestValidation:
    """Test eval request validation."""

    def test_validate_eval_request_valid(self, temp_governance):
        """Test validating a valid eval request."""
        eval_args = {
            "experiment_id": "exp_001",
            "metrics": ["auc", "accuracy"]
        }

        is_valid, error = temp_governance.validate_tool_request("eval", eval_args)

        assert is_valid
        assert error is None

    def test_validate_eval_request_missing_experiment_id(self, temp_governance):
        """Test validation fails when experiment_id missing."""
        eval_args = {
            "metrics": ["auc"]
        }

        is_valid, error = temp_governance.validate_tool_request("eval", eval_args)

        assert not is_valid
        assert "experiment_id" in error

    def test_validate_eval_request_missing_metrics(self, temp_governance):
        """Test validation fails when metrics missing."""
        eval_args = {
            "experiment_id": "exp_001"
        }

        is_valid, error = temp_governance.validate_tool_request("eval", eval_args)

        assert not is_valid
        assert "metrics" in error


# ============================================================================
# Mode Permission Tests
# ============================================================================

@pytest.mark.unit
class TestModePermissions:
    """Test mode permission checks."""

    def test_check_permission_semi_mode(self, temp_governance):
        """Test that SEMI mode requires approval."""
        # Set SEMI mode
        state = temp_governance.memory.load_system_state()
        state.mode = OperatingMode.SEMI
        temp_governance.memory.save_system_state(state)

        is_allowed, message = temp_governance.check_mode_permission("train", requires_approval=True)

        assert not is_allowed
        assert "approval" in message.lower()

    def test_check_permission_auto_mode_blocks_train(self, temp_governance):
        """Test that AUTO mode blocks training."""
        # Set AUTO mode
        state = temp_governance.memory.load_system_state()
        state.mode = OperatingMode.AUTO
        temp_governance.memory.save_system_state(state)

        is_allowed, message = temp_governance.check_mode_permission("train")

        assert not is_allowed
        assert "training" in message.lower()

    def test_check_permission_full_mode_allows_all(self, temp_governance):
        """Test that FULL mode allows all tools."""
        # Set FULL mode
        state = temp_governance.memory.load_system_state()
        state.mode = OperatingMode.FULL
        temp_governance.memory.save_system_state(state)

        is_allowed, message = temp_governance.check_mode_permission("train")

        assert is_allowed
        assert message is None


# ============================================================================
# Transactional Execution Tests
# ============================================================================

@pytest.mark.unit
class TestTransactionalExecution:
    """Test transactional tool execution."""

    def test_tool_transaction_success(self, temp_governance):
        """Test successful transaction."""
        # Modify memory in transaction
        with temp_governance.tool_transaction("test_tool", cycle_id=1):
            state = temp_governance.memory.load_system_state()
            state.status = "running"
            temp_governance.memory.save_system_state(state)

        # Changes should be committed
        loaded = temp_governance.memory.load_system_state()
        assert loaded.status == "running"

    def test_tool_transaction_rollback(self, temp_governance):
        """Test transaction rollback on error."""
        # Get initial state
        initial_state = temp_governance.memory.load_system_state()
        initial_status = initial_state.status

        # Try transaction that fails
        try:
            with temp_governance.tool_transaction("test_tool", cycle_id=1):
                state = temp_governance.memory.load_system_state()
                state.status = "running"
                temp_governance.memory.save_system_state(state)
                raise Exception("Simulated error")
        except Exception:
            pass

        # Changes should be rolled back
        loaded = temp_governance.memory.load_system_state()
        assert loaded.status == initial_status

    def test_execute_with_rollback_success(self, temp_governance):
        """Test execute_with_rollback with success."""
        tool_args = {
            "experiment_id": "exp_001",
            "config": {"learning_rate": 0.001}
        }

        # Set FULL mode to allow execution
        state = temp_governance.memory.load_system_state()
        state.mode = OperatingMode.FULL
        temp_governance.memory.save_system_state(state)

        result = temp_governance.execute_with_rollback(
            tool_name="train",
            tool_args=tool_args,
            cycle_id=1,
            role="test",
            execution_callback=lambda: {"status": "success"}
        )

        assert result["status"] == "success"

    def test_execute_with_rollback_failure(self, temp_governance):
        """Test execute_with_rollback with failure and rollback."""
        # Get initial state
        initial_directive = temp_governance.memory.load_directive()
        initial_cycle_id = initial_directive.cycle_id

        tool_args = {
            "experiment_id": "exp_002",
            "config": {"learning_rate": 0.001}
        }

        # Set FULL mode
        state = temp_governance.memory.load_system_state()
        state.mode = OperatingMode.FULL
        temp_governance.memory.save_system_state(state)

        # Execution that fails
        def failing_execution():
            # Modify memory
            directive = temp_governance.memory.load_directive()
            directive.cycle_id = 999
            temp_governance.memory.save_directive(directive)

            # Fail
            raise Exception("Execution failed")

        with pytest.raises(ToolExecutionError):
            temp_governance.execute_with_rollback(
                tool_name="train",
                tool_args=tool_args,
                cycle_id=1,
                role="test",
                execution_callback=failing_execution
            )

        # Verify rollback
        loaded_directive = temp_governance.memory.load_directive()
        assert loaded_directive.cycle_id == initial_cycle_id


# ============================================================================
# Audit Logging Tests
# ============================================================================

@pytest.mark.unit
class TestAuditLogging:
    """Test audit trail logging."""

    def test_log_tool_execution(self, temp_governance):
        """Test logging tool execution."""
        temp_governance._log_tool_execution(
            tool_name="train",
            tool_args={"experiment_id": "exp_001"},
            cycle_id=1,
            role="architect",
            status="started"
        )

        # Verify log exists
        assert temp_governance.audit_log_path.exists()

        # Read log
        with open(temp_governance.audit_log_path, 'r') as f:
            log_entry = json.loads(f.readline())

        assert log_entry["tool"] == "train"
        assert log_entry["cycle_id"] == 1
        assert log_entry["status"] == "started"

    def test_get_audit_trail(self, temp_governance):
        """Test getting audit trail."""
        # Log multiple executions
        for i in range(5):
            temp_governance._log_tool_execution(
                tool_name="train",
                tool_args={"experiment_id": f"exp_{i:03d}"},
                cycle_id=i,
                role="test",
                status="success"
            )

        # Get all entries
        trail = temp_governance.get_audit_trail()
        assert len(trail) == 5

        # Get entries for specific cycle
        cycle_trail = temp_governance.get_audit_trail(cycle_id=2)
        assert len(cycle_trail) == 1
        assert cycle_trail[0]["cycle_id"] == 2


# ============================================================================
# Resource Limit Tests
# ============================================================================

@pytest.mark.unit
class TestResourceLimits:
    """Test resource limit checks."""

    def test_check_resource_limits_within_limit(self, temp_governance):
        """Test resource limits when within bounds."""
        within_limits, error = temp_governance.check_resource_limits()

        assert within_limits
        assert error is None

    def test_check_resource_limits_exceeded(self, temp_governance):
        """Test resource limits when exceeded."""
        # Add many active experiments
        state = temp_governance.memory.load_system_state()

        for i in range(15):  # More than max (10)
            state.active_experiments.append(
                ActiveExperiment(
                    experiment_id=f"exp_{i:03d}",
                    status="running",
                    started_at="2025-11-16T00:00:00Z"
                )
            )

        temp_governance.memory.save_system_state(state)

        within_limits, error = temp_governance.check_resource_limits()

        assert not within_limits
        assert "limit reached" in error
