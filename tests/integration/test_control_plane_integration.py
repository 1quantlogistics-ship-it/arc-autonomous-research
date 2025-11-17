"""
Integration tests for Control Plane with MemoryHandler.

Tests the control plane's integration with the v1.1.0 memory handler,
schema validation, and config system.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from config import ARCSettings
from memory_handler import MemoryHandler, get_memory_handler, reset_memory_handler
from schemas import (
    Directive, DirectiveMode, Objective,
    SystemState, OperatingMode,
    Constraints, ForbiddenRange
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_arc_env(tmp_path, monkeypatch):
    """Create temporary ARC environment with control plane."""
    # Create settings with temp directory
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    # Initialize memory
    memory = MemoryHandler(settings)
    memory.initialize_memory(force=True)

    # Monkeypatch environment variables
    monkeypatch.setenv("ARC_ENVIRONMENT", "test")
    monkeypatch.setenv("ARC_HOME", str(tmp_path / "arc"))
    monkeypatch.setenv("ARC_LLM_ENDPOINT", "http://localhost:8000/v1")

    # Reset global memory handler to use test settings
    reset_memory_handler()

    yield settings, memory

    # Cleanup
    reset_memory_handler()


@pytest.fixture
def control_plane_client(temp_arc_env):
    """Create a test client for the control plane."""
    settings, memory = temp_arc_env

    # Import control plane (will use our monkeypatched environment)
    from api.control_plane import app

    client = TestClient(app)
    return client


# ============================================================================
# Root Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneRoot:
    """Test control plane root endpoint."""

    def test_root_endpoint(self, control_plane_client):
        """Test GET / returns service info."""
        response = control_plane_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "ARC Control Plane"
        assert data["version"] == "1.1.0"
        assert data["status"] == "operational"


# ============================================================================
# Status Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneStatus:
    """Test status endpoint with schema validation."""

    def test_get_status(self, control_plane_client):
        """Test GET /status returns validated state."""
        response = control_plane_client.get("/status")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "mode" in data
        assert "status" in data
        assert "current_cycle" in data
        assert "current_objective" in data

        # Check mode is valid
        assert data["mode"] in [m.value for m in OperatingMode]

    def test_get_status_with_query(self, control_plane_client):
        """Test GET /status with query filter."""
        response = control_plane_client.get("/status?query=mode")

        assert response.status_code == 200
        data = response.json()

        # Should only return mode field
        assert "mode" in data


# ============================================================================
# Mode Change Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneMode:
    """Test mode change endpoint."""

    def test_set_mode_valid(self, control_plane_client):
        """Test POST /mode with valid mode."""
        response = control_plane_client.post("/mode?mode=AUTO")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "mode_changed"
        assert data["new_mode"] == "AUTO"
        assert data["old_mode"] == "SEMI"

    def test_set_mode_invalid(self, control_plane_client):
        """Test POST /mode with invalid mode."""
        response = control_plane_client.post("/mode?mode=INVALID")

        assert response.status_code == 400
        assert "Invalid mode" in response.json()["detail"]

    def test_mode_change_persists(self, control_plane_client, temp_arc_env):
        """Test that mode changes are persisted to memory."""
        settings, memory = temp_arc_env

        # Change mode
        response = control_plane_client.post("/mode?mode=FULL")
        assert response.status_code == 200

        # Reload state and verify
        state = memory.load_system_state()
        assert state.mode == OperatingMode.FULL


# ============================================================================
# Exec Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneExec:
    """Test exec endpoint with validation."""

    def test_exec_blocked_command(self, control_plane_client):
        """Test that non-allowlisted commands are blocked."""
        request = {
            "command": "rm -rf /",
            "role": "test",
            "cycle_id": 1,
            "requires_approval": False
        }

        response = control_plane_client.post("/exec", json=request)

        assert response.status_code == 400
        assert "not in allowlist" in response.json()["detail"]

    def test_exec_allowed_command(self, control_plane_client):
        """Test executing an allowed command."""
        request = {
            "command": "ls",
            "role": "test",
            "cycle_id": 1,
            "requires_approval": False
        }

        # Set mode to FULL to bypass approval
        control_plane_client.post("/mode?mode=FULL")

        response = control_plane_client.post("/exec", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executed"
        assert "returncode" in data

    def test_exec_requires_approval_in_semi_mode(self, control_plane_client):
        """Test that SEMI mode requires approval."""
        # Ensure SEMI mode
        control_plane_client.post("/mode?mode=SEMI")

        request = {
            "command": "ls",
            "role": "test",
            "cycle_id": 1,
            "requires_approval": True
        }

        response = control_plane_client.post("/exec", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending_approval"


# ============================================================================
# Train Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneTrain:
    """Test training endpoint with constraint validation."""

    def test_train_valid_config(self, control_plane_client):
        """Test training with valid config."""
        request = {
            "experiment_id": "exp_001",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "requires_approval": False
        }

        # Set to FULL mode
        control_plane_client.post("/mode?mode=FULL")

        response = control_plane_client.post("/train", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["experiment_id"] == "exp_001"

    def test_train_violates_constraints(self, control_plane_client, temp_arc_env):
        """Test training with config that violates constraints."""
        settings, memory = temp_arc_env

        # Add constraint
        constraints = memory.load_constraints()
        constraints.forbidden_ranges.append(
            ForbiddenRange(
                param="learning_rate",
                min=0.0001,
                max=0.01,
                reason="Unsafe range for learning rate"
            )
        )
        memory.save_constraints(constraints)

        # Try to train with lr outside safe range
        request = {
            "experiment_id": "exp_002",
            "config": {
                "learning_rate": 0.1  # Above max=0.01
            },
            "requires_approval": False
        }

        control_plane_client.post("/mode?mode=FULL")
        response = control_plane_client.post("/train", json=request)

        assert response.status_code == 400
        assert "validation_failed" in str(response.json()["detail"])

    def test_train_updates_active_experiments(self, control_plane_client, temp_arc_env):
        """Test that training adds to active experiments."""
        settings, memory = temp_arc_env

        request = {
            "experiment_id": "exp_003",
            "config": {"batch_size": 16},
            "requires_approval": False
        }

        # Set to FULL mode
        control_plane_client.post("/mode?mode=FULL")

        response = control_plane_client.post("/train", json=request)
        assert response.status_code == 200

        # Check system state
        state = memory.load_system_state()
        assert len(state.active_experiments) > 0
        assert any(exp.experiment_id == "exp_003" for exp in state.active_experiments)


# ============================================================================
# Archive/Rollback Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneArchiveRollback:
    """Test archive and rollback endpoints."""

    def test_archive_creates_snapshot(self, control_plane_client, temp_arc_env):
        """Test that archive creates a snapshot."""
        settings, memory = temp_arc_env

        request = {
            "cycle_id": 1,
            "reason": "Test snapshot"
        }

        response = control_plane_client.post("/archive", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"
        assert "snapshot_id" in data

        # Verify snapshot exists
        snapshot_id = data["snapshot_id"]
        snapshot_dir = settings.snapshots_dir / snapshot_id
        assert snapshot_dir.exists()
        assert (snapshot_dir / "directive.json").exists()
        assert (snapshot_dir / "metadata.json").exists()

    def test_rollback_restores_memory(self, control_plane_client, temp_arc_env):
        """Test that rollback restores memory state."""
        settings, memory = temp_arc_env

        # Get initial state
        initial_directive = memory.load_directive()
        initial_cycle_id = initial_directive.cycle_id

        # Create snapshot
        archive_response = control_plane_client.post(
            "/archive",
            json={"cycle_id": 1, "reason": "Before modification"}
        )
        snapshot_id = archive_response.json()["snapshot_id"]

        # Modify memory
        directive = memory.load_directive()
        directive.cycle_id = 999
        directive.notes = "Modified"
        memory.save_directive(directive)

        # Verify modification
        modified = memory.load_directive()
        assert modified.cycle_id == 999

        # Rollback
        rollback_response = control_plane_client.post(
            "/rollback",
            json={"snapshot_id": snapshot_id}
        )

        assert rollback_response.status_code == 200

        # Verify restoration
        restored = memory.load_directive()
        assert restored.cycle_id == initial_cycle_id
        assert restored.notes != "Modified"

    def test_rollback_validates_restored_memory(self, control_plane_client, temp_arc_env):
        """Test that rollback validates restored memory."""
        settings, memory = temp_arc_env

        # Create valid snapshot
        archive_response = control_plane_client.post(
            "/archive",
            json={"cycle_id": 1, "reason": "Valid snapshot"}
        )
        snapshot_id = archive_response.json()["snapshot_id"]

        # Corrupt snapshot (make it invalid)
        snapshot_dir = settings.snapshots_dir / snapshot_id
        directive_file = snapshot_dir / "directive.json"
        directive_file.write_text('{"invalid": "schema"}')

        # Try to rollback
        rollback_response = control_plane_client.post(
            "/rollback",
            json={"snapshot_id": snapshot_id}
        )

        # Should fail validation
        assert rollback_response.status_code == 500
        assert "validation_failed" in str(rollback_response.json()["detail"])


# ============================================================================
# Eval Endpoint Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneEval:
    """Test eval endpoint."""

    def test_eval_nonexistent_experiment(self, control_plane_client):
        """Test eval for nonexistent experiment."""
        request = {
            "experiment_id": "nonexistent",
            "metrics": ["auc", "accuracy"]
        }

        response = control_plane_client.post("/eval", json=request)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_eval_existing_experiment(self, control_plane_client, temp_arc_env):
        """Test eval for existing experiment."""
        settings, memory = temp_arc_env

        # Create experiment with results
        exp_dir = settings.experiments_dir / "exp_test"
        exp_dir.mkdir(parents=True)

        results = {
            "auc": 0.95,
            "accuracy": 0.92,
            "f1": 0.93
        }

        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f)

        # Eval
        request = {
            "experiment_id": "exp_test",
            "metrics": ["auc", "accuracy"]
        }

        response = control_plane_client.post("/eval", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "exp_test"
        assert data["metrics"]["auc"] == 0.95
        assert data["metrics"]["accuracy"] == 0.92
        assert "f1" not in data["metrics"]  # Not requested


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestControlPlaneErrorHandling:
    """Test error handling with schema validation."""

    def test_corrupt_memory_returns_500(self, control_plane_client, temp_arc_env):
        """Test that corrupted memory returns 500 error."""
        settings, memory = temp_arc_env

        # Corrupt system state
        state_file = settings.memory_dir / "system_state.json"
        state_file.write_text('{"broken": "schema"}')

        # Try to get status
        response = control_plane_client.get("/status")

        assert response.status_code == 500
        assert "validation error" in response.json()["detail"].lower()
