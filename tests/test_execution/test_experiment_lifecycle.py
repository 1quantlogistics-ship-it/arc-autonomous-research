"""
Tests for Experiment Lifecycle System.

Dev 2 - Bulletproof Execution: Tests for ExperimentState, ExperimentRecord,
and ExperimentRegistry.

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from execution.experiment_lifecycle import (
    ExperimentState,
    ExperimentRecord,
    ExperimentRegistry,
    get_experiment_registry,
    reset_experiment_registry,
    VALID_TRANSITIONS,
)


class TestExperimentState:
    """Test ExperimentState enum."""

    def test_state_values(self):
        """Test all state values exist."""
        assert ExperimentState.PENDING.value == "pending"
        assert ExperimentState.QUEUED.value == "queued"
        assert ExperimentState.RUNNING.value == "running"
        assert ExperimentState.PAUSED.value == "paused"
        assert ExperimentState.COMPLETED.value == "completed"
        assert ExperimentState.FAILED.value == "failed"
        assert ExperimentState.CRASHED.value == "crashed"
        assert ExperimentState.TIMEOUT.value == "timeout"
        assert ExperimentState.CANCELLED.value == "cancelled"

    def test_is_terminal(self):
        """Test terminal state detection."""
        # Terminal states
        assert ExperimentState.COMPLETED.is_terminal is True
        assert ExperimentState.FAILED.is_terminal is True
        assert ExperimentState.CRASHED.is_terminal is True
        assert ExperimentState.TIMEOUT.is_terminal is True
        assert ExperimentState.CANCELLED.is_terminal is True

        # Non-terminal states
        assert ExperimentState.PENDING.is_terminal is False
        assert ExperimentState.QUEUED.is_terminal is False
        assert ExperimentState.RUNNING.is_terminal is False
        assert ExperimentState.PAUSED.is_terminal is False

    def test_is_active(self):
        """Test active state detection."""
        # Active states
        assert ExperimentState.PENDING.is_active is True
        assert ExperimentState.QUEUED.is_active is True
        assert ExperimentState.RUNNING.is_active is True
        assert ExperimentState.PAUSED.is_active is True

        # Inactive states
        assert ExperimentState.COMPLETED.is_active is False
        assert ExperimentState.FAILED.is_active is False
        assert ExperimentState.CRASHED.is_active is False


class TestExperimentRecord:
    """Test ExperimentRecord dataclass."""

    def test_create_record(self):
        """Test creating an experiment record."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        assert record.experiment_id == "exp_001"
        assert record.cycle_id == 1
        assert record.proposal_id == "prop_001"
        assert record.state == ExperimentState.PENDING
        assert record.state_history == []
        assert record.created_at is not None

    def test_valid_transition(self):
        """Test valid state transition."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        # PENDING -> QUEUED (valid)
        result = record.transition_to(ExperimentState.QUEUED, "Scheduled for execution")

        assert result is True
        assert record.state == ExperimentState.QUEUED
        assert record.queued_at is not None
        assert len(record.state_history) == 1
        assert record.state_history[0]["from_state"] == "pending"
        assert record.state_history[0]["to_state"] == "queued"
        assert record.state_history[0]["reason"] == "Scheduled for execution"

    def test_invalid_transition(self):
        """Test invalid state transition."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        # PENDING -> COMPLETED (invalid - must go through QUEUED -> RUNNING first)
        result = record.transition_to(ExperimentState.COMPLETED)

        assert result is False
        assert record.state == ExperimentState.PENDING
        assert len(record.state_history) == 0

    def test_full_lifecycle(self):
        """Test full experiment lifecycle."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        # PENDING -> QUEUED -> RUNNING -> COMPLETED
        assert record.transition_to(ExperimentState.QUEUED) is True
        assert record.transition_to(ExperimentState.RUNNING) is True
        assert record.started_at is not None
        assert record.transition_to(ExperimentState.COMPLETED) is True
        assert record.completed_at is not None

        assert len(record.state_history) == 3
        assert record.state == ExperimentState.COMPLETED

    def test_duration_calculation(self):
        """Test duration calculation."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        # Not started yet
        assert record.duration_seconds is None

        # Start
        record.transition_to(ExperimentState.QUEUED)
        record.transition_to(ExperimentState.RUNNING)

        # Duration should be > 0 while running
        import time
        time.sleep(0.1)
        assert record.duration_seconds > 0

    def test_to_dict(self):
        """Test serialization to dict."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
            config={"learning_rate": 0.001},
            metrics={"auc": 0.85},
        )

        data = record.to_dict()

        assert data["experiment_id"] == "exp_001"
        assert data["cycle_id"] == 1
        assert data["state"] == "pending"  # String, not enum
        assert data["config"]["learning_rate"] == 0.001
        assert data["metrics"]["auc"] == 0.85

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "experiment_id": "exp_001",
            "cycle_id": 1,
            "proposal_id": "prop_001",
            "state": "running",
            "created_at": "2025-01-01T00:00:00",
            "config": {"learning_rate": 0.001},
        }

        record = ExperimentRecord.from_dict(data)

        assert record.experiment_id == "exp_001"
        assert record.state == ExperimentState.RUNNING
        assert record.config["learning_rate"] == 0.001


class TestExperimentRegistry:
    """Test ExperimentRegistry class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def registry(self, temp_storage):
        """Create registry with temp storage."""
        reset_experiment_registry()
        return ExperimentRegistry(storage_path=temp_storage)

    def test_register_experiment(self, registry):
        """Test registering an experiment."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        registered = registry.register(record)

        assert registered.experiment_id == "exp_001"
        assert registry.get("exp_001") is not None

    def test_register_duplicate_fails(self, registry):
        """Test registering duplicate experiment fails."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )

        registry.register(record)

        with pytest.raises(ValueError):
            registry.register(record)

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent experiment."""
        result = registry.get("nonexistent")
        assert result is None

    def test_update_state(self, registry):
        """Test updating experiment state."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )
        registry.register(record)

        result = registry.update_state(
            "exp_001",
            ExperimentState.QUEUED,
            reason="Starting execution",
        )

        assert result is True
        updated = registry.get("exp_001")
        assert updated.state == ExperimentState.QUEUED

    def test_update_state_with_kwargs(self, registry):
        """Test updating state with additional fields."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )
        registry.register(record)

        # Move to running and add PID
        registry.update_state("exp_001", ExperimentState.QUEUED)
        registry.update_state(
            "exp_001",
            ExperimentState.RUNNING,
            pid=12345,
            gpu_id=0,
        )

        updated = registry.get("exp_001")
        assert updated.pid == 12345
        assert updated.gpu_id == 0

    def test_update_metrics(self, registry):
        """Test updating experiment metrics."""
        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )
        registry.register(record)

        # Add metrics
        registry.update_metrics("exp_001", {"auc": 0.85, "loss": 0.15})

        updated = registry.get("exp_001")
        assert updated.metrics["auc"] == 0.85
        assert updated.metrics["loss"] == 0.15

        # Merge more metrics
        registry.update_metrics("exp_001", {"accuracy": 0.9})

        updated = registry.get("exp_001")
        assert "auc" in updated.metrics  # Still there
        assert updated.metrics["accuracy"] == 0.9

    def test_get_by_state(self, registry):
        """Test getting experiments by state."""
        # Register experiments in different states
        for i in range(3):
            record = ExperimentRecord(
                experiment_id=f"exp_{i:03d}",
                cycle_id=1,
                proposal_id=f"prop_{i:03d}",
            )
            registry.register(record)

        # Move some to different states
        registry.update_state("exp_001", ExperimentState.QUEUED)
        registry.update_state("exp_002", ExperimentState.QUEUED)
        registry.update_state("exp_002", ExperimentState.RUNNING)

        pending = registry.get_by_state(ExperimentState.PENDING)
        queued = registry.get_by_state(ExperimentState.QUEUED)
        running = registry.get_by_state(ExperimentState.RUNNING)

        assert len(pending) == 1
        assert len(queued) == 1
        assert len(running) == 1

    def test_get_by_cycle(self, registry):
        """Test getting experiments by cycle."""
        # Cycle 1 experiments
        for i in range(2):
            registry.register(ExperimentRecord(
                experiment_id=f"exp_c1_{i}",
                cycle_id=1,
                proposal_id=f"prop_{i}",
            ))

        # Cycle 2 experiments
        for i in range(3):
            registry.register(ExperimentRecord(
                experiment_id=f"exp_c2_{i}",
                cycle_id=2,
                proposal_id=f"prop_{i}",
            ))

        cycle1_exps = registry.get_by_cycle(1)
        cycle2_exps = registry.get_by_cycle(2)

        assert len(cycle1_exps) == 2
        assert len(cycle2_exps) == 3

    def test_get_active(self, registry):
        """Test getting active experiments."""
        # Register and progress experiments
        for i in range(4):
            registry.register(ExperimentRecord(
                experiment_id=f"exp_{i:03d}",
                cycle_id=1,
                proposal_id=f"prop_{i:03d}",
            ))

        # exp_000: PENDING (active)
        # exp_001: QUEUED (active)
        registry.update_state("exp_001", ExperimentState.QUEUED)
        # exp_002: RUNNING (active)
        registry.update_state("exp_002", ExperimentState.QUEUED)
        registry.update_state("exp_002", ExperimentState.RUNNING)
        # exp_003: COMPLETED (not active)
        registry.update_state("exp_003", ExperimentState.QUEUED)
        registry.update_state("exp_003", ExperimentState.RUNNING)
        registry.update_state("exp_003", ExperimentState.COMPLETED)

        active = registry.get_active()

        assert len(active) == 3  # PENDING, QUEUED, RUNNING

    def test_state_callback(self, registry):
        """Test state change callback."""
        callback_calls = []

        def on_state_change(record, old_state, new_state):
            callback_calls.append({
                "experiment_id": record.experiment_id,
                "old": old_state,
                "new": new_state,
            })

        registry.add_state_callback(on_state_change)

        record = ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        )
        registry.register(record)

        registry.update_state("exp_001", ExperimentState.QUEUED)
        registry.update_state("exp_001", ExperimentState.RUNNING)

        assert len(callback_calls) == 2
        assert callback_calls[0]["old"] == ExperimentState.PENDING
        assert callback_calls[0]["new"] == ExperimentState.QUEUED
        assert callback_calls[1]["old"] == ExperimentState.QUEUED
        assert callback_calls[1]["new"] == ExperimentState.RUNNING

    def test_get_stats(self, registry):
        """Test registry statistics."""
        # Register experiments in different states
        for i in range(5):
            registry.register(ExperimentRecord(
                experiment_id=f"exp_{i:03d}",
                cycle_id=1,
                proposal_id=f"prop_{i:03d}",
            ))

        # Move to different states
        registry.update_state("exp_001", ExperimentState.QUEUED)
        registry.update_state("exp_001", ExperimentState.RUNNING)
        registry.update_state("exp_001", ExperimentState.COMPLETED)

        registry.update_state("exp_002", ExperimentState.QUEUED)
        registry.update_state("exp_002", ExperimentState.RUNNING)
        registry.update_state("exp_002", ExperimentState.FAILED)

        stats = registry.get_stats()

        assert stats["total_experiments"] == 5
        assert stats["completed_experiments"] == 1
        assert stats["failed_experiments"] == 1
        assert stats["active_experiments"] == 3  # PENDING: 3


class TestPersistence:
    """Test persistence and recovery."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_persistence_on_register(self, temp_storage):
        """Test experiments are persisted on registration."""
        registry = ExperimentRegistry(storage_path=temp_storage)

        registry.register(ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        ))

        # Check file exists
        filepath = Path(temp_storage) / "exp_001.json"
        assert filepath.exists()

        # Verify content
        with open(filepath) as f:
            data = json.load(f)

        assert data["experiment_id"] == "exp_001"
        assert data["state"] == "pending"

    def test_persistence_on_update(self, temp_storage):
        """Test experiments are persisted on update."""
        registry = ExperimentRegistry(storage_path=temp_storage)

        registry.register(ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        ))

        registry.update_state("exp_001", ExperimentState.QUEUED)

        # Verify updated content
        filepath = Path(temp_storage) / "exp_001.json"
        with open(filepath) as f:
            data = json.load(f)

        assert data["state"] == "queued"

    def test_load_on_startup(self, temp_storage):
        """Test experiments are loaded on startup."""
        # Create registry and add experiment
        registry1 = ExperimentRegistry(storage_path=temp_storage)
        registry1.register(ExperimentRecord(
            experiment_id="exp_001",
            cycle_id=1,
            proposal_id="prop_001",
        ))
        registry1.update_state("exp_001", ExperimentState.QUEUED)
        registry1.update_state("exp_001", ExperimentState.RUNNING)
        registry1.update_state("exp_001", ExperimentState.COMPLETED)

        # Create new registry - should load from disk
        registry2 = ExperimentRegistry(storage_path=temp_storage)

        record = registry2.get("exp_001")
        assert record is not None
        assert record.state == ExperimentState.COMPLETED

    def test_recover_incomplete_on_startup(self, temp_storage):
        """Test incomplete experiments are recovered as crashed."""
        # Create experiment file manually with RUNNING state
        exp_data = {
            "experiment_id": "exp_001",
            "cycle_id": 1,
            "proposal_id": "prop_001",
            "state": "running",
            "state_history": [],
            "created_at": datetime.now().isoformat(),
            "config": {},
            "metrics": {},
            "metadata": {},
        }

        filepath = Path(temp_storage) / "exp_001.json"
        with open(filepath, "w") as f:
            json.dump(exp_data, f)

        # Create registry - should recover
        registry = ExperimentRegistry(storage_path=temp_storage)

        record = registry.get("exp_001")
        assert record.state == ExperimentState.CRASHED
        assert len(record.state_history) == 1
        assert "Recovered on startup" in record.state_history[0]["reason"]


class TestSingleton:
    """Test singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_experiment_registry()
        yield
        reset_experiment_registry()

    def test_get_same_instance(self):
        """Test get_experiment_registry returns same instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            instance1 = get_experiment_registry(storage_path=temp_dir)
            instance2 = get_experiment_registry()

            assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test reset creates new instance."""
        with tempfile.TemporaryDirectory() as temp_dir1:
            instance1 = get_experiment_registry(storage_path=temp_dir1)
            reset_experiment_registry()

            with tempfile.TemporaryDirectory() as temp_dir2:
                instance2 = get_experiment_registry(storage_path=temp_dir2)

                assert instance1 is not instance2


class TestValidTransitions:
    """Test valid state transitions."""

    def test_terminal_states_have_no_transitions(self):
        """Verify terminal states have no valid transitions."""
        terminal_states = [
            ExperimentState.COMPLETED,
            ExperimentState.FAILED,
            ExperimentState.CRASHED,
            ExperimentState.TIMEOUT,
            ExperimentState.CANCELLED,
        ]

        for state in terminal_states:
            assert VALID_TRANSITIONS[state] == [], f"{state} should have no transitions"

    def test_pending_transitions(self):
        """Test PENDING state transitions."""
        valid = VALID_TRANSITIONS[ExperimentState.PENDING]
        assert ExperimentState.QUEUED in valid
        assert ExperimentState.CANCELLED in valid
        assert ExperimentState.RUNNING not in valid  # Must go through QUEUED

    def test_running_transitions(self):
        """Test RUNNING state transitions."""
        valid = VALID_TRANSITIONS[ExperimentState.RUNNING]
        assert ExperimentState.COMPLETED in valid
        assert ExperimentState.FAILED in valid
        assert ExperimentState.CRASHED in valid
        assert ExperimentState.TIMEOUT in valid
        assert ExperimentState.PAUSED in valid
        assert ExperimentState.CANCELLED in valid
