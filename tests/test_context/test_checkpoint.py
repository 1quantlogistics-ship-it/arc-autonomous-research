"""
Tests for Checkpoint Manager.

Phase G - Crash recovery is the critical path.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime

from context.checkpoint_manager import (
    CycleCheckpoint,
    CheckpointManager,
    get_checkpoint_manager,
    reset_checkpoint_manager,
)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create a CheckpointManager with temp directory."""
    reset_checkpoint_manager()
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir, max_checkpoints=5)


class TestCycleCheckpoint:
    """Tests for CycleCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """CycleCheckpoint should create with all fields."""
        checkpoint = CycleCheckpoint(
            cycle_id=1,
            phase="planning",
            agent_states={"proposer": {"status": "ready"}},
            memory_snapshot={"directive": {"mode": "explore"}}
        )

        assert checkpoint.cycle_id == 1
        assert checkpoint.phase == "planning"
        assert checkpoint.agent_states == {"proposer": {"status": "ready"}}
        assert checkpoint.memory_snapshot == {"directive": {"mode": "explore"}}
        assert checkpoint.timestamp is not None
        assert checkpoint.checkpoint_id.startswith("ckpt_1_planning_")

    def test_checkpoint_id_generation(self):
        """Checkpoint ID should be auto-generated."""
        checkpoint = CycleCheckpoint(
            cycle_id=5,
            phase="executing",
            agent_states={},
            memory_snapshot={}
        )

        assert checkpoint.checkpoint_id.startswith("ckpt_5_executing_")
        assert len(checkpoint.checkpoint_id) > 20

    def test_checkpoint_to_dict(self):
        """to_dict should return serializable dict."""
        checkpoint = CycleCheckpoint(
            cycle_id=1,
            phase="voting",
            agent_states={"agent1": "state1"},
            memory_snapshot={"key": "value"}
        )

        d = checkpoint.to_dict()
        assert d["cycle_id"] == 1
        assert d["phase"] == "voting"
        assert d["agent_states"] == {"agent1": "state1"}
        assert "timestamp" in d
        assert "checkpoint_id" in d

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_checkpoint_from_dict(self):
        """from_dict should recreate checkpoint."""
        original = CycleCheckpoint(
            cycle_id=3,
            phase="reviewing",
            agent_states={"critic": {"approved": True}},
            memory_snapshot={"results": [1, 2, 3]}
        )

        d = original.to_dict()
        restored = CycleCheckpoint.from_dict(d)

        assert restored.cycle_id == original.cycle_id
        assert restored.phase == original.phase
        assert restored.agent_states == original.agent_states
        assert restored.checkpoint_id == original.checkpoint_id

    def test_checkpoint_age(self):
        """age_seconds should calculate correctly."""
        checkpoint = CycleCheckpoint(
            cycle_id=1,
            phase="planning",
            agent_states={},
            memory_snapshot={}
        )

        # Age should be very small for fresh checkpoint
        assert checkpoint.age_seconds < 1.0

        # Sleep briefly and check age increased
        time.sleep(0.1)
        assert checkpoint.age_seconds >= 0.1


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_manager_creation(self, temp_checkpoint_dir):
        """Manager should initialize correctly."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=10
        )

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.max_checkpoints == 10
        assert temp_checkpoint_dir.exists()

    def test_save_checkpoint(self, checkpoint_manager, temp_checkpoint_dir):
        """save_checkpoint should create file atomically."""
        path = checkpoint_manager.save_checkpoint(
            cycle_id=1,
            phase="planning",
            agent_states={"test": "state"},
            memory_snapshot={"mem": "snap"}
        )

        assert path.exists()
        assert path.suffix == ".json"
        assert "ckpt_1_planning_" in path.name

        # No temp files should remain
        temp_files = list(temp_checkpoint_dir.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_save_checkpoint_with_metadata(self, checkpoint_manager):
        """Metadata should be saved with checkpoint."""
        path = checkpoint_manager.save_checkpoint(
            cycle_id=2,
            phase="voting",
            agent_states={},
            memory_snapshot={},
            metadata={"version": "2.0", "custom": "data"}
        )

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["version"] == "2.0"
        assert data["metadata"]["custom"] == "data"

    def test_invalid_phase_rejected(self, checkpoint_manager):
        """Invalid phase should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid phase"):
            checkpoint_manager.save_checkpoint(
                cycle_id=1,
                phase="invalid_phase",
                agent_states={},
                memory_snapshot={}
            )

    def test_load_checkpoint(self, checkpoint_manager):
        """load_checkpoint should retrieve saved checkpoint."""
        checkpoint_manager.save_checkpoint(
            cycle_id=1,
            phase="executing",
            agent_states={"executor": "running"},
            memory_snapshot={"config": {"lr": 0.001}}
        )

        # Get the checkpoint ID from the saved file
        checkpoints = checkpoint_manager.list_checkpoints()
        checkpoint_id = checkpoints[0]["checkpoint_id"]

        loaded = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded is not None
        assert loaded.cycle_id == 1
        assert loaded.phase == "executing"
        assert loaded.agent_states == {"executor": "running"}

    def test_load_latest_checkpoint(self, checkpoint_manager):
        """load_latest_checkpoint should return most recent."""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="planning",
            agent_states={}, memory_snapshot={}
        )
        time.sleep(0.05)
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="voting",
            agent_states={}, memory_snapshot={}
        )
        time.sleep(0.05)
        checkpoint_manager.save_checkpoint(
            cycle_id=2, phase="planning",
            agent_states={"latest": True}, memory_snapshot={}
        )

        latest = checkpoint_manager.load_latest_checkpoint()

        assert latest is not None
        assert latest.cycle_id == 2
        assert latest.phase == "planning"
        assert latest.agent_states == {"latest": True}

    def test_load_latest_no_checkpoints(self, temp_checkpoint_dir):
        """load_latest should return None when no checkpoints exist."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.load_latest_checkpoint() is None

    def test_get_recovery_point(self, checkpoint_manager):
        """get_recovery_point should find best checkpoint for cycle."""
        # Save checkpoints for multiple cycles
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="planning", agent_states={}, memory_snapshot={}
        )
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="voting", agent_states={}, memory_snapshot={}
        )
        checkpoint_manager.save_checkpoint(
            cycle_id=2, phase="planning", agent_states={}, memory_snapshot={}
        )

        # Recovery point for cycle 1 should be the voting checkpoint
        recovery = checkpoint_manager.get_recovery_point(1)
        assert recovery is not None
        assert recovery.cycle_id == 1
        assert recovery.phase == "voting"  # Most recent for cycle 1

        # Recovery point for cycle 2 should be its checkpoint
        recovery2 = checkpoint_manager.get_recovery_point(2)
        assert recovery2 is not None
        assert recovery2.cycle_id == 2

    def test_get_recovery_point_fallback(self, checkpoint_manager):
        """get_recovery_point should fall back to earlier cycle if none exists."""
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="reviewing", agent_states={}, memory_snapshot={}
        )

        # No checkpoint for cycle 5, should fall back to cycle 1
        recovery = checkpoint_manager.get_recovery_point(5)
        assert recovery is not None
        assert recovery.cycle_id == 1

    def test_max_checkpoints_cleanup(self, temp_checkpoint_dir):
        """Old checkpoints should be cleaned up beyond max."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=3
        )

        # Save 5 checkpoints
        for i in range(5):
            manager.save_checkpoint(
                cycle_id=i, phase="planning",
                agent_states={}, memory_snapshot={}
            )
            time.sleep(0.05)  # Ensure different timestamps

        # Should only have 3 checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Should have the newest 3 (cycles 2, 3, 4)
        cycle_ids = [c["cycle_id"] for c in checkpoints]
        assert 0 not in cycle_ids
        assert 1 not in cycle_ids
        assert 4 in cycle_ids

    def test_list_checkpoints(self, checkpoint_manager):
        """list_checkpoints should return all with metadata."""
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="planning", agent_states={}, memory_snapshot={}
        )
        checkpoint_manager.save_checkpoint(
            cycle_id=2, phase="voting", agent_states={}, memory_snapshot={}
        )

        checkpoints = checkpoint_manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert all("checkpoint_id" in c for c in checkpoints)
        assert all("cycle_id" in c for c in checkpoints)
        assert all("phase" in c for c in checkpoints)
        assert all("timestamp" in c for c in checkpoints)
        assert all("file_size" in c for c in checkpoints)

    def test_delete_checkpoint(self, checkpoint_manager):
        """delete_checkpoint should remove specific checkpoint."""
        checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="planning", agent_states={}, memory_snapshot={}
        )
        checkpoint_manager.save_checkpoint(
            cycle_id=2, phase="planning", agent_states={}, memory_snapshot={}
        )

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 2

        # Delete first checkpoint
        deleted = checkpoint_manager.delete_checkpoint(checkpoints[0]["checkpoint_id"])
        assert deleted is True

        # Verify only one remains
        remaining = checkpoint_manager.list_checkpoints()
        assert len(remaining) == 1

    def test_delete_nonexistent_checkpoint(self, checkpoint_manager):
        """delete_checkpoint should return False for missing checkpoint."""
        result = checkpoint_manager.delete_checkpoint("nonexistent_id")
        assert result is False

    def test_clear_all_checkpoints(self, checkpoint_manager):
        """clear_all_checkpoints should remove everything."""
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                cycle_id=i, phase="planning",
                agent_states={}, memory_snapshot={}
            )

        assert len(checkpoint_manager.list_checkpoints()) == 3

        deleted = checkpoint_manager.clear_all_checkpoints()
        assert deleted == 3
        assert len(checkpoint_manager.list_checkpoints()) == 0

    def test_checkpoint_stats(self, checkpoint_manager):
        """get_checkpoint_stats should return useful statistics."""
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                cycle_id=i, phase="planning",
                agent_states={"data": "x" * 100},
                memory_snapshot={}
            )

        stats = checkpoint_manager.get_checkpoint_stats()

        assert stats["count"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["oldest"] is not None
        assert stats["newest"] is not None
        assert len(stats["cycles_covered"]) == 3


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_checkpoint_manager_singleton(self, temp_checkpoint_dir):
        """get_checkpoint_manager should return same instance."""
        reset_checkpoint_manager()

        manager1 = get_checkpoint_manager(checkpoint_dir=temp_checkpoint_dir)
        manager2 = get_checkpoint_manager()

        assert manager1 is manager2

    def test_reset_checkpoint_manager(self, temp_checkpoint_dir):
        """reset_checkpoint_manager should clear singleton."""
        manager1 = get_checkpoint_manager(checkpoint_dir=temp_checkpoint_dir)
        reset_checkpoint_manager()
        manager2 = get_checkpoint_manager(checkpoint_dir=temp_checkpoint_dir)

        assert manager1 is not manager2


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_no_temp_files_on_success(self, checkpoint_manager, temp_checkpoint_dir):
        """Successful saves should not leave temp files."""
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                cycle_id=i, phase="planning",
                agent_states={}, memory_snapshot={}
            )

        temp_files = list(temp_checkpoint_dir.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_checkpoint_content_valid(self, checkpoint_manager):
        """Saved checkpoints should have valid JSON content."""
        path = checkpoint_manager.save_checkpoint(
            cycle_id=1, phase="planning",
            agent_states={"nested": {"deep": {"value": 123}}},
            memory_snapshot={"list": [1, 2, 3]}
        )

        with open(path) as f:
            data = json.load(f)

        assert data["cycle_id"] == 1
        assert data["agent_states"]["nested"]["deep"]["value"] == 123
        assert data["memory_snapshot"]["list"] == [1, 2, 3]
