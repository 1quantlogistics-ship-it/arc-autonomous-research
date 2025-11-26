"""
Tests for Training Harness.

Phase G Bulletproof Execution - Tests for training harness that runs inside subprocess.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from execution.training_harness import (
    TrainingHarness,
    TrainingConfig,
    TrainingState,
)
from execution.ipc_protocol import IPCProtocol, MessageType


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Should create config with defaults."""
        config = TrainingConfig()

        assert config.num_epochs == 10
        assert config.checkpoint_every == 1
        assert config.checkpoint_dir == "./checkpoints"

    def test_from_dict(self):
        """Should create config from dictionary."""
        d = {
            "num_epochs": 50,
            "checkpoint_every": 5,
            "checkpoint_dir": "/tmp/checkpoints",
            "model_config": {"type": "resnet50"},
            "optimizer_config": {"lr": 0.001},
        }

        config = TrainingConfig.from_dict(d)

        assert config.num_epochs == 50
        assert config.checkpoint_every == 5
        assert config.model_config["type"] == "resnet50"

    def test_from_dict_partial(self):
        """Should use defaults for missing keys."""
        d = {"num_epochs": 20}

        config = TrainingConfig.from_dict(d)

        assert config.num_epochs == 20
        assert config.checkpoint_every == 1  # default


class TestTrainingState:
    """Tests for TrainingState enum."""

    def test_all_states(self):
        """Should have all expected states."""
        assert TrainingState.INITIALIZING.value == "initializing"
        assert TrainingState.READY.value == "ready"
        assert TrainingState.TRAINING.value == "training"
        assert TrainingState.CHECKPOINTING.value == "checkpointing"
        assert TrainingState.STOPPING.value == "stopping"
        assert TrainingState.COMPLETED.value == "completed"
        assert TrainingState.FAILED.value == "failed"


class TestTrainingHarness:
    """Tests for TrainingHarness class."""

    @pytest.fixture
    def ipc_dir(self):
        """Create temporary IPC directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TrainingConfig(
            num_epochs=2,
            checkpoint_every=1,
            checkpoint_dir="/tmp/test_checkpoints",
        )

    @pytest.fixture
    def harness(self, ipc_dir, config):
        """Create TrainingHarness instance."""
        return TrainingHarness(
            experiment_id="test_exp",
            ipc_dir=ipc_dir,
            config=config,
        )

    def test_initialization(self, harness):
        """Should initialize correctly."""
        assert harness.experiment_id == "test_exp"
        assert harness.state == TrainingState.INITIALIZING
        assert harness.current_epoch == 0
        assert harness.current_batch == 0

    def test_initial_state(self, harness):
        """Should start in initializing state."""
        assert harness.state == TrainingState.INITIALIZING
        assert not harness._stop_requested
        assert not harness._checkpoint_requested

    def test_should_stop_initially_false(self, harness):
        """Should not stop initially."""
        assert harness.should_stop() is False

    def test_should_checkpoint_initially_false(self, harness):
        """Should not checkpoint initially."""
        assert harness.should_checkpoint() is False

    def test_report_progress(self, harness, ipc_dir):
        """Should report progress via IPC."""
        harness.report_progress(
            epoch=1,
            batch=10,
            total_batches=100,
            loss=0.5,
        )

        assert harness.current_epoch == 1
        assert harness.current_batch == 10
        assert harness.total_batches == 100

    def test_report_metrics(self, harness):
        """Should report and track best metrics."""
        harness.report_metrics({"val_loss": 0.5, "auc": 0.8})
        assert harness.best_metrics["val_loss"] == 0.5

        # Better metrics should update best
        harness.report_metrics({"val_loss": 0.3, "auc": 0.85})
        assert harness.best_metrics["val_loss"] == 0.3

        # Worse metrics should not update best
        harness.report_metrics({"val_loss": 0.4, "auc": 0.82})
        assert harness.best_metrics["val_loss"] == 0.3


class TestTrainingHarnessIntegration:
    """Integration tests with IPC."""

    @pytest.fixture
    def setup(self):
        """Set up harness with parent IPC for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipc_dir = Path(tmpdir)

            # Create parent IPC to simulate SubprocessExecutor
            parent_ipc = IPCProtocol(ipc_dir, "test_exp", is_parent=True)

            # Create harness (child)
            config = TrainingConfig(num_epochs=2)
            harness = TrainingHarness(
                experiment_id="test_exp",
                ipc_dir=ipc_dir,
                config=config,
            )

            yield {
                "harness": harness,
                "parent_ipc": parent_ipc,
                "ipc_dir": ipc_dir,
            }

    def test_stop_request_via_ipc(self, setup):
        """Should handle stop request via IPC."""
        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Parent sends stop
        parent_ipc.send_stop(reason="test")

        # Harness should detect stop
        assert harness.should_stop() is True
        assert harness._stop_requested is True

    def test_checkpoint_request_via_ipc(self, setup):
        """Should handle checkpoint request via IPC."""
        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Parent sends checkpoint request
        parent_ipc.send_checkpoint_request()

        # Check for commands
        harness._check_ipc_commands()

        assert harness._checkpoint_requested is True

    def test_ready_signal(self, setup):
        """Should send ready signal."""
        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Harness sends ready
        harness.ipc.send_ready()

        # Parent should receive
        msg = parent_ipc.receive(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == MessageType.READY


class TestTrainingHarnessCheckpointing:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def setup(self):
        """Set up harness for checkpoint testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipc_dir = Path(tmpdir) / "ipc"
            ipc_dir.mkdir()
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            config = TrainingConfig(
                num_epochs=5,
                checkpoint_every=2,
                checkpoint_dir=str(checkpoint_dir),
            )

            harness = TrainingHarness(
                experiment_id="checkpoint_test",
                ipc_dir=ipc_dir,
                config=config,
            )

            yield {
                "harness": harness,
                "checkpoint_dir": checkpoint_dir,
            }

    def test_save_checkpoint(self, setup):
        """Should save checkpoint to file."""
        harness = setup["harness"]
        checkpoint_dir = setup["checkpoint_dir"]

        # Set some state
        harness.current_epoch = 3
        harness.best_metrics = {"val_loss": 0.25}

        # Save checkpoint
        path = harness._save_checkpoint(epoch=3, metrics={"train_loss": 0.3})

        assert Path(path).exists()
        assert "checkpoint_test" in path
        assert "epoch3" in path

        # Verify contents
        with open(path) as f:
            data = json.load(f)

        assert data["experiment_id"] == "checkpoint_test"
        assert data["epoch"] == 3
        assert data["best_metrics"]["val_loss"] == 0.25
