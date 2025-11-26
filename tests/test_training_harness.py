"""
Tests for Training Harness.

Phase G Bulletproof Execution - Tests for training harness that runs inside subprocess.
Updated for unified specs with status.json/metrics.json IPC.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from execution.training_harness import (
    TrainingHarness,
    TrainingConfig,
    TrainingState,
    StatusValue,
    GracefulKiller,
    StatusWriter,
    MetricsWriter,
    atomic_write_json,
)


class TestAtomicWriteJson:
    """Tests for atomic JSON writing."""

    def test_atomic_write_creates_file(self):
        """Should create file atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            atomic_write_json(path, data)

            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded == data

    def test_atomic_write_no_temp_file_left(self):
        """Should not leave temporary file behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            atomic_write_json(path, {"key": "value"})

            files = list(Path(tmpdir).iterdir())
            assert len(files) == 1
            assert files[0].name == "test.json"


class TestGracefulKiller:
    """Tests for GracefulKiller signal handler."""

    def test_initial_state(self):
        """Should not request kill initially."""
        killer = GracefulKiller()
        assert killer.kill_requested is False
        killer.restore()

    def test_restore_signals(self):
        """Should be able to restore original handlers."""
        killer = GracefulKiller()
        # Should not raise
        killer.restore()


class TestStatusWriter:
    """Tests for StatusWriter."""

    def test_writes_status_file(self):
        """Should write status.json with correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            writer = StatusWriter(status_file, "exp_001")

            writer.write(
                StatusValue.RUNNING.value,
                current_epoch=5,
                total_epochs=10,
                phase="training",
                message="Training epoch 5/10",
            )

            assert status_file.exists()
            data = json.loads(status_file.read_text())

            assert data["status"] == "running"
            assert data["experiment_id"] == "exp_001"
            assert data["current_epoch"] == 5
            assert data["total_epochs"] == 10
            assert data["phase"] == "training"
            assert data["error"] is None
            assert "updated_at" in data

    def test_writes_error_status(self):
        """Should write error information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            writer = StatusWriter(status_file, "exp_001")

            writer.write(
                StatusValue.FAILED.value,
                error="CUDA out of memory",
                message="Training failed",
            )

            data = json.loads(status_file.read_text())
            assert data["status"] == "failed"
            assert data["error"] == "CUDA out of memory"


class TestMetricsWriter:
    """Tests for MetricsWriter."""

    def test_records_epoch_metrics(self):
        """Should record metrics with history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            writer = MetricsWriter(metrics_file)

            writer.record_epoch(0, {"train_loss": 0.65, "val_loss": 0.58, "auc": 0.72})
            writer.record_epoch(1, {"train_loss": 0.55, "val_loss": 0.48, "auc": 0.78})

            data = json.loads(metrics_file.read_text())

            assert len(data["history"]) == 2
            assert data["history"][0]["epoch"] == 0
            assert data["history"][1]["auc"] == 0.78
            assert data["best"]["best_auc"] == 0.78
            assert data["best"]["best_auc_epoch"] == 1
            assert data["latest"]["epoch"] == 1

    def test_tracks_best_metrics(self):
        """Should track best AUC and val_loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            writer = MetricsWriter(metrics_file)

            writer.record_epoch(0, {"auc": 0.70, "val_loss": 0.60})
            writer.record_epoch(1, {"auc": 0.80, "val_loss": 0.50})
            writer.record_epoch(2, {"auc": 0.75, "val_loss": 0.55})

            data = json.loads(metrics_file.read_text())

            assert data["best"]["best_auc"] == 0.80
            assert data["best"]["best_auc_epoch"] == 1
            assert data["best"]["best_val_loss"] == 0.50
            assert data["best"]["best_val_loss_epoch"] == 1

    def test_load_existing_metrics(self):
        """Should load existing metrics file when resuming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"

            # Write existing metrics
            existing = {
                "history": [{"epoch": 0, "auc": 0.72}],
                "best": {"best_auc": 0.72, "best_auc_epoch": 0},
                "latest": {"epoch": 0},
            }
            metrics_file.write_text(json.dumps(existing))

            writer = MetricsWriter(metrics_file)
            writer.load_existing()

            assert len(writer.history) == 1
            assert writer.best["best_auc"] == 0.72


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Should create config with defaults."""
        config = TrainingConfig()

        assert config.num_epochs == 10
        assert config.checkpoint_every == 1
        assert config.checkpoint_dir == "./checkpoints"

    def test_from_dict_unified_format(self):
        """Should create config from unified spec format."""
        d = {
            "experiment_id": "exp_001",
            "model": {"architecture": "resnet50", "pretrained": True},
            "data": {"batch_size": 16, "image_size": 224},
            "optimizer": {"type": "adamw", "lr": 0.0001},
            "scheduler": {"type": "cosine", "T_max": 100},
            "loss": {"type": "focal", "gamma": 2.0},
            "training": {
                "num_epochs": 100,
                "checkpoint_every": 10,
                "early_stopping_patience": 15,
            },
        }

        config = TrainingConfig.from_dict(d)

        assert config.experiment_id == "exp_001"
        assert config.num_epochs == 100
        assert config.checkpoint_every == 10
        assert config.early_stopping_patience == 15
        assert config.model_config["architecture"] == "resnet50"
        assert config.optimizer_config["lr"] == 0.0001

    def test_from_dict_legacy_format(self):
        """Should handle legacy format with top-level num_epochs."""
        d = {
            "num_epochs": 50,
            "checkpoint_every": 5,
            "model_config": {"type": "resnet50"},
        }

        config = TrainingConfig.from_dict(d)

        assert config.num_epochs == 50
        assert config.checkpoint_every == 5

    def test_to_dict(self):
        """Should convert config to dictionary."""
        config = TrainingConfig(
            experiment_id="exp_001",
            num_epochs=50,
            checkpoint_every=5,
        )

        d = config.to_dict()

        assert d["experiment_id"] == "exp_001"
        assert d["training"]["num_epochs"] == 50


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


class TestStatusValue:
    """Tests for StatusValue enum (IPC status values)."""

    def test_all_status_values(self):
        """Should have all status values from shared contract."""
        assert StatusValue.PENDING.value == "pending"
        assert StatusValue.STARTING.value == "starting"
        assert StatusValue.RUNNING.value == "running"
        assert StatusValue.CHECKPOINTING.value == "checkpointing"
        assert StatusValue.COMPLETED.value == "completed"
        assert StatusValue.FAILED.value == "failed"
        assert StatusValue.TIMEOUT.value == "timeout"
        assert StatusValue.CANCELLED.value == "cancelled"
        assert StatusValue.OOM.value == "oom"


class TestTrainingHarness:
    """Tests for TrainingHarness class."""

    @pytest.fixture
    def tmpdir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TrainingConfig(
            experiment_id="test_exp",
            num_epochs=2,
            checkpoint_every=1,
        )

    @pytest.fixture
    def harness(self, tmpdir, config):
        """Create TrainingHarness instance."""
        return TrainingHarness(
            experiment_id="test_exp",
            config=config,
            status_file=tmpdir / "status.json",
            metrics_file=tmpdir / "metrics.json",
            checkpoint_dir=tmpdir / "checkpoints",
        )

    def test_initialization(self, harness):
        """Should initialize correctly."""
        assert harness.experiment_id == "test_exp"
        assert harness.state == TrainingState.INITIALIZING
        assert harness.current_epoch == 0
        assert harness.start_epoch == 0

    def test_initial_state(self, harness):
        """Should start in initializing state."""
        assert harness.state == TrainingState.INITIALIZING
        assert not harness.killer.kill_requested
        assert not harness._checkpoint_requested

    def test_should_stop_initially_false(self, harness):
        """Should not stop initially."""
        assert harness.should_stop() is False

    def test_should_checkpoint_initially_false(self, harness):
        """Should not checkpoint initially."""
        assert harness.should_checkpoint() is False

    def test_report_progress(self, harness):
        """Should report progress."""
        harness.report_progress(
            epoch=1,
            batch=10,
            total_batches=100,
            loss=0.5,
        )

        assert harness.current_epoch == 1

    def test_report_metrics(self, harness):
        """Should report and track best metrics."""
        harness.report_metrics({"epoch": 0, "val_loss": 0.5, "auc": 0.8})
        assert harness.best_metrics["auc"] == 0.8

        # Better metrics should update best
        harness.report_metrics({"epoch": 1, "val_loss": 0.3, "auc": 0.85})
        assert harness.best_metrics["auc"] == 0.85

        # Worse metrics should not update best
        harness.report_metrics({"epoch": 2, "val_loss": 0.4, "auc": 0.82})
        assert harness.best_metrics["auc"] == 0.85

    def test_request_checkpoint(self, harness):
        """Should be able to request checkpoint."""
        assert not harness.should_checkpoint()
        harness.request_checkpoint()
        assert harness.should_checkpoint()

    def test_writes_initial_status(self, tmpdir, config):
        """Should write initial status on creation."""
        harness = TrainingHarness(
            experiment_id="test_exp",
            config=config,
            status_file=tmpdir / "status.json",
            metrics_file=tmpdir / "metrics.json",
            checkpoint_dir=tmpdir / "checkpoints",
        )

        status_file = tmpdir / "status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["status"] == "starting"
        assert data["experiment_id"] == "test_exp"


class TestTrainingHarnessCheckpointing:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def setup(self):
        """Set up harness for checkpoint testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"

            config = TrainingConfig(
                experiment_id="checkpoint_test",
                num_epochs=5,
                checkpoint_every=2,
            )

            harness = TrainingHarness(
                experiment_id="checkpoint_test",
                config=config,
                status_file=tmpdir / "status.json",
                metrics_file=tmpdir / "metrics.json",
                checkpoint_dir=checkpoint_dir,
            )

            yield {
                "harness": harness,
                "checkpoint_dir": checkpoint_dir,
                "tmpdir": tmpdir,
            }

    def test_save_checkpoint(self, setup):
        """Should save checkpoint to file with proper naming."""
        harness = setup["harness"]
        checkpoint_dir = setup["checkpoint_dir"]

        # Set some state
        harness.current_epoch = 3
        harness.best_metrics = {"val_loss": 0.25, "auc": 0.85}

        # Save checkpoint
        path = harness._save_checkpoint(
            epoch=3,
            metrics={"train_loss": 0.3, "auc": 0.85},
            auc=0.85,
        )

        assert Path(path).exists()
        assert "epoch_3" in path
        assert "auc_0.8500" in path

        # Verify contents
        with open(path) as f:
            data = json.load(f)

        assert data["experiment_id"] == "checkpoint_test"
        assert data["epoch"] == 3
        assert data["best_metrics"]["auc"] == 0.85

    def test_latest_checkpoint_created(self, setup):
        """Should create latest.pt copy."""
        harness = setup["harness"]
        checkpoint_dir = setup["checkpoint_dir"]

        harness._save_checkpoint(epoch=1, metrics={"auc": 0.75}, auc=0.75)

        latest = checkpoint_dir / "latest.pt"
        assert latest.exists()

    def test_get_best_checkpoint(self, setup):
        """Should find best checkpoint by AUC."""
        harness = setup["harness"]
        checkpoint_dir = setup["checkpoint_dir"]

        # Save multiple checkpoints
        harness._save_checkpoint(epoch=0, metrics={"auc": 0.70}, auc=0.70)
        harness._save_checkpoint(epoch=1, metrics={"auc": 0.85}, auc=0.85)
        harness._save_checkpoint(epoch=2, metrics={"auc": 0.80}, auc=0.80)

        best = harness.get_best_checkpoint()
        assert best is not None
        assert "auc_0.8500" in best

    def test_maybe_resume_checkpoint(self, setup):
        """Should resume from existing checkpoint."""
        harness = setup["harness"]
        checkpoint_dir = setup["checkpoint_dir"]

        # Create a checkpoint manually (use .json for JSON format)
        ckpt_data = {
            "epoch": 5,
            "best_metrics": {"auc": 0.80},
        }
        ckpt_path = checkpoint_dir / "checkpoint_test_epoch5.json"
        ckpt_path.write_text(json.dumps(ckpt_data))

        # Resume
        harness.maybe_resume_checkpoint()

        assert harness.start_epoch == 6
        assert harness.best_metrics["auc"] == 0.80


class TestTrainingHarnessWithLegacyIPC:
    """Tests for backward compatibility with legacy IPC protocol."""

    @pytest.fixture
    def setup(self):
        """Set up harness with legacy IPC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            ipc_dir = tmpdir / "ipc"
            ipc_dir.mkdir()

            # Create parent IPC to simulate SubprocessExecutor
            from execution.ipc_protocol import IPCProtocol, MessageType

            parent_ipc = IPCProtocol(ipc_dir, "test_exp", is_parent=True)

            config = TrainingConfig(num_epochs=2)
            harness = TrainingHarness(
                experiment_id="test_exp",
                config=config,
                status_file=tmpdir / "status.json",
                metrics_file=tmpdir / "metrics.json",
                checkpoint_dir=tmpdir / "checkpoints",
                ipc_dir=ipc_dir,
            )

            yield {
                "harness": harness,
                "parent_ipc": parent_ipc,
                "ipc_dir": ipc_dir,
                "tmpdir": tmpdir,
            }

    def test_stop_request_via_legacy_ipc(self, setup):
        """Should handle stop request via legacy IPC."""
        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Parent sends stop
        parent_ipc.send_stop(reason="test")

        # Harness should detect stop (via should_stop which calls _check_ipc_commands)
        assert harness.should_stop() is True
        assert harness.killer.kill_requested is True

    def test_checkpoint_request_via_legacy_ipc(self, setup):
        """Should handle checkpoint request via legacy IPC."""
        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Parent sends checkpoint request
        parent_ipc.send_checkpoint_request()

        # Check for commands
        harness._check_ipc_commands()

        assert harness._checkpoint_requested is True

    def test_ready_signal_via_legacy_ipc(self, setup):
        """Should send ready signal via legacy IPC."""
        from execution.ipc_protocol import MessageType

        harness = setup["harness"]
        parent_ipc = setup["parent_ipc"]

        # Harness sends ready
        harness.ipc.send_ready()

        # Parent should receive
        msg = parent_ipc.receive(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == MessageType.READY


class TestTrainingHarnessCLI:
    """Tests for CLI entry point."""

    def test_help_message(self):
        """Should show help message."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "execution.training_harness", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )

        assert result.returncode == 0
        assert "--experiment-id" in result.stdout
        assert "--config" in result.stdout
        assert "--status-file" in result.stdout
        assert "--metrics-file" in result.stdout
        assert "--checkpoint-dir" in result.stdout
