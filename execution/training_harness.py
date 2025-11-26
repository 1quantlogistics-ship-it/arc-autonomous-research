"""
Training Harness - Runs Inside Subprocess

This module runs inside the training subprocess and provides:
- Training loop execution with graceful shutdown
- Signal handling for SIGTERM/SIGINT
- Emergency checkpointing on crash/timeout
- IPC communication via status.json and metrics.json files
- GPU isolation enforcement

Can be run as a module:
    python -m execution.training_harness \
        --experiment-id exp_001 \
        --config /path/to/config.json \
        --status-file /path/to/status.json \
        --metrics-file /path/to/metrics.json \
        --checkpoint-dir /path/to/checkpoints/

Author: ARC Team (Dev 1)
Created: 2025-11-26
Updated: 2025-11-26 - Unified specs with status.json/metrics.json IPC
"""

# =============================================================================
# GPU ISOLATION - MUST BE BEFORE ANY TORCH IMPORTS
# =============================================================================
import os
import sys

def _enforce_gpu():
    """Enforce GPU isolation before torch is imported."""
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda is None:
        # Default to GPU 0 if not set (training GPU should be set by parent)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[training_harness] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", file=sys.stderr)

_enforce_gpu()

# =============================================================================
# NOW SAFE TO IMPORT TORCH AND OTHER MODULES
# =============================================================================
import argparse
import atexit
import json
import logging
import signal
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)


# =============================================================================
# STATUS VALUES (SHARED CONTRACT)
# =============================================================================
class StatusValue(Enum):
    """Valid status values for status.json."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    OOM = "oom"


class TrainingState(Enum):
    """Internal training state machine."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# GRACEFUL KILLER - SIGNAL HANDLING
# =============================================================================
class GracefulKiller:
    """
    Catches SIGTERM/SIGINT for graceful shutdown.
    Training loop checks self.kill_requested each batch.
    On signal: set flag, training saves checkpoint and exits cleanly.
    """

    def __init__(self):
        self.kill_requested = False
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        print(f"[training_harness] Received signal {signum}, will checkpoint and exit", file=sys.stderr)
        logger.info(f"Received signal {signum}, requesting graceful shutdown")
        self.kill_requested = True

    def restore(self):
        """Restore original signal handlers."""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)


# =============================================================================
# ATOMIC FILE WRITER
# =============================================================================
def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically to prevent corruption."""
    path = Path(path)
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp_path.rename(path)  # Atomic on POSIX


# =============================================================================
# STATUS WRITER
# =============================================================================
class StatusWriter:
    """
    Writes status.json with atomic write pattern.
    Updates current state, epoch, phase, errors.
    """

    def __init__(self, status_file: Path, experiment_id: str):
        self.status_file = Path(status_file)
        self.experiment_id = experiment_id
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        status: str,
        current_epoch: int = 0,
        total_epochs: int = 0,
        phase: str = "initializing",
        error: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Write status to file."""
        data = {
            "status": status,
            "experiment_id": self.experiment_id,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "phase": phase,
            "updated_at": time.time(),
            "error": error,
            "message": message or f"Status: {status}",
        }
        atomic_write_json(self.status_file, data)
        logger.debug(f"Status updated: {status}")


# =============================================================================
# METRICS WRITER
# =============================================================================
class MetricsWriter:
    """
    Writes metrics.json with atomic write pattern.
    Tracks history, best metrics, and latest values.
    Called after each epoch.
    """

    def __init__(self, metrics_file: Path):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[dict] = []
        self.best: Dict[str, Any] = {
            "best_auc": 0.0,
            "best_auc_epoch": 0,
            "best_val_loss": float("inf"),
            "best_val_loss_epoch": 0,
        }
        self.latest: Dict[str, Any] = {}

    def record_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for an epoch."""
        entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics,
        }
        self.history.append(entry)
        self.latest = entry.copy()

        # Update best metrics
        auc = metrics.get("auc", metrics.get("val_auc", 0))
        val_loss = metrics.get("val_loss", float("inf"))

        if auc > self.best["best_auc"]:
            self.best["best_auc"] = auc
            self.best["best_auc_epoch"] = epoch

        if val_loss < self.best["best_val_loss"]:
            self.best["best_val_loss"] = val_loss
            self.best["best_val_loss_epoch"] = epoch

        self._write()

    def _write(self) -> None:
        """Write metrics to file."""
        data = {
            "history": self.history,
            "best": self.best,
            "latest": self.latest,
        }
        atomic_write_json(self.metrics_file, data)
        logger.debug(f"Metrics updated: epoch {self.latest.get('epoch', '?')}")

    def load_existing(self) -> None:
        """Load existing metrics file if resuming."""
        if self.metrics_file.exists():
            try:
                data = json.loads(self.metrics_file.read_text())
                self.history = data.get("history", [])
                self.best = data.get("best", self.best)
                self.latest = data.get("latest", {})
                logger.info(f"Loaded existing metrics: {len(self.history)} epochs")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load existing metrics: {e}")


# =============================================================================
# TRAINING CONFIG
# =============================================================================
@dataclass
class TrainingConfig:
    """
    Training configuration.

    Matches the experiment config schema from the unified spec.
    """
    experiment_id: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    loss_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors
    num_epochs: int = 10
    checkpoint_every: int = 1
    checkpoint_dir: str = "./checkpoints"
    early_stopping_patience: int = 15

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary (matches unified spec format)."""
        # Handle nested training config
        training = data.get("training", {})

        return cls(
            experiment_id=data.get("experiment_id", ""),
            model_config=data.get("model", data.get("model_config", {})),
            optimizer_config=data.get("optimizer", data.get("optimizer_config", {})),
            scheduler_config=data.get("scheduler", data.get("scheduler_config", {})),
            data_config=data.get("data", data.get("data_config", {})),
            loss_config=data.get("loss", data.get("loss_config", {})),
            training_config=training,
            num_epochs=training.get("num_epochs", data.get("num_epochs", 10)),
            checkpoint_every=training.get("checkpoint_every", data.get("checkpoint_every", 10)),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
            early_stopping_patience=training.get("early_stopping_patience", 15),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model": self.model_config,
            "optimizer": self.optimizer_config,
            "scheduler": self.scheduler_config,
            "data": self.data_config,
            "loss": self.loss_config,
            "training": {
                "num_epochs": self.num_epochs,
                "checkpoint_every": self.checkpoint_every,
                "early_stopping_patience": self.early_stopping_patience,
            },
            "checkpoint_dir": self.checkpoint_dir,
        }


# =============================================================================
# TRAINING HARNESS
# =============================================================================
class TrainingHarness:
    """
    Main orchestrator for training subprocess.

    Lifecycle:
    1. __init__: Parse args, load config
    2. run(): Main entry point
       - setup_model()
       - setup_data()
       - setup_optimizer()
       - setup_loss()
       - maybe_resume_checkpoint()
       - training_loop()
       - save_final_checkpoint()
    """

    def __init__(
        self,
        experiment_id: str,
        config: TrainingConfig,
        status_file: Path,
        metrics_file: Path,
        checkpoint_dir: Path,
        ipc_dir: Optional[Path] = None,
    ):
        """
        Initialize training harness.

        Args:
            experiment_id: Unique experiment identifier
            config: Training configuration
            status_file: Path to status.json for IPC
            metrics_file: Path to metrics.json for IPC
            checkpoint_dir: Directory for checkpoints
            ipc_dir: Optional IPC directory for legacy protocol
        """
        self.experiment_id = experiment_id
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # IPC writers
        self.status_writer = StatusWriter(status_file, experiment_id)
        self.metrics_writer = MetricsWriter(metrics_file)

        # Legacy IPC protocol (for backwards compatibility)
        self.ipc = None
        self.MessageType = None
        if ipc_dir:
            try:
                from execution.ipc_protocol import IPCProtocol, MessageType
                self.ipc = IPCProtocol(ipc_dir, experiment_id, is_parent=False)
                self.MessageType = MessageType
            except ImportError:
                logger.warning("IPC protocol not available, using file-based IPC only")

        # Signal handler
        self.killer = GracefulKiller()

        # Training state
        self.state = TrainingState.INITIALIZING
        self._checkpoint_requested = False
        self.current_epoch = 0
        self.start_epoch = 0
        self.total_epochs = config.num_epochs
        self.best_metrics: Dict[str, float] = {}
        self.latest_checkpoint: Optional[str] = None

        # Track epochs without improvement for early stopping
        self._epochs_without_improvement = 0

        # Register cleanup on exit
        atexit.register(self._on_exit)

        # Write initial status
        self.status_writer.write(
            StatusValue.STARTING.value,
            phase="initializing",
            message=f"Harness initialized for {experiment_id}",
        )

        logger.info(f"TrainingHarness initialized for {experiment_id}")

    def _on_exit(self) -> None:
        """Cleanup on exit."""
        if self.state == TrainingState.TRAINING:
            logger.warning("Abnormal exit during training, attempting emergency checkpoint")
            try:
                self._emergency_checkpoint()
            except Exception as e:
                logger.error(f"Emergency checkpoint failed: {e}")

    def run(self, train_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            train_fn: Optional custom training function
                      Signature: train_fn(harness, config) -> Dict[str, Any]

        Returns:
            Final training results
        """
        try:
            self._setup()

            self.state = TrainingState.TRAINING
            self.status_writer.write(
                StatusValue.RUNNING.value,
                current_epoch=self.start_epoch,
                total_epochs=self.total_epochs,
                phase="training",
                message="Training started",
            )

            # Run training
            if train_fn is not None:
                results = train_fn(self, self.config)
            else:
                results = self._default_training_loop()

            self.state = TrainingState.COMPLETED
            self.status_writer.write(
                StatusValue.COMPLETED.value,
                current_epoch=self.current_epoch,
                total_epochs=self.total_epochs,
                phase="completed",
                message="Training completed successfully",
            )

            if self.ipc:
                self.ipc.send_complete(results)

            return results

        except Exception as e:
            self.state = TrainingState.FAILED
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(f"Training failed: {error_msg}\n{tb}")

            # Check if it's an OOM error
            status = StatusValue.FAILED.value
            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                status = StatusValue.OOM.value

            self.status_writer.write(
                status,
                current_epoch=self.current_epoch,
                total_epochs=self.total_epochs,
                phase="failed",
                error=error_msg,
                message=f"Training failed: {error_msg[:100]}",
            )

            if self.ipc:
                self.ipc.send_error(error_msg, tb)

            # Try emergency checkpoint
            self._emergency_checkpoint()

            raise

    def _setup(self) -> None:
        """Set up training (load checkpoints, etc.)."""
        self.status_writer.write(
            StatusValue.STARTING.value,
            phase="setup",
            message="Loading configuration and checkpoints",
        )

        # Load existing metrics if resuming
        self.metrics_writer.load_existing()

        # Check for existing checkpoints to resume from
        self.maybe_resume_checkpoint()

        # Signal ready if using legacy IPC
        if self.ipc:
            self.ipc.send_ready()

    def maybe_resume_checkpoint(self) -> None:
        """Resume from latest checkpoint if available."""
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        # Also check for .json checkpoints (legacy format)
        checkpoints.extend(self.checkpoint_dir.glob("checkpoint_*.json"))

        # Filter out latest.pt symlink/copy
        checkpoints = [c for c in checkpoints if c.name != "latest.pt"]

        if not checkpoints:
            self.start_epoch = 0
            logger.info("No checkpoints found, starting from epoch 0")
            return

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Resuming from checkpoint: {latest}")

        try:
            if latest.suffix == ".json":
                # Legacy JSON checkpoint
                with open(latest) as f:
                    ckpt = json.load(f)
                self.start_epoch = ckpt.get("epoch", 0) + 1
                self.best_metrics = ckpt.get("best_metrics", {})
            else:
                # PyTorch checkpoint
                import torch
                ckpt = torch.load(latest, map_location="cpu")
                self.start_epoch = ckpt.get("epoch", 0) + 1
                self.best_metrics = ckpt.get("best_metrics", {})
                # Note: model/optimizer state loading is done by the custom train_fn

            self.current_epoch = self.start_epoch
            logger.info(f"Resuming from epoch {self.start_epoch}")

        except Exception as e:
            logger.warning(f"Could not load checkpoint {latest}: {e}")
            self.start_epoch = 0

    def _default_training_loop(self) -> Dict[str, Any]:
        """
        Default training loop implementation.

        This is a stub that should be replaced with actual training logic.
        It simulates training for testing purposes.
        """
        logger.info("Running default training loop (stub)")

        batches_per_epoch = 100  # Simulated

        for epoch in range(self.start_epoch, self.total_epochs):
            self.current_epoch = epoch

            # Check for graceful shutdown
            if self.killer.kill_requested:
                logger.info("Graceful shutdown requested, saving checkpoint and exiting")
                self._save_checkpoint(epoch, self.best_metrics, is_final=True)
                break

            # Update status
            self.status_writer.write(
                StatusValue.RUNNING.value,
                current_epoch=epoch,
                total_epochs=self.total_epochs,
                phase="training",
                message=f"Training epoch {epoch}/{self.total_epochs}",
            )

            # Simulate epoch training
            epoch_metrics = self._simulate_epoch(epoch, batches_per_epoch)

            # Record metrics
            self.metrics_writer.record_epoch(epoch, epoch_metrics)

            if self.ipc:
                self.ipc.send_metrics(epoch_metrics)

            # Update best metrics and check early stopping
            improved = self._update_best_metrics(epoch_metrics)

            # Checkpoint if needed
            should_checkpoint = (
                (epoch + 1) % self.config.checkpoint_every == 0 or
                self._checkpoint_requested or
                improved
            )

            if should_checkpoint:
                auc = epoch_metrics.get("auc", epoch_metrics.get("val_auc", 0))
                self._save_checkpoint(epoch, epoch_metrics, auc=auc)
                self._checkpoint_requested = False

            # Early stopping check
            if self._epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping: no improvement for {self._epochs_without_improvement} epochs")
                break

        return {
            "final_epoch": self.current_epoch,
            "best_metrics": self.best_metrics,
            "last_checkpoint": self.latest_checkpoint,
            "total_epochs_trained": self.current_epoch - self.start_epoch + 1,
        }

    def _simulate_epoch(self, epoch: int, batches_per_epoch: int) -> Dict[str, Any]:
        """Simulate a training epoch (for testing)."""
        for batch in range(batches_per_epoch):
            if self.killer.kill_requested:
                break

            # Check for checkpoint requests via legacy IPC
            if self.ipc:
                self._check_ipc_commands()

            # Simulate batch processing
            time.sleep(0.01)

            # Report progress periodically
            if batch % 10 == 0:
                loss = 1.0 / (epoch + 1) + 0.1 * (1 - batch / batches_per_epoch)
                if self.ipc:
                    self.ipc.send_progress(
                        epoch=epoch,
                        batch=batch,
                        total_batches=batches_per_epoch,
                        loss=loss,
                    )

        # Simulated epoch metrics
        return {
            "epoch": epoch,
            "train_loss": 0.65 - epoch * 0.02,
            "val_loss": 0.58 - epoch * 0.015,
            "auc": 0.72 + epoch * 0.01,
            "sensitivity": 0.68 + epoch * 0.008,
            "specificity": 0.76 + epoch * 0.005,
            "lr": 0.0001 * (0.95 ** epoch),
        }

    def _check_ipc_commands(self) -> None:
        """Check for IPC commands from parent (legacy protocol)."""
        if not self.ipc:
            return

        messages = self.ipc.receive_all()
        for msg in messages:
            if msg.msg_type == self.MessageType.STOP:
                logger.info(f"Stop command received: {msg.payload}")
                self.killer.kill_requested = True

            elif msg.msg_type == self.MessageType.CHECKPOINT:
                logger.info("Checkpoint request received")
                self._checkpoint_requested = True

    def _update_best_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Update best metrics and return True if improved."""
        improved = False

        # Check AUC improvement
        auc = metrics.get("auc", metrics.get("val_auc"))
        if auc and (not self.best_metrics or auc > self.best_metrics.get("auc", 0)):
            self.best_metrics = metrics.copy()
            self._epochs_without_improvement = 0
            improved = True
        else:
            self._epochs_without_improvement += 1

        return improved

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        auc: Optional[float] = None,
        is_final: bool = False,
    ) -> str:
        """
        Save a training checkpoint.

        Checkpoint naming convention:
        - epoch_N_auc_X.XXXX.pt for regular checkpoints
        - latest.pt for symlink/copy of most recent
        - emergency_epoch_N.pt for crash recovery
        """
        self.state = TrainingState.CHECKPOINTING
        self.status_writer.write(
            StatusValue.CHECKPOINTING.value,
            current_epoch=epoch,
            total_epochs=self.total_epochs,
            phase="checkpointing",
            message=f"Saving checkpoint at epoch {epoch}",
        )

        # Build checkpoint filename
        if auc is not None:
            checkpoint_name = f"epoch_{epoch}_auc_{auc:.4f}.pt"
        else:
            checkpoint_name = f"epoch_{epoch}.pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Checkpoint data (JSON format for stub, would be torch.save in real impl)
        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "epoch": epoch,
            "metrics": metrics,
            "best_metrics": self.best_metrics,
            "config": self.config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save checkpoint (JSON for stub)
        atomic_write_json(checkpoint_path, checkpoint_data)

        # Update latest.pt symlink/copy
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        # Use copy instead of symlink for portability
        import shutil
        shutil.copy2(checkpoint_path, latest_path)

        self.latest_checkpoint = str(checkpoint_path)

        if self.ipc:
            self.ipc.send_checkpoint_saved(str(checkpoint_path))

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if not is_final:
            self.state = TrainingState.TRAINING

        return str(checkpoint_path)

    def _emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on crash."""
        try:
            logger.warning("Saving emergency checkpoint")

            checkpoint_name = f"emergency_epoch_{self.current_epoch}.pt"
            checkpoint_path = self.checkpoint_dir / checkpoint_name

            checkpoint_data = {
                "experiment_id": self.experiment_id,
                "type": "emergency",
                "epoch": self.current_epoch,
                "best_metrics": self.best_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            atomic_write_json(checkpoint_path, checkpoint_data)

            if self.ipc:
                self.ipc.send_checkpoint_saved(str(checkpoint_path))

            logger.info(f"Emergency checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Emergency checkpoint failed: {e}")

    # =========================================================================
    # PUBLIC API FOR CUSTOM TRAINING FUNCTIONS
    # =========================================================================

    def report_progress(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        **extra
    ) -> None:
        """Report training progress to parent process."""
        self.current_epoch = epoch

        if self.ipc:
            self.ipc.send_progress(epoch, batch, total_batches, loss, **extra)

    def report_metrics(self, metrics: Dict[str, float]) -> None:
        """Report metrics to parent process."""
        epoch = metrics.get("epoch", self.current_epoch)
        self.metrics_writer.record_epoch(epoch, metrics)

        if self.ipc:
            self.ipc.send_metrics(metrics)

        self._update_best_metrics(metrics)

    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.ipc:
            self._check_ipc_commands()
        return self.killer.kill_requested

    def should_checkpoint(self) -> bool:
        """Check if checkpoint was requested."""
        return self._checkpoint_requested

    def request_checkpoint(self) -> None:
        """Request a checkpoint at next opportunity."""
        self._checkpoint_requested = True

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint by AUC."""
        checkpoints = list(self.checkpoint_dir.glob("epoch_*_auc_*.pt"))
        if not checkpoints:
            return None

        # Parse AUC from filename and find best
        def get_auc(p: Path) -> float:
            try:
                # epoch_N_auc_X.XXXX.pt
                parts = p.stem.split("_")
                auc_idx = parts.index("auc") + 1
                return float(parts[auc_idx])
            except (ValueError, IndexError):
                return 0.0

        return str(max(checkpoints, key=get_auc))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point when run as module."""
    parser = argparse.ArgumentParser(description="ARC Training Harness")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--status-file", required=True, help="Status file path (IPC)")
    parser.add_argument("--metrics-file", required=True, help="Metrics file path (IPC)")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--ipc-dir", default=None, help="Legacy IPC directory (optional)")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Log GPU info
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config = TrainingConfig.from_dict(config_dict)

    # Create and run harness
    harness = TrainingHarness(
        experiment_id=args.experiment_id,
        config=config,
        status_file=Path(args.status_file),
        metrics_file=Path(args.metrics_file),
        checkpoint_dir=Path(args.checkpoint_dir),
        ipc_dir=Path(args.ipc_dir) if args.ipc_dir else None,
    )

    try:
        results = harness.run()
        logger.info(f"Training completed: {results}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
