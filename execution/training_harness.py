"""
Training Harness - Runs Inside Subprocess

This module runs inside the training subprocess and provides:
- Training loop execution with graceful shutdown
- Signal handling for SIGTERM
- Emergency checkpointing on crash/timeout
- IPC communication with parent process

Can be run as a module:
    python -m execution.training_harness --experiment-id exp_001 --ipc-dir /tmp/arc_ipc/exp_001

Author: ARC Team (Dev 1)
Created: 2025-11-26
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """Training state machine."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        model_config: Model architecture config
        optimizer_config: Optimizer settings
        data_config: Dataset configuration
        num_epochs: Number of training epochs
        checkpoint_every: Save checkpoint every N epochs
        checkpoint_dir: Directory for checkpoints
    """
    model_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    num_epochs: int = 10
    checkpoint_every: int = 1
    checkpoint_dir: str = "./checkpoints"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            model_config=data.get("model_config", {}),
            optimizer_config=data.get("optimizer_config", {}),
            data_config=data.get("data_config", {}),
            num_epochs=data.get("num_epochs", 10),
            checkpoint_every=data.get("checkpoint_every", 1),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
        )


class TrainingHarness:
    """
    Training harness that runs inside subprocess.

    Handles:
    - Signal handling for graceful shutdown
    - Emergency checkpointing on crash
    - IPC communication with parent
    - Training loop orchestration
    """

    def __init__(
        self,
        experiment_id: str,
        ipc_dir: Path,
        config: TrainingConfig,
    ):
        """
        Initialize training harness.

        Args:
            experiment_id: Unique experiment identifier
            ipc_dir: Directory for IPC files
            config: Training configuration
        """
        self.experiment_id = experiment_id
        self.ipc_dir = Path(ipc_dir)
        self.config = config

        self.state = TrainingState.INITIALIZING
        self._stop_requested = False
        self._checkpoint_requested = False

        # IPC protocol (import here to avoid circular imports at module level)
        from execution.ipc_protocol import IPCProtocol, MessageType
        self.ipc = IPCProtocol(ipc_dir, experiment_id, is_parent=False)
        self.MessageType = MessageType

        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.best_metrics: Dict[str, float] = {}
        self.latest_checkpoint: Optional[str] = None

        # Set up signal handlers
        self._setup_signal_handlers()

        # Register cleanup on exit
        atexit.register(self._on_exit)

        logger.info(f"TrainingHarness initialized for {experiment_id}")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame) -> None:
        """Handle SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, requesting stop")
        self._stop_requested = True
        self._checkpoint_requested = True

    def _on_exit(self) -> None:
        """Cleanup on exit."""
        if self.state == TrainingState.TRAINING:
            logger.warning("Abnormal exit during training, attempting emergency checkpoint")
            try:
                self._save_emergency_checkpoint()
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
            self.state = TrainingState.READY
            self.ipc.send_ready()

            # Wait for start command (or proceed if we have config)
            start_msg = self.ipc.wait_for(self.MessageType.START, timeout=30)
            if start_msg:
                # Merge any start-time config updates
                start_config = start_msg.payload.get("config", {})
                if start_config:
                    self.config = TrainingConfig.from_dict({
                        **self.config.__dict__,
                        **start_config
                    })

            self.state = TrainingState.TRAINING

            # Run training
            if train_fn is not None:
                results = train_fn(self, self.config)
            else:
                results = self._default_training_loop()

            self.state = TrainingState.COMPLETED
            self.ipc.send_complete(results)

            return results

        except Exception as e:
            self.state = TrainingState.FAILED
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(f"Training failed: {error_msg}\n{tb}")

            self.ipc.send_error(error_msg, tb)

            # Try emergency checkpoint
            try:
                self._save_emergency_checkpoint()
            except Exception:
                pass

            raise

    def _default_training_loop(self) -> Dict[str, Any]:
        """
        Default training loop implementation.

        This is a stub that should be replaced with actual training logic.
        It simulates training for testing purposes.
        """
        logger.info("Running default training loop (stub)")

        num_epochs = self.config.num_epochs
        batches_per_epoch = 100  # Simulated

        self.total_batches = batches_per_epoch

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            if self._stop_requested:
                logger.info("Stop requested, ending training")
                break

            for batch in range(batches_per_epoch):
                self.current_batch = batch

                # Check for stop/checkpoint requests
                self._check_ipc_commands()

                if self._stop_requested:
                    break

                # Simulate batch processing
                time.sleep(0.01)  # Small delay for simulation

                # Report progress periodically
                if batch % 10 == 0:
                    loss = 1.0 / (epoch + 1) + 0.1 * (1 - batch / batches_per_epoch)
                    self.ipc.send_progress(
                        epoch=epoch,
                        batch=batch,
                        total_batches=batches_per_epoch,
                        loss=loss,
                    )

            # End of epoch
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": 1.0 / (epoch + 1),
                "val_loss": 1.1 / (epoch + 1),
            }
            self.ipc.send_metrics(epoch_metrics)

            # Checkpoint if needed
            if (epoch + 1) % self.config.checkpoint_every == 0 or self._checkpoint_requested:
                self._save_checkpoint(epoch, epoch_metrics)
                self._checkpoint_requested = False

            # Update best metrics
            if not self.best_metrics or epoch_metrics["val_loss"] < self.best_metrics.get("val_loss", float("inf")):
                self.best_metrics = epoch_metrics.copy()

        return {
            "final_epoch": self.current_epoch,
            "best_metrics": self.best_metrics,
            "last_checkpoint": self.latest_checkpoint,
        }

    def _check_ipc_commands(self) -> None:
        """Check for IPC commands from parent."""
        messages = self.ipc.receive_all()
        for msg in messages:
            if msg.msg_type == self.MessageType.STOP:
                logger.info(f"Stop command received: {msg.payload}")
                self._stop_requested = True

            elif msg.msg_type == self.MessageType.CHECKPOINT:
                logger.info("Checkpoint request received")
                self._checkpoint_requested = True

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> str:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Current metrics

        Returns:
            Path to saved checkpoint
        """
        self.state = TrainingState.CHECKPOINTING

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"checkpoint_{self.experiment_id}_epoch{epoch}.json"
        checkpoint_path = checkpoint_dir / checkpoint_name

        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "epoch": epoch,
            "batch": self.current_batch,
            "metrics": metrics,
            "best_metrics": self.best_metrics,
            "config": self.config.__dict__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Atomic write
        temp_path = checkpoint_path.with_suffix(".tmp")
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        temp_path.rename(checkpoint_path)

        self.latest_checkpoint = str(checkpoint_path)
        self.ipc.send_checkpoint_saved(str(checkpoint_path))

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        self.state = TrainingState.TRAINING
        return str(checkpoint_path)

    def _save_emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on crash."""
        logger.warning("Saving emergency checkpoint")

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"emergency_{self.experiment_id}_{int(time.time())}.json"
        checkpoint_path = checkpoint_dir / checkpoint_name

        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "type": "emergency",
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "best_metrics": self.best_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.ipc.send_checkpoint_saved(str(checkpoint_path))
        logger.info(f"Emergency checkpoint saved: {checkpoint_path}")

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
        self.current_batch = batch
        self.total_batches = total_batches
        self.ipc.send_progress(epoch, batch, total_batches, loss, **extra)

    def report_metrics(self, metrics: Dict[str, float]) -> None:
        """Report metrics to parent process."""
        self.ipc.send_metrics(metrics)
        # Update best if this is better
        if "val_loss" in metrics:
            if not self.best_metrics or metrics["val_loss"] < self.best_metrics.get("val_loss", float("inf")):
                self.best_metrics = metrics.copy()

    def should_stop(self) -> bool:
        """Check if training should stop."""
        self._check_ipc_commands()
        return self._stop_requested

    def should_checkpoint(self) -> bool:
        """Check if checkpoint was requested."""
        return self._checkpoint_requested


def main():
    """Main entry point when run as module."""
    parser = argparse.ArgumentParser(description="ARC Training Harness")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    parser.add_argument("--ipc-dir", required=True, help="IPC directory")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config = TrainingConfig.from_dict(config_dict)

    # Create and run harness
    harness = TrainingHarness(
        experiment_id=args.experiment_id,
        ipc_dir=args.ipc_dir,
        config=config,
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
