"""
Subprocess Executor - Process Isolation for Training Jobs

Spawns training jobs as separate subprocesses to ensure:
- Training crashes cannot kill the ARC system
- Timeout enforcement with SIGTERM→SIGKILL escalation
- Clean resource cleanup on termination

Author: ARC Team (Dev 1)
Created: 2025-11-26
"""

import os
import sys
import signal
import subprocess
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from threading import Thread, Event

from execution.ipc_protocol import IPCProtocol, MessageType, IPCMessage

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class ExecutionResult:
    """
    Result of subprocess execution.

    Attributes:
        experiment_id: Experiment identifier
        status: Final execution status
        exit_code: Process exit code
        start_time: When execution started
        end_time: When execution ended
        duration_seconds: Total runtime
        metrics: Collected metrics (if any)
        error: Error message (if failed)
        checkpoint_path: Path to last checkpoint (if saved)
    """
    experiment_id: str
    status: ExecutionStatus
    exit_code: Optional[int] = None
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
        }


class SubprocessExecutor:
    """
    Executes training jobs in isolated subprocesses.

    Features:
    - Process isolation (crashes don't kill parent)
    - Timeout enforcement with graceful escalation
    - IPC-based communication with training harness
    - Progress monitoring and metric collection

    Usage:
        executor = SubprocessExecutor()
        result = executor.execute(
            experiment_id="exp_001",
            config=training_config,
            timeout_seconds=3600,
        )
    """

    DEFAULT_TIMEOUT = 3600  # 1 hour
    SIGTERM_GRACE_PERIOD = 30  # seconds before SIGKILL
    POLL_INTERVAL = 1.0  # seconds between status checks

    def __init__(
        self,
        ipc_base_dir: Optional[Path] = None,
        python_executable: Optional[str] = None,
        harness_module: str = "execution.training_harness",
    ):
        """
        Initialize SubprocessExecutor.

        Args:
            ipc_base_dir: Base directory for IPC files
            python_executable: Python interpreter to use
            harness_module: Module containing training harness
        """
        self.ipc_base_dir = Path(ipc_base_dir) if ipc_base_dir else Path("/tmp/arc_ipc")
        self.python_executable = python_executable or sys.executable
        self.harness_module = harness_module

        self._current_process: Optional[subprocess.Popen] = None
        self._current_ipc: Optional[IPCProtocol] = None
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        self._progress_callback: Optional[Callable] = None
        self._latest_metrics: Dict[str, Any] = {}
        self._latest_checkpoint: Optional[str] = None

        logger.info(f"SubprocessExecutor initialized: ipc_dir={self.ipc_base_dir}")

    def execute(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        timeout_seconds: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ExecutionResult:
        """
        Execute training in a subprocess.

        Args:
            experiment_id: Unique experiment identifier
            config: Training configuration
            timeout_seconds: Maximum runtime before timeout
            progress_callback: Called with progress updates

        Returns:
            ExecutionResult with status and collected data
        """
        self._progress_callback = progress_callback
        self._stop_event.clear()
        self._latest_metrics = {}
        self._latest_checkpoint = None

        start_time = datetime.now(timezone.utc)

        # Set up IPC
        ipc_dir = self.ipc_base_dir / experiment_id
        ipc_dir.mkdir(parents=True, exist_ok=True)
        self._current_ipc = IPCProtocol(ipc_dir, experiment_id, is_parent=True)

        # Prepare environment
        env = os.environ.copy()
        env["ARC_EXPERIMENT_ID"] = experiment_id
        env["ARC_IPC_DIR"] = str(ipc_dir)

        # Write config for subprocess
        config_path = ipc_dir / "config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        try:
            # Build command
            cmd = [
                self.python_executable,
                "-m", self.harness_module,
                "--experiment-id", experiment_id,
                "--ipc-dir", str(ipc_dir),
                "--config", str(config_path),
            ]

            logger.info(f"Starting subprocess for {experiment_id}: {' '.join(cmd)}")

            # Start subprocess
            self._current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )

            # Start monitoring thread
            self._monitor_thread = Thread(
                target=self._monitor_process,
                args=(timeout_seconds,),
                daemon=True,
            )
            self._monitor_thread.start()

            # Wait for completion or timeout
            result = self._wait_for_completion(experiment_id, timeout_seconds)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result.start_time = start_time.isoformat()
            result.end_time = end_time.isoformat()
            result.duration_seconds = duration

            return result

        except Exception as e:
            logger.error(f"Execution error for {experiment_id}: {e}")
            self._cleanup_process()

            return ExecutionResult(
                experiment_id=experiment_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time.isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )

        finally:
            self._cleanup_ipc()

    def _wait_for_completion(
        self,
        experiment_id: str,
        timeout_seconds: float
    ) -> ExecutionResult:
        """Wait for subprocess to complete."""
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            # Check if process exited
            if self._current_process is None:
                break

            exit_code = self._current_process.poll()

            if exit_code is not None:
                # Process exited
                if exit_code == 0:
                    return ExecutionResult(
                        experiment_id=experiment_id,
                        status=ExecutionStatus.COMPLETED,
                        exit_code=exit_code,
                        metrics=self._latest_metrics,
                        checkpoint_path=self._latest_checkpoint,
                    )
                else:
                    stderr = self._current_process.stderr.read().decode() if self._current_process.stderr else ""
                    return ExecutionResult(
                        experiment_id=experiment_id,
                        status=ExecutionStatus.FAILED,
                        exit_code=exit_code,
                        metrics=self._latest_metrics,
                        checkpoint_path=self._latest_checkpoint,
                        error=f"Process exited with code {exit_code}: {stderr[:500]}",
                    )

            # Check for messages from child
            self._process_messages()

            # Check if stop was requested
            if self._stop_event.is_set():
                self._terminate_process()
                return ExecutionResult(
                    experiment_id=experiment_id,
                    status=ExecutionStatus.KILLED,
                    metrics=self._latest_metrics,
                    checkpoint_path=self._latest_checkpoint,
                )

            time.sleep(self.POLL_INTERVAL)

        # Timeout - try graceful shutdown first
        logger.warning(f"Timeout for {experiment_id}, initiating graceful shutdown")
        return self._handle_timeout(experiment_id)

    def _handle_timeout(self, experiment_id: str) -> ExecutionResult:
        """Handle execution timeout with graceful shutdown."""
        if self._current_process is None:
            return ExecutionResult(
                experiment_id=experiment_id,
                status=ExecutionStatus.TIMEOUT,
                metrics=self._latest_metrics,
            )

        # Request graceful stop
        if self._current_ipc:
            try:
                self._current_ipc.send_stop(reason="timeout")
                self._current_ipc.send_checkpoint_request()
            except Exception as e:
                logger.warning(f"Failed to send stop command: {e}")

        # Send SIGTERM
        logger.info(f"Sending SIGTERM to {experiment_id}")
        self._send_signal(signal.SIGTERM)

        # Wait for graceful shutdown
        grace_start = time.time()
        while time.time() - grace_start < self.SIGTERM_GRACE_PERIOD:
            if self._current_process.poll() is not None:
                # Process exited gracefully
                return ExecutionResult(
                    experiment_id=experiment_id,
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=self._current_process.returncode,
                    metrics=self._latest_metrics,
                    checkpoint_path=self._latest_checkpoint,
                )
            time.sleep(0.5)

        # SIGKILL as last resort
        logger.warning(f"SIGTERM failed, sending SIGKILL to {experiment_id}")
        self._send_signal(signal.SIGKILL)
        self._current_process.wait(timeout=5)

        return ExecutionResult(
            experiment_id=experiment_id,
            status=ExecutionStatus.KILLED,
            exit_code=self._current_process.returncode if self._current_process else None,
            metrics=self._latest_metrics,
            checkpoint_path=self._latest_checkpoint,
        )

    def _process_messages(self) -> None:
        """Process incoming IPC messages."""
        if self._current_ipc is None:
            return

        messages = self._current_ipc.receive_all()
        for msg in messages:
            self._handle_message(msg)

    def _handle_message(self, msg: IPCMessage) -> None:
        """Handle a single IPC message."""
        if msg.msg_type == MessageType.PROGRESS:
            if self._progress_callback:
                self._progress_callback(msg.payload)
            logger.debug(f"Progress: {msg.payload}")

        elif msg.msg_type == MessageType.METRICS:
            self._latest_metrics.update(msg.payload.get("metrics", {}))
            logger.info(f"Metrics update: {self._latest_metrics}")

        elif msg.msg_type == MessageType.CHECKPOINT_SAVED:
            self._latest_checkpoint = msg.payload.get("path")
            logger.info(f"Checkpoint saved: {self._latest_checkpoint}")

        elif msg.msg_type == MessageType.ERROR:
            logger.error(f"Child error: {msg.payload}")

        elif msg.msg_type == MessageType.COMPLETE:
            self._latest_metrics.update(msg.payload.get("results", {}))
            logger.info(f"Training complete: {msg.payload}")

        elif msg.msg_type == MessageType.READY:
            logger.info("Child process ready")

    def _monitor_process(self, timeout: float) -> None:
        """Background thread to monitor process."""
        deadline = time.time() + timeout

        while not self._stop_event.is_set():
            if self._current_process is None:
                break

            if self._current_process.poll() is not None:
                break

            if time.time() > deadline:
                logger.warning("Monitor detected timeout")
                break

            time.sleep(1)

    def _send_signal(self, sig: signal.Signals) -> None:
        """Send signal to process and its children."""
        if self._current_process is None:
            return

        try:
            # Kill process group if possible (Linux/macOS)
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(self._current_process.pid), sig)
            else:
                self._current_process.send_signal(sig)
        except ProcessLookupError:
            pass  # Process already dead
        except Exception as e:
            logger.warning(f"Failed to send signal {sig}: {e}")

    def _terminate_process(self) -> None:
        """Terminate the current process."""
        if self._current_process is None:
            return

        self._send_signal(signal.SIGTERM)

        try:
            self._current_process.wait(timeout=self.SIGTERM_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            self._send_signal(signal.SIGKILL)
            self._current_process.wait(timeout=5)

    def _cleanup_process(self) -> None:
        """Clean up process resources."""
        if self._current_process is not None:
            try:
                self._current_process.kill()
                self._current_process.wait(timeout=5)
            except Exception:
                pass
            self._current_process = None

    def _cleanup_ipc(self) -> None:
        """Clean up IPC resources."""
        if self._current_ipc is not None:
            try:
                self._current_ipc.cleanup()
            except Exception:
                pass
            self._current_ipc = None

    def stop(self) -> None:
        """Request graceful stop of current execution."""
        self._stop_event.set()
        if self._current_ipc:
            try:
                self._current_ipc.send_stop(reason="user_requested")
            except Exception:
                pass

    def request_checkpoint(self) -> None:
        """Request the running process to save a checkpoint."""
        if self._current_ipc:
            try:
                self._current_ipc.send_checkpoint_request()
            except Exception as e:
                logger.warning(f"Failed to request checkpoint: {e}")

    @property
    def is_running(self) -> bool:
        """Check if a subprocess is currently running."""
        return (
            self._current_process is not None and
            self._current_process.poll() is None
        )

    @property
    def latest_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics."""
        return self._latest_metrics.copy()
