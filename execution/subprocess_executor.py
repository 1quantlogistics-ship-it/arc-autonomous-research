"""
Subprocess Executor - Process Isolation for Training Jobs

Spawns training jobs as separate subprocesses to ensure:
- Training crashes cannot kill the ARC system
- Timeout enforcement with SIGTERM→SIGKILL escalation
- Clean resource cleanup on termination
- GPU isolation via CUDA_VISIBLE_DEVICES

Author: ARC Team (Dev 1)
Created: 2025-11-26
Updated: 2025-11-26 - Unified specs with status.json/metrics.json IPC
"""

import json
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
from typing import Dict, Any, Optional, Callable, NamedTuple
from threading import Thread, Event

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution (matches status.json values)."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    CANCELLED = "cancelled"
    OOM = "oom"


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


class ExperimentHandle(NamedTuple):
    """Handle to a running experiment."""
    experiment_id: str
    process: subprocess.Popen
    work_dir: Path
    config_file: Path
    status_file: Path
    metrics_file: Path
    checkpoint_dir: Path
    stdout_log: Path
    stderr_log: Path


class SubprocessExecutor:
    """
    Executes training jobs in isolated subprocesses.

    Features:
    - Process isolation (crashes don't kill parent)
    - Timeout enforcement with graceful escalation
    - File-based IPC via status.json and metrics.json
    - Progress monitoring and metric collection
    - GPU isolation via CUDA_VISIBLE_DEVICES

    Usage:
        executor = SubprocessExecutor(
            training_gpu_id=1,
            experiments_dir="./experiments",
        )
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
        experiments_dir: Optional[Path] = None,
        training_gpu_id: int = 0,
        python_executable: Optional[str] = None,
        harness_module: str = "execution.training_harness",
        default_timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize SubprocessExecutor.

        Args:
            experiments_dir: Directory for experiment files
            training_gpu_id: GPU index for training (isolated from inference)
            python_executable: Python interpreter to use
            harness_module: Module containing training harness
            default_timeout: Default timeout in seconds
        """
        self.experiments_dir = Path(experiments_dir) if experiments_dir else Path("./experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.training_gpu_id = training_gpu_id
        self.python_executable = python_executable or sys.executable
        self.harness_module = harness_module
        self.default_timeout = default_timeout

        # Current execution state
        self._current_process: Optional[subprocess.Popen] = None
        self._current_handle: Optional[ExperimentHandle] = None
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        self._progress_callback: Optional[Callable] = None
        self._latest_metrics: Dict[str, Any] = {}
        self._latest_checkpoint: Optional[str] = None

        # Legacy IPC protocol (optional, for backwards compatibility)
        self._current_ipc = None

        logger.info(f"SubprocessExecutor initialized: experiments_dir={self.experiments_dir}, gpu={training_gpu_id}")

    def submit_experiment(
        self,
        experiment_id: str,
        experiment_config: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> ExperimentHandle:
        """
        Submit an experiment for execution (non-blocking).

        Args:
            experiment_id: Unique experiment identifier
            experiment_config: Training configuration
            timeout_seconds: Optional timeout override

        Returns:
            ExperimentHandle for monitoring the experiment
        """
        timeout = timeout_seconds or self.default_timeout

        # Set up experiment directory structure
        work_dir = self.experiments_dir / experiment_id
        work_dir.mkdir(parents=True, exist_ok=True)

        config_file = work_dir / "config.json"
        status_file = work_dir / "status.json"
        metrics_file = work_dir / "metrics.json"
        checkpoint_dir = work_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        stdout_log = work_dir / "stdout.log"
        stderr_log = work_dir / "stderr.log"

        # Write config
        config_file.write_text(json.dumps(experiment_config, indent=2))

        # Build command
        cmd = [
            self.python_executable,
            "-m", self.harness_module,
            "--experiment-id", experiment_id,
            "--config", str(config_file),
            "--status-file", str(status_file),
            "--metrics-file", str(metrics_file),
            "--checkpoint-dir", str(checkpoint_dir),
        ]

        # Environment with GPU isolation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.training_gpu_id)
        env["PYTHONPATH"] = str(Path.cwd())
        env["ARC_EXPERIMENT_ID"] = experiment_id

        logger.info(f"Starting subprocess for {experiment_id}: {' '.join(cmd)}")
        logger.info(f"GPU isolation: CUDA_VISIBLE_DEVICES={self.training_gpu_id}")

        # Spawn subprocess
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(work_dir),
            stdout=open(stdout_log, "w"),
            stderr=open(stderr_log, "w"),
            start_new_session=True,  # New process group for clean termination
        )

        handle = ExperimentHandle(
            experiment_id=experiment_id,
            process=process,
            work_dir=work_dir,
            config_file=config_file,
            status_file=status_file,
            metrics_file=metrics_file,
            checkpoint_dir=checkpoint_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

        self._current_process = process
        self._current_handle = handle

        return handle

    def wait_for_experiment(
        self,
        experiment_id: str,
        timeout_seconds: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Wait for an experiment to complete.

        Args:
            experiment_id: Experiment identifier
            timeout_seconds: Optional timeout override
            progress_callback: Called with progress updates

        Returns:
            Result dictionary with status and metrics
        """
        if self._current_handle is None or self._current_handle.experiment_id != experiment_id:
            return {"status": "error", "error": f"No running experiment with id {experiment_id}"}

        timeout = timeout_seconds or self.default_timeout
        handle = self._current_handle
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Check if process exited
            exit_code = handle.process.poll()

            if exit_code is not None:
                # Process exited - read final status
                status_data = self._read_status_file(handle.status_file)
                metrics_data = self._read_metrics_file(handle.metrics_file)

                return {
                    "status": status_data.get("status", "completed" if exit_code == 0 else "failed"),
                    "exit_code": exit_code,
                    "metrics": metrics_data.get("best", {}),
                    "latest_checkpoint": self._find_latest_checkpoint(handle.checkpoint_dir),
                    "error": status_data.get("error"),
                }

            # Read status and metrics
            status_data = self._read_status_file(handle.status_file)
            metrics_data = self._read_metrics_file(handle.metrics_file)

            if progress_callback and metrics_data.get("latest"):
                progress_callback(metrics_data["latest"])

            if self._stop_event.is_set():
                self._terminate_experiment(handle)
                return {"status": "cancelled"}

            time.sleep(self.POLL_INTERVAL)

        # Timeout - initiate graceful shutdown
        logger.warning(f"Timeout for {experiment_id}, initiating graceful shutdown")
        return self._handle_timeout(handle)

    def execute(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        timeout_seconds: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ExecutionResult:
        """
        Execute training in a subprocess (blocking).

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

        try:
            # Submit experiment
            handle = self.submit_experiment(experiment_id, config, int(timeout_seconds))

            # Start monitoring thread
            self._monitor_thread = Thread(
                target=self._monitor_process,
                args=(timeout_seconds,),
                daemon=True,
            )
            self._monitor_thread.start()

            # Wait for completion
            result_dict = self.wait_for_experiment(
                experiment_id,
                timeout_seconds=int(timeout_seconds),
                progress_callback=progress_callback,
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Map status string to enum
            status_str = result_dict.get("status", "failed")
            status = self._map_status(status_str)

            return ExecutionResult(
                experiment_id=experiment_id,
                status=status,
                exit_code=result_dict.get("exit_code"),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                metrics=result_dict.get("metrics", {}),
                error=result_dict.get("error"),
                checkpoint_path=result_dict.get("latest_checkpoint"),
            )

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

    def _read_status_file(self, status_file: Path) -> Dict[str, Any]:
        """Read status.json file."""
        if not status_file.exists():
            return {}
        try:
            return json.loads(status_file.read_text())
        except (json.JSONDecodeError, IOError):
            return {}

    def _read_metrics_file(self, metrics_file: Path) -> Dict[str, Any]:
        """Read metrics.json file."""
        if not metrics_file.exists():
            return {}
        try:
            return json.loads(metrics_file.read_text())
        except (json.JSONDecodeError, IOError):
            return {}

    def _find_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[str]:
        """Find the latest checkpoint in the directory."""
        if not checkpoint_dir.exists():
            return None

        # Check for latest.pt first
        latest = checkpoint_dir / "latest.pt"
        if latest.exists():
            return str(latest)

        # Find most recent checkpoint by modification time
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return None

        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    def _map_status(self, status_str: str) -> ExecutionStatus:
        """Map status string to ExecutionStatus enum."""
        try:
            return ExecutionStatus(status_str)
        except ValueError:
            return ExecutionStatus.FAILED

    def _handle_timeout(self, handle: ExperimentHandle) -> Dict[str, Any]:
        """Handle execution timeout with graceful shutdown."""
        # Send SIGTERM first
        logger.info(f"Sending SIGTERM to {handle.experiment_id}")
        self._send_signal(handle.process, signal.SIGTERM)

        # Wait for graceful shutdown
        grace_start = time.time()
        while time.time() - grace_start < self.SIGTERM_GRACE_PERIOD:
            if handle.process.poll() is not None:
                # Process exited gracefully
                status_data = self._read_status_file(handle.status_file)
                return {
                    "status": "timeout",
                    "exit_code": handle.process.returncode,
                    "metrics": self._read_metrics_file(handle.metrics_file).get("best", {}),
                    "latest_checkpoint": self._find_latest_checkpoint(handle.checkpoint_dir),
                }
            time.sleep(0.5)

        # SIGKILL as last resort
        logger.warning(f"SIGTERM failed, sending SIGKILL to {handle.experiment_id}")
        self._send_signal(handle.process, signal.SIGKILL)
        try:
            handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

        return {
            "status": "killed",
            "exit_code": handle.process.returncode,
            "metrics": self._read_metrics_file(handle.metrics_file).get("best", {}),
            "latest_checkpoint": self._find_latest_checkpoint(handle.checkpoint_dir),
        }

    def _terminate_experiment(self, handle: ExperimentHandle) -> None:
        """Terminate an experiment gracefully."""
        self._send_signal(handle.process, signal.SIGTERM)
        try:
            handle.process.wait(timeout=self.SIGTERM_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            self._send_signal(handle.process, signal.SIGKILL)
            handle.process.wait(timeout=5)

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

    def _send_signal(self, process: subprocess.Popen, sig: signal.Signals) -> None:
        """Send signal to process and its children."""
        try:
            # Kill process group if possible (Linux/macOS)
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass  # Process already dead
        except Exception as e:
            logger.warning(f"Failed to send signal {sig}: {e}")

    def _cleanup_process(self) -> None:
        """Clean up process resources."""
        if self._current_process is not None:
            try:
                self._current_process.kill()
                self._current_process.wait(timeout=5)
            except Exception:
                pass
            self._current_process = None
        self._current_handle = None

    def stop(self) -> None:
        """Request graceful stop of current execution."""
        self._stop_event.set()
        if self._current_handle:
            self._terminate_experiment(self._current_handle)

    def request_checkpoint(self) -> None:
        """
        Request the running process to save a checkpoint.

        Note: With file-based IPC, we send SIGUSR1 to trigger checkpoint.
        The training harness interprets this as a checkpoint request.
        """
        if self._current_process and self._current_process.poll() is None:
            try:
                # Send SIGUSR1 as checkpoint request signal
                self._current_process.send_signal(signal.SIGUSR1)
                logger.info("Checkpoint request sent via SIGUSR1")
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
        if self._current_handle:
            data = self._read_metrics_file(self._current_handle.metrics_file)
            return data.get("latest", {})
        return {}

    @property
    def current_experiment_id(self) -> Optional[str]:
        """Get the ID of the currently running experiment."""
        return self._current_handle.experiment_id if self._current_handle else None

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get status of an experiment by ID.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Status dictionary from status.json
        """
        status_file = self.experiments_dir / experiment_id / "status.json"
        return self._read_status_file(status_file)

    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get metrics of an experiment by ID.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Metrics dictionary from metrics.json
        """
        metrics_file = self.experiments_dir / experiment_id / "metrics.json"
        return self._read_metrics_file(metrics_file)

    def list_experiments(self) -> list:
        """List all experiment IDs in the experiments directory."""
        if not self.experiments_dir.exists():
            return []
        return [
            d.name for d in self.experiments_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]
