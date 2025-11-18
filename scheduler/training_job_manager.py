"""
Training Job Manager: Async training execution with status tracking and job control

Manages long-running training jobs asynchronously to prevent blocking ARC operations.

Features:
- Async job execution in background threads
- Job registry with persistent storage (jobs/active.json)
- Status tracking (queued, running, completed, failed, cancelled)
- Cancel/resume capabilities
- Graceful shutdown and rollback
- Heartbeat monitoring
- Transactional safety

This is critical for autonomous operation where training jobs may run 10-60 minutes.
"""

import json
import logging
import threading
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from queue import Queue
import shutil

from config import get_settings

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Training job states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMING = "resuming"


@dataclass
class TrainingJob:
    """Training job metadata."""
    job_id: str
    experiment_id: str
    task_type: str  # segmentation or classification
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoint_dir: str = ""
    log_dir: str = ""
    gpu_id: Optional[int] = None
    process_id: Optional[int] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_epoch: int = 0
    total_epochs: int = 0
    last_heartbeat: Optional[str] = None
    cycle_id: int = 0
    # Retry and recovery fields
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 60  # Exponential backoff base
    timeout_seconds: Optional[int] = None  # None = no timeout
    auto_resume: bool = True
    last_checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """Create from dictionary."""
        data = data.copy()
        data['status'] = JobStatus(data['status'])
        return cls(**data)


class TrainingJobManager:
    """
    Manages asynchronous training job execution.

    Features:
    - Non-blocking job submission
    - Background worker threads
    - Persistent job registry
    - Status tracking and heartbeat monitoring
    - Graceful cancellation and rollback
    """

    def __init__(
        self,
        registry_path: Optional[str] = None,
        max_concurrent_jobs: int = 2,
        heartbeat_interval: int = 30
    ):
        """
        Initialize job manager.

        Args:
            registry_path: Path to job registry file
            max_concurrent_jobs: Maximum concurrent training jobs
            heartbeat_interval: Seconds between heartbeat updates
        """
        settings = get_settings()

        # Job registry file
        if registry_path is None:
            registry_dir = settings.home / "jobs"
            registry_dir.mkdir(parents=True, exist_ok=True)
            registry_path = str(registry_dir / "active.json")

        self.registry_path = Path(registry_path)

        # Job storage
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue: Queue = Queue()

        # Configuration
        self.max_concurrent_jobs = max_concurrent_jobs
        self.heartbeat_interval = heartbeat_interval

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()

        # Load existing jobs from registry
        self._load_registry()

        # Start worker threads
        self._start_workers()

        logger.info(f"TrainingJobManager initialized: {max_concurrent_jobs} workers")

    def submit_job(
        self,
        job_id: str,
        experiment_id: str,
        task_type: str,
        training_function: Callable,
        training_args: Dict[str, Any],
        checkpoint_dir: str,
        log_dir: str,
        gpu_id: Optional[int] = None,
        cycle_id: int = 0,
        max_retries: int = 3,
        timeout_seconds: Optional[int] = None,
        auto_resume: bool = True
    ) -> TrainingJob:
        """
        Submit training job for async execution.

        Args:
            job_id: Unique job identifier
            experiment_id: Experiment identifier
            task_type: Task type (segmentation or classification)
            training_function: Function to call for training
            training_args: Arguments to pass to training function
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            gpu_id: GPU to use
            cycle_id: Research cycle ID
            max_retries: Maximum number of retry attempts on failure (default: 3)
            timeout_seconds: Job timeout in seconds (None = no timeout)
            auto_resume: Whether to auto-resume from last checkpoint on retry

        Returns:
            TrainingJob instance
        """
        logger.info(f"Submitting job: {job_id} (max_retries={max_retries}, timeout={timeout_seconds})")

        # Create job record
        job = TrainingJob(
            job_id=job_id,
            experiment_id=experiment_id,
            task_type=task_type,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow().isoformat(),
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            gpu_id=gpu_id,
            cycle_id=cycle_id,
            total_epochs=training_args.get('epochs', 0),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            auto_resume=auto_resume
        )

        # Store job
        with self.lock:
            self.jobs[job_id] = job
            self._save_registry()

        # Queue job for execution
        self.job_queue.put((job, training_function, training_args))

        logger.info(f"Job queued: {job_id}")
        return job

    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get current job status.

        Args:
            job_id: Job identifier

        Returns:
            TrainingJob or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(
        self,
        status_filter: Optional[JobStatus] = None
    ) -> List[TrainingJob]:
        """
        List all jobs, optionally filtered by status.

        Args:
            status_filter: Filter by status

        Returns:
            List of TrainingJob instances
        """
        with self.lock:
            jobs = list(self.jobs.values())

            if status_filter:
                jobs = [j for j in jobs if j.status == status_filter]

            # Sort by created_at descending
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            return jobs

    def cancel_job(self, job_id: str, rollback: bool = True) -> bool:
        """
        Cancel running job.

        Args:
            job_id: Job to cancel
            rollback: Whether to delete checkpoint artifacts

        Returns:
            True if cancelled successfully
        """
        logger.info(f"Cancelling job: {job_id}")

        with self.lock:
            job = self.jobs.get(job_id)

            if not job:
                logger.error(f"Job not found: {job_id}")
                return False

            if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
                logger.warning(f"Job {job_id} cannot be cancelled (status: {job.status.value})")
                return False

            # Kill process if running
            if job.process_id and job.status == JobStatus.RUNNING:
                try:
                    import psutil
                    process = psutil.Process(job.process_id)
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info(f"Process {job.process_id} terminated")
                except Exception as e:
                    logger.error(f"Failed to kill process {job.process_id}: {e}")

            # Update status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow().isoformat()

            # Rollback artifacts if requested
            if rollback:
                self._rollback_artifacts(job)

            self._save_registry()

        logger.info(f"Job cancelled: {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """
        Resume cancelled or failed job.

        Args:
            job_id: Job to resume

        Returns:
            True if resumed successfully
        """
        logger.info(f"Resuming job: {job_id}")

        with self.lock:
            job = self.jobs.get(job_id)

            if not job:
                logger.error(f"Job not found: {job_id}")
                return False

            if job.status not in [JobStatus.CANCELLED, JobStatus.FAILED]:
                logger.warning(f"Job {job_id} cannot be resumed (status: {job.status.value})")
                return False

            # Update status
            job.status = JobStatus.RESUMING
            job.started_at = None
            job.completed_at = None
            job.process_id = None
            job.return_code = None
            job.error_message = None

            self._save_registry()

        # TODO: Re-queue job with checkpoint resumption
        logger.warning("Job resume not fully implemented yet")

        return True

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Remove completed/failed jobs older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 3600)
        removed = 0

        with self.lock:
            jobs_to_remove = []

            for job_id, job in self.jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    created_time = datetime.fromisoformat(job.created_at).timestamp()

                    if created_time < cutoff:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                removed += 1

            if removed > 0:
                self._save_registry()

        logger.info(f"Cleaned up {removed} old jobs")
        return removed

    def shutdown(self):
        """Gracefully shutdown job manager."""
        logger.info("Shutting down TrainingJobManager")

        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        # Save final registry
        self._save_registry()

        logger.info("TrainingJobManager shutdown complete")

    # Private methods

    def _start_workers(self):
        """Start background worker threads."""
        self.running = True

        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TrainingWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker: TrainingWorker-{i}")

    def _worker_loop(self):
        """Background worker loop."""
        while self.running:
            try:
                # Get job from queue (blocking with timeout)
                job, training_fn, training_args = self.job_queue.get(timeout=1)

                # Execute job
                self._execute_job(job, training_fn, training_args)

                self.job_queue.task_done()

            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logger.error(f"Worker error: {e}")

    def _execute_job(
        self,
        job: TrainingJob,
        training_fn: Callable,
        training_args: Dict[str, Any]
    ):
        """
        Execute training job with retry logic and timeout support.

        Args:
            job: Job to execute
            training_fn: Training function
            training_args: Training arguments
        """
        logger.info(f"Executing job: {job.job_id} (attempt {job.retry_count + 1}/{job.max_retries + 1})")

        # Update status to running
        with self.lock:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow().isoformat()
            job.last_heartbeat = datetime.utcnow().isoformat()
            self._save_registry()

        try:
            # Execute training function with optional timeout
            if job.timeout_seconds:
                result = self._execute_with_timeout(
                    training_fn,
                    training_args,
                    job.timeout_seconds,
                    job
                )
            else:
                result = training_fn(**training_args)

            # Save last checkpoint if available
            if result.get('best_checkpoint_path'):
                with self.lock:
                    job.last_checkpoint_path = result['best_checkpoint_path']
                    self._save_registry()

            # Update job with results
            with self.lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow().isoformat()
                job.return_code = result.get('return_code', 0)
                job.progress = 1.0
                job.current_epoch = job.total_epochs
                self._save_registry()

            logger.info(f"Job completed: {job.job_id}")

        except Exception as e:
            logger.error(f"Job failed: {job.job_id} - {e}")

            # Determine if we should retry
            should_retry = job.retry_count < job.max_retries

            if should_retry:
                # Update retry count and schedule retry
                with self.lock:
                    job.retry_count += 1
                    job.error_message = f"Attempt {job.retry_count}/{job.max_retries + 1} failed: {str(e)}"
                    job.status = JobStatus.QUEUED  # Back to queued for retry
                    self._save_registry()

                # Calculate exponential backoff delay
                delay = job.retry_delay_seconds * (2 ** (job.retry_count - 1))
                logger.info(f"Retrying job {job.job_id} in {delay} seconds (attempt {job.retry_count + 1}/{job.max_retries + 1})")

                # Wait before retry
                time.sleep(delay)

                # Modify training args for resume if auto_resume enabled
                retry_args = training_args.copy()
                if job.auto_resume and job.last_checkpoint_path:
                    logger.info(f"Auto-resuming from checkpoint: {job.last_checkpoint_path}")
                    retry_args['resume_from_checkpoint'] = job.last_checkpoint_path

                # Re-queue job
                self.job_queue.put((job, training_fn, retry_args))

            else:
                # No more retries - mark as failed
                with self.lock:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.utcnow().isoformat()
                    job.error_message = f"Failed after {job.retry_count + 1} attempts: {str(e)}"
                    self._save_registry()

                # Rollback artifacts on final failure
                logger.warning(f"Job {job.job_id} exhausted all retries - rolling back artifacts")
                self._rollback_artifacts(job)

    def _execute_with_timeout(
        self,
        training_fn: Callable,
        training_args: Dict[str, Any],
        timeout_seconds: int,
        job: TrainingJob
    ) -> Dict[str, Any]:
        """
        Execute training function with timeout.

        Args:
            training_fn: Training function to execute
            training_args: Arguments for training function
            timeout_seconds: Timeout in seconds
            job: Job being executed

        Returns:
            Training result dict

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        import concurrent.futures

        logger.info(f"Executing job {job.job_id} with {timeout_seconds}s timeout")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(training_fn, **training_args)

            try:
                result = future.result(timeout=timeout_seconds)
                return result

            except concurrent.futures.TimeoutError:
                logger.error(f"Job {job.job_id} exceeded timeout of {timeout_seconds}s")

                # Try to gracefully cancel
                future.cancel()

                raise TimeoutError(
                    f"Training exceeded timeout of {timeout_seconds}s "
                    f"(epoch {job.current_epoch}/{job.total_epochs})"
                )

    def auto_clean_broken_experiments(self) -> int:
        """
        Automatically clean up broken experiment directories.

        Identifies and removes experiments from failed jobs that:
        - Have no valid checkpoints
        - Have empty or corrupt directories
        - Were cancelled or failed without retries remaining

        Returns:
            Number of experiments cleaned
        """
        logger.info("Running auto-clean for broken experiments")

        cleaned = 0

        with self.lock:
            # Get all failed or cancelled jobs
            broken_jobs = [
                job for job in self.jobs.values()
                if job.status in [JobStatus.FAILED, JobStatus.CANCELLED]
                and job.retry_count >= job.max_retries
            ]

        for job in broken_jobs:
            try:
                exp_dir = Path(job.checkpoint_dir).parent

                if not exp_dir.exists():
                    continue

                # Check if experiment directory is broken
                is_broken = False

                # Check 1: No valid checkpoints
                checkpoint_dir = Path(job.checkpoint_dir)
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob("*.pt"))
                    if len(checkpoints) == 0:
                        is_broken = True
                        logger.info(f"Broken experiment {job.experiment_id}: No checkpoints found")

                # Check 2: Empty experiment directory
                if exp_dir.exists():
                    total_size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
                    if total_size < 1024:  # Less than 1KB
                        is_broken = True
                        logger.info(f"Broken experiment {job.experiment_id}: Directory nearly empty")

                # Clean if broken
                if is_broken:
                    logger.warning(f"Cleaning broken experiment: {job.experiment_id}")
                    shutil.rmtree(exp_dir)
                    cleaned += 1

            except Exception as e:
                logger.error(f"Failed to clean experiment {job.experiment_id}: {e}")

        logger.info(f"Auto-clean complete: {cleaned} broken experiments removed")
        return cleaned

    def _rollback_artifacts(self, job: TrainingJob):
        """
        Rollback checkpoint artifacts on failure/cancellation.

        Args:
            job: Job to rollback
        """
        logger.info(f"Rolling back artifacts for job: {job.job_id}")

        try:
            # Remove checkpoint directory
            checkpoint_dir = Path(job.checkpoint_dir)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Removed checkpoint dir: {checkpoint_dir}")

            # Keep logs for debugging

        except Exception as e:
            logger.error(f"Rollback failed for job {job.job_id}: {e}")

    def _load_registry(self):
        """Load job registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                for job_id, job_data in data.items():
                    self.jobs[job_id] = TrainingJob.from_dict(job_data)

                logger.info(f"Loaded {len(self.jobs)} jobs from registry")

            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save job registry to disk."""
        try:
            data = {
                job_id: job.to_dict()
                for job_id, job in self.jobs.items()
            }

            # Atomic write
            temp_path = self.registry_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.registry_path)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


# Global singleton instance
_job_manager: Optional[TrainingJobManager] = None
_lock = threading.Lock()


def get_job_manager(
    registry_path: Optional[str] = None,
    max_concurrent_jobs: int = 2
) -> TrainingJobManager:
    """
    Get global job manager instance (singleton).

    Args:
        registry_path: Path to job registry
        max_concurrent_jobs: Max concurrent jobs

    Returns:
        TrainingJobManager instance
    """
    global _job_manager

    if _job_manager is None:
        with _lock:
            if _job_manager is None:
                _job_manager = TrainingJobManager(
                    registry_path=registry_path,
                    max_concurrent_jobs=max_concurrent_jobs
                )

    return _job_manager


def shutdown_job_manager():
    """Shutdown global job manager."""
    global _job_manager

    if _job_manager:
        _job_manager.shutdown()
        _job_manager = None
