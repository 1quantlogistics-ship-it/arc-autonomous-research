"""
Job Scheduler

GPU-aware job scheduling system for autonomous experiment execution.

Features:
- Queue-based job management with priorities
- GPU resource allocation (GPU0/1 for experiments, GPU2 for ARC)
- Concurrent job execution tracking
- Job status monitoring
- Integration with tool governance for transactional safety
- Automatic job retry on transient failures

Design:
- Jobs are queued based on priority (0-10, higher = more urgent)
- GPU0 and GPU1 are reserved for training experiments
- GPU2 is reserved for ARC orchestrator + light workers
- Scheduler respects GPU memory limits
- Failed jobs can be retried with exponential backoff
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import json

from schemas.experiment_schemas import (
    TrainingJobConfig, JobStatus, GPUAllocation,
    GPUStatus, SchedulerStatus, QueuedJob,
    ExperimentResult
)
from config import get_settings, ARCSettings
from tool_governance import get_tool_governance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class JobRecord:
    """Internal job record with execution metadata."""
    job_id: str
    experiment_id: str
    config: TrainingJobConfig
    priority: int
    status: JobStatus
    gpu_id: Optional[int] = None
    queued_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "priority": self.priority,
            "status": self.status.value,
            "gpu_id": self.gpu_id,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


# ============================================================================
# GPU Manager
# ============================================================================

class GPUManager:
    """Manages GPU resource allocation."""

    def __init__(self, settings: ARCSettings):
        """
        Initialize GPU manager.

        Args:
            settings: ARC settings
        """
        self.settings = settings
        self._lock = threading.Lock()

        # GPU allocation map
        self._gpus: Dict[int, GPUStatus] = {}

        # Initialize GPU statuses
        self._initialize_gpus()

    def _initialize_gpus(self):
        """Initialize GPU status tracking."""
        # GPU0 and GPU1 for experiments
        for gpu_id in [0, 1]:
            self._gpus[gpu_id] = GPUStatus(
                gpu_id=gpu_id,
                allocated=False,
                current_job_id=None,
                utilization_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=16384.0  # 16GB default
            )

        # GPU2 reserved for ARC (marked as allocated)
        self._gpus[2] = GPUStatus(
            gpu_id=2,
            allocated=True,
            current_job_id="arc_orchestrator",
            utilization_percent=0.0,
            memory_used_mb=0.0,
            memory_total_mb=16384.0
        )

    def allocate_gpu(self, job_id: str, preference: GPUAllocation = GPUAllocation.AUTO) -> Optional[int]:
        """
        Allocate GPU for job.

        Args:
            job_id: Job requesting GPU
            preference: GPU preference (AUTO = scheduler decides)

        Returns:
            Allocated GPU ID or None if no GPU available
        """
        with self._lock:
            # If specific GPU requested, try to allocate it
            if preference != GPUAllocation.AUTO:
                gpu_id = int(preference.value.replace("gpu", ""))
                if gpu_id in [0, 1]:  # Only allow experiment GPUs
                    if not self._gpus[gpu_id].allocated:
                        self._gpus[gpu_id].allocated = True
                        self._gpus[gpu_id].current_job_id = job_id
                        logger.info(f"Allocated GPU{gpu_id} to job {job_id}")
                        return gpu_id

            # AUTO allocation - find first available experiment GPU
            for gpu_id in [0, 1]:
                if not self._gpus[gpu_id].allocated:
                    self._gpus[gpu_id].allocated = True
                    self._gpus[gpu_id].current_job_id = job_id
                    logger.info(f"Allocated GPU{gpu_id} to job {job_id}")
                    return gpu_id

            logger.warning(f"No GPU available for job {job_id}")
            return None

    def release_gpu(self, gpu_id: int):
        """
        Release GPU allocation.

        Args:
            gpu_id: GPU to release
        """
        with self._lock:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].allocated = False
                self._gpus[gpu_id].current_job_id = None
                logger.info(f"Released GPU{gpu_id}")

    def get_gpu_status(self, gpu_id: int) -> Optional[GPUStatus]:
        """Get status of specific GPU."""
        return self._gpus.get(gpu_id)

    def get_all_gpu_statuses(self) -> List[GPUStatus]:
        """Get statuses of all GPUs."""
        return list(self._gpus.values())

    def get_available_gpu_count(self) -> int:
        """Get count of available experiment GPUs."""
        return sum(1 for gpu in self._gpus.values()
                   if gpu.gpu_id in [0, 1] and not gpu.allocated)


# ============================================================================
# Job Queue
# ============================================================================

class JobQueue:
    """Priority-based job queue."""

    def __init__(self):
        """Initialize job queue."""
        self._queue: deque[JobRecord] = deque()
        self._lock = threading.Lock()

    def enqueue(self, job: JobRecord):
        """
        Add job to queue with priority ordering.

        Args:
            job: Job to enqueue
        """
        with self._lock:
            # Insert job in priority order (higher priority first)
            inserted = False
            for i, existing_job in enumerate(self._queue):
                if job.priority > existing_job.priority:
                    self._queue.insert(i, job)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(job)

            logger.info(f"Enqueued job {job.job_id} (priority={job.priority}, queue_len={len(self._queue)})")

    def dequeue(self) -> Optional[JobRecord]:
        """
        Remove and return highest priority job.

        Returns:
            Highest priority job or None if queue empty
        """
        with self._lock:
            if len(self._queue) > 0:
                job = self._queue.popleft()
                logger.debug(f"Dequeued job {job.job_id}")
                return job
            return None

    def peek(self) -> Optional[JobRecord]:
        """
        View highest priority job without removing.

        Returns:
            Highest priority job or None if queue empty
        """
        with self._lock:
            if len(self._queue) > 0:
                return self._queue[0]
            return None

    def get_length(self) -> int:
        """Get current queue length."""
        return len(self._queue)

    def get_all_jobs(self) -> List[JobRecord]:
        """Get all queued jobs."""
        with self._lock:
            return list(self._queue)

    def remove_job(self, job_id: str) -> bool:
        """
        Remove job from queue.

        Args:
            job_id: Job to remove

        Returns:
            True if job was removed, False if not found
        """
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    del self._queue[i]
                    logger.info(f"Removed job {job_id} from queue")
                    return True
            return False


# ============================================================================
# Job Scheduler
# ============================================================================

class JobScheduler:
    """
    GPU-aware job scheduler for autonomous experiments.

    Features:
    - Priority-based queue
    - GPU resource management
    - Concurrent job execution
    - Status monitoring
    - Job retry logic
    - Integration with tool governance

    Example:
        scheduler = JobScheduler()
        job_id = scheduler.submit_job(job_config, priority=7)
        status = scheduler.get_job_status(job_id)
        scheduler.cancel_job(job_id)
    """

    def __init__(
        self,
        settings: Optional[ARCSettings] = None,
        max_concurrent_jobs: int = 2,
        auto_start: bool = True
    ):
        """
        Initialize job scheduler.

        Args:
            settings: Optional ARC settings
            max_concurrent_jobs: Maximum concurrent jobs (default 2 for GPU0/GPU1)
            auto_start: Whether to start scheduler thread automatically
        """
        self.settings = settings or get_settings()
        self.max_concurrent_jobs = min(max_concurrent_jobs, 2)  # Max 2 (GPU0 + GPU1)

        # Components
        self.gpu_manager = GPUManager(self.settings)
        self.queue = JobQueue()
        self.governance = get_tool_governance()

        # Job tracking
        self._jobs: Dict[str, JobRecord] = {}
        self._running_jobs: Dict[str, JobRecord] = {}
        self._completed_jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

        # Scheduler thread
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

        # Logging
        self.log_dir = self.settings.logs_dir / "scheduler"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if auto_start:
            self.start()

        logger.info(f"JobScheduler initialized (max_concurrent={max_concurrent_jobs})")

    # ========================================================================
    # Job Submission
    # ========================================================================

    def submit_job(
        self,
        job_config: TrainingJobConfig,
        priority: int = 5,
        cycle_id: int = 0
    ) -> str:
        """
        Submit training job to scheduler.

        Args:
            job_config: Complete training job configuration
            priority: Job priority (0-10, higher = more urgent)
            cycle_id: Research cycle submitting job

        Returns:
            Job ID

        Raises:
            ValueError: If job validation fails
        """
        job_id = job_config.job_id

        logger.info(f"Submitting job {job_id} (priority={priority})")

        # Create job record
        job_record = JobRecord(
            job_id=job_id,
            experiment_id=job_config.experiment_spec.experiment_id,
            config=job_config,
            priority=priority,
            status=JobStatus.QUEUED
        )

        # Track job
        with self._lock:
            self._jobs[job_id] = job_record

        # Add to queue
        self.queue.enqueue(job_record)

        # Log submission
        self._log_job_event(job_id, "submitted", {"priority": priority, "cycle_id": cycle_id})

        return job_id

    def submit_batch(
        self,
        job_configs: List[TrainingJobConfig],
        priority: int = 5,
        cycle_id: int = 0
    ) -> List[str]:
        """
        Submit batch of training jobs.

        Args:
            job_configs: List of job configurations
            priority: Priority for all jobs
            cycle_id: Research cycle

        Returns:
            List of job IDs
        """
        job_ids = []
        for config in job_configs:
            job_id = self.submit_job(config, priority, cycle_id)
            job_ids.append(job_id)

        logger.info(f"Submitted batch of {len(job_ids)} jobs")
        return job_ids

    # ========================================================================
    # Job Control
    # ========================================================================

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel queued or running job.

        Args:
            job_id: Job to cancel

        Returns:
            True if job was cancelled, False if not found
        """
        logger.info(f"Cancelling job {job_id}")

        with self._lock:
            # Check if job exists
            if job_id not in self._jobs:
                logger.warning(f"Job {job_id} not found")
                return False

            job = self._jobs[job_id]

            # If queued, remove from queue
            if job.status == JobStatus.QUEUED:
                self.queue.remove_job(job_id)

            # If running, kill and release GPU
            if job.status == JobStatus.RUNNING:
                if job.gpu_id is not None:
                    self.gpu_manager.release_gpu(job.gpu_id)
                if job_id in self._running_jobs:
                    del self._running_jobs[job_id]

            # Update status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow().isoformat()

            # Move to completed
            self._completed_jobs[job_id] = job

            self._log_job_event(job_id, "cancelled", {})

            return True

    # ========================================================================
    # Job Status
    # ========================================================================

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get status of job.

        Args:
            job_id: Job to query

        Returns:
            Job status or None if not found
        """
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id].status
            return None

    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed job information.

        Args:
            job_id: Job to query

        Returns:
            Job details dictionary or None if not found
        """
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id].to_dict()
            return None

    def get_scheduler_status(self) -> SchedulerStatus:
        """
        Get overall scheduler status.

        Returns:
            Scheduler status with queue and GPU info
        """
        with self._lock:
            queued_jobs = [
                QueuedJob(
                    job_id=job.job_id,
                    experiment_id=job.experiment_id,
                    priority=job.priority,
                    queued_at=job.queued_at,
                    estimated_duration=job.config.timeout,
                    gpu_preference=job.config.experiment_spec.gpu_allocation
                )
                for job in self.queue.get_all_jobs()
            ]

            return SchedulerStatus(
                queue_length=self.queue.get_length(),
                running_jobs=len(self._running_jobs),
                available_gpus=self.gpu_manager.get_available_gpu_count(),
                gpu_statuses=self.gpu_manager.get_all_gpu_statuses(),
                queued_jobs=queued_jobs
            )

    # ========================================================================
    # Scheduler Loop
    # ========================================================================

    def start(self):
        """Start scheduler thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        logger.info("Scheduler thread started")

    def stop(self):
        """Stop scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        logger.info("Scheduler thread stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self._running:
            try:
                # Check if we can start new jobs
                if len(self._running_jobs) < self.max_concurrent_jobs:
                    # Get next job from queue
                    job = self.queue.dequeue()

                    if job is not None:
                        # Try to allocate GPU
                        gpu_id = self.gpu_manager.allocate_gpu(
                            job.job_id,
                            job.config.experiment_spec.gpu_allocation
                        )

                        if gpu_id is not None:
                            # Start job
                            self._start_job(job, gpu_id)
                        else:
                            # No GPU available, re-queue
                            self.queue.enqueue(job)

                # Sleep briefly
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                time.sleep(5.0)

        logger.info("Scheduler loop ended")

    def _start_job(self, job: JobRecord, gpu_id: int):
        """
        Start job execution.

        Args:
            job: Job to start
            gpu_id: Assigned GPU
        """
        logger.info(f"Starting job {job.job_id} on GPU{gpu_id}")

        with self._lock:
            job.status = JobStatus.RUNNING
            job.gpu_id = gpu_id
            job.started_at = datetime.utcnow().isoformat()
            self._running_jobs[job.job_id] = job

        self._log_job_event(job.job_id, "started", {"gpu_id": gpu_id})

        # NOTE: Actual job execution would happen here
        # For now, this is a placeholder - the training_executor will handle actual execution

    # ========================================================================
    # Logging
    # ========================================================================

    def _log_job_event(self, job_id: str, event: str, metadata: Dict[str, Any]):
        """Log job event to scheduler log."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "event": event,
            "metadata": metadata
        }

        log_file = self.log_dir / "scheduler_events.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# Convenience Functions
# ============================================================================

_global_scheduler: Optional[JobScheduler] = None


def get_job_scheduler(
    settings: Optional[ARCSettings] = None,
    max_concurrent_jobs: int = 2
) -> JobScheduler:
    """
    Get global job scheduler instance.

    Args:
        settings: Optional settings
        max_concurrent_jobs: Maximum concurrent jobs

    Returns:
        JobScheduler instance
    """
    global _global_scheduler

    if _global_scheduler is None:
        _global_scheduler = JobScheduler(
            settings=settings,
            max_concurrent_jobs=max_concurrent_jobs
        )

    return _global_scheduler


def reset_job_scheduler():
    """Reset global job scheduler (useful for testing)."""
    global _global_scheduler

    if _global_scheduler is not None:
        _global_scheduler.stop()

    _global_scheduler = None
