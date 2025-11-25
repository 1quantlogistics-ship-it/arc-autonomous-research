"""
Job Scheduler Package

GPU-aware job scheduling for autonomous experiment execution.
Also includes timing utilities for profiling (Phase F).
"""

# Lazy imports to avoid circular dependency issues
def __getattr__(name):
    if name in ("JobScheduler", "get_job_scheduler", "reset_job_scheduler"):
        from scheduler.job_scheduler import JobScheduler, get_job_scheduler, reset_job_scheduler
        return {
            "JobScheduler": JobScheduler,
            "get_job_scheduler": get_job_scheduler,
            "reset_job_scheduler": reset_job_scheduler,
        }[name]
    elif name in ("CycleProfiler", "AsyncBatchOptimizer", "OperationTimer", "PerformanceTracker", "timed_async", "TimingMetric"):
        from scheduler.timing import CycleProfiler, AsyncBatchOptimizer, OperationTimer, PerformanceTracker, timed_async, TimingMetric
        return {
            "CycleProfiler": CycleProfiler,
            "AsyncBatchOptimizer": AsyncBatchOptimizer,
            "OperationTimer": OperationTimer,
            "PerformanceTracker": PerformanceTracker,
            "timed_async": timed_async,
            "TimingMetric": TimingMetric,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "JobScheduler",
    "get_job_scheduler",
    "reset_job_scheduler",
    # Phase F timing utilities
    "CycleProfiler",
    "AsyncBatchOptimizer",
    "OperationTimer",
    "PerformanceTracker",
    "timed_async",
    "TimingMetric",
]
