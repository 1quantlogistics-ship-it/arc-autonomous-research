"""
Async timing utilities for profiling experiment cycles.

Phase F - Infrastructure & Stability Track
"""
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """Single timing measurement."""
    name: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


class CycleProfiler:
    """
    Profiles async operations within experiment cycles.
    Thread-safe for concurrent operations.
    """

    def __init__(self, cycle_id: str):
        self.cycle_id = cycle_id
        self.metrics: Dict[str, List[TimingMetric]] = defaultdict(list)
        self._start_time = time.perf_counter()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def measure(self, operation_name: str):
        """Context manager to measure operation duration."""
        metric = TimingMetric(name=operation_name, start_time=time.perf_counter())
        try:
            yield metric
        finally:
            metric.end_time = time.perf_counter()
            async with self._lock:
                self.metrics[operation_name].append(metric)

            # Log slow operations (>5s)
            if metric.duration_s > 5.0:
                logger.warning(
                    f"[{self.cycle_id}] Slow operation: {operation_name} "
                    f"took {metric.duration_s:.2f}s"
                )

    def get_summary(self) -> Dict:
        """Get timing summary for the cycle."""
        total_time = time.perf_counter() - self._start_time

        summary = {
            'cycle_id': self.cycle_id,
            'total_time_s': total_time,
            'operations': {}
        }

        for op_name, metrics in self.metrics.items():
            durations = [m.duration_s for m in metrics]
            summary['operations'][op_name] = {
                'count': len(metrics),
                'total_s': sum(durations),
                'avg_s': sum(durations) / len(durations) if durations else 0,
                'max_s': max(durations) if durations else 0,
                'min_s': min(durations) if durations else 0,
                'pct_of_cycle': (sum(durations) / total_time) * 100 if total_time > 0 else 0
            }

        # Identify bottlenecks (>20% of cycle time)
        summary['bottlenecks'] = [
            op for op, stats in summary['operations'].items()
            if stats['pct_of_cycle'] > 20
        ]

        return summary

    def log_summary(self):
        """Log timing summary."""
        summary = self.get_summary()
        logger.info(f"=== Cycle Timing Summary: {self.cycle_id} ===")
        logger.info(f"Total cycle time: {summary['total_time_s']:.2f}s")

        for op, stats in sorted(
            summary['operations'].items(),
            key=lambda x: x[1]['total_s'],
            reverse=True
        ):
            logger.info(
                f"  {op}: {stats['total_s']:.2f}s ({stats['pct_of_cycle']:.1f}%) "
                f"[calls: {stats['count']}, avg: {stats['avg_s']:.2f}s]"
            )

        if summary['bottlenecks']:
            logger.warning(f"Bottlenecks detected: {summary['bottlenecks']}")


def timed_async(operation_name: str = None):
    """Decorator for timing async functions."""
    def decorator(func):
        name = operation_name or func.__name__

        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if duration > 1.0:
                    logger.debug(f"{name} completed in {duration:.2f}s")

        return wrapper
    return decorator


class AsyncBatchOptimizer:
    """
    Optimizes async operations by batching concurrent calls.
    Use for operations that can be parallelized.
    """

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def gather_with_limit(self, coroutines: list):
        """Run coroutines with concurrency limit."""
        async def limited(coro):
            async with self.semaphore:
                return await coro

        return await asyncio.gather(*[limited(c) for c in coroutines])

    async def map_async(self, func: Callable, items: list) -> List[Any]:
        """Map async function over items with concurrency limit."""
        return await self.gather_with_limit([func(item) for item in items])


class OperationTimer:
    """
    Simple context manager for timing synchronous operations.
    """

    def __init__(self, operation_name: str, log_threshold_s: float = 1.0):
        self.operation_name = operation_name
        self.log_threshold_s = log_threshold_s
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.duration_s > self.log_threshold_s:
            logger.debug(f"{self.operation_name} completed in {self.duration_s:.2f}s")
        return False

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000


class PerformanceTracker:
    """
    Tracks performance metrics across multiple cycles.
    Useful for identifying trends and regressions.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.cycle_summaries: List[Dict] = []

    def record_cycle(self, summary: Dict):
        """Record a cycle summary."""
        self.cycle_summaries.append(summary)
        if len(self.cycle_summaries) > self.max_history:
            self.cycle_summaries.pop(0)

    def get_operation_trend(self, operation_name: str) -> Dict:
        """Get trend data for a specific operation."""
        times = []
        for summary in self.cycle_summaries:
            if operation_name in summary.get('operations', {}):
                times.append(summary['operations'][operation_name]['avg_s'])

        if not times:
            return {'operation': operation_name, 'data_points': 0}

        return {
            'operation': operation_name,
            'data_points': len(times),
            'avg_s': sum(times) / len(times),
            'min_s': min(times),
            'max_s': max(times),
            'trend': 'improving' if len(times) > 1 and times[-1] < times[0] else 'stable'
        }

    def get_bottleneck_frequency(self) -> Dict[str, int]:
        """Get frequency of each operation appearing as bottleneck."""
        frequency = defaultdict(int)
        for summary in self.cycle_summaries:
            for bottleneck in summary.get('bottlenecks', []):
                frequency[bottleneck] += 1
        return dict(frequency)
