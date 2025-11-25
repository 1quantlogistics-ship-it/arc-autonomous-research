"""
Tests for async timing utilities.

Phase F - Infrastructure & Stability Track
"""
import pytest
import asyncio
from scheduler.timing import (
    CycleProfiler, timed_async, AsyncBatchOptimizer,
    OperationTimer, PerformanceTracker, TimingMetric
)


@pytest.mark.asyncio
async def test_profiler_measures_operations():
    """Profiler should measure operation durations."""
    profiler = CycleProfiler("test-cycle")

    async with profiler.measure("fast_op"):
        await asyncio.sleep(0.01)

    async with profiler.measure("slow_op"):
        await asyncio.sleep(0.05)

    summary = profiler.get_summary()
    assert "fast_op" in summary['operations']
    assert "slow_op" in summary['operations']
    assert summary['operations']['slow_op']['total_s'] > summary['operations']['fast_op']['total_s']


@pytest.mark.asyncio
async def test_profiler_multiple_calls():
    """Profiler should aggregate multiple calls to same operation."""
    profiler = CycleProfiler("test-cycle")

    for _ in range(3):
        async with profiler.measure("repeated_op"):
            await asyncio.sleep(0.01)

    summary = profiler.get_summary()
    assert summary['operations']['repeated_op']['count'] == 3


@pytest.mark.asyncio
async def test_profiler_bottleneck_detection():
    """Profiler should identify bottlenecks (>20% of cycle time)."""
    profiler = CycleProfiler("test-cycle")

    # Create a dominant operation
    async with profiler.measure("bottleneck_op"):
        await asyncio.sleep(0.1)

    async with profiler.measure("fast_op"):
        await asyncio.sleep(0.01)

    summary = profiler.get_summary()
    assert "bottleneck_op" in summary['bottlenecks']


@pytest.mark.asyncio
async def test_batch_optimizer_limits_concurrency():
    """Batch optimizer should limit concurrent operations."""
    optimizer = AsyncBatchOptimizer(max_concurrent=2)
    running = []
    max_running = 0

    async def track_concurrency(i):
        nonlocal max_running
        running.append(i)
        max_running = max(max_running, len(running))
        await asyncio.sleep(0.05)
        running.remove(i)
        return i

    results = await optimizer.map_async(track_concurrency, list(range(5)))

    assert max_running <= 2
    assert sorted(results) == list(range(5))


@pytest.mark.asyncio
async def test_batch_optimizer_preserves_order():
    """Results should be in same order as input."""
    optimizer = AsyncBatchOptimizer(max_concurrent=3)

    async def double(x):
        await asyncio.sleep(0.01)
        return x * 2

    results = await optimizer.map_async(double, [1, 2, 3, 4, 5])
    assert results == [2, 4, 6, 8, 10]


def test_timing_metric_properties():
    """TimingMetric should calculate durations correctly."""
    metric = TimingMetric(name="test", start_time=0.0, end_time=1.5)
    assert metric.duration_s == 1.5
    assert metric.duration_ms == 1500.0


def test_operation_timer_context_manager():
    """OperationTimer should measure sync operations."""
    import time

    with OperationTimer("test_op") as timer:
        time.sleep(0.05)

    assert timer.duration_s >= 0.05
    assert timer.duration_ms >= 50


def test_performance_tracker_records_cycles():
    """PerformanceTracker should record and maintain history."""
    tracker = PerformanceTracker(max_history=3)

    for i in range(5):
        tracker.record_cycle({
            'cycle_id': f'cycle-{i}',
            'operations': {'op1': {'avg_s': 0.1 * i}}
        })

    # Should only keep last 3
    assert len(tracker.cycle_summaries) == 3
    assert tracker.cycle_summaries[0]['cycle_id'] == 'cycle-2'


def test_performance_tracker_operation_trend():
    """PerformanceTracker should calculate operation trends."""
    tracker = PerformanceTracker()

    # Simulate improving performance
    tracker.record_cycle({'operations': {'op1': {'avg_s': 1.0}}})
    tracker.record_cycle({'operations': {'op1': {'avg_s': 0.8}}})
    tracker.record_cycle({'operations': {'op1': {'avg_s': 0.5}}})

    trend = tracker.get_operation_trend('op1')
    assert trend['data_points'] == 3
    assert trend['trend'] == 'improving'


def test_performance_tracker_bottleneck_frequency():
    """PerformanceTracker should track bottleneck frequency."""
    tracker = PerformanceTracker()

    tracker.record_cycle({'bottlenecks': ['op1', 'op2']})
    tracker.record_cycle({'bottlenecks': ['op1']})
    tracker.record_cycle({'bottlenecks': ['op1', 'op3']})

    freq = tracker.get_bottleneck_frequency()
    assert freq['op1'] == 3
    assert freq['op2'] == 1
    assert freq['op3'] == 1


@pytest.mark.asyncio
async def test_timed_async_decorator():
    """timed_async decorator should work without errors."""
    @timed_async("test_operation")
    async def slow_operation():
        await asyncio.sleep(0.01)
        return "done"

    result = await slow_operation()
    assert result == "done"


@pytest.mark.asyncio
async def test_profiler_concurrent_access():
    """Profiler should handle concurrent measurements safely."""
    profiler = CycleProfiler("concurrent-test")

    async def measure_op(name, delay):
        async with profiler.measure(name):
            await asyncio.sleep(delay)

    # Run multiple measurements concurrently
    await asyncio.gather(
        measure_op("op1", 0.01),
        measure_op("op2", 0.02),
        measure_op("op1", 0.01),  # Same name as first
    )

    summary = profiler.get_summary()
    assert summary['operations']['op1']['count'] == 2
    assert summary['operations']['op2']['count'] == 1
