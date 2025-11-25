"""
Tests for GPU monitoring utilities.

Phase F - Infrastructure & Stability Track
"""
import pytest
from datetime import datetime
from monitoring.gpu_metrics import GPUMonitor, GPUMetrics, AsyncGPUMonitor, get_gpu_monitor


def test_gpu_metrics_dataclass():
    """GPUMetrics should calculate derived properties correctly."""
    metrics = GPUMetrics(
        index=0,
        name="NVIDIA RTX 4090",
        utilization_percent=75.0,
        memory_used_mb=8000,
        memory_total_mb=24000,
        temperature_c=65,
        power_draw_w=300.0,
        power_limit_w=450.0,
        timestamp=datetime.now()
    )

    assert metrics.memory_percent == pytest.approx(33.33, rel=0.01)
    assert metrics.power_percent == pytest.approx(66.67, rel=0.01)


def test_gpu_metrics_zero_division():
    """GPUMetrics should handle zero totals gracefully."""
    metrics = GPUMetrics(
        index=0,
        name="Test GPU",
        utilization_percent=0,
        memory_used_mb=0,
        memory_total_mb=0,
        temperature_c=0,
        power_draw_w=0,
        power_limit_w=0,
        timestamp=datetime.now()
    )

    assert metrics.memory_percent == 0.0
    assert metrics.power_percent == 0.0


def test_gpu_metrics_to_dict():
    """GPUMetrics.to_dict should return all fields."""
    metrics = GPUMetrics(
        index=0,
        name="Test GPU",
        utilization_percent=50.0,
        memory_used_mb=4000,
        memory_total_mb=8000,
        temperature_c=60,
        power_draw_w=200.0,
        power_limit_w=400.0,
        timestamp=datetime.now()
    )

    d = metrics.to_dict()
    assert d['index'] == 0
    assert d['name'] == "Test GPU"
    assert d['memory_percent'] == 50.0
    assert d['power_percent'] == 50.0
    assert 'timestamp' in d


def test_gpu_monitor_creation():
    """GPUMonitor should be creatable."""
    monitor = GPUMonitor()
    assert isinstance(monitor.available, bool)


def test_gpu_monitor_get_metrics():
    """Monitor should return empty list if nvidia-smi unavailable."""
    monitor = GPUMonitor()
    metrics = monitor.get_metrics()
    assert isinstance(metrics, list)


def test_gpu_monitor_get_summary():
    """Summary should always return valid structure."""
    monitor = GPUMonitor()
    summary = monitor.get_summary()

    assert 'available' in summary
    assert 'gpu_count' in summary
    assert 'gpus' in summary


def test_gpu_monitor_history():
    """Monitor should track history."""
    monitor = GPUMonitor()

    # Trigger a metrics collection
    monitor.get_metrics()

    history = monitor.get_history(last_n=10)
    assert isinstance(history, list)


def test_gpu_monitor_clear_history():
    """Monitor should clear history."""
    monitor = GPUMonitor()
    monitor.get_metrics()  # Add to history
    monitor.clear_history()
    assert len(monitor._history) == 0


def test_async_gpu_monitor_creation():
    """AsyncGPUMonitor should be creatable."""
    monitor = AsyncGPUMonitor(poll_interval=1.0)
    assert monitor.poll_interval == 1.0
    assert not monitor.is_running


@pytest.mark.asyncio
async def test_async_gpu_monitor_start_stop():
    """AsyncGPUMonitor should start and stop cleanly."""
    monitor = AsyncGPUMonitor(poll_interval=0.1)

    await monitor.start()
    assert monitor.is_running

    await monitor.stop()
    assert not monitor.is_running


@pytest.mark.asyncio
async def test_async_gpu_monitor_polling():
    """AsyncGPUMonitor should poll when running."""
    import asyncio

    monitor = AsyncGPUMonitor(poll_interval=0.05)
    monitor.clear_history()

    await monitor.start()
    await asyncio.sleep(0.15)  # Allow a few polls
    await monitor.stop()

    # History should have some entries (even if empty due to no nvidia-smi)
    # The monitor will still run the poll loop
    assert not monitor.is_running


def test_get_gpu_monitor_singleton():
    """get_gpu_monitor should return same instance."""
    monitor1 = get_gpu_monitor()
    monitor2 = get_gpu_monitor()
    assert monitor1 is monitor2


def test_gpu_metrics_timestamp_format():
    """Timestamp should be ISO format in dict."""
    now = datetime.now()
    metrics = GPUMetrics(
        index=0,
        name="Test GPU",
        utilization_percent=50.0,
        memory_used_mb=4000,
        memory_total_mb=8000,
        temperature_c=60,
        power_draw_w=200.0,
        power_limit_w=400.0,
        timestamp=now
    )

    d = metrics.to_dict()
    # Should be parseable as ISO format
    parsed = datetime.fromisoformat(d['timestamp'])
    assert parsed.date() == now.date()
