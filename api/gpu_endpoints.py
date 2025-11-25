"""
REST API endpoints for GPU monitoring.

Phase F - Infrastructure & Stability Track
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from monitoring.gpu_metrics import get_gpu_monitor

router = APIRouter(prefix="/api/gpu", tags=["GPU Monitoring"])


@router.get("/status")
async def get_gpu_status():
    """
    Get current GPU status summary.

    Returns:
        Summary including GPU count, utilization, memory, and temperature.
    """
    monitor = get_gpu_monitor()
    return monitor.get_summary()


@router.get("/metrics")
async def get_gpu_metrics():
    """
    Get detailed current metrics for all GPUs.

    Returns:
        List of detailed metrics per GPU.
    """
    monitor = get_gpu_monitor()
    metrics = monitor.get_metrics()
    return {"gpus": [m.to_dict() for m in metrics]}


@router.get("/history")
async def get_gpu_history(last_n: int = 100):
    """
    Get historical GPU metrics.

    Args:
        last_n: Number of historical samples to return (default 100)

    Returns:
        List of historical metric samples.
    """
    if last_n < 1 or last_n > 1000:
        raise HTTPException(status_code=400, detail="last_n must be between 1 and 1000")

    monitor = get_gpu_monitor()
    return {"history": monitor.get_history(last_n)}


@router.post("/monitor/start")
async def start_monitoring(poll_interval: float = 1.0):
    """
    Start background GPU monitoring.

    Args:
        poll_interval: Polling interval in seconds (default 1.0)

    Returns:
        Status confirmation.
    """
    if poll_interval < 0.1 or poll_interval > 60:
        raise HTTPException(status_code=400, detail="poll_interval must be between 0.1 and 60 seconds")

    monitor = get_gpu_monitor()
    monitor.poll_interval = poll_interval
    await monitor.start()
    return {"status": "started", "poll_interval": poll_interval}


@router.post("/monitor/stop")
async def stop_monitoring():
    """
    Stop background GPU monitoring.

    Returns:
        Status confirmation.
    """
    monitor = get_gpu_monitor()
    await monitor.stop()
    return {"status": "stopped"}


@router.get("/monitor/status")
async def get_monitor_status():
    """
    Get GPU monitor status.

    Returns:
        Monitor running status and configuration.
    """
    monitor = get_gpu_monitor()
    return {
        "running": monitor.is_running,
        "poll_interval": monitor.poll_interval,
        "available": monitor.available,
        "history_size": len(monitor._history)
    }


@router.post("/history/clear")
async def clear_history():
    """
    Clear GPU metric history.

    Returns:
        Status confirmation.
    """
    monitor = get_gpu_monitor()
    monitor.clear_history()
    return {"status": "cleared"}
