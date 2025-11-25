"""
GPU metrics collection using nvidia-smi.
Lightweight polling without external dependencies.

Phase F - Infrastructure & Stability Track
"""
import subprocess
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Metrics for a single GPU."""
    index: int
    name: str
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    timestamp: datetime

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @property
    def power_percent(self) -> float:
        if self.power_limit_w == 0:
            return 0.0
        return (self.power_draw_w / self.power_limit_w) * 100

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'name': self.name,
            'utilization_percent': self.utilization_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'memory_percent': round(self.memory_percent, 2),
            'temperature_c': self.temperature_c,
            'power_draw_w': self.power_draw_w,
            'power_limit_w': self.power_limit_w,
            'power_percent': round(self.power_percent, 2),
            'timestamp': self.timestamp.isoformat()
        }


class GPUMonitor:
    """
    Monitors GPU metrics using nvidia-smi.
    """

    NVIDIA_SMI_QUERY = (
        "index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit"
    )

    def __init__(self):
        self._available = self._check_nvidia_smi()
        self._history: List[List[GPUMetrics]] = []
        self._max_history = 1000  # Keep last 1000 samples

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not available")
            return False

    @property
    def available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._available

    def get_metrics(self) -> List[GPUMetrics]:
        """Get current metrics for all GPUs."""
        if not self._available:
            return []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={self.NVIDIA_SMI_QUERY}",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )

            metrics = []
            timestamp = datetime.now()

            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    try:
                        metrics.append(GPUMetrics(
                            index=int(parts[0]),
                            name=parts[1],
                            utilization_percent=float(parts[2]) if parts[2] not in ['[N/A]', 'N/A'] else 0,
                            memory_used_mb=int(parts[3]) if parts[3] not in ['[N/A]', 'N/A'] else 0,
                            memory_total_mb=int(parts[4]) if parts[4] not in ['[N/A]', 'N/A'] else 0,
                            temperature_c=int(parts[5]) if parts[5] not in ['[N/A]', 'N/A'] else 0,
                            power_draw_w=float(parts[6]) if parts[6] not in ['[N/A]', 'N/A'] else 0,
                            power_limit_w=float(parts[7]) if parts[7] not in ['[N/A]', 'N/A'] else 0,
                            timestamp=timestamp
                        ))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse GPU metrics line: {line}, error: {e}")

            # Store in history
            if metrics:
                self._history.append(metrics)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

            return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"nvidia-smi failed: {e}")
            return []

    def get_summary(self) -> Dict:
        """Get summary of current GPU state."""
        metrics = self.get_metrics()
        if not metrics:
            return {'available': self._available, 'gpu_count': 0, 'gpus': []}

        return {
            'available': True,
            'gpu_count': len(metrics),
            'gpus': [m.to_dict() for m in metrics],
            'total_memory_used_mb': sum(m.memory_used_mb for m in metrics),
            'total_memory_mb': sum(m.memory_total_mb for m in metrics),
            'avg_utilization': sum(m.utilization_percent for m in metrics) / len(metrics),
            'max_temperature': max(m.temperature_c for m in metrics)
        }

    def get_history(self, last_n: int = 100) -> List[Dict]:
        """Get historical metrics."""
        history = self._history[-last_n:]
        return [
            {
                'timestamp': samples[0].timestamp.isoformat() if samples else None,
                'gpus': [m.to_dict() for m in samples]
            }
            for samples in history
        ]

    def clear_history(self):
        """Clear metric history."""
        self._history.clear()


class AsyncGPUMonitor(GPUMonitor):
    """
    Async GPU monitor with background polling.
    """

    def __init__(self, poll_interval: float = 1.0):
        super().__init__()
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background polling."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"GPU monitoring started (interval: {self.poll_interval}s)")

    async def stop(self):
        """Stop background polling."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GPU monitoring stopped")

    async def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            self.get_metrics()
            await asyncio.sleep(self.poll_interval)

    @property
    def is_running(self) -> bool:
        """Check if background polling is running."""
        return self._running


# Global monitor instance
_monitor: Optional[AsyncGPUMonitor] = None


def get_gpu_monitor() -> AsyncGPUMonitor:
    """Get or create global GPU monitor."""
    global _monitor
    if _monitor is None:
        _monitor = AsyncGPUMonitor()
    return _monitor
