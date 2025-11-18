"""
System Monitoring Utilities for ARC Mission Control UI

Provides real-time telemetry for:
- GPU health (memory, utilization, temperature)
- CPU usage
- RAM usage
- Disk usage
- System uptime

This is the data source for Dev 2's Silicon Valley-grade UI.
"""

import logging
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitors system health metrics for UI display.

    Provides clean, structured data for beautiful dashboards.
    """

    def __init__(self):
        """Initialize system monitor."""
        self.start_time = time.time()
        logger.info("SystemMonitor initialized")

    def get_gpu_health(self) -> List[Dict[str, Any]]:
        """
        Get GPU health metrics for all available GPUs.

        Returns:
            List of GPU health dicts with:
            - id: GPU index
            - mem: Memory usage in GB
            - util: Utilization percentage
            - temp: Temperature in Celsius
            - name: GPU name
        """
        gpus = []

        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get GPU name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / (1024 ** 3)
                mem_total_gb = mem_info.total / (1024 ** 3)
                mem_percent = (mem_info.used / mem_info.total) * 100

                # Get utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_percent = util_info.gpu

                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None

                gpus.append({
                    "id": i,
                    "name": name,
                    "mem": round(mem_used_gb, 1),
                    "mem_total": round(mem_total_gb, 1),
                    "mem_percent": round(mem_percent, 1),
                    "util": util_percent,
                    "temp": temp,
                    "status": self._get_gpu_status(util_percent, mem_percent, temp)
                })

            pynvml.nvmlShutdown()

        except ImportError:
            logger.warning("pynvml not available - GPU monitoring disabled")
        except Exception as e:
            logger.error(f"Failed to get GPU health: {e}")

        return gpus

    def _get_gpu_status(self, util: int, mem_percent: float, temp: Optional[int]) -> str:
        """
        Determine GPU health status for UI color coding.

        Args:
            util: GPU utilization percentage
            mem_percent: Memory usage percentage
            temp: Temperature in Celsius

        Returns:
            Status: "healthy", "warning", or "critical"
        """
        # Critical conditions
        if temp and temp > 85:
            return "critical"
        if mem_percent > 95:
            return "critical"

        # Warning conditions
        if temp and temp > 75:
            return "warning"
        if mem_percent > 85:
            return "warning"
        if util > 95:
            return "warning"

        return "healthy"

    def get_cpu_usage(self) -> float:
        """
        Get CPU usage percentage.

        Returns:
            CPU usage as float (0-100)
        """
        try:
            return round(psutil.cpu_percent(interval=0.5), 1)
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return 0.0

    def get_ram_usage(self) -> Dict[str, Any]:
        """
        Get RAM usage metrics.

        Returns:
            Dict with:
            - percent: Usage percentage
            - used_gb: Used RAM in GB
            - total_gb: Total RAM in GB
            - available_gb: Available RAM in GB
        """
        try:
            mem = psutil.virtual_memory()

            return {
                "percent": round(mem.percent, 1),
                "used_gb": round(mem.used / (1024 ** 3), 1),
                "total_gb": round(mem.total / (1024 ** 3), 1),
                "available_gb": round(mem.available / (1024 ** 3), 1),
                "status": "critical" if mem.percent > 95 else "warning" if mem.percent > 85 else "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get RAM usage: {e}")
            return {"percent": 0, "used_gb": 0, "total_gb": 0, "available_gb": 0, "status": "unknown"}

    def get_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """
        Get disk usage metrics.

        Args:
            path: Path to check disk usage for

        Returns:
            Dict with:
            - percent: Usage percentage
            - used_gb: Used disk in GB
            - total_gb: Total disk in GB
            - free_gb: Free disk in GB
        """
        try:
            disk = psutil.disk_usage(path)

            return {
                "percent": round(disk.percent, 1),
                "used_gb": round(disk.used / (1024 ** 3), 1),
                "total_gb": round(disk.total / (1024 ** 3), 1),
                "free_gb": round(disk.free / (1024 ** 3), 1),
                "status": "critical" if disk.percent > 95 else "warning" if disk.percent > 85 else "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {"percent": 0, "used_gb": 0, "total_gb": 0, "free_gb": 0, "status": "unknown"}

    def get_uptime(self) -> str:
        """
        Get system uptime in HH:MM:SS format.

        Returns:
            Uptime string like "03:22:18"
        """
        try:
            uptime_seconds = time.time() - self.start_time
            uptime_delta = timedelta(seconds=int(uptime_seconds))

            # Format as HH:MM:SS
            hours = uptime_delta.seconds // 3600
            minutes = (uptime_delta.seconds % 3600) // 60
            seconds = uptime_delta.seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception as e:
            logger.error(f"Failed to get uptime: {e}")
            return "00:00:00"

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get complete system health snapshot.

        This is the primary endpoint for UI `/ui/system/health`.

        Returns:
            Dict with all health metrics in UI-ready format
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "gpu": self.get_gpu_health(),
            "cpu_usage": self.get_cpu_usage(),
            "ram": self.get_ram_usage(),
            "disk": self.get_disk_usage(),
            "uptime": self.get_uptime(),
            "status": self._get_overall_status()
        }

    def _get_overall_status(self) -> str:
        """
        Determine overall system health status.

        Returns:
            Status: "healthy", "warning", or "critical"
        """
        # Check GPU health
        gpus = self.get_gpu_health()
        if any(gpu["status"] == "critical" for gpu in gpus):
            return "critical"

        # Check RAM
        ram = self.get_ram_usage()
        if ram["status"] == "critical":
            return "critical"

        # Check disk
        disk = self.get_disk_usage()
        if disk["status"] == "critical":
            return "critical"

        # Check for warnings
        if any(gpu["status"] == "warning" for gpu in gpus):
            return "warning"
        if ram["status"] == "warning":
            return "warning"
        if disk["status"] == "warning":
            return "warning"

        return "healthy"


# Global singleton
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """
    Get global system monitor instance (singleton).

    Returns:
        SystemMonitor instance
    """
    global _system_monitor

    if _system_monitor is None:
        _system_monitor = SystemMonitor()

    return _system_monitor
