"""
Metrics Streaming System

Dev 2 - Bulletproof Execution: Live metrics streaming from training processes.

Features:
- File-based metrics watching (JSON/JSONL files)
- Callback integration for real-time updates
- Aggregation and windowing for dashboard
- Integration with ExperimentRegistry

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread, Event
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class MetricsCallback(Protocol):
    """Protocol for metrics callback functions."""

    def __call__(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int,
        timestamp: str,
    ) -> None:
        """
        Called when new metrics are received.

        Args:
            experiment_id: Source experiment
            metrics: Dictionary of metric name -> value
            step: Training step/epoch
            timestamp: When metrics were recorded
        """
        ...


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    experiment_id: str
    step: int
    epoch: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsSnapshot":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MetricsWindow:
    """Sliding window for metrics aggregation."""
    experiment_id: str
    window_size: int = 100
    _snapshots: List[MetricsSnapshot] = field(default_factory=list)

    def add(self, snapshot: MetricsSnapshot) -> None:
        """Add snapshot to window."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def get_latest(self) -> Optional[MetricsSnapshot]:
        """Get most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_average(self, metric_name: str) -> Optional[float]:
        """Calculate average of a metric over the window."""
        values = [
            s.metrics.get(metric_name)
            for s in self._snapshots
            if metric_name in s.metrics
        ]
        return sum(values) / len(values) if values else None

    def get_min_max(self, metric_name: str) -> Optional[tuple]:
        """Get min and max of a metric over the window."""
        values = [
            s.metrics.get(metric_name)
            for s in self._snapshots
            if metric_name in s.metrics
        ]
        return (min(values), max(values)) if values else None

    def get_trend(self, metric_name: str) -> Optional[str]:
        """Determine if metric is improving/declining/stable."""
        if len(self._snapshots) < 2:
            return None

        values = [
            s.metrics.get(metric_name)
            for s in self._snapshots
            if metric_name in s.metrics
        ]

        if len(values) < 2:
            return None

        # Compare first and last thirds
        third = len(values) // 3
        if third < 1:
            third = 1

        first_avg = sum(values[:third]) / third
        last_avg = sum(values[-third:]) / third

        diff = last_avg - first_avg
        threshold = 0.01 * abs(first_avg) if first_avg != 0 else 0.001

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"


class MetricsFileWatcher:
    """
    Watches a metrics file for updates.

    Supports:
    - JSON files (single object, rewritten each update)
    - JSONL files (append-only, one JSON per line)
    """

    def __init__(
        self,
        filepath: Path,
        experiment_id: str,
        callback: MetricsCallback,
        poll_interval: float = 1.0,
    ):
        """
        Initialize file watcher.

        Args:
            filepath: Path to metrics file
            experiment_id: Associated experiment
            callback: Callback for new metrics
            poll_interval: Seconds between polls
        """
        self.filepath = Path(filepath)
        self.experiment_id = experiment_id
        self.callback = callback
        self.poll_interval = poll_interval

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._last_position = 0
        self._last_mtime = 0.0
        self._is_jsonl = filepath.suffix.lower() == ".jsonl"

    def start(self) -> None:
        """Start watching the file."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started watching metrics file: {self.filepath}")

    def stop(self) -> None:
        """Stop watching the file."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info(f"Stopped watching metrics file: {self.filepath}")

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while not self._stop_event.is_set():
            try:
                if self.filepath.exists():
                    mtime = self.filepath.stat().st_mtime

                    if mtime > self._last_mtime:
                        self._last_mtime = mtime
                        self._process_updates()

            except Exception as e:
                logger.error(f"Error watching {self.filepath}: {e}")

            self._stop_event.wait(self.poll_interval)

    def _process_updates(self) -> None:
        """Process new content in the file."""
        try:
            if self._is_jsonl:
                self._process_jsonl()
            else:
                self._process_json()

        except Exception as e:
            logger.error(f"Error processing metrics file {self.filepath}: {e}")

    def _process_json(self) -> None:
        """Process JSON file (complete rewrite each update)."""
        with open(self.filepath) as f:
            data = json.load(f)

        # Extract metrics
        metrics = {}
        step = data.get("step", data.get("iteration", 0))
        timestamp = data.get("timestamp", datetime.now().isoformat())

        # Flatten nested metrics
        for key, value in data.items():
            if isinstance(value, (int, float)) and key not in {"step", "iteration", "epoch"}:
                metrics[key] = float(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics[f"{key}/{subkey}"] = float(subvalue)

        if metrics:
            self.callback(
                experiment_id=self.experiment_id,
                metrics=metrics,
                step=step,
                timestamp=timestamp,
            )

    def _process_jsonl(self) -> None:
        """Process JSONL file (append-only, read new lines)."""
        with open(self.filepath) as f:
            f.seek(self._last_position)
            new_lines = f.readlines()
            self._last_position = f.tell()

        for line in new_lines:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                metrics = {}
                step = data.get("step", data.get("iteration", 0))
                timestamp = data.get("timestamp", datetime.now().isoformat())

                for key, value in data.items():
                    if isinstance(value, (int, float)) and key not in {"step", "iteration", "epoch"}:
                        metrics[key] = float(value)

                if metrics:
                    self.callback(
                        experiment_id=self.experiment_id,
                        metrics=metrics,
                        step=step,
                        timestamp=timestamp,
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON line in {self.filepath}: {e}")


class MetricsStreamer:
    """
    Central metrics streaming manager.

    Coordinates multiple file watchers and aggregates metrics.
    """

    def __init__(self, poll_interval: float = 1.0):
        """
        Initialize metrics streamer.

        Args:
            poll_interval: Default polling interval for watchers
        """
        self.poll_interval = poll_interval

        self._watchers: Dict[str, MetricsFileWatcher] = {}
        self._windows: Dict[str, MetricsWindow] = {}
        self._callbacks: List[MetricsCallback] = []
        self._lock = Lock()

        # History for persistence
        self._history: Dict[str, List[MetricsSnapshot]] = {}
        self._max_history = 10000  # Max snapshots per experiment

    def register_experiment(
        self,
        experiment_id: str,
        metrics_file: str,
        window_size: int = 100,
    ) -> None:
        """
        Register an experiment for metrics streaming.

        Args:
            experiment_id: Experiment identifier
            metrics_file: Path to metrics file (JSON or JSONL)
            window_size: Size of metrics window
        """
        with self._lock:
            if experiment_id in self._watchers:
                logger.warning(f"Experiment {experiment_id} already registered")
                return

            filepath = Path(metrics_file)

            watcher = MetricsFileWatcher(
                filepath=filepath,
                experiment_id=experiment_id,
                callback=self._on_metrics,
                poll_interval=self.poll_interval,
            )

            self._watchers[experiment_id] = watcher
            self._windows[experiment_id] = MetricsWindow(
                experiment_id=experiment_id,
                window_size=window_size,
            )
            self._history[experiment_id] = []

            watcher.start()
            logger.info(f"Registered metrics stream for experiment: {experiment_id}")

    def unregister_experiment(self, experiment_id: str) -> None:
        """Stop streaming metrics for an experiment."""
        with self._lock:
            watcher = self._watchers.pop(experiment_id, None)
            if watcher:
                watcher.stop()
                logger.info(f"Unregistered metrics stream for experiment: {experiment_id}")

    def add_callback(self, callback: MetricsCallback) -> None:
        """Add a callback for all metrics updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: MetricsCallback) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_latest_metrics(self, experiment_id: str) -> Optional[Dict[str, float]]:
        """Get latest metrics for an experiment."""
        with self._lock:
            window = self._windows.get(experiment_id)
            if window:
                snapshot = window.get_latest()
                return snapshot.metrics if snapshot else None
            return None

    def get_metrics_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of metrics for an experiment."""
        with self._lock:
            window = self._windows.get(experiment_id)
            if not window:
                return None

            latest = window.get_latest()
            if not latest:
                return None

            summary = {
                "experiment_id": experiment_id,
                "latest_step": latest.step,
                "latest_timestamp": latest.timestamp,
                "metrics": {},
            }

            # For each metric, get latest value, average, min/max, trend
            for metric_name, value in latest.metrics.items():
                min_max = window.get_min_max(metric_name)
                summary["metrics"][metric_name] = {
                    "current": value,
                    "average": window.get_average(metric_name),
                    "min": min_max[0] if min_max else None,
                    "max": min_max[1] if min_max else None,
                    "trend": window.get_trend(metric_name),
                }

            return summary

    def get_metrics_history(
        self,
        experiment_id: str,
        last_n: Optional[int] = None,
    ) -> List[MetricsSnapshot]:
        """Get metrics history for an experiment."""
        with self._lock:
            history = self._history.get(experiment_id, [])
            if last_n:
                return history[-last_n:]
            return history.copy()

    def get_active_streams(self) -> List[str]:
        """Get list of active experiment streams."""
        with self._lock:
            return list(self._watchers.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get streamer statistics."""
        with self._lock:
            return {
                "active_streams": len(self._watchers),
                "total_callbacks": len(self._callbacks),
                "experiments": {
                    exp_id: {
                        "history_size": len(self._history.get(exp_id, [])),
                        "window_size": self._windows[exp_id].window_size if exp_id in self._windows else 0,
                    }
                    for exp_id in self._watchers
                },
            }

    def stop_all(self) -> None:
        """Stop all watchers."""
        with self._lock:
            for watcher in self._watchers.values():
                watcher.stop()
            self._watchers.clear()

    def _on_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int,
        timestamp: str,
    ) -> None:
        """Internal callback for metrics updates."""
        snapshot = MetricsSnapshot(
            experiment_id=experiment_id,
            step=step,
            metrics=metrics,
            timestamp=timestamp,
        )

        with self._lock:
            # Add to window
            window = self._windows.get(experiment_id)
            if window:
                window.add(snapshot)

            # Add to history
            history = self._history.get(experiment_id, [])
            history.append(snapshot)
            if len(history) > self._max_history:
                history.pop(0)
            self._history[experiment_id] = history

        # Fire callbacks
        for callback in self._callbacks:
            try:
                callback(
                    experiment_id=experiment_id,
                    metrics=metrics,
                    step=step,
                    timestamp=timestamp,
                )
            except Exception as e:
                logger.error(f"Metrics callback error: {e}")


# Singleton pattern
_streamer: Optional[MetricsStreamer] = None
_streamer_lock = Lock()


def get_metrics_streamer(poll_interval: float = 1.0) -> MetricsStreamer:
    """
    Get or create the global MetricsStreamer instance.

    Args:
        poll_interval: Default polling interval

    Returns:
        The singleton MetricsStreamer instance
    """
    global _streamer

    with _streamer_lock:
        if _streamer is None:
            _streamer = MetricsStreamer(poll_interval=poll_interval)
        return _streamer


def reset_metrics_streamer() -> None:
    """Reset the global MetricsStreamer instance (for testing)."""
    global _streamer
    with _streamer_lock:
        if _streamer is not None:
            _streamer.stop_all()
        _streamer = None
