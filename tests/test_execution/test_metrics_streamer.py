"""
Tests for Metrics Streaming System.

Dev 2 - Bulletproof Execution: Tests for MetricsStreamer, MetricsSnapshot,
MetricsWindow, and file watching.

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import pytest
import json
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime

from execution.metrics_streamer import (
    MetricsSnapshot,
    MetricsWindow,
    MetricsFileWatcher,
    MetricsStreamer,
    get_metrics_streamer,
    reset_metrics_streamer,
)


class TestMetricsSnapshot:
    """Test MetricsSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a metrics snapshot."""
        snapshot = MetricsSnapshot(
            experiment_id="exp_001",
            step=100,
            metrics={"loss": 0.5, "auc": 0.85},
        )

        assert snapshot.experiment_id == "exp_001"
        assert snapshot.step == 100
        assert snapshot.metrics["loss"] == 0.5
        assert snapshot.metrics["auc"] == 0.85
        assert snapshot.timestamp is not None

    def test_to_dict(self):
        """Test serialization to dict."""
        snapshot = MetricsSnapshot(
            experiment_id="exp_001",
            step=100,
            epoch=5,
            metrics={"loss": 0.5},
        )

        data = snapshot.to_dict()

        assert data["experiment_id"] == "exp_001"
        assert data["step"] == 100
        assert data["epoch"] == 5
        assert data["metrics"]["loss"] == 0.5

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "experiment_id": "exp_001",
            "step": 100,
            "epoch": 5,
            "metrics": {"loss": 0.5},
            "timestamp": "2025-01-01T00:00:00",
        }

        snapshot = MetricsSnapshot.from_dict(data)

        assert snapshot.experiment_id == "exp_001"
        assert snapshot.step == 100
        assert snapshot.metrics["loss"] == 0.5


class TestMetricsWindow:
    """Test MetricsWindow for metrics aggregation."""

    def test_add_snapshot(self):
        """Test adding snapshots to window."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        for i in range(10):
            snapshot = MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"loss": 1.0 - i * 0.1},
            )
            window.add(snapshot)

        latest = window.get_latest()
        assert latest.step == 9
        assert latest.metrics["loss"] == pytest.approx(0.1)

    def test_window_size_limit(self):
        """Test window respects size limit."""
        window = MetricsWindow(experiment_id="exp_001", window_size=5)

        for i in range(10):
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"value": i},
            ))

        # Should only have last 5
        assert len(window._snapshots) == 5
        assert window._snapshots[0].step == 5  # Oldest
        assert window._snapshots[-1].step == 9  # Newest

    def test_get_average(self):
        """Test calculating average."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        for i in range(5):
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"value": i * 2},  # 0, 2, 4, 6, 8
            ))

        avg = window.get_average("value")
        assert avg == pytest.approx(4.0)  # (0+2+4+6+8)/5 = 4

    def test_get_min_max(self):
        """Test getting min/max."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        for i in [5, 2, 8, 1, 9, 3]:
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"value": i},
            ))

        min_max = window.get_min_max("value")
        assert min_max == (1, 9)

    def test_get_trend_improving(self):
        """Test detecting improving trend."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        # Loss decreasing = improving
        for i in range(10):
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"loss": 1.0 - i * 0.05},  # 1.0 -> 0.55
            ))

        trend = window.get_trend("loss")
        assert trend == "declining"  # Loss going down

        # AUC increasing = improving
        for i in range(10):
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"auc": 0.5 + i * 0.05},  # 0.5 -> 0.95
            ))

        trend = window.get_trend("auc")
        assert trend == "improving"

    def test_get_trend_stable(self):
        """Test detecting stable trend."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        # Nearly constant values
        for i in range(10):
            window.add(MetricsSnapshot(
                experiment_id="exp_001",
                step=i,
                metrics={"value": 0.5 + (i % 2) * 0.001},  # Tiny variance
            ))

        trend = window.get_trend("value")
        assert trend == "stable"

    def test_get_nonexistent_metric(self):
        """Test getting nonexistent metric returns None."""
        window = MetricsWindow(experiment_id="exp_001", window_size=100)

        window.add(MetricsSnapshot(
            experiment_id="exp_001",
            step=0,
            metrics={"loss": 0.5},
        ))

        assert window.get_average("nonexistent") is None
        assert window.get_min_max("nonexistent") is None
        assert window.get_trend("nonexistent") is None


class TestMetricsFileWatcher:
    """Test MetricsFileWatcher."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_watch_json_file(self, temp_dir):
        """Test watching JSON file for updates."""
        filepath = Path(temp_dir) / "metrics.json"
        received = []

        def callback(experiment_id, metrics, step, timestamp):
            received.append({
                "experiment_id": experiment_id,
                "metrics": metrics.copy(),
                "step": step,
            })

        watcher = MetricsFileWatcher(
            filepath=filepath,
            experiment_id="exp_001",
            callback=callback,
            poll_interval=0.1,
        )

        try:
            watcher.start()

            # Write metrics
            with open(filepath, "w") as f:
                json.dump({"step": 10, "loss": 0.5, "auc": 0.85}, f)

            # Wait for watcher to pick it up
            time.sleep(0.3)

            assert len(received) >= 1
            assert received[-1]["step"] == 10
            assert received[-1]["metrics"]["loss"] == 0.5
            assert received[-1]["metrics"]["auc"] == 0.85

        finally:
            watcher.stop()

    def test_watch_jsonl_file(self, temp_dir):
        """Test watching JSONL file for new lines."""
        filepath = Path(temp_dir) / "metrics.jsonl"
        received = []

        def callback(experiment_id, metrics, step, timestamp):
            received.append({
                "step": step,
                "metrics": metrics.copy(),
            })

        watcher = MetricsFileWatcher(
            filepath=filepath,
            experiment_id="exp_001",
            callback=callback,
            poll_interval=0.1,
        )

        try:
            watcher.start()

            # Write initial lines
            with open(filepath, "w") as f:
                f.write(json.dumps({"step": 1, "loss": 0.9}) + "\n")
                f.write(json.dumps({"step": 2, "loss": 0.8}) + "\n")

            time.sleep(0.3)

            # Append more
            with open(filepath, "a") as f:
                f.write(json.dumps({"step": 3, "loss": 0.7}) + "\n")

            time.sleep(0.3)

            # Should have received all 3
            assert len(received) >= 3

        finally:
            watcher.stop()


class TestMetricsStreamer:
    """Test MetricsStreamer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def streamer(self):
        """Create streamer instance."""
        reset_metrics_streamer()
        return MetricsStreamer(poll_interval=0.1)

    def test_register_experiment(self, streamer, temp_dir):
        """Test registering experiment for streaming."""
        filepath = Path(temp_dir) / "metrics.json"
        filepath.touch()

        streamer.register_experiment(
            experiment_id="exp_001",
            metrics_file=str(filepath),
        )

        assert "exp_001" in streamer.get_active_streams()

    def test_unregister_experiment(self, streamer, temp_dir):
        """Test unregistering experiment."""
        filepath = Path(temp_dir) / "metrics.json"
        filepath.touch()

        streamer.register_experiment("exp_001", str(filepath))
        streamer.unregister_experiment("exp_001")

        assert "exp_001" not in streamer.get_active_streams()

    def test_add_callback(self, streamer, temp_dir):
        """Test adding callback."""
        filepath = Path(temp_dir) / "metrics.json"
        received = []

        def callback(experiment_id, metrics, step, timestamp):
            received.append({"experiment_id": experiment_id, "step": step})

        streamer.add_callback(callback)
        streamer.register_experiment("exp_001", str(filepath))

        # Write metrics
        with open(filepath, "w") as f:
            json.dump({"step": 100, "loss": 0.5}, f)

        time.sleep(0.3)

        assert len(received) >= 1
        assert received[-1]["experiment_id"] == "exp_001"
        assert received[-1]["step"] == 100

    def test_get_latest_metrics(self, streamer, temp_dir):
        """Test getting latest metrics."""
        filepath = Path(temp_dir) / "metrics.json"

        streamer.register_experiment("exp_001", str(filepath))

        # Write metrics
        with open(filepath, "w") as f:
            json.dump({"step": 50, "loss": 0.3, "auc": 0.9}, f)

        time.sleep(0.3)

        latest = streamer.get_latest_metrics("exp_001")

        assert latest is not None
        assert latest["loss"] == 0.3
        assert latest["auc"] == 0.9

    def test_get_metrics_summary(self, streamer, temp_dir):
        """Test getting metrics summary."""
        filepath = Path(temp_dir) / "metrics.jsonl"

        streamer.register_experiment("exp_001", str(filepath), window_size=10)

        # Write multiple metric points
        with open(filepath, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "step": i * 10,
                    "loss": 1.0 - i * 0.1,
                    "auc": 0.5 + i * 0.1,
                }) + "\n")

        time.sleep(0.3)

        summary = streamer.get_metrics_summary("exp_001")

        assert summary is not None
        assert "loss" in summary["metrics"]
        assert "auc" in summary["metrics"]
        assert summary["metrics"]["loss"]["current"] is not None
        assert summary["metrics"]["auc"]["trend"] is not None

    def test_get_metrics_history(self, streamer, temp_dir):
        """Test getting metrics history."""
        filepath = Path(temp_dir) / "metrics.jsonl"

        streamer.register_experiment("exp_001", str(filepath))

        # Write metrics
        with open(filepath, "w") as f:
            for i in range(10):
                f.write(json.dumps({"step": i, "loss": 0.5}) + "\n")

        time.sleep(0.3)

        # Get all history
        history = streamer.get_metrics_history("exp_001")
        assert len(history) == 10

        # Get last 5
        history_5 = streamer.get_metrics_history("exp_001", last_n=5)
        assert len(history_5) == 5

    def test_get_stats(self, streamer, temp_dir):
        """Test getting streamer stats."""
        filepath1 = Path(temp_dir) / "metrics1.json"
        filepath2 = Path(temp_dir) / "metrics2.json"
        filepath1.touch()
        filepath2.touch()

        streamer.register_experiment("exp_001", str(filepath1))
        streamer.register_experiment("exp_002", str(filepath2))

        stats = streamer.get_stats()

        assert stats["active_streams"] == 2
        assert "exp_001" in stats["experiments"]
        assert "exp_002" in stats["experiments"]

    def test_stop_all(self, streamer, temp_dir):
        """Test stopping all watchers."""
        filepath = Path(temp_dir) / "metrics.json"
        filepath.touch()

        streamer.register_experiment("exp_001", str(filepath))
        streamer.register_experiment("exp_002", str(filepath))

        streamer.stop_all()

        assert len(streamer.get_active_streams()) == 0


class TestSingleton:
    """Test singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_metrics_streamer()
        yield
        reset_metrics_streamer()

    def test_get_same_instance(self):
        """Test get_metrics_streamer returns same instance."""
        instance1 = get_metrics_streamer()
        instance2 = get_metrics_streamer()

        assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test reset creates new instance."""
        instance1 = get_metrics_streamer()
        reset_metrics_streamer()
        instance2 = get_metrics_streamer()

        assert instance1 is not instance2


class TestIntegration:
    """Integration tests with experiment lifecycle."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_experiment_flow(self, temp_dir):
        """Test full experiment metrics flow."""
        from execution.experiment_lifecycle import (
            ExperimentRecord,
            ExperimentRegistry,
            ExperimentState,
        )

        # Setup
        registry = ExperimentRegistry(storage_path=temp_dir)
        streamer = MetricsStreamer(poll_interval=0.1)

        metrics_updates = []

        def on_metrics(experiment_id, metrics, step, timestamp):
            metrics_updates.append({
                "experiment_id": experiment_id,
                "step": step,
                "metrics": metrics.copy(),
            })
            # Update registry with latest metrics
            registry.update_metrics(experiment_id, metrics)

        streamer.add_callback(on_metrics)

        try:
            # Create experiment
            record = ExperimentRecord(
                experiment_id="exp_001",
                cycle_id=1,
                proposal_id="prop_001",
            )
            registry.register(record)

            # Start experiment
            registry.update_state("exp_001", ExperimentState.QUEUED)
            registry.update_state("exp_001", ExperimentState.RUNNING)

            # Setup metrics streaming
            metrics_file = Path(temp_dir) / "exp_001_metrics.jsonl"
            streamer.register_experiment("exp_001", str(metrics_file))

            # Simulate training writing metrics
            with open(metrics_file, "w") as f:
                for epoch in range(3):
                    f.write(json.dumps({
                        "step": epoch * 100,
                        "epoch": epoch,
                        "loss": 1.0 - epoch * 0.2,
                        "auc": 0.5 + epoch * 0.15,
                    }) + "\n")
                    f.flush()
                    time.sleep(0.15)

            time.sleep(0.3)

            # Complete experiment
            registry.update_state("exp_001", ExperimentState.COMPLETED)

            # Verify
            final_record = registry.get("exp_001")
            assert final_record.state == ExperimentState.COMPLETED
            assert len(metrics_updates) >= 3
            assert "auc" in final_record.metrics

        finally:
            streamer.stop_all()
