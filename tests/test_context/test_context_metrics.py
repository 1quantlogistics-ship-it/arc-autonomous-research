"""
Tests for Context Metrics Collector.

Phase G - Dashboard integration for token monitoring.
"""

import pytest
import time
from datetime import datetime, timezone

from monitoring.context_metrics import (
    ContextMetrics,
    CompressionStats,
    ContextMetricsCollector,
    get_context_metrics_collector,
    reset_context_metrics_collector,
)


@pytest.fixture
def collector():
    """Create a fresh ContextMetricsCollector."""
    reset_context_metrics_collector()
    return ContextMetricsCollector(history_size=100)


@pytest.fixture
def sample_metrics():
    """Create sample ContextMetrics."""
    return ContextMetrics(
        tokens_used=50000,
        tokens_limit=180000,
        usage_percent=27.78,
        tier1_tokens=4000,
        tier2_tokens=16000,
        compression_applied=False,
        tokens_saved=0,
        agent_name="proposer",
        model_name="claude-3-sonnet"
    )


class TestContextMetrics:
    """Tests for ContextMetrics dataclass."""

    def test_metrics_creation(self, sample_metrics):
        """ContextMetrics should store all fields."""
        assert sample_metrics.tokens_used == 50000
        assert sample_metrics.tokens_limit == 180000
        assert sample_metrics.usage_percent == 27.78
        assert sample_metrics.agent_name == "proposer"
        assert sample_metrics.model_name == "claude-3-sonnet"

    def test_tokens_available(self, sample_metrics):
        """tokens_available should calculate correctly."""
        assert sample_metrics.tokens_available == 130000

    def test_tokens_available_zero_when_over(self):
        """tokens_available should be 0 when over budget."""
        metrics = ContextMetrics(
            tokens_used=200000,
            tokens_limit=180000,
            usage_percent=111.11
        )
        assert metrics.tokens_available == 0

    def test_is_over_budget_false(self, sample_metrics):
        """is_over_budget should be False when within budget."""
        assert sample_metrics.is_over_budget is False

    def test_is_over_budget_true(self):
        """is_over_budget should be True when exceeded."""
        metrics = ContextMetrics(
            tokens_used=200000,
            tokens_limit=180000,
            usage_percent=111.11
        )
        assert metrics.is_over_budget is True

    def test_to_dict(self, sample_metrics):
        """to_dict should return complete dict."""
        d = sample_metrics.to_dict()

        assert d["tokens_used"] == 50000
        assert d["tokens_limit"] == 180000
        assert d["agent_name"] == "proposer"
        assert "timestamp" in d

    def test_from_dict(self, sample_metrics):
        """from_dict should recreate metrics."""
        d = sample_metrics.to_dict()
        restored = ContextMetrics.from_dict(d)

        assert restored.tokens_used == sample_metrics.tokens_used
        assert restored.agent_name == sample_metrics.agent_name

    def test_default_timestamp(self):
        """Timestamp should default to now."""
        before = datetime.now(timezone.utc).isoformat()
        metrics = ContextMetrics(
            tokens_used=0,
            tokens_limit=100,
            usage_percent=0
        )
        after = datetime.now(timezone.utc).isoformat()

        assert before <= metrics.timestamp <= after


class TestCompressionStats:
    """Tests for CompressionStats dataclass."""

    def test_initial_stats(self):
        """Initial stats should be zero."""
        stats = CompressionStats()

        assert stats.total_compressions == 0
        assert stats.total_tokens_saved == 0
        assert stats.aggressive_count == 0
        assert stats.moderate_count == 0
        assert stats.light_count == 0
        assert stats.average_compression_ratio == 1.0
        assert stats.last_compression is None

    def test_record_compression_aggressive(self):
        """Recording aggressive compression should update stats."""
        stats = CompressionStats()
        stats.record_compression("aggressive", 1000, 0.3)

        assert stats.total_compressions == 1
        assert stats.total_tokens_saved == 1000
        assert stats.aggressive_count == 1
        assert stats.last_compression is not None

    def test_record_compression_moderate(self):
        """Recording moderate compression should update stats."""
        stats = CompressionStats()
        stats.record_compression("moderate", 500, 0.6)

        assert stats.moderate_count == 1

    def test_record_compression_light(self):
        """Recording light compression should update stats."""
        stats = CompressionStats()
        stats.record_compression("light", 200, 0.8)

        assert stats.light_count == 1

    def test_average_ratio_updates(self):
        """Average compression ratio should update with EMA."""
        stats = CompressionStats()

        # First compression sets the average
        stats.record_compression("light", 100, 0.8)
        assert stats.average_compression_ratio == 0.8

        # Subsequent compressions use EMA
        stats.record_compression("moderate", 200, 0.5)
        # EMA: 0.1 * 0.5 + 0.9 * 0.8 = 0.77
        assert 0.7 < stats.average_compression_ratio < 0.8

    def test_to_dict(self):
        """to_dict should return complete dict."""
        stats = CompressionStats()
        stats.record_compression("aggressive", 1000, 0.3)

        d = stats.to_dict()
        assert d["total_compressions"] == 1
        assert d["total_tokens_saved"] == 1000
        assert d["aggressive_count"] == 1


class TestContextMetricsCollector:
    """Tests for ContextMetricsCollector class."""

    def test_collector_creation(self, collector):
        """Collector should initialize with empty state."""
        assert collector.history_size == 100
        assert len(collector._metrics_history) == 0
        assert len(collector._current_usage) == 0

    def test_record_metrics(self, collector, sample_metrics):
        """record should store metrics in history and current."""
        collector.record(sample_metrics)

        assert len(collector._metrics_history) == 1
        assert "proposer" in collector._current_usage

    def test_record_multiple_agents(self, collector):
        """Multiple agents should be tracked separately."""
        metrics1 = ContextMetrics(
            tokens_used=10000, tokens_limit=180000, usage_percent=5.56,
            agent_name="proposer"
        )
        metrics2 = ContextMetrics(
            tokens_used=20000, tokens_limit=57600, usage_percent=34.72,
            agent_name="critic"
        )

        collector.record(metrics1)
        collector.record(metrics2)

        current = collector.get_current_usage()
        assert "proposer" in current
        assert "critic" in current
        assert current["proposer"]["tokens_used"] == 10000
        assert current["critic"]["tokens_used"] == 20000

    def test_record_updates_current(self, collector):
        """Recording same agent should update current."""
        metrics1 = ContextMetrics(
            tokens_used=10000, tokens_limit=180000, usage_percent=5.56,
            agent_name="proposer"
        )
        metrics2 = ContextMetrics(
            tokens_used=50000, tokens_limit=180000, usage_percent=27.78,
            agent_name="proposer"
        )

        collector.record(metrics1)
        collector.record(metrics2)

        current = collector.get_current_usage()
        assert current["proposer"]["tokens_used"] == 50000

    def test_record_compression(self, collector):
        """record_compression should update stats."""
        collector.record_compression("moderate", 10000, 5000)

        stats = collector.get_compression_stats()
        assert stats["total_compressions"] == 1
        assert stats["total_tokens_saved"] == 5000
        assert stats["moderate_count"] == 1

    def test_get_current_usage(self, collector, sample_metrics):
        """get_current_usage should return all agents."""
        collector.record(sample_metrics)

        current = collector.get_current_usage()
        assert isinstance(current, dict)
        assert "proposer" in current
        assert current["proposer"]["tokens_used"] == 50000

    def test_get_agent_usage(self, collector, sample_metrics):
        """get_agent_usage should return specific agent."""
        collector.record(sample_metrics)

        usage = collector.get_agent_usage("proposer")
        assert usage is not None
        assert usage.tokens_used == 50000

    def test_get_agent_usage_not_found(self, collector):
        """get_agent_usage should return None for unknown agent."""
        assert collector.get_agent_usage("unknown") is None

    def test_get_history(self, collector):
        """get_history should return historical metrics."""
        for i in range(5):
            metrics = ContextMetrics(
                tokens_used=i * 10000,
                tokens_limit=180000,
                usage_percent=i * 5.56,
                agent_name=f"agent_{i}"
            )
            collector.record(metrics)

        history = collector.get_history()
        assert len(history) == 5

    def test_get_history_last_n(self, collector):
        """get_history should limit to last_n entries."""
        for i in range(10):
            metrics = ContextMetrics(
                tokens_used=i * 10000,
                tokens_limit=180000,
                usage_percent=i * 5.56
            )
            collector.record(metrics)

        history = collector.get_history(last_n=3)
        assert len(history) == 3

    def test_get_history_filter_agent(self, collector):
        """get_history should filter by agent name."""
        for i in range(5):
            metrics1 = ContextMetrics(
                tokens_used=i * 10000, tokens_limit=180000,
                usage_percent=i * 5.56, agent_name="proposer"
            )
            metrics2 = ContextMetrics(
                tokens_used=i * 5000, tokens_limit=57600,
                usage_percent=i * 8.68, agent_name="critic"
            )
            collector.record(metrics1)
            collector.record(metrics2)

        proposer_history = collector.get_history(agent_name="proposer")
        assert len(proposer_history) == 5
        assert all(m["agent_name"] == "proposer" for m in proposer_history)

    def test_get_summary(self, collector):
        """get_summary should return aggregate stats."""
        metrics1 = ContextMetrics(
            tokens_used=50000, tokens_limit=180000, usage_percent=27.78,
            agent_name="proposer"
        )
        metrics2 = ContextMetrics(
            tokens_used=30000, tokens_limit=57600, usage_percent=52.08,
            agent_name="critic"
        )

        collector.record(metrics1)
        collector.record(metrics2)

        summary = collector.get_summary()

        assert summary["agents_tracked"] == 2
        assert summary["total_tokens_used"] == 80000
        assert summary["history_entries"] == 2
        assert "compression_stats" in summary
        assert "current_usage" in summary

    def test_get_summary_over_budget(self, collector):
        """get_summary should count over-budget agents."""
        metrics1 = ContextMetrics(
            tokens_used=50000, tokens_limit=180000, usage_percent=27.78,
            agent_name="normal"
        )
        metrics2 = ContextMetrics(
            tokens_used=70000, tokens_limit=57600, usage_percent=121.53,
            agent_name="over_budget"
        )

        collector.record(metrics1)
        collector.record(metrics2)

        summary = collector.get_summary()
        assert summary["agents_over_budget"] == 1

    def test_get_usage_trends(self, collector):
        """get_usage_trends should calculate trend data."""
        for i in range(20):
            metrics = ContextMetrics(
                tokens_used=i * 5000,
                tokens_limit=180000,
                usage_percent=i * 2.78
            )
            collector.record(metrics)

        trends = collector.get_usage_trends()

        assert trends["data_points"] == 20
        assert trends["avg_usage_percent"] > 0
        assert trends["max_usage_percent"] >= trends["min_usage_percent"]
        assert trends["trend"] in ["increasing", "decreasing", "stable", "insufficient_data"]

    def test_get_usage_trends_empty(self, collector):
        """get_usage_trends should handle empty history."""
        trends = collector.get_usage_trends()

        assert trends["data_points"] == 0
        assert trends["trend"] == "stable"

    def test_get_usage_trends_increasing(self, collector):
        """get_usage_trends should detect increasing trend."""
        # Create clearly increasing data
        for i in range(20):
            metrics = ContextMetrics(
                tokens_used=i * 10000,
                tokens_limit=180000,
                usage_percent=i * 5.56
            )
            collector.record(metrics)

        trends = collector.get_usage_trends()
        assert trends["trend"] == "increasing"

    def test_clear_history(self, collector, sample_metrics):
        """clear_history should remove all history."""
        collector.record(sample_metrics)
        assert len(collector._metrics_history) > 0

        collector.clear_history()
        assert len(collector._metrics_history) == 0

    def test_reset(self, collector, sample_metrics):
        """reset should clear everything."""
        collector.record(sample_metrics)
        collector.record_compression("light", 100, 50)

        collector.reset()

        assert len(collector._metrics_history) == 0
        assert len(collector._current_usage) == 0
        assert collector._compression_stats.total_compressions == 0

    def test_history_size_limit(self):
        """History should respect size limit."""
        collector = ContextMetricsCollector(history_size=5)

        for i in range(10):
            metrics = ContextMetrics(
                tokens_used=i * 1000,
                tokens_limit=10000,
                usage_percent=i * 10
            )
            collector.record(metrics)

        # Should only keep last 5
        history = collector.get_history()
        assert len(history) == 5


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_context_metrics_collector_singleton(self):
        """get_context_metrics_collector should return same instance."""
        reset_context_metrics_collector()

        collector1 = get_context_metrics_collector()
        collector2 = get_context_metrics_collector()

        assert collector1 is collector2

    def test_reset_context_metrics_collector(self):
        """reset_context_metrics_collector should clear singleton."""
        collector1 = get_context_metrics_collector()
        reset_context_metrics_collector()
        collector2 = get_context_metrics_collector()

        assert collector1 is not collector2


class TestThreadSafety:
    """Tests for thread safety (basic validation)."""

    def test_concurrent_records(self, collector):
        """Multiple records should not corrupt state."""
        import threading

        def record_metrics(agent_id):
            for i in range(100):
                metrics = ContextMetrics(
                    tokens_used=i * 100,
                    tokens_limit=10000,
                    usage_percent=i,
                    agent_name=f"agent_{agent_id}"
                )
                collector.record(metrics)

        threads = [
            threading.Thread(target=record_metrics, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all agents tracked
        current = collector.get_current_usage()
        assert len(current) == 5
