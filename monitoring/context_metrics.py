"""
Context Metrics Collector for Dashboard Integration

Phase G - Extends existing monitoring/ from Phase F with context-aware metrics.

Provides real-time token usage tracking, compression statistics,
and integration with the monitoring dashboard.
"""

import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextMetrics:
    """
    Metrics for context/token usage at a point in time.

    Attributes:
        tokens_used: Current tokens used
        tokens_limit: Token limit for the model
        usage_percent: Percentage of limit used
        tier1_tokens: Tokens in hot tier (working memory)
        tier2_tokens: Tokens in warm tier (recent history)
        compression_applied: Whether compression was applied
        tokens_saved: Tokens saved by compression (if applied)
        agent_name: Name of the agent (if applicable)
        model_name: Name of the model
        timestamp: When metrics were recorded
    """
    tokens_used: int
    tokens_limit: int
    usage_percent: float
    tier1_tokens: int = 0
    tier2_tokens: int = 0
    compression_applied: bool = False
    tokens_saved: int = 0
    agent_name: str = ""
    model_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMetrics":
        """Create from dictionary."""
        return cls(**data)

    @property
    def tokens_available(self) -> int:
        """Tokens still available within limit."""
        return max(0, self.tokens_limit - self.tokens_used)

    @property
    def is_over_budget(self) -> bool:
        """Check if over token budget."""
        return self.tokens_used > self.tokens_limit


@dataclass
class CompressionStats:
    """
    Statistics about compression operations.

    Tracks compression effectiveness over time.
    """
    total_compressions: int = 0
    total_tokens_saved: int = 0
    aggressive_count: int = 0
    moderate_count: int = 0
    light_count: int = 0
    average_compression_ratio: float = 1.0
    last_compression: Optional[str] = None

    def record_compression(
        self,
        strategy: str,
        tokens_saved: int,
        compression_ratio: float
    ) -> None:
        """Record a compression operation."""
        self.total_compressions += 1
        self.total_tokens_saved += tokens_saved
        self.last_compression = datetime.now(timezone.utc).isoformat()

        # Update strategy counts
        if strategy == "aggressive":
            self.aggressive_count += 1
        elif strategy == "moderate":
            self.moderate_count += 1
        elif strategy == "light":
            self.light_count += 1

        # Update running average
        if self.total_compressions == 1:
            self.average_compression_ratio = compression_ratio
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_compression_ratio = (
                alpha * compression_ratio +
                (1 - alpha) * self.average_compression_ratio
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ContextMetricsCollector:
    """
    Collects and manages context metrics for the dashboard.

    Provides:
    - Real-time token usage tracking per agent
    - Compression statistics
    - Historical metrics for trend analysis
    - Integration with existing GPU monitoring
    """

    DEFAULT_HISTORY_SIZE = 1000

    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE):
        """
        Initialize the metrics collector.

        Args:
            history_size: Maximum history entries to retain
        """
        self.history_size = history_size
        self._metrics_history: deque = deque(maxlen=history_size)
        self._current_usage: Dict[str, ContextMetrics] = {}
        self._compression_stats = CompressionStats()
        self._lock = Lock()

        logger.info(f"ContextMetricsCollector initialized with history_size={history_size}")

    def record(self, metrics: ContextMetrics) -> None:
        """
        Record context metrics.

        Args:
            metrics: ContextMetrics to record
        """
        with self._lock:
            # Add to history
            self._metrics_history.append(metrics)

            # Update current usage for agent
            key = metrics.agent_name or "default"
            self._current_usage[key] = metrics

            # Log if over budget
            if metrics.is_over_budget:
                logger.warning(
                    f"Token budget exceeded for {key}: "
                    f"{metrics.tokens_used}/{metrics.tokens_limit} "
                    f"({metrics.usage_percent:.1f}%)"
                )

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int
    ) -> None:
        """
        Record a compression operation.

        Args:
            strategy: Compression strategy used
            original_tokens: Original token count
            compressed_tokens: Compressed token count
        """
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        with self._lock:
            self._compression_stats.record_compression(
                strategy, tokens_saved, compression_ratio
            )

        logger.debug(
            f"Compression recorded: {strategy}, "
            f"saved {tokens_saved} tokens ({(1-compression_ratio)*100:.1f}%)"
        )

    def get_current_usage(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current token usage for all tracked agents.

        Returns:
            Dict mapping agent names to their current metrics
        """
        with self._lock:
            return {
                name: metrics.to_dict()
                for name, metrics in self._current_usage.items()
            }

    def get_agent_usage(self, agent_name: str) -> Optional[ContextMetrics]:
        """
        Get current usage for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ContextMetrics for the agent, or None if not found
        """
        with self._lock:
            return self._current_usage.get(agent_name)

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dict with compression statistics
        """
        with self._lock:
            return self._compression_stats.to_dict()

    def get_history(
        self,
        last_n: Optional[int] = None,
        agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics.

        Args:
            last_n: Number of entries to return (default: all)
            agent_name: Filter by agent name (optional)

        Returns:
            List of historical metrics dicts
        """
        with self._lock:
            history = list(self._metrics_history)

        # Filter by agent if specified
        if agent_name:
            history = [m for m in history if m.agent_name == agent_name]

        # Limit entries
        if last_n:
            history = history[-last_n:]

        return [m.to_dict() for m in history]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of context metrics.

        Returns:
            Dict with summary statistics
        """
        with self._lock:
            current = {
                name: metrics.to_dict()
                for name, metrics in self._current_usage.items()
            }

            # Calculate totals
            total_used = sum(m.tokens_used for m in self._current_usage.values())
            total_limit = sum(m.tokens_limit for m in self._current_usage.values())
            agents_count = len(self._current_usage)
            over_budget = sum(1 for m in self._current_usage.values() if m.is_over_budget)

            return {
                "agents_tracked": agents_count,
                "agents_over_budget": over_budget,
                "total_tokens_used": total_used,
                "total_tokens_limit": total_limit,
                "overall_usage_percent": round(total_used / total_limit * 100, 2) if total_limit > 0 else 0,
                "history_entries": len(self._metrics_history),
                "compression_stats": self._compression_stats.to_dict(),
                "current_usage": current
            }

    def get_usage_trends(
        self,
        agent_name: Optional[str] = None,
        last_n: int = 100
    ) -> Dict[str, Any]:
        """
        Get usage trends over time.

        Args:
            agent_name: Filter by agent (optional)
            last_n: Number of data points

        Returns:
            Dict with trend data
        """
        history = self.get_history(last_n=last_n, agent_name=agent_name)

        if not history:
            return {
                "data_points": 0,
                "avg_usage_percent": 0,
                "max_usage_percent": 0,
                "min_usage_percent": 0,
                "compressions_triggered": 0,
                "trend": "stable"
            }

        usage_percents = [m["usage_percent"] for m in history]
        compressions = sum(1 for m in history if m["compression_applied"])

        # Calculate trend
        if len(usage_percents) >= 10:
            first_half = sum(usage_percents[:len(usage_percents)//2]) / (len(usage_percents)//2)
            second_half = sum(usage_percents[len(usage_percents)//2:]) / (len(usage_percents) - len(usage_percents)//2)

            if second_half > first_half * 1.1:
                trend = "increasing"
            elif second_half < first_half * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "data_points": len(history),
            "avg_usage_percent": round(sum(usage_percents) / len(usage_percents), 2),
            "max_usage_percent": round(max(usage_percents), 2),
            "min_usage_percent": round(min(usage_percents), 2),
            "compressions_triggered": compressions,
            "trend": trend
        }

    def clear_history(self) -> None:
        """Clear metrics history."""
        with self._lock:
            self._metrics_history.clear()
            logger.info("Context metrics history cleared")

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._metrics_history.clear()
            self._current_usage.clear()
            self._compression_stats = CompressionStats()
            logger.info("Context metrics collector reset")


# Singleton pattern for global access
_context_metrics_collector: Optional[ContextMetricsCollector] = None
_collector_lock = Lock()


def get_context_metrics_collector(
    history_size: int = ContextMetricsCollector.DEFAULT_HISTORY_SIZE
) -> ContextMetricsCollector:
    """
    Get or create the global ContextMetricsCollector instance.

    Args:
        history_size: Maximum history entries to retain

    Returns:
        The singleton ContextMetricsCollector instance
    """
    global _context_metrics_collector

    with _collector_lock:
        if _context_metrics_collector is None:
            _context_metrics_collector = ContextMetricsCollector(history_size)
        return _context_metrics_collector


def reset_context_metrics_collector() -> None:
    """Reset the global ContextMetricsCollector instance (for testing)."""
    global _context_metrics_collector
    with _collector_lock:
        _context_metrics_collector = None
