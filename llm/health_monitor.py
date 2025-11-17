"""
LLM Health Monitor: Model availability and performance tracking
================================================================

Monitors health of all LLM endpoints and enables automatic failover.

Features:
- Periodic health checks
- Model availability tracking
- Circuit breaker pattern
- Auto-failover to MockLLMClient
- Performance metrics collection
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from llm.models import ModelConfig, get_model_config, list_all_models
from llm.client import LLMClient
from llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Health metrics for a model endpoint"""
    model_id: str
    state: ModelState = ModelState.UNKNOWN
    last_check_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_error: Optional[str] = None
    response_times: List[float] = field(default_factory=list)

    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def update_response_time(self, response_time_ms: float):
        """Update average response time"""
        self.response_times.append(response_time_ms)
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        self.avg_response_time_ms = sum(self.response_times) / len(self.response_times)


class HealthMonitor:
    """
    Monitors health of all LLM endpoints.

    Features:
    - Periodic health checks
    - Circuit breaker pattern (fail fast when model is down)
    - Auto-failover to MockLLMClient
    - Performance metrics tracking
    """

    def __init__(
        self,
        check_interval_seconds: int = 60,
        failure_threshold: int = 3,
        success_threshold: int = 2,
        timeout_ms: int = 5000,
        enable_auto_monitoring: bool = True
    ):
        """
        Initialize health monitor.

        Args:
            check_interval_seconds: How often to check model health
            failure_threshold: Consecutive failures before marking as FAILED
            success_threshold: Consecutive successes to recover from FAILED
            timeout_ms: Health check timeout in milliseconds
            enable_auto_monitoring: Start background monitoring thread
        """
        self.check_interval = check_interval_seconds
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_ms = timeout_ms

        # Model health state
        self.health_metrics: Dict[str, HealthMetrics] = {}

        # Circuit breaker state
        self.circuit_open: Dict[str, bool] = {}

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Initialize metrics for all models
        for model_config in list_all_models():
            self.health_metrics[model_config.model_id] = HealthMetrics(
                model_id=model_config.model_id,
                state=ModelState.OFFLINE if model_config.offline else ModelState.UNKNOWN
            )
            self.circuit_open[model_config.model_id] = False

        if enable_auto_monitoring:
            self.start_monitoring()

        logger.info(f"HealthMonitor initialized (interval={check_interval_seconds}s)")

    def start_monitoring(self):
        """Start background health monitoring thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Background health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Background health monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self.check_all_models()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(self.check_interval * 10):
                if not self.monitoring_active:
                    break
                time.sleep(0.1)

    def check_model_health(self, model_id: str) -> ModelState:
        """
        Check health of a specific model.

        Args:
            model_id: Model to check

        Returns:
            Current model state
        """
        config = get_model_config(model_id)
        if not config:
            logger.warning(f"Unknown model: {model_id}")
            return ModelState.UNKNOWN

        metrics = self.health_metrics.get(model_id)
        if not metrics:
            metrics = HealthMetrics(model_id=model_id)
            self.health_metrics[model_id] = metrics

        # Skip offline models (always OFFLINE)
        if config.offline or config.model_id == "mock-llm":
            metrics.state = ModelState.OFFLINE
            return ModelState.OFFLINE

        # Perform health check
        start_time = time.time()
        try:
            client = LLMClient(
                endpoint=config.endpoint,
                model_name=config.model_id,
                timeout=self.timeout_ms // 1000
            )

            # Simple health check: generate short response
            response = client.generate(
                prompt="Health check",
                max_tokens=10,
                temperature=0.0
            )

            # Success
            response_time_ms = (time.time() - start_time) * 1000
            metrics.update_response_time(response_time_ms)
            metrics.total_requests += 1
            metrics.successful_requests += 1
            metrics.consecutive_failures = 0
            metrics.last_success_time = datetime.now()
            metrics.last_check_time = datetime.now()
            metrics.last_error = None

            # Update state
            if metrics.consecutive_failures == 0:
                metrics.state = ModelState.HEALTHY
                self.circuit_open[model_id] = False

            logger.debug(f"Health check passed for {model_id} ({response_time_ms:.0f}ms)")

            return metrics.state

        except Exception as e:
            # Failure
            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.consecutive_failures += 1
            metrics.last_check_time = datetime.now()
            metrics.last_error = str(e)

            # Update state based on failure count
            if metrics.consecutive_failures >= self.failure_threshold:
                metrics.state = ModelState.FAILED
                self.circuit_open[model_id] = True
                logger.error(f"Model {model_id} marked as FAILED after {metrics.consecutive_failures} failures")
            else:
                metrics.state = ModelState.DEGRADED

            logger.warning(f"Health check failed for {model_id}: {e}")

            return metrics.state

    def check_all_models(self):
        """Check health of all online models"""
        for model_id in self.health_metrics.keys():
            config = get_model_config(model_id)
            if config and not config.offline:
                self.check_model_health(model_id)

    def is_model_available(self, model_id: str) -> bool:
        """
        Check if model is available (not in circuit breaker state)

        Args:
            model_id: Model to check

        Returns:
            True if model is available
        """
        # Offline models always available (local/mock)
        config = get_model_config(model_id)
        if config and (config.offline or config.model_id == "mock-llm"):
            return True

        # Check circuit breaker
        if self.circuit_open.get(model_id, False):
            return False

        # Check state
        metrics = self.health_metrics.get(model_id)
        if not metrics:
            return False

        return metrics.state in [ModelState.HEALTHY, ModelState.DEGRADED, ModelState.UNKNOWN]

    def get_best_available_model(self, preferred_model_id: str) -> str:
        """
        Get best available model (with fallback)

        Args:
            preferred_model_id: Preferred model

        Returns:
            Model ID to use (may be mock if all failed)
        """
        # Check if preferred model is available
        if self.is_model_available(preferred_model_id):
            return preferred_model_id

        logger.warning(f"Preferred model {preferred_model_id} unavailable, looking for fallback")

        # Find best fallback
        available_models = [
            (model_id, metrics)
            for model_id, metrics in self.health_metrics.items()
            if self.is_model_available(model_id) and model_id != "mock-llm"
        ]

        if available_models:
            # Sort by success rate
            available_models.sort(key=lambda x: x[1].success_rate(), reverse=True)
            fallback_id = available_models[0][0]
            logger.info(f"Falling back to {fallback_id}")
            return fallback_id

        # Last resort: mock
        logger.warning("All models unavailable, falling back to MockLLMClient")
        return "mock-llm"

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary for all models

        Returns:
            Health summary dictionary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "overall": {
                "total_models": len(self.health_metrics),
                "healthy": 0,
                "degraded": 0,
                "failed": 0,
                "offline": 0,
                "unknown": 0
            }
        }

        for model_id, metrics in self.health_metrics.items():
            summary["models"][model_id] = {
                "state": metrics.state.value,
                "success_rate": metrics.success_rate(),
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "consecutive_failures": metrics.consecutive_failures,
                "last_check": metrics.last_check_time.isoformat() if metrics.last_check_time else None,
                "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                "circuit_open": self.circuit_open.get(model_id, False)
            }

            # Update overall counts
            state_key = metrics.state.value
            if state_key in summary["overall"]:
                summary["overall"][state_key] += 1

        return summary

    def reset_circuit_breaker(self, model_id: str):
        """Manually reset circuit breaker for a model"""
        if model_id in self.circuit_open:
            self.circuit_open[model_id] = False
            metrics = self.health_metrics.get(model_id)
            if metrics:
                metrics.consecutive_failures = 0
                metrics.state = ModelState.UNKNOWN
            logger.info(f"Circuit breaker reset for {model_id}")

    def __repr__(self) -> str:
        healthy = sum(1 for m in self.health_metrics.values() if m.state == ModelState.HEALTHY)
        total = len(self.health_metrics)
        return f"<HealthMonitor healthy={healthy}/{total}>"


# Global health monitor instance
_global_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """
    Get global health monitor instance

    Returns:
        HealthMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor(enable_auto_monitoring=True)
    return _global_monitor
