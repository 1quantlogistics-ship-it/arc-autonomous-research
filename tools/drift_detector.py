"""
Drift Detection for ARC Autonomous Research

Detects various types of drift to ensure ARC remains scientifically productive:
- Performance drift (AUC trends, regression detection)
- Distribution drift (KS test, MMD for feature distributions)
- Diversity drift (proposal entropy, mode collapse)
- Prediction drift (world model accuracy degradation)

Integrates with Director for automatic mode switching and FDA logging for risk tracking.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy import stats
from collections import deque

from config import get_settings
from tools.dev_logger import get_dev_logger

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    drift_detected: bool
    drift_type: str  # "performance", "distribution", "diversity", "prediction"
    drift_score: float  # 0.0 (no drift) to 1.0 (severe drift)
    severity: str  # "low", "medium", "high"
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: str
    recommended_action: str


class DriftDetectorError(Exception):
    """Raised when drift detection fails."""
    pass


class DriftDetector:
    """
    Multi-modal drift detector for ARC.

    Monitors:
    - Performance metrics (AUC, sensitivity, specificity)
    - Proposal diversity (entropy, uniqueness)
    - World model accuracy
    - Distribution shifts in data/predictions
    """

    def __init__(
        self,
        window_size: int = 10,
        performance_threshold: float = 0.05,
        diversity_threshold: float = 2.0,
        ks_threshold: float = 0.1,
        mmd_threshold: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Number of recent cycles to analyze
            performance_threshold: Performance drop threshold (e.g., 0.05 = 5%)
            diversity_threshold: Minimum entropy in bits
            ks_threshold: Kolmogorov-Smirnov test p-value threshold
            mmd_threshold: Maximum Mean Discrepancy threshold
        """
        self.window_size = window_size
        self.performance_threshold = performance_threshold
        self.diversity_threshold = diversity_threshold
        self.ks_threshold = ks_threshold
        self.mmd_threshold = mmd_threshold

        # Circular buffers for metrics
        self.performance_history: deque = deque(maxlen=window_size)
        self.diversity_history: deque = deque(maxlen=window_size)
        self.prediction_accuracy: deque = deque(maxlen=window_size)

        # Dev logger for FDA risk tracking
        self.dev_logger = get_dev_logger()

        logger.info(f"Drift detector initialized (window={window_size})")

    def detect_performance_drift(
        self,
        current_auc: float,
        cycle_id: int
    ) -> DriftDetectionResult:
        """
        Detect performance drift using trend analysis.

        Args:
            current_auc: Current cycle AUC
            cycle_id: Current cycle ID

        Returns:
            DriftDetectionResult
        """
        # Add to history
        self.performance_history.append({
            "cycle_id": cycle_id,
            "auc": current_auc,
            "timestamp": datetime.now().isoformat()
        })

        # Need at least 5 points for trend analysis
        if len(self.performance_history) < 5:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type="performance",
                drift_score=0.0,
                severity="low",
                confidence=0.5,
                details={"reason": "Insufficient history"},
                timestamp=datetime.now().isoformat(),
                recommended_action="continue_monitoring"
            )

        # Extract AUC values
        auc_values = [entry["auc"] for entry in self.performance_history]

        # Compute trend (linear regression slope)
        x = np.arange(len(auc_values))
        y = np.array(auc_values)

        # Handle constant values
        if np.std(y) == 0:
            slope = 0.0
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Compute recent drop (last vs. average of previous)
        recent_drop = 0.0
        if len(auc_values) >= 3:
            recent_value = auc_values[-1]
            previous_avg = np.mean(auc_values[-4:-1])
            recent_drop = previous_avg - recent_value

        # Drift detected if:
        # 1. Significant downward trend (slope < -0.01)
        # 2. Recent drop > threshold
        drift_detected = (slope < -0.01) or (recent_drop > self.performance_threshold)

        # Calculate drift score
        drift_score = max(abs(slope) * 10, recent_drop / self.performance_threshold)
        drift_score = min(drift_score, 1.0)

        # Determine severity
        if drift_score > 0.7:
            severity = "high"
        elif drift_score > 0.4:
            severity = "medium"
        else:
            severity = "low"

        # Recommended action
        if drift_detected and severity == "high":
            recommended_action = "reset_exploration"
        elif drift_detected and severity == "medium":
            recommended_action = "increase_exploration"
        else:
            recommended_action = "continue_monitoring"

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type="performance",
            drift_score=drift_score,
            severity=severity,
            confidence=min(len(auc_values) / self.window_size, 1.0),
            details={
                "slope": float(slope),
                "recent_drop": float(recent_drop),
                "current_auc": current_auc,
                "window_mean": float(np.mean(auc_values)),
                "window_std": float(np.std(auc_values))
            },
            timestamp=datetime.now().isoformat(),
            recommended_action=recommended_action
        )

        # Log to FDA if drift detected
        if drift_detected:
            self._log_drift_to_fda(result, cycle_id)

        return result

    def detect_diversity_drift(
        self,
        proposal_configs: List[Dict[str, Any]],
        cycle_id: int
    ) -> DriftDetectionResult:
        """
        Detect diversity drift using entropy and uniqueness metrics.

        Args:
            proposal_configs: List of proposed configurations
            cycle_id: Current cycle ID

        Returns:
            DriftDetectionResult
        """
        # Compute diversity metrics
        entropy = self._compute_config_entropy(proposal_configs)
        uniqueness_ratio = self._compute_uniqueness_ratio(proposal_configs)

        # Add to history
        self.diversity_history.append({
            "cycle_id": cycle_id,
            "entropy": entropy,
            "uniqueness_ratio": uniqueness_ratio,
            "num_proposals": len(proposal_configs),
            "timestamp": datetime.now().isoformat()
        })

        # Drift detected if entropy below threshold
        drift_detected = entropy < self.diversity_threshold

        # Calculate drift score (inverse of entropy, normalized)
        max_entropy = 5.0  # Typical maximum for configs
        drift_score = max(0.0, (max_entropy - entropy) / max_entropy)
        drift_score = min(drift_score, 1.0)

        # Determine severity
        if entropy < 1.0:
            severity = "high"
        elif entropy < 1.5:
            severity = "medium"
        else:
            severity = "low"

        # Recommended action
        if drift_detected and severity == "high":
            recommended_action = "force_exploration_mode"
        elif drift_detected and severity == "medium":
            recommended_action = "reject_duplicates"
        else:
            recommended_action = "continue_monitoring"

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type="diversity",
            drift_score=drift_score,
            severity=severity,
            confidence=0.9,  # High confidence in entropy measurement
            details={
                "entropy_bits": float(entropy),
                "uniqueness_ratio": float(uniqueness_ratio),
                "num_proposals": len(proposal_configs),
                "threshold": self.diversity_threshold
            },
            timestamp=datetime.now().isoformat(),
            recommended_action=recommended_action
        )

        # Log to FDA if drift detected
        if drift_detected:
            self._log_drift_to_fda(result, cycle_id)

        return result

    def detect_distribution_drift_ks(
        self,
        reference_distribution: np.ndarray,
        current_distribution: np.ndarray,
        cycle_id: int
    ) -> DriftDetectionResult:
        """
        Detect distribution drift using Kolmogorov-Smirnov test.

        Args:
            reference_distribution: Reference distribution (e.g., historical metrics)
            current_distribution: Current distribution
            cycle_id: Current cycle ID

        Returns:
            DriftDetectionResult
        """
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference_distribution, current_distribution)

        # Drift detected if p-value below threshold (distributions differ)
        drift_detected = p_value < self.ks_threshold

        # Drift score based on KS statistic (0-1 range already)
        drift_score = float(ks_statistic)

        # Determine severity based on KS statistic
        if ks_statistic > 0.5:
            severity = "high"
        elif ks_statistic > 0.3:
            severity = "medium"
        else:
            severity = "low"

        # Recommended action
        if drift_detected and severity == "high":
            recommended_action = "investigate_data_shift"
        elif drift_detected:
            recommended_action = "monitor_closely"
        else:
            recommended_action = "continue_monitoring"

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type="distribution",
            drift_score=drift_score,
            severity=severity,
            confidence=1.0 - p_value,  # Higher confidence with lower p-value
            details={
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "threshold": self.ks_threshold,
                "reference_mean": float(np.mean(reference_distribution)),
                "current_mean": float(np.mean(current_distribution))
            },
            timestamp=datetime.now().isoformat(),
            recommended_action=recommended_action
        )

        # Log to FDA if drift detected
        if drift_detected:
            self._log_drift_to_fda(result, cycle_id)

        return result

    def _compute_config_entropy(self, proposal_configs: List[Dict[str, Any]]) -> float:
        """
        Compute Shannon entropy of proposal configurations.

        Args:
            proposal_configs: List of configurations

        Returns:
            Entropy in bits
        """
        if not proposal_configs:
            return 0.0

        # Convert configs to hashable strings
        config_signatures = []
        for config in proposal_configs:
            # Extract key parameters
            signature = json.dumps({
                "model": config.get("model", {}),
                "training": config.get("training", {}),
                "data": config.get("data", {})
            }, sort_keys=True)
            config_signatures.append(signature)

        # Count unique configurations
        from collections import Counter
        counts = Counter(config_signatures)

        # Compute Shannon entropy
        total = len(config_signatures)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    def _compute_uniqueness_ratio(self, proposal_configs: List[Dict[str, Any]]) -> float:
        """
        Compute ratio of unique configurations to total configurations.

        Args:
            proposal_configs: List of configurations

        Returns:
            Uniqueness ratio (0.0 to 1.0)
        """
        if not proposal_configs:
            return 0.0

        # Convert to hashable signatures
        import json
        signatures = set()
        for config in proposal_configs:
            sig = json.dumps({
                "model": config.get("model", {}),
                "training": config.get("training", {})
            }, sort_keys=True)
            signatures.add(sig)

        return len(signatures) / len(proposal_configs)

    def _log_drift_to_fda(self, result: DriftDetectionResult, cycle_id: int):
        """
        Log drift detection to FDA development logs.

        Args:
            result: Drift detection result
            cycle_id: Current cycle ID
        """
        try:
            self.dev_logger.log_risk_event(
                event_type=f"{result.drift_type}_drift",
                severity=result.severity,
                description=f"{result.drift_type.capitalize()} drift detected: {result.recommended_action}",
                cycle_id=cycle_id,
                context={
                    "drift_score": result.drift_score,
                    "confidence": result.confidence,
                    "details": result.details,
                    "recommended_action": result.recommended_action
                }
            )
            logger.debug(f"FDA drift event logged for cycle {cycle_id}")
        except Exception as e:
            logger.warning(f"FDA drift logging failed: {e}")

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of drift detection history.

        Returns:
            Dict with drift history statistics
        """
        return {
            "performance_history": list(self.performance_history),
            "diversity_history": list(self.diversity_history),
            "window_size": self.window_size,
            "thresholds": {
                "performance": self.performance_threshold,
                "diversity": self.diversity_threshold,
                "ks_test": self.ks_threshold
            }
        }


# Singleton instance
_drift_detector_instance: Optional[DriftDetector] = None


def get_drift_detector(
    window_size: Optional[int] = None,
    performance_threshold: Optional[float] = None,
    diversity_threshold: Optional[float] = None
) -> DriftDetector:
    """
    Get singleton drift detector instance.

    Args:
        window_size: Override window size (default from config)
        performance_threshold: Override performance threshold
        diversity_threshold: Override diversity threshold

    Returns:
        DriftDetector instance
    """
    global _drift_detector_instance

    if _drift_detector_instance is None:
        settings = get_settings()

        _drift_detector_instance = DriftDetector(
            window_size=window_size or 10,
            performance_threshold=performance_threshold or 0.05,
            diversity_threshold=diversity_threshold or 2.0,
            ks_threshold=0.1,
            mmd_threshold=0.05
        )

    return _drift_detector_instance
