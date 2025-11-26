"""
Medical Imaging Metrics for Clinical-Grade Evaluation.

Phase G - Implements pAUC, AUC-PR, calibration metrics, and constrained sensitivity.
These metrics are specifically designed for glaucoma detection evaluation where
clinical relevance is paramount.

Author: ARC Team (Dev 1)
Created: 2025-11-26
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MedicalMetricsConfig:
    """Configuration for medical metrics computation."""
    pauc_fpr_range: Tuple[float, float] = (0.0, 0.2)  # Clinically relevant FPR range
    sensitivity_at_specificity: float = 0.95  # FDA threshold
    calibration_bins: int = 10
    confidence_level: float = 0.95  # For CIs
    n_bootstrap: int = 1000  # Bootstrap samples for CIs


class MedicalMetrics:
    """
    Comprehensive medical imaging metrics for clinical evaluation.

    Provides:
    - Partial AUC (pAUC) for clinically relevant FPR ranges
    - AUC-PR for imbalanced datasets
    - Sensitivity/Specificity at thresholds
    - Calibration metrics (Brier, ECE, MCE)
    - Bootstrap confidence intervals
    """

    def __init__(self, config: Optional[MedicalMetricsConfig] = None):
        self.config = config or MedicalMetricsConfig()

    def compute_partial_auc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        fpr_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True
    ) -> float:
        """
        Compute partial AUC in clinically relevant FPR range.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_score: Predicted probabilities
            fpr_range: (min_fpr, max_fpr) range, defaults to config
            normalize: If True, normalize pAUC to 0-1 scale

        Returns:
            Partial AUC value (normalized if requested)
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.error("sklearn required for partial AUC computation")
            return 0.0

        fpr_range = fpr_range or self.config.pauc_fpr_range
        max_fpr = fpr_range[1]

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        # Check for valid input
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true, returning 0.0")
            return 0.0

        try:
            pauc = roc_auc_score(y_true, y_score, max_fpr=max_fpr)
            if normalize:
                # Normalize to 0-1 range: pAUC / max_fpr
                pauc = pauc / max_fpr
            return float(pauc)
        except Exception as e:
            logger.error(f"Error computing partial AUC: {e}")
            return 0.0

    def compute_auc_pr(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        with_ci: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute AUC-PR (Area Under Precision-Recall Curve) with confidence intervals.

        Critical for imbalanced medical datasets where positive class is rare.

        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            with_ci: Compute bootstrap confidence intervals

        Returns:
            (auc_pr, ci_lower, ci_upper) or (auc_pr, nan, nan) if with_ci=False
        """
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            logger.error("sklearn required for AUC-PR computation")
            return (0.0, np.nan, np.nan)

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true")
            return (0.0, np.nan, np.nan)

        try:
            auc_pr = average_precision_score(y_true, y_score)

            if with_ci:
                _, ci_lower, ci_upper = self.bootstrap_confidence_interval(
                    y_true, y_score,
                    lambda yt, ys: average_precision_score(yt, ys),
                    n_bootstrap=self.config.n_bootstrap
                )
                return (float(auc_pr), float(ci_lower), float(ci_upper))
            else:
                return (float(auc_pr), np.nan, np.nan)

        except Exception as e:
            logger.error(f"Error computing AUC-PR: {e}")
            return (0.0, np.nan, np.nan)

    def sensitivity_at_specificity(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_specificity: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Find sensitivity at target specificity threshold.

        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            target_specificity: Target specificity (default from config)

        Returns:
            (sensitivity, threshold)
        """
        try:
            from sklearn.metrics import roc_curve
        except ImportError:
            logger.error("sklearn required for ROC computation")
            return (0.0, 0.5)

        target_specificity = target_specificity or self.config.sensitivity_at_specificity

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(np.unique(y_true)) < 2:
            return (0.0, 0.5)

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            # Specificity = 1 - FPR
            specificity = 1 - fpr

            # Find threshold closest to target specificity
            idx = np.argmin(np.abs(specificity - target_specificity))

            return (float(tpr[idx]), float(thresholds[idx]) if idx < len(thresholds) else 0.5)

        except Exception as e:
            logger.error(f"Error computing sensitivity at specificity: {e}")
            return (0.0, 0.5)

    def specificity_at_sensitivity(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_sensitivity: float = 0.95
    ) -> Tuple[float, float]:
        """
        Find specificity at target sensitivity threshold.

        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            target_sensitivity: Target sensitivity (default 0.95)

        Returns:
            (specificity, threshold)
        """
        try:
            from sklearn.metrics import roc_curve
        except ImportError:
            logger.error("sklearn required for ROC computation")
            return (0.0, 0.5)

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(np.unique(y_true)) < 2:
            return (0.0, 0.5)

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)

            # Find threshold closest to target sensitivity (TPR)
            idx = np.argmin(np.abs(tpr - target_sensitivity))
            specificity = 1 - fpr[idx]

            return (float(specificity), float(thresholds[idx]) if idx < len(thresholds) else 0.5)

        except Exception as e:
            logger.error(f"Error computing specificity at sensitivity: {e}")
            return (0.0, 0.5)

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics for reliability assessment.

        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration (default from config)

        Returns:
            {
                'brier_score': float,
                'expected_calibration_error': float,
                'maximum_calibration_error': float,
                'reliability_diagram_data': dict
            }
        """
        n_bins = n_bins or self.config.calibration_bins

        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()

        # Brier Score
        brier_score = np.mean((y_prob - y_true) ** 2)

        # Calibration binning
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_count = np.sum(mask)
            else:
                bin_acc = 0.0
                bin_conf = (bin_edges[i] + bin_edges[i + 1]) / 2
                bin_count = 0

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)

        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)

        # Expected Calibration Error (ECE)
        total_samples = np.sum(bin_counts)
        if total_samples > 0:
            ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total_samples
        else:
            ece = 0.0

        # Maximum Calibration Error (MCE)
        valid_bins = bin_counts > 0
        if np.any(valid_bins):
            mce = np.max(np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins]))
        else:
            mce = 0.0

        return {
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'reliability_diagram_data': {
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        }

    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_fn: Callable,
        n_bootstrap: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence intervals for any metric.

        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            metric_fn: Function(y_true, y_score) -> float
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility

        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        n_bootstrap = n_bootstrap or self.config.n_bootstrap
        rng = np.random.RandomState(random_state)

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        n_samples = len(y_true)

        # Point estimate
        point_estimate = metric_fn(y_true, y_score)

        # Bootstrap
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            indices = rng.randint(0, n_samples, n_samples)
            try:
                estimate = metric_fn(y_true[indices], y_score[indices])
                bootstrap_estimates.append(estimate)
            except Exception:
                continue

        if len(bootstrap_estimates) < 10:
            logger.warning("Not enough successful bootstrap samples")
            return (float(point_estimate), np.nan, np.nan)

        bootstrap_estimates = np.array(bootstrap_estimates)
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

        return (float(point_estimate), float(ci_lower), float(ci_upper))

    def compute_auc_roc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        with_ci: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute standard AUC-ROC with confidence intervals.

        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            with_ci: Compute bootstrap confidence intervals

        Returns:
            (auc_roc, ci_lower, ci_upper)
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.error("sklearn required for AUC-ROC computation")
            return (0.0, np.nan, np.nan)

        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(np.unique(y_true)) < 2:
            return (0.0, np.nan, np.nan)

        try:
            auc_roc = roc_auc_score(y_true, y_score)

            if with_ci:
                _, ci_lower, ci_upper = self.bootstrap_confidence_interval(
                    y_true, y_score,
                    lambda yt, ys: roc_auc_score(yt, ys),
                    n_bootstrap=self.config.n_bootstrap
                )
                return (float(auc_roc), float(ci_lower), float(ci_upper))
            else:
                return (float(auc_roc), np.nan, np.nan)

        except Exception as e:
            logger.error(f"Error computing AUC-ROC: {e}")
            return (0.0, np.nan, np.nan)


class MetricsReporter:
    """Generate clinical-grade metrics reports."""

    def __init__(self, config: Optional[MedicalMetricsConfig] = None):
        self.metrics = MedicalMetrics(config)

    def generate_clinical_report(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive clinical metrics report.

        Includes:
        - AUC-ROC with 95% CI
        - Partial AUC (FPR 0-0.2)
        - AUC-PR with 95% CI
        - Sensitivity @ 95% specificity
        - Specificity @ 95% sensitivity
        - Brier score
        - ECE and MCE
        - Reliability diagram data
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        # AUC-ROC
        auc_roc, auc_roc_ci_lo, auc_roc_ci_hi = self.metrics.compute_auc_roc(y_true, y_score)

        # Partial AUC
        pauc = self.metrics.compute_partial_auc(y_true, y_score)

        # AUC-PR
        auc_pr, auc_pr_ci_lo, auc_pr_ci_hi = self.metrics.compute_auc_pr(y_true, y_score)

        # Operating points
        sens_at_95_spec, thresh_sens = self.metrics.sensitivity_at_specificity(y_true, y_score, 0.95)
        spec_at_95_sens, thresh_spec = self.metrics.specificity_at_sensitivity(y_true, y_score, 0.95)

        # Calibration
        calibration = self.metrics.compute_calibration_metrics(y_true, y_score)

        return {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(np.sum(1 - y_true)),
            'auc_roc': {
                'value': auc_roc,
                'ci_lower': auc_roc_ci_lo,
                'ci_upper': auc_roc_ci_hi
            },
            'partial_auc': {
                'value': pauc,
                'fpr_range': self.metrics.config.pauc_fpr_range
            },
            'auc_pr': {
                'value': auc_pr,
                'ci_lower': auc_pr_ci_lo,
                'ci_upper': auc_pr_ci_hi
            },
            'sensitivity_at_95_specificity': {
                'sensitivity': sens_at_95_spec,
                'threshold': thresh_sens
            },
            'specificity_at_95_sensitivity': {
                'specificity': spec_at_95_sens,
                'threshold': thresh_spec
            },
            'calibration': calibration
        }

    def compare_models(
        self,
        results: List[Dict[str, Any]],
        primary_metric: str = 'auc_roc'
    ) -> Dict[str, Any]:
        """
        Statistical comparison of multiple models.

        Args:
            results: List of report dicts from generate_clinical_report
            primary_metric: Metric to rank by ('auc_roc', 'partial_auc', 'auc_pr')

        Returns:
            Comparison summary with rankings
        """
        if len(results) < 2:
            return {'error': 'Need at least 2 models to compare'}

        # Extract primary metric values
        model_scores = []
        for r in results:
            if primary_metric == 'auc_roc':
                score = r.get('auc_roc', {}).get('value', 0)
            elif primary_metric == 'partial_auc':
                score = r.get('partial_auc', {}).get('value', 0)
            elif primary_metric == 'auc_pr':
                score = r.get('auc_pr', {}).get('value', 0)
            else:
                score = 0

            model_scores.append({
                'model_name': r.get('model_name', 'unknown'),
                'score': score
            })

        # Rank by score
        model_scores.sort(key=lambda x: x['score'], reverse=True)

        return {
            'primary_metric': primary_metric,
            'rankings': model_scores,
            'best_model': model_scores[0]['model_name'] if model_scores else None,
            'score_range': {
                'max': model_scores[0]['score'] if model_scores else 0,
                'min': model_scores[-1]['score'] if model_scores else 0
            }
        }


# Convenience functions
def compute_partial_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.2
) -> float:
    """Compute normalized partial AUC in FPR range [0, max_fpr]."""
    metrics = MedicalMetrics()
    return metrics.compute_partial_auc(y_true, y_score, fpr_range=(0.0, max_fpr))


def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-PR (Average Precision)."""
    metrics = MedicalMetrics()
    auc_pr, _, _ = metrics.compute_auc_pr(y_true, y_score, with_ci=False)
    return auc_pr


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score for calibration."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    metrics = MedicalMetrics()
    calibration = metrics.compute_calibration_metrics(y_true, y_prob, n_bins)
    return calibration['expected_calibration_error']
