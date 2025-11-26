"""
Tests for Medical Metrics module.

Phase G - Tests for pAUC, AUC-PR, calibration metrics, and clinical reporting.
"""

import pytest
import numpy as np

from tools.medical_metrics import (
    MedicalMetrics,
    MedicalMetricsConfig,
    MetricsReporter,
    compute_partial_auc,
    compute_auc_pr,
    compute_brier_score,
    compute_ece,
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Generate scores that correlate with labels
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive
    y_score = np.random.beta(2, 5, n_samples)  # Base scores
    # Make positive samples have higher scores on average
    y_score[y_true == 1] += 0.3
    y_score = np.clip(y_score, 0, 1)

    return y_true, y_score


@pytest.fixture
def perfect_predictions():
    """Predictions with perfect separation."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    return y_true, y_score


@pytest.fixture
def metrics():
    """Create MedicalMetrics instance."""
    return MedicalMetrics()


class TestPartialAUC:
    """Tests for partial AUC computation."""

    def test_pauc_perfect_classifier(self, metrics, perfect_predictions):
        """Perfect classifier should have high pAUC."""
        y_true, y_score = perfect_predictions
        pauc = metrics.compute_partial_auc(y_true, y_score)
        # Perfect separation should give high pAUC
        assert pauc > 0.9

    def test_pauc_random_classifier(self, metrics):
        """Random classifier should have pAUC around 0.5."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_score = np.random.uniform(0, 1, 1000)
        pauc = metrics.compute_partial_auc(y_true, y_score)
        # Random should be around 0.5 (with tolerance)
        assert 0.3 < pauc < 0.7

    def test_pauc_normalized(self, metrics, sample_predictions):
        """Normalized pAUC should be in [0, 1]."""
        y_true, y_score = sample_predictions
        pauc = metrics.compute_partial_auc(y_true, y_score, normalize=True)
        assert 0 <= pauc <= 1

    def test_pauc_custom_range(self, metrics, sample_predictions):
        """Custom FPR range should work."""
        y_true, y_score = sample_predictions
        pauc_narrow = metrics.compute_partial_auc(y_true, y_score, fpr_range=(0.0, 0.1))
        pauc_wide = metrics.compute_partial_auc(y_true, y_score, fpr_range=(0.0, 0.5))
        # Both should be valid values
        assert 0 <= pauc_narrow <= 1
        assert 0 <= pauc_wide <= 1

    def test_pauc_single_class(self, metrics):
        """Single class should return 0."""
        y_true = np.zeros(100)
        y_score = np.random.uniform(0, 1, 100)
        pauc = metrics.compute_partial_auc(y_true, y_score)
        assert pauc == 0.0


class TestAUCPR:
    """Tests for AUC-PR computation."""

    def test_auc_pr_returns_tuple(self, metrics, sample_predictions):
        """AUC-PR should return (value, ci_lower, ci_upper)."""
        y_true, y_score = sample_predictions
        result = metrics.compute_auc_pr(y_true, y_score)
        assert len(result) == 3
        assert isinstance(result[0], float)

    def test_auc_pr_value_range(self, metrics, sample_predictions):
        """AUC-PR should be in [0, 1]."""
        y_true, y_score = sample_predictions
        auc_pr, _, _ = metrics.compute_auc_pr(y_true, y_score, with_ci=False)
        assert 0 <= auc_pr <= 1

    def test_auc_pr_perfect_classifier(self, metrics, perfect_predictions):
        """Perfect classifier should have high AUC-PR."""
        y_true, y_score = perfect_predictions
        auc_pr, _, _ = metrics.compute_auc_pr(y_true, y_score, with_ci=False)
        assert auc_pr > 0.9

    def test_auc_pr_with_ci(self, metrics, sample_predictions):
        """CI should bracket the point estimate."""
        y_true, y_score = sample_predictions
        auc_pr, ci_low, ci_high = metrics.compute_auc_pr(y_true, y_score, with_ci=True)
        assert ci_low <= auc_pr <= ci_high


class TestSensitivitySpecificity:
    """Tests for sensitivity/specificity at thresholds."""

    def test_sensitivity_at_specificity_returns_tuple(self, metrics, sample_predictions):
        """Should return (sensitivity, threshold)."""
        y_true, y_score = sample_predictions
        result = metrics.sensitivity_at_specificity(y_true, y_score, 0.95)
        assert len(result) == 2
        sens, thresh = result
        assert 0 <= sens <= 1
        assert 0 <= thresh <= 1

    def test_specificity_at_sensitivity_returns_tuple(self, metrics, sample_predictions):
        """Should return (specificity, threshold)."""
        y_true, y_score = sample_predictions
        result = metrics.specificity_at_sensitivity(y_true, y_score, 0.95)
        assert len(result) == 2
        spec, thresh = result
        assert 0 <= spec <= 1
        assert 0 <= thresh <= 1

    def test_high_specificity_lower_sensitivity(self, metrics, sample_predictions):
        """Higher specificity requirement should give lower sensitivity."""
        y_true, y_score = sample_predictions
        sens_95, _ = metrics.sensitivity_at_specificity(y_true, y_score, 0.95)
        sens_80, _ = metrics.sensitivity_at_specificity(y_true, y_score, 0.80)
        # Higher specificity = lower sensitivity (tradeoff)
        assert sens_95 <= sens_80


class TestCalibration:
    """Tests for calibration metrics."""

    def test_brier_score_range(self, metrics, sample_predictions):
        """Brier score should be in [0, 1]."""
        y_true, y_score = sample_predictions
        result = metrics.compute_calibration_metrics(y_true, y_score)
        assert 0 <= result['brier_score'] <= 1

    def test_ece_range(self, metrics, sample_predictions):
        """ECE should be in [0, 1]."""
        y_true, y_score = sample_predictions
        result = metrics.compute_calibration_metrics(y_true, y_score)
        assert 0 <= result['expected_calibration_error'] <= 1

    def test_mce_range(self, metrics, sample_predictions):
        """MCE should be in [0, 1]."""
        y_true, y_score = sample_predictions
        result = metrics.compute_calibration_metrics(y_true, y_score)
        assert 0 <= result['maximum_calibration_error'] <= 1

    def test_reliability_diagram_data(self, metrics, sample_predictions):
        """Should return reliability diagram data."""
        y_true, y_score = sample_predictions
        result = metrics.compute_calibration_metrics(y_true, y_score, n_bins=10)
        rd_data = result['reliability_diagram_data']

        assert 'bin_accuracies' in rd_data
        assert 'bin_confidences' in rd_data
        assert 'bin_counts' in rd_data
        assert len(rd_data['bin_accuracies']) == 10

    def test_perfectly_calibrated(self, metrics):
        """Perfectly calibrated predictions should have low ECE."""
        # Create perfectly calibrated predictions
        np.random.seed(42)
        y_prob = np.random.uniform(0, 1, 1000)
        y_true = (np.random.uniform(0, 1, 1000) < y_prob).astype(int)

        result = metrics.compute_calibration_metrics(y_true, y_prob)
        # Should have low ECE
        assert result['expected_calibration_error'] < 0.15


class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_returns_triple(self, metrics, sample_predictions):
        """Bootstrap should return (estimate, ci_low, ci_high)."""
        y_true, y_score = sample_predictions

        def simple_metric(yt, ys):
            return np.mean(ys)

        result = metrics.bootstrap_confidence_interval(
            y_true, y_score, simple_metric, n_bootstrap=100
        )
        assert len(result) == 3

    def test_bootstrap_ci_brackets_estimate(self, metrics, sample_predictions):
        """CI should bracket the point estimate."""
        y_true, y_score = sample_predictions

        def auc_metric(yt, ys):
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(yt, ys)

        estimate, ci_low, ci_high = metrics.bootstrap_confidence_interval(
            y_true, y_score, auc_metric, n_bootstrap=100
        )
        assert ci_low <= estimate <= ci_high


class TestMetricsReporter:
    """Tests for MetricsReporter class."""

    def test_clinical_report_structure(self, sample_predictions):
        """Clinical report should have expected structure."""
        y_true, y_score = sample_predictions
        reporter = MetricsReporter()
        report = reporter.generate_clinical_report(y_true, y_score, "test_model")

        assert 'model_name' in report
        assert 'auc_roc' in report
        assert 'partial_auc' in report
        assert 'auc_pr' in report
        assert 'sensitivity_at_95_specificity' in report
        assert 'specificity_at_95_sensitivity' in report
        assert 'calibration' in report

    def test_clinical_report_values(self, sample_predictions):
        """Clinical report values should be valid."""
        y_true, y_score = sample_predictions
        reporter = MetricsReporter()
        report = reporter.generate_clinical_report(y_true, y_score, "test_model")

        assert report['model_name'] == "test_model"
        assert 0 <= report['auc_roc']['value'] <= 1
        assert 0 <= report['partial_auc']['value'] <= 1
        assert 0 <= report['auc_pr']['value'] <= 1

    def test_compare_models(self, sample_predictions):
        """Model comparison should rank by metric."""
        y_true, y_score = sample_predictions
        reporter = MetricsReporter()

        # Create two "models" with different scores
        report1 = reporter.generate_clinical_report(y_true, y_score, "model_a")
        y_score_worse = y_score * 0.5 + 0.25  # Degrade scores
        report2 = reporter.generate_clinical_report(y_true, y_score_worse, "model_b")

        comparison = reporter.compare_models([report1, report2])

        assert 'rankings' in comparison
        assert len(comparison['rankings']) == 2
        assert comparison['best_model'] is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_partial_auc_func(self, sample_predictions):
        """compute_partial_auc should work."""
        y_true, y_score = sample_predictions
        pauc = compute_partial_auc(y_true, y_score)
        assert 0 <= pauc <= 1

    def test_compute_auc_pr_func(self, sample_predictions):
        """compute_auc_pr should work."""
        y_true, y_score = sample_predictions
        auc_pr = compute_auc_pr(y_true, y_score)
        assert 0 <= auc_pr <= 1

    def test_compute_brier_score_func(self, sample_predictions):
        """compute_brier_score should work."""
        y_true, y_score = sample_predictions
        brier = compute_brier_score(y_true, y_score)
        assert 0 <= brier <= 1

    def test_compute_ece_func(self, sample_predictions):
        """compute_ece should work."""
        y_true, y_score = sample_predictions
        ece = compute_ece(y_true, y_score)
        assert 0 <= ece <= 1
