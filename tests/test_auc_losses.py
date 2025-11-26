"""
Tests for AUC Loss Functions.

Phase G - Tests for AUCMarginLoss, PartialAUCLoss, CompositeMedicalLoss.
"""

import pytest
import torch
import torch.nn as nn

from tools.loss_functions import (
    AUCMarginLoss,
    PartialAUCLoss,
    CompositeMedicalLoss,
    LossFactory,
)


@pytest.fixture
def sample_batch():
    """Generate sample batch for testing."""
    torch.manual_seed(42)
    batch_size = 100

    # Create mixed positive/negative batch
    targets = torch.zeros(batch_size)
    targets[:30] = 1  # 30% positive

    # Create predictions (logits)
    predictions = torch.randn(batch_size)
    # Make positive samples have slightly higher scores
    predictions[:30] += 0.5

    return predictions, targets


@pytest.fixture
def balanced_batch():
    """Balanced batch with 50/50 split."""
    torch.manual_seed(42)
    batch_size = 100

    targets = torch.zeros(batch_size)
    targets[:50] = 1

    predictions = torch.randn(batch_size)
    predictions[:50] += 1.0  # Good separation

    return predictions, targets


class TestAUCMarginLoss:
    """Tests for AUCMarginLoss."""

    def test_forward_returns_scalar(self, sample_batch):
        """Loss should return a scalar tensor."""
        predictions, targets = sample_batch
        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_loss_positive_value(self, sample_batch):
        """Loss should be non-negative."""
        predictions, targets = sample_batch
        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() >= 0

    def test_loss_decreases_with_separation(self):
        """Better separation should give lower loss."""
        torch.manual_seed(42)
        targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float)

        # Poor separation
        pred_poor = torch.tensor([0.5, 0.4, 0.6, 0.3, 0.5, 0.5, 0.6, 0.4, 0.7, 0.5])
        # Good separation
        pred_good = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1, 0.8, 0.9, 0.85, 0.95, 0.9])

        loss_fn = AUCMarginLoss(margin=0.5, num_samples=100)
        loss_poor = loss_fn(pred_poor, targets)
        loss_good = loss_fn(pred_good, targets)

        assert loss_good.item() < loss_poor.item()

    def test_gradient_flow(self, sample_batch):
        """Loss should allow gradient computation."""
        predictions, targets = sample_batch
        predictions.requires_grad = True

        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_no_positive_samples(self):
        """Should handle case with no positive samples."""
        predictions = torch.randn(10)
        targets = torch.zeros(10)

        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_no_negative_samples(self):
        """Should handle case with no negative samples."""
        predictions = torch.randn(10)
        targets = torch.ones(10)

        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_margin_parameter(self, sample_batch):
        """Different margins should produce different losses."""
        predictions, targets = sample_batch

        loss_m1 = AUCMarginLoss(margin=0.5)
        loss_m2 = AUCMarginLoss(margin=2.0)

        l1 = loss_m1(predictions, targets)
        l2 = loss_m2(predictions, targets)

        # Larger margin should typically give larger loss
        assert l1.item() != l2.item()

    def test_num_samples_parameter(self, balanced_batch):
        """Different num_samples should work."""
        predictions, targets = balanced_batch

        loss_100 = AUCMarginLoss(num_samples=100)
        loss_1000 = AUCMarginLoss(num_samples=1000)

        l1 = loss_100(predictions, targets)
        l2 = loss_1000(predictions, targets)

        # Both should be valid
        assert l1.item() >= 0
        assert l2.item() >= 0


class TestPartialAUCLoss:
    """Tests for PartialAUCLoss."""

    def test_forward_returns_scalar(self, sample_batch):
        """Loss should return a scalar tensor."""
        predictions, targets = sample_batch
        loss_fn = PartialAUCLoss()
        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_loss_positive_value(self, sample_batch):
        """Loss should be non-negative."""
        predictions, targets = sample_batch
        loss_fn = PartialAUCLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() >= 0

    def test_gradient_flow(self, sample_batch):
        """Loss should allow gradient computation."""
        predictions, targets = sample_batch
        predictions.requires_grad = True

        loss_fn = PartialAUCLoss()
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_fpr_range_parameter(self, sample_batch):
        """Different FPR ranges should work."""
        predictions, targets = sample_batch

        loss_narrow = PartialAUCLoss(fpr_range=(0.0, 0.1))
        loss_wide = PartialAUCLoss(fpr_range=(0.0, 0.5))

        l1 = loss_narrow(predictions, targets)
        l2 = loss_wide(predictions, targets)

        assert l1.item() >= 0
        assert l2.item() >= 0

    def test_no_positive_samples(self):
        """Should handle no positive samples."""
        predictions = torch.randn(10)
        targets = torch.zeros(10)

        loss_fn = PartialAUCLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_regularization(self, sample_batch):
        """Lambda regularization should affect loss."""
        predictions, targets = sample_batch

        loss_no_reg = PartialAUCLoss(lambda_reg=0.0)
        loss_with_reg = PartialAUCLoss(lambda_reg=1.0)

        l1 = loss_no_reg(predictions, targets)
        l2 = loss_with_reg(predictions, targets)

        # With regularization should have different value
        assert abs(l1.item() - l2.item()) > 0.001


class TestCompositeMedicalLoss:
    """Tests for CompositeMedicalLoss."""

    def test_forward_returns_dict(self, sample_batch):
        """Loss should return dict with components."""
        predictions, targets = sample_batch
        loss_fn = CompositeMedicalLoss()
        result = loss_fn(predictions, targets)

        assert isinstance(result, dict)
        assert 'total' in result
        assert 'bce' in result
        assert 'focal' in result
        assert 'auc' in result

    def test_total_is_weighted_sum(self, sample_batch):
        """Total should be weighted sum of components."""
        predictions, targets = sample_batch
        loss_fn = CompositeMedicalLoss(
            bce_weight=0.3,
            focal_weight=0.3,
            auc_weight=0.4
        )
        result = loss_fn(predictions, targets)

        expected_total = (
            0.3 * result['bce'] +
            0.3 * result['focal'] +
            0.4 * result['auc']
        )

        assert torch.allclose(result['total'], expected_total, atol=1e-6)

    def test_gradient_flow(self, sample_batch):
        """Loss should allow gradient computation."""
        predictions, targets = sample_batch
        predictions.requires_grad = True

        loss_fn = CompositeMedicalLoss()
        result = loss_fn(predictions, targets)
        result['total'].backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_custom_weights(self, sample_batch):
        """Custom weights should be respected."""
        predictions, targets = sample_batch

        # Heavy BCE weight
        loss_bce_heavy = CompositeMedicalLoss(bce_weight=0.8, focal_weight=0.1, auc_weight=0.1)
        # Heavy AUC weight
        loss_auc_heavy = CompositeMedicalLoss(bce_weight=0.1, focal_weight=0.1, auc_weight=0.8)

        r1 = loss_bce_heavy(predictions, targets)
        r2 = loss_auc_heavy(predictions, targets)

        # Results should differ due to weighting
        assert r1['total'].item() != r2['total'].item()

    def test_components_positive(self, sample_batch):
        """All loss components should be non-negative."""
        predictions, targets = sample_batch
        loss_fn = CompositeMedicalLoss()
        result = loss_fn(predictions, targets)

        assert result['bce'].item() >= 0
        assert result['focal'].item() >= 0
        assert result['auc'].item() >= 0
        assert result['total'].item() >= 0


class TestLossFactoryExtensions:
    """Tests for LossFactory with new losses."""

    def test_factory_creates_auc_margin(self):
        """Factory should create AUCMarginLoss."""
        config = {'type': 'auc_margin', 'margin': 1.5}
        loss = LossFactory.create(config)

        assert isinstance(loss, AUCMarginLoss)
        assert loss.margin == 1.5

    def test_factory_creates_partial_auc(self):
        """Factory should create PartialAUCLoss."""
        config = {'type': 'partial_auc', 'fpr_range': (0.0, 0.1)}
        loss = LossFactory.create(config)

        assert isinstance(loss, PartialAUCLoss)
        assert loss.fpr_max == 0.1

    def test_factory_creates_composite_medical(self):
        """Factory should create CompositeMedicalLoss."""
        config = {
            'type': 'composite_medical',
            'bce_weight': 0.4,
            'focal_weight': 0.3,
            'auc_weight': 0.3
        }
        loss = LossFactory.create(config)

        assert isinstance(loss, CompositeMedicalLoss)
        assert loss.weights['bce'] == 0.4

    def test_factory_compound_with_auc(self, sample_batch):
        """Factory should handle compound loss with AUC."""
        config = {
            'type': 'compound',
            'components': [
                {'type': 'bce', 'weight': 0.5},
                {'type': 'auc_margin', 'margin': 1.0, 'weight': 0.5}
            ],
            'learnable_weights': False
        }
        loss = LossFactory.create(config)

        predictions, targets = sample_batch
        # CompoundLoss returns dict, so check it works
        result = loss(predictions, targets.float())
        assert 'total' in result


class TestDeviceCompatibility:
    """Tests for device compatibility (CPU)."""

    def test_auc_margin_cpu(self, sample_batch):
        """AUCMarginLoss should work on CPU."""
        predictions, targets = sample_batch
        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)
        assert loss.device.type == 'cpu'

    def test_partial_auc_cpu(self, sample_batch):
        """PartialAUCLoss should work on CPU."""
        predictions, targets = sample_batch
        loss_fn = PartialAUCLoss()
        loss = loss_fn(predictions, targets)
        assert loss.device.type == 'cpu'

    def test_composite_cpu(self, sample_batch):
        """CompositeMedicalLoss should work on CPU."""
        predictions, targets = sample_batch
        loss_fn = CompositeMedicalLoss()
        result = loss_fn(predictions, targets)
        assert result['total'].device.type == 'cpu'


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Should handle single sample batch."""
        predictions = torch.tensor([0.5])
        targets = torch.tensor([1.0])

        # These will return 0 since can't form pairs
        auc_loss = AUCMarginLoss()(predictions, targets)
        assert auc_loss.item() == 0.0

    def test_very_small_batch(self):
        """Should handle very small batch."""
        predictions = torch.tensor([0.1, 0.9])
        targets = torch.tensor([0.0, 1.0])

        auc_loss = AUCMarginLoss(num_samples=10)(predictions, targets)
        assert auc_loss.item() >= 0

    def test_all_same_predictions(self, sample_batch):
        """Should handle case where all predictions are same."""
        _, targets = sample_batch
        predictions = torch.ones(100) * 0.5

        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)
        # Should produce some loss since margin violation
        assert loss.item() >= 0

    def test_extreme_predictions(self, sample_batch):
        """Should handle extreme prediction values."""
        _, targets = sample_batch
        predictions = torch.randn(100) * 100  # Large values

        loss_fn = AUCMarginLoss()
        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
