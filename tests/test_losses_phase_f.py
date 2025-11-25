"""
Tests for Phase F loss functions (Lovasz, Boundary, CompoundLoss).

Author: ARC Team (Dev 2)
Created: 2025-11-24
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from tools.loss_functions import (
    LovaszSoftmax,
    LovaszHinge,
    BoundaryLoss,
    CompoundLoss,
    LossFactory,
    lovasz_grad,
    DiceLoss,
    FocalLoss,
)


class TestLovaszGrad:
    """Test Lovasz gradient computation."""

    def test_lovasz_grad_basic(self):
        """Test basic Lovasz gradient."""
        gt = torch.tensor([1, 1, 0, 0])
        grad = lovasz_grad(gt)

        assert grad.shape == gt.shape
        assert grad.sum().item() > 0

    def test_lovasz_grad_all_ones(self):
        """Test Lovasz gradient with all positive."""
        gt = torch.tensor([1, 1, 1, 1])
        grad = lovasz_grad(gt)

        assert grad.shape == gt.shape

    def test_lovasz_grad_all_zeros(self):
        """Test Lovasz gradient with all negative."""
        gt = torch.tensor([0, 0, 0, 0])
        grad = lovasz_grad(gt)

        assert grad.shape == gt.shape


class TestLovaszSoftmax:
    """Test LovaszSoftmax loss."""

    @pytest.fixture
    def sample_input(self):
        """Create sample segmentation input."""
        batch_size = 2
        num_classes = 3
        H, W = 32, 32

        logits = torch.randn(batch_size, num_classes, H, W)
        labels = torch.randint(0, num_classes, (batch_size, H, W))

        return logits, labels

    def test_forward_basic(self, sample_input):
        """Test basic forward pass."""
        logits, labels = sample_input

        loss_fn = LovaszSoftmax()
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_forward_per_image(self, sample_input):
        """Test per-image mode."""
        logits, labels = sample_input

        loss_fn = LovaszSoftmax(per_image=True)
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_classes_all(self, sample_input):
        """Test with classes='all'."""
        logits, labels = sample_input

        loss_fn = LovaszSoftmax(classes='all')
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0

    def test_gradient_flow(self, sample_input):
        """Test gradients flow correctly."""
        logits, labels = sample_input
        logits.requires_grad = True

        loss_fn = LovaszSoftmax()
        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestLovaszHinge:
    """Test LovaszHinge loss."""

    @pytest.fixture
    def sample_input(self):
        """Create sample binary segmentation input."""
        batch_size = 2
        H, W = 32, 32

        logits = torch.randn(batch_size, 1, H, W)
        labels = torch.randint(0, 2, (batch_size, H, W)).float()

        return logits, labels

    def test_forward_basic(self, sample_input):
        """Test basic forward pass."""
        logits, labels = sample_input

        loss_fn = LovaszHinge()
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_3d_logits(self, sample_input):
        """Test with 3D logits (no channel dim)."""
        logits, labels = sample_input
        logits_3d = logits.squeeze(1)

        loss_fn = LovaszHinge()
        loss = loss_fn(logits_3d, labels)

        assert loss.dim() == 0

    def test_per_image_false(self, sample_input):
        """Test without per-image averaging."""
        logits, labels = sample_input

        loss_fn = LovaszHinge(per_image=False)
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0


class TestBoundaryLoss:
    """Test BoundaryLoss."""

    @pytest.fixture
    def sample_input(self):
        """Create sample segmentation input."""
        batch_size = 2
        num_classes = 3
        H, W = 32, 32

        logits = torch.randn(batch_size, num_classes, H, W)
        labels = torch.randint(0, num_classes, (batch_size, H, W))

        return logits, labels

    def test_forward_without_dist_maps(self, sample_input):
        """Test forward pass without distance maps (fallback mode)."""
        logits, labels = sample_input

        loss_fn = BoundaryLoss()
        loss = loss_fn(logits, labels)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_with_dist_maps(self, sample_input):
        """Test forward pass with precomputed distance maps."""
        logits, labels = sample_input
        batch_size, num_classes, H, W = logits.shape

        # Create mock distance maps
        dist_maps = torch.randn(batch_size, num_classes, H, W)

        loss_fn = BoundaryLoss()
        loss = loss_fn(logits, labels, dist_maps=dist_maps)

        assert loss.dim() == 0

    def test_compute_distance_map(self):
        """Test distance map computation."""
        label = np.zeros((32, 32), dtype=np.int64)
        label[8:24, 8:24] = 1  # Center square

        dist_map = BoundaryLoss.compute_distance_map(label, num_classes=2)

        assert dist_map.shape == (2, 32, 32)


class TestCompoundLoss:
    """Test CompoundLoss."""

    @pytest.fixture
    def sample_classification_input(self):
        """Create sample classification input."""
        batch_size = 4
        logits = torch.randn(batch_size, 1)
        labels = torch.randint(0, 2, (batch_size, 1)).float()
        return logits, labels

    def test_forward_fixed_weights(self, sample_classification_input):
        """Test forward with fixed weights."""
        logits, labels = sample_classification_input

        losses = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]
        compound = CompoundLoss(
            losses=losses,
            weights=[0.5, 0.5],
            learnable_weights=False
        )

        result = compound(logits, labels)

        assert 'total' in result
        assert 'weights' in result
        assert result['total'].dim() == 0

    def test_forward_learnable_weights(self, sample_classification_input):
        """Test forward with learnable weights."""
        logits, labels = sample_classification_input

        losses = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]
        compound = CompoundLoss(
            losses=losses,
            learnable_weights=True
        )

        result = compound(logits, labels)

        assert 'total' in result
        # Check weights are parameters
        assert compound.log_weights.requires_grad

    def test_get_weights(self, sample_classification_input):
        """Test get_weights method."""
        losses = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]
        compound = CompoundLoss(
            losses=losses,
            weights=[1.0, 2.0],
            learnable_weights=False,
            normalize_weights=True
        )

        weights = compound.get_weights()

        # Should be softmax normalized
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradient_to_weights(self, sample_classification_input):
        """Test gradients flow to learnable weights."""
        logits, labels = sample_classification_input

        losses = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]
        compound = CompoundLoss(
            losses=losses,
            learnable_weights=True
        )

        result = compound(logits, labels)
        result['total'].backward()

        assert compound.log_weights.grad is not None


class TestLossFactory:
    """Test LossFactory."""

    def test_create_simple_loss(self):
        """Test creating a simple loss."""
        config = {'type': 'dice'}
        loss = LossFactory.create(config)

        assert isinstance(loss, DiceLoss)

    def test_create_focal_loss(self):
        """Test creating focal loss with params."""
        config = {'type': 'focal', 'gamma': 3.0}
        loss = LossFactory.create(config)

        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 3.0

    def test_create_compound_loss(self):
        """Test creating compound loss."""
        config = {
            'type': 'compound',
            'components': [
                {'type': 'dice', 'weight': 0.5},
                {'type': 'focal', 'gamma': 2.0, 'weight': 0.5}
            ],
            'learnable_weights': True
        }
        loss = LossFactory.create(config)

        assert isinstance(loss, CompoundLoss)
        assert len(loss.losses) == 2

    def test_create_unknown_loss(self):
        """Test error for unknown loss type."""
        config = {'type': 'unknown_loss'}

        with pytest.raises(ValueError, match="Unknown loss type"):
            LossFactory.create(config)

    def test_create_lovasz_softmax(self):
        """Test creating Lovasz-Softmax loss."""
        config = {'type': 'lovasz_softmax', 'classes': 'present'}
        loss = LossFactory.create(config)

        assert isinstance(loss, LovaszSoftmax)


class TestLossConfigIntegration:
    """Test integration with loss_config schema."""

    def _load_loss_config_module(self):
        """Load loss_config module directly."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "loss_config",
            "/Users/bengibson/arc-autonomous-research/schemas/loss_config.py"
        )
        loss_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loss_module)
        return loss_module

    def test_example_lovasz_softmax(self):
        """Test creating loss from schema example."""
        loss_module = self._load_loss_config_module()
        LossConfig = loss_module.LossConfig

        config = LossConfig.example_lovasz_softmax()

        assert config.primary_loss.value == 'lovasz_softmax'
        assert config.hyperparameters.lovasz_classes == 'present'

    def test_example_compound_loss(self):
        """Test compound loss schema example."""
        loss_module = self._load_loss_config_module()
        LossConfig = loss_module.LossConfig
        LossType = loss_module.LossType

        config = LossConfig.example_compound_dice_focal_lovasz()

        assert config.primary_loss == LossType.COMPOUND
        assert len(config.compound_components) == 3

    def test_validate_compound_requires_components(self):
        """Test that compound loss requires components."""
        loss_module = self._load_loss_config_module()
        LossConfig = loss_module.LossConfig
        LossType = loss_module.LossType

        with pytest.raises(ValueError):
            LossConfig(
                name="invalid_compound",
                primary_loss=LossType.COMPOUND,
                primary_weight=1.0,
                compound_components=None  # Should fail
            )
