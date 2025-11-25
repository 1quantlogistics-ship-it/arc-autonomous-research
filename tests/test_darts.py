"""
Tests for DARTS (Differentiable Architecture Search).

Author: ARC Team (Dev 2)
Created: 2025-11-24
"""

import pytest
import torch
import torch.nn as nn

from tools.darts import (
    DARTSConfig,
    DARTSNetwork,
    DARTSCell,
    MixedOp,
    DARTSOperations,
    DARTSTrainer,
    create_darts_network,
    genotype_to_dict,
    DARTS_PRIMITIVES,
    MEDICAL_SAFE_PRIMITIVES,
    Zero,
    Identity,
    SepConv,
    DilConv,
    FactorizedReduce,
)


class TestDARTSOperations:
    """Test individual DARTS operations."""

    def test_zero_operation(self):
        """Test zero (edge removal) operation."""
        x = torch.randn(2, 16, 32, 32)

        zero = Zero(stride=1)
        out = zero(x)

        assert out.shape == x.shape
        assert torch.all(out == 0)

    def test_zero_with_stride(self):
        """Test zero operation with stride."""
        x = torch.randn(2, 16, 32, 32)

        zero = Zero(stride=2)
        out = zero(x)

        assert out.shape == (2, 16, 16, 16)
        assert torch.all(out == 0)

    def test_identity(self):
        """Test identity operation."""
        x = torch.randn(2, 16, 32, 32)

        identity = Identity()
        out = identity(x)

        assert torch.equal(out, x)

    def test_sep_conv(self):
        """Test separable convolution."""
        x = torch.randn(2, 16, 32, 32)

        sep_conv = SepConv(16, 16, 3, stride=1, padding=1)
        out = sep_conv(x)

        assert out.shape == x.shape

    def test_dil_conv(self):
        """Test dilated convolution."""
        x = torch.randn(2, 16, 32, 32)

        dil_conv = DilConv(16, 16, 3, stride=1, padding=2, dilation=2)
        out = dil_conv(x)

        assert out.shape == x.shape

    def test_factorized_reduce(self):
        """Test factorized reduce."""
        x = torch.randn(2, 16, 32, 32)

        reduce = FactorizedReduce(16, 16)
        out = reduce(x)

        assert out.shape == (2, 16, 16, 16)


class TestMixedOp:
    """Test mixed operation."""

    def test_forward(self):
        """Test mixed operation forward."""
        x = torch.randn(2, 16, 32, 32)

        mixed_op = MixedOp(16, stride=1, primitives=MEDICAL_SAFE_PRIMITIVES)
        weights = torch.softmax(torch.randn(len(MEDICAL_SAFE_PRIMITIVES)), dim=0)
        out = mixed_op(x, weights)

        assert out.shape == x.shape

    def test_forward_with_reduction(self):
        """Test mixed operation with stride=2."""
        x = torch.randn(2, 16, 32, 32)

        mixed_op = MixedOp(16, stride=2, primitives=MEDICAL_SAFE_PRIMITIVES)
        weights = torch.softmax(torch.randn(len(MEDICAL_SAFE_PRIMITIVES)), dim=0)
        out = mixed_op(x, weights)

        assert out.shape == (2, 16, 16, 16)


class TestDARTSCell:
    """Test DARTS cell."""

    def test_cell_forward(self):
        """Test cell forward pass."""
        C = 16
        cell = DARTSCell(
            C_prev_prev=C, C_prev=C, C=C,
            reduction=False, reduction_prev=False,
            primitives=MEDICAL_SAFE_PRIMITIVES
        )

        s0 = torch.randn(2, C, 32, 32)
        s1 = torch.randn(2, C, 32, 32)
        weights = torch.softmax(torch.randn(cell.num_edges, len(MEDICAL_SAFE_PRIMITIVES)), dim=-1)

        out = cell(s0, s1, weights)

        # Output channels = C * num_nodes (concatenated)
        assert out.shape == (2, C * cell.num_nodes, 32, 32)

    def test_reduction_cell(self):
        """Test reduction cell."""
        C = 16
        cell = DARTSCell(
            C_prev_prev=C, C_prev=C, C=C,
            reduction=True, reduction_prev=False,
            primitives=MEDICAL_SAFE_PRIMITIVES
        )

        s0 = torch.randn(2, C, 32, 32)
        s1 = torch.randn(2, C, 32, 32)
        weights = torch.softmax(torch.randn(cell.num_edges, len(MEDICAL_SAFE_PRIMITIVES)), dim=-1)

        out = cell(s0, s1, weights)

        # Spatial dimensions reduced by 2
        assert out.shape[2] == 16
        assert out.shape[3] == 16


class TestDARTSConfig:
    """Test DARTS configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DARTSConfig()

        assert config.init_channels == 16
        assert config.num_cells == 8
        assert config.max_params == 10_000_000
        assert config.primitives == MEDICAL_SAFE_PRIMITIVES

    def test_custom_config(self):
        """Test custom configuration."""
        config = DARTSConfig(
            init_channels=32,
            num_cells=4,
            primitives=DARTS_PRIMITIVES
        )

        assert config.init_channels == 32
        assert config.num_cells == 4
        assert config.primitives == DARTS_PRIMITIVES


class TestDARTSNetwork:
    """Test DARTS network."""

    @pytest.fixture
    def small_network(self):
        """Create small network for testing."""
        config = DARTSConfig(
            init_channels=8,
            num_cells=4,
            num_nodes=2
        )
        return DARTSNetwork(config, num_classes=2)

    def test_forward_pass(self, small_network):
        """Test forward pass."""
        x = torch.randn(2, 3, 64, 64)
        out = small_network(x)

        assert out.shape == (2, 2)

    def test_arch_parameters(self, small_network):
        """Test architecture parameters are separate."""
        arch_params = small_network.arch_parameters()

        assert len(arch_params) == 2  # normal + reduce
        assert all(p.requires_grad for p in arch_params)

    def test_network_parameters(self, small_network):
        """Test network parameters exclude arch params."""
        arch_params = set(id(p) for p in small_network.arch_parameters())
        network_params = small_network.network_parameters()

        for p in network_params:
            assert id(p) not in arch_params

    def test_get_genotype(self, small_network):
        """Test genotype extraction."""
        genotype = small_network.get_genotype()

        assert 'normal' in genotype
        assert 'reduce' in genotype
        assert len(genotype['normal']) > 0
        assert len(genotype['reduce']) > 0

    def test_gradient_flow(self, small_network):
        """Test gradients flow to both arch and network params."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 2, (2,))

        loss = nn.functional.cross_entropy(small_network(x), y)
        loss.backward()

        # Check arch params have gradients
        for p in small_network.arch_parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)

        # Check some network params have gradients
        has_grad = False
        for p in small_network.network_parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        assert has_grad


class TestDARTSTrainer:
    """Test DARTS trainer."""

    @pytest.fixture
    def mock_loaders(self):
        """Create mock data loaders."""
        class MockLoader:
            def __init__(self):
                self.data = [
                    (torch.randn(2, 3, 32, 32), torch.randint(0, 2, (2,)))
                    for _ in range(3)
                ]
                self.idx = 0

            def __iter__(self):
                self.idx = 0
                return self

            def __next__(self):
                if self.idx >= len(self.data):
                    raise StopIteration
                item = self.data[self.idx]
                self.idx += 1
                return item

            def __len__(self):
                return len(self.data)

        return MockLoader(), MockLoader()

    def test_train_step(self, mock_loaders):
        """Test single training step."""
        train_loader, val_loader = mock_loaders

        config = DARTSConfig(init_channels=8, num_cells=2, num_nodes=2)
        model = DARTSNetwork(config, num_classes=2)

        trainer = DARTSTrainer(model, train_loader, val_loader, device='cpu')

        x_train, y_train = next(iter(train_loader))
        x_val, y_val = next(iter(val_loader))

        losses = trainer.train_step(x_train, y_train, x_val, y_val)

        assert 'loss_train' in losses
        assert 'loss_val' in losses


class TestCreateDARTSNetwork:
    """Test factory function."""

    def test_create_medical_safe(self):
        """Test creating network with medical-safe primitives."""
        model = create_darts_network(
            num_classes=2,
            init_channels=8,
            num_cells=4,
            use_medical_safe=True
        )

        assert model.config.primitives == MEDICAL_SAFE_PRIMITIVES

    def test_create_standard(self):
        """Test creating network with standard primitives."""
        model = create_darts_network(
            num_classes=2,
            init_channels=8,
            num_cells=4,
            use_medical_safe=False
        )

        assert model.config.primitives == DARTS_PRIMITIVES


class TestGenotypeToDict:
    """Test genotype serialization."""

    def test_genotype_to_dict(self):
        """Test converting genotype to dict."""
        genotype = {
            'normal': [('sep_conv_3x3', 0), ('skip_connect', 1)],
            'reduce': [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)]
        }

        result = genotype_to_dict(genotype)

        assert result['normal'] == [('sep_conv_3x3', 0), ('skip_connect', 1)]
        assert result['reduce'] == [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)]


class TestDARTSSchemaIntegration:
    """Test integration with architecture_grammar schema."""

    def test_darts_search_config(self):
        """Test DARTSSearchConfig schema."""
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "architecture_grammar",
            "/Users/bengibson/arc-autonomous-research/schemas/architecture_grammar.py"
        )
        arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch_module)
        DARTSSearchConfig = arch_module.DARTSSearchConfig

        config = DARTSSearchConfig(
            enabled=True,
            init_channels=16,
            num_cells=8
        )

        assert config.enabled
        assert config.init_channels == 16

    def test_validate_darts_config(self):
        """Test DARTS config validation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "architecture_grammar",
            "/Users/bengibson/arc-autonomous-research/schemas/architecture_grammar.py"
        )
        arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch_module)
        DARTSSearchConfig = arch_module.DARTSSearchConfig
        validate_darts_config = arch_module.validate_darts_config

        config = DARTSSearchConfig(
            enabled=True,
            search_epochs=50
        )

        is_valid, error = validate_darts_config(config)
        assert is_valid

    def test_validate_darts_config_low_epochs(self):
        """Test validation fails for low epochs."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "architecture_grammar",
            "/Users/bengibson/arc-autonomous-research/schemas/architecture_grammar.py"
        )
        arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch_module)
        DARTSSearchConfig = arch_module.DARTSSearchConfig
        validate_darts_config = arch_module.validate_darts_config

        config = DARTSSearchConfig(
            enabled=True,
            search_epochs=15
        )

        is_valid, error = validate_darts_config(config)
        assert not is_valid
        assert "search_epochs" in error
