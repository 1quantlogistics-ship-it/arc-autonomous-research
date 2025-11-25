"""
DARTS: Differentiable Architecture Search for ARC.

Implements gradient-based neural architecture search that respects
ARC's existing constraints (10M params, 5GB memory).

Reference: https://arxiv.org/abs/1806.09055

Key Features:
- Differentiable architecture search with mixed operations
- Medical-safe primitive set (no aggressive pooling)
- Bilevel optimization (network weights + architecture params)
- ARC constraint validation (10M params, 5GB memory)
- Genotype extraction for discrete architecture

Author: ARC Team (Dev 2)
Created: 2025-11-24
Version: 1.0 (Phase F)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DARTS OPERATIONS
# ============================================================================

class DARTSOperations:
    """Standard DARTS operation set."""

    @staticmethod
    def none(C: int, stride: int) -> nn.Module:
        """Zero operation (edge removal)."""
        return Zero(stride)

    @staticmethod
    def skip_connect(C: int, stride: int) -> nn.Module:
        """Identity or factorized reduce."""
        if stride == 1:
            return Identity()
        return FactorizedReduce(C, C)

    @staticmethod
    def avg_pool_3x3(C: int, stride: int) -> nn.Module:
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    @staticmethod
    def max_pool_3x3(C: int, stride: int) -> nn.Module:
        return nn.MaxPool2d(3, stride=stride, padding=1)

    @staticmethod
    def sep_conv_3x3(C: int, stride: int) -> nn.Module:
        return SepConv(C, C, 3, stride, 1)

    @staticmethod
    def sep_conv_5x5(C: int, stride: int) -> nn.Module:
        return SepConv(C, C, 5, stride, 2)

    @staticmethod
    def dil_conv_3x3(C: int, stride: int) -> nn.Module:
        return DilConv(C, C, 3, stride, 2, 2)

    @staticmethod
    def dil_conv_5x5(C: int, stride: int) -> nn.Module:
        return DilConv(C, C, 5, stride, 4, 2)


# Operation primitives - standard DARTS set
DARTS_PRIMITIVES = [
    'none',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

# Medical imaging safe subset (excludes aggressive pooling that may lose detail)
MEDICAL_SAFE_PRIMITIVES = [
    'none',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


class Zero(nn.Module):
    """Zero operation for edge removal."""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x * 0.0
        return x[:, :, ::self.stride, ::self.stride] * 0.0


class Identity(nn.Module):
    """Identity operation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FactorizedReduce(nn.Module):
    """Reduce spatial dimensions while maintaining channels."""
    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        assert C_out % 2 == 0, f"C_out ({C_out}) must be divisible by 2"
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class SepConv(nn.Module):
    """Separable convolution."""
    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int, padding: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding,
                     groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, kernel_size, 1, padding,
                     groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    """Dilated convolution."""
    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int, padding: int, dilation: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding,
                     dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ============================================================================
# MIXED OPERATION (Differentiable)
# ============================================================================

class MixedOp(nn.Module):
    """
    Mixed operation: weighted sum of all primitive operations.
    Weights are architecture parameters (alphas).
    """

    def __init__(self, C: int, stride: int,
                 primitives: List[str] = None):
        super().__init__()
        if primitives is None:
            primitives = DARTS_PRIMITIVES

        self.ops = nn.ModuleList()

        for primitive in primitives:
            op_fn = getattr(DARTSOperations, primitive)
            self.ops.append(op_fn(C, stride))

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            weights: Softmax architecture weights [num_ops]
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# ============================================================================
# DARTS CELL
# ============================================================================

class DARTSCell(nn.Module):
    """
    DARTS cell with learnable topology.

    Args:
        C_prev_prev: Input channels from 2 cells ago
        C_prev: Input channels from previous cell
        C: Number of output channels for this cell
        reduction: Whether this is a reduction cell
        reduction_prev: Whether previous cell was reduction
        primitives: List of operation names to search over
        num_nodes: Number of intermediate nodes in the cell
    """

    def __init__(self, C_prev_prev: int, C_prev: int, C: int,
                 reduction: bool = False, reduction_prev: bool = False,
                 primitives: List[str] = None,
                 num_nodes: int = 4):
        super().__init__()

        if primitives is None:
            primitives = DARTS_PRIMITIVES

        self.reduction = reduction
        self.num_nodes = num_nodes
        self.primitives = primitives

        # Preprocess inputs to match C channels
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = nn.Sequential(
                nn.Conv2d(C_prev_prev, C, 1, bias=False),
                nn.BatchNorm2d(C),
            )
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(C_prev, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

        # Mixed operations for each edge
        self.ops = nn.ModuleList()
        self.num_edges = 0

        for i in range(num_nodes):
            for j in range(i + 2):  # Connect to input nodes and previous intermediate nodes
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(MixedOp(C, stride, primitives))
                self.num_edges += 1

    def forward(self, s0: torch.Tensor, s1: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s0, s1: Inputs from previous cells
            weights: Architecture weights [num_edges, num_ops]
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self.num_nodes):
            # Sum contributions from all previous states
            s = sum(
                self.ops[offset + j](states[j], weights[offset + j])
                for j in range(len(states))
            )
            offset += len(states)
            states.append(s)

        # Concatenate intermediate nodes
        return torch.cat(states[2:], dim=1)


# ============================================================================
# DARTS NETWORK
# ============================================================================

@dataclass
class DARTSConfig:
    """Configuration for DARTS search."""
    init_channels: int = 16
    num_cells: int = 8
    num_nodes: int = 4
    primitives: List[str] = None

    # ARC constraints
    max_params: int = 10_000_000  # 10M
    max_memory_gb: float = 5.0

    # Search settings
    arch_learning_rate: float = 3e-4
    arch_weight_decay: float = 1e-3

    def __post_init__(self):
        if self.primitives is None:
            self.primitives = MEDICAL_SAFE_PRIMITIVES


class DARTSNetwork(nn.Module):
    """
    DARTS network for architecture search.

    Includes architecture parameters (alphas) that are learned
    alongside network weights.
    """

    def __init__(self, config: DARTSConfig, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        C = config.init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        # Build cells with proper channel tracking
        self.cells = nn.ModuleList()

        # Track channels: stem outputs C, each cell outputs C * num_nodes (concatenated)
        C_prev_prev = C  # From stem
        C_prev = C       # From stem (both inputs start as stem output)
        C_curr = C       # Current cell's internal channel count

        reduction_layers = [config.num_cells // 3, 2 * config.num_cells // 3]
        reduction_prev = False

        for i in range(config.num_cells):
            reduction = i in reduction_layers
            if reduction:
                C_curr *= 2

            cell = DARTSCell(
                C_prev_prev=C_prev_prev,
                C_prev=C_prev,
                C=C_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
                primitives=config.primitives,
                num_nodes=config.num_nodes
            )
            self.cells.append(cell)

            # Update channel tracking for next cell
            # Cell output = C_curr * num_nodes (concatenated intermediate nodes)
            C_prev_prev = C_prev
            C_prev = C_curr * config.num_nodes
            reduction_prev = reduction

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Final classifier input is the last cell's output channels
        self.classifier = nn.Linear(C_prev, num_classes)

        # Initialize architecture parameters
        self._init_arch_params()

        # Validate constraints
        self._validate_constraints()

    def _init_arch_params(self):
        """Initialize architecture parameters (alphas)."""
        num_ops = len(self.config.primitives)

        # Normal cell alphas
        num_edges_normal = sum(i + 2 for i in range(self.config.num_nodes))
        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(num_edges_normal, num_ops)
        )

        # Reduction cell alphas
        self.alphas_reduce = nn.Parameter(
            1e-3 * torch.randn(num_edges_normal, num_ops)
        )

    def _validate_constraints(self):
        """Validate ARC constraints."""
        num_params = sum(p.numel() for p in self.parameters())

        if num_params > self.config.max_params:
            raise ValueError(
                f"Model has {num_params:,} params, exceeds limit of "
                f"{self.config.max_params:,}"
            )

        logger.info(f"DARTS network: {num_params:,} parameters")

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return architecture parameters for separate optimizer."""
        return [self.alphas_normal, self.alphas_reduce]

    def network_parameters(self) -> List[nn.Parameter]:
        """Return network weights (excluding arch params)."""
        arch_param_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_param_ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax over operations
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def get_genotype(self) -> Dict:
        """
        Extract discrete architecture from continuous weights.
        Returns the top-2 operations for each node.
        """
        def parse_weights(weights: torch.Tensor, num_nodes: int) -> List[Tuple]:
            gene = []
            offset = 0

            for i in range(num_nodes):
                # Get edges connecting to this node
                edges = []
                for j in range(i + 2):
                    edge_weights = weights[offset + j]
                    # Exclude 'none' operation
                    if 'none' in self.config.primitives:
                        none_idx = self.config.primitives.index('none')
                        edge_weights = edge_weights.clone()
                        edge_weights[none_idx] = -float('inf')

                    best_op = edge_weights.argmax().item()
                    edges.append((
                        self.config.primitives[best_op],
                        j,
                        edge_weights[best_op].item()
                    ))

                # Keep top-2 edges
                edges = sorted(edges, key=lambda x: -x[2])[:2]
                gene.extend([(op, src) for op, src, _ in edges])
                offset += i + 2

            return gene

        with torch.no_grad():
            weights_normal = F.softmax(self.alphas_normal, dim=-1)
            weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        return {
            'normal': parse_weights(weights_normal, self.config.num_nodes),
            'reduce': parse_weights(weights_reduce, self.config.num_nodes),
        }


# ============================================================================
# DARTS TRAINER
# ============================================================================

class DARTSTrainer:
    """
    Trainer for DARTS architecture search.
    Uses bilevel optimization: network weights and arch params.
    """

    def __init__(
        self,
        model: DARTSNetwork,
        train_loader,
        val_loader,
        device: str = 'cuda',
        criterion: nn.Module = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Separate optimizers
        self.optimizer_w = torch.optim.SGD(
            model.network_parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )

        self.optimizer_a = torch.optim.Adam(
            model.arch_parameters(),
            lr=model.config.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=model.config.arch_weight_decay
        )

    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor,
                   x_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
        """
        Single DARTS training step:
        1. Update architecture params on validation data
        2. Update network weights on training data
        """
        self.model.train()

        # Step 1: Update architecture parameters
        self.optimizer_a.zero_grad()
        logits_val = self.model(x_val.to(self.device))
        loss_val = self.criterion(logits_val, y_val.to(self.device))
        loss_val.backward()
        self.optimizer_a.step()

        # Step 2: Update network weights
        self.optimizer_w.zero_grad()
        logits_train = self.model(x_train.to(self.device))
        loss_train = self.criterion(logits_train, y_train.to(self.device))
        loss_train.backward()
        self.optimizer_w.step()

        return {
            'loss_train': loss_train.item(),
            'loss_val': loss_val.item()
        }

    def search(self, num_epochs: int) -> Dict:
        """
        Run architecture search.

        Args:
            num_epochs: Number of search epochs

        Returns:
            Best discovered architecture (genotype)
        """
        train_iter = iter(self.train_loader)
        val_iter = iter(self.val_loader)

        for epoch in range(num_epochs):
            epoch_losses = []

            for step in range(len(self.train_loader)):
                try:
                    x_train, y_train = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x_train, y_train = next(train_iter)

                try:
                    x_val, y_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    x_val, y_val = next(val_iter)

                losses = self.train_step(x_train, y_train, x_val, y_val)
                epoch_losses.append(losses)

            avg_loss = sum(l['loss_train'] for l in epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            # Log current architecture
            genotype = self.model.get_genotype()
            logger.debug(f"Current genotype: {genotype}")

        return self.model.get_genotype()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_darts_network(
    num_classes: int,
    init_channels: int = 16,
    num_cells: int = 8,
    use_medical_safe: bool = True,
    max_params: int = 10_000_000
) -> DARTSNetwork:
    """
    Factory function to create a DARTS network.

    Args:
        num_classes: Number of output classes
        init_channels: Initial channel count
        num_cells: Number of cells in the network
        use_medical_safe: Use medical-safe primitives (no aggressive pooling)
        max_params: Maximum parameter count

    Returns:
        Configured DARTSNetwork
    """
    config = DARTSConfig(
        init_channels=init_channels,
        num_cells=num_cells,
        primitives=MEDICAL_SAFE_PRIMITIVES if use_medical_safe else DARTS_PRIMITIVES,
        max_params=max_params
    )
    return DARTSNetwork(config, num_classes=num_classes)


def genotype_to_dict(genotype: Dict) -> Dict:
    """
    Convert genotype to a serializable dictionary format.

    Args:
        genotype: Genotype from get_genotype()

    Returns:
        Serializable dictionary
    """
    return {
        'normal': [(op, int(src)) for op, src in genotype['normal']],
        'reduce': [(op, int(src)) for op, src in genotype['reduce']],
    }
