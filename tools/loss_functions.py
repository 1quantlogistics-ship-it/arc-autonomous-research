"""
Custom Loss Functions for ARC Glaucoma Detection.

Implements advanced loss functions for handling class imbalance and multi-task learning
in glaucoma detection models. All losses are PyTorch-compatible and FDA-compliant.

Loss Types:
- FocalLoss: Handles class imbalance via focusing parameter
- DiceLoss: Segmentation-inspired overlap loss
- TverskyLoss: Asymmetric Dice with FP/FN trade-off control
- CombinedLoss: Weighted combination of multiple losses
- MultiTaskLoss: Multi-task learning wrapper
- LovaszSoftmax: Better IoU optimization for multi-class segmentation (Phase F)
- LovaszHinge: Binary IoU optimization (Phase F)
- BoundaryLoss: Distance-weighted boundary loss for medical imaging (Phase F)
- CompoundLoss: Learnable weighted combination of losses (Phase F)

Author: ARC Team (Dev 1, Dev 2)
Created: 2025-11-19
Version: 1.1 (Phase F Enhanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - γ (gamma): Focusing parameter (default: 2.0)
    - α (alpha): Class balance weight (default: 0.25)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class balance weight (None = no weighting, float = positive class weight)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (batch_size, num_classes) or (batch_size,)
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        # Binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply class balance weight if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-inspired classification.

    Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice Coefficient

    Useful for imbalanced datasets as it focuses on overlap rather than pixel accuracy.

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (batch_size, ...)
            targets: Ground truth labels (batch_size, ...)

        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff

        # Reduction already applied (single value)
        return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for asymmetric Dice with FP/FN trade-off control.

    Reference: Salehi et al., "Tversky Loss Function for Image Segmentation" (2017)

    TL = 1 - (TP + smooth) / (TP + α*FP + β*FN + smooth)

    where:
    - α (alpha): Weight for false positives
    - β (beta): Weight for false negatives
    - α + β = 1 typically

    For glaucoma detection:
    - β > α: Prioritize recall (minimize false negatives)
    - β < α: Prioritize precision (minimize false positives)

    Args:
        alpha: FP weight (default: 0.3)
        beta: FN weight (default: 0.7) - higher to prioritize recall
        smooth: Smoothing factor (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        reduction: str = 'mean'
    ):
        super(TverskyLoss, self).__init__()

        # Validate alpha + beta
        if abs(alpha + beta - 1.0) > 0.01:
            raise ValueError(f"Alpha ({alpha}) + Beta ({beta}) should sum to 1.0")

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (batch_size, ...)
            targets: Ground truth labels (batch_size, ...)

        Returns:
            Tversky loss value
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute TP, FP, FN
        TP = (probs_flat * targets_flat).sum()
        FP = (probs_flat * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()

        # Compute Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1.0 - tversky_index

        return tversky_loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.

    Useful for combining BCE + Dice, or other loss combinations.

    Args:
        loss1: First loss function
        loss2: Second loss function
        weight1: Weight for first loss (default: 0.5)
        weight2: Weight for second loss (default: 0.5)

    Example:
        combined = CombinedLoss(
            nn.BCEWithLogitsLoss(),
            DiceLoss(),
            weight1=0.6,
            weight2=0.4
        )
    """

    def __init__(
        self,
        loss1: nn.Module,
        loss2: nn.Module,
        weight1: float = 0.5,
        weight2: float = 0.5
    ):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model
            targets: Ground truth labels

        Returns:
            Weighted combined loss
        """
        l1 = self.loss1(inputs, targets)
        l2 = self.loss2(inputs, targets)

        return self.weight1 * l1 + self.weight2 * l2


class MultiTaskLoss(nn.Module):
    """
    Multi-task learning loss with primary classification + auxiliary tasks.

    Used for training models with auxiliary heads (e.g., DRI prediction, CDR prediction)
    to improve feature learning for primary glaucoma classification.

    Args:
        primary_loss: Loss for primary classification task
        primary_weight: Weight for primary loss (should be ≥ 0.6 for clinical safety)
        auxiliary_losses: Dict of {task_name: (loss_fn, weight)}

    Example:
        multi_task_loss = MultiTaskLoss(
            primary_loss=FocalLoss(gamma=2.0),
            primary_weight=0.7,
            auxiliary_losses={
                "dri_prediction": (nn.MSELoss(), 0.2),
                "cdr_prediction": (nn.SmoothL1Loss(), 0.1)
            }
        )
    """

    def __init__(
        self,
        primary_loss: nn.Module,
        primary_weight: float = 0.7,
        auxiliary_losses: Optional[Dict[str, tuple]] = None
    ):
        super(MultiTaskLoss, self).__init__()

        # Validate primary weight
        if primary_weight < 0.6:
            raise ValueError(
                f"Primary weight ({primary_weight}) must be ≥ 0.6 for clinical safety. "
                f"Classification must remain the primary objective."
            )

        self.primary_loss = primary_loss
        self.primary_weight = primary_weight
        self.auxiliary_losses = auxiliary_losses or {}

        # Validate total weight
        aux_weight_sum = sum(weight for _, weight in self.auxiliary_losses.values())
        total_weight = primary_weight + aux_weight_sum

        if total_weight > 1.0 + 1e-6:
            raise ValueError(
                f"Total loss weight ({total_weight:.3f}) exceeds 1.0. "
                f"Primary: {primary_weight}, Auxiliary: {aux_weight_sum:.3f}"
            )

    def forward(
        self,
        primary_inputs: torch.Tensor,
        primary_targets: torch.Tensor,
        auxiliary_inputs: Optional[Dict[str, torch.Tensor]] = None,
        auxiliary_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            primary_inputs: Logits for primary classification
            primary_targets: Ground truth for primary task
            auxiliary_inputs: Dict of {task_name: logits} for auxiliary tasks
            auxiliary_targets: Dict of {task_name: targets} for auxiliary tasks

        Returns:
            Dict with:
            - 'total': Total weighted loss
            - 'primary': Primary task loss
            - '<task_name>': Each auxiliary task loss
        """
        # Compute primary loss
        primary_loss_value = self.primary_loss(primary_inputs, primary_targets)
        total_loss = self.primary_weight * primary_loss_value

        loss_dict = {
            'primary': primary_loss_value,
            'total': total_loss.clone()
        }

        # Compute auxiliary losses
        if auxiliary_inputs is not None and auxiliary_targets is not None:
            for task_name, (loss_fn, weight) in self.auxiliary_losses.items():
                if task_name in auxiliary_inputs and task_name in auxiliary_targets:
                    aux_loss = loss_fn(auxiliary_inputs[task_name], auxiliary_targets[task_name])
                    loss_dict[task_name] = aux_loss
                    loss_dict['total'] = loss_dict['total'] + weight * aux_loss

        return loss_dict


def build_loss_from_config(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Build PyTorch loss function from LossConfig schema.

    Translates ARC LossConfig schema into actual PyTorch loss implementations.

    Args:
        loss_config: Dict from LossConfig.to_dict()

    Returns:
        PyTorch loss module

    Example:
        loss_config = {
            "primary_loss": "focal",
            "primary_weight": 1.0,
            "hyperparameters": {"focal_gamma": 2.0, "focal_alpha": 0.75},
            "class_weighting": "balanced",
            "auxiliary_tasks": [...]
        }

        loss = build_loss_from_config(loss_config)
    """
    primary_loss_type = loss_config.get("primary_loss", "bce")
    hyperparams = loss_config.get("hyperparameters", {}) or {}
    class_weighting = loss_config.get("class_weighting", "none")
    auxiliary_tasks = loss_config.get("auxiliary_tasks", [])
    primary_weight = loss_config.get("primary_weight", 1.0)

    # Build primary loss
    if primary_loss_type == "focal":
        gamma = hyperparams.get("focal_gamma", 2.0)
        alpha = hyperparams.get("focal_alpha", None)
        primary_loss = FocalLoss(gamma=gamma, alpha=alpha)

    elif primary_loss_type == "dice":
        primary_loss = DiceLoss()

    elif primary_loss_type == "tversky":
        alpha = hyperparams.get("tversky_alpha", 0.3)
        beta = hyperparams.get("tversky_beta", 0.7)
        primary_loss = TverskyLoss(alpha=alpha, beta=beta)

    elif primary_loss_type == "bce_dice":
        # Combined BCE + Dice
        weight_bce = hyperparams.get("combination_weight", 0.5)
        weight_dice = 1.0 - weight_bce
        primary_loss = CombinedLoss(
            nn.BCEWithLogitsLoss(),
            DiceLoss(),
            weight1=weight_bce,
            weight2=weight_dice
        )

    elif primary_loss_type == "weighted_bce" or class_weighting != "none":
        # BCE with class weights
        # Note: pos_weight should be computed from actual class distribution
        # For now, use a placeholder that will be filled by training executor
        primary_loss = nn.BCEWithLogitsLoss()

    else:  # Default: regular BCE
        primary_loss = nn.BCEWithLogitsLoss()

    # If no auxiliary tasks, return primary loss
    if not auxiliary_tasks:
        return primary_loss

    # Build auxiliary losses
    auxiliary_losses = {}
    for task_config in auxiliary_tasks:
        task_type = task_config["task_type"]
        task_weight = task_config["weight"]
        task_loss_type = task_config.get("loss_type", "mse")

        # Map auxiliary loss type to PyTorch loss
        if task_loss_type == "mse":
            aux_loss_fn = nn.MSELoss()
        elif task_loss_type == "mae":
            aux_loss_fn = nn.L1Loss()
        elif task_loss_type == "smooth_l1" or task_loss_type == "huber":
            aux_loss_fn = nn.SmoothL1Loss()
        else:
            aux_loss_fn = nn.MSELoss()  # Default

        auxiliary_losses[task_type] = (aux_loss_fn, task_weight)

    # Return multi-task loss
    return MultiTaskLoss(
        primary_loss=primary_loss,
        primary_weight=primary_weight,
        auxiliary_losses=auxiliary_losses
    )


# ============================================================================
# PHASE F: LOVASZ LOSSES - Better for IoU optimization
# ============================================================================

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors.
    See Algorithm 1 in https://arxiv.org/abs/1705.08790

    Args:
        gt_sorted: Ground truth labels sorted by prediction error

    Returns:
        Lovasz gradient for the sorted ground truth
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor,
                        classes: str = 'present') -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss.

    Args:
        probas: [P, C] softmax probabilities
        labels: [P] ground truth labels
        classes: 'all', 'present', or list of class indices

    Returns:
        Lovasz-Softmax loss value
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []

    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)


class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax loss for multi-class semantic segmentation.

    Better than cross-entropy for optimizing IoU/Jaccard directly.
    Reference: https://arxiv.org/abs/1705.08790

    Args:
        classes: 'all' for all classes, 'present' for classes in batch
        per_image: Compute loss per image then average
        ignore_index: Class index to ignore
    """

    def __init__(self, classes: str = 'present', per_image: bool = False,
                 ignore_index: int = -100):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] raw predictions
            labels: [B, H, W] ground truth

        Returns:
            Lovasz-Softmax loss value
        """
        probas = F.softmax(logits, dim=1)

        if self.per_image:
            losses = []
            for prob, lab in zip(probas, labels):
                # Flatten
                prob_flat = prob.permute(1, 2, 0).reshape(-1, prob.size(0))
                lab_flat = lab.reshape(-1)

                # Remove ignored
                if self.ignore_index is not None:
                    valid = lab_flat != self.ignore_index
                    prob_flat = prob_flat[valid]
                    lab_flat = lab_flat[valid]

                losses.append(lovasz_softmax_flat(prob_flat, lab_flat, self.classes))
            return torch.stack(losses).mean()
        else:
            # Flatten all
            B, C, H, W = probas.shape
            probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)

            if self.ignore_index is not None:
                valid = labels_flat != self.ignore_index
                probas_flat = probas_flat[valid]
                labels_flat = labels_flat[valid]

            return lovasz_softmax_flat(probas_flat, labels_flat, self.classes)


class LovaszHinge(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes IoU directly using hinge loss formulation.

    Args:
        per_image: Compute loss per image then average
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W] raw predictions
            labels: [B, H, W] binary ground truth

        Returns:
            Lovasz-Hinge loss value
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        if self.per_image:
            losses = []
            for logit, label in zip(logits, labels):
                logit_flat = logit.reshape(-1)
                label_flat = label.reshape(-1)
                losses.append(self._lovasz_hinge_flat(logit_flat, label_flat))
            return torch.stack(losses).mean()
        else:
            return self._lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))

    def _lovasz_hinge_flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary Lovasz hinge loss."""
        if len(labels) == 0:
            return logits.sum() * 0.0

        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


# ============================================================================
# PHASE F: BOUNDARY LOSS - For medical imaging with class imbalance
# ============================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary loss for medical image segmentation.
    Uses distance transform to weight boundary regions more heavily.

    Reference: https://arxiv.org/abs/1812.07032

    Args:
        theta0: Initial boundary weight
        theta: Rate of boundary weight increase
    """

    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dist_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] predictions
            labels: [B, H, W] ground truth
            dist_maps: [B, C, H, W] precomputed distance transforms (optional)

        Returns:
            Boundary loss value
        """
        probas = F.softmax(logits, dim=1)

        if dist_maps is None:
            # If no distance maps provided, fall back to simple boundary detection
            logger.warning("BoundaryLoss: dist_maps not provided, using edge detection fallback")
            return self._edge_fallback(probas, labels)

        # Boundary loss component
        boundary_loss = (probas * dist_maps).sum() / probas.numel()

        return boundary_loss

    def _edge_fallback(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fallback boundary loss using Sobel edge detection."""
        # Simple edge-based boundary weighting
        B, C, H, W = probas.shape

        # Create edge weights from labels
        labels_onehot = F.one_hot(labels.long(), num_classes=C).permute(0, 3, 1, 2).float()

        # Sobel-like edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=probas.dtype, device=probas.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=probas.dtype, device=probas.device).view(1, 1, 3, 3)

        edges = torch.zeros_like(labels_onehot)
        for c in range(C):
            channel = labels_onehot[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edges[:, c:c+1, :, :] = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

        # Weight predictions by edge importance
        weighted_loss = (probas * edges).sum() / probas.numel()
        return weighted_loss

    @staticmethod
    def compute_distance_map(label: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Compute distance transform for boundary loss.
        Call this during data loading.

        Args:
            label: [H, W] ground truth label array
            num_classes: Number of classes

        Returns:
            [C, H, W] signed distance transforms per class
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            logger.error("scipy required for distance_transform_edt")
            return np.zeros((num_classes,) + label.shape, dtype=np.float32)

        dist_maps = np.zeros((num_classes,) + label.shape, dtype=np.float32)

        for c in range(num_classes):
            mask = (label == c).astype(np.uint8)
            if mask.sum() > 0:
                # Distance inside
                dist_inside = distance_transform_edt(mask)
                # Distance outside (negate)
                dist_outside = distance_transform_edt(1 - mask)
                # Signed distance (negative inside, positive outside)
                dist_maps[c] = dist_outside - dist_inside

        return dist_maps


# ============================================================================
# PHASE F: COMPOUND LOSS - Learnable combination of multiple losses
# ============================================================================

class CompoundLoss(nn.Module):
    """
    Compound loss with learnable or fixed weights.
    Combines multiple loss functions optimally.

    Args:
        losses: List of loss functions
        weights: Initial weights (learned if learnable_weights=True)
        learnable_weights: Whether to learn loss weights during training
        normalize_weights: Softmax normalize weights
        loss_names: Names for each loss (for logging)
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
        learnable_weights: bool = True,
        normalize_weights: bool = True,
        loss_names: Optional[List[str]] = None
    ):
        super().__init__()

        self.losses = nn.ModuleList(losses)
        self.learnable_weights = learnable_weights
        self.normalize_weights = normalize_weights
        self.loss_names = loss_names or [f"loss_{i}" for i in range(len(losses))]

        # Initialize weights
        if weights is None:
            weights = [1.0] * len(losses)

        if learnable_weights:
            # Use log-space for numerical stability
            self.log_weights = nn.Parameter(torch.log(torch.tensor(weights, dtype=torch.float32)))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def get_weights(self) -> torch.Tensor:
        """Get current (possibly normalized) weights."""
        if self.learnable_weights:
            weights = torch.exp(self.log_weights)
        else:
            weights = self.weights

        if self.normalize_weights:
            weights = F.softmax(weights, dim=0)

        return weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute weighted sum of losses.

        Args:
            logits: Model predictions
            labels: Ground truth labels
            **kwargs: Additional arguments passed to individual losses

        Returns:
            Dict with 'total' loss and individual loss values
        """
        weights = self.get_weights()

        loss_values = {}
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for i, (loss_fn, name) in enumerate(zip(self.losses, self.loss_names)):
            try:
                loss_val = loss_fn(logits, labels, **kwargs)
            except TypeError:
                # Some losses don't accept kwargs
                loss_val = loss_fn(logits, labels)

            loss_values[name] = loss_val
            total_loss = total_loss + weights[i] * loss_val

        loss_values['total'] = total_loss
        loss_values['weights'] = {
            name: w.item() for name, w in zip(self.loss_names, weights)
        }

        return loss_values

    def get_weight_summary(self) -> Dict[str, float]:
        """Get current weight values for logging."""
        weights = self.get_weights()
        return {name: w.item() for name, w in zip(self.loss_names, weights)}


# ============================================================================
# PHASE G: AUC-OPTIMIZATION LOSSES - Direct AUC optimization
# ============================================================================

class AUCMarginLoss(nn.Module):
    """
    AUC-margin loss for direct AUC optimization.

    Optimizes pairwise ranking with margin constraint.
    Reference: "Optimizing AUC using Margin-based Algorithms"

    Instead of enumerating all positive-negative pairs (O(n^2)),
    samples a fixed number of pairs for efficiency.

    Args:
        margin: Margin for ranking constraint (default: 1.0)
        gamma: Surrogate loss temperature (default: 0.1)
        num_samples: Number of pairs to sample per batch (default: 1000)
    """

    def __init__(
        self,
        margin: float = 1.0,
        gamma: float = 0.1,
        num_samples: int = 1000
    ):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.num_samples = num_samples

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AUC-margin loss.

        Samples positive-negative pairs and optimizes margin ranking.
        Uses surrogate hinge loss for differentiability.

        Args:
            predictions: Predicted scores/logits (batch_size,)
            targets: Binary labels (batch_size,)

        Returns:
            AUC-margin loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        # Get positive and negative indices
        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_pos == 0 or n_neg == 0:
            # Return zero loss if no pairs can be formed
            return predictions.sum() * 0.0

        # Sample pairs
        num_pairs = min(self.num_samples, n_pos * n_neg)

        # Random sampling with replacement
        pos_sample = pos_indices[torch.randint(n_pos, (num_pairs,), device=predictions.device)]
        neg_sample = neg_indices[torch.randint(n_neg, (num_pairs,), device=predictions.device)]

        # Get scores for sampled pairs
        pos_scores = predictions[pos_sample]
        neg_scores = predictions[neg_sample]

        # Compute surrogate hinge loss: max(0, margin - (s_pos - s_neg))
        # Using softplus as differentiable approximation to hinge
        score_diff = pos_scores - neg_scores
        loss = F.softplus(self.margin - score_diff, beta=1.0/self.gamma)

        return loss.mean()


class PartialAUCLoss(nn.Module):
    """
    Partial AUC loss for clinically relevant FPR range.

    Focuses optimization on low FPR region critical for screening.
    Reference: "Optimizing Partial Area Under ROC Curve"

    Uses two-way partial AUC formulation for efficiency.

    Args:
        fpr_range: (min_fpr, max_fpr) for partial AUC (default: 0.0, 0.2)
        lambda_reg: Regularization weight (default: 0.1)
        temperature: Softmax temperature for weighting (default: 1.0)
        num_samples: Pairs to sample per batch (default: 1000)
    """

    def __init__(
        self,
        fpr_range: tuple = (0.0, 0.2),
        lambda_reg: float = 0.1,
        temperature: float = 1.0,
        num_samples: int = 1000
    ):
        super().__init__()
        self.fpr_min = fpr_range[0]
        self.fpr_max = fpr_range[1]
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute partial AUC loss in specified FPR range.

        Uses importance weighting to focus on low FPR region.

        Args:
            predictions: Predicted scores/logits (batch_size,)
            targets: Binary labels (batch_size,)

        Returns:
            Partial AUC loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_pos == 0 or n_neg == 0:
            return predictions.sum() * 0.0

        pos_scores = predictions[pos_mask]
        neg_scores = predictions[neg_mask]

        # For partial AUC in [0, max_fpr], we weight negative samples
        # by how likely they are to contribute to the FPR in that range
        # Higher negative scores contribute more to FPR

        # Sort negative scores (descending = highest first)
        neg_sorted, neg_sort_idx = torch.sort(neg_scores, descending=True)

        # Take only top (max_fpr * n_neg) negatives
        num_neg_to_use = max(1, int(self.fpr_max * n_neg))
        top_neg_scores = neg_sorted[:num_neg_to_use]

        # Sample pairs between positives and top negatives
        num_pairs = min(self.num_samples, n_pos * num_neg_to_use)

        pos_sample = torch.randint(n_pos, (num_pairs,), device=predictions.device)
        neg_sample = torch.randint(num_neg_to_use, (num_pairs,), device=predictions.device)

        sampled_pos = pos_scores[pos_sample]
        sampled_neg = top_neg_scores[neg_sample]

        # Surrogate loss: want pos > neg
        score_diff = sampled_pos - sampled_neg
        loss = F.softplus(-score_diff / self.temperature)

        # Add regularization to prevent score explosion
        reg_loss = self.lambda_reg * (predictions.pow(2).mean())

        return loss.mean() + reg_loss


class CompositeMedicalLoss(nn.Module):
    """
    Composite loss combining classification and AUC optimization.

    Balances:
    - BCE for probability calibration
    - Focal for hard example mining
    - AUC-margin for ranking optimization

    Args:
        bce_weight: Weight for BCE loss (default: 0.3)
        focal_weight: Weight for Focal loss (default: 0.3)
        auc_weight: Weight for AUC-margin loss (default: 0.4)
        focal_gamma: Gamma for Focal loss (default: 2.0)
        auc_margin: Margin for AUC loss (default: 1.0)
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        focal_weight: float = 0.3,
        auc_weight: float = 0.4,
        focal_gamma: float = 2.0,
        auc_margin: float = 1.0
    ):
        super().__init__()
        self.weights = {
            'bce': bce_weight,
            'focal': focal_weight,
            'auc': auc_weight
        }
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.auc = AUCMarginLoss(margin=auc_margin)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss with component breakdown.

        Args:
            predictions: Model logits
            targets: Binary labels

        Returns:
            Dict with 'total', 'bce', 'focal', 'auc' loss values
        """
        targets_float = targets.float()

        bce_loss = self.bce(predictions, targets_float)
        focal_loss = self.focal(predictions, targets_float)
        auc_loss = self.auc(predictions, targets_float)

        total = (
            self.weights['bce'] * bce_loss +
            self.weights['focal'] * focal_loss +
            self.weights['auc'] * auc_loss
        )

        return {
            'total': total,
            'bce': bce_loss,
            'focal': focal_loss,
            'auc': auc_loss
        }


# ============================================================================
# PHASE F/G: LOSS FACTORY - Easy creation of configured losses
# ============================================================================

class LossFactory:
    """Factory for creating loss functions from config."""

    REGISTRY = {
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss,
        'focal': FocalLoss,
        'dice': DiceLoss,
        'tversky': TverskyLoss,
        'lovasz_softmax': LovaszSoftmax,
        'lovasz_hinge': LovaszHinge,
        'boundary': BoundaryLoss,
        'auc_margin': AUCMarginLoss,
        'partial_auc': PartialAUCLoss,
        'composite_medical': CompositeMedicalLoss,
    }

    @classmethod
    def create(cls, config: Dict) -> nn.Module:
        """
        Create loss from config dict.

        Example config:
        {
            'type': 'compound',
            'components': [
                {'type': 'dice', 'weight': 0.5},
                {'type': 'focal', 'gamma': 2.0, 'weight': 0.3},
                {'type': 'lovasz_softmax', 'weight': 0.2}
            ],
            'learnable_weights': True
        }

        Args:
            config: Loss configuration dictionary

        Returns:
            Configured loss module
        """
        loss_type = config.get('type', 'cross_entropy')

        if loss_type == 'compound':
            losses = []
            weights = []
            names = []

            for comp in config['components']:
                comp = comp.copy()  # Don't modify original
                comp_type = comp.pop('type')
                weight = comp.pop('weight', 1.0)

                loss_cls = cls.REGISTRY.get(comp_type)
                if loss_cls is None:
                    raise ValueError(f"Unknown loss type: {comp_type}")

                losses.append(loss_cls(**comp))
                weights.append(weight)
                names.append(comp_type)

            return CompoundLoss(
                losses=losses,
                weights=weights,
                learnable_weights=config.get('learnable_weights', True),
                loss_names=names
            )
        else:
            loss_cls = cls.REGISTRY.get(loss_type)
            if loss_cls is None:
                raise ValueError(f"Unknown loss type: {loss_type}")

            # Remove 'type' from config and pass rest as kwargs
            config = {k: v for k, v in config.items() if k != 'type'}
            return loss_cls(**config)
