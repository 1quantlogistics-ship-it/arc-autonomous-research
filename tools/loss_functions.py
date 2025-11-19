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

Author: ARC Team (Dev 1)
Created: 2025-11-19
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


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
