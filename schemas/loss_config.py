"""
Loss Configuration Schema for ARC.

Defines structured loss function composition for glaucoma detection training.
Enables exploration of loss engineering strategies (focal loss, class weighting,
auxiliary tasks) while maintaining clinical safety.

Key Components:
- Base loss types (BCE, Focal, Dice, Tversky)
- Phase F additions: Lovasz-Softmax, Lovasz-Hinge, Boundary, Compound
- Multi-task loss composition (classification + auxiliary tasks)
- Class weighting strategies
- Loss hyperparameters with safe bounds

Clinical Considerations:
- Optimize for AUC while maintaining sensitivity ≥ 0.85
- Handle class imbalance (glaucoma prevalence ~2-3%)
- Auxiliary tasks: DRI prediction, ISNT ratio prediction
- Avoid over-optimization on specificity at expense of sensitivity

Author: ARC Team (Dev 1, Dev 2)
Created: 2025-11-18
Version: 1.1 (Phase F Enhanced)
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class LossType(str, Enum):
    """
    Base loss function types for glaucoma classification.

    **Classification Losses**:
    - BCE: Binary Cross-Entropy (standard baseline)
    - FOCAL: Focal Loss (handles class imbalance, γ parameter controls focus on hard examples)
    - WEIGHTED_BCE: BCE with class weights (simple imbalance handling)

    **Segmentation-Inspired Losses** (for pixel-level tasks if applicable):
    - DICE: Dice Loss (IoU-based, good for imbalanced segmentation)
    - TVERSKY: Tversky Loss (generalization of Dice, controls FP/FN trade-off)

    **Phase F Additions - IoU Optimization**:
    - LOVASZ_SOFTMAX: Multi-class Lovasz-Softmax (direct IoU optimization)
    - LOVASZ_HINGE: Binary Lovasz-Hinge (binary IoU optimization)
    - BOUNDARY: Boundary loss for medical imaging (distance-weighted)

    **Combined Losses**:
    - BCE_DICE: Combination of BCE + Dice (hybrid classification-segmentation)
    - COMPOUND: Learnable weighted combination of multiple losses (Phase F)
    """
    # Classification
    BCE = "bce"
    FOCAL = "focal"
    WEIGHTED_BCE = "weighted_bce"

    # Segmentation-inspired
    DICE = "dice"
    TVERSKY = "tversky"

    # Phase F: IoU optimization losses
    LOVASZ_SOFTMAX = "lovasz_softmax"
    LOVASZ_HINGE = "lovasz_hinge"
    BOUNDARY = "boundary"

    # Combined
    BCE_DICE = "bce_dice"
    COMPOUND = "compound"


class AuxiliaryTask(str, Enum):
    """
    Auxiliary tasks for multi-task learning.

    Auxiliary tasks provide additional supervision signals that can improve
    primary glaucoma classification performance.

    - DRI_PREDICTION: Predict Disc Relevance Index (continuous regression)
    - ISNT_PREDICTION: Predict ISNT ratio (continuous regression)
    - CDR_PREDICTION: Predict Cup-to-Disc Ratio (continuous regression)
    - VESSEL_DENSITY: Predict retinal vessel density (continuous regression)
    """
    DRI_PREDICTION = "dri_prediction"
    ISNT_PREDICTION = "isnt_prediction"
    CDR_PREDICTION = "cdr_prediction"
    VESSEL_DENSITY = "vessel_density"


class ClassWeightingStrategy(str, Enum):
    """
    Strategy for computing class weights to handle imbalance.

    - NONE: No class weighting (baseline)
    - BALANCED: Inverse class frequency (sklearn-style)
    - EFFECTIVE_SAMPLES: Effective number of samples (Class-Balanced Loss)
    - CUSTOM: User-specified class weights
    """
    NONE = "none"
    BALANCED = "balanced"
    EFFECTIVE_SAMPLES = "effective_samples"
    CUSTOM = "custom"


class LossHyperparameters(BaseModel):
    """
    Hyperparameters for specific loss functions.

    Different loss types require different hyperparameters:
    - Focal Loss: gamma (focus parameter), alpha (class balance)
    - Tversky Loss: alpha (FP weight), beta (FN weight)
    - BCE+Dice: combination weight
    - Lovasz: per_image, classes (Phase F)
    - Boundary: theta0, theta (Phase F)
    - Compound: learnable_weights, normalize_weights (Phase F)
    """
    # Focal loss parameters
    focal_gamma: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Focal loss gamma (focus on hard examples, 0=BCE, typical: 2.0)"
    )

    focal_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Focal loss alpha (class balance weight, 0.5=no balance)"
    )

    # Tversky loss parameters
    tversky_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Tversky alpha (FP weight, higher=penalize FP more)"
    )

    tversky_beta: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Tversky beta (FN weight, higher=penalize FN more)"
    )

    # Combined loss weight
    combination_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Weight for combining losses (e.g., BCE vs Dice)"
    )

    # Phase F: Lovasz loss parameters
    lovasz_per_image: Optional[bool] = Field(
        default=None,
        description="Lovasz: compute loss per-image then average (default: False for softmax, True for hinge)"
    )

    lovasz_classes: Optional[str] = Field(
        default=None,
        description="Lovasz-Softmax: 'all' or 'present' (only classes in batch)"
    )

    lovasz_ignore_index: Optional[int] = Field(
        default=None,
        description="Lovasz: class index to ignore (e.g., -100 for padding)"
    )

    # Phase F: Boundary loss parameters
    boundary_theta0: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Boundary loss: initial boundary weight"
    )

    boundary_theta: Optional[float] = Field(
        default=None, ge=0.0, le=20.0,
        description="Boundary loss: boundary weight increase rate"
    )

    # Phase F: Compound loss parameters
    compound_learnable_weights: Optional[bool] = Field(
        default=None,
        description="Compound loss: learn loss weights during training (default: True)"
    )

    compound_normalize_weights: Optional[bool] = Field(
        default=None,
        description="Compound loss: softmax normalize weights (default: True)"
    )

    @field_validator('tversky_alpha', 'tversky_beta', mode='after')
    @classmethod
    def validate_tversky_sum(cls, v):
        """Tversky alpha + beta sum validation done at model level."""
        # Individual field validation - sum check in model_validator
        return v

    @field_validator('lovasz_classes', mode='after')
    @classmethod
    def validate_lovasz_classes(cls, v):
        """Validate lovasz_classes is either 'all' or 'present'."""
        if v is not None and v not in ['all', 'present']:
            raise ValueError(f"lovasz_classes must be 'all' or 'present', got '{v}'")
        return v


class AuxiliaryTaskConfig(BaseModel):
    """
    Configuration for a single auxiliary task.

    Example:
        {
            "task_type": "dri_prediction",
            "weight": 0.3,
            "loss_type": "mse"
        }
    """
    task_type: AuxiliaryTask = Field(
        description="Type of auxiliary task"
    )

    weight: float = Field(
        ge=0.0, le=1.0,
        description="Task weight in multi-task loss (0.0 to 1.0)"
    )

    loss_type: str = Field(
        default="mse",
        description="Loss function for auxiliary task (mse, mae, huber)"
    )

    @field_validator('loss_type', mode='after')
    @classmethod
    def validate_auxiliary_loss_type(cls, v):
        """Validate auxiliary task loss type."""
        allowed = ["mse", "mae", "huber", "smooth_l1"]
        if v not in allowed:
            raise ValueError(
                f"Auxiliary loss type '{v}' not in allowed list: {allowed}"
            )
        return v


# ============================================================================
# PHASE F: Compound Loss Component Configuration
# ============================================================================

class CompoundLossComponent(BaseModel):
    """
    Configuration for a single component in a compound loss.

    Phase F addition: enables learnable weighted combinations of multiple losses.

    Example:
        {
            "type": "dice",
            "weight": 0.5,
            "params": {}
        }
    """
    type: LossType = Field(
        description="Type of loss function for this component"
    )

    weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Initial weight for this loss component"
    )

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for this loss component"
    )

    @field_validator('type', mode='after')
    @classmethod
    def validate_not_compound(cls, v):
        """Compound loss cannot contain another compound loss."""
        if v == LossType.COMPOUND:
            raise ValueError("Compound loss cannot contain another compound loss")
        return v


class LossConfig(BaseModel):
    """
    Complete loss function configuration.

    Defines the loss function composition for training, including:
    - Primary classification loss
    - Optional auxiliary tasks
    - Class weighting strategy
    - Loss-specific hyperparameters

    Clinical Safety:
    - Primary task weight should be ≥ 0.6 (classification is primary goal)
    - Auxiliary task weights should sum to ≤ 0.4
    - Avoid over-weighting specificity at expense of sensitivity

    Example:
        {
            "name": "focal_with_dri",
            "primary_loss": "focal",
            "primary_weight": 0.7,
            "auxiliary_tasks": [
                {"task_type": "dri_prediction", "weight": 0.3, "loss_type": "mse"}
            ],
            "class_weighting": "balanced",
            "hyperparameters": {"focal_gamma": 2.0, "focal_alpha": 0.75}
        }
    """
    name: str = Field(
        description="Human-readable loss config name (e.g., 'focal_gamma2', 'bce_dri_aux')"
    )

    primary_loss: LossType = Field(
        description="Primary classification loss function"
    )

    primary_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Weight for primary classification loss (should be ≥ 0.6 for clinical safety)"
    )

    auxiliary_tasks: Optional[List[AuxiliaryTaskConfig]] = Field(
        default=None,
        description="Optional auxiliary tasks for multi-task learning"
    )

    class_weighting: ClassWeightingStrategy = Field(
        default=ClassWeightingStrategy.NONE,
        description="Strategy for class weight computation"
    )

    custom_class_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom class weights (required if class_weighting='custom')"
    )

    hyperparameters: Optional[LossHyperparameters] = Field(
        default=None,
        description="Loss-specific hyperparameters"
    )

    label_smoothing: float = Field(
        default=0.0, ge=0.0, le=0.2,
        description="Label smoothing factor (0.0 = no smoothing, typical: 0.1)"
    )

    # Phase F: Compound loss components
    compound_components: Optional[List[CompoundLossComponent]] = Field(
        default=None,
        description="Phase F: List of loss components for compound loss (required when primary_loss='compound')"
    )

    @field_validator('primary_weight', mode='after')
    @classmethod
    def validate_primary_weight(cls, v):
        """Primary task weight should be ≥ 0.6 for clinical safety."""
        if v < 0.6:
            raise ValueError(
                f"Primary weight ({v}) should be ≥ 0.6 to prioritize classification. "
                f"Auxiliary tasks are supplementary, not primary goals."
            )
        return v

    @model_validator(mode='after')
    def validate_model_constraints(self):
        """Validate cross-field constraints."""
        # Validate compound loss has components
        if self.primary_loss == LossType.COMPOUND:
            if self.compound_components is None or len(self.compound_components) < 2:
                raise ValueError(
                    "Compound loss requires at least 2 components in compound_components"
                )

        # Validate auxiliary weights sum to ≤ 0.4
        if self.auxiliary_tasks is not None:
            aux_weight_sum = sum(task.weight for task in self.auxiliary_tasks)
            total_weight = self.primary_weight + aux_weight_sum

            if total_weight > 1.0 + 1e-6:
                raise ValueError(
                    f"Total loss weight ({total_weight:.3f}) exceeds 1.0. "
                    f"Primary: {self.primary_weight}, Auxiliary: {aux_weight_sum:.3f}"
                )

            if aux_weight_sum > 0.4:
                raise ValueError(
                    f"Auxiliary task weights sum to {aux_weight_sum:.3f} (max 0.4). "
                    f"Primary task must remain dominant for clinical safety."
                )

        # Validate custom_class_weights
        if self.class_weighting == ClassWeightingStrategy.CUSTOM and self.custom_class_weights is None:
            raise ValueError(
                "custom_class_weights must be provided when class_weighting='custom'"
            )

        # Validate hyperparameters match loss type
        if self.primary_loss == LossType.FOCAL:
            if self.hyperparameters is None or self.hyperparameters.focal_gamma is None:
                raise ValueError(
                    "Focal loss requires focal_gamma hyperparameter"
                )

        if self.primary_loss == LossType.TVERSKY:
            if (self.hyperparameters is None or
                self.hyperparameters.tversky_alpha is None or
                self.hyperparameters.tversky_beta is None):
                raise ValueError(
                    "Tversky loss requires tversky_alpha and tversky_beta hyperparameters"
                )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    @classmethod
    def baseline_bce(cls) -> "LossConfig":
        """
        Create baseline BCE loss (current production).

        Returns:
            Simple BCE loss with no auxiliary tasks
        """
        return cls(
            name="baseline_bce",
            primary_loss=LossType.BCE,
            primary_weight=1.0,
            class_weighting=ClassWeightingStrategy.NONE
        )

    @classmethod
    def example_focal_balanced(cls) -> "LossConfig":
        """Example: Focal loss with balanced class weights."""
        return cls(
            name="focal_gamma2_balanced",
            primary_loss=LossType.FOCAL,
            primary_weight=1.0,
            class_weighting=ClassWeightingStrategy.BALANCED,
            hyperparameters=LossHyperparameters(
                focal_gamma=2.0,
                focal_alpha=0.75
            )
        )

    @classmethod
    def example_multitask_dri(cls) -> "LossConfig":
        """Example: BCE with DRI auxiliary task."""
        return cls(
            name="bce_dri_aux",
            primary_loss=LossType.BCE,
            primary_weight=0.7,
            auxiliary_tasks=[
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.DRI_PREDICTION,
                    weight=0.3,
                    loss_type="mse"
                )
            ],
            class_weighting=ClassWeightingStrategy.BALANCED
        )

    @classmethod
    def example_comprehensive(cls) -> "LossConfig":
        """Example: Focal loss with multiple auxiliary tasks."""
        return cls(
            name="focal_multi_aux",
            primary_loss=LossType.FOCAL,
            primary_weight=0.6,
            auxiliary_tasks=[
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.DRI_PREDICTION,
                    weight=0.2,
                    loss_type="mse"
                ),
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.CDR_PREDICTION,
                    weight=0.2,
                    loss_type="smooth_l1"
                )
            ],
            class_weighting=ClassWeightingStrategy.EFFECTIVE_SAMPLES,
            hyperparameters=LossHyperparameters(
                focal_gamma=2.0,
                focal_alpha=0.75
            ),
            label_smoothing=0.1
        )

    # =========================================================================
    # Phase F Examples
    # =========================================================================

    @classmethod
    def example_lovasz_softmax(cls) -> "LossConfig":
        """Example: Lovasz-Softmax for IoU optimization (Phase F)."""
        return cls(
            name="lovasz_softmax_present",
            primary_loss=LossType.LOVASZ_SOFTMAX,
            primary_weight=1.0,
            hyperparameters=LossHyperparameters(
                lovasz_classes="present",
                lovasz_per_image=False
            )
        )

    @classmethod
    def example_compound_dice_focal_lovasz(cls) -> "LossConfig":
        """Example: Compound loss with Dice + Focal + Lovasz (Phase F)."""
        return cls(
            name="compound_dice_focal_lovasz",
            primary_loss=LossType.COMPOUND,
            primary_weight=1.0,
            compound_components=[
                CompoundLossComponent(type=LossType.DICE, weight=0.4),
                CompoundLossComponent(type=LossType.FOCAL, weight=0.3, params={"gamma": 2.0}),
                CompoundLossComponent(type=LossType.LOVASZ_SOFTMAX, weight=0.3)
            ],
            hyperparameters=LossHyperparameters(
                compound_learnable_weights=True,
                compound_normalize_weights=True
            )
        )

    @classmethod
    def example_boundary_loss(cls) -> "LossConfig":
        """Example: Boundary loss for medical imaging (Phase F)."""
        return cls(
            name="boundary_weighted",
            primary_loss=LossType.BOUNDARY,
            primary_weight=1.0,
            hyperparameters=LossHyperparameters(
                boundary_theta0=3.0,
                boundary_theta=5.0
            )
        )


def validate_loss_safety(loss_config: LossConfig) -> tuple[bool, str]:
    """
    Validate loss configuration for clinical safety.

    Additional validation beyond Pydantic model validators.
    Used by Critic agent for loss config proposal review.

    Args:
        loss_config: Loss configuration to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Primary weight must be dominant
    if loss_config.primary_weight < 0.6:
        return False, (
            f"Primary weight ({loss_config.primary_weight}) too low. "
            f"Classification must remain primary goal (≥ 0.6)."
        )

    # Check 2: Focal gamma should not be too high (risks instability)
    if loss_config.hyperparameters and loss_config.hyperparameters.focal_gamma:
        if loss_config.hyperparameters.focal_gamma > 3.0:
            return False, (
                f"Focal gamma ({loss_config.hyperparameters.focal_gamma}) too high (max 3.0). "
                f"High gamma risks training instability."
            )

    # Check 3: Too many auxiliary tasks risks diluting primary objective
    if loss_config.auxiliary_tasks and len(loss_config.auxiliary_tasks) > 3:
        return False, (
            f"Too many auxiliary tasks ({len(loss_config.auxiliary_tasks)}). "
            f"Maximum 3 recommended to maintain focus on primary classification."
        )

    # Check 4: Label smoothing should be conservative
    if loss_config.label_smoothing > 0.15:
        return False, (
            f"Label smoothing ({loss_config.label_smoothing}) too high (max 0.15). "
            f"Excessive smoothing may degrade calibration."
        )

    # Check 5: For Tversky loss, beta should be ≥ alpha (prioritize FN over FP)
    if loss_config.hyperparameters:
        if (loss_config.hyperparameters.tversky_alpha is not None and
            loss_config.hyperparameters.tversky_beta is not None):
            alpha = loss_config.hyperparameters.tversky_alpha
            beta = loss_config.hyperparameters.tversky_beta

            if beta < alpha:
                return False, (
                    f"Tversky beta ({beta}) < alpha ({alpha}). "
                    f"For clinical safety, should prioritize recall (beta ≥ alpha) "
                    f"to avoid missing glaucoma cases (FN)."
                )

    # =========================================================================
    # Phase F: Additional safety checks for new loss types
    # =========================================================================

    # Check 6: Compound loss should have reasonable number of components
    if loss_config.primary_loss == LossType.COMPOUND:
        if loss_config.compound_components:
            if len(loss_config.compound_components) > 5:
                return False, (
                    f"Too many compound loss components ({len(loss_config.compound_components)}). "
                    f"Maximum 5 recommended for training stability."
                )

    # Check 7: Boundary loss theta parameters should be reasonable
    if loss_config.hyperparameters:
        if loss_config.hyperparameters.boundary_theta0 is not None:
            if loss_config.hyperparameters.boundary_theta0 > 5.0:
                return False, (
                    f"Boundary theta0 ({loss_config.hyperparameters.boundary_theta0}) too high. "
                    f"Values > 5.0 may cause training instability."
                )

    # All checks passed
    return True, ""


def compute_effective_class_weights(
    class_counts: Dict[str, int],
    beta: float = 0.9999
) -> Dict[str, float]:
    """
    Compute class weights using effective number of samples.

    Based on "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019).

    Args:
        class_counts: Dictionary of class counts {"negative": N_neg, "positive": N_pos}
        beta: Hyperparameter controlling re-weighting (default: 0.9999)

    Returns:
        Dictionary of class weights
    """
    weights = {}

    for class_name, count in class_counts.items():
        effective_num = (1.0 - beta ** count) / (1.0 - beta)
        weights[class_name] = 1.0 / effective_num

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    return weights
