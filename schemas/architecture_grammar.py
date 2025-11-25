"""
Architecture Grammar Schema for ARC.

Defines the valid architecture design space as composable components for
glaucoma detection models. Enables structured architecture search beyond
fixed templates.

Key Components:
- Fusion strategies for combining CNN features + clinical indicators
- Backbone architectures (CNN, ViT)
- Attention mechanism specifications
- Composition validation rules

Author: ARC Team (Dev 1)
Created: 2025-11-18
Version: 1.0
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class FusionType(str, Enum):
    """
    Fusion strategies for combining visual features and clinical indicators.

    - LATE: Concatenate CNN embedding + clinical vector → MLP (baseline)
    - FILM: Feature-wise Linear Modulation (clinical indicators modulate CNN features)
    - CROSS_ATTENTION: Clinical indicators as queries, CNN features as keys/values
    - GATED: Learned soft gates weight CNN vs clinical contribution per-sample
    """
    LATE = "late"
    FILM = "film"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"


class BackboneType(str, Enum):
    """
    Visual feature extraction backbone architectures.

    - EFFICIENTNET_B3: Current baseline, efficient CNN (default)
    - CONVNEXT_TINY: Modern ConvNet architecture
    - CONVNEXT_SMALL: Larger ConvNeXt variant
    - DEIT_SMALL: Vision Transformer (requires attention_config)
    - VIT_BASE: Larger Vision Transformer
    """
    EFFICIENTNET_B3 = "efficientnet_b3"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    DEIT_SMALL = "deit_small"
    VIT_BASE = "vit_base"


class PretrainedWeights(str, Enum):
    """
    Pretrained weight sources for transfer learning.

    - IMAGENET: Standard ImageNet-1K pretrained weights
    - MEDICAL: Medical imaging pretrained (CheXpert, MIMIC-CXR, etc.)
    - NONE: Random initialization
    """
    IMAGENET = "imagenet"
    MEDICAL = "medical"
    NONE = "none"


class AttentionConfig(BaseModel):
    """
    Attention mechanism configuration for transformer backbones and fusion.

    Required for ViT-based backbones (DEIT_SMALL, VIT_BASE).
    Optional for attention-based fusion (CROSS_ATTENTION).
    """
    num_heads: int = Field(
        ge=1, le=16,
        description="Number of attention heads (typically 4, 8, or 12)"
    )

    embed_dim: int = Field(
        ge=64, le=1024,
        description="Embedding dimension (must be divisible by num_heads)"
    )

    depth: Optional[int] = Field(
        default=None, ge=1, le=24,
        description="Number of transformer blocks (ViT only)"
    )

    dropout: float = Field(
        default=0.0, ge=0.0, le=0.5,
        description="Attention dropout rate"
    )

    @model_validator(mode='after')
    def validate_embed_divisibility(self):
        """Ensure embed_dim is divisible by num_heads."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self


class FusionConfig(BaseModel):
    """
    Configuration for fusion module hyperparameters.

    Different fusion types require different parameters:
    - LATE: hidden_dims for MLP head
    - FILM: num_film_layers
    - CROSS_ATTENTION: attention_config
    - GATED: gating_hidden_dim
    """
    hidden_dims: Optional[List[int]] = Field(
        default=None,
        description="MLP hidden layer dimensions for late fusion"
    )

    num_film_layers: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Number of FiLM conditioning layers"
    )

    gating_hidden_dim: Optional[int] = Field(
        default=None, ge=16, le=512,
        description="Hidden dimension for gating network"
    )

    dropout: float = Field(
        default=0.1, ge=0.0, le=0.5,
        description="Fusion module dropout rate"
    )


class ArchitectureGrammar(BaseModel):
    """
    Complete architecture specification using structured grammar.

    Replaces free-form architecture configs with validated, composable design.
    Enables ARC to explore beyond fixed templates.

    Example:
        {
            "fusion_type": "film",
            "backbone": "convnext_tiny",
            "pretrained": "medical",
            "attention_config": null,
            "fusion_config": {"num_film_layers": 3, "dropout": 0.1}
        }
    """
    fusion_type: FusionType = Field(
        description="Fusion strategy for visual + clinical features"
    )

    backbone: BackboneType = Field(
        description="Visual feature extraction backbone"
    )

    pretrained: PretrainedWeights = Field(
        default=PretrainedWeights.IMAGENET,
        description="Pretrained weight initialization"
    )

    attention_config: Optional[AttentionConfig] = Field(
        default=None,
        description="Attention parameters (required for ViT backbones and cross-attention fusion)"
    )

    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig,
        description="Fusion module hyperparameters"
    )

    num_clinical_features: int = Field(
        default=4,
        ge=1, le=20,
        description="Number of clinical indicator inputs (CDR, ISNT, vessel density, entropy)"
    )

    num_classes: int = Field(
        default=1,
        ge=1, le=10,
        description="Number of output classes (1 for binary glaucoma classification)"
    )

    @model_validator(mode='after')
    def validate_architecture_constraints(self):
        """
        Validate architecture constraints.

        Rules:
        - ViT backbones (DEIT, VIT) REQUIRE attention_config
        - Cross-attention fusion REQUIRES attention_config
        - Fusion config parameters match fusion_type requirements
        """
        # ViT backbones require attention config
        if self.backbone in [BackboneType.DEIT_SMALL, BackboneType.VIT_BASE]:
            if self.attention_config is None:
                raise ValueError(
                    f"{self.backbone} backbone requires attention_config with num_heads, embed_dim, and depth"
                )
            if self.attention_config.depth is None:
                raise ValueError(
                    f"ViT backbones require attention_config.depth to be specified"
                )

        # Cross-attention fusion requires attention config
        if self.fusion_type == FusionType.CROSS_ATTENTION:
            if self.attention_config is None:
                raise ValueError(
                    "cross_attention fusion requires attention_config with num_heads and embed_dim"
                )

        # Validate and set default fusion_config values
        if self.fusion_config is not None:
            if self.fusion_type == FusionType.LATE:
                if self.fusion_config.hidden_dims is None:
                    self.fusion_config.hidden_dims = [256, 128]

            elif self.fusion_type == FusionType.FILM:
                if self.fusion_config.num_film_layers is None:
                    self.fusion_config.num_film_layers = 3

            elif self.fusion_type == FusionType.GATED:
                if self.fusion_config.gating_hidden_dim is None:
                    self.fusion_config.gating_hidden_dim = 128

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    @classmethod
    def from_baseline(cls) -> "ArchitectureGrammar":
        """
        Create baseline architecture (current production config).

        Returns:
            ArchitectureGrammar with late fusion + EfficientNet-B3
        """
        return cls(
            fusion_type=FusionType.LATE,
            backbone=BackboneType.EFFICIENTNET_B3,
            pretrained=PretrainedWeights.IMAGENET,
            attention_config=None,
            fusion_config=FusionConfig(
                hidden_dims=[256, 128],
                dropout=0.1
            )
        )

    @classmethod
    def example_film(cls) -> "ArchitectureGrammar":
        """Example: FiLM fusion with ConvNeXt backbone."""
        return cls(
            fusion_type=FusionType.FILM,
            backbone=BackboneType.CONVNEXT_TINY,
            pretrained=PretrainedWeights.MEDICAL,
            attention_config=None,
            fusion_config=FusionConfig(
                num_film_layers=3,
                dropout=0.1
            )
        )

    @classmethod
    def example_cross_attention(cls) -> "ArchitectureGrammar":
        """Example: Cross-attention fusion with ViT backbone."""
        return cls(
            fusion_type=FusionType.CROSS_ATTENTION,
            backbone=BackboneType.DEIT_SMALL,
            pretrained=PretrainedWeights.IMAGENET,
            attention_config=AttentionConfig(
                num_heads=8,
                embed_dim=384,
                depth=12,
                dropout=0.0
            ),
            fusion_config=FusionConfig(dropout=0.1)
        )


class ArchitectureFamily(str, Enum):
    """
    High-level architecture families for clustering experiments.

    Used by Historian to track which architectural approaches work best.
    """
    CNN_LATE_FUSION = "cnn_late_fusion"
    CNN_FILM = "cnn_film"
    CNN_GATED = "cnn_gated"
    VIT_CROSS_ATTENTION = "vit_cross_attention"
    VIT_LATE_FUSION = "vit_late_fusion"
    HYBRID = "hybrid"


def classify_architecture_family(grammar: ArchitectureGrammar) -> ArchitectureFamily:
    """
    Classify architecture into high-level family for clustering.

    Args:
        grammar: Architecture grammar specification

    Returns:
        ArchitectureFamily classification
    """
    is_vit = grammar.backbone in [BackboneType.DEIT_SMALL, BackboneType.VIT_BASE]
    fusion = grammar.fusion_type

    if is_vit and fusion == FusionType.CROSS_ATTENTION:
        return ArchitectureFamily.VIT_CROSS_ATTENTION
    elif is_vit and fusion == FusionType.LATE:
        return ArchitectureFamily.VIT_LATE_FUSION
    elif not is_vit and fusion == FusionType.FILM:
        return ArchitectureFamily.CNN_FILM
    elif not is_vit and fusion == FusionType.GATED:
        return ArchitectureFamily.CNN_GATED
    elif not is_vit and fusion == FusionType.LATE:
        return ArchitectureFamily.CNN_LATE_FUSION
    else:
        return ArchitectureFamily.HYBRID


def validate_grammar_compatibility(grammar: ArchitectureGrammar) -> tuple[bool, str]:
    """
    Validate architecture grammar for clinical safety and feasibility.

    Additional validation beyond Pydantic model validators.
    Used by Critic agent for proposal review.

    Args:
        grammar: Architecture grammar to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: ViT backbones require sufficient GPU memory
    if grammar.backbone in [BackboneType.VIT_BASE] and grammar.attention_config:
        if grammar.attention_config.embed_dim > 768:
            return False, "VIT_BASE with embed_dim > 768 may exceed GPU memory"

    # Check 2: FiLM with too many layers risks overfitting
    if grammar.fusion_type == FusionType.FILM and grammar.fusion_config.num_film_layers:
        if grammar.fusion_config.num_film_layers > 5:
            return False, "num_film_layers > 5 risks overfitting on small datasets"

    # Check 3: Cross-attention requires compatible dimensions
    if grammar.fusion_type == FusionType.CROSS_ATTENTION:
        if grammar.attention_config is None:
            return False, "Cross-attention fusion requires attention_config"

        # Clinical features must be compatible with embed_dim
        if grammar.num_clinical_features > grammar.attention_config.embed_dim:
            return False, (
                f"num_clinical_features ({grammar.num_clinical_features}) "
                f"exceeds attention embed_dim ({grammar.attention_config.embed_dim})"
            )

    # Check 4: Late fusion MLP should not be too deep
    if grammar.fusion_type == FusionType.LATE and grammar.fusion_config.hidden_dims:
        if len(grammar.fusion_config.hidden_dims) > 4:
            return False, "Late fusion MLP depth > 4 layers risks overfitting"

    return True, ""


# ============================================================================
# PHASE F: DARTS (Differentiable Architecture Search) Configuration
# ============================================================================

class DARTSPrimitive(str, Enum):
    """
    Available DARTS primitives for architecture search.

    Phase F addition: enables gradient-based NAS.
    """
    NONE = "none"
    SKIP_CONNECT = "skip_connect"
    AVG_POOL_3X3 = "avg_pool_3x3"
    MAX_POOL_3X3 = "max_pool_3x3"
    SEP_CONV_3X3 = "sep_conv_3x3"
    SEP_CONV_5X5 = "sep_conv_5x5"
    DIL_CONV_3X3 = "dil_conv_3x3"
    DIL_CONV_5X5 = "dil_conv_5x5"


# Predefined primitive sets
DARTS_STANDARD_PRIMITIVES = [
    DARTSPrimitive.NONE,
    DARTSPrimitive.SKIP_CONNECT,
    DARTSPrimitive.AVG_POOL_3X3,
    DARTSPrimitive.MAX_POOL_3X3,
    DARTSPrimitive.SEP_CONV_3X3,
    DARTSPrimitive.SEP_CONV_5X5,
    DARTSPrimitive.DIL_CONV_3X3,
    DARTSPrimitive.DIL_CONV_5X5,
]

# Medical-safe primitives (no aggressive pooling)
DARTS_MEDICAL_PRIMITIVES = [
    DARTSPrimitive.NONE,
    DARTSPrimitive.SKIP_CONNECT,
    DARTSPrimitive.SEP_CONV_3X3,
    DARTSPrimitive.SEP_CONV_5X5,
    DARTSPrimitive.DIL_CONV_3X3,
    DARTSPrimitive.DIL_CONV_5X5,
]


class DARTSSearchConfig(BaseModel):
    """
    Configuration for DARTS architecture search.

    Phase F addition: enables differentiable neural architecture search
    within ARC's constraints.
    """

    enabled: bool = Field(
        default=False,
        description="Enable DARTS search for this experiment"
    )

    # Cell configuration
    init_channels: int = Field(
        default=16, ge=8, le=64,
        description="Initial channel count"
    )

    num_cells: int = Field(
        default=8, ge=4, le=20,
        description="Number of cells in the network"
    )

    num_nodes: int = Field(
        default=4, ge=2, le=6,
        description="Number of intermediate nodes per cell"
    )

    # Operation set
    primitives: List[DARTSPrimitive] = Field(
        default_factory=lambda: DARTS_MEDICAL_PRIMITIVES.copy(),
        description="Operations to search over"
    )

    use_medical_safe: bool = Field(
        default=True,
        description="Use medical-safe primitives (excludes aggressive pooling)"
    )

    # Training configuration
    search_epochs: int = Field(
        default=50, ge=10,
        description="Number of architecture search epochs"
    )

    arch_learning_rate: float = Field(
        default=3e-4, gt=0,
        description="Learning rate for architecture parameters"
    )

    arch_weight_decay: float = Field(
        default=1e-3, ge=0,
        description="Weight decay for architecture parameters"
    )

    # Constraints (from ARC)
    max_params: int = Field(
        default=10_000_000,
        description="Maximum parameter count (10M for ARC)"
    )

    max_memory_gb: float = Field(
        default=5.0,
        description="Maximum GPU memory in GB"
    )

    @field_validator('primitives', mode='after')
    @classmethod
    def validate_primitives(cls, v):
        """Ensure primitives are valid."""
        if v is not None and len(v) < 2:
            raise ValueError("Need at least 2 primitives for meaningful search")
        return v

    @model_validator(mode='after')
    def apply_medical_safe_defaults(self):
        """Apply medical-safe primitives if flag is set and primitives not provided."""
        if self.use_medical_safe and self.primitives is None:
            self.primitives = DARTS_MEDICAL_PRIMITIVES.copy()
        return self


class DARTSGenotype(BaseModel):
    """
    Discovered architecture genotype from DARTS search.

    Represents the discrete architecture extracted from
    continuous architecture weights.
    """

    normal: List[tuple] = Field(
        description="Normal cell genotype: list of (operation, source_node) tuples"
    )

    reduce: List[tuple] = Field(
        description="Reduction cell genotype: list of (operation, source_node) tuples"
    )

    search_config: Optional[DARTSSearchConfig] = Field(
        default=None,
        description="Configuration used for the search"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'normal': [(op, int(src)) for op, src in self.normal],
            'reduce': [(op, int(src)) for op, src in self.reduce],
            'search_config': self.search_config.dict() if self.search_config else None
        }


# Extend ArchitectureGrammar to optionally include DARTS config
class ExtendedArchitectureGrammar(ArchitectureGrammar):
    """
    Extended architecture grammar with DARTS search support.

    Phase F addition: adds DARTS NAS configuration.
    """

    darts_config: Optional[DARTSSearchConfig] = Field(
        default=None,
        description="DARTS architecture search configuration (Phase F)"
    )

    discovered_genotype: Optional[DARTSGenotype] = Field(
        default=None,
        description="Genotype discovered from DARTS search"
    )

    @classmethod
    def example_darts_search(cls) -> "ExtendedArchitectureGrammar":
        """Example: DARTS search with medical-safe primitives."""
        return cls(
            fusion_type=FusionType.LATE,
            backbone=BackboneType.EFFICIENTNET_B3,
            pretrained=PretrainedWeights.IMAGENET,
            fusion_config=FusionConfig(hidden_dims=[256, 128]),
            darts_config=DARTSSearchConfig(
                enabled=True,
                init_channels=16,
                num_cells=8,
                use_medical_safe=True,
                search_epochs=50
            )
        )


def validate_darts_config(config: DARTSSearchConfig) -> tuple[bool, str]:
    """
    Validate DARTS configuration for ARC constraints.

    Args:
        config: DARTS search configuration

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Parameter budget
    estimated_params = config.init_channels * config.num_cells * config.num_nodes * 10000
    if estimated_params > config.max_params * 1.5:
        return False, (
            f"Estimated params ({estimated_params:,}) may exceed limit ({config.max_params:,}). "
            f"Reduce init_channels or num_cells."
        )

    # Check 2: Search epochs should be reasonable
    if config.search_epochs < 20:
        return False, (
            f"search_epochs ({config.search_epochs}) too low. "
            f"Need at least 20 epochs for meaningful architecture search."
        )

    # Check 3: Medical-safe check
    if config.use_medical_safe:
        unsafe_ops = {DARTSPrimitive.AVG_POOL_3X3, DARTSPrimitive.MAX_POOL_3X3}
        for prim in config.primitives:
            if prim in unsafe_ops:
                return False, (
                    f"use_medical_safe=True but primitives include {prim.value}. "
                    f"Pooling operations may lose important detail in medical images."
                )

    return True, ""
