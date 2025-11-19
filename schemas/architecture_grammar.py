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
from pydantic import BaseModel, Field, validator


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

    @validator('embed_dim')
    def validate_embed_divisibility(cls, v, values):
        """Ensure embed_dim is divisible by num_heads."""
        num_heads = values.get('num_heads', 1)
        if v % num_heads != 0:
            raise ValueError(
                f"embed_dim ({v}) must be divisible by num_heads ({num_heads})"
            )
        return v


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

    @validator('attention_config', always=True)
    def validate_attention_requirements(cls, v, values):
        """
        Validate attention_config requirements based on backbone and fusion type.

        Rules:
        - ViT backbones (DEIT, VIT) REQUIRE attention_config
        - Cross-attention fusion REQUIRES attention_config
        - Other combinations: attention_config is optional
        """
        backbone = values.get('backbone')
        fusion_type = values.get('fusion_type')

        # ViT backbones require attention config
        if backbone in [BackboneType.DEIT_SMALL, BackboneType.VIT_BASE]:
            if v is None:
                raise ValueError(
                    f"{backbone} backbone requires attention_config with num_heads, embed_dim, and depth"
                )
            if v.depth is None:
                raise ValueError(
                    f"ViT backbones require attention_config.depth to be specified"
                )

        # Cross-attention fusion requires attention config
        if fusion_type == FusionType.CROSS_ATTENTION:
            if v is None:
                raise ValueError(
                    "cross_attention fusion requires attention_config with num_heads and embed_dim"
                )

        return v

    @validator('fusion_config', always=True)
    def validate_fusion_config(cls, v, values):
        """
        Validate fusion_config parameters match fusion_type requirements.

        Rules:
        - LATE fusion: should have hidden_dims
        - FILM fusion: should have num_film_layers
        - GATED fusion: should have gating_hidden_dim
        - CROSS_ATTENTION: inherits from attention_config
        """
        fusion_type = values.get('fusion_type')

        if fusion_type == FusionType.LATE:
            if v.hidden_dims is None:
                # Provide default MLP architecture
                v.hidden_dims = [256, 128]

        elif fusion_type == FusionType.FILM:
            if v.num_film_layers is None:
                # Provide default number of FiLM layers
                v.num_film_layers = 3

        elif fusion_type == FusionType.GATED:
            if v.gating_hidden_dim is None:
                # Provide default gating network hidden dim
                v.gating_hidden_dim = 128

        return v

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
