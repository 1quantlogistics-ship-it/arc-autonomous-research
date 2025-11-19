"""
Experiment Engine Schemas

Pydantic models for experiment specifications, training jobs, and results.
These schemas ensure type safety for the autonomous experiment execution system.

Integration:
- Used by AcuVue tool interfaces for validation
- Integrated with ToolGovernance for constraint checking
- Compatible with Control Plane schema validation
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from pathlib import Path

# Import architecture grammar (Phase E)
try:
    from schemas.architecture_grammar import ArchitectureGrammar
    ARCHITECTURE_GRAMMAR_AVAILABLE = True
except ImportError:
    ARCHITECTURE_GRAMMAR_AVAILABLE = False


# ============================================================================
# Enumerations
# ============================================================================

class JobStatus(str, Enum):
    """Training job status states."""
    QUEUED = "queued"          # Waiting in scheduler queue
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Execution failed
    CANCELLED = "cancelled"    # User/supervisor cancelled
    TIMEOUT = "timeout"        # Exceeded time limit
    OOM = "oom"               # Out of memory error


class GPUAllocation(str, Enum):
    """GPU assignment for jobs."""
    GPU0 = "gpu0"              # Primary experiment GPU
    GPU1 = "gpu1"              # Secondary experiment GPU
    GPU2 = "gpu2"              # ARC + light workers
    AUTO = "auto"              # Scheduler decides


class ArchitectureFamily(str, Enum):
    """Supported model architecture families."""
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    VIT = "vit"                # Vision Transformer
    DENSENET = "densenet"
    INCEPTION = "inception"
    CUSTOM = "custom"


class DatasetType(str, Enum):
    """Dataset categories."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    CROSS_VAL = "cross_val"


class PreprocessingType(str, Enum):
    """Preprocessing operation types."""
    NORMALIZE = "normalize"
    RESIZE = "resize"
    AUGMENT = "augment"
    CROP = "crop"
    FILTER = "filter"
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Evaluation metric types."""
    AUC = "auc"
    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    LOSS = "loss"


# ============================================================================
# Preprocessing Schemas
# ============================================================================

class PreprocessingStep(BaseModel):
    """Individual preprocessing operation."""
    type: PreprocessingType
    params: Dict[str, Any] = Field(default_factory=dict)
    order: int = Field(ge=0, description="Execution order (lower = earlier)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "normalize",
                "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "order": 1
            }
        }
    )


class PreprocessingChain(BaseModel):
    """Complete preprocessing pipeline."""
    chain_id: str = Field(description="Unique preprocessing chain identifier")
    steps: List[PreprocessingStep] = Field(default_factory=list)
    description: Optional[str] = None

    @field_validator('steps')
    @classmethod
    def validate_steps_order(cls, v: List[PreprocessingStep]) -> List[PreprocessingStep]:
        """Ensure steps are properly ordered."""
        if len(v) > 0:
            orders = [step.order for step in v]
            if len(orders) != len(set(orders)):
                raise ValueError("Preprocessing steps must have unique order values")
        return sorted(v, key=lambda x: x.order)


# ============================================================================
# Hyperparameter Schemas
# ============================================================================

class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    type: Literal["adam", "sgd", "adamw", "rmsprop"] = "adam"
    learning_rate: float = Field(gt=0, le=1.0, description="Learning rate")
    weight_decay: float = Field(ge=0, le=0.1, default=0.0)
    momentum: Optional[float] = Field(ge=0, le=1.0, default=None)
    beta1: Optional[float] = Field(ge=0, le=1.0, default=0.9)
    beta2: Optional[float] = Field(ge=0, le=1.0, default=0.999)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "beta1": 0.9,
                "beta2": 0.999
            }
        }
    )


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: Literal["step", "cosine", "plateau", "none"] = "step"
    step_size: Optional[int] = Field(ge=1, default=None)
    gamma: Optional[float] = Field(gt=0, le=1.0, default=None)
    patience: Optional[int] = Field(ge=1, default=None)

    @field_validator('step_size')
    @classmethod
    def validate_step_scheduler(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure step scheduler has step_size."""
        if info.data.get('type') == 'step' and v is None:
            raise ValueError("step scheduler requires step_size")
        return v


class HyperparameterConfig(BaseModel):
    """Complete hyperparameter configuration."""
    batch_size: int = Field(ge=1, le=512, default=32)
    epochs: int = Field(ge=1, le=500, default=100)
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    early_stopping_patience: Optional[int] = Field(ge=1, default=10)
    gradient_clip: Optional[float] = Field(gt=0, default=None)
    mixed_precision: bool = Field(default=False, description="Use mixed precision training")


# ============================================================================
# Architecture Schemas
# ============================================================================

class ArchitectureConfig(BaseModel):
    """Model architecture specification."""
    family: ArchitectureFamily
    variant: str = Field(description="Specific variant (e.g., 'resnet50', 'efficientnet-b0')")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    num_classes: int = Field(ge=1, description="Number of output classes")
    dropout: float = Field(ge=0, le=0.9, default=0.5)
    custom_config: Optional[Dict[str, Any]] = Field(default=None, description="Architecture-specific config")

    # Phase E: Architecture Grammar support
    grammar: Optional["ArchitectureGrammar"] = Field(
        default=None,
        description="Structured architecture grammar (Phase E). If provided, overrides family/variant."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "family": "resnet",
                "variant": "resnet50",
                "pretrained": True,
                "num_classes": 2,
                "dropout": 0.5
            }
        }
    )

    @field_validator('grammar')
    @classmethod
    def validate_grammar_available(cls, v):
        """Ensure grammar is available if specified."""
        if v is not None and not ARCHITECTURE_GRAMMAR_AVAILABLE:
            raise ValueError("ArchitectureGrammar not available - install Phase E schemas")
        return v


# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetConfig(BaseModel):
    """Dataset specification."""
    dataset_id: str = Field(description="Unique dataset identifier")
    dataset_type: DatasetType
    data_path: str = Field(description="Path to dataset")
    preprocessing_chain_id: Optional[str] = None
    train_split: float = Field(ge=0.1, le=0.9, default=0.8)
    val_split: float = Field(ge=0.05, le=0.5, default=0.1)
    test_split: float = Field(ge=0.05, le=0.5, default=0.1)

    @field_validator('test_split')
    @classmethod
    def validate_splits(cls, v: float, info) -> float:
        """Ensure splits sum to 1.0."""
        train = info.data.get('train_split', 0.8)
        val = info.data.get('val_split', 0.1)
        total = train + val + v
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Dataset splits must sum to 1.0 (got {total})")
        return v


# ============================================================================
# Experiment Specification Schema
# ============================================================================

class ExperimentSpec(BaseModel):
    """Complete experiment specification.

    This is the core schema for defining autonomous research experiments.
    It combines architecture, hyperparameters, datasets, and preprocessing.
    """
    experiment_id: str = Field(description="Unique experiment identifier")
    description: str = Field(description="Human-readable experiment description")

    # Core configuration
    architecture: ArchitectureConfig
    hyperparameters: HyperparameterConfig
    dataset: DatasetConfig
    preprocessing: Optional[PreprocessingChain] = None

    # Metadata
    cycle_id: int = Field(ge=0, description="Research cycle that generated this experiment")
    novelty_class: Literal["exploit", "explore", "wildcat"] = "exploit"
    priority: int = Field(ge=0, le=10, default=5, description="Scheduling priority (10 = highest)")

    # Resource constraints
    max_training_time: int = Field(ge=60, le=86400, default=3600, description="Max training time in seconds")
    gpu_allocation: GPUAllocation = GPUAllocation.AUTO

    # Tags and organization
    tags: List[str] = Field(default_factory=list)
    parent_experiment_id: Optional[str] = Field(None, description="Parent experiment if this is a variant")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_id": "exp_001_resnet50_lr0.001",
                "description": "ResNet50 baseline with Adam optimizer",
                "architecture": {
                    "family": "resnet",
                    "variant": "resnet50",
                    "pretrained": True,
                    "num_classes": 2,
                    "dropout": 0.5
                },
                "hyperparameters": {
                    "batch_size": 32,
                    "epochs": 100,
                    "optimizer": {
                        "type": "adam",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0001
                    }
                },
                "dataset": {
                    "dataset_id": "acuvue_v1",
                    "dataset_type": "train",
                    "data_path": "/data/acuvue",
                    "train_split": 0.8,
                    "val_split": 0.1,
                    "test_split": 0.1
                },
                "cycle_id": 1,
                "novelty_class": "exploit",
                "priority": 5,
                "max_training_time": 3600
            }
        }
    )


# ============================================================================
# Training Job Schema
# ============================================================================

class TrainingJobConfig(BaseModel):
    """Training job runtime configuration."""
    job_id: str = Field(description="Unique job identifier (usually same as experiment_id)")
    experiment_spec: ExperimentSpec

    # Runtime settings
    gpu_id: Optional[int] = Field(None, ge=0, le=7, description="Assigned GPU ID")
    checkpoint_dir: str = Field(description="Directory for saving checkpoints")
    log_dir: str = Field(description="Directory for training logs")

    # Execution control
    resume_from_checkpoint: Optional[str] = Field(None, description="Path to checkpoint to resume from")
    save_checkpoints: bool = Field(default=True)
    checkpoint_interval: int = Field(ge=1, default=10, description="Save checkpoint every N epochs")

    # Monitoring
    log_interval: int = Field(ge=1, default=100, description="Log metrics every N batches")
    validation_interval: int = Field(ge=1, default=1, description="Validate every N epochs")

    # Safety
    max_retries: int = Field(ge=0, le=3, default=1, description="Retry count on failure")
    timeout: int = Field(ge=60, description="Job timeout in seconds")

    # Metadata
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    submitted_by: str = Field(default="arc_orchestrator")

    model_config = ConfigDict(validate_assignment=True)


# ============================================================================
# Results Schemas
# ============================================================================

class MetricResult(BaseModel):
    """Individual metric result."""
    metric_type: MetricType
    value: float
    epoch: int = Field(ge=0)
    split: DatasetType = DatasetType.VAL

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metric_type": "auc",
                "value": 0.872,
                "epoch": 50,
                "split": "val"
            }
        }
    )


class CheckpointInfo(BaseModel):
    """Checkpoint metadata."""
    checkpoint_path: str
    epoch: int = Field(ge=0)
    metrics: List[MetricResult] = Field(default_factory=list)
    is_best: bool = Field(default=False)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ExperimentResult(BaseModel):
    """Complete experiment execution results.

    This schema captures all outcomes from a training job,
    including metrics, checkpoints, and execution metadata.
    """
    experiment_id: str
    job_id: str

    # Execution status
    status: JobStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = Field(None, ge=0)

    # Results
    metrics: List[MetricResult] = Field(default_factory=list)
    best_metrics: Dict[str, float] = Field(default_factory=dict, description="Best value per metric")
    final_metrics: Dict[str, float] = Field(default_factory=dict, description="Final epoch metrics")

    # Artifacts
    checkpoints: List[CheckpointInfo] = Field(default_factory=list)
    best_checkpoint_path: Optional[str] = None
    log_file_path: Optional[str] = None

    # Execution info
    gpu_id: Optional[int] = None
    peak_memory_mb: Optional[float] = Field(None, ge=0)
    epochs_completed: int = Field(ge=0, default=0)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(ge=0, default=0)

    # Metadata
    experiment_spec: ExperimentSpec

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_id": "exp_001_resnet50_lr0.001",
                "job_id": "job_001",
                "status": "completed",
                "started_at": "2025-11-18T10:00:00Z",
                "completed_at": "2025-11-18T11:30:00Z",
                "duration_seconds": 5400,
                "best_metrics": {
                    "auc": 0.872,
                    "accuracy": 0.834
                },
                "epochs_completed": 100,
                "gpu_id": 0
            }
        }
    )


# ============================================================================
# Scheduler Schemas
# ============================================================================

class QueuedJob(BaseModel):
    """Job in scheduler queue."""
    job_id: str
    experiment_id: str
    priority: int = Field(ge=0, le=10)
    queued_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    estimated_duration: Optional[int] = Field(None, ge=0, description="Estimated duration in seconds")
    gpu_preference: GPUAllocation = GPUAllocation.AUTO


class GPUStatus(BaseModel):
    """GPU resource status."""
    gpu_id: int = Field(ge=0, le=7)
    allocated: bool = Field(default=False)
    current_job_id: Optional[str] = None
    utilization_percent: float = Field(ge=0, le=100, default=0.0)
    memory_used_mb: float = Field(ge=0, default=0.0)
    memory_total_mb: float = Field(ge=0, default=0.0)
    temperature_celsius: Optional[float] = Field(None, ge=0, le=100)


class SchedulerStatus(BaseModel):
    """Overall scheduler status."""
    queue_length: int = Field(ge=0)
    running_jobs: int = Field(ge=0)
    available_gpus: int = Field(ge=0)
    gpu_statuses: List[GPUStatus] = Field(default_factory=list)
    queued_jobs: List[QueuedJob] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Validation Functions
# ============================================================================

def validate_experiment_spec(spec: Dict[str, Any]) -> ExperimentSpec:
    """
    Validate experiment specification dictionary.

    Args:
        spec: Raw experiment specification

    Returns:
        Validated ExperimentSpec instance

    Raises:
        ValidationError: If specification is invalid
    """
    return ExperimentSpec(**spec)


def validate_training_job_config(config: Dict[str, Any]) -> TrainingJobConfig:
    """
    Validate training job configuration dictionary.

    Args:
        config: Raw job configuration

    Returns:
        Validated TrainingJobConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return TrainingJobConfig(**config)


def validate_experiment_result(result: Dict[str, Any]) -> ExperimentResult:
    """
    Validate experiment result dictionary.

    Args:
        result: Raw experiment result

    Returns:
        Validated ExperimentResult instance

    Raises:
        ValidationError: If result is invalid
    """
    return ExperimentResult(**result)
