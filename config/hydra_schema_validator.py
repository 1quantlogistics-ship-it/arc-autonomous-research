"""
Hydra Schema Validator: Ensure configs match AcuVue Hydra format
=================================================================

Validates that ARC-generated configs are compatible with AcuVue's Hydra schema.
Based on reference config: /Users/bengibson/Desktop/AcuVue_repo/configs/phase02_baseline.yaml

Key validation:
- Required top-level sections
- Parameter types and ranges
- Dataset name validity
- Architecture compatibility
"""

import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Validation issue with severity."""
    severity: str  # "error", "warning"
    field: str
    message: str

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class HydraSchemaValidator:
    """
    Validates configs against AcuVue Hydra schema.

    Reference schema from phase02_baseline.yaml:
    - training: epochs, batch_size, learning_rate, save_best_only, patience
    - optimizer: type, weight_decay
    - scheduler: enabled, type, warmup_epochs
    - model: in_channels, out_channels, architecture
    - data: source, data_root, num_samples, train_split, val_split, test_split,
            image_size, use_augmentation
    - augmentation: horizontal_flip, vertical_flip, rotation_degrees, brightness, contrast
    - system: device, seed, log_level, num_workers
    - logging: log_every_n_steps, save_train_images, wandb
    - checkpoint: save_dir, save_frequency, metric_name, metric_mode
    - evaluation: metrics, compute_every_n_epochs, save_predictions
    """

    def __init__(self):
        """Initialize validator with AcuVue schema rules."""

        # Required top-level sections
        self.required_sections = [
            "training", "optimizer", "model", "data", "system"
        ]

        # Optional sections
        self.optional_sections = [
            "scheduler", "augmentation", "logging", "checkpoint", "evaluation"
        ]

        # Valid dataset names (from AcuVue)
        self.valid_datasets = [
            "synthetic", "rim_one", "refuge", "drions", "combined"
        ]

        # Valid model architectures
        self.valid_architectures = [
            "unet",  # Phase 02
            "effnet_fusion"  # Phase 03
        ]

        # Valid optimizer types
        self.valid_optimizers = ["adam", "adamw", "sgd"]

        # Valid scheduler types
        self.valid_schedulers = ["cosine", "step", "plateau"]

        # Valid devices
        self.valid_devices = ["auto", "cuda", "cpu"]

        # Valid log levels
        self.valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        # Valid metrics
        self.valid_metrics = [
            "dice", "iou", "accuracy", "sensitivity", "specificity", "precision", "recall", "f1"
        ]

        # Parameter schemas with types and ranges
        self.param_schemas = {
            "training.epochs": {"type": int, "min": 1, "max": 100},
            "training.batch_size": {"type": int, "min": 1, "max": 32},
            "training.learning_rate": {"type": float, "min": 1e-6, "max": 0.1},
            "training.patience": {"type": int, "min": -1, "max": 50},  # -1 = disabled

            "optimizer.weight_decay": {"type": float, "min": 0.0, "max": 0.1},

            "scheduler.warmup_epochs": {"type": int, "min": 0, "max": 10},

            "model.in_channels": {"type": int, "min": 1, "max": 4},
            "model.out_channels": {"type": int, "min": 1, "max": 10},

            "data.num_samples": {"type": int, "min": 10, "max": 10000},
            "data.train_split": {"type": float, "min": 0.5, "max": 0.9},
            "data.val_split": {"type": float, "min": 0.05, "max": 0.3},
            "data.test_split": {"type": float, "min": 0.05, "max": 0.3},
            "data.image_size": {"type": int, "min": 128, "max": 1024},

            "augmentation.horizontal_flip": {"type": float, "min": 0.0, "max": 1.0},
            "augmentation.vertical_flip": {"type": float, "min": 0.0, "max": 1.0},
            "augmentation.rotation_degrees": {"type": (int, float), "min": 0, "max": 45},
            "augmentation.brightness": {"type": float, "min": 0.0, "max": 1.0},
            "augmentation.contrast": {"type": float, "min": 0.0, "max": 1.0},

            "system.seed": {"type": int, "min": 0, "max": 10000},
            "system.num_workers": {"type": int, "min": 0, "max": 16},

            "logging.log_every_n_steps": {"type": int, "min": 1, "max": 1000},

            "checkpoint.save_frequency": {"type": int, "min": 1, "max": 10},

            "evaluation.compute_every_n_epochs": {"type": int, "min": 1, "max": 10},
        }

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate config against Hydra schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required sections
        for section in self.required_sections:
            if section not in config:
                issues.append(ValidationIssue(
                    severity="error",
                    field=section,
                    message=f"Required section '{section}' is missing"
                ))

        # Validate training section
        if "training" in config:
            issues.extend(self._validate_training(config["training"]))

        # Validate optimizer section
        if "optimizer" in config:
            issues.extend(self._validate_optimizer(config["optimizer"]))

        # Validate model section
        if "model" in config:
            issues.extend(self._validate_model(config["model"]))

        # Validate data section
        if "data" in config:
            issues.extend(self._validate_data(config["data"]))

        # Validate system section
        if "system" in config:
            issues.extend(self._validate_system(config["system"]))

        # Validate optional sections if present
        if "scheduler" in config:
            issues.extend(self._validate_scheduler(config["scheduler"]))

        if "augmentation" in config:
            issues.extend(self._validate_augmentation(config["augmentation"]))

        if "evaluation" in config:
            issues.extend(self._validate_evaluation(config["evaluation"]))

        # Validate parameter values
        issues.extend(self._validate_parameters(config))

        # Determine if valid (no errors, warnings are okay)
        has_errors = any(issue.severity == "error" for issue in issues)
        is_valid = not has_errors

        return is_valid, issues

    def _validate_training(self, training: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate training section."""
        issues = []

        required_fields = ["epochs", "batch_size", "learning_rate"]
        for field in required_fields:
            if field not in training:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"training.{field}",
                    message=f"Required field '{field}' is missing"
                ))

        return issues

    def _validate_optimizer(self, optimizer: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate optimizer section."""
        issues = []

        if "type" not in optimizer:
            issues.append(ValidationIssue(
                severity="error",
                field="optimizer.type",
                message="Required field 'type' is missing"
            ))
        elif optimizer["type"] not in self.valid_optimizers:
            issues.append(ValidationIssue(
                severity="error",
                field="optimizer.type",
                message=f"Invalid optimizer '{optimizer['type']}'. Must be one of: {self.valid_optimizers}"
            ))

        return issues

    def _validate_model(self, model: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate model section."""
        issues = []

        required_fields = ["in_channels", "out_channels", "architecture"]
        for field in required_fields:
            if field not in model:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"model.{field}",
                    message=f"Required field '{field}' is missing"
                ))

        if "architecture" in model and model["architecture"] not in self.valid_architectures:
            issues.append(ValidationIssue(
                severity="warning",
                field="model.architecture",
                message=f"Architecture '{model['architecture']}' not in known set: {self.valid_architectures}"
            ))

        return issues

    def _validate_data(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data section."""
        issues = []

        required_fields = ["source", "image_size"]
        for field in required_fields:
            if field not in data:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"data.{field}",
                    message=f"Required field '{field}' is missing"
                ))

        if "source" in data and data["source"] not in self.valid_datasets:
            issues.append(ValidationIssue(
                severity="warning",
                field="data.source",
                message=f"Dataset '{data['source']}' not in known set: {self.valid_datasets}"
            ))

        # Check splits sum to 1.0
        if all(key in data for key in ["train_split", "val_split", "test_split"]):
            total = data["train_split"] + data["val_split"] + data["test_split"]
            if abs(total - 1.0) > 0.01:
                issues.append(ValidationIssue(
                    severity="error",
                    field="data.splits",
                    message=f"Train/val/test splits must sum to 1.0, got {total:.3f}"
                ))

        return issues

    def _validate_system(self, system: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate system section."""
        issues = []

        if "device" in system and system["device"] not in self.valid_devices:
            issues.append(ValidationIssue(
                severity="error",
                field="system.device",
                message=f"Invalid device '{system['device']}'. Must be one of: {self.valid_devices}"
            ))

        if "log_level" in system and system["log_level"] not in self.valid_log_levels:
            issues.append(ValidationIssue(
                severity="warning",
                field="system.log_level",
                message=f"Log level '{system['log_level']}' not standard. Valid: {self.valid_log_levels}"
            ))

        return issues

    def _validate_scheduler(self, scheduler: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate scheduler section."""
        issues = []

        if "type" in scheduler and scheduler["type"] not in self.valid_schedulers:
            issues.append(ValidationIssue(
                severity="warning",
                field="scheduler.type",
                message=f"Scheduler type '{scheduler['type']}' not in known set: {self.valid_schedulers}"
            ))

        return issues

    def _validate_augmentation(self, augmentation: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate augmentation section."""
        issues = []

        # All augmentation parameters should be floats between 0-1 or rotation degrees
        for key, value in augmentation.items():
            if key == "rotation_degrees":
                if not isinstance(value, (int, float)) or value < 0 or value > 45:
                    issues.append(ValidationIssue(
                        severity="error",
                        field=f"augmentation.{key}",
                        message=f"Rotation degrees must be between 0-45, got {value}"
                    ))
            elif not isinstance(value, (int, float)) or value < 0 or value > 1:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"augmentation.{key}",
                    message=f"Augmentation parameter must be between 0-1, got {value}"
                ))

        return issues

    def _validate_evaluation(self, evaluation: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate evaluation section."""
        issues = []

        if "metrics" in evaluation:
            if not isinstance(evaluation["metrics"], list):
                issues.append(ValidationIssue(
                    severity="error",
                    field="evaluation.metrics",
                    message="Metrics must be a list"
                ))
            else:
                for metric in evaluation["metrics"]:
                    if metric not in self.valid_metrics:
                        issues.append(ValidationIssue(
                            severity="warning",
                            field="evaluation.metrics",
                            message=f"Metric '{metric}' not in known set: {self.valid_metrics}"
                        ))

        return issues

    def _validate_parameters(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate parameter values against schema."""
        issues = []

        for param_path, schema in self.param_schemas.items():
            # Navigate to nested parameter
            parts = param_path.split(".")
            value = config

            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break

            if value is None:
                continue  # Parameter not present (may be optional)

            # Validate type
            expected_type = schema["type"]
            if isinstance(expected_type, tuple):
                # Multiple types allowed
                if not isinstance(value, expected_type):
                    issues.append(ValidationIssue(
                        severity="error",
                        field=param_path,
                        message=f"Expected one of {expected_type}, got {type(value).__name__}"
                    ))
            else:
                if not isinstance(value, expected_type):
                    issues.append(ValidationIssue(
                        severity="error",
                        field=param_path,
                        message=f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    ))

            # Validate range
            if isinstance(value, (int, float)):
                if "min" in schema and value < schema["min"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        field=param_path,
                        message=f"Value {value} below minimum {schema['min']}"
                    ))

                if "max" in schema and value > schema["max"]:
                    issues.append(ValidationIssue(
                        severity="error",
                        field=param_path,
                        message=f"Value {value} above maximum {schema['max']}"
                    ))

        return issues

    def validate_from_file(self, config_path: str) -> Tuple[bool, List[ValidationIssue]]:
        """
        Load and validate YAML config file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Tuple of (is_valid, list of issues)
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            return self.validate(config)

        except yaml.YAMLError as e:
            return False, [ValidationIssue(
                severity="error",
                field="yaml",
                message=f"Failed to parse YAML: {e}"
            )]
        except FileNotFoundError:
            return False, [ValidationIssue(
                severity="error",
                field="file",
                message=f"Config file not found: {config_path}"
            )]


def get_hydra_validator() -> HydraSchemaValidator:
    """Factory function to get validator instance."""
    return HydraSchemaValidator()
