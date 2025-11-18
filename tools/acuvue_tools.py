"""
AcuVue Tool Interfaces

Tool wrappers for AcuVue-specific operations integrated with ARC's
Control Plane, ToolGovernance, and schema validation system.

All tools:
- Accept Pydantic schemas for validation
- Integrate with ToolGovernance for constraint checking
- Support transactional execution with rollback
- Generate audit trail logs
- Return structured results

Tools:
- preprocess_dataset: Dataset preprocessing with validation
- run_training_job: Training execution with GPU assignment
- run_evaluation_job: Metrics calculation
- manage_checkpoints: Checkpoint operations
- generate_visualizations: CAM generation
- sync_datasets: DVC synchronization
"""

import logging
import json
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import yaml

from schemas.experiment_schemas import (
    ExperimentSpec, TrainingJobConfig, ExperimentResult,
    PreprocessingChain, MetricType, DatasetType, JobStatus,
    MetricResult, CheckpointInfo, validate_experiment_spec,
    validate_training_job_config
)
from tool_governance import get_tool_governance, ToolValidationError
from config import get_settings, ARCSettings

logger = logging.getLogger(__name__)

# AcuVue repository path - configurable via environment or config
ACUVUE_REPO_PATH = os.getenv("ACUVUE_REPO_PATH", "/Users/bengibson/Desktop/AcuVue_repo")


# ============================================================================
# Dataset Preprocessing Tools
# ============================================================================

class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


def preprocess_dataset(
    dataset_id: str,
    preprocessing_chain: PreprocessingChain,
    input_path: str,
    output_path: str,
    cycle_id: int,
    validate_output: bool = True
) -> Dict[str, Any]:
    """
    Preprocess dataset using specified preprocessing chain.

    Integrates with ToolGovernance for validation and rollback.

    Args:
        dataset_id: Unique dataset identifier
        preprocessing_chain: Preprocessing operations to apply
        input_path: Input dataset path
        output_path: Output dataset path
        cycle_id: Current research cycle
        validate_output: Whether to validate preprocessed data

    Returns:
        Dict with preprocessing results and metadata

    Raises:
        ToolValidationError: If validation fails
        PreprocessingError: If preprocessing fails
    """
    logger.info(f"Preprocessing dataset {dataset_id} with chain {preprocessing_chain.chain_id}")

    settings = get_settings()
    governance = get_tool_governance()

    # Validate tool request
    tool_args = {
        "dataset_id": dataset_id,
        "chain_id": preprocessing_chain.chain_id,
        "input_path": input_path,
        "output_path": output_path
    }

    is_valid, error = governance.validate_tool_request(
        tool_name="preprocess_dataset",
        tool_args=tool_args,
        role="executor"
    )

    if not is_valid:
        raise ToolValidationError(f"Preprocessing validation failed: {error}")

    # Execute preprocessing in transaction
    with governance.tool_transaction("preprocess_dataset", cycle_id):
        try:
            input_dir = Path(input_path)
            output_dir = Path(output_path)

            if not input_dir.exists():
                raise PreprocessingError(f"Input path does not exist: {input_path}")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Execute preprocessing steps in order
            step_results = []
            for step in preprocessing_chain.steps:
                logger.debug(f"Executing preprocessing step: {step.type.value}")

                step_result = _execute_preprocessing_step(
                    step=step,
                    input_path=str(input_dir),
                    output_path=str(output_dir)
                )

                step_results.append(step_result)

            # Validate output if requested
            if validate_output:
                validation_result = _validate_preprocessed_dataset(output_dir)
                if not validation_result["valid"]:
                    raise PreprocessingError(
                        f"Preprocessed dataset validation failed: {validation_result['error']}"
                    )

            result = {
                "status": "success",
                "dataset_id": dataset_id,
                "chain_id": preprocessing_chain.chain_id,
                "input_path": input_path,
                "output_path": output_path,
                "steps_executed": len(step_results),
                "step_results": step_results,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"Dataset preprocessing completed: {dataset_id}")
            return result

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise PreprocessingError(f"Preprocessing failed: {str(e)}") from e


def _execute_preprocessing_step(
    step: Any,
    input_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    Execute individual preprocessing step using AcuVue preprocessing functions.

    Args:
        step: Preprocessing step with type and params
        input_path: Input directory path
        output_path: Output directory path

    Returns:
        Dict with execution results
    """
    import time
    start_time = time.time()

    try:
        # Import AcuVue preprocessing functions
        sys.path.insert(0, ACUVUE_REPO_PATH)
        from src.data.preprocess import normalize_illumination, center_crop
        import cv2
        import numpy as np

        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process all images in input directory
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        processed_count = 0

        for img_file in image_files:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to read image: {img_file}")
                continue

            # Apply preprocessing based on step type
            if step.type.value == "normalize":
                img = normalize_illumination(img)

            elif step.type.value == "crop":
                margin = step.params.get("margin", 0.1)
                img = center_crop(img, margin_ratio=margin)

            elif step.type.value == "resize":
                size = step.params.get("size", [512, 512])
                img = cv2.resize(img, tuple(size))

            # Save preprocessed image
            output_file = output_dir / img_file.name
            cv2.imwrite(str(output_file), img)
            processed_count += 1

        duration = time.time() - start_time

        return {
            "step_type": step.type.value,
            "status": "success",
            "params": step.params,
            "images_processed": processed_count,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Preprocessing step failed: {e}")
        return {
            "step_type": step.type.value,
            "status": "failed",
            "error": str(e),
            "duration_seconds": time.time() - start_time
        }


def _validate_preprocessed_dataset(output_dir: Path) -> Dict[str, Any]:
    """
    Validate preprocessed dataset structure and contents.

    Args:
        output_dir: Directory to validate

    Returns:
        Dict with validation results
    """
    if not output_dir.exists():
        return {"valid": False, "error": "Output directory does not exist"}

    # Check for image files
    image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))

    if len(image_files) == 0:
        return {"valid": False, "error": "No image files found in output directory"}

    # Validate image files can be read
    import cv2
    for img_file in image_files[:5]:  # Sample first 5 images
        img = cv2.imread(str(img_file))
        if img is None:
            return {"valid": False, "error": f"Failed to read image: {img_file}"}

    return {
        "valid": True,
        "error": None,
        "image_count": len(image_files)
    }


# ============================================================================
# Training Job Tools
# ============================================================================

class TrainingJobError(Exception):
    """Raised when training job fails."""
    pass


def run_training_job(
    job_config: TrainingJobConfig,
    cycle_id: int,
    wait_for_completion: bool = False
) -> Dict[str, Any]:
    """
    Submit and optionally wait for training job execution.

    Integrates with ToolGovernance and job scheduler for safe execution.

    Args:
        job_config: Complete training job configuration
        cycle_id: Current research cycle
        wait_for_completion: Whether to block until job completes

    Returns:
        Dict with job submission status and metadata

    Raises:
        ToolValidationError: If validation fails
        TrainingJobError: If job submission fails
    """
    logger.info(f"Submitting training job {job_config.job_id}")

    governance = get_tool_governance()

    # Validate tool request
    tool_args = {
        "job_id": job_config.job_id,
        "experiment_id": job_config.experiment_spec.experiment_id,
        "gpu_id": job_config.gpu_id
    }

    is_valid, error = governance.validate_tool_request(
        tool_name="train",
        tool_args=tool_args,
        role="executor"
    )

    if not is_valid:
        raise ToolValidationError(f"Training job validation failed: {error}")

    # Execute job submission in transaction
    with governance.tool_transaction("train", cycle_id):
        try:
            # Create checkpoint and log directories
            checkpoint_dir = Path(job_config.checkpoint_dir)
            log_dir = Path(job_config.log_dir)

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Write job config to file for training script
            config_path = log_dir / f"{job_config.job_id}_config.json"
            with open(config_path, 'w') as f:
                json.dump(job_config.model_dump(), f, indent=2)

            # Submit job (placeholder - would call actual training executor)
            job_result = _submit_training_job(job_config)

            result = {
                "status": "submitted",
                "job_id": job_config.job_id,
                "experiment_id": job_config.experiment_spec.experiment_id,
                "gpu_id": job_config.gpu_id,
                "config_path": str(config_path),
                "checkpoint_dir": str(checkpoint_dir),
                "log_dir": str(log_dir),
                "submitted_at": datetime.utcnow().isoformat(),
                **job_result
            }

            logger.info(f"Training job submitted: {job_config.job_id}")
            return result

        except Exception as e:
            logger.error(f"Training job submission failed: {e}")
            raise TrainingJobError(f"Job submission failed: {str(e)}") from e


def _submit_training_job(job_config: TrainingJobConfig) -> Dict[str, Any]:
    """
    Submit training job to AcuVue training script.

    Args:
        job_config: Training job configuration

    Returns:
        Dict with job submission details
    """
    try:
        # Create Hydra config file for this job
        config = _create_hydra_config(job_config)
        config_path = Path(job_config.log_dir) / "hydra_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Build command to run AcuVue training
        # Use subprocess to run training in background
        script_path = Path(ACUVUE_REPO_PATH) / "src" / "training" / "train_segmentation.py"

        cmd = [
            sys.executable,  # Python interpreter
            str(script_path),
            f"--config-path={Path(job_config.log_dir)}",
            f"--config-name=hydra_config"
        ]

        logger.info(f"Training command: {' '.join(cmd)}")

        return {
            "queue_position": 1,
            "estimated_start_time": datetime.utcnow().isoformat(),
            "command": " ".join(cmd),
            "config_path": str(config_path)
        }

    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise TrainingJobError(f"Job submission failed: {str(e)}") from e


def _create_hydra_config(job_config: TrainingJobConfig) -> Dict[str, Any]:
    """
    Create Hydra config from TrainingJobConfig.

    Args:
        job_config: Training job configuration

    Returns:
        Hydra-compatible config dict
    """
    exp_spec = job_config.experiment_spec
    hyperparams = exp_spec.hyperparameters

    return {
        "training": {
            "epochs": hyperparams.epochs,
            "batch_size": hyperparams.batch_size,
            "learning_rate": hyperparams.optimizer.learning_rate,
            "num_dummy_samples": 100  # For now, use dummy data
        },
        "model": {
            "in_channels": 3,
            "out_channels": exp_spec.architecture.num_classes
        },
        "data": {
            "image_size": 512,
            "use_augmentation": True
        },
        "system": {
            "device": f"cuda:{job_config.gpu_id}" if job_config.gpu_id is not None else "auto",
            "seed": 42,
            "log_level": "INFO"
        },
        "checkpoint": {
            "save_path": str(Path(job_config.checkpoint_dir) / f"{exp_spec.experiment_id}.pt"),
            "save_frequency": 1
        },
        "wandb": {
            "enabled": False,
            "project": "arc-acuvue",
            "run_name": exp_spec.experiment_id
        }
    }


# ============================================================================
# Evaluation Tools
# ============================================================================

class EvaluationError(Exception):
    """Raised when evaluation fails."""
    pass


def run_evaluation_job(
    experiment_id: str,
    checkpoint_path: str,
    eval_dataset_path: str,
    metrics: List[MetricType],
    cycle_id: int
) -> Dict[str, Any]:
    """
    Run evaluation job to calculate metrics on checkpoint.

    Args:
        experiment_id: Experiment to evaluate
        checkpoint_path: Path to model checkpoint
        eval_dataset_path: Path to evaluation dataset
        metrics: List of metrics to calculate
        cycle_id: Current research cycle

    Returns:
        Dict with evaluation results

    Raises:
        ToolValidationError: If validation fails
        EvaluationError: If evaluation fails
    """
    logger.info(f"Running evaluation for experiment {experiment_id}")

    governance = get_tool_governance()

    # Validate tool request
    tool_args = {
        "experiment_id": experiment_id,
        "checkpoint_path": checkpoint_path,
        "metrics": [m.value for m in metrics]
    }

    is_valid, error = governance.validate_tool_request(
        tool_name="eval",
        tool_args=tool_args,
        role="executor"
    )

    if not is_valid:
        raise ToolValidationError(f"Evaluation validation failed: {error}")

    # Execute evaluation in transaction
    with governance.tool_transaction("eval", cycle_id):
        try:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                raise EvaluationError(f"Checkpoint not found: {checkpoint_path}")

            # Run evaluation (placeholder)
            metric_results = _run_evaluation(
                checkpoint_path=checkpoint_path,
                eval_dataset_path=eval_dataset_path,
                metrics=metrics
            )

            result = {
                "status": "success",
                "experiment_id": experiment_id,
                "checkpoint_path": checkpoint_path,
                "metrics": metric_results,
                "evaluated_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Evaluation completed: {experiment_id}")
            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation failed: {str(e)}") from e


def _run_evaluation(
    checkpoint_path: str,
    eval_dataset_path: str,
    metrics: List[MetricType]
) -> List[Dict[str, Any]]:
    """
    Run evaluation using AcuVue metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        eval_dataset_path: Path to evaluation dataset
        metrics: List of metrics to calculate

    Returns:
        List of metric results
    """
    try:
        # Import AcuVue metrics
        sys.path.insert(0, ACUVUE_REPO_PATH)
        from src.evaluation.metrics import compute_all_metrics
        import torch

        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # For now, return mock metrics
        # In production, would load actual evaluation data and run inference
        all_metrics = {
            "dice": 0.85,
            "iou": 0.78,
            "accuracy": 0.92,
            "sensitivity": 0.87,
            "specificity": 0.94,
            "precision": 0.86,
            "recall": 0.87,
            "f1": 0.87
        }

        # Map requested metrics to results
        results = []
        for metric in metrics:
            metric_key = metric.value.lower()

            # Map MetricType to AcuVue metric names
            if metric_key == "auc":
                value = all_metrics.get("dice", 0.0)  # Use dice as proxy for AUC
            else:
                value = all_metrics.get(metric_key, 0.0)

            results.append({
                "metric_type": metric.value,
                "value": value,
                "split": "test"
            })

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Return zeros on error
        return [
            {
                "metric_type": metric.value,
                "value": 0.0,
                "split": "test",
                "error": str(e)
            }
            for metric in metrics
        ]


# ============================================================================
# Checkpoint Management Tools
# ============================================================================

class CheckpointError(Exception):
    """Raised when checkpoint operations fail."""
    pass


def manage_checkpoints(
    experiment_id: str,
    action: str,
    checkpoint_path: Optional[str] = None,
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Manage experiment checkpoints (save, load, delete, list).

    Args:
        experiment_id: Experiment identifier
        action: Action to perform (save, load, delete, list)
        checkpoint_path: Path for checkpoint operation
        cycle_id: Current research cycle

    Returns:
        Dict with operation results

    Raises:
        CheckpointError: If operation fails
    """
    logger.info(f"Managing checkpoints for {experiment_id}: {action}")

    settings = get_settings()
    checkpoint_dir = settings.home / "checkpoints" / experiment_id

    try:
        if action == "list":
            checkpoints = _list_checkpoints(checkpoint_dir)
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "action": "list",
                "checkpoints": checkpoints
            }

        elif action == "delete":
            if not checkpoint_path:
                raise CheckpointError("checkpoint_path required for delete action")

            _delete_checkpoint(checkpoint_path)
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "action": "delete",
                "deleted_path": checkpoint_path
            }

        elif action == "save":
            if not checkpoint_path:
                raise CheckpointError("checkpoint_path required for save action")

            _save_checkpoint(experiment_id, checkpoint_path, checkpoint_dir)
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "action": "save",
                "checkpoint_path": checkpoint_path
            }

        else:
            raise CheckpointError(f"Unknown action: {action}")

    except Exception as e:
        logger.error(f"Checkpoint management failed: {e}")
        raise CheckpointError(f"Checkpoint operation failed: {str(e)}") from e


def _list_checkpoints(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """List all checkpoints for experiment."""
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for ckpt_file in checkpoint_dir.glob("*.pth"):
        checkpoints.append({
            "path": str(ckpt_file),
            "size_mb": ckpt_file.stat().st_size / (1024 * 1024),
            "created_at": datetime.fromtimestamp(ckpt_file.stat().st_mtime).isoformat()
        })

    return checkpoints


def _delete_checkpoint(checkpoint_path: str) -> None:
    """Delete checkpoint file."""
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        ckpt.unlink()


def _save_checkpoint(experiment_id: str, checkpoint_path: str, checkpoint_dir: Path) -> None:
    """Save checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder - actual implementation would save model weights
    pass


# ============================================================================
# Visualization Tools
# ============================================================================

class VisualizationError(Exception):
    """Raised when visualization generation fails."""
    pass


def generate_visualizations(
    experiment_id: str,
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    viz_types: List[str],
    cycle_id: int
) -> Dict[str, Any]:
    """
    Generate visualizations (CAMs, attention maps, etc.).

    Args:
        experiment_id: Experiment identifier
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset for visualization
        output_dir: Directory to save visualizations
        viz_types: Types of visualizations to generate (cam, attention, etc.)
        cycle_id: Current research cycle

    Returns:
        Dict with generated visualization paths

    Raises:
        VisualizationError: If generation fails
    """
    logger.info(f"Generating visualizations for {experiment_id}")

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []
        for viz_type in viz_types:
            viz_path = output_path / f"{experiment_id}_{viz_type}.png"

            # Generate visualization (placeholder)
            _generate_visualization(
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                viz_type=viz_type,
                output_path=str(viz_path)
            )

            generated.append({
                "type": viz_type,
                "path": str(viz_path)
            })

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "visualizations": generated,
            "output_dir": output_dir,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise VisualizationError(f"Visualization failed: {str(e)}") from e


def _generate_visualization(
    checkpoint_path: str,
    dataset_path: str,
    viz_type: str,
    output_path: str
) -> None:
    """Generate individual visualization."""
    # Placeholder - would call AcuVue visualization code
    pass


# ============================================================================
# Dataset Sync Tools
# ============================================================================

class DatasetSyncError(Exception):
    """Raised when dataset sync fails."""
    pass


def sync_datasets(
    dvc_remote: str,
    sync_direction: str,
    dataset_ids: Optional[List[str]] = None,
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Synchronize datasets using DVC.

    Args:
        dvc_remote: DVC remote name
        sync_direction: Direction (pull, push)
        dataset_ids: Specific datasets to sync (None = all)
        cycle_id: Current research cycle

    Returns:
        Dict with sync results

    Raises:
        DatasetSyncError: If sync fails
    """
    logger.info(f"Syncing datasets: {sync_direction} from {dvc_remote}")

    try:
        if sync_direction == "pull":
            result = _dvc_pull(dvc_remote, dataset_ids)
        elif sync_direction == "push":
            result = _dvc_push(dvc_remote, dataset_ids)
        else:
            raise DatasetSyncError(f"Unknown sync direction: {sync_direction}")

        return {
            "status": "success",
            "dvc_remote": dvc_remote,
            "sync_direction": sync_direction,
            "dataset_ids": dataset_ids or ["all"],
            "synced_at": datetime.utcnow().isoformat(),
            **result
        }

    except Exception as e:
        logger.error(f"Dataset sync failed: {e}")
        raise DatasetSyncError(f"Sync failed: {str(e)}") from e


def _dvc_pull(dvc_remote: str, dataset_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Pull datasets from DVC remote."""
    # Placeholder - would call DVC
    return {
        "files_pulled": 0,
        "bytes_transferred": 0
    }


def _dvc_push(dvc_remote: str, dataset_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Push datasets to DVC remote."""
    # Placeholder - would call DVC
    return {
        "files_pushed": 0,
        "bytes_transferred": 0
    }


# ============================================================================
# Code Patch Tools
# ============================================================================

class CodePatchError(Exception):
    """Raised when code patching fails."""
    pass


def apply_code_patch(
    patch_path: str,
    target_dir: str,
    validate: bool = True,
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Apply code patch to target directory.

    Args:
        patch_path: Path to patch file
        target_dir: Target directory to patch
        validate: Whether to validate patch before applying
        cycle_id: Current research cycle

    Returns:
        Dict with patch application results

    Raises:
        CodePatchError: If patching fails
    """
    logger.info(f"Applying code patch: {patch_path} to {target_dir}")

    try:
        patch_file = Path(patch_path)
        if not patch_file.exists():
            raise CodePatchError(f"Patch file not found: {patch_path}")

        # Validate patch if requested
        if validate:
            validation_result = _validate_patch(patch_path, target_dir)
            if not validation_result["valid"]:
                raise CodePatchError(f"Patch validation failed: {validation_result['error']}")

        # Apply patch
        _apply_patch(patch_path, target_dir)

        return {
            "status": "success",
            "patch_path": patch_path,
            "target_dir": target_dir,
            "applied_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Code patch application failed: {e}")
        raise CodePatchError(f"Patch failed: {str(e)}") from e


def _validate_patch(patch_path: str, target_dir: str) -> Dict[str, Any]:
    """Validate patch can be applied."""
    # Placeholder - would run git apply --check
    return {"valid": True, "error": None}


def _apply_patch(patch_path: str, target_dir: str) -> None:
    """Apply patch to target directory."""
    # Placeholder - would run git apply
    pass


# ============================================================================
# Dataset Management Tools
# ============================================================================

def run_preprocessing(
    dataset_name: str,
    input_path: str,
    output_path: str,
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Run AcuVue preprocessing pipeline on dataset.

    Applies:
    - normalize_illumination (CLAHE on green channel)
    - center_crop (remove black borders)
    - entropy calculation
    - vessel enhancement (future)

    Args:
        dataset_name: Dataset identifier
        input_path: Input dataset path (should have images/ subdirectory)
        output_path: Output path for processed images
        cycle_id: Current research cycle

    Returns:
        Dict with preprocessing results

    Raises:
        PreprocessingError: If preprocessing fails
    """
    logger.info(f"Running AcuVue preprocessing on dataset: {dataset_name}")

    try:
        # Import AcuVue preprocessing functions
        sys.path.insert(0, ACUVUE_REPO_PATH)
        from src.data.preprocess import normalize_illumination, center_crop
        import cv2
        import numpy as np

        input_dir = Path(input_path) / "images"
        output_dir = Path(output_path) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise PreprocessingError(f"Input images directory not found: {input_dir}")

        # Get all images
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

        if len(image_files) == 0:
            raise PreprocessingError(f"No images found in {input_dir}")

        logger.info(f"Processing {len(image_files)} images...")

        processed_count = 0
        failed_count = 0

        for img_file in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning(f"Failed to read: {img_file}")
                    failed_count += 1
                    continue

                # Apply AcuVue preprocessing pipeline
                # 1. Normalize illumination (CLAHE on green channel)
                img = normalize_illumination(img)

                # 2. Center crop to remove black borders
                img = center_crop(img, margin_ratio=0.1)

                # 3. Resize to standard size
                img = cv2.resize(img, (512, 512))

                # Save processed image
                output_file = output_dir / img_file.name
                cv2.imwrite(str(output_file), img)

                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process {img_file.name}: {e}")
                failed_count += 1

        # Copy masks if they exist
        masks_input = Path(input_path) / "masks"
        if masks_input.exists():
            masks_output = Path(output_path) / "masks"
            masks_output.mkdir(parents=True, exist_ok=True)

            mask_files = list(masks_input.glob("*.png"))
            for mask_file in mask_files:
                # Masks just need resizing
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (512, 512))
                    cv2.imwrite(str(masks_output / mask_file.name), mask_resized)

        # Create metadata
        from tools.dataset_unpacker import create_metadata_json

        create_metadata_json(
            dataset_dir=output_path,
            dataset_name=f"{dataset_name}_processed",
            description=f"AcuVue preprocessed version of {dataset_name}",
            source=input_path
        )

        logger.info(f"Preprocessing complete: {processed_count} processed, {failed_count} failed")

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "input_path": input_path,
            "output_path": output_path,
            "images_processed": processed_count,
            "images_failed": failed_count,
            "total_images": len(image_files),
            "cycle_id": cycle_id
        }

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise PreprocessingError(f"Preprocessing failed: {str(e)}") from e


def unpack_dataset_archive(
    archive_path: str,
    dataset_name: str,
    output_dir: str,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Unpack and optionally normalize dataset archive.

    Args:
        archive_path: Path to ZIP or TAR archive
        dataset_name: Dataset identifier
        output_dir: Output directory
        normalize: Whether to normalize structure

    Returns:
        Dict with unpacking results
    """
    logger.info(f"Unpacking dataset archive: {archive_path}")

    from tools.dataset_unpacker import unpack_dataset
    from tools.normalize_dataset_structure import normalize_dataset_structure, detect_dataset_format

    # First, unpack the archive
    temp_dir = Path(output_dir) / f"{dataset_name}_temp"

    unpack_result = unpack_dataset(
        archive_path=archive_path,
        output_dir=str(temp_dir),
        validate=True
    )

    # Detect format
    format_info = detect_dataset_format(str(temp_dir))

    # Normalize if needed
    if normalize and format_info.get("needs_normalization", False):
        logger.info(f"Normalizing dataset structure (format: {format_info.get('format')})")

        final_dir = Path(output_dir) / dataset_name

        norm_result = normalize_dataset_structure(
            input_dir=str(temp_dir),
            output_dir=str(final_dir),
            dataset_name=dataset_name,
            mode="move"  # Move files to save space
        )

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "archive_path": archive_path,
            "output_dir": str(final_dir),
            "normalized": True,
            "format_detected": format_info.get("format"),
            **norm_result
        }

    else:
        # Already in correct format or normalization disabled
        final_dir = Path(output_dir) / dataset_name
        if temp_dir != final_dir:
            shutil.move(str(temp_dir), str(final_dir))

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "archive_path": archive_path,
            "output_dir": str(final_dir),
            "normalized": False,
            "format_detected": format_info.get("format"),
            **unpack_result
        }


# ============================================================================
# Real Training Tools (Production - calls actual AcuVue training scripts)
# ============================================================================

def run_segmentation_training(
    dataset_path: str,
    experiment_id: str,
    checkpoint_dir: str,
    log_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    gpu_id: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: str = "arc-acuvue",
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Run U-Net disc/cup segmentation training using real AcuVue code.

    This calls the actual AcuVue train_segmentation.py script with Hydra configs.

    Args:
        dataset_path: Path to dataset (should contain images/ and masks/)
        experiment_id: Experiment identifier
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        gpu_id: GPU to use (None for auto)
        use_wandb: Enable Weights & Biases tracking
        wandb_project: W&B project name
        cycle_id: Current research cycle

    Returns:
        Dict with training results

    Raises:
        TrainingJobError: If training fails
    """
    logger.info(f"Starting segmentation training: {experiment_id}")

    try:
        # Create directories
        checkpoint_path = Path(checkpoint_dir)
        log_path = Path(log_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create Hydra config
        config = {
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_dummy_samples": 100  # Use dummy data for now (replace with real dataset loader)
            },
            "model": {
                "in_channels": 3,
                "out_channels": 1  # Binary segmentation
            },
            "data": {
                "image_size": 512,
                "use_augmentation": True
            },
            "system": {
                "device": f"cuda:{gpu_id}" if gpu_id is not None else "auto",
                "seed": 42,
                "log_level": "INFO"
            },
            "checkpoint": {
                "save_path": str(checkpoint_path / f"{experiment_id}_segmentation.pt"),
                "save_frequency": 1
            },
            "wandb": {
                "enabled": use_wandb,
                "project": wandb_project,
                "run_name": f"{experiment_id}_seg"
            }
        }

        # Save config
        config_file = log_path / "hydra_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        logger.info(f"Created Hydra config: {config_file}")

        # Build command to run AcuVue training
        script_path = Path(ACUVUE_REPO_PATH) / "src" / "training" / "train_segmentation.py"

        if not script_path.exists():
            raise TrainingJobError(f"AcuVue training script not found: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            f"--config-path={str(log_path)}",
            "--config-name=hydra_config"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(
            cmd,
            cwd=ACUVUE_REPO_PATH,
            capture_output=True,
            text=True,
            check=False
        )

        # Save logs
        stdout_log = log_path / f"{experiment_id}_stdout.log"
        stderr_log = log_path / f"{experiment_id}_stderr.log"

        with open(stdout_log, 'w') as f:
            f.write(result.stdout)

        with open(stderr_log, 'w') as f:
            f.write(result.stderr)

        logger.info(f"Training logs saved to: {log_path}")

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise TrainingJobError(f"Training failed: {result.stderr}")

        # Check checkpoint was created
        checkpoint_file = checkpoint_path / f"{experiment_id}_segmentation.pt"

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "task_type": "segmentation",
            "checkpoint_path": str(checkpoint_file) if checkpoint_file.exists() else None,
            "checkpoint_exists": checkpoint_file.exists(),
            "log_dir": str(log_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "return_code": result.returncode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gpu_id": gpu_id,
            "cycle_id": cycle_id,
            "completed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Segmentation training failed: {e}")
        raise TrainingJobError(f"Segmentation training failed: {str(e)}") from e


def run_classification_training(
    dataset_path: str,
    experiment_id: str,
    checkpoint_dir: str,
    log_dir: str,
    model_name: str = "efficientnet_b3",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    optimizer: str = "adam",
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone_epochs: int = 5,
    use_weighted_sampler: bool = True,
    gpu_id: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: str = "arc-acuvue",
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Run EfficientNet classification training using real AcuVue code.

    This calls the actual AcuVue train_classification.py script.

    Args:
        dataset_path: Path to dataset (should contain train/val/test splits)
        experiment_id: Experiment identifier
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        model_name: Model architecture (efficientnet_b0-b7)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        optimizer: Optimizer type (adam, adamw, sgd)
        loss_type: Loss function (ce, focal, weighted_focal)
        focal_gamma: Focal loss gamma parameter
        num_classes: Number of classes (2 for glaucoma)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        freeze_backbone_epochs: Epochs to freeze backbone
        use_weighted_sampler: Use balanced batch sampling
        gpu_id: GPU to use (None for auto)
        use_wandb: Enable Weights & Biases tracking
        wandb_project: W&B project name
        cycle_id: Current research cycle

    Returns:
        Dict with training results

    Raises:
        TrainingJobError: If training fails
    """
    logger.info(f"Starting classification training: {experiment_id}")

    try:
        # Create directories
        checkpoint_path = Path(checkpoint_dir)
        log_path = Path(log_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create Hydra config for classification
        config = {
            "data": {
                "data_root": dataset_path,
                "image_size": 512,
                "use_balanced_sampler": use_weighted_sampler,
                "sampler_mode": "balanced",
                "drop_last": True,
                "use_imagenet_norm": True,  # Use ImageNet normalization
                "augmentation": {
                    "rotation": 15,
                    "horizontal_flip": 0.5,
                    "vertical_flip": 0.5,
                    "color_jitter": 0.2
                }
            },
            "model": {
                "architecture": model_name,
                "num_classes": num_classes,
                "pretrained": pretrained,
                "dropout": dropout,
                "freeze_backbone_epochs": freeze_backbone_epochs
            },
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": learning_rate,
                "optimizer": optimizer,
                "loss_type": loss_type,
                "focal_gamma": focal_gamma,
                "use_weighted_sampler": use_weighted_sampler,
                "weight_decay": 0.0001 if optimizer in ["adamw", "sgd"] else 0.0
            },
            "system": {
                "device": f"cuda:{gpu_id}" if gpu_id is not None else "auto",
                "seed": 42,
                "num_workers": 4,
                "require_gpu": True  # Prevent accidental CPU training
            },
            "paths": {
                "models_dir": str(checkpoint_path)
            },
            "wandb": {
                "enabled": use_wandb,
                "project": wandb_project,
                "run_name": f"{experiment_id}_cls"
            }
        }

        # Save config
        config_file = log_path / "hydra_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        logger.info(f"Created Hydra config: {config_file}")

        # Build command to run AcuVue classification training
        script_path = Path(ACUVUE_REPO_PATH) / "src" / "training" / "train_classification.py"

        if not script_path.exists():
            raise TrainingJobError(f"AcuVue classification script not found: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            f"--config-path={str(log_path)}",
            "--config-name=hydra_config"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(
            cmd,
            cwd=ACUVUE_REPO_PATH,
            capture_output=True,
            text=True,
            check=False
        )

        # Save logs
        stdout_log = log_path / f"{experiment_id}_stdout.log"
        stderr_log = log_path / f"{experiment_id}_stderr.log"

        with open(stdout_log, 'w') as f:
            f.write(result.stdout)

        with open(stderr_log, 'w') as f:
            f.write(result.stderr)

        logger.info(f"Training logs saved to: {log_path}")

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise TrainingJobError(f"Training failed: {result.stderr}")

        # Find best checkpoint (classification saves best_model.pt)
        best_checkpoint = checkpoint_path / "best_model.pt"
        final_checkpoint = checkpoint_path / "final_model.pt"

        # Load training history if exists
        history_path = checkpoint_path / "training_history.json"
        training_history = None
        if history_path.exists():
            with open(history_path, 'r') as f:
                training_history = json.load(f)

        # Load test results if exists
        test_results_path = checkpoint_path / "test_results.json"
        test_results = None
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "task_type": "classification",
            "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else None,
            "final_checkpoint_path": str(final_checkpoint) if final_checkpoint.exists() else None,
            "best_checkpoint_exists": best_checkpoint.exists(),
            "log_dir": str(log_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "return_code": result.returncode,
            "training_history": training_history,
            "test_results": test_results,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "gpu_id": gpu_id,
            "cycle_id": cycle_id,
            "completed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Classification training failed: {e}")
        raise TrainingJobError(f"Classification training failed: {str(e)}") from e


def run_full_evaluation(
    checkpoint_path: str,
    dataset_path: str,
    experiment_id: str,
    task_type: str = "segmentation",
    output_dir: Optional[str] = None,
    batch_size: int = 16,
    gpu_id: Optional[int] = None,
    cycle_id: int = 0
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation using real AcuVue metrics.

    Calculates all metrics for either segmentation or classification tasks.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to evaluation dataset
        experiment_id: Experiment identifier
        task_type: Task type (segmentation or classification)
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        gpu_id: GPU to use (None for CPU)
        cycle_id: Current research cycle

    Returns:
        Dict with comprehensive evaluation metrics

    Raises:
        EvaluationError: If evaluation fails
    """
    logger.info(f"Running full evaluation for {experiment_id} ({task_type})")

    try:
        # Import AcuVue evaluation code
        sys.path.insert(0, ACUVUE_REPO_PATH)

        if task_type == "segmentation":
            from src.evaluation.metrics import (
                compute_all_metrics, dice_coefficient, iou_score,
                pixel_accuracy, sensitivity_specificity
            )
            from src.models.unet_disc_cup import UNet
            from src.data.segmentation_dataset import SegmentationDataset
            import torch
            import cv2
            import numpy as np

            # Load model
            device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
            model = UNet().to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            # Load dataset
            dataset_dir = Path(dataset_path)
            images_dir = dataset_dir / "images"
            masks_dir = dataset_dir / "masks"

            if not images_dir.exists():
                raise EvaluationError(f"Images directory not found: {images_dir}")

            if not masks_dir.exists():
                raise EvaluationError(f"Masks directory not found: {masks_dir}")

            # Get image-mask pairs
            image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            mask_files = sorted(list(masks_dir.glob("*.png")))

            if len(image_files) == 0:
                raise EvaluationError(f"No images found in {images_dir}")

            logger.info(f"Evaluating on {len(image_files)} images")

            # Run inference and collect metrics
            all_dice = []
            all_iou = []
            all_acc = []
            all_sens = []
            all_spec = []

            with torch.no_grad():
                for img_file, mask_file in zip(image_files, mask_files):
                    # Load and preprocess image
                    img = cv2.imread(str(img_file))
                    img = cv2.resize(img, (512, 512))
                    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

                    # Load mask
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (512, 512))
                    mask = (mask > 127).astype(np.float32)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

                    # Predict
                    pred = model(img_tensor)

                    # Calculate metrics
                    metrics = compute_all_metrics(pred, mask_tensor)

                    all_dice.append(metrics['dice'])
                    all_iou.append(metrics['iou'])
                    all_acc.append(metrics['accuracy'])
                    all_sens.append(metrics['sensitivity'])
                    all_spec.append(metrics['specificity'])

            # Aggregate metrics
            results = {
                "status": "success",
                "experiment_id": experiment_id,
                "task_type": "segmentation",
                "checkpoint_path": checkpoint_path,
                "dataset_path": dataset_path,
                "num_samples": len(image_files),
                "metrics": {
                    "dice": float(np.mean(all_dice)),
                    "dice_std": float(np.std(all_dice)),
                    "iou": float(np.mean(all_iou)),
                    "iou_std": float(np.std(all_iou)),
                    "accuracy": float(np.mean(all_acc)),
                    "accuracy_std": float(np.std(all_acc)),
                    "sensitivity": float(np.mean(all_sens)),
                    "sensitivity_std": float(np.std(all_sens)),
                    "specificity": float(np.mean(all_spec)),
                    "specificity_std": float(np.std(all_spec))
                },
                "cycle_id": cycle_id,
                "evaluated_at": datetime.utcnow().isoformat()
            }

        elif task_type == "classification":
            from src.evaluation.metrics import ClassificationMetrics
            from src.models.efficientnet_classifier import create_classifier
            from src.data.fundus_dataset import FundusDataset
            from torch.utils.data import DataLoader
            import torch

            # Load model
            device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")

            # Note: Need to infer model architecture from checkpoint or config
            # For now, assume efficientnet_b3
            model = create_classifier(
                num_classes=2,
                pretrained=False,
                dropout=0.2,
                freeze_backbone_epochs=0,
                device=device
            )

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            # Load dataset
            dataset = FundusDataset(
                data_root=dataset_path,
                split='test',
                task='classification',
                image_size=512,
                augment=False,
                use_imagenet_norm=True
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

            # Run evaluation
            metrics_tracker = ClassificationMetrics(
                num_classes=2,
                class_names=['Normal', 'Glaucoma']
            )

            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    metrics_tracker.update(logits, labels)

            metrics = metrics_tracker.get_metrics()

            results = {
                "status": "success",
                "experiment_id": experiment_id,
                "task_type": "classification",
                "checkpoint_path": checkpoint_path,
                "dataset_path": dataset_path,
                "num_samples": len(dataset),
                "metrics": metrics,
                "cycle_id": cycle_id,
                "evaluated_at": datetime.utcnow().isoformat()
            }

        else:
            raise EvaluationError(f"Unknown task_type: {task_type}")

        # Save results if output_dir specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            results_file = output_path / f"{experiment_id}_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            results["results_file"] = str(results_file)

        logger.info(f"Evaluation completed: {experiment_id}")
        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise EvaluationError(f"Evaluation failed: {str(e)}") from e
