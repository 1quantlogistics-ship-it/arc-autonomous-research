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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from schemas.experiment_schemas import (
    ExperimentSpec, TrainingJobConfig, ExperimentResult,
    PreprocessingChain, MetricType, DatasetType, JobStatus,
    MetricResult, CheckpointInfo, validate_experiment_spec,
    validate_training_job_config
)
from tool_governance import get_tool_governance, ToolValidationError
from config import get_settings, ARCSettings

logger = logging.getLogger(__name__)


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
    """Execute individual preprocessing step."""
    # This is a placeholder - actual implementation would call AcuVue preprocessing
    # For now, return mock result
    return {
        "step_type": step.type.value,
        "status": "success",
        "params": step.params,
        "duration_seconds": 0.1
    }


def _validate_preprocessed_dataset(output_dir: Path) -> Dict[str, Any]:
    """Validate preprocessed dataset structure and contents."""
    # Placeholder validation
    if not output_dir.exists():
        return {"valid": False, "error": "Output directory does not exist"}

    return {"valid": True, "error": None}


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
    """Submit training job to scheduler."""
    # Placeholder - actual implementation would interact with job scheduler
    return {
        "queue_position": 1,
        "estimated_start_time": datetime.utcnow().isoformat()
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
    """Run actual evaluation."""
    # Placeholder - would call AcuVue evaluation code
    return [
        {
            "metric_type": metric.value,
            "value": 0.85,
            "split": "test"
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
