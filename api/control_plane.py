import os
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
import uvicorn

# Import v1.1.0 infrastructure
from config import get_settings, ARCSettings
from memory_handler import get_memory_handler, MemoryHandler, ValidationFailedError, AtomicWriteError
from schemas import (
    Directive, HistorySummary, Constraints, SystemState,
    OperatingMode, ActiveExperiment
)
from tool_governance import get_tool_governance, ToolGovernance, ToolValidationError, ToolExecutionError

# Import experiment engine components
from schemas.experiment_schemas import (
    ExperimentSpec, TrainingJobConfig, ExperimentResult,
    JobStatus, SchedulerStatus, validate_experiment_spec,
    validate_training_job_config
)
from scheduler.job_scheduler import get_job_scheduler
from tools.acuvue_tools import (
    preprocess_dataset, run_training_job, run_evaluation_job,
    manage_checkpoints, generate_visualizations
)

# Initialize settings, memory handler, and tool governance
settings = get_settings()
memory = get_memory_handler(settings)
governance = get_tool_governance(settings, memory)
scheduler = get_job_scheduler(settings, max_concurrent_jobs=2)

# Configure logging with config-driven paths
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.logs_dir / 'control_plane.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title='ARC Control Plane', version='1.1.0')

# Models
class ExecRequest(BaseModel):
    command: str
    role: str
    cycle_id: int
    requires_approval: bool = True

class TrainRequest(BaseModel):
    experiment_id: str
    config: Dict[str, Any]
    requires_approval: bool = True

class StatusRequest(BaseModel):
    query: Optional[str] = None

class EvalRequest(BaseModel):
    experiment_id: str
    metrics: List[str]

class ArchiveRequest(BaseModel):
    cycle_id: int
    reason: str

class RollbackRequest(BaseModel):
    snapshot_id: str

# Experiment Engine Request Models
class ExperimentCreateRequest(BaseModel):
    experiment_spec: Dict[str, Any]  # Will be validated as ExperimentSpec
    cycle_id: int

class ExperimentScheduleRequest(BaseModel):
    job_config: Dict[str, Any]  # Will be validated as TrainingJobConfig
    priority: int = 5
    cycle_id: int = 0

class ExperimentCancelRequest(BaseModel):
    experiment_id: str

class PreprocessDatasetRequest(BaseModel):
    dataset_id: str
    preprocessing_chain: Dict[str, Any]
    input_path: str
    output_path: str
    cycle_id: int

# Helper functions
def load_system_state() -> SystemState:
    """Load system state with schema validation."""
    return memory.load_system_state()

def save_system_state(state: SystemState) -> None:
    """Save system state with schema validation."""
    memory.save_system_state(state)

def load_directive() -> Directive:
    """Load directive with schema validation."""
    return memory.load_directive()

def load_constraints() -> Constraints:
    """Load constraints with schema validation."""
    return memory.load_constraints()

def validate_command(command: str) -> bool:
    """Validate command against allowlist from config."""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    base_cmd = cmd_parts[0]
    return base_cmd in settings.allowed_commands

def check_mode_permission(action: str) -> bool:
    """Check if action is allowed in current mode."""
    state = load_system_state()
    mode = state.mode

    if mode == OperatingMode.SEMI:
        # All actions require approval in SEMI mode
        return True
    elif mode == OperatingMode.AUTO:
        # Auto mode allows most actions except training
        return action != 'train'
    elif mode == OperatingMode.FULL:
        # Full autonomy
        return True
    else:
        return False

# Endpoints
@app.get('/')
async def root():
    return {
        'service': 'ARC Control Plane',
        'version': '1.1.0',
        'status': 'operational'
    }

@app.get('/status')
async def get_status(query: Optional[str] = None):
    """Get system status with validated schema."""
    try:
        state = load_system_state()
        directive = load_directive()

        status = {
            'mode': state.mode.value,
            'arc_version': state.arc_version,
            'status': state.status,
            'last_cycle': state.last_cycle_timestamp,
            'active_experiments': [exp.dict() for exp in state.active_experiments],
            'current_cycle': directive.cycle_id,
            'current_objective': directive.objective.value
        }

        if query:
            # Filter status based on query
            filtered = {k: v for k, v in status.items() if query.lower() in k.lower()}
            return filtered if filtered else status

        return status
    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except Exception as e:
        logger.error(f'Status check failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/exec')
async def execute_command(req: ExecRequest):
    """Execute command with safety validation and schema-validated logging."""
    try:
        logger.info(f'Exec request from {req.role}: {req.command}')

        # Check mode permission
        if not check_mode_permission('exec'):
            raise HTTPException(status_code=403, detail='Action not permitted in current mode')

        # Validate command
        if not validate_command(req.command):
            logger.warning(f'Command blocked: {req.command}')
            raise HTTPException(status_code=400, detail='Command not in allowlist')

        # In SEMI mode, always require approval
        state = load_system_state()
        if state.mode == OperatingMode.SEMI and req.requires_approval:
            return {
                'status': 'pending_approval',
                'command': req.command,
                'role': req.role,
                'cycle_id': req.cycle_id,
                'message': 'Command requires human approval in SEMI mode'
            }

        # Execute command
        result = subprocess.run(
            req.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Log execution (use config-driven path)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'role': req.role,
            'cycle_id': req.cycle_id,
            'command': req.command,
            'returncode': result.returncode,
            'stdout': result.stdout[:1000],  # Truncate for logging
            'stderr': result.stderr[:1000]
        }

        log_file = settings.logs_dir / f'exec_cycle_{req.cycle_id}.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return {
            'status': 'executed',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail='Command execution timeout')
    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except Exception as e:
        logger.error(f'Exec failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def start_training(req: TrainRequest):
    """Start training job with constraint validation and schema-validated state updates."""
    try:
        logger.info(f'Train request for experiment {req.experiment_id}')

        # Check mode permission
        if not check_mode_permission('train'):
            raise HTTPException(status_code=403, detail='Training not permitted in current mode')

        # Load constraints with validation
        constraints = load_constraints()

        # Validate config against constraints
        validation_errors = []
        for param, value in req.config.items():
            for forbidden in constraints.forbidden_ranges:
                if forbidden.param == param:
                    if forbidden.min is not None and value < forbidden.min:
                        validation_errors.append(f'Parameter {param}={value} below safe range (min={forbidden.min})')
                    if forbidden.max is not None and value > forbidden.max:
                        validation_errors.append(f'Parameter {param}={value} above safe range (max={forbidden.max})')

        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={'error': 'validation_failed', 'issues': validation_errors}
            )

        # In SEMI mode, require approval
        state = load_system_state()
        if state.mode == OperatingMode.SEMI and req.requires_approval:
            return {
                'status': 'pending_approval',
                'experiment_id': req.experiment_id,
                'config': req.config,
                'message': 'Training requires human approval in SEMI mode'
            }

        # Add to active experiments using transaction
        with memory.transaction():
            new_experiment = ActiveExperiment(
                experiment_id=req.experiment_id,
                status='queued',
                started_at=datetime.now().isoformat()
            )
            state.active_experiments.append(new_experiment)
            save_system_state(state)

        return {
            'status': 'queued',
            'experiment_id': req.experiment_id,
            'message': 'Training job queued'
        }

    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Train request failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/eval')
async def evaluate_experiment(req: EvalRequest):
    """Evaluate experiment metrics using config-driven paths."""
    try:
        exp_dir = settings.experiments_dir / req.experiment_id
        if not exp_dir.exists():
            raise HTTPException(status_code=404, detail='Experiment not found')

        # Load experiment results
        results_path = exp_dir / 'results.json'
        if not results_path.exists():
            raise HTTPException(status_code=404, detail='Results not found')

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Extract requested metrics
        metrics = {m: results.get(m) for m in req.metrics if m in results}

        return {
            'experiment_id': req.experiment_id,
            'metrics': metrics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Eval failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/archive')
async def archive_cycle(req: ArchiveRequest):
    """Archive cycle data using memory handler's backup system."""
    try:
        # Create backup using memory handler
        backup_dir = memory.backup_memory()

        # Save metadata
        metadata = {
            'snapshot_id': backup_dir.name,
            'cycle_id': req.cycle_id,
            'reason': req.reason,
            'timestamp': datetime.now().isoformat()
        }

        with open(backup_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f'Archived cycle {req.cycle_id} to {backup_dir.name}')

        return {
            'status': 'archived',
            'snapshot_id': backup_dir.name
        }

    except Exception as e:
        logger.error(f'Archive failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/rollback')
async def rollback_to_snapshot(req: RollbackRequest):
    """Rollback to previous snapshot using memory handler's restore system."""
    try:
        snapshot_dir = settings.snapshots_dir / req.snapshot_id
        if not snapshot_dir.exists():
            raise HTTPException(status_code=404, detail='Snapshot not found')

        # Restore memory using memory handler
        memory.restore_memory(snapshot_dir)

        # Validate restored memory
        is_valid, errors = memory.validate_all_memory()
        if not is_valid:
            logger.error(f'Restored memory validation failed: {errors}')
            raise HTTPException(
                status_code=500,
                detail={'error': 'restore_validation_failed', 'issues': errors}
            )

        logger.info(f'Rolled back to snapshot {req.snapshot_id}')

        return {
            'status': 'rolled_back',
            'snapshot_id': req.snapshot_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Rollback failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/mode')
async def set_mode(mode: str):
    """Change ARC operating mode with schema validation."""
    try:
        # Validate mode using schema enum
        try:
            new_mode = OperatingMode(mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid mode. Must be one of: {[m.value for m in OperatingMode]}'
            )

        # Update state atomically
        with memory.transaction():
            state = load_system_state()
            old_mode = state.mode
            state.mode = new_mode
            save_system_state(state)

        logger.info(f'Mode changed: {old_mode.value} -> {new_mode.value}')

        return {
            'status': 'mode_changed',
            'old_mode': old_mode.value,
            'new_mode': new_mode.value
        }

    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Mode change failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Experiment Engine Endpoints
# ============================================================================

@app.post('/experiments/create')
async def create_experiment(req: ExperimentCreateRequest):
    """
    Create and validate experiment specification.

    Args:
        req: Experiment creation request with spec and cycle_id

    Returns:
        Validated experiment specification
    """
    try:
        logger.info(f'Creating experiment for cycle {req.cycle_id}')

        # Validate experiment spec using Pydantic schema
        experiment_spec = validate_experiment_spec(req.experiment_spec)

        # Store experiment spec (using memory handler for persistence)
        exp_file = settings.experiments_dir / experiment_spec.experiment_id / 'spec.json'
        exp_file.parent.mkdir(parents=True, exist_ok=True)

        with open(exp_file, 'w') as f:
            json.dump(experiment_spec.model_dump(), f, indent=2)

        logger.info(f'Created experiment {experiment_spec.experiment_id}')

        return {
            'status': 'created',
            'experiment_id': experiment_spec.experiment_id,
            'spec': experiment_spec.model_dump()
        }

    except ValidationError as e:
        logger.error(f'Experiment validation failed: {e}')
        raise HTTPException(
            status_code=400,
            detail={'error': 'validation_failed', 'issues': str(e)}
        )
    except Exception as e:
        logger.error(f'Experiment creation failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/experiments/schedule')
async def schedule_experiment(req: ExperimentScheduleRequest):
    """
    Schedule training job for execution.

    Args:
        req: Job scheduling request with config and priority

    Returns:
        Job submission status
    """
    try:
        logger.info(f'Scheduling experiment job (priority={req.priority})')

        # Validate job config
        job_config = validate_training_job_config(req.job_config)

        # Submit to scheduler
        job_id = scheduler.submit_job(
            job_config=job_config,
            priority=req.priority,
            cycle_id=req.cycle_id
        )

        # Get scheduler status
        scheduler_status = scheduler.get_scheduler_status()

        logger.info(f'Scheduled job {job_id} (queue_length={scheduler_status.queue_length})')

        return {
            'status': 'scheduled',
            'job_id': job_id,
            'experiment_id': job_config.experiment_spec.experiment_id,
            'priority': req.priority,
            'queue_position': scheduler_status.queue_length,
            'available_gpus': scheduler_status.available_gpus
        }

    except ValidationError as e:
        logger.error(f'Job config validation failed: {e}')
        raise HTTPException(
            status_code=400,
            detail={'error': 'validation_failed', 'issues': str(e)}
        )
    except Exception as e:
        logger.error(f'Job scheduling failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/experiments/status/{experiment_id}')
async def get_experiment_status(experiment_id: str):
    """
    Get status of training job.

    Args:
        experiment_id: Experiment/job identifier

    Returns:
        Job status and details
    """
    try:
        # Get job status from scheduler
        job_status = scheduler.get_job_status(experiment_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail=f'Job not found: {experiment_id}')

        # Get detailed job info
        job_details = scheduler.get_job_details(experiment_id)

        return {
            'experiment_id': experiment_id,
            'status': job_status.value,
            'details': job_details
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Status check failed for {experiment_id}: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/experiments/cancel/{experiment_id}')
async def cancel_experiment(experiment_id: str):
    """
    Cancel queued or running training job.

    Args:
        experiment_id: Job to cancel

    Returns:
        Cancellation status
    """
    try:
        logger.info(f'Cancelling experiment {experiment_id}')

        # Cancel job in scheduler
        cancelled = scheduler.cancel_job(experiment_id)

        if not cancelled:
            raise HTTPException(status_code=404, detail=f'Job not found or already completed: {experiment_id}')

        logger.info(f'Cancelled job {experiment_id}')

        return {
            'status': 'cancelled',
            'experiment_id': experiment_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Cancellation failed for {experiment_id}: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/scheduler/queue')
async def get_scheduler_queue():
    """
    Get current scheduler queue status.

    Returns:
        Queue status with jobs and GPU allocation
    """
    try:
        scheduler_status = scheduler.get_scheduler_status()

        return {
            'queue_length': scheduler_status.queue_length,
            'running_jobs': scheduler_status.running_jobs,
            'available_gpus': scheduler_status.available_gpus,
            'gpu_statuses': [gpu.model_dump() for gpu in scheduler_status.gpu_statuses],
            'queued_jobs': [job.model_dump() for job in scheduler_status.queued_jobs],
            'timestamp': scheduler_status.timestamp
        }

    except Exception as e:
        logger.error(f'Queue status check failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/scheduler/gpus')
async def get_gpu_status():
    """
    Get GPU allocation status.

    Returns:
        GPU status for all GPUs
    """
    try:
        scheduler_status = scheduler.get_scheduler_status()

        return {
            'gpus': [gpu.model_dump() for gpu in scheduler_status.gpu_statuses],
            'available_count': scheduler_status.available_gpus,
            'timestamp': scheduler_status.timestamp
        }

    except Exception as e:
        logger.error(f'GPU status check failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/datasets/preprocess')
async def preprocess_dataset_endpoint(req: PreprocessDatasetRequest):
    """
    Preprocess dataset using specified preprocessing chain.

    Args:
        req: Preprocessing request

    Returns:
        Preprocessing results
    """
    try:
        from schemas.experiment_schemas import PreprocessingChain

        logger.info(f'Preprocessing dataset {req.dataset_id}')

        # Validate preprocessing chain
        chain = PreprocessingChain(**req.preprocessing_chain)

        # Execute preprocessing with tool governance
        result = preprocess_dataset(
            dataset_id=req.dataset_id,
            preprocessing_chain=chain,
            input_path=req.input_path,
            output_path=req.output_path,
            cycle_id=req.cycle_id
        )

        logger.info(f'Preprocessing completed for {req.dataset_id}')

        return result

    except ValidationError as e:
        logger.error(f'Preprocessing chain validation failed: {e}')
        raise HTTPException(
            status_code=400,
            detail={'error': 'validation_failed', 'issues': str(e)}
        )
    except ToolValidationError as e:
        logger.error(f'Tool validation failed: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dataset Management Endpoints
# ============================================================================

class DatasetUnpackRequest(BaseModel):
    archive_path: str
    dataset_name: str
    normalize: bool = True


class DatasetRegisterRequest(BaseModel):
    dataset_dir: str
    dataset_name: str
    push_to_dvc: bool = True


class DatasetPreprocessRequest(BaseModel):
    dataset_name: str
    input_path: str
    output_path: str


@app.post('/datasets/unpack')
async def unpack_dataset_endpoint(req: DatasetUnpackRequest):
    """
    Unpack and normalize dataset archive.

    Args:
        req: Dataset unpacking request

    Returns:
        Unpacking and normalization results
    """
    try:
        from tools.acuvue_tools import unpack_dataset_archive

        logger.info(f'Unpacking dataset: {req.dataset_name}')

        result = unpack_dataset_archive(
            archive_path=req.archive_path,
            dataset_name=req.dataset_name,
            output_dir=str(settings.home / "workspace" / "data"),
            normalize=req.normalize
        )

        logger.info(f'Dataset unpacked: {req.dataset_name}')

        return result

    except Exception as e:
        logger.error(f'Dataset unpacking failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/datasets/register')
async def register_dataset_endpoint(req: DatasetRegisterRequest):
    """
    Register dataset with DVC for version control.

    Args:
        req: Dataset registration request

    Returns:
        Registration results with SHA256 hash
    """
    try:
        from tools.dvc_tools import register_dataset_with_dvc

        logger.info(f'Registering dataset with DVC: {req.dataset_name}')

        result = register_dataset_with_dvc(
            dataset_dir=req.dataset_dir,
            dataset_name=req.dataset_name,
            push=req.push_to_dvc
        )

        logger.info(f'Dataset registered: {req.dataset_name}')

        return result

    except Exception as e:
        logger.error(f'Dataset registration failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/datasets/preprocess')
async def preprocess_dataset_full(req: DatasetPreprocessRequest):
    """
    Run full AcuVue preprocessing pipeline on dataset.

    Args:
        req: Preprocessing request

    Returns:
        Preprocessing results
    """
    try:
        from tools.acuvue_tools import run_preprocessing

        logger.info(f'Running AcuVue preprocessing: {req.dataset_name}')

        result = run_preprocessing(
            dataset_name=req.dataset_name,
            input_path=req.input_path,
            output_path=req.output_path
        )

        logger.info(f'Preprocessing complete: {req.dataset_name}')

        return result

    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/datasets/list')
async def list_datasets():
    """
    List all registered datasets.

    Returns:
        List of registered datasets with metadata
    """
    try:
        from tools.dvc_tools import list_registered_datasets

        datasets = list_registered_datasets()

        return {
            "datasets": datasets,
            "count": len(datasets)
        }

    except Exception as e:
        logger.error(f'Failed to list datasets: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/datasets/{dataset_name}/info')
async def get_dataset_info(dataset_name: str):
    """
    Get information about specific dataset.

    Args:
        dataset_name: Dataset identifier

    Returns:
        Dataset metadata
    """
    try:
        from tools.dvc_tools import get_dataset_info

        info = get_dataset_info(dataset_name)

        if info is None:
            raise HTTPException(status_code=404, detail=f'Dataset not found: {dataset_name}')

        return {
            "dataset_name": dataset_name,
            **info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to get dataset info: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/datasets/{dataset_name}/validate')
async def validate_dataset(dataset_name: str):
    """
    Validate dataset integrity by checking SHA256 hash.

    Args:
        dataset_name: Dataset to validate

    Returns:
        Validation results
    """
    try:
        from tools.dvc_tools import validate_dataset_integrity

        result = validate_dataset_integrity(dataset_name)

        return result

    except Exception as e:
        logger.error(f'Dataset validation failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Training Endpoints
# ============================================================================

class SegmentationTrainingRequest(BaseModel):
    """Request to train segmentation model."""
    dataset_path: str
    experiment_id: str
    checkpoint_dir: str
    log_dir: str
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.001
    gpu_id: Optional[int] = None
    use_wandb: bool = False
    wandb_project: str = "arc-acuvue"
    cycle_id: int = 0


class ClassificationTrainingRequest(BaseModel):
    """Request to train classification model."""
    dataset_path: str
    experiment_id: str
    checkpoint_dir: str
    log_dir: str
    model_name: str = "efficientnet_b3"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.0001
    optimizer: str = "adam"
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone_epochs: int = 5
    use_weighted_sampler: bool = True
    gpu_id: Optional[int] = None
    use_wandb: bool = False
    wandb_project: str = "arc-acuvue"
    cycle_id: int = 0


class EvaluationRequest(BaseModel):
    """Request to evaluate trained model."""
    checkpoint_path: str
    dataset_path: str
    experiment_id: str
    task_type: str = "segmentation"
    output_dir: Optional[str] = None
    batch_size: int = 16
    gpu_id: Optional[int] = None
    cycle_id: int = 0


@app.post('/train/segmentation')
async def train_segmentation_endpoint(req: SegmentationTrainingRequest):
    """
    Train U-Net segmentation model using AcuVue training pipeline.

    This calls the real AcuVue train_segmentation.py script.

    Args:
        req: Segmentation training request

    Returns:
        Training results with checkpoint path
    """
    logger.info(f'Training segmentation model: {req.experiment_id}')

    try:
        from tools.acuvue_tools import run_segmentation_training

        result = run_segmentation_training(
            dataset_path=req.dataset_path,
            experiment_id=req.experiment_id,
            checkpoint_dir=req.checkpoint_dir,
            log_dir=req.log_dir,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            gpu_id=req.gpu_id,
            use_wandb=req.use_wandb,
            wandb_project=req.wandb_project,
            cycle_id=req.cycle_id
        )

        return result

    except Exception as e:
        logger.error(f'Segmentation training failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/train/classifier')
async def train_classifier_endpoint(req: ClassificationTrainingRequest):
    """
    Train EfficientNet classification model using AcuVue training pipeline.

    This calls the real AcuVue train_classification.py script.

    Args:
        req: Classification training request

    Returns:
        Training results with checkpoint paths and metrics
    """
    logger.info(f'Training classification model: {req.experiment_id}')

    try:
        from tools.acuvue_tools import run_classification_training

        result = run_classification_training(
            dataset_path=req.dataset_path,
            experiment_id=req.experiment_id,
            checkpoint_dir=req.checkpoint_dir,
            log_dir=req.log_dir,
            model_name=req.model_name,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            optimizer=req.optimizer,
            loss_type=req.loss_type,
            focal_gamma=req.focal_gamma,
            num_classes=req.num_classes,
            pretrained=req.pretrained,
            dropout=req.dropout,
            freeze_backbone_epochs=req.freeze_backbone_epochs,
            use_weighted_sampler=req.use_weighted_sampler,
            gpu_id=req.gpu_id,
            use_wandb=req.use_wandb,
            wandb_project=req.wandb_project,
            cycle_id=req.cycle_id
        )

        return result

    except Exception as e:
        logger.error(f'Classification training failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@app.post('/eval/run')
async def evaluate_model_endpoint(req: EvaluationRequest):
    """
    Run comprehensive evaluation on trained model.

    Calculates all metrics using real AcuVue evaluation code.

    Args:
        req: Evaluation request

    Returns:
        Comprehensive evaluation metrics
    """
    logger.info(f'Evaluating model: {req.experiment_id}')

    try:
        from tools.acuvue_tools import run_full_evaluation

        result = run_full_evaluation(
            checkpoint_path=req.checkpoint_path,
            dataset_path=req.dataset_path,
            experiment_id=req.experiment_id,
            task_type=req.task_type,
            output_dir=req.output_dir,
            batch_size=req.batch_size,
            gpu_id=req.gpu_id,
            cycle_id=req.cycle_id
        )

        return result

    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job Management Endpoints
# ============================================================================

@app.get('/jobs/status/{job_id}')
async def get_job_status(job_id: str):
    """
    Get training job status.

    Args:
        job_id: Job identifier

    Returns:
        Job status and metadata
    """
    try:
        from scheduler.training_job_manager import get_job_manager

        job_manager = get_job_manager()
        job = job_manager.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return job.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to get job status: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/jobs/list')
async def list_jobs(status: Optional[str] = None):
    """
    List all training jobs, optionally filtered by status.

    Args:
        status: Filter by status (queued, running, completed, failed, cancelled)

    Returns:
        List of job records
    """
    try:
        from scheduler.training_job_manager import get_job_manager, JobStatus

        job_manager = get_job_manager()

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        jobs = job_manager.list_jobs(status_filter=status_filter)

        return {
            "jobs": [job.to_dict() for job in jobs],
            "total": len(jobs)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to list jobs: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/jobs/cancel/{job_id}')
async def cancel_job(job_id: str, rollback: bool = True):
    """
    Cancel running training job.

    Args:
        job_id: Job to cancel
        rollback: Whether to delete checkpoint artifacts (default: true)

    Returns:
        Cancellation result
    """
    logger.info(f'Cancelling job: {job_id}')

    try:
        from scheduler.training_job_manager import get_job_manager

        job_manager = get_job_manager()
        success = job_manager.cancel_job(job_id, rollback=rollback)

        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to cancel job: {job_id}")

        return {
            "status": "cancelled",
            "job_id": job_id,
            "rollback": rollback
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to cancel job: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/jobs/resume/{job_id}')
async def resume_job(job_id: str):
    """
    Resume cancelled or failed training job.

    Args:
        job_id: Job to resume

    Returns:
        Resume result
    """
    logger.info(f'Resuming job: {job_id}')

    try:
        from scheduler.training_job_manager import get_job_manager

        job_manager = get_job_manager()
        success = job_manager.resume_job(job_id)

        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to resume job: {job_id}")

        return {
            "status": "resuming",
            "job_id": job_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to resume job: {e}')
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # Ensure directories exist using config
    settings.ensure_directories()

    # Start server
    uvicorn.run(app, host='0.0.0.0', port=8002, log_level='info')
