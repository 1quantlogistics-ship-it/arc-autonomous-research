"""
Distributed training utilities for multi-GPU setups.

Phase F - Infrastructure & Stability Track
"""
import os
import logging
from typing import Optional

# Conditional torch imports
try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define stubs for type hints
    DDP = None
    DistributedSampler = None

from config.gpu_config import GPUConfig, get_gpu_config

logger = logging.getLogger(__name__)


def setup_distributed(rank: int = 0, world_size: int = 1, backend: str = "nccl") -> bool:
    """
    Initialize distributed training environment.

    For RunPod: This is called automatically when using torchrun or
    the distributed launcher.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend (nccl, gloo, mpi)

    Returns:
        True if distributed setup succeeded
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - skipping distributed setup")
        return False

    if world_size <= 1:
        logger.info("Single GPU mode - skipping distributed setup")
        return False

    # RunPod sets these automatically with torchrun
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(rank)
        logger.info(f"Distributed setup complete: rank {rank}/{world_size}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training."""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model,
    device_id: int,
    find_unused_parameters: bool = False
):
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: PyTorch model
        device_id: GPU device ID for this process
        find_unused_parameters: Set True if model has unused params

    Returns:
        DDP-wrapped model
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - returning unwrapped model")
        return model

    model = model.to(f"cuda:{device_id}")

    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters
    )


def get_distributed_sampler(
    dataset,
    shuffle: bool = True,
    seed: int = 42
) -> Optional['DistributedSampler']:
    """
    Get sampler for distributed training.

    Args:
        dataset: PyTorch dataset
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling

    Returns:
        DistributedSampler or None if not in distributed mode
    """
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return None

    return DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        seed=seed
    )


class DistributedTrainer:
    """
    High-level trainer that handles distributed setup automatically.
    """

    def __init__(self, model, config: GPUConfig = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DistributedTrainer")

        self.config = config or get_gpu_config()
        self.model = model
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1

    def setup(self) -> 'DistributedTrainer':
        """Initialize training environment."""
        if self.config.use_distributed and self.config.device_count > 1:
            # Get rank from environment (set by torchrun)
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = self.config.device_count

            self.is_distributed = setup_distributed(
                rank=self.rank,
                world_size=self.world_size,
                backend=self.config.backend.value
            )

            if self.is_distributed:
                self.model = wrap_model_ddp(self.model, self.rank)
                logger.info(f"Model wrapped with DDP on rank {self.rank}")
        else:
            # Single GPU or CPU
            device = self.config.primary_device
            self.model = self.model.to(device)
            logger.info(f"Model loaded on {device}")

        return self

    def cleanup(self):
        """Clean up resources."""
        cleanup_distributed()

    def is_main_process(self) -> bool:
        """Check if this is the main process (for logging, checkpointing)."""
        return self.rank == 0

    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint (only on main process)."""
        if not self.is_main_process():
            return

        state = {
            'model_state_dict': (
                self.model.module.state_dict()
                if self.is_distributed
                else self.model.state_dict()
            ),
            **kwargs
        }
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.primary_device)

        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint

    def synchronize(self):
        """Synchronize all processes."""
        if self.is_distributed and TORCH_AVAILABLE:
            dist.barrier()


def get_world_info() -> dict:
    """
    Get information about the distributed world.

    Returns:
        Dict with rank, world_size, and backend info
    """
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return {
            'distributed': False,
            'rank': 0,
            'world_size': 1,
            'backend': None
        }

    return {
        'distributed': True,
        'rank': dist.get_rank(),
        'world_size': dist.get_world_size(),
        'backend': dist.get_backend()
    }
