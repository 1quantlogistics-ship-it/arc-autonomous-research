"""
GPU configuration for distributed training.
Compatible with RunPod multi-GPU instances.

Phase F - Infrastructure & Stability Track
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# Conditional torch import for environments without GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DistributedBackend(Enum):
    NCCL = "nccl"      # Best for GPU-GPU (RunPod default)
    GLOO = "gloo"      # Fallback, works everywhere
    MPI = "mpi"        # For HPC environments


@dataclass
class GPUConfig:
    """Configuration for GPU training."""

    # Distributed training settings
    use_distributed: bool = True  # Use DDP when multiple GPUs available
    backend: DistributedBackend = DistributedBackend.NCCL

    # Memory management
    max_memory_per_gpu_gb: float = 5.0  # From existing constraints
    gradient_checkpointing: bool = True  # Save memory
    mixed_precision: bool = True  # FP16 training

    # RunPod specific
    runpod_mode: bool = field(default_factory=lambda: os.getenv("RUNPOD_POD_ID") is not None)

    def __post_init__(self):
        # Auto-detect device count
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._device_count = torch.cuda.device_count()
        else:
            self._device_count = 0

        if self._device_count <= 1:
            self.use_distributed = False

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def primary_device(self) -> str:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @property
    def available_devices(self) -> List[str]:
        return [f"cuda:{i}" for i in range(self._device_count)]

    def get_device_for_rank(self, rank: int) -> str:
        """Get device for given process rank in distributed setup."""
        if self._device_count == 0:
            return "cpu"
        return f"cuda:{rank % self._device_count}"


def get_gpu_config() -> GPUConfig:
    """Get GPU configuration, respecting environment overrides."""
    config = GPUConfig()

    # Environment overrides for RunPod
    if os.getenv("DISABLE_DISTRIBUTED"):
        config.use_distributed = False
    if os.getenv("GPU_MEMORY_LIMIT"):
        config.max_memory_per_gpu_gb = float(os.getenv("GPU_MEMORY_LIMIT"))

    return config


def validate_gpu_memory(model, config: GPUConfig) -> bool:
    """
    Check if model fits within memory constraints.

    Args:
        model: PyTorch model
        config: GPU configuration

    Returns:
        True if model fits within memory limits
    """
    if not TORCH_AVAILABLE:
        return True

    param_memory_gb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 3)

    # Rough estimate: params + gradients + optimizer states = 4x params
    estimated_total = param_memory_gb * 4

    return estimated_total <= config.max_memory_per_gpu_gb


def get_gpu_memory_info() -> List[dict]:
    """
    Get current GPU memory usage for all GPUs.

    Returns:
        List of dicts with memory info per GPU
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return []

    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

        info.append({
            'device': i,
            'name': torch.cuda.get_device_name(i),
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'total_gb': round(total, 2),
            'free_gb': round(total - allocated, 2),
            'utilization_pct': round((allocated / total) * 100, 1) if total > 0 else 0
        })

    return info
