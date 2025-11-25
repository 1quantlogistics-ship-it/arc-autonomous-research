"""
Tests for distributed training utilities.

Phase F - Infrastructure & Stability Track
"""
import pytest
import os

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.gpu_config import GPUConfig, get_gpu_config, validate_gpu_memory, get_gpu_memory_info


def test_gpu_config_creation():
    """GPUConfig should be creatable regardless of GPU availability."""
    config = GPUConfig()
    assert isinstance(config.device_count, int)
    assert config.device_count >= 0


def test_gpu_config_primary_device():
    """Primary device should be cuda:0 or cpu."""
    config = get_gpu_config()
    assert config.primary_device in ["cuda:0", "cpu"]


def test_gpu_config_environment_override():
    """Environment variables should override config."""
    os.environ["DISABLE_DISTRIBUTED"] = "1"
    os.environ["GPU_MEMORY_LIMIT"] = "8.0"

    try:
        config = get_gpu_config()
        assert config.use_distributed is False
        assert config.max_memory_per_gpu_gb == 8.0
    finally:
        del os.environ["DISABLE_DISTRIBUTED"]
        del os.environ["GPU_MEMORY_LIMIT"]


def test_gpu_config_device_for_rank():
    """get_device_for_rank should return valid device string."""
    config = GPUConfig()

    if config.device_count == 0:
        assert config.get_device_for_rank(0) == "cpu"
    else:
        assert config.get_device_for_rank(0) == "cuda:0"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_validate_gpu_memory_small_model():
    """Small model should pass memory validation."""
    small_model = torch.nn.Linear(10, 10)
    config = GPUConfig()
    config.max_memory_per_gpu_gb = 5.0
    assert validate_gpu_memory(small_model, config)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_get_gpu_memory_info():
    """get_gpu_memory_info should return list."""
    info = get_gpu_memory_info()
    assert isinstance(info, list)

    if torch.cuda.is_available():
        assert len(info) > 0
        assert 'device' in info[0]
        assert 'total_gb' in info[0]


# Tests for distributed module
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_distributed_trainer_creation():
    """DistributedTrainer should be creatable."""
    from tools.distributed import DistributedTrainer

    model = torch.nn.Linear(10, 10)
    config = GPUConfig()
    config.use_distributed = False  # Force single-GPU mode for test

    trainer = DistributedTrainer(model, config)
    assert trainer.rank == 0
    assert trainer.world_size == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_distributed_trainer_setup_single_gpu():
    """DistributedTrainer should work in single-GPU mode."""
    from tools.distributed import DistributedTrainer

    model = torch.nn.Linear(10, 10)
    config = GPUConfig()
    config.use_distributed = False

    trainer = DistributedTrainer(model, config)
    trainer.setup()

    assert trainer.is_main_process()
    assert not trainer.is_distributed

    trainer.cleanup()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_get_world_info():
    """get_world_info should return valid info."""
    from tools.distributed import get_world_info

    info = get_world_info()
    assert 'distributed' in info
    assert 'rank' in info
    assert 'world_size' in info


def test_distributed_backend_enum():
    """DistributedBackend enum should have expected values."""
    from config.gpu_config import DistributedBackend

    assert DistributedBackend.NCCL.value == "nccl"
    assert DistributedBackend.GLOO.value == "gloo"
    assert DistributedBackend.MPI.value == "mpi"


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs"
)
def test_distributed_trainer_multi_gpu():
    """Test distributed setup with multiple GPUs (requires 2+ GPUs)."""
    # This test would run in CI with multi-GPU runners
    pass
