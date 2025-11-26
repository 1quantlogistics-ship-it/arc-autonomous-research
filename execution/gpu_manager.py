"""
GPU Manager - GPU Memory Estimation and Monitoring

Provides GPU memory pre-flight checks and monitoring:
- Memory estimation before experiment start
- Real-time GPU usage monitoring
- Multi-GPU management

Author: ARC Team (Dev 1)
Created: 2025-11-26
"""

import logging
import subprocess
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """
    Information about a single GPU.

    Attributes:
        index: GPU index
        name: GPU model name
        memory_total_mb: Total memory in MB
        memory_used_mb: Used memory in MB
        memory_free_mb: Free memory in MB
        utilization_percent: GPU utilization percentage
        temperature_c: Temperature in Celsius
    """
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: float = 0.0
    temperature_c: int = 0

    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_usage_percent": round(self.memory_usage_percent, 1),
            "utilization_percent": self.utilization_percent,
            "temperature_c": self.temperature_c,
        }


@dataclass
class MemoryEstimate:
    """
    Memory estimate for an experiment configuration.

    Attributes:
        model_memory_mb: Estimated model parameter memory
        optimizer_memory_mb: Estimated optimizer state memory
        activation_memory_mb: Estimated activation memory (varies with batch size)
        buffer_memory_mb: Additional buffer for framework overhead
        total_estimated_mb: Total estimated memory
        confidence: Estimate confidence (low/medium/high)
    """
    model_memory_mb: int
    optimizer_memory_mb: int
    activation_memory_mb: int
    buffer_memory_mb: int = 500  # Default 500MB buffer
    confidence: str = "medium"

    @property
    def total_estimated_mb(self) -> int:
        """Total estimated memory."""
        return (
            self.model_memory_mb +
            self.optimizer_memory_mb +
            self.activation_memory_mb +
            self.buffer_memory_mb
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_memory_mb": self.model_memory_mb,
            "optimizer_memory_mb": self.optimizer_memory_mb,
            "activation_memory_mb": self.activation_memory_mb,
            "buffer_memory_mb": self.buffer_memory_mb,
            "total_estimated_mb": self.total_estimated_mb,
            "confidence": self.confidence,
        }


class GPUManager:
    """
    Manages GPU resources and memory estimation.

    Features:
    - Query GPU status via nvidia-smi
    - Estimate memory requirements for experiments
    - Pre-flight checks before experiment start
    - GPU selection for experiments
    """

    # Memory multipliers for common architectures
    PARAM_MEMORY_BYTES = 4  # float32
    OPTIMIZER_MULTIPLIER = 2  # Adam stores m and v
    ACTIVATION_MULTIPLIER = 2  # Typical activation memory relative to model

    # Common model parameter counts (millions)
    MODEL_PARAM_ESTIMATES = {
        "resnet18": 11.7,
        "resnet34": 21.8,
        "resnet50": 25.6,
        "resnet101": 44.5,
        "resnet152": 60.2,
        "efficientnet_b0": 5.3,
        "efficientnet_b1": 7.8,
        "efficientnet_b2": 9.2,
        "efficientnet_b3": 12.0,
        "efficientnet_b4": 19.0,
        "efficientnet_b5": 30.0,
        "vit_small": 22.0,
        "vit_base": 86.0,
        "vit_large": 307.0,
    }

    def __init__(self, memory_buffer_percent: float = 0.15):
        """
        Initialize GPU manager.

        Args:
            memory_buffer_percent: Safety buffer as percentage of GPU memory
        """
        self.memory_buffer_percent = memory_buffer_percent
        self._nvidia_smi_available = self._check_nvidia_smi()

        if self._nvidia_smi_available:
            logger.info("GPUManager initialized with nvidia-smi support")
        else:
            logger.warning("nvidia-smi not available, GPU features limited")

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_gpu_info(self) -> List[GPUInfo]:
        """
        Get information about all available GPUs.

        Returns:
            List of GPUInfo for each GPU
        """
        if not self._nvidia_smi_available:
            return []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {result.stderr}")
                return []

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_used_mb=int(parts[3]),
                        memory_free_mb=int(parts[4]),
                        utilization_percent=float(parts[5]) if parts[5] != '[N/A]' else 0,
                        temperature_c=int(parts[6]) if parts[6] != '[N/A]' else 0,
                    ))

            return gpus

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return []

    def get_gpu(self, index: int) -> Optional[GPUInfo]:
        """Get info for a specific GPU."""
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if gpu.index == index:
                return gpu
        return None

    def estimate_memory(
        self,
        config: Dict[str, Any],
    ) -> MemoryEstimate:
        """
        Estimate GPU memory requirements for a configuration.

        Args:
            config: Training configuration containing:
                - model_type: Model architecture name
                - batch_size: Training batch size
                - image_size: Input image size
                - num_params: Optional explicit parameter count

        Returns:
            MemoryEstimate with breakdown
        """
        # Get parameter count
        model_type = config.get("model_type", "resnet50").lower()
        num_params_millions = config.get("num_params")

        if num_params_millions is None:
            # Look up from known models
            for key, params in self.MODEL_PARAM_ESTIMATES.items():
                if key in model_type:
                    num_params_millions = params
                    break

            if num_params_millions is None:
                # Default estimate based on typical medical imaging models
                num_params_millions = 25.0
                confidence = "low"
            else:
                confidence = "medium"
        else:
            confidence = "high"

        # Calculate memory components
        num_params = int(num_params_millions * 1e6)

        # Model parameters (float32)
        model_memory_mb = (num_params * self.PARAM_MEMORY_BYTES) // (1024 * 1024)

        # Optimizer states (Adam has 2x model size for momentum)
        optimizer = config.get("optimizer", "adam").lower()
        if optimizer in ["adam", "adamw"]:
            optimizer_memory_mb = model_memory_mb * self.OPTIMIZER_MULTIPLIER
        elif optimizer == "sgd":
            optimizer_memory_mb = model_memory_mb // 2  # Just momentum
        else:
            optimizer_memory_mb = model_memory_mb

        # Activation memory (depends on batch size and image size)
        batch_size = config.get("batch_size", 32)
        image_size = config.get("image_size", 224)

        # Simplified activation estimate
        # Actual depends on architecture, but this gives reasonable ballpark
        activation_per_sample_mb = (image_size * image_size * 3 * 4) // (1024 * 1024)  # RGB float32
        activation_per_sample_mb *= self.ACTIVATION_MULTIPLIER  # Account for intermediate activations
        activation_memory_mb = activation_per_sample_mb * batch_size

        # Buffer for PyTorch/framework overhead
        buffer_mb = int(500 + (model_memory_mb * 0.1))  # Base + 10% model size

        return MemoryEstimate(
            model_memory_mb=model_memory_mb,
            optimizer_memory_mb=optimizer_memory_mb,
            activation_memory_mb=activation_memory_mb,
            buffer_memory_mb=buffer_mb,
            confidence=confidence,
        )

    def preflight_check(
        self,
        config: Dict[str, Any],
        gpu_index: int = 0,
    ) -> Tuple[bool, str]:
        """
        Perform pre-flight check before starting experiment.

        Args:
            config: Training configuration
            gpu_index: Target GPU index

        Returns:
            (success, message) tuple
        """
        # Get GPU info
        gpu = self.get_gpu(gpu_index)
        if gpu is None:
            if not self._nvidia_smi_available:
                return True, "GPU monitoring not available, proceeding without check"
            return False, f"GPU {gpu_index} not found"

        # Estimate memory requirement
        estimate = self.estimate_memory(config)

        # Check available memory with buffer
        required_with_buffer = int(
            estimate.total_estimated_mb * (1 + self.memory_buffer_percent)
        )

        if gpu.memory_free_mb < required_with_buffer:
            return False, (
                f"Insufficient GPU memory. "
                f"Required: {required_with_buffer}MB (estimated {estimate.total_estimated_mb}MB + {self.memory_buffer_percent*100:.0f}% buffer), "
                f"Available: {gpu.memory_free_mb}MB on GPU {gpu_index}"
            )

        # Check GPU temperature
        if gpu.temperature_c > 85:
            return False, f"GPU {gpu_index} temperature too high: {gpu.temperature_c}°C"

        return True, (
            f"Pre-flight check passed. "
            f"Estimated: {estimate.total_estimated_mb}MB ({estimate.confidence} confidence), "
            f"Available: {gpu.memory_free_mb}MB on GPU {gpu_index} ({gpu.name})"
        )

    def select_best_gpu(
        self,
        required_memory_mb: Optional[int] = None
    ) -> Optional[int]:
        """
        Select the best available GPU for training.

        Prefers GPU with most free memory that meets requirements.

        Args:
            required_memory_mb: Minimum required memory

        Returns:
            GPU index or None if no suitable GPU found
        """
        gpus = self.get_gpu_info()
        if not gpus:
            return None

        # Filter by required memory
        if required_memory_mb is not None:
            gpus = [g for g in gpus if g.memory_free_mb >= required_memory_mb]

        if not gpus:
            return None

        # Select GPU with most free memory
        best_gpu = max(gpus, key=lambda g: g.memory_free_mb)
        return best_gpu.index

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of all GPU status.

        Returns:
            Dictionary with GPU status information
        """
        gpus = self.get_gpu_info()

        if not gpus:
            return {
                "available": False,
                "message": "No GPUs detected" if self._nvidia_smi_available else "nvidia-smi not available",
                "gpus": [],
            }

        total_memory = sum(g.memory_total_mb for g in gpus)
        total_used = sum(g.memory_used_mb for g in gpus)
        total_free = sum(g.memory_free_mb for g in gpus)

        return {
            "available": True,
            "gpu_count": len(gpus),
            "total_memory_mb": total_memory,
            "used_memory_mb": total_used,
            "free_memory_mb": total_free,
            "usage_percent": round((total_used / total_memory) * 100, 1) if total_memory > 0 else 0,
            "gpus": [g.to_dict() for g in gpus],
        }


# Singleton instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get or create the global GPUManager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def reset_gpu_manager() -> None:
    """Reset the global GPUManager instance (for testing)."""
    global _gpu_manager
    _gpu_manager = None
