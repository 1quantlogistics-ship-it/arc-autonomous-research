"""
Execution Module - Bulletproof Training Infrastructure

Provides subprocess isolation for training jobs to ensure:
- Training crashes cannot kill the ARC system
- Timeout enforcement with SIGTERM→SIGKILL escalation
- Emergency checkpointing on crash/timeout
- GPU memory pre-validation before experiment start

Dev 1 Components:
- SubprocessExecutor: Process isolation and lifecycle management
- TrainingHarness: Training loop with graceful shutdown
- GPUManager: GPU memory estimation and monitoring
- IPCProtocol: JSON file-based inter-process communication

Dev 2 Components:
- ExperimentLifecycle: State machine and registry for experiments
- MetricsStreamer: Live metrics streaming with callbacks

Author: ARC Team
Created: 2025-11-26
"""

# Dev 1 components (conditional - may not exist yet)
try:
    from execution.ipc_protocol import (
        IPCProtocol,
        IPCMessage,
        MessageType,
    )
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False

try:
    from execution.subprocess_executor import (
        SubprocessExecutor,
        ExecutionResult,
        ExecutionStatus,
    )
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False

try:
    from execution.training_harness import (
        TrainingHarness,
        TrainingConfig,
        TrainingState,
    )
    TRAINING_HARNESS_AVAILABLE = True
except ImportError:
    TRAINING_HARNESS_AVAILABLE = False

try:
    from execution.gpu_manager import (
        GPUManager,
        GPUInfo,
        MemoryEstimate,
    )
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False

# Dev 2 components
from execution.experiment_lifecycle import (
    ExperimentState,
    ExperimentRecord,
    ExperimentRegistry,
    get_experiment_registry,
    reset_experiment_registry,
)
from execution.metrics_streamer import (
    MetricsStreamer,
    MetricsCallback,
    MetricsSnapshot,
    MetricsWindow,
    get_metrics_streamer,
    reset_metrics_streamer,
)

__all__ = [
    # Dev 2: Experiment lifecycle
    "ExperimentState",
    "ExperimentRecord",
    "ExperimentRegistry",
    "get_experiment_registry",
    "reset_experiment_registry",
    # Dev 2: Metrics streaming
    "MetricsStreamer",
    "MetricsCallback",
    "MetricsSnapshot",
    "MetricsWindow",
    "get_metrics_streamer",
    "reset_metrics_streamer",
]

# Conditionally add Dev 1 exports
if IPC_AVAILABLE:
    __all__.extend(["IPCProtocol", "IPCMessage", "MessageType"])

if SUBPROCESS_AVAILABLE:
    __all__.extend(["SubprocessExecutor", "ExecutionResult", "ExecutionStatus"])

if TRAINING_HARNESS_AVAILABLE:
    __all__.extend(["TrainingHarness", "TrainingConfig", "TrainingState"])

if GPU_MANAGER_AVAILABLE:
    __all__.extend(["GPUManager", "GPUInfo", "MemoryEstimate"])
