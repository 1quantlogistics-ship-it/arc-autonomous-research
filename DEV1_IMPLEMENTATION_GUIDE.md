# Dev 1 Implementation Guide
## ARC Infrastructure & GPU Optimization Tasks

**Status**: Phase 1.1 Complete | In Progress
**Environment**: Mac Development (GPU features to be tested on RunPod)
**Branch**: `feature/dev1-gpu-optimization` (recommended)

---

## ✅ COMPLETED: Phase 1.1 - Historian Timeout Fix

### Changes Made:

1. **config.py** - Added historian-specific timeout:
```python
historian_timeout: int = Field(
    default=600,
    ge=60,
    le=1800,
    description="Historian LLM timeout in seconds (longer for deep reasoning)"
)
```

2. **llm/router.py** - Role-specific timeout support:
- Added `from config import get_settings`
- Modified `get_client_for_role()` to pass role to `_create_client()`
- Modified `_create_client()` to accept `role` parameter
- Implemented role-specific timeout:
```python
settings = get_settings()
timeout = settings.historian_timeout if role == "historian" else settings.llm_timeout
```

3. **.env.production** - Added configuration:
```bash
ARC_LLM_TIMEOUT=120
ARC_HISTORIAN_TIMEOUT=600
ARC_LLM_MAX_RETRIES=3
ARC_LLM_RETRY_DELAY=2.0
```

### Testing (Mac):
```python
# Test timeout configuration loads
from config import get_settings
settings = get_settings()
assert settings.historian_timeout == 600
assert settings.llm_timeout == 120
```

---

## 🔄 IN PROGRESS: Phase 1.2 - Retry-on-Timeout Logic

### Implementation Plan:

**File to Modify**: `api/multi_agent_orchestrator.py`

**Approach**: Wrap Historian calls in retry decorator

```python
from functools import wraps
import time

def retry_on_timeout(max_retries=3, initial_delay=2.0):
    """Retry decorator with exponential backoff for timeout errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TimeoutError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"Timeout on attempt {attempt + 1}, retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed after {max_retries} attempts")
                        raise
                except Exception as e:
                    # Non-timeout errors: fail immediately
                    raise

            raise last_exception
        return wrapper
    return decorator
```

**Usage in Orchestrator**:
```python
@retry_on_timeout(max_retries=3)
def _run_historian_update(self, cycle_id: int):
    """Run historian update with automatic retry on timeout."""
    return self.historian.process(cycle_id=cycle_id)
```

**Testing (Mac)**:
```python
# Mock timeout error to test retry
def mock_historian_with_timeout():
    raise TimeoutError("LLM request timed out")

# Should retry 3 times
result = retry_on_timeout(max_retries=3)(mock_historian_with_timeout)
```

---

## 📋 TODO: Phase 1.3 - Continuous Research Script

### Create: `scripts/run_continuous_research.py`

```python
#!/usr/bin/env python3
"""
Continuous Research Loop for ARC
Replaces bash script with robust Python error handling
"""

import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.multi_agent_orchestrator import MultiAgentOrchestrator
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinuousResearchLoop:
    """
    Runs ARC in continuous autonomous mode with auto-restart on failures.
    """

    def __init__(
        self,
        memory_path: Optional[str] = None,
        max_cycles: Optional[int] = None,
        dry_run: bool = False
    ):
        self.memory_path = memory_path
        self.max_cycles = max_cycles
        self.dry_run = dry_run
        self.should_stop = False
        self.cycle_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown on SIGTERM/SIGINT."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.should_stop = True

    def run(self):
        """Main continuous research loop."""
        logger.info("Starting continuous research loop...")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Max cycles: {self.max_cycles or 'unlimited'}")

        while not self.should_stop:
            if self.max_cycles and self.cycle_count >= self.max_cycles:
                logger.info(f"Reached max cycles ({self.max_cycles}), stopping.")
                break

            try:
                self.cycle_count += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"CYCLE {self.cycle_count} - {datetime.now().isoformat()}")
                logger.info(f"{'='*80}\n")

                if self.dry_run:
                    logger.info("[DRY RUN] Would execute research cycle here")
                    time.sleep(2)  # Simulate cycle
                else:
                    self._run_cycle()

                logger.info(f"Cycle {self.cycle_count} completed successfully")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping...")
                break

            except Exception as e:
                logger.error(f"Cycle {self.cycle_count} failed: {e}")
                logger.error("Auto-restarting in 10 seconds...")
                time.sleep(10)
                # Continue to next cycle (auto-restart)

        logger.info(f"Continuous research loop stopped after {self.cycle_count} cycles")

    def _run_cycle(self):
        """Execute a single research cycle."""
        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator(
            memory_path=self.memory_path,
            offline_mode=False
        )

        # Run autonomous cycle
        result = orchestrator.run_autonomous_cycle(
            cycle_id=self.cycle_count,
            wait_for_completion=True,
            timeout=3600
        )

        logger.info(f"Cycle result: {result.get('status')}")
        logger.info(f"Experiments run: {len(result.get('experiments', []))}")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ARC Continuous Research Loop")
    parser.add_argument('--memory-path', type=str, help="Memory directory path")
    parser.add_argument('--max-cycles', type=int, help="Maximum number of cycles")
    parser.add_argument('--dry-run', action='store_true', help="Dry run mode (no actual execution)")

    args = parser.parse_args()

    loop = ContinuousResearchLoop(
        memory_path=args.memory_path,
        max_cycles=args.max_cycles,
        dry_run=args.dry_run
    )

    try:
        loop.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Dry run test (Mac)
python scripts/run_continuous_research.py --dry-run --max-cycles 5

# Full mode (RunPod)
python scripts/run_continuous_research.py --max-cycles 100

# Infinite mode
python scripts/run_continuous_research.py
```

---

## 📋 TODO: Phase 2 - GPU Infrastructure Code

### Task 2.1: Multi-GPU Training Support

**File**: `tools/acuvue_tools.py`

Add multi-GPU parameter to training functions:

```python
def run_classification_training(
    job_config: TrainingJobConfig,
    cycle_id: int,
    gpu_ids: Optional[List[int]] = None,  # NEW: Multi-GPU support
    enable_deepspeed: bool = False,        # NEW: DeepSpeed flag
    wait_for_completion: bool = True,
    dummy_mode: bool = False
) -> Dict[str, Any]:
    """
    Run classification training with multi-GPU support.
    """
    if gpu_ids is None:
        gpu_ids = [job_config.gpu_id]

    # Configure Hydra for multi-GPU
    hydra_config = _create_hydra_config(
        job_config=job_config,
        gpu_ids=gpu_ids,
        enable_deepspeed=enable_deepspeed
    )

    # Rest of implementation...
```

**Hydra Config Generation**:
```python
def _create_hydra_config(
    job_config: TrainingJobConfig,
    gpu_ids: List[int],
    enable_deepspeed: bool = False
) -> Dict[str, Any]:
    """Create Hydra config with multi-GPU support."""

    # Auto-scale batch size by GPU count
    settings = get_settings()
    effective_batch_size = settings.base_batch_size * len(gpu_ids)

    config = {
        "training": {
            "batch_size": effective_batch_size,
            "epochs": settings.default_epochs,  # Use config default (5)
            "devices": gpu_ids if len(gpu_ids) > 1 else gpu_ids[0],
            "strategy": "ddp" if len(gpu_ids) > 1 else None,
        },
        "system": {
            "device": "cuda" if gpu_ids else "cpu",
            "num_gpus": len(gpu_ids),
        }
    }

    if enable_deepspeed:
        config["training"]["strategy"] = "deepspeed"
        config["deepspeed"] = {
            "stage": 2,
            "offload_optimizer": False,
            "offload_parameters": False,
        }

    return config
```

### Task 2.2: GPU Allocation Policy

**File**: `scheduler/job_scheduler.py`

```python
# GPU allocation constants
TRAINING_GPUS = [0, 1]  # GPUs for training
INFERENCE_GPU = 2        # GPU reserved for LLM

class JobScheduler:
    def allocate_gpu(self, job_type: str = "training") -> int:
        """
        Allocate GPU based on job type.

        Args:
            job_type: "training" or "inference"

        Returns:
            GPU ID
        """
        if job_type == "inference":
            return INFERENCE_GPU

        # For training: find least loaded GPU from training pool
        gpu_loads = self._get_gpu_loads(TRAINING_GPUS)
        return min(gpu_loads, key=gpu_loads.get)

    def _get_gpu_loads(self, gpu_ids: List[int]) -> Dict[int, int]:
        """Get current job count per GPU."""
        loads = {gpu_id: 0 for gpu_id in gpu_ids}

        for job in self.active_jobs:
            if job.gpu_id in loads:
                loads[job.gpu_id] += 1

        return loads
```

### Task 2.3: Parallel Job Execution

**File**: `scheduler/training_job_manager.py`

Increase worker pool size:

```python
def __init__(self, max_concurrent_jobs: int = 3):  # Increase from 2 to 3
    self.max_concurrent_jobs = max_concurrent_jobs
    self.worker_threads = []

    # Start worker threads
    for i in range(max_concurrent_jobs):
        worker = threading.Thread(
            target=self._worker_loop,
            name=f"JobWorker-{i}",
            daemon=True
        )
        worker.start()
        self.worker_threads.append(worker)
```

### Task 2.4: Configuration Updates

**File**: `config.py`

```python
# Add these fields to ARCSettings class:

min_epochs: int = Field(
    default=3,
    ge=1,
    le=100,
    description="Minimum training epochs"
)

default_epochs: int = Field(
    default=5,
    ge=1,
    le=100,
    description="Default training epochs"
)

base_batch_size: int = Field(
    default=16,
    ge=1,
    le=256,
    description="Base batch size (before GPU scaling)"
)

batch_size_per_gpu: int = Field(
    default=16,
    ge=1,
    le=128,
    description="Batch size multiplier per GPU"
)
```

---

## 📋 TODO: Phase 3 - GPU Monitoring

### Create: `tools/gpu_monitor.py`

```python
"""
GPU Monitoring with Mock Mode for Mac Development
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import random

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU statistics."""
    gpu_id: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int
    temperature_c: int
    fan_speed_percent: int
    power_usage_w: float


class GPUMonitor:
    """
    Monitor GPU statistics with automatic fallback to mock mode.
    """

    def __init__(self, mock_mode: Optional[bool] = None):
        """
        Initialize GPU monitor.

        Args:
            mock_mode: Force mock mode (None = auto-detect)
        """
        if mock_mode is None:
            # Auto-detect: use mock if pynvml not available
            self.mock_mode = not PYNVML_AVAILABLE
        else:
            self.mock_mode = mock_mode

        if not self.mock_mode:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU Monitor initialized: {self.device_count} GPUs detected")
            except Exception as e:
                logger.warning(f"Failed to initialize pynvml: {e}, using mock mode")
                self.mock_mode = True
                self.device_count = 3  # Mock 3 GPUs
        else:
            self.device_count = 3  # Mock 3 GPUs
            logger.info("GPU Monitor initialized in mock mode")

    def get_gpu_stats(self) -> List[GPUStats]:
        """
        Get statistics for all GPUs.

        Returns:
            List of GPUStats
        """
        if self.mock_mode:
            return self._mock_gpu_stats()
        else:
            return self._real_gpu_stats()

    def _real_gpu_stats(self) -> List[GPUStats]:
        """Get real GPU stats from nvidia-smi."""
        stats = []

        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used // (1024 ** 2)  # MB
                memory_total = mem_info.total // (1024 ** 2)  # MB

                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Get fan speed (may not be available on all GPUs)
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan = 0

                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0

                # Get name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                stats.append(GPUStats(
                    gpu_id=i,
                    name=name,
                    memory_used_mb=memory_used,
                    memory_total_mb=memory_total,
                    utilization_percent=util.gpu,
                    temperature_c=temp,
                    fan_speed_percent=fan,
                    power_usage_w=power
                ))

            except Exception as e:
                logger.error(f"Error reading GPU {i}: {e}")

        return stats

    def _mock_gpu_stats(self) -> List[GPUStats]:
        """Generate mock GPU stats for development."""
        return [
            GPUStats(
                gpu_id=0,
                name="NVIDIA A40 (Mock)",
                memory_used_mb=random.randint(5000, 30000),
                memory_total_mb=45634,
                utilization_percent=random.randint(0, 100),
                temperature_c=random.randint(40, 75),
                fan_speed_percent=random.randint(30, 80),
                power_usage_w=random.uniform(50.0, 300.0)
            ),
            GPUStats(
                gpu_id=1,
                name="NVIDIA A40 (Mock)",
                memory_used_mb=random.randint(5000, 30000),
                memory_total_mb=45634,
                utilization_percent=random.randint(0, 100),
                temperature_c=random.randint(40, 75),
                fan_speed_percent=random.randint(30, 80),
                power_usage_w=random.uniform(50.0, 300.0)
            ),
            GPUStats(
                gpu_id=2,
                name="NVIDIA A40 (Mock)",
                memory_used_mb=random.randint(2000, 10000),
                memory_total_mb=45634,
                utilization_percent=random.randint(0, 50),  # LLM GPU - lower usage
                temperature_c=random.randint(40, 65),
                fan_speed_percent=random.randint(20, 60),
                power_usage_w=random.uniform(30.0, 150.0)
            )
        ]

    def shutdown(self):
        """Cleanup GPU monitor."""
        if not self.mock_mode:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Global instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(mock_mode: Optional[bool] = None) -> GPUMonitor:
    """Get global GPU monitor instance."""
    global _gpu_monitor

    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(mock_mode=mock_mode)

    return _gpu_monitor
```

**Add to requirements.txt**:
```
nvidia-ml-py3>=12.535.108
```

**Testing on Mac**:
```python
from tools.gpu_monitor import get_gpu_monitor

# Will automatically use mock mode
monitor = get_gpu_monitor()
stats = monitor.get_gpu_stats()

for gpu in stats:
    print(f"GPU {gpu.gpu_id}: {gpu.name}")
    print(f"  Util: {gpu.utilization_percent}%")
    print(f"  VRAM: {gpu.memory_used_mb}/{gpu.memory_total_mb} MB")
    print(f"  Temp: {gpu.temperature_c}°C")
```

---

## 📋 Remaining Implementation Tasks

### Phase 4: Async Cycle Timing
- Convert Historian to async
- Add cycle pipelining
- Reduce poll intervals

### Phase 5: FDA Documentation Export
- Add provenance tracking
- Create export functionality
- Generate compliance reports

### Phase 6: Testing & Documentation
- Write unit tests
- Update deployment docs
- Test on RunPod

---

## Testing Strategy

### Mac Development:
✅ Configuration loading
✅ Timeout settings
✅ Mock GPU monitoring
✅ Retry logic
✅ Continuous loop (dry-run)
✅ Export functionality

### RunPod Validation:
- Real GPU detection
- Multi-GPU training
- Parallel job execution
- GPU utilization metrics
- Performance benchmarks

---

## Deployment Checklist

- [ ] Complete all phases
- [ ] Run unit tests
- [ ] Update .env.production
- [ ] Test on Mac (CPU mode)
- [ ] Deploy to RunPod
- [ ] Validate GPU utilization >85%
- [ ] Monitor for stability
- [ ] Document any issues

---

**Last Updated**: 2025-11-18
**Author**: Dev 1 (Claude Code)
**Status**: Phase 1.1 Complete, Phase 1.2-1.3 In Progress
