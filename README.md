# ARC - Autonomous Research Collective

**Version:** 1.3.0 (Phase F Complete)
**Status:** Production-Ready with Multi-GPU & Advanced ML
**License:** MIT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Phase F](https://img.shields.io/badge/Phase%20F-Complete-brightgreen.svg)](https://github.com/1quantlogistics-ship-it/arc-autonomous-research)
[![Tests](https://img.shields.io/badge/Tests-77%20Passing-brightgreen.svg)](https://github.com/1quantlogistics-ship-it/arc-autonomous-research)

## Overview

ARC (Autonomous Research Collective) is a **multi-agent autonomous ML research framework** that uses LLM-based reasoning agents to design, execute, and learn from machine learning experiments. It now features **advanced experiment design capabilities** including architecture search, loss engineering, curriculum learning, and multi-objective optimization.

### What's New in Version 1.3.0 (Phase F)

**Infrastructure & Stability:**

-  **Exponential Backoff Retry Logic** - Per-agent retry policies with configurable attempts, delays, and jitter
-  **Async Cycle Timing** - CycleProfiler and AsyncBatchOptimizer for performance monitoring
-  **Multi-GPU Training (DDP)** - PyTorch DistributedDataParallel with automatic GPU discovery
-  **GPU Monitoring** - Real-time nvidia-smi metrics with REST API endpoints

**ML Capabilities:**

-  **Enhanced Loss Functions** - Lovasz-Softmax, Lovasz-Hinge, and Boundary losses
-  **Compound Loss Builder** - Combine multiple losses with configurable weights
-  **DARTS NAS** - Differentiable Architecture Search with mixed operations
-  **Pareto Visualization** - Interactive 2D/3D Pareto frontier plots with Plotly

**Phase F Stats:**
- 77 tests passing
- 17 new files added
- 2,000+ lines of new production code
- All Phase E "acknowledged gaps" resolved
- RunPod deployment ready

### Previous: Version 1.2.0 (Phase E)

 **19 advanced ML capabilities** added in Phase E:

-  **Architecture Grammar & NAS** - Neural Architecture Search with constraint validation
-  **Augmentation Policy Learning** - AutoAugment with 14 safe operations
-  **Loss Engineering** - Focal loss, multi-task learning, class weighting
-  **Curriculum Learning** - Progressive difficulty with 4 pacing strategies
-  **Multi-Objective Optimization** - Pareto frontier tracking with hypervolume metrics

### Core Features

**Production Infrastructure (Phase F):**
-  **Multi-GPU Training** - PyTorch DDP with automatic GPU discovery and load balancing
-  **GPU Monitoring API** - Real-time metrics via REST endpoints
-  **Retry-on-Timeout** - Exponential backoff with per-agent policies
-  **Async Timing** - Cycle profiling and batch optimization

**Multi-Agent Governance (Phase D):**
- 🤖 **9 Specialized Agents** with democratic voting and weighted consensus
- 🧠 **Single Local LLM** - All agents use DeepSeek R1 via vLLM (localhost:8000)
- 🛡️ **Supervisor Veto Power** - Final safety gatekeeper with override authority
- 📊 **FDA-Aligned Logging** - Automatic traceability and provenance tracking
- ⚙️ **Role-Specific Timeouts** - Configurable per-agent reasoning time
- 🐳 **RunPod Deployment** - Production Docker with GPU support

**Advanced Experiment Design (Phase E + F):**
-  **Architecture Search (NAS)** - Random, evolutionary, ENAS, **DARTS** strategies
-  **Augmentation Policy** - AutoAugment with FDA-safe operations
-  **Loss Engineering** - Focal, Dice, Tversky, **Lovasz, Boundary, Compound** losses
-  **Curriculum Learning** - Progressive training from easy to hard
-  **Multi-Objective Optimization** - Pareto frontier with **interactive visualization**

**Infrastructure:**
-  **Safety-First Design** - SEMI/AUTO/FULL autonomy modes
-  **File-Based Protocol** - JSON inter-agent communication
-  **Real GPU Training** - PyTorch integration with experiment tracking
-  **Enhanced Dashboard** - 8 tabs with real-time monitoring
-  **Snapshot & Rollback** - State preservation and restoration
-  **Offline Operation** - Full functionality without network (mock mode)

---

## Table of Contents

- [Architecture](#architecture)
- [Phase F: Production Infrastructure](#phase-f-production-infrastructure)
- [Phase E: Advanced Experiment Design](#phase-e-advanced-experiment-design)
- [Agent Roles](#agent-roles)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Clinical Safety](#clinical-safety)
- [Testing](#testing)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│     Multi-Agent Orchestrator + Consensus Engine              │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                     AGENT REGISTRY (9 Agents)                │
│                                                               │
│  Strategic:        │ Proposal:           │ Safety:           │
│  • Director (2.0)  │ • Architect (1.5)   │ • Critic (2.0)    │
│                    │ • Explorer (1.2)    │ • Critic 2 (1.8)  │
│                    │ • Param Sci (1.5)   │ • Supervisor (3.0)│
│                                                               │
│  Memory:           │ Execution:                               │
│  • Historian (1.0) │ • Executor (1.0)                        │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                  PHASE E: EXPERIMENT DESIGN                  │
│  Architecture │ Augmentation │ Loss │ Curriculum │ Multi-Obj │
│     Grammar   │    Policy    │ Eng  │  Learning  │   Optim   │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                     LLM ROUTING LAYER                        │
│           All Agents → DeepSeek R1 (localhost:8000)          │
│              Single local vLLM server required               │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                  FILE-BASED PROTOCOL MEMORY                  │
│  directive.json │ proposals.json │ reviews.json │ votes.jsonl│
│  pareto_history.json │ curriculum_history.json │ ...         │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│              FDA DEVELOPMENT LOGGING                         │
│  experiments/ │ cycles/ │ data/ │ risk/ │ git_commits/       │
└──────────────────────────────────────────────────────────────┘

Numbers in parentheses = Voting weights
```

---

## Phase F: Production Infrastructure

Phase F resolves all previously acknowledged gaps and adds production-ready infrastructure for RunPod deployment.

### 1. Retry-on-Timeout Logic

**Files:** `llm/retry.py`, `config/retry_config.py`

Exponential backoff retry utilities with per-agent policies:

```python
from llm.retry import retry_with_backoff, RetryConfig
from config.retry_config import get_agent_retry_policy

# Get agent-specific policy
policy = get_agent_retry_policy("historian")  # 5 attempts, 600s timeout

# Use decorator
@retry_with_backoff(max_attempts=3, base_delay=1.0, max_delay=30.0)
async def call_llm(prompt: str) -> str:
    return await client.complete(prompt)

# Or use context manager
async with RetryContext(policy) as ctx:
    result = await ctx.execute(call_llm, prompt)
```

**Features:**
- Configurable max attempts, base delay, max delay
- Jitter to prevent thundering herd
- Per-agent policies (Historian: 5 attempts/600s, others: 3 attempts/120s)
- FDA-compliant logging of retry events

### 2. Async Cycle Timing

**File:** `scheduler/timing.py`

Performance monitoring and batch optimization:

```python
from scheduler.timing import CycleProfiler, AsyncBatchOptimizer

# Profile research cycles
profiler = CycleProfiler()
with profiler.measure("agent_reasoning"):
    result = await agent.reason(context)

# Get timing report
report = profiler.get_report()
print(f"Agent reasoning: {report['agent_reasoning']['mean']:.2f}s")

# Optimize batch sizes
optimizer = AsyncBatchOptimizer(target_latency=2.0)
optimal_batch = optimizer.suggest_batch_size(current_throughput=10.5)
```

### 3. Multi-GPU Training (DDP)

**Files:** `tools/distributed.py`, `config/gpu_config.py`

PyTorch DistributedDataParallel wrapper:

```python
from tools.distributed import DDPWrapper, DistributedTrainer
from config.gpu_config import GPUConfig

# Automatic GPU discovery
config = GPUConfig.auto_detect()
print(f"Found {config.num_gpus} GPUs: {config.device_ids}")

# Wrap model for DDP
wrapper = DDPWrapper(model, config)
ddp_model = wrapper.wrap()

# Or use high-level trainer
trainer = DistributedTrainer(
    model=model,
    train_loader=train_loader,
    config=config
)
trainer.train(num_epochs=100)
```

**Features:**
- Automatic GPU discovery via nvidia-smi
- Configurable device IDs and world size
- Gradient synchronization strategies
- Mixed precision support (FP16/BF16)
- Checkpoint saving/loading with DDP

### 4. GPU Monitoring

**Files:** `monitoring/gpu_metrics.py`, `api/gpu_endpoints.py`

Real-time GPU metrics via REST API:

```python
from monitoring.gpu_metrics import GPUMonitor

# Get current metrics
monitor = GPUMonitor()
metrics = monitor.get_metrics()

for gpu in metrics:
    print(f"GPU {gpu['index']}: {gpu['utilization']}% util, "
          f"{gpu['memory_used']}/{gpu['memory_total']} MB, "
          f"{gpu['temperature']}°C")
```

**REST Endpoints:**
- `GET /api/gpu/metrics` - Current GPU metrics
- `GET /api/gpu/history` - Historical metrics (last hour)
- `GET /api/gpu/alerts` - Active alerts (temp >85°C, memory >90%)

### 5. Enhanced Loss Functions

**File:** `tools/loss_functions.py` (extended)

New segmentation-aware losses:

```python
from tools.loss_functions import LovaszSoftmax, LovaszHinge, BoundaryLoss, CompoundLoss

# Lovasz-Softmax for multi-class segmentation
lovasz = LovaszSoftmax(classes='present', per_image=False)
loss = lovasz(predictions, targets)

# Boundary loss for edge-aware training
boundary = BoundaryLoss(theta0=3, theta=5)
loss = boundary(predictions, distance_maps)

# Compound loss combining multiple objectives
compound = CompoundLoss([
    (FocalLoss(gamma=2.0), 0.5),
    (LovaszSoftmax(), 0.3),
    (BoundaryLoss(), 0.2)
])
loss = compound(predictions, targets)
```

### 6. DARTS Neural Architecture Search

**Files:** `tools/darts.py`, `schemas/architecture_grammar.py` (extended)

Differentiable Architecture Search:

```python
from tools.darts import DARTSSearcher, MixedOp
from schemas.architecture_grammar import DARTSSearchConfig

# Configure search space
config = DARTSSearchConfig(
    num_cells=8,
    num_nodes=4,
    primitives=['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3',
                'max_pool_3x3', 'avg_pool_3x3', 'skip_connect'],
    num_epochs=50
)

# Run architecture search
searcher = DARTSSearcher(config)
genotype = searcher.search(train_loader, val_loader)

# Derive final architecture
final_model = searcher.derive_architecture(genotype)
print(f"Best architecture: {genotype}")
```

### 7. Pareto Visualization

**Files:** `tools/pareto_viz.py`, `api/visualization_endpoints.py`

Interactive Pareto frontier plots:

```python
from tools.pareto_viz import ParetoFront

# Create Pareto front from experiments
front = ParetoFront.from_experiments(experiments, objectives=['auc', 'sensitivity', 'latency'])

# Generate interactive 2D plot
fig = front.plot_2d('auc', 'sensitivity',
                    highlight_optimal=True,
                    show_dominated=True)
fig.write_html('pareto_2d.html')

# Generate 3D plot
fig = front.plot_3d('auc', 'sensitivity', 'latency',
                    colorscale='Viridis')
fig.write_html('pareto_3d.html')

# Get optimal solutions
optimal = front.get_pareto_optimal()
print(f"Found {len(optimal)} Pareto-optimal experiments")
```

**REST Endpoints:**
- `GET /api/viz/pareto?objectives=auc,sensitivity` - 2D Pareto plot
- `GET /api/viz/pareto3d?objectives=auc,sensitivity,latency` - 3D Pareto plot
- `GET /api/viz/pareto/data` - Raw Pareto front data

---

## Phase E: Advanced Experiment Design

Phase E adds **19 sophisticated ML capabilities** enabling ARC to autonomously explore advanced training techniques while maintaining clinical safety.

### Part 1: Architecture Grammar + Augmentation Policy

#### 1. Architecture Grammar (NAS)

**Schema:** `schemas/architecture_grammar.py` (593 lines)

Define neural architecture search spaces with constraint validation:

**Features:**
- **Layer Types**: Conv2D, Residual, Attention, Transformer, Pooling, BatchNorm, Dropout
- **Search Spaces**: Input/output channels, kernel sizes, strides, activation functions
- **NAS Strategies**: Random search, evolutionary, ENAS, DARTS, reinforcement learning
- **Constraint Validation**: Parameter count ≤ 10M, GPU memory ≤ 5GB, DRI ≥ 0.6

**Example:**
```python
from schemas.architecture_grammar import ArchitectureGrammar, LayerType, NASStrategy

# Define search space
grammar = ArchitectureGrammar(
    name="resnet_search_space",
    layer_groups=[
        {
            "type": LayerType.RESIDUAL_BLOCK,
            "num_layers": 3,
            "channels": [64, 128, 256]
        }
    ],
    nas_strategy=NASStrategy.EVOLUTIONARY,
    max_params=10_000_000,  # 10M parameter limit
    max_gpu_memory_gb=5.0
)
```

**Clinical Safety:**
- Maximum 10M parameters (deployment feasibility)
- GPU memory capped at 5GB
- DRI ≥ 0.6 (Disc Relevance Index - image quality preservation)

#### 2. Augmentation Policy

**Schema:** `schemas/augmentation_policy.py` (632 lines)

AutoAugment-style augmentation policy learning with medical imaging safety:

**14 Safe Operations:**
- **Geometric**: Rotate (±15°), horizontal flip, scale (0.9-1.1), translate (±10%)
- **Intensity**: Brightness (±10%), contrast (±10%), gamma (0.9-1.1)
- **Advanced**: Gaussian noise (σ≤0.01), Gaussian blur (kernel≤3), elastic deformation

**Forbidden Operations** (Medical Imaging Safety):
- Color jitter (hue/saturation changes)
- Cutout / random erasing
- Strong elastic deformation
- Aggressive downsampling

**Example:**
```python
from schemas.augmentation_policy import (
    AugmentationPolicy, AugmentationOp, AugmentationOpType
)

# Define safe augmentation policy
policy = AugmentationPolicy(
    name="safe_medical_augmentation",
    operations=[
        AugmentationOp(
            op_type=AugmentationOpType.ROTATE,
            magnitude=10.0,  # ±10 degrees
            probability=0.5
        ),
        AugmentationOp(
            op_type=AugmentationOpType.BRIGHTNESS,
            magnitude=0.08,  # ±8%
            probability=0.3
        )
    ],
    dri_constraint=0.6  # Maintain image quality
)
```

**Evolution Strategies:**
- Random search
- Grid search
- Mutation (30% rate)
- Crossover (2-parent)
- Population-based training (PBT)

### Part 2: Loss Engineering + Curriculum Learning

#### 3. Loss Configuration

**Schemas:**
- `schemas/loss_config.py` (472 lines)
- `tools/loss_functions.py` (544 lines - PyTorch implementations)

Advanced loss functions for class imbalance and multi-task learning:

**Base Loss Types:**
- **BCE**: Binary Cross-Entropy (baseline)
- **Focal Loss**: γ parameter for hard example focus (γ=2.0 typical)
- **Weighted BCE**: Class weights (inverse frequency)
- **Dice Loss**: Segmentation-inspired IoU-based
- **Tversky Loss**: Configurable FP/FN trade-off (α, β parameters)
- **Combined**: Hybrid strategies (e.g., BCE+Dice)

**Multi-Task Learning:**
- Primary task: Glaucoma classification (weight ≥ 0.6)
- Auxiliary tasks: DRI prediction, CDR prediction, ISNT ratio, vessel density
- Safety: Primary weight must be ≥ 0.6 for clinical focus

**Example:**
```python
from schemas.loss_config import (
    LossConfig, LossType, AuxiliaryTask, ClassWeightingStrategy
)

# Focal loss with auxiliary task
loss_config = LossConfig(
    name="focal_with_dri",
    primary_loss=LossType.FOCAL,
    primary_weight=0.7,
    auxiliary_tasks=[
        {
            "task_type": AuxiliaryTask.DRI_PREDICTION,
            "weight": 0.3,
            "loss_type": "mse"
        }
    ],
    class_weighting=ClassWeightingStrategy.BALANCED,
    hyperparameters={
        "focal_gamma": 2.0,
        "focal_alpha": 0.75
    }
)
```

**Class Weighting Strategies:**
- None (baseline)
- Balanced (inverse class frequency)
- Effective Samples (Class-Balanced Loss, Cui et al. 2019)
- Custom weights

#### 4. Curriculum Learning

**Schema:** `schemas/curriculum_strategy.py` (496 lines)

Progressive training from easy to hard samples:

**Difficulty Metrics:**
- Image quality (contrast, sharpness, SNR)
- Disease severity (mild → moderate → severe)
- CDR ratio (easy → challenging cases)
- Diagnostic confidence (high certainty → ambiguous)

**Pacing Strategies:**
- **Linear**: Constant difficulty increase
- **Exponential**: Accelerating difficulty curve
- **Adaptive**: Performance-based progression
- **Step-based**: Threshold transitions

**Example:**
```python
from schemas.curriculum_strategy import (
    CurriculumStrategy, DifficultyMetric, PacingStrategy
)

# Define 3-stage curriculum
curriculum = CurriculumStrategy(
    name="quality_based_curriculum",
    difficulty_metric=DifficultyMetric.IMAGE_QUALITY,
    pacing_strategy=PacingStrategy.LINEAR,
    stages=[
        {
            "stage_id": 0,
            "name": "easy",
            "difficulty_range": (0.0, 0.3),
            "num_epochs": 20,
            "dri_threshold": 0.7
        },
        {
            "stage_id": 1,
            "name": "medium",
            "difficulty_range": (0.3, 0.7),
            "num_epochs": 30,
            "dri_threshold": 0.65
        },
        {
            "stage_id": 2,
            "name": "hard",
            "difficulty_range": (0.7, 1.0),
            "num_epochs": 30,
            "dri_threshold": 0.6
        }
    ],
    min_sensitivity=0.85  # Safety constraint
)
```

**Clinical Safety:**
- Sensitivity ≥ 0.85 throughout all stages
- DRI constraints enforced per stage
- Automatic rollback if metrics degrade

### Part 3: Multi-Objective Optimization

#### 5. Multi-Objective Optimization

**Schema:** `schemas/multi_objective.py` (652 lines)

Pareto frontier tracking for trade-off exploration:

**Core Components:**
- **ObjectiveSpec**: Define metric, weight, direction (maximize/minimize), constraints
- **ParetoFront**: Track non-dominated solutions
- **Dominance Checking**: Identify dominated vs non-dominated experiments
- **Hypervolume**: Quality metric for Pareto front

**Example:**
```python
from schemas.multi_objective import (
    ObjectiveSpec, MultiObjectiveConfig, OptimizationDirection
)

# Balanced AUC, Sensitivity, Specificity optimization
objectives = [
    ObjectiveSpec(
        metric_name="auc",
        weight=0.4,
        direction=OptimizationDirection.MAXIMIZE
    ),
    ObjectiveSpec(
        metric_name="sensitivity",
        weight=0.3,
        direction=OptimizationDirection.MAXIMIZE,
        constraint={"type": ">=", "value": 0.85}
    ),
    ObjectiveSpec(
        metric_name="specificity",
        weight=0.3,
        direction=OptimizationDirection.MAXIMIZE
    )
]

# Historian tracks Pareto frontier evolution
pareto_front = historian.get_pareto_frontier(objectives)
print(f"Pareto-optimal solutions: {pareto_front['num_pareto_optimal']}")
print(f"Hypervolume: {pareto_front['hypervolume']:.3f}")
```

**Pareto Frontier Analysis:**
- Dominance relationships
- Pareto ranking (rank 0 = optimal)
- Hypervolume computation (2D, 3D, N-D)
- Evolution tracking across cycles
- Trade-off correlation analysis

**World Model Integration:**
```python
# Multi-objective prediction
predictions = world_model.predict_multi_objective(
    config,
    objectives=["auc", "sensitivity", "specificity"]
)

# Suggest Pareto-optimal experiments
suggestions = world_model.suggest_pareto_optimal_experiments(
    candidate_configs,
    objectives=objectives,
    acquisition="hypervolume"  # or "ucb", "ei"
)
```

---

## Agent Roles

### 9 Specialized Agents with Democratic Voting

| Agent               | Model            | Weight | Responsibility                       | Phase E Enhancements |
|---------------------|------------------|--------|--------------------------------------|---------------------|
| **Director**        | Claude Sonnet    | 2.0    | Strategic planning, mode control     | - |
| **Architect**       | DeepSeek R1      | 1.5    | Experiment design                    | - |
| **Explorer** ⭐      | Qwen 2.5         | 1.2    | Parameter space exploration          | **Augmentation policy proposals** |
| **Param Scientist** ⭐| DeepSeek R1     | 1.5    | Hyperparameter optimization          | **Architecture NAS, Loss configs** |
| **Instructor** 🆕   | DeepSeek R1      | 1.3    | **Curriculum design**                | **Augmentation + Curriculum** |
| **Critic**          | Qwen 2.5         | 2.0    | Primary safety review                | **Architecture + Augmentation validation** |
| **Critic Secondary** ⭐| DeepSeek R1    | 1.8    | Secondary safety, prevent groupthink | - |
| **Supervisor** ⭐    | Llama 3 (Local)  | **3.0**| **Final validation, veto power**     | - |
| **Historian**       | DeepSeek R1      | 1.0    | Memory management + world model      | **Curriculum + Pareto tracking** |
| **Executor**        | DeepSeek R1      | 1.0    | Training execution                   | - |

⭐ = New in Phase D | 🆕 = Enhanced in Phase E

### Phase E Agent Capabilities

**Parameter Scientist:**
- `propose_architectures()` - NAS proposals with constraint checking
- `propose_loss_configs()` - Focal loss, multi-task learning strategies

**Explorer:**
- `propose_augmentation_policies()` - Evolutionary augmentation search
- Mutation, crossover, random policy generation

**Instructor** (New method):
- `propose_curriculum_strategy()` - Difficulty metric + pacing strategy design
- `propose_augmentation_strategy()` - AutoAugment policy proposals

**Historian:**
- `track_curriculum_progression()` - Log stage transitions to `curriculum_history.json`
- `get_pareto_frontier()` - Extract non-dominated experiments
- `track_pareto_evolution()` - Save Pareto snapshots, track hypervolume
- `analyze_objective_tradeoffs()` - Correlation analysis + recommendations

**Critic:**
- Architecture grammar safety validation
- Augmentation policy safety checks (DRI ≥ 0.6)

**World Model:**
- `predict_multi_objective()` - Predict AUC, sensitivity, specificity simultaneously
- `suggest_pareto_optimal_experiments()` - Multi-objective acquisition functions

---
### 🆕 Phase D Agent Roles

| Agent               | Model            | Weight | Responsibility                       | Timeout |
|---------------------|------------------|--------|--------------------------------------|---------|
| Director            | Claude Sonnet    | 2.0    | Strategic planning, mode control     | 120s    |
| Architect           | DeepSeek R1      | 1.5    | Experiment design                    | 120s    |
| **Explorer** ⭐      | Qwen 2.5         | 1.2    | Parameter space exploration          | 120s    |
| **Param Scientist** ⭐| DeepSeek R1     | 1.5    | Hyperparameter optimization          | 120s    |
| Critic              | Qwen 2.5         | 2.0    | Primary safety review                | 120s    |
| **Critic Secondary** ⭐| DeepSeek R1    | 1.8    | Secondary safety, prevent groupthink | 120s    |
| **Supervisor** ⭐    | Llama 3 (Local)  | **3.0**| **Final validation, veto power**     | 120s    |
| **Historian** 🔧    | DeepSeek R1      | 1.0    | Memory management                    | **600s**|
| Executor            | DeepSeek R1      | 1.0    | Training execution                   | 120s    |

⭐ = New in Phase D | 🔧 = Enhanced timeout support

## 🆕 FDA-Aligned Development Logging

ARC now includes automatic development logging that demonstrates professional, methodical development for regulatory contexts (FDA, ISO 13485, GMLP Principle 9).

### What Gets Logged Automatically

**Experiment Logging** (`dev_logs/experiments/`)
- Complete config (model, dataset, hyperparameters)
- All metrics (AUC, sensitivity, specificity, accuracy)
- Model and dataset versions
- Reasoning summaries
- Execution status and duration
- Checkpoint paths

**Research Cycle Logging** (`dev_logs/cycles/`)
- Agents involved in each cycle
- Proposals generated and approved
- Decision reasoning
- Failures and warnings
- Supervisor vetoes and conflicts
- Cycle duration

**Risk Event Logging** (`dev_logs/risk/`)
- Cycle crashes (high severity)
- LLM timeouts (medium severity)
- Supervisor vetoes (low severity)
- Experiment failures (medium severity)
- Training errors with context

**Data Provenance Logging** (`dev_logs/data/`)
- Dataset preprocessing operations
- Input/output checksums (MD5)
- Transformations applied
- File counts and validation
- Processing metadata

**Git Commit Tracking** (`dev_logs/git_commits/`)
- Automatic commit logging
- Code change tracking

**System Snapshots** (`dev_logs/system_snapshots/`)
- Per-cycle system state
- Configuration snapshots
- Reproducibility support

### Log Formats

All logs written in dual format:
- **JSONL** (`.jsonl`): Machine-readable, line-delimited JSON
- **TXT** (`.txt`): Human-readable summaries

### FDA Compliance Features

 **Traceability**: Every decision tracked from proposal to result
 **Structured Iteration**: Cycle-by-cycle progression documented
 **Controlled Changes**: Git commits + system snapshots
 **Reproducibility**: Full config + checksums captured
 **Process Awareness**: Agent reasoning and decisions logged
 **Risk Awareness**: Timeouts, crashes, vetoes tracked

**Note**: This is *lightweight documentation* showing professional development, NOT full QMS/DHF/ISO compliance. Demonstrates methodical approach and traceability for regulatory review.

## Components

### Control Plane (`api/control_plane.py`)
FastAPI service for orchestration, safety validation, and state management.
- **Port:** 8002
- **Endpoints:** `/status`, `/exec`, `/train`, `/archive`, `/rollback`, `/mode`

### Dashboard (`api/dashboard.py`)
Streamlit web interface for monitoring and control.
- **Port:** 8501
- **Features:** Memory visualization, experiment tracking, live metrics

### Orchestrators
- **multi_agent_orchestrator.py**: Full 9-agent democratic research cycle
- **training_executor.py**: GPU training with experiment tracking
- **complete_research_loop.py**: End-to-end autonomous research

### Training Integration (`tools/acuvue_tools.py`)
AcuVue medical imaging tools with:
- Dataset preprocessing with provenance tracking
- PyTorch training with GPU support
- Evaluation and metrics calculation
- Checkpoint management
- CAM visualization generation

## Installation

### Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.1+ (for GPU training)
- **Docker**: 20.10+ (for containerized deployment)
- **Git**: 2.30+

### LLM Server Requirement

> **IMPORTANT:** ARC requires a local LLM server running before startup.

All 10 agents use a single local LLM endpoint. You must have **vLLM** (or compatible OpenAI-format server) running:

```bash
# Option 1: vLLM with DeepSeek R1 (recommended)
pip install vllm
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2

# Option 2: Ollama (simpler setup)
ollama serve
ollama run deepseek-r1:32b

# Option 3: Any OpenAI-compatible server
# Must respond at http://localhost:8000/generate
```

**Server must be running on `http://localhost:8000`** before starting ARC.

| Setting | Value |
|---------|-------|
| Endpoint | `http://localhost:8000/generate` |
| Model | `deepseek-r1` (or any 32B+ model) |
| Min VRAM | 48GB (for 32B model) |
| Recommended | 2x A100 80GB or 4x RTX 4090 |

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/1quantlogistics-ship-it/arc-autonomous-research.git
cd arc-autonomous-research/arc_clean

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.production .env
# Edit .env with your API keys and settings

# 5. Initialize memory directory
mkdir -p /workspace/arc/memory
mkdir -p /workspace/arc/experiments
```

### RunPod Deployment

```bash
# 1. Build Docker image
docker build -t arc:latest .

# 2. Run container
docker run -d \
  --name arc-pod \
  --gpus all \
  -p 8002:8002 \
  -p 8501:8501 \
  -v /workspace:/workspace \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  arc:latest

# 3. Access dashboard
# http://your-runpod-ip:8501
```

---

## Quick Start

### 4-Step Startup Sequence

```bash
# Terminal 1: Control Plane API (FastAPI)
cd arc_clean
python -m api.control_plane
# Running on http://localhost:8002

# Terminal 2: Dashboard (Streamlit)
cd arc_clean
streamlit run api/dashboard.py --server.port=8501
# Dashboard: http://localhost:8501

# Terminal 3: Background State Poller (Optional)
cd arc_clean
python -m api.ui_state_poller
# Polls memory every 5s, caches for UI

# Terminal 4: Run Research Cycle
cd arc_clean
python orchestrators/multi_agent_orchestrator.py --mode=SEMI --cycles=5
```

### Access Points

- **Dashboard**: http://localhost:8501 (Streamlit UI)
- **Control Plane**: http://localhost:8002 (FastAPI)
- **API Docs**: http://localhost:8002/docs (Swagger UI)

---

## Usage Examples

### Example 1: Focal Loss for Class Imbalance

```python
from config.experiment_config_generator import ExperimentConfigGenerator
from schemas.loss_config import LossConfig, LossType, ClassWeightingStrategy

# Initialize config generator
generator = ExperimentConfigGenerator()

# Create focal loss configuration
proposal = {
    "changes": {
        "loss_config": LossConfig(
            name="focal_gamma2_balanced",
            primary_loss=LossType.FOCAL,
            primary_weight=1.0,
            class_weighting=ClassWeightingStrategy.BALANCED,
            hyperparameters={
                "focal_gamma": 2.0,
                "focal_alpha": 0.75
            }
        ).to_dict()
    }
}

# Generate training config
config = generator.generate_config("exp_focal_001", proposal)

# Config now contains:
# {
#   "loss_type": "focal",
#   "loss_params": {"gamma": 2.0, "alpha": 0.75},
#   "class_weighting": "balanced",
#   ...
# }
```

### Example 2: Curriculum Learning

```python
from schemas.curriculum_strategy import (
    CurriculumStrategy, DifficultyMetric, PacingStrategy
)

# Design 3-stage curriculum
curriculum = CurriculumStrategy(
    name="severity_curriculum",
    difficulty_metric=DifficultyMetric.DISEASE_SEVERITY,
    pacing_strategy=PacingStrategy.ADAPTIVE,
    stages=[
        {"stage_id": 0, "name": "mild", "difficulty_range": (0.0, 0.3), "num_epochs": 20},
        {"stage_id": 1, "name": "moderate", "difficulty_range": (0.3, 0.7), "num_epochs": 25},
        {"stage_id": 2, "name": "severe", "difficulty_range": (0.7, 1.0), "num_epochs": 25}
    ],
    min_sensitivity=0.85
)

# Track progression
historian.track_curriculum_progression(
    experiment_id="exp_curriculum_001",
    curriculum_name="severity_curriculum"
)

# Analyze effectiveness
analysis = historian.analyze_curriculum_effectiveness(
    curriculum_name="severity_curriculum",
    baseline_name="baseline_no_curriculum"
)
print(f"Improvement: {analysis['improvement_pct']:.1f}%")
```

### Example 3: Multi-Objective Optimization

```python
from schemas.multi_objective import ObjectiveSpec, OptimizationDirection

# Define objectives
objectives = [
    ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
    ObjectiveSpec(
        metric_name="sensitivity",
        weight=0.5,
        direction=OptimizationDirection.MAXIMIZE,
        constraint={"type": ">=", "value": 0.85}
    )
]

# Get current Pareto frontier
result = historian.get_pareto_frontier(objectives)

print(f"Pareto-optimal experiments: {result['num_pareto_optimal']}")
print(f"Hypervolume: {result['hypervolume']:.3f}")

# Track evolution over time
evolution = historian.track_pareto_evolution(objectives, cycle_id=10)
print(f"Hypervolume improvement: {evolution['hypervolume_improvement']:.3f}")

# Analyze trade-offs
tradeoffs = historian.analyze_objective_tradeoffs(objectives)
for trade in tradeoffs['tradeoffs']:
    print(f"Trade-off: {trade['metric_1']} vs {trade['metric_2']} (r={trade['correlation']:.2f})")
```

### Example 4: Running a Full Research Cycle

```python
from api.multi_agent_orchestrator import MultiAgentOrchestrator

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator(
    mode="SEMI",  # Requires human approval
    offline=False  # Use real models
)

# Run research cycle
for cycle in range(5):
    print(f"\n=== Cycle {cycle + 1} ===")

    # 1. Historian summarizes history
    summary = orchestrator.run_historian()

    # 2. Director sets strategy
    directive = orchestrator.run_director(summary)

    # 3. Agents propose experiments
    proposals = orchestrator.run_proposal_agents(directive, summary)

    # 4. Democratic voting
    votes = orchestrator.run_voting(proposals)

    # 5. Supervisor validation
    approved = orchestrator.run_supervisor(votes)

    # 6. Execute approved experiments
    if approved:
        results = orchestrator.run_executor(approved)
        print(f"Completed: {results['experiment_id']}")
```

---

## Configuration

### Environment Variables

```bash
# .env file

# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Selection
DIRECTOR_MODEL=claude-sonnet-4.5
ARCHITECT_MODEL=deepseek-r1
CRITIC_MODEL=qwen2.5-32b

# Autonomy Mode
ARC_MODE=SEMI  # SEMI, AUTO, or FULL
See `.env.production` for complete configuration template.

## Memory Protocol

ARC uses file-based JSON protocol for all agent communication:

**Core Protocol Files:**
- `memory/directive.json` - Strategic directives from Director
- `memory/proposals.json` - Experiment ideas from Architect
- `memory/reviews.json` - Safety evaluations from Critic
- `memory/history_summary.json` - Learning history from Historian
- `memory/constraints.json` - Forbidden parameter ranges
- `memory/system_state.json` - Global ARC state

**Phase D Decision Logs:**
- `memory/decisions/voting_history.jsonl` - Multi-agent vote records
- `memory/decisions/supervisor_decisions.jsonl` - Supervisor decisions
- `memory/decisions/overrides.jsonl` - Consensus override log

**🆕 FDA Development Logs:**
- `dev_logs/experiments/experiment_history.jsonl` - All experiments
- `dev_logs/cycles/cycle_history.jsonl` - All research cycles
- `dev_logs/risk/risk_events.jsonl` - Risk tracking
- `dev_logs/data/data_provenance.jsonl` - Dataset operations
- `dev_logs/git_commits/commit_history.jsonl` - Code changes
- `dev_logs/system_snapshots/` - System state snapshots

## Validation Status

### Phase C (v0.9.0)
✅ **Smoketest #1 (Structural)** - PASSED
✅ **Smoketest #2 (Training Pipeline)** - PASSED
- Single-LLM architecture validated
- All 5 agents operational
- Real GPU training successful
- Full research loop complete

### Phase D (v1.1.0-alpha)
 **Multi-Agent Infrastructure** - COMPLETE
- 9 specialized agent classes implemented
- Agent registry and discovery system
- Democratic voting mechanism
- Supervisor veto power
- Offline operation (mock mode)
- Enhanced dashboard (8 tabs)
- Configuration system (YAML)

 **Production Enhancements** - COMPLETE
- FDA-aligned development logging
- Role-specific timeout support (Historian 600s)
- Data provenance tracking with checksums
- Risk event monitoring
- RunPod deployment configuration
- Docker containerization

### Phase F (v1.3.0)
 **Infrastructure & Stability** - COMPLETE
- Exponential backoff retry logic with per-agent policies
- Async cycle timing (CycleProfiler, AsyncBatchOptimizer)
- Multi-GPU training (PyTorch DDP)
- GPU monitoring with REST API

 **ML Capabilities** - COMPLETE
- Lovasz-Softmax, Lovasz-Hinge, Boundary losses
- Compound loss builder
- DARTS neural architecture search
- Interactive Pareto visualization (2D/3D)

# Timeouts (seconds)
DIRECTOR_TIMEOUT=120
HISTORIAN_TIMEOUT=600

# Paths
WORKSPACE_PATH=/workspace/arc
MEMORY_PATH=/workspace/arc/memory

# RunPod Settings
RUNPOD_GPU_COUNT=2
RUNPOD_GPU_TYPE=A40
```

### Agent Configuration

Edit `config/agents.yaml`:

```yaml
agents:
  - id: parameter_scientist_001
    role: parameter_scientist
    model: deepseek-r1
    voting_weight: 1.5
    capabilities:
      - proposal_generation
      - exploration
    phase_e_enabled: true  # Enable Phase E features
```

### Multi-Objective Configuration

```python
# In your experiment proposal
proposal = {
    "objectives": [
        {"metric_name": "auc", "weight": 0.5, "direction": "maximize"},
        {
            "metric_name": "sensitivity",
            "weight": 0.3,
            "direction": "maximize",
            "constraint": {"type": ">=", "value": 0.85}
        },
        {"metric_name": "specificity", "weight": 0.2, "direction": "maximize"}
    ]
}
```

---

## Clinical Safety

### Phase E Safety Guarantees

All advanced ML techniques enforce FDA-compliance and clinical safety:

#### Architecture Search
- ✅ Parameter count ≤ 10M (deployment feasibility)
- ✅ GPU memory ≤ 5GB (hardware constraints)
- ✅ DRI ≥ 0.6 (image quality preservation)

#### Augmentation Policy
- ✅ Rotation limited to ±15° (maintains clinical orientation)
- ✅ No blur beyond σ=3.0 (preserves diagnostic features)
- ✅ No noise beyond σ=0.01 (maintains SNR)
- ✅ No color jitter (preserves tissue appearance)
- ✅ DRI ≥ 0.6 enforced on all policies

#### Loss Engineering
- ✅ Primary classification weight ≥ 0.6 (glaucoma detection is primary)
- ✅ Auxiliary task weights ≤ 0.4 (supplementary only)
- ✅ Focal gamma ≤ 3.0 (training stability)
- ✅ Label smoothing ≤ 0.15 (calibration preservation)
- ✅ Tversky β ≥ α (prioritize recall over precision)

#### Curriculum Learning
- ✅ Sensitivity ≥ 0.85 throughout all stages (minimize false negatives)
- ✅ DRI ≥ 0.6 at hardest difficulty (maintain image quality)
- ✅ Automatic rollback if metrics degrade
- ✅ Stage progression only when safe

#### Multi-Objective Optimization
- ✅ AUC must be included as objective (primary performance metric)
- ✅ Sensitivity constraint ≥ 0.85 (false negative prevention)
- ✅ Pareto fronts validated for clinical compliance
- ✅ Dominated solutions filtered from recommendations

### FDA Development Logging

All Phase E features integrate with FDA-aligned logging:

```python
# Automatic logging of:
# - Architecture search attempts
# - Augmentation policy evolution
# - Loss configuration changes
# - Curriculum progression
# - Pareto frontier evolution
# - All safety constraint violations
```

**Logged to:** `dev_logs/experiments/`, `dev_logs/cycles/`, `dev_logs/data/`

---

## Testing

### Phase E Test Suite

**File:** `tests/test_multi_objective_e2e.py` (714 lines, 30+ tests)

```bash
# Run comprehensive Phase E tests
cd arc_clean
pytest tests/test_multi_objective_e2e.py -v

# Test categories:
# - ObjectiveSpec validation (5 tests)
# - Pareto frontier computation (8 tests)
# - Clinical safety validation (6 tests)
# - Historian Pareto tracking (4 tests)
# - World model multi-objective (3 tests)
# - Config generator integration (4 tests)
```

**Test Coverage:**
- ✅ Architecture grammar validation
- ✅ Augmentation policy safety checks
- ✅ Loss config validation (Pydantic + clinical)
- ✅ Curriculum strategy validation
- ✅ Pareto dominance computation
- ✅ Hypervolume calculation (2D, 3D, N-D)
- ✅ Multi-objective predictions
- ✅ Config translation methods

### Running All Tests

```bash
# Unit tests
pytest tests/agents/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# All tests with coverage
pytest --cov=arc_clean tests/
```

---

## Development

### Project Structure

```
arc_clean/
├── agents/                  # 9 specialized agent implementations
│   ├── director_agent.py
│   ├── architect_agent.py
│   ├── critic_agent.py
│   ├── critic_secondary.py
│   ├── explorer.py
│   ├── parameter_scientist.py
│   ├── instructor_agent.py
│   ├── historian_agent.py
│   ├── supervisor.py
│   ├── executor_agent.py
│   └── base.py, registry.py, protocol.py
│
├── llm/                     # Phase F: LLM utilities
│   └── retry.py                     (Exponential backoff retry)
│
├── scheduler/               # Phase F: Timing utilities
│   └── timing.py                    (CycleProfiler, AsyncBatchOptimizer)
│
├── monitoring/              # Phase F: GPU monitoring
│   └── gpu_metrics.py               (nvidia-smi wrapper)
│
├── schemas/                 # Phase E + F schemas
│   ├── architecture_grammar.py      (NAS + DARTS configs)
│   ├── augmentation_policy.py       (AutoAugment)
│   ├── loss_config.py               (Extended with Lovasz, Boundary)
│   ├── curriculum_strategy.py       (Curriculum learning)
│   ├── multi_objective.py           (Pareto optimization)
│   └── experiment_schemas.py
│
├── tools/                   # Phase E + F tools
│   ├── loss_functions.py            (Extended: Lovasz, Boundary, Compound)
│   ├── distributed.py               (Phase F: DDP wrapper)
│   ├── darts.py                     (Phase F: DARTS NAS)
│   ├── pareto_viz.py                (Phase F: Pareto visualization)
│   ├── world_model.py               (multi-objective predictions)
│   ├── acuvue_tools.py
│   ├── dataset_fusion.py
│   ├── drift_detector.py
│   ├── failure_predictor.py
│   ├── mode_collapse_engine.py
│   └── dev_logger.py
│
├── config/                  # Configuration management
│   ├── experiment_config_generator.py
│   ├── retry_config.py              (Phase F: Per-agent retry policies)
│   ├── gpu_config.py                (Phase F: GPU configuration)
│   ├── loader.py
│   ├── agents.example.yaml
│   └── models.example.yaml
│
├── api/                     # Core services
│   ├── control_plane.py
│   ├── multi_agent_orchestrator.py
│   ├── training_executor.py
│   ├── dashboard.py
│   ├── gpu_endpoints.py             (Phase F: GPU monitoring API)
│   ├── visualization_endpoints.py   (Phase F: Pareto viz API)
│   ├── ui_endpoints.py
│   └── ui_state_poller.py
│
├── tests/                   # Comprehensive test suite (77 tests)
│   ├── test_multi_objective_e2e.py
│   ├── test_phase_f_*.py            (Phase F tests)
│   ├── unit/
│   ├── integration/
│   └── mocks/
│
├── memory/                  # Protocol memory (JSON files)
│   ├── directive.json
│   ├── proposals.json
│   ├── pareto_history.json          (new - Phase E)
│   ├── curriculum_history.json      (new - Phase E)
│   └── decisions/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md (this file)
```

### Phase F File Summary

**New Files (17):**

*Infrastructure:*
1. `llm/retry.py` - Exponential backoff retry utilities
2. `config/retry_config.py` - Per-agent retry policies
3. `scheduler/timing.py` - CycleProfiler, AsyncBatchOptimizer
4. `tools/distributed.py` - DDP wrapper, DistributedTrainer
5. `config/gpu_config.py` - GPU configuration
6. `monitoring/gpu_metrics.py` - nvidia-smi wrapper
7. `api/gpu_endpoints.py` - GPU monitoring REST endpoints

*ML Capabilities:*
8. `tools/loss_functions.py` - Extended with Lovasz, Boundary, Compound
9. `schemas/loss_config.py` - Extended loss schemas
10. `tools/darts.py` - DARTS neural architecture search
11. `schemas/architecture_grammar.py` - Extended with DARTS configs
12. `tools/pareto_viz.py` - Pareto visualization
13. `api/visualization_endpoints.py` - Visualization REST endpoints

*Tests:*
14-17. `tests/test_phase_f_*.py` - Comprehensive Phase F tests

**Total:** 2,000+ lines of new production code, 77 tests passing

### Phase E File Summary

**New Files (7):**
1. `schemas/architecture_grammar.py` - 593 lines
2. `schemas/augmentation_policy.py` - 632 lines
3. `schemas/loss_config.py` - 472 lines
4. `schemas/curriculum_strategy.py` - 496 lines
5. `schemas/multi_objective.py` - 652 lines
6. `tools/loss_functions.py` - 544 lines
7. `tests/test_multi_objective_e2e.py` - 714 lines

**Total:** 4,103 lines of new code

---

## Documentation

### Core Documentation

- **[PHASE_D_PLAN.md](PHASE_D_PLAN.md)** - Multi-agent architecture details
- **[PHASE_2_3_SUMMARY.md](PHASE_2_3_SUMMARY.md)** - Infrastructure leverage & intelligence
- **[PHASE_4A_EXECUTION_INTEGRATION.md](PHASE_4A_EXECUTION_INTEGRATION.md)** - Training execution
- **[PHASE_4B_INTELLIGENCE_LAYER.md](PHASE_4B_INTELLIGENCE_LAYER.md)** - World model & adaptive strategy
- **[DEV_2_UI_ARCHITECTURE.md](DEV_2_UI_ARCHITECTURE.md)** - Dashboard & UI backend
- **[CONTROL_PLANE_INTEGRATION_COMPLETE.md](CONTROL_PLANE_INTEGRATION_COMPLETE.md)** - API integration

### Phase E Documentation

**Schemas** (inline docstrings):
- Architecture grammar: 130+ lines of documentation
- Augmentation policy: 150+ lines of documentation
- Loss config: 140+ lines of documentation
- Curriculum strategy: 120+ lines of documentation
- Multi-objective: 180+ lines of documentation

**Examples:**
- All schemas include factory methods with examples
- All schemas include validation examples
- All schemas include clinical safety examples

### External Resources

- **FDA Guidance**: [Software as a Medical Device (SaMD)](https://www.fda.gov/medical-devices/software-medical-device-samd)
- **GMLP Principle 9**: Development tracking and traceability
- **ISO 13485**: Quality management for medical devices
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Focal Loss Paper**: Lin et al., 2017 (https://arxiv.org/abs/1708.02002)
- **Class-Balanced Loss**: Cui et al., 2019 (https://arxiv.org/abs/1901.05555)

---

## Contributing

We welcome contributions! Areas of interest:

**Phase G Ideas:**
- Advanced curriculum strategies (competence-based pacing)
- High-dimensional Pareto visualization (>3 objectives)
- Reinforcement learning-based NAS
- Federated learning support
- Model compression and quantization

**General Improvements:**
- Additional test coverage
- Documentation improvements
- Bug fixes
- Performance optimizations
- New agent roles

**Process:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use ARC in your research, please cite:

```bibtex
@software{arc_autonomous_research_2025,
  title = {ARC: Autonomous Research Collective},
  author = {ARC Development Team},
  year = {2025},
  version = {1.3.0},
  url = {https://github.com/1quantlogistics-ship-it/arc-autonomous-research},
  note = {Phase F: Production Infrastructure - Multi-GPU Training, GPU Monitoring,
          Retry Logic, DARTS NAS, Enhanced Losses, Pareto Visualization}
}
```

**Phase F Features:**
- Multi-GPU Training (PyTorch DDP)
- GPU Monitoring with REST API
- Exponential Backoff Retry Logic
- DARTS Neural Architecture Search
- Lovasz, Boundary, and Compound Losses
- Interactive Pareto Visualization

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

Copyright (c) 2025 ARC Development Team

---

## Acknowledgments

- **Claude Sonnet 4.5** - Strategic planning and architecture design
- **DeepSeek R1** - Deep reasoning and analysis
- **Qwen 2.5 32B** - Safety review and exploration
- **Llama 3 8B** - Offline validation
- **PyTorch Team** - ML framework
- **Streamlit Team** - Dashboard framework
- **FastAPI Team** - API framework
- **RunPod** - GPU infrastructure

---

**ARC v1.3.0 (Phase F Complete)**
*Production-Ready Multi-Agent Autonomous ML Research with Multi-GPU Training*

For questions, issues, or feedback, please open an issue on GitHub.
