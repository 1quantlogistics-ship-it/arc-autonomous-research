# ARC - Autonomous Research Collective

**Version:** 1.2.0 (Phase D + Phase E Complete)
**Status:** Advanced Experiment Design - Production Ready
**License:** MIT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Phase E](https://img.shields.io/badge/Phase%20E-Complete-brightgreen.svg)](https://github.com/1quantlogistics-ship-it/arc-autonomous-research)

## Overview

ARC (Autonomous Research Collective) is a **multi-agent autonomous ML research framework** that uses LLM-based reasoning agents to design, execute, and learn from machine learning experiments. It now features **advanced experiment design capabilities** including architecture search, loss engineering, curriculum learning, and multi-objective optimization.

### What's New in Version 1.2.0 (Phase E)

🎉 **19 new advanced ML capabilities** added across 3 development weeks:

- ✅ **Architecture Grammar & NAS** - Neural Architecture Search with constraint validation
- ✅ **Augmentation Policy Learning** - AutoAugment with 14 safe operations
- ✅ **Loss Engineering** - Focal loss, multi-task learning, class weighting
- ✅ **Curriculum Learning** - Progressive difficulty with 4 pacing strategies
- ✅ **Multi-Objective Optimization** - Pareto frontier tracking with hypervolume metrics

**Phase E Stats:**
- 5,500+ lines of production code
- 7 new schemas (2,845 lines)
- 30+ comprehensive end-to-end tests
- Full backward compatibility with Phase D
- FDA-compliant clinical safety constraints

### Core Features

**Multi-Agent Governance (Phase D):**
- 🤖 **9 Specialized Agents** with democratic voting and weighted consensus
- 🧠 **Heterogeneous Models** - Different LLMs per role (Claude, DeepSeek, Qwen, Llama)
- 🛡️ **Supervisor Veto Power** - Final safety gatekeeper with override authority
- 📊 **FDA-Aligned Logging** - Automatic traceability and provenance tracking
- ⚙️ **Role-Specific Timeouts** - Configurable per-agent reasoning time
- 🐳 **RunPod Deployment** - Production Docker with GPU support

**Advanced Experiment Design (Phase E):**
- 🏗️ **Architecture Search (NAS)** - Random, evolutionary, ENAS, DARTS strategies
- 🔄 **Augmentation Policy** - AutoAugment with FDA-safe operations
- ⚖️ **Loss Engineering** - Focal, Dice, Tversky, multi-task learning
- 📈 **Curriculum Learning** - Progressive training from easy to hard
- 🎯 **Multi-Objective Optimization** - Pareto frontier with hypervolume tracking

**Infrastructure:**
- 🔒 **Safety-First Design** - SEMI/AUTO/FULL autonomy modes
- 📁 **File-Based Protocol** - JSON inter-agent communication
- 🔬 **Real GPU Training** - PyTorch integration with experiment tracking
- 📊 **Enhanced Dashboard** - 8 tabs with real-time monitoring
- 💾 **Snapshot & Rollback** - State preservation and restoration
- 🌐 **Offline Operation** - Full functionality without network (mock mode)

---

## Table of Contents

- [Architecture](#architecture)
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
│  Claude Sonnet 4.5 │ DeepSeek R1 │ Qwen 2.5 │ Llama 3 8B    │
│  (Strategy)        │ (Analysis)  │ (Safety) │ (Validator)   │
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

## Phase E: Advanced Experiment Design

Phase E adds **19 sophisticated ML capabilities** enabling ARC to autonomously explore advanced training techniques while maintaining clinical safety.

### Week 1: Architecture Grammar + Augmentation Policy

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

### Week 2: Loss Engineering + Curriculum Learning

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

### Week 3: Multi-Objective Optimization

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

## Installation

### Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.1+ (for GPU training)
- **Docker**: 20.10+ (for containerized deployment)
- **Git**: 2.30+

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
├── schemas/                 # Phase E: 5 new schemas (2,845 lines)
│   ├── architecture_grammar.py      (593 lines - NAS)
│   ├── augmentation_policy.py       (632 lines - AutoAugment)
│   ├── loss_config.py               (472 lines - Loss engineering)
│   ├── curriculum_strategy.py       (496 lines - Curriculum learning)
│   ├── multi_objective.py           (652 lines - Pareto optimization)
│   └── experiment_schemas.py
│
├── tools/                   # Phase E: Loss functions + existing tools
│   ├── loss_functions.py            (544 lines - PyTorch losses)
│   ├── world_model.py               (extended - multi-objective)
│   ├── acuvue_tools.py
│   ├── dataset_fusion.py
│   ├── drift_detector.py
│   ├── failure_predictor.py
│   ├── mode_collapse_engine.py
│   └── dev_logger.py
│
├── config/                  # Configuration management
│   ├── experiment_config_generator.py  (extended - Phase E translation)
│   ├── loader.py
│   ├── agents.example.yaml
│   └── models.example.yaml
│
├── api/                     # Core services
│   ├── control_plane.py
│   ├── multi_agent_orchestrator.py
│   ├── training_executor.py
│   ├── dashboard.py
│   ├── ui_endpoints.py
│   └── ui_state_poller.py
│
├── tests/                   # Comprehensive test suite
│   ├── test_multi_objective_e2e.py  (714 lines - Phase E tests)
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

**Modified Files (7):**
1. `agents/parameter_scientist.py` - +200 lines (architecture, loss proposals)
2. `agents/instructor_agent.py` - +180 lines (augmentation, curriculum)
3. `agents/explorer.py` - +150 lines (augmentation evolution)
4. `agents/historian_agent.py` - +380 lines (curriculum + Pareto tracking)
5. `agents/critic_agent.py` - +120 lines (architecture + augmentation validation)
6. `config/experiment_config_generator.py` - +200 lines (translation methods)
7. `tools/world_model.py` - +140 lines (multi-objective predictions)

**Total:** +1,370 lines of enhancements

**Grand Total:** 5,473 lines added in Phase E

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

**Phase E Enhancements:**
- Additional NAS strategies (DARTS gradient-based search)
- More augmentation operations (elastic deformation variants)
- Additional loss functions (Lovasz-Softmax, IoU-based)
- Advanced curriculum strategies (competence-based pacing)
- High-dimensional Pareto visualization (>3 objectives)

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
  version = {1.2.0},
  url = {https://github.com/1quantlogistics-ship-it/arc-autonomous-research},
  note = {Phase E: Advanced Experiment Design - Architecture Search,
          Loss Engineering, Curriculum Learning, Multi-Objective Optimization}
}
```

**Phase E Features:**
- Architecture Grammar & NAS (Week 1)
- Augmentation Policy Learning (Week 1)
- Loss Engineering & Multi-Task Learning (Week 2)
- Curriculum Learning (Week 2)
- Multi-Objective Optimization with Pareto Frontier Tracking (Week 3)

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

**ARC v1.2.0 (Phase D + Phase E Complete)**
*Multi-Agent Autonomous ML Research with Advanced Experiment Design*

For questions, issues, or feedback, please open an issue on GitHub.
