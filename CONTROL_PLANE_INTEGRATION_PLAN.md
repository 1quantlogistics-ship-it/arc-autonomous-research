# Control Plane Integration Plan

**Status**: Memory Handler ✅ Complete | Control Plane Integration ⏳ Next

---

## What's Been Built

### ✅ Phase 1: Foundation (COMPLETE)
- **schemas.py** (458 lines) - Pydantic models for all memory files
- **config.py** (562 lines) - Environment-aware configuration
- **memory_handler.py** (550 lines) - Unified memory I/O with validation
- **Test suite** (440+ lines) - 48 unit tests for memory handler

---

## Next Steps: Control Plane Integration

### Task 1: Refactor control_plane.py

**Current State** ([api/control_plane.py](api/control_plane.py)):
- ❌ Hard-coded paths (`/workspace/arc/memory`, etc.)
- ❌ Raw JSON I/O (no validation)
- ❌ Dict-based memory access
- ❌ No atomic writes
- ❌ Manual error handling

**Target State**:
- ✅ Config-driven paths
- ✅ Schema-validated I/O via MemoryHandler
- ✅ Pydantic models throughout
- ✅ Atomic writes with rollback
- ✅ Structured error responses

**Changes Required**:

```python
# OLD (lines 1-27)
MEMORY_DIR = '/workspace/arc/memory'  # Hard-coded
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LOGS_DIR = '/workspace/arc/logs'

# NEW
from config import get_settings
from memory_handler import get_memory_handler

settings = get_settings()
memory = get_memory_handler(settings)
```

```python
# OLD (lines 65-83)
def load_system_state() -> Dict[str, Any]:
    path = os.path.join(MEMORY_DIR, 'system_state.json')
    with open(path, 'r') as f:
        return json.load(f)

# NEW
def load_system_state() -> SystemState:
    return memory.load_system_state()
```

**Affected Functions** (15 total):
1. `load_system_state()` → Use `memory.load_system_state()`
2. `save_system_state()` → Use `memory.save_system_state(model)`
3. `load_directive()` → Use `memory.load_directive()`
4. `load_constraints()` → Use `memory.load_constraints()`
5. `validate_command()` → Use `settings.allowed_commands`
6. `check_mode_permission()` → Update to use SystemState model
7. Logging setup → Use `settings.log_level`, `settings.logs_dir`

**Affected Endpoints** (7 total):
1. `GET /` → Update version to 1.1.0
2. `GET /status` → Return Pydantic models, not dicts
3. `POST /exec` → Schema-validate requests, atomic logging
4. `POST /train` → Validate TrainRequest against constraints
5. `POST /eval` → Schema-validate experiment results
6. `POST /archive` → Use memory.backup_memory()
7. `POST /rollback` → Use memory.restore_memory()

**Estimated Time**: 4-6 hours

---

### Task 2: Add Schema Validation to All Endpoints

**Pattern for Each Endpoint**:

```python
@app.post('/train')
async def train_experiment(req: TrainRequest):
    """Execute training with full validation."""
    try:
        # Load constraints with validation
        constraints = memory.load_constraints()

        # Validate config against constraints
        errors = validate_training_config(req.config, constraints)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"error": "validation_failed", "issues": errors}
            )

        # Load system state
        state = memory.load_system_state()

        # Update state atomically
        with memory.transaction():
            state.status = "training"
            state.active_experiments.append(ActiveExperiment(
                experiment_id=req.experiment_id,
                status="running",
                started_at=datetime.utcnow().isoformat()
            ))
            memory.save_system_state(state)

        # Execute training...

        return {
            "status": "started",
            "experiment_id": req.experiment_id
        }

    except ValidationFailedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MemoryHandlerError as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Endpoints to Update**:
1. `/exec` - Validate ExecRequest
2. `/train` - Validate against Constraints
3. `/eval` - Validate EvalRequest + results
4. `/status` - Return validated SystemState
5. `/archive` - Create validated snapshot
6. `/rollback` - Restore and validate

**Estimated Time**: 3-4 hours

---

### Task 3: Build Orchestrator Base Skeleton

**Goal**: Agent-agnostic orchestrator spine

**File**: `orchestrator_base.py` (new)

**Structure**:

```python
"""
Orchestrator Base Skeleton

Provides the execution spine for ARC cycles.
Agent logic is pluggable - this handles:
- Cycle context management
- Memory load/validate/save
- Step dispatching
- Error handling and rollback
- Logging and metrics
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from memory_handler import MemoryHandler, get_memory_handler
from config import ARCSettings, get_settings
from schemas import *


class OrchestratorPhase(Enum):
    """Orchestrator execution phases."""
    INIT = "init"
    HISTORIAN = "historian"
    DIRECTOR = "director"
    ARCHITECT = "architect"
    CRITIC = "critic"
    EXECUTOR = "executor"
    UPDATE = "update"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CycleContext:
    """Context object passed through orchestrator phases."""
    cycle_id: int
    directive: Optional[Directive] = None
    proposals: Optional[Proposals] = None
    reviews: Optional[Reviews] = None
    history: Optional[HistorySummary] = None
    constraints: Optional[Constraints] = None
    state: Optional[SystemState] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class OrchestratorBase:
    """
    Base orchestrator for ARC research cycles.

    Handles execution flow, memory management, and error recovery.
    Agent implementations are injected as callbacks.
    """

    def __init__(
        self,
        settings: Optional[ARCSettings] = None,
        memory: Optional[MemoryHandler] = None
    ):
        self.settings = settings or get_settings()
        self.memory = memory or get_memory_handler(settings)

        # Agent callbacks (injected by multi-agent system)
        self.agent_callbacks: Dict[str, Callable] = {}

        # Execution hooks
        self.before_phase_hooks: List[Callable] = []
        self.after_phase_hooks: List[Callable] = []
        self.error_hooks: List[Callable] = []

    def register_agent(self, phase: str, callback: Callable):
        """Register an agent callback for a phase."""
        self.agent_callbacks[phase] = callback

    def run_cycle(self, cycle_id: int) -> CycleContext:
        """
        Execute a complete research cycle.

        Args:
            cycle_id: Cycle number

        Returns:
            CycleContext with results
        """
        context = CycleContext(cycle_id=cycle_id)

        try:
            # Phase 1: Load memory
            context = self._phase_load_memory(context)

            # Phase 2: Historian update
            if "historian" in self.agent_callbacks:
                context = self.agent_callbacks["historian"](context)

            # Phase 3: Director directive
            if "director" in self.agent_callbacks:
                context = self.agent_callbacks["director"](context)

            # Phase 4: Architect proposals
            if "architect" in self.agent_callbacks:
                context = self.agent_callbacks["architect"](context)

            # Phase 5: Critic review
            if "critic" in self.agent_callbacks:
                context = self.agent_callbacks["critic"](context)

            # Phase 6: Executor execution
            if "executor" in self.agent_callbacks:
                context = self.agent_callbacks["executor"](context)

            # Phase 7: Save memory
            context = self._phase_save_memory(context)

        except Exception as e:
            context.errors.append(str(e))
            self._handle_error(context, e)

        return context

    def _phase_load_memory(self, context: CycleContext) -> CycleContext:
        """Load all memory with validation."""
        try:
            context.history = self.memory.load_history_summary()
            context.constraints = self.memory.load_constraints()
            context.directive = self.memory.load_directive()
            context.state = self.memory.load_system_state()
        except Exception as e:
            context.errors.append(f"Memory load failed: {e}")
            raise

        return context

    def _phase_save_memory(self, context: CycleContext) -> CycleContext:
        """Save all memory atomically."""
        try:
            with self.memory.transaction():
                if context.directive:
                    self.memory.save_directive(context.directive)
                if context.history:
                    self.memory.save_history_summary(context.history)
                if context.proposals:
                    self.memory.save_proposals(context.proposals)
                if context.reviews:
                    self.memory.save_reviews(context.reviews)
                if context.state:
                    self.memory.save_system_state(context.state)
        except Exception as e:
            context.errors.append(f"Memory save failed: {e}")
            raise

        return context

    def _handle_error(self, context: CycleContext, error: Exception):
        """Handle orchestrator errors."""
        for hook in self.error_hooks:
            hook(context, error)

        # Log error
        import logging
        logging.error(f"Cycle {context.cycle_id} failed: {error}")

        # Create snapshot for debugging
        self.memory.backup_memory()
```

**Integration with Phase D Agents**:

```python
# In agents/supervisor.py or similar
from orchestrator_base import OrchestratorBase, CycleContext

orchestrator = OrchestratorBase()

# Register Phase D agents
orchestrator.register_agent("historian", historian_agent.process)
orchestrator.register_agent("director", director_agent.process)
orchestrator.register_agent("architect", architect_agent.process)
orchestrator.register_agent("critic", critic_agent.process)
orchestrator.register_agent("executor", executor_agent.process)

# Run cycle
result = orchestrator.run_cycle(cycle_id=10)
```

**Estimated Time**: 3-4 hours

---

### Task 4: Expand Test Coverage

**New Test Files Needed**:

1. **tests/integration/test_control_plane_integration.py**
   - Test control plane with real memory handler
   - Test endpoint validation
   - Test error responses
   - Test mode switching

2. **tests/integration/test_orchestrator_base.py**
   - Test full cycle execution
   - Test phase transitions
   - Test error handling and rollback
   - Test agent callback system

3. **tests/integration/test_phase_d_compatibility.py**
   - Test memory handler with Phase D agents
   - Test orchestrator with multi-agent system
   - Test consensus integration
   - Test supervisor oversight

**Estimated Time**: 6-8 hours

---

## Integration Checklist

### Control Plane (4-6 hours)
- [ ] Replace hard-coded paths with config
- [ ] Replace JSON I/O with MemoryHandler
- [ ] Update all helper functions to use schemas
- [ ] Add schema validation to all endpoints
- [ ] Add structured error responses
- [ ] Update logging to use config
- [ ] Test all endpoints with validation

### Orchestrator Base (3-4 hours)
- [ ] Create orchestrator_base.py
- [ ] Implement CycleContext
- [ ] Implement phase dispatching
- [ ] Add transaction support
- [ ] Add error handling and rollback
- [ ] Add agent callback registration
- [ ] Test with mock agents

### Testing (6-8 hours)
- [ ] Integration tests for control plane
- [ ] Integration tests for orchestrator
- [ ] End-to-end tests with Phase D agents
- [ ] Performance tests for memory handler
- [ ] Concurrency tests for multi-agent
- [ ] Error recovery tests

### Documentation (2-3 hours)
- [ ] Update control plane API docs
- [ ] Document orchestrator usage
- [ ] Add migration guide for existing code
- [ ] Update Phase D integration docs

---

## Total Estimated Time

**15-21 hours** for complete integration

**Breakdown**:
- Control Plane: 4-6 hours
- Orchestrator Base: 3-4 hours
- Testing: 6-8 hours
- Documentation: 2-3 hours

---

## Success Criteria

✅ All control plane endpoints use schema validation
✅ All memory I/O goes through MemoryHandler
✅ No hard-coded paths remain
✅ All tests pass (unit + integration)
✅ Phase D agents work with new infrastructure
✅ Backward compatibility maintained
✅ Performance meets requirements (<100ms for memory ops)

---

## Next Session Plan

**Priority 1**: Refactor control_plane.py (4-6 hours)
1. Update imports and initialization
2. Replace all helper functions
3. Update all endpoints
4. Test each endpoint individually

**Priority 2**: Build orchestrator_base.py (3-4 hours)
1. Implement core structure
2. Add phase dispatching
3. Integrate with memory handler
4. Test with mock agents

**Priority 3**: Integration testing (6-8 hours)
1. Write integration tests
2. Test with Phase D agents
3. Performance testing
4. Error recovery testing

---

**Status**: Foundation complete, ready for integration work!
