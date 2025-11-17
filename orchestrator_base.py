"""
Orchestrator Base Skeleton

Provides the execution spine for ARC cycles.
Agent logic is pluggable - this handles:
- Cycle context management
- Memory load/validate/save
- Step dispatching
- Error handling and rollback
- Logging and metrics

This orchestrator is agent-agnostic. Agent implementations (Phase D multi-agent
system, single-agent systems, etc.) are injected as callbacks.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from memory_handler import MemoryHandler, get_memory_handler, ValidationFailedError
from config import ARCSettings, get_settings
from schemas import (
    Directive, HistorySummary, Constraints, SystemState,
    Proposals, Reviews
)


# Configure logging
logger = logging.getLogger(__name__)


class OrchestratorPhase(Enum):
    """Orchestrator execution phases."""
    INIT = "init"
    LOAD_MEMORY = "load_memory"
    HISTORIAN = "historian"
    DIRECTOR = "director"
    ARCHITECT = "architect"
    CRITIC = "critic"
    EXECUTOR = "executor"
    SAVE_MEMORY = "save_memory"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CycleContext:
    """
    Context object passed through orchestrator phases.

    Contains all data needed during a research cycle including memory
    state, agent outputs, and error tracking.
    """
    cycle_id: int
    phase: OrchestratorPhase = OrchestratorPhase.INIT

    # Memory state
    directive: Optional[Directive] = None
    proposals: Optional[Proposals] = None
    reviews: Optional[Reviews] = None
    history: Optional[HistorySummary] = None
    constraints: Optional[Constraints] = None
    state: Optional[SystemState] = None

    # Agent outputs (flexible dictionary for agent-specific data)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metrics
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.utcnow().isoformat()


class OrchestratorBase:
    """
    Base orchestrator for ARC research cycles.

    Handles execution flow, memory management, and error recovery.
    Agent implementations are injected as callbacks.

    Example:
        orchestrator = OrchestratorBase()

        # Register Phase D agents
        orchestrator.register_agent("historian", historian_agent.process)
        orchestrator.register_agent("director", director_agent.process)
        orchestrator.register_agent("architect", architect_agent.process)
        orchestrator.register_agent("critic", critic_agent.process)
        orchestrator.register_agent("executor", executor_agent.process)

        # Run cycle
        result = orchestrator.run_cycle(cycle_id=10)
    """

    def __init__(
        self,
        settings: Optional[ARCSettings] = None,
        memory: Optional[MemoryHandler] = None
    ):
        """
        Initialize orchestrator.

        Args:
            settings: Optional ARC settings. If None, uses get_settings()
            memory: Optional memory handler. If None, uses get_memory_handler()
        """
        self.settings = settings or get_settings()
        self.memory = memory or get_memory_handler(settings)

        # Agent callbacks (injected by multi-agent system)
        self.agent_callbacks: Dict[str, Callable[[CycleContext], CycleContext]] = {}

        # Execution hooks
        self.before_phase_hooks: List[Callable[[CycleContext, OrchestratorPhase], None]] = []
        self.after_phase_hooks: List[Callable[[CycleContext, OrchestratorPhase], None]] = []
        self.error_hooks: List[Callable[[CycleContext, Exception], None]] = []

        logger.info(f"OrchestratorBase initialized with memory_dir={self.memory.memory_dir}")

    # ========================================================================
    # Agent Registration
    # ========================================================================

    def register_agent(self, phase: str, callback: Callable[[CycleContext], CycleContext]):
        """
        Register an agent callback for a phase.

        Args:
            phase: Phase name (e.g., "historian", "director", "architect")
            callback: Agent callback function that takes and returns CycleContext
        """
        self.agent_callbacks[phase] = callback
        logger.info(f"Registered agent callback for phase: {phase}")

    def register_before_phase_hook(self, hook: Callable[[CycleContext, OrchestratorPhase], None]):
        """Register a hook to run before each phase."""
        self.before_phase_hooks.append(hook)

    def register_after_phase_hook(self, hook: Callable[[CycleContext, OrchestratorPhase], None]):
        """Register a hook to run after each phase."""
        self.after_phase_hooks.append(hook)

    def register_error_hook(self, hook: Callable[[CycleContext, Exception], None]):
        """Register a hook to run on errors."""
        self.error_hooks.append(hook)

    # ========================================================================
    # Main Cycle Execution
    # ========================================================================

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
            logger.info(f"Starting cycle {cycle_id}")

            # Phase 1: Load memory
            context = self._run_phase(context, OrchestratorPhase.LOAD_MEMORY, self._phase_load_memory)

            # Phase 2: Historian update
            if "historian" in self.agent_callbacks:
                context = self._run_phase(context, OrchestratorPhase.HISTORIAN, self.agent_callbacks["historian"])

            # Phase 3: Director directive
            if "director" in self.agent_callbacks:
                context = self._run_phase(context, OrchestratorPhase.DIRECTOR, self.agent_callbacks["director"])

            # Phase 4: Architect proposals
            if "architect" in self.agent_callbacks:
                context = self._run_phase(context, OrchestratorPhase.ARCHITECT, self.agent_callbacks["architect"])

            # Phase 5: Critic review
            if "critic" in self.agent_callbacks:
                context = self._run_phase(context, OrchestratorPhase.CRITIC, self.agent_callbacks["critic"])

            # Phase 6: Executor execution
            if "executor" in self.agent_callbacks:
                context = self._run_phase(context, OrchestratorPhase.EXECUTOR, self.agent_callbacks["executor"])

            # Phase 7: Save memory
            context = self._run_phase(context, OrchestratorPhase.SAVE_MEMORY, self._phase_save_memory)

            # Mark complete
            context.phase = OrchestratorPhase.COMPLETE
            context.completed_at = datetime.utcnow().isoformat()
            logger.info(f"Cycle {cycle_id} completed successfully")

        except Exception as e:
            context.phase = OrchestratorPhase.ERROR
            context.errors.append(str(e))
            logger.error(f"Cycle {cycle_id} failed: {e}", exc_info=True)
            self._handle_error(context, e)

        return context

    def _run_phase(
        self,
        context: CycleContext,
        phase: OrchestratorPhase,
        callback: Callable[[CycleContext], CycleContext]
    ) -> CycleContext:
        """
        Run a single phase with hooks.

        Args:
            context: Cycle context
            phase: Phase to run
            callback: Phase callback

        Returns:
            Updated context
        """
        # Update phase
        context.phase = phase
        logger.info(f"Entering phase: {phase.value}")

        # Before hooks
        for hook in self.before_phase_hooks:
            try:
                hook(context, phase)
            except Exception as e:
                logger.warning(f"Before-phase hook failed: {e}")

        # Execute phase
        context = callback(context)

        # After hooks
        for hook in self.after_phase_hooks:
            try:
                hook(context, phase)
            except Exception as e:
                logger.warning(f"After-phase hook failed: {e}")

        logger.info(f"Completed phase: {phase.value}")
        return context

    # ========================================================================
    # Memory Management Phases
    # ========================================================================

    def _phase_load_memory(self, context: CycleContext) -> CycleContext:
        """
        Load all memory with validation.

        Args:
            context: Cycle context

        Returns:
            Context with loaded memory
        """
        try:
            logger.debug("Loading memory files")
            context.history = self.memory.load_history_summary()
            context.constraints = self.memory.load_constraints()
            context.directive = self.memory.load_directive()
            context.state = self.memory.load_system_state()

            # Optionally load proposals/reviews if they exist
            if self.memory.file_exists('proposals.json'):
                try:
                    context.proposals = self.memory.load_proposals()
                except Exception as e:
                    logger.warning(f"Failed to load proposals: {e}")

            if self.memory.file_exists('reviews.json'):
                try:
                    context.reviews = self.memory.load_reviews()
                except Exception as e:
                    logger.warning(f"Failed to load reviews: {e}")

            logger.debug("Memory loaded successfully")

        except ValidationFailedError as e:
            context.errors.append(f"Memory validation failed: {e}")
            logger.error(f"Memory validation failed: {e}")
            raise
        except Exception as e:
            context.errors.append(f"Memory load failed: {e}")
            logger.error(f"Memory load failed: {e}")
            raise

        return context

    def _phase_save_memory(self, context: CycleContext) -> CycleContext:
        """
        Save all memory atomically.

        Args:
            context: Cycle context

        Returns:
            Context after save
        """
        try:
            logger.debug("Saving memory files")

            with self.memory.transaction():
                # Save all modified memory
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

            logger.debug("Memory saved successfully")

        except Exception as e:
            context.errors.append(f"Memory save failed: {e}")
            logger.error(f"Memory save failed: {e}")
            raise

        return context

    # ========================================================================
    # Error Handling
    # ========================================================================

    def _handle_error(self, context: CycleContext, error: Exception):
        """
        Handle orchestrator errors.

        Args:
            context: Cycle context
            error: Exception that occurred
        """
        # Run error hooks
        for hook in self.error_hooks:
            try:
                hook(context, error)
            except Exception as e:
                logger.error(f"Error hook failed: {e}")

        # Create snapshot for debugging
        try:
            backup_dir = self.memory.backup_memory()
            logger.info(f"Error snapshot created: {backup_dir}")
        except Exception as e:
            logger.error(f"Failed to create error snapshot: {e}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_cycle_stats(self, context: CycleContext) -> Dict[str, Any]:
        """
        Get statistics about a cycle execution.

        Args:
            context: Cycle context

        Returns:
            Dictionary with cycle stats
        """
        stats = {
            'cycle_id': context.cycle_id,
            'phase': context.phase.value,
            'started_at': context.started_at,
            'completed_at': context.completed_at,
            'error_count': len(context.errors),
            'warning_count': len(context.warnings),
            'has_errors': len(context.errors) > 0
        }

        # Calculate duration if completed
        if context.started_at and context.completed_at:
            from datetime import datetime
            start = datetime.fromisoformat(context.started_at)
            end = datetime.fromisoformat(context.completed_at)
            stats['duration_seconds'] = (end - start).total_seconds()

        return stats

    def validate_context(self, context: CycleContext) -> tuple[bool, List[str]]:
        """
        Validate that context has required data for saving.

        Args:
            context: Cycle context

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if context.directive is None:
            errors.append("Missing directive")
        if context.state is None:
            errors.append("Missing system state")
        if context.constraints is None:
            errors.append("Missing constraints")
        if context.history is None:
            errors.append("Missing history summary")

        return len(errors) == 0, errors


# ============================================================================
# Convenience Functions
# ============================================================================

def create_orchestrator(
    settings: Optional[ARCSettings] = None,
    memory: Optional[MemoryHandler] = None
) -> OrchestratorBase:
    """
    Create an orchestrator instance.

    Args:
        settings: Optional settings
        memory: Optional memory handler

    Returns:
        OrchestratorBase instance
    """
    return OrchestratorBase(settings=settings, memory=memory)
