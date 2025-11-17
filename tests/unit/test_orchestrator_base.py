"""
Unit tests for Orchestrator Base.

Tests the orchestrator's ability to manage research cycles,
dispatch phases, handle errors, and integrate with memory handler.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from orchestrator_base import (
    OrchestratorBase, CycleContext, OrchestratorPhase,
    create_orchestrator
)
from memory_handler import MemoryHandler, ValidationFailedError
from schemas import Directive, DirectiveMode, Objective, SystemState, OperatingMode
from config import ARCSettings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_orchestrator(tmp_path):
    """Create an orchestrator with temporary directory."""
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    memory = MemoryHandler(settings)
    memory.initialize_memory(force=True)

    orchestrator = OrchestratorBase(settings=settings, memory=memory)

    return orchestrator


@pytest.fixture
def mock_agent_callback():
    """Create a mock agent callback."""
    def callback(context: CycleContext) -> CycleContext:
        context.agent_outputs["mock_agent"] = "completed"
        return context

    return callback


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorInit:
    """Test orchestrator initialization."""

    def test_create_orchestrator(self, temp_orchestrator):
        """Test creating an orchestrator."""
        assert temp_orchestrator is not None
        assert temp_orchestrator.memory is not None
        assert temp_orchestrator.settings is not None

    def test_orchestrator_with_default_settings(self):
        """Test creating orchestrator with default settings."""
        orchestrator = create_orchestrator()
        assert orchestrator is not None

    def test_agent_callbacks_empty(self, temp_orchestrator):
        """Test that agent callbacks start empty."""
        assert len(temp_orchestrator.agent_callbacks) == 0


# ============================================================================
# Agent Registration Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorAgentRegistration:
    """Test agent registration."""

    def test_register_agent(self, temp_orchestrator, mock_agent_callback):
        """Test registering an agent callback."""
        temp_orchestrator.register_agent("historian", mock_agent_callback)

        assert "historian" in temp_orchestrator.agent_callbacks
        assert temp_orchestrator.agent_callbacks["historian"] == mock_agent_callback

    def test_register_multiple_agents(self, temp_orchestrator):
        """Test registering multiple agents."""
        callbacks = {
            "historian": lambda ctx: ctx,
            "director": lambda ctx: ctx,
            "architect": lambda ctx: ctx,
            "critic": lambda ctx: ctx,
            "executor": lambda ctx: ctx
        }

        for phase, callback in callbacks.items():
            temp_orchestrator.register_agent(phase, callback)

        assert len(temp_orchestrator.agent_callbacks) == 5
        for phase in callbacks:
            assert phase in temp_orchestrator.agent_callbacks

    def test_register_hooks(self, temp_orchestrator):
        """Test registering hooks."""
        before_hook = Mock()
        after_hook = Mock()
        error_hook = Mock()

        temp_orchestrator.register_before_phase_hook(before_hook)
        temp_orchestrator.register_after_phase_hook(after_hook)
        temp_orchestrator.register_error_hook(error_hook)

        assert before_hook in temp_orchestrator.before_phase_hooks
        assert after_hook in temp_orchestrator.after_phase_hooks
        assert error_hook in temp_orchestrator.error_hooks


# ============================================================================
# CycleContext Tests
# ============================================================================

@pytest.mark.unit
class TestCycleContext:
    """Test cycle context object."""

    def test_create_context(self):
        """Test creating a cycle context."""
        context = CycleContext(cycle_id=1)

        assert context.cycle_id == 1
        assert context.phase == OrchestratorPhase.INIT
        assert context.errors == []
        assert context.warnings == []
        assert context.agent_outputs == {}
        assert context.started_at is not None

    def test_context_tracks_errors(self):
        """Test context error tracking."""
        context = CycleContext(cycle_id=1)

        context.errors.append("Test error")
        context.warnings.append("Test warning")

        assert len(context.errors) == 1
        assert len(context.warnings) == 1


# ============================================================================
# Memory Loading Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorMemoryLoading:
    """Test memory loading phase."""

    def test_load_memory_phase(self, temp_orchestrator):
        """Test loading memory in a phase."""
        context = CycleContext(cycle_id=1)
        context = temp_orchestrator._phase_load_memory(context)

        assert context.directive is not None
        assert isinstance(context.directive, Directive)
        assert context.history is not None
        assert context.constraints is not None
        assert context.state is not None

    def test_load_memory_validates(self, temp_orchestrator, tmp_path):
        """Test that loading validates schemas."""
        # Corrupt a memory file
        bad_file = temp_orchestrator.memory.memory_dir / "directive.json"
        bad_file.write_text('{"invalid": "schema"}')

        context = CycleContext(cycle_id=1)

        with pytest.raises(ValidationFailedError):
            temp_orchestrator._phase_load_memory(context)


# ============================================================================
# Memory Saving Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorMemorySaving:
    """Test memory saving phase."""

    def test_save_memory_phase(self, temp_orchestrator):
        """Test saving memory in a phase."""
        # Load initial context
        context = CycleContext(cycle_id=1)
        context = temp_orchestrator._phase_load_memory(context)

        # Modify directive
        context.directive.cycle_id = 99
        context.directive.notes = "Updated via orchestrator"

        # Save
        context = temp_orchestrator._phase_save_memory(context)

        # Reload and verify
        reloaded = temp_orchestrator.memory.load_directive()
        assert reloaded.cycle_id == 99
        assert reloaded.notes == "Updated via orchestrator"

    def test_save_memory_is_atomic(self, temp_orchestrator):
        """Test that save is atomic (rolls back on error)."""
        # Load initial context
        context = CycleContext(cycle_id=1)
        context = temp_orchestrator._phase_load_memory(context)

        initial_cycle_id = context.directive.cycle_id

        # Modify directive
        context.directive.cycle_id = 99

        # Simulate error during save by making state None (will fail validation)
        context.state = None

        # Save should fail
        with pytest.raises(Exception):
            temp_orchestrator._phase_save_memory(context)

        # Reload and verify rollback
        reloaded = temp_orchestrator.memory.load_directive()
        assert reloaded.cycle_id == initial_cycle_id  # Not 99


# ============================================================================
# Cycle Execution Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorCycleExecution:
    """Test full cycle execution."""

    def test_run_cycle_without_agents(self, temp_orchestrator):
        """Test running a cycle without any agents registered."""
        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.cycle_id == 1
        assert result.phase == OrchestratorPhase.COMPLETE
        assert len(result.errors) == 0
        assert result.directive is not None
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_run_cycle_with_agent(self, temp_orchestrator):
        """Test running a cycle with a registered agent."""
        # Register a simple agent
        def historian_agent(context: CycleContext) -> CycleContext:
            context.agent_outputs["historian"] = "History updated"
            return context

        temp_orchestrator.register_agent("historian", historian_agent)

        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert "historian" in result.agent_outputs
        assert result.agent_outputs["historian"] == "History updated"

    def test_run_cycle_with_multiple_agents(self, temp_orchestrator):
        """Test running a cycle with multiple agents."""
        # Register multiple agents
        agents = {
            "historian": lambda ctx: ctx,
            "director": lambda ctx: ctx,
            "architect": lambda ctx: ctx,
            "critic": lambda ctx: ctx,
            "executor": lambda ctx: ctx
        }

        for phase, callback in agents.items():
            temp_orchestrator.register_agent(phase, callback)

        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert len(result.errors) == 0

    def test_run_cycle_calls_hooks(self, temp_orchestrator):
        """Test that cycle execution calls hooks."""
        before_hook = Mock()
        after_hook = Mock()

        temp_orchestrator.register_before_phase_hook(before_hook)
        temp_orchestrator.register_after_phase_hook(after_hook)

        temp_orchestrator.run_cycle(cycle_id=1)

        # Should be called for load_memory and save_memory phases (at least)
        assert before_hook.call_count >= 2
        assert after_hook.call_count >= 2


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorErrorHandling:
    """Test error handling."""

    def test_run_cycle_handles_agent_error(self, temp_orchestrator):
        """Test that orchestrator handles agent errors gracefully."""
        # Register agent that raises error
        def failing_agent(context: CycleContext) -> CycleContext:
            raise ValueError("Simulated agent error")

        temp_orchestrator.register_agent("historian", failing_agent)

        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.ERROR
        assert len(result.errors) > 0
        assert "Simulated agent error" in result.errors[0]

    def test_error_triggers_snapshot(self, temp_orchestrator):
        """Test that errors trigger snapshot creation."""
        # Register failing agent
        def failing_agent(context: CycleContext) -> CycleContext:
            raise ValueError("Simulated error")

        temp_orchestrator.register_agent("historian", failing_agent)

        result = temp_orchestrator.run_cycle(cycle_id=1)

        # Check that snapshot was created
        snapshots = list(temp_orchestrator.settings.snapshots_dir.glob("backup_*"))
        assert len(snapshots) > 0

    def test_error_hook_is_called(self, temp_orchestrator):
        """Test that error hooks are called on error."""
        error_hook = Mock()
        temp_orchestrator.register_error_hook(error_hook)

        # Register failing agent
        def failing_agent(context: CycleContext) -> CycleContext:
            raise ValueError("Simulated error")

        temp_orchestrator.register_agent("historian", failing_agent)

        temp_orchestrator.run_cycle(cycle_id=1)

        # Error hook should be called
        error_hook.assert_called_once()


# ============================================================================
# Utility Methods Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorUtilities:
    """Test utility methods."""

    def test_get_cycle_stats(self, temp_orchestrator):
        """Test getting cycle statistics."""
        result = temp_orchestrator.run_cycle(cycle_id=1)
        stats = temp_orchestrator.get_cycle_stats(result)

        assert stats['cycle_id'] == 1
        assert stats['phase'] == OrchestratorPhase.COMPLETE.value
        assert stats['error_count'] == 0
        assert stats['has_errors'] is False
        assert 'duration_seconds' in stats

    def test_validate_context(self, temp_orchestrator):
        """Test context validation."""
        # Empty context should fail
        context = CycleContext(cycle_id=1)
        is_valid, errors = temp_orchestrator.validate_context(context)

        assert not is_valid
        assert len(errors) > 0

        # Load memory to make context valid
        context = temp_orchestrator._phase_load_memory(context)
        is_valid, errors = temp_orchestrator.validate_context(context)

        assert is_valid
        assert len(errors) == 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.unit
class TestOrchestratorIntegration:
    """Test orchestrator integration scenarios."""

    def test_full_cycle_with_memory_updates(self, temp_orchestrator):
        """Test full cycle that updates memory."""
        # Register agent that modifies directive
        def director_agent(context: CycleContext) -> CycleContext:
            context.directive.cycle_id += 1
            context.directive.notes = "Updated by director"
            return context

        temp_orchestrator.register_agent("director", director_agent)

        # Run cycle
        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE

        # Verify memory was saved
        reloaded = temp_orchestrator.memory.load_directive()
        assert reloaded.notes == "Updated by director"

    def test_cycle_with_agent_chain(self, temp_orchestrator):
        """Test cycle with multiple agents passing data."""
        # Chain of agents
        def historian_agent(context: CycleContext) -> CycleContext:
            context.agent_outputs["step"] = 1
            return context

        def director_agent(context: CycleContext) -> CycleContext:
            context.agent_outputs["step"] = context.agent_outputs["step"] + 1
            return context

        def architect_agent(context: CycleContext) -> CycleContext:
            context.agent_outputs["step"] = context.agent_outputs["step"] + 1
            return context

        temp_orchestrator.register_agent("historian", historian_agent)
        temp_orchestrator.register_agent("director", director_agent)
        temp_orchestrator.register_agent("architect", architect_agent)

        result = temp_orchestrator.run_cycle(cycle_id=1)

        assert result.agent_outputs["step"] == 3
        assert result.phase == OrchestratorPhase.COMPLETE
