"""
Integration tests for Phase D Multi-Agent compatibility.

Tests that the v1.1.0 memory handler and orchestrator work correctly
with the Phase D multi-agent system.
"""

import pytest
from pathlib import Path

from memory_handler import MemoryHandler, get_memory_handler, reset_memory_handler
from orchestrator_base import OrchestratorBase, CycleContext, OrchestratorPhase
from config import ARCSettings
from schemas import (
    Directive, DirectiveMode, Objective,
    SystemState, OperatingMode,
    Proposals, Proposal, NoveltyClass, ExpectedImpact,
    Reviews, Review, ReviewDecision
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def phase_d_env(tmp_path):
    """Create Phase D environment with memory handler."""
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    memory = MemoryHandler(settings)
    memory.initialize_memory(force=True)

    orchestrator = OrchestratorBase(settings=settings, memory=memory)

    yield settings, memory, orchestrator

    reset_memory_handler()


# ============================================================================
# Memory Handler with Phase D Schemas
# ============================================================================

@pytest.mark.integration
class TestPhaseD_MemoryHandler:
    """Test memory handler with Phase D schemas."""

    def test_load_save_proposals(self, phase_d_env):
        """Test loading and saving proposals (Architect output)."""
        settings, memory, orchestrator = phase_d_env

        # Create proposals
        proposals = Proposals(
            cycle_id=1,
            proposals=[
                Proposal(
                    proposal_id="prop_001",
                    description="Test proposal",
                    novelty_class=NoveltyClass.EXPLOIT,
                    expected_impact=ExpectedImpact.MEDIUM,
                    rationale="Testing Phase D integration"
                )
            ]
        )

        # Save
        memory.save_proposals(proposals)

        # Reload and verify
        loaded = memory.load_proposals()
        assert loaded.cycle_id == 1
        assert len(loaded.proposals) == 1
        assert loaded.proposals[0].proposal_id == "prop_001"

    def test_load_save_reviews(self, phase_d_env):
        """Test loading and saving reviews (Critic output)."""
        settings, memory, orchestrator = phase_d_env

        # Create reviews
        reviews = Reviews(
            cycle_id=1,
            reviews=[
                Review(
                    proposal_id="prop_001",
                    decision=ReviewDecision.APPROVE,
                    confidence=0.85,
                    reasoning="Proposal meets safety criteria"
                )
            ]
        )

        # Save
        memory.save_reviews(reviews)

        # Reload and verify
        loaded = memory.load_reviews()
        assert loaded.cycle_id == 1
        assert len(loaded.reviews) == 1
        assert loaded.reviews[0].decision == ReviewDecision.APPROVE

    def test_transaction_with_multiple_phase_d_files(self, phase_d_env):
        """Test transaction with proposals and reviews."""
        settings, memory, orchestrator = phase_d_env

        with memory.transaction():
            # Save proposals
            proposals = Proposals(
                cycle_id=1,
                proposals=[
                    Proposal(
                        proposal_id="prop_002",
                        description="Transaction test",
                        novelty_class=NoveltyClass.EXPLORE,
                        expected_impact=ExpectedImpact.HIGH,
                        rationale="Testing atomicity"
                    )
                ]
            )
            memory.save_proposals(proposals)

            # Save reviews
            reviews = Reviews(
                cycle_id=1,
                reviews=[
                    Review(
                        proposal_id="prop_002",
                        decision=ReviewDecision.APPROVE,
                        confidence=0.9,
                        reasoning="High impact proposal"
                    )
                ]
            )
            memory.save_reviews(reviews)

        # Both should be saved
        assert memory.file_exists('proposals.json')
        assert memory.file_exists('reviews.json')

        loaded_proposals = memory.load_proposals()
        loaded_reviews = memory.load_reviews()

        assert len(loaded_proposals.proposals) == 1
        assert len(loaded_reviews.reviews) == 1


# ============================================================================
# Orchestrator with Phase D Agent Callbacks
# ============================================================================

@pytest.mark.integration
class TestPhaseD_Orchestrator:
    """Test orchestrator with Phase D agent callbacks."""

    def test_historian_agent_callback(self, phase_d_env):
        """Test orchestrator with historian agent."""
        settings, memory, orchestrator = phase_d_env

        # Mock historian agent
        def historian_agent(context: CycleContext) -> CycleContext:
            # Historian updates history summary
            context.history.total_cycles += 1
            context.history.last_update = "2025-11-16T00:00:00Z"
            context.agent_outputs["historian"] = "History updated"
            return context

        orchestrator.register_agent("historian", historian_agent)

        result = orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert "historian" in result.agent_outputs
        assert result.history.total_cycles == 1

    def test_director_agent_callback(self, phase_d_env):
        """Test orchestrator with director agent."""
        settings, memory, orchestrator = phase_d_env

        # Mock director agent
        def director_agent(context: CycleContext) -> CycleContext:
            # Director updates directive
            context.directive.mode = DirectiveMode.REFINE
            context.directive.notes = "Director updated directive"
            context.agent_outputs["director"] = "Directive set"
            return context

        orchestrator.register_agent("director", director_agent)

        result = orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert result.directive.mode == DirectiveMode.REFINE
        assert "director" in result.agent_outputs

    def test_architect_agent_callback(self, phase_d_env):
        """Test orchestrator with architect agent."""
        settings, memory, orchestrator = phase_d_env

        # Mock architect agent
        def architect_agent(context: CycleContext) -> CycleContext:
            # Architect creates proposals
            context.proposals = Proposals(
                cycle_id=context.cycle_id,
                proposals=[
                    Proposal(
                        proposal_id="arch_001",
                        description="Architect proposal",
                        novelty_class=NoveltyClass.EXPLOIT,
                        expected_impact=ExpectedImpact.HIGH,
                        rationale="Generated by architect"
                    )
                ]
            )
            context.agent_outputs["architect"] = "Proposals generated"
            return context

        orchestrator.register_agent("architect", architect_agent)

        result = orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert result.proposals is not None
        assert len(result.proposals.proposals) == 1
        assert "architect" in result.agent_outputs

        # Verify saved to memory
        loaded_proposals = memory.load_proposals()
        assert len(loaded_proposals.proposals) == 1

    def test_critic_agent_callback(self, phase_d_env):
        """Test orchestrator with critic agent."""
        settings, memory, orchestrator = phase_d_env

        # Mock critic agent
        def critic_agent(context: CycleContext) -> CycleContext:
            # Critic creates reviews
            if context.proposals:
                context.reviews = Reviews(
                    cycle_id=context.cycle_id,
                    reviews=[
                        Review(
                            proposal_id=prop.proposal_id,
                            decision=ReviewDecision.APPROVE,
                            confidence=0.8,
                            reasoning="Approved by critic"
                        )
                        for prop in context.proposals.proposals
                    ]
                )
            context.agent_outputs["critic"] = "Reviews completed"
            return context

        # Also need architect to create proposals first
        def architect_agent(context: CycleContext) -> CycleContext:
            context.proposals = Proposals(
                cycle_id=context.cycle_id,
                proposals=[
                    Proposal(
                        proposal_id="crit_001",
                        description="Proposal for review",
                        novelty_class=NoveltyClass.EXPLORE,
                        expected_impact=ExpectedImpact.MEDIUM,
                        rationale="Test proposal"
                    )
                ]
            )
            return context

        orchestrator.register_agent("architect", architect_agent)
        orchestrator.register_agent("critic", critic_agent)

        result = orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.COMPLETE
        assert result.reviews is not None
        assert len(result.reviews.reviews) == 1
        assert result.reviews.reviews[0].decision == ReviewDecision.APPROVE

        # Verify saved to memory
        loaded_reviews = memory.load_reviews()
        assert len(loaded_reviews.reviews) == 1

    def test_full_phase_d_pipeline(self, phase_d_env):
        """Test full Phase D pipeline: Historian → Director → Architect → Critic."""
        settings, memory, orchestrator = phase_d_env

        # Mock all Phase D agents
        def historian_agent(context: CycleContext) -> CycleContext:
            context.history.total_cycles += 1
            context.agent_outputs["historian"] = "done"
            return context

        def director_agent(context: CycleContext) -> CycleContext:
            context.directive.cycle_id = context.cycle_id
            context.directive.mode = DirectiveMode.REFINE
            context.agent_outputs["director"] = "done"
            return context

        def architect_agent(context: CycleContext) -> CycleContext:
            context.proposals = Proposals(
                cycle_id=context.cycle_id,
                proposals=[
                    Proposal(
                        proposal_id=f"prop_{i}",
                        description=f"Proposal {i}",
                        novelty_class=NoveltyClass.EXPLOIT,
                        expected_impact=ExpectedImpact.MEDIUM,
                        rationale=f"Rationale {i}"
                    )
                    for i in range(3)
                ]
            )
            context.agent_outputs["architect"] = "done"
            return context

        def critic_agent(context: CycleContext) -> CycleContext:
            context.reviews = Reviews(
                cycle_id=context.cycle_id,
                reviews=[
                    Review(
                        proposal_id=prop.proposal_id,
                        decision=ReviewDecision.APPROVE if i % 2 == 0 else ReviewDecision.REJECT,
                        confidence=0.85,
                        reasoning=f"Review for {prop.proposal_id}"
                    )
                    for i, prop in enumerate(context.proposals.proposals)
                ]
            )
            context.agent_outputs["critic"] = "done"
            return context

        # Register all agents
        orchestrator.register_agent("historian", historian_agent)
        orchestrator.register_agent("director", director_agent)
        orchestrator.register_agent("architect", architect_agent)
        orchestrator.register_agent("critic", critic_agent)

        # Run cycle
        result = orchestrator.run_cycle(cycle_id=5)

        # Verify completion
        assert result.phase == OrchestratorPhase.COMPLETE
        assert len(result.errors) == 0

        # Verify all agents ran
        assert result.agent_outputs["historian"] == "done"
        assert result.agent_outputs["director"] == "done"
        assert result.agent_outputs["architect"] == "done"
        assert result.agent_outputs["critic"] == "done"

        # Verify memory state
        assert result.history.total_cycles == 1
        assert result.directive.cycle_id == 5
        assert len(result.proposals.proposals) == 3
        assert len(result.reviews.reviews) == 3

        # Verify persistence
        loaded_directive = memory.load_directive()
        loaded_proposals = memory.load_proposals()
        loaded_reviews = memory.load_reviews()

        assert loaded_directive.cycle_id == 5
        assert len(loaded_proposals.proposals) == 3
        assert len(loaded_reviews.reviews) == 3


# ============================================================================
# Error Recovery in Phase D Context
# ============================================================================

@pytest.mark.integration
class TestPhaseD_ErrorRecovery:
    """Test error recovery in Phase D context."""

    def test_critic_error_triggers_rollback(self, phase_d_env):
        """Test that critic error doesn't corrupt memory."""
        settings, memory, orchestrator = phase_d_env

        # Get initial state
        initial_directive = memory.load_directive()
        initial_cycle_id = initial_directive.cycle_id

        # Mock architect and failing critic
        def architect_agent(context: CycleContext) -> CycleContext:
            context.proposals = Proposals(
                cycle_id=context.cycle_id,
                proposals=[
                    Proposal(
                        proposal_id="fail_001",
                        description="Will cause critic to fail",
                        novelty_class=NoveltyClass.WILDCAT,
                        expected_impact=ExpectedImpact.VERY_HIGH,
                        rationale="High risk"
                    )
                ]
            )
            return context

        def failing_critic_agent(context: CycleContext) -> CycleContext:
            raise ValueError("Critic rejected all proposals")

        orchestrator.register_agent("architect", architect_agent)
        orchestrator.register_agent("critic", failing_critic_agent)

        # Run cycle (should fail)
        result = orchestrator.run_cycle(cycle_id=10)

        assert result.phase == OrchestratorPhase.ERROR
        assert len(result.errors) > 0

        # Verify memory not corrupted
        current_directive = memory.load_directive()
        assert current_directive.cycle_id == initial_cycle_id

    def test_snapshot_created_on_phase_d_error(self, phase_d_env):
        """Test that snapshot is created when Phase D agent fails."""
        settings, memory, orchestrator = phase_d_env

        # Failing agent
        def failing_agent(context: CycleContext) -> CycleContext:
            raise RuntimeError("Simulated Phase D error")

        orchestrator.register_agent("historian", failing_agent)

        # Run cycle
        result = orchestrator.run_cycle(cycle_id=1)

        assert result.phase == OrchestratorPhase.ERROR

        # Verify snapshot was created
        snapshots = list(settings.snapshots_dir.glob("backup_*"))
        assert len(snapshots) > 0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.integration
class TestPhaseD_Performance:
    """Test performance with Phase D workloads."""

    def test_memory_operations_under_100ms(self, phase_d_env):
        """Test that memory operations meet <100ms requirement."""
        import time
        settings, memory, orchestrator = phase_d_env

        # Test load operation
        start = time.perf_counter()
        memory.load_all_memory()
        load_time = (time.perf_counter() - start) * 1000

        assert load_time < 100, f"Load took {load_time:.2f}ms (should be <100ms)"

        # Test save operation
        directive = memory.load_directive()
        directive.cycle_id += 1

        start = time.perf_counter()
        memory.save_directive(directive)
        save_time = (time.perf_counter() - start) * 1000

        assert save_time < 100, f"Save took {save_time:.2f}ms (should be <100ms)"

    def test_cycle_with_all_agents_completes(self, phase_d_env):
        """Test that full cycle with all agents completes in reasonable time."""
        import time
        settings, memory, orchestrator = phase_d_env

        # Register simple agents
        for phase in ["historian", "director", "architect", "critic", "executor"]:
            orchestrator.register_agent(phase, lambda ctx: ctx)

        start = time.perf_counter()
        result = orchestrator.run_cycle(cycle_id=1)
        duration = time.perf_counter() - start

        assert result.phase == OrchestratorPhase.COMPLETE
        # Should complete quickly for mock agents
        assert duration < 1.0, f"Cycle took {duration:.2f}s"
