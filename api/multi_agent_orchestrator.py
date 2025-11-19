"""
Multi-Agent Orchestrator: Democratic Research Cycle Coordinator
================================================================

Coordinates the full multi-agent research cycle with:
- Supervisor oversight and veto power
- Democratic consensus voting
- Parallel proposal generation
- Multi-critic safety review
- Heterogeneous model routing
- Complete decision audit trail

This orchestrator replaces the single-LLM sequential pipeline with
a true multi-agent governance system.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.registry import AgentRegistry
from agents.director_agent import DirectorAgent
from agents.architect_agent import ArchitectAgent
from agents.critic_agent import CriticAgent
from agents.critic_secondary import CriticSecondaryAgent
from agents.historian_agent import HistorianAgent
from agents.executor_agent import ExecutorAgent
from agents.explorer import ExplorerAgent
from agents.parameter_scientist import ParameterScientistAgent
from agents.supervisor import SupervisorAgent

from consensus.voting import VotingSystem, VoteResult
from consensus.conflict_resolution import ConflictResolver, ConflictResolutionStrategy

from llm.router import LLMRouter
from llm.health_monitor import get_health_monitor

from config.loader import get_config_loader
from llm.decision_logger import get_decision_logger, LogEventType
from tools.dev_logger import get_dev_logger
from tools.drift_detector import get_drift_detector
from tools.mode_collapse_engine import get_mode_collapse_engine

# Import training executor for autonomous execution
try:
    from api.training_executor import get_training_executor, TrainingExecutionError, JobStatus
    TRAINING_EXECUTOR_AVAILABLE = True
except ImportError:
    logger.warning("Training executor not available - experiments will not be executed")
    TRAINING_EXECUTOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Multi-agent research cycle orchestrator.

    Coordinates all 9 agents in a democratic decision-making process:
    1. Supervisor pre-check (system validation)
    2. Historian updates history from previous cycle
    3. Director sets strategic direction
    4. Parallel proposal generation (Architect, Explorer, Parameter Scientist)
    5. Multi-critic review (Primary Critic, Secondary Critic)
    6. Democratic voting on proposals
    7. Conflict resolution if needed
    8. Supervisor final validation (can veto)
    9. Executor prepares approved experiments
    10. Decision logging for audit trail
    """

    def __init__(
        self,
        memory_path: str = "/workspace/arc/memory",
        offline_mode: bool = False,
        config_profile: str = "development"
    ):
        """
        Initialize multi-agent orchestrator.

        Args:
            memory_path: Path to shared memory directory
            offline_mode: Use MockLLMClient for all agents
            config_profile: Configuration profile (development, staging, production)
        """
        self.memory_path = Path(memory_path)
        self.offline_mode = offline_mode
        self.config_loader = get_config_loader()
        self.config_profile = config_profile

        # Ensure memory directories exist
        self.memory_path.mkdir(parents=True, exist_ok=True)
        (self.memory_path / "decisions").mkdir(exist_ok=True)

        # Initialize LLM router
        self.llm_router = LLMRouter(offline_mode=offline_mode)

        # Initialize agent registry
        self.registry = AgentRegistry()

        # Initialize voting system
        self.voting_system = VotingSystem(
            consensus_threshold=0.66,
            min_votes_required=2,
            enable_confidence_weighting=True
        )

        # Initialize conflict resolver
        self.conflict_resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.CONSERVATIVE,
            supervisor_override_threshold=0.9
        )

        # Initialize decision logger
        log_dir = str(self.memory_path / "logs")
        self.decision_logger = get_decision_logger(log_dir=log_dir)

        # Initialize FDA development logger
        self.dev_logger = get_dev_logger()

        # Initialize drift detector
        self.drift_detector = get_drift_detector()

        # Initialize mode collapse engine
        self.mode_collapse_engine = get_mode_collapse_engine()

        # Initialize training executor (for autonomous operation)
        self.training_executor = None
        if TRAINING_EXECUTOR_AVAILABLE and not offline_mode:
            try:
                self.training_executor = get_training_executor(
                    memory_path=str(self.memory_path),
                    poll_interval=10,
                    max_concurrent_jobs=3
                )
                logger.info("Training executor initialized - autonomous execution enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize training executor: {e}")

        # Initialize agents
        self._initialize_agents()

        logger.info(f"MultiAgentOrchestrator initialized (offline={offline_mode})")

    def _initialize_agents(self):
        """Initialize and register all 9 agents."""
        logger.info("Initializing agents...")

        # Strategic agent (using local DeepSeek R1 instead of Claude)
        self.director = DirectorAgent(
            agent_id="director_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=2.0,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.director)
        self.director.activate()

        # Proposal generation agents
        self.architect = ArchitectAgent(
            agent_id="architect_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=1.5,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.architect)
        self.architect.activate()

        self.explorer = ExplorerAgent(
            agent_id="explorer_001",
            model="qwen2.5-32b",
            llm_router=self.llm_router,
            voting_weight=1.2,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.explorer)
        self.explorer.activate()

        self.parameter_scientist = ParameterScientistAgent(
            agent_id="parameter_scientist_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=1.5,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.parameter_scientist)
        self.parameter_scientist.activate()

        # Safety review agents
        self.primary_critic = CriticAgent(
            agent_id="critic_001",
            model="qwen2.5-32b",
            llm_router=self.llm_router,
            voting_weight=2.0,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.primary_critic)
        self.primary_critic.activate()

        self.secondary_critic = CriticSecondaryAgent(
            agent_id="critic_secondary_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=1.8,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.secondary_critic)
        self.secondary_critic.activate()

        # Supervisor (veto power)
        self.supervisor = SupervisorAgent(
            agent_id="supervisor_001",
            model="llama-3-8b-local",
            llm_router=self.llm_router,
            voting_weight=3.0,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.supervisor)
        self.supervisor.activate()

        # Memory and execution agents
        self.historian = HistorianAgent(
            agent_id="historian_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=1.0,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.historian)
        self.historian.activate()

        self.executor = ExecutorAgent(
            agent_id="executor_001",
            model="deepseek-r1",
            llm_router=self.llm_router,
            voting_weight=1.0,
            memory_path=str(self.memory_path)
        )
        self.registry.register(self.executor)
        self.executor.activate()

        logger.info(f"Initialized {len(self.registry)} agents")

    def run_research_cycle(self, cycle_id: int) -> Dict[str, Any]:
        """
        Execute a complete multi-agent research cycle.

        Args:
            cycle_id: Current cycle number

        Returns:
            Cycle results with decisions and metrics
        """
        logger.info(f"=== Starting Multi-Agent Research Cycle {cycle_id} ===")
        cycle_start = time.time()

        # Log cycle start
        self.decision_logger.log_cycle_event(
            cycle_id=cycle_id,
            event_type=LogEventType.CYCLE_STARTED,
            metadata={"offline_mode": self.offline_mode}
        )

        results = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "decisions": {},
            "metrics": {}
        }

        try:
            # Stage 1: Supervisor Pre-Check
            logger.info("Stage 1: Supervisor Pre-Check")
            precheck_result = self._supervisor_precheck(cycle_id)
            results["stages"]["precheck"] = precheck_result
            if not precheck_result["passed"]:
                logger.warning(f"Supervisor pre-check failed: {precheck_result['reason']}")
                return results

            # Stage 2: Historian Updates
            logger.info("Stage 2: Historian Updates History")
            historian_result = self._historian_update(cycle_id)
            results["stages"]["historian"] = historian_result

            # Stage 3: Director Strategic Planning
            logger.info("Stage 3: Director Strategic Planning")
            directive_result = self._director_planning(cycle_id)
            results["stages"]["director"] = directive_result

            # Stage 4: Parallel Proposal Generation
            logger.info("Stage 4: Parallel Proposal Generation")
            proposals_result = self._generate_proposals(cycle_id)
            results["stages"]["proposals"] = proposals_result

            if not proposals_result.get("proposals"):
                logger.warning("No proposals generated, ending cycle")
                return results

            # Stage 5: Multi-Critic Review
            logger.info("Stage 5: Multi-Critic Safety Review")
            reviews_result = self._critic_review(cycle_id, proposals_result["proposals"])
            results["stages"]["reviews"] = reviews_result

            # Stage 6: Democratic Voting
            logger.info("Stage 6: Democratic Consensus Voting")
            voting_result = self._conduct_voting(cycle_id, proposals_result["proposals"])
            results["stages"]["voting"] = voting_result
            results["decisions"]["voting_outcomes"] = voting_result["vote_results"]

            # Stage 7: Conflict Resolution (if needed)
            logger.info("Stage 7: Conflict Resolution")
            resolution_result = self._resolve_conflicts(cycle_id, voting_result)
            results["stages"]["conflict_resolution"] = resolution_result

            # Stage 8: Supervisor Final Validation
            logger.info("Stage 8: Supervisor Final Validation")
            supervisor_result = self._supervisor_validation(cycle_id, resolution_result)
            results["stages"]["supervisor"] = supervisor_result
            results["decisions"]["supervisor_decisions"] = supervisor_result["decisions"]

            # Stage 9: Executor Preparation
            logger.info("Stage 9: Executor Preparation")
            execution_result = self._executor_preparation(cycle_id, supervisor_result["approved_proposals"])
            results["stages"]["execution"] = execution_result

            # Stage 10: Log Decisions
            logger.info("Stage 10: Decision Logging")
            self._log_cycle_decisions(cycle_id, results)

            # Calculate cycle metrics
            cycle_duration = time.time() - cycle_start
            results["metrics"]["cycle_duration_seconds"] = cycle_duration
            results["metrics"]["total_proposals"] = len(proposals_result.get("proposals", []))
            results["metrics"]["approved_proposals"] = len(supervisor_result.get("approved_proposals", []))
            results["metrics"]["consensus_rate"] = self._calculate_consensus_rate(voting_result)

            logger.info(f"=== Cycle {cycle_id} Complete ({cycle_duration:.2f}s) ===")
            logger.info(f"Proposals: {results['metrics']['total_proposals']}, Approved: {results['metrics']['approved_proposals']}")

            # Log cycle completion
            self.decision_logger.log_cycle_event(
                cycle_id=cycle_id,
                event_type=LogEventType.CYCLE_COMPLETED,
                metadata={
                    "duration_seconds": cycle_duration,
                    "total_proposals": results['metrics']['total_proposals'],
                    "approved_proposals": results['metrics']['approved_proposals'],
                    "consensus_rate": results['metrics']['consensus_rate']
                }
            )

            # FDA Development Logging: Log research cycle
            self._log_cycle_to_fda(cycle_id, results, cycle_duration)

            # Drift Detection: Check for performance and diversity drift
            self._run_drift_detection(cycle_id, results)

            # Snapshot system state for FDA traceability
            self.dev_logger.snapshot_system_state(cycle_id=cycle_id)

            return results

        except Exception as e:
            logger.error(f"Cycle {cycle_id} failed: {e}", exc_info=True)
            results["error"] = str(e)
            results["status"] = "failed"

            # FDA Development Logging: Log risk event for cycle failure
            self.dev_logger.log_risk_event(
                event_type="cycle_crash",
                severity="high",
                description=f"Research cycle {cycle_id} crashed with exception: {str(e)}",
                cycle_id=cycle_id,
                context={"exception_type": type(e).__name__, "traceback": str(e)}
            )

            return results

    def _supervisor_precheck(self, cycle_id: int) -> Dict[str, Any]:
        """Supervisor validates system state before cycle."""
        supervisor = self.registry.get_agent("supervisor_001")

        # Check system state
        system_state_path = self.memory_path / "system_state.json"
        if not system_state_path.exists():
            return {"passed": False, "reason": "system_state.json missing"}

        # Check all agents are healthy
        health_report = self.registry.get_health_report()
        unhealthy = health_report["unhealthy_agents"]

        if unhealthy > 0:
            logger.warning(f"{unhealthy} unhealthy agents detected")

        return {
            "passed": True,
            "healthy_agents": health_report["healthy_agents"],
            "total_agents": health_report["total_agents"],
            "timestamp": datetime.now().isoformat()
        }

    def _historian_update(self, cycle_id: int) -> Dict[str, Any]:
        """Historian updates history from previous cycle results."""
        historian = self.registry.get_agent("historian_001")

        # For now, historian reads existing history
        # In a real cycle, it would incorporate results from previous cycle
        try:
            result = historian.process({"cycle_id": cycle_id})

            return {
                "status": "updated",
                "timestamp": datetime.now().isoformat()
            }
        except TimeoutError as e:
            # FDA Development Logging: Log timeout as risk event
            self.dev_logger.log_risk_event(
                event_type="llm_timeout",
                severity="medium",
                description=f"Historian timed out during cycle {cycle_id}",
                cycle_id=cycle_id,
                context={"agent": "historian", "error": str(e)}
            )
            logger.warning(f"Historian timeout in cycle {cycle_id}: {e}")
            return {
                "status": "timeout",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _director_planning(self, cycle_id: int) -> Dict[str, Any]:
        """Director sets strategic direction using adaptive strategy."""
        director = self.registry.get_agent("director_001")
        historian = self.registry.get_agent("historian_001")

        # Compute adaptive strategy based on performance trends
        strategy = director.compute_adaptive_strategy(
            historian=historian,
            stagnation_threshold=0.01,
            regression_threshold=-0.05,
            window=5
        )

        # Update directive with computed strategy
        directive = {
            "mode": strategy["mode"],
            "novelty_budget": strategy["novelty_budget"],
            "objective": strategy.get("objective", "Continue research"),
            "focus_areas": strategy.get("focus_areas", []),
            "reasoning": strategy.get("reasoning", ""),
            "strategy_type": strategy.get("strategy_type", "adaptive")
        }

        # Save directive to file for other agents to read
        directive_path = self.memory_path / "directive.json"
        directive_path.write_text(json.dumps(directive, indent=2))

        logger.info(f"Director adaptive strategy: mode={directive['mode']}, "
                   f"novelty_budget={directive['novelty_budget']}, "
                   f"reasoning={directive['reasoning']}")

        # Log to agent cognition feed for UI visibility
        self._log_agent_decision(
            agent="Director",
            action="strategy_decision",
            message=f"Mode: {directive['mode']} | {directive['reasoning']}",
            metadata={
                "mode": directive['mode'],
                "novelty_budget": directive['novelty_budget'],
                "strategy_type": directive.get("strategy_type", "adaptive")
            }
        )

        return {
            "directive": directive,
            "mode": directive.get("mode", "explore"),
            "novelty_budget": directive.get("novelty_budget", {}),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_proposals(self, cycle_id: int) -> Dict[str, Any]:
        """Parallel proposal generation from multiple agents."""
        # Get proposal generation agents
        architect = self.registry.get_agent("architect_001")
        explorer = self.registry.get_agent("explorer_001")
        param_scientist = self.registry.get_agent("parameter_scientist_001")

        all_proposals = []

        # Architect proposals (always active)
        arch_result = architect.process({"cycle_id": cycle_id})
        all_proposals.extend(arch_result.get("proposals", []))

        # Explorer proposals (always active - full multi-agent orchestrator)
        exp_result = explorer.process({"cycle_id": cycle_id})
        all_proposals.extend(exp_result.get("proposals", []))

        # Parameter Scientist proposals (always active - full multi-agent orchestrator)
        param_result = param_scientist.process({"cycle_id": cycle_id})
        all_proposals.extend(param_result.get("proposals", []))

        logger.info(f"Generated {len(all_proposals)} proposals: "
                   f"{len(arch_result.get('proposals', []))} from Architect, "
                   f"{len(exp_result.get('proposals', []))} from Explorer, "
                   f"{len(param_result.get('proposals', []))} from Parameter Scientist")

        # Mode Collapse Detection: Check for diversity collapse before filtering
        collapse_result = self.mode_collapse_engine.detect_mode_collapse(
            proposals=all_proposals,
            cycle_id=cycle_id
        )

        if collapse_result.collapse_detected:
            logger.warning(
                f"Mode collapse detected: {collapse_result.collapse_type} "
                f"(severity: {collapse_result.severity})"
            )

            # Trigger exploration mode if collapse is severe
            if collapse_result.severity in ["high", "critical"]:
                self.mode_collapse_engine.trigger_exploration_mode(
                    cycle_id=cycle_id,
                    reason=f"{collapse_result.collapse_type}: {collapse_result.recommended_action}"
                )

        # Apply diversity enforcement (mode collapse engine)
        filtered_proposals, diversity_action = self.mode_collapse_engine.enforce_diversity(
            proposals=all_proposals,
            cycle_id=cycle_id
        )

        # Apply additional historical duplicate filter
        filtered_proposals = self._filter_duplicate_proposals(filtered_proposals, cycle_id)
        removed_count = len(all_proposals) - len(filtered_proposals)

        if removed_count > 0:
            logger.warning(f"Removed {removed_count} duplicate/recent proposals to enforce diversity")
            # Log to cognition feed
            self._log_agent_decision(
                agent="Orchestrator",
                action="diversity_filter",
                message=f"Rejected {removed_count} duplicate/recent proposals to maintain diversity",
                metadata={
                    "removed_count": removed_count,
                    "cycle_id": cycle_id,
                    "mode_collapse_detected": collapse_result.collapse_detected,
                    "collapse_type": collapse_result.collapse_type if collapse_result.collapse_detected else None
                }
            )

        return {
            "proposals": filtered_proposals,
            "total_count": len(filtered_proposals),
            "filtered_count": removed_count,
            "mode_collapse_detected": collapse_result.collapse_detected,
            "collapse_details": collapse_result.details if collapse_result.collapse_detected else None,
            "timestamp": datetime.now().isoformat()
        }

    def _critic_review(self, cycle_id: int, proposals: List[Dict]) -> Dict[str, Any]:
        """Multi-critic safety review."""
        critic = self.registry.get_agent("critic_001")
        critic_secondary = self.registry.get_agent("critic_secondary_001")

        # Primary critic review
        primary_review = critic.process({"cycle_id": cycle_id, "proposals": proposals})

        # Secondary critic review (independent)
        secondary_review = critic_secondary.process({"cycle_id": cycle_id, "proposals": proposals})

        return {
            "primary_reviews": primary_review.get("reviews", []),
            "secondary_reviews": secondary_review.get("secondary_reviews", []),
            "timestamp": datetime.now().isoformat()
        }

    def _conduct_voting(self, cycle_id: int, proposals: List[Dict]) -> Dict[str, Any]:
        """Conduct democratic voting on all proposals."""
        vote_results = []

        for proposal in proposals:
            # Collect votes from all active agents
            votes = []
            proposal_id = proposal.get("experiment_id", f"proposal_{cycle_id}")

            for agent in self.registry.get_active_agents():
                vote = agent.vote_on_proposal(proposal)

                # Log individual vote
                self.decision_logger.log_vote(
                    cycle_id=cycle_id,
                    proposal_id=proposal_id,
                    agent_id=agent.agent_id,
                    agent_role=agent.role,
                    voting_weight=agent.voting_weight,
                    decision=vote["decision"],
                    confidence=vote["confidence"],
                    reasoning=vote["reasoning"],
                    constraints_checked=vote.get("constraints_checked", []),
                    metadata={"proposal_type": proposal.get("type", "unknown")}
                )

                votes.append({
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "decision": vote["decision"],
                    "confidence": vote["confidence"],
                    "voting_weight": agent.voting_weight,
                    "reasoning": vote["reasoning"]
                })

            # Conduct vote
            vote_result = self.voting_system.conduct_vote(proposal, votes)
            vote_results.append(vote_result)

            # Log consensus result
            vote_distribution = {}
            for vote in votes:
                decision = vote["decision"]
                vote_distribution[decision] = vote_distribution.get(decision, 0) + 1

            self.decision_logger.log_consensus(
                cycle_id=cycle_id,
                proposal_id=proposal_id,
                total_votes=vote_result.total_votes,
                weighted_score=vote_result.weighted_score,
                consensus_reached=vote_result.consensus_reached,
                final_decision=vote_result.decision.value,
                confidence=vote_result.confidence,
                vote_distribution=vote_distribution,
                participating_agents=[v["agent_id"] for v in votes],
                metadata={"proposal_type": proposal.get("type", "unknown")}
            )

            # Also log to legacy format for backward compatibility
            self._log_vote(cycle_id, proposal, vote_result)

        return {
            "vote_results": [vr.to_dict() for vr in vote_results],
            "timestamp": datetime.now().isoformat()
        }

    def _resolve_conflicts(self, cycle_id: int, voting_result: Dict) -> Dict[str, Any]:
        """Resolve voting conflicts using conflict resolution strategies."""
        vote_results = voting_result["vote_results"]
        resolutions = []

        for vote_dict in vote_results:
            # Reconstruct VoteResult
            from consensus.voting import VoteDecision
            vote_result = VoteResult(
                proposal_id=vote_dict["proposal_id"],
                total_votes=vote_dict["total_votes"],
                weighted_score=vote_dict["weighted_score"],
                consensus_reached=vote_dict["consensus_reached"],
                decision=VoteDecision(vote_dict["decision"]),
                votes=vote_dict["votes"],
                confidence=vote_dict["confidence"],
                metadata=vote_dict["metadata"]
            )

            # Resolve if no consensus
            if not vote_result.consensus_reached:
                # Detect controversy
                controversy = self.conflict_resolver.detect_controversy(vote_result)

                resolution = self.conflict_resolver.resolve_conflict(vote_result)
                resolutions.append(resolution)

                # Log conflict with structured logger
                self.decision_logger.log_conflict(
                    cycle_id=cycle_id,
                    proposal_id=vote_result.proposal_id,
                    conflict_type=controversy.get("reason", "low_consensus"),
                    entropy=controversy.get("entropy", 0.0),
                    resolution_strategy=resolution.get("resolution_strategy", "unknown"),
                    original_decision=vote_result.decision.value,
                    final_decision=resolution.get("final_decision", "reject"),
                    override_applied=resolution.get("override_applied", False),
                    reasoning=resolution.get("reasoning", ""),
                    metadata={
                        "controversial": controversy.get("controversial", False),
                        "vote_distribution": controversy.get("decision_distribution", {})
                    }
                )

                # Also log to legacy format
                self._log_conflict_resolution(cycle_id, vote_result.proposal_id, resolution)

        return {
            "resolutions": resolutions,
            "conflicts_resolved": len(resolutions),
            "timestamp": datetime.now().isoformat()
        }

    def _supervisor_validation(self, cycle_id: int, resolution_result: Dict) -> Dict[str, Any]:
        """Supervisor final validation with veto power."""
        supervisor = self.registry.get_agent("supervisor_001")

        # Supervisor reviews all decisions
        supervisor_result = supervisor.process({"cycle_id": cycle_id})

        decisions = supervisor_result.get("decisions", [])
        approved_proposals = [
            d["experiment_id"] for d in decisions
            if d["decision"] == "approve"
        ]

        # Log supervisor decisions with structured logger
        for decision in decisions:
            proposal_id = decision.get("experiment_id", f"proposal_{cycle_id}")

            # Extract supervisor decision details
            supervisor_decision = decision.get("decision", "reject")
            consensus_decision = decision.get("consensus_decision", "unknown")
            override_consensus = decision.get("override_consensus", False)

            self.decision_logger.log_supervisor_decision(
                cycle_id=cycle_id,
                proposal_id=proposal_id,
                supervisor_decision=supervisor_decision,
                risk_assessment=decision.get("risk_level", "unknown"),
                consensus_decision=consensus_decision,
                override_consensus=override_consensus,
                confidence=decision.get("confidence", 0.5),
                reasoning=decision.get("reasoning", ""),
                constraints_violated=decision.get("constraints_violated", []),
                safety_concerns=decision.get("safety_concerns", []),
                metadata={
                    "proposal_type": decision.get("type", "unknown"),
                    "supervisor_agent": "supervisor_001"
                }
            )

            # Also log to legacy format
            self._log_supervisor_decision(cycle_id, decision)

            # Log to cognition feed for UI visibility
            if supervisor_decision == "reject" or override_consensus:
                action_type = "veto_proposal" if override_consensus else "reject_proposal"
                message = f"{action_type.upper()}: {proposal_id} | {decision.get('reasoning', 'No reason provided')}"
                self._log_agent_decision(
                    agent="Supervisor",
                    action=action_type,
                    message=message,
                    metadata={
                        "proposal_id": proposal_id,
                        "risk_level": decision.get("risk_level", "unknown"),
                        "override_consensus": override_consensus,
                        "constraints_violated": decision.get("constraints_violated", [])
                    }
                )

                # FDA Development Logging: Log vetoes as risk events
                if override_consensus:
                    self.dev_logger.log_risk_event(
                        event_type="supervisor_veto",
                        severity="low",
                        description=f"Supervisor vetoed proposal {proposal_id}: {decision.get('reasoning', 'No reason')}",
                        cycle_id=cycle_id,
                        context={
                            "proposal_id": proposal_id,
                            "risk_level": decision.get("risk_level", "unknown"),
                            "constraints_violated": decision.get("constraints_violated", [])
                        }
                    )

        return {
            "decisions": decisions,
            "approved_proposals": approved_proposals,
            "override_count": sum(1 for d in decisions if d.get("override_consensus", False)),
            "timestamp": datetime.now().isoformat()
        }

    def _executor_preparation(
        self,
        cycle_id: int,
        approved_proposals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executor prepares and submits approved experiments for execution.

        Args:
            cycle_id: Current cycle ID
            approved_proposals: List of approved proposal dicts

        Returns:
            Dict with execution status and submitted jobs
        """
        if not approved_proposals:
            return {"status": "no_approved_proposals", "executions": []}

        executor = self.registry.get_agent("executor_001")

        # If training executor available, submit jobs
        if self.training_executor:
            try:
                logger.info(f"Submitting {len(approved_proposals)} approved proposals for training")

                submitted_jobs = self.training_executor.submit_batch(
                    proposals=approved_proposals,
                    requires_approval=False  # Already approved by multi-agent consensus
                )

                return {
                    "status": "submitted",
                    "approved_count": len(approved_proposals),
                    "submitted_count": len(submitted_jobs),
                    "job_ids": [job.experiment_id for job in submitted_jobs],
                    "timestamp": datetime.now().isoformat()
                }

            except TrainingExecutionError as e:
                logger.error(f"Failed to submit training jobs: {e}")
                return {
                    "status": "submission_failed",
                    "error": str(e),
                    "approved_count": len(approved_proposals),
                    "submitted_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
        else:
            # Offline mode or training executor not available
            logger.info(f"Training executor not available - logging {len(approved_proposals)} approved proposals")
            return {
                "status": "prepared_offline",
                "approved_count": len(approved_proposals),
                "approved_experiments": [p.get("experiment_id") for p in approved_proposals],
                "timestamp": datetime.now().isoformat()
            }

    def _log_vote(self, cycle_id: int, proposal: Dict, vote_result: VoteResult):
        """Log voting result to decisions/voting_history.jsonl"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            "proposal_id": proposal.get("experiment_id", "unknown"),
            **vote_result.to_dict()
        }

        log_path = self.memory_path / "decisions" / "voting_history.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _log_conflict_resolution(self, cycle_id: int, proposal_id: str, resolution: Dict):
        """Log conflict resolution to decisions/conflict_resolution.jsonl"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            "proposal_id": proposal_id,
            **resolution
        }

        log_path = self.memory_path / "decisions" / "conflict_resolution.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _log_supervisor_decision(self, cycle_id: int, decision: Dict):
        """Log supervisor decision to decisions/supervisor_decisions.jsonl"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            **decision
        }

        log_path = self.memory_path / "decisions" / "supervisor_decisions.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _log_cycle_decisions(self, cycle_id: int, results: Dict):
        """Log complete cycle decisions."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            "summary": {
                "total_proposals": results["metrics"]["total_proposals"],
                "approved_proposals": results["metrics"]["approved_proposals"],
                "consensus_rate": results["metrics"]["consensus_rate"],
                "cycle_duration": results["metrics"]["cycle_duration_seconds"]
            }
        }

        log_path = self.memory_path / "decisions" / "cycle_summary.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _calculate_consensus_rate(self, voting_result: Dict) -> float:
        """Calculate percentage of votes that reached consensus."""
        vote_results = voting_result["vote_results"]
        if not vote_results:
            return 0.0

        consensus_count = sum(1 for v in vote_results if v["consensus_reached"])
        return consensus_count / len(vote_results)

    def _filter_duplicate_proposals(
        self,
        proposals: List[Dict],
        cycle_id: int,
        lookback_cycles: int = 10
    ) -> List[Dict]:
        """
        Filter out duplicate or recently-run experiments to enforce diversity.

        Args:
            proposals: List of proposed experiments
            cycle_id: Current cycle ID
            lookback_cycles: How many recent cycles to check for duplicates

        Returns:
            Filtered list of unique proposals
        """
        # Load training history to check for duplicates
        history = self.read_memory("training_history.json") or {}
        recent_experiments = history.get("recent_experiments", [])

        # Extract config signatures from recent experiments
        recent_signatures = set()
        for exp in recent_experiments[-lookback_cycles * 3:]:  # ~3 experiments per cycle
            config = exp.get("config", {})
            signature = self._compute_config_signature(config)
            recent_signatures.add(signature)

        # Filter proposals
        unique_proposals = []
        seen_signatures = set()

        for proposal in proposals:
            config = proposal.get("config", {})
            signature = self._compute_config_signature(config)

            # Check if duplicate within current batch
            if signature in seen_signatures:
                logger.debug(f"Skipping duplicate proposal in current batch: {proposal.get('experiment_id')}")
                continue

            # Check if recently run
            if signature in recent_signatures:
                logger.debug(f"Skipping recently-run proposal: {proposal.get('experiment_id')}")
                continue

            # Unique proposal
            unique_proposals.append(proposal)
            seen_signatures.add(signature)

        return unique_proposals

    def _compute_config_signature(self, config: Dict[str, Any]) -> str:
        """
        Compute a signature hash for a config to detect duplicates.

        Args:
            config: Experiment configuration

        Returns:
            Signature string
        """
        # Extract key parameters that define experiment uniqueness
        key_params = {
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "data": config.get("data", {})
        }

        # Sort and serialize for consistent hashing
        import hashlib
        signature_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()

    def read_memory(self, filename: str) -> Optional[Dict]:
        """Read a JSON file from memory directory."""
        path = self.memory_path / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                return None
        return None

    def _log_cycle_to_fda(self, cycle_id: int, results: Dict, cycle_duration: float) -> None:
        """
        Log research cycle to FDA development logs.

        Args:
            cycle_id: Cycle ID
            results: Complete cycle results
            cycle_duration: Cycle duration in seconds
        """
        # Extract agent summaries
        agents_involved = []
        for stage_name, stage_data in results.get("stages", {}).items():
            if stage_name == "proposals":
                agents_involved.extend(["Architect", "Explorer", "ParameterScientist"])
            elif stage_name == "reviews":
                agents_involved.extend(["PrimaryCritic", "SecondaryCritic"])
            elif stage_name == "supervisor":
                agents_involved.append("Supervisor")

        # Extract proposals and decisions
        proposals_count = results.get("metrics", {}).get("total_proposals", 0)
        approved_count = results.get("metrics", {}).get("approved_proposals", 0)

        # Extract supervisor decisions for warnings
        supervisor_stage = results.get("stages", {}).get("supervisor", {})
        vetoes = supervisor_stage.get("override_count", 0)

        # Extract voting conflicts
        conflict_stage = results.get("stages", {}).get("conflict_resolution", {})
        conflicts = conflict_stage.get("conflicts_resolved", 0)

        # Determine if cycle had failures/warnings
        failures = []
        warnings = []

        if vetoes > 0:
            warnings.append(f"Supervisor vetoed {vetoes} proposals")

        if conflicts > 0:
            warnings.append(f"Resolved {conflicts} voting conflicts")

        precheck_stage = results.get("stages", {}).get("precheck", {})
        if not precheck_stage.get("passed", True):
            failures.append(f"Pre-check failed: {precheck_stage.get('reason', 'unknown')}")

        # Log to FDA dev logger
        self.dev_logger.log_cycle(
            cycle_id=cycle_id,
            agents_involved=agents_involved,
            proposals_generated=proposals_count,
            proposals_approved=approved_count,
            reasoning_summary=self._generate_cycle_reasoning_summary(results),
            decisions_made=[],  # Detailed decisions logged separately
            failures=failures,
            warnings=warnings,
            duration_seconds=cycle_duration
        )

    def _generate_cycle_reasoning_summary(self, results: Dict) -> str:
        """Generate human-readable reasoning summary from cycle results."""
        summary_parts = []

        # Director strategy
        director_stage = results.get("stages", {}).get("director", {})
        directive = director_stage.get("directive", {})
        if directive:
            mode = directive.get("mode", "unknown")
            reasoning = directive.get("reasoning", "")
            summary_parts.append(f"Director Strategy: {mode} mode - {reasoning}")

        # Proposals
        metrics = results.get("metrics", {})
        proposals = metrics.get("total_proposals", 0)
        approved = metrics.get("approved_proposals", 0)
        summary_parts.append(f"Proposals: {proposals} generated, {approved} approved")

        # Consensus
        consensus_rate = metrics.get("consensus_rate", 0)
        summary_parts.append(f"Consensus Rate: {consensus_rate:.1%}")

        return " | ".join(summary_parts)

    def _run_drift_detection(self, cycle_id: int, results: Dict) -> None:
        """
        Run drift detection after cycle completion.

        Args:
            cycle_id: Cycle ID
            results: Complete cycle results
        """
        try:
            # Performance drift detection (stub - need actual AUC from results)
            # In a real implementation, would extract AUC from completed experiments
            # For now, use consensus_rate as a proxy metric
            consensus_rate = results.get("metrics", {}).get("consensus_rate", 0.0)

            # Diversity drift detection
            proposals_stage = results.get("stages", {}).get("proposals", {})
            proposals = proposals_stage.get("proposals", [])

            if proposals:
                # Check diversity drift
                diversity_result = self.drift_detector.detect_diversity_drift(
                    proposal_configs=proposals,
                    cycle_id=cycle_id
                )

                logger.info(f"Diversity drift check: detected={diversity_result.drift_detected}, "
                          f"score={diversity_result.drift_score:.2f}, "
                          f"action={diversity_result.recommended_action}")

                # Store drift result in cycle results
                results["drift_detection"] = {
                    "diversity": {
                        "detected": diversity_result.drift_detected,
                        "score": diversity_result.drift_score,
                        "severity": diversity_result.severity,
                        "action": diversity_result.recommended_action,
                        "details": diversity_result.details
                    }
                }

                # TODO: Trigger Director mode change if high drift
                # if diversity_result.drift_detected and diversity_result.severity == "high":
                #     self._trigger_exploration_mode(cycle_id)

        except Exception as e:
            logger.warning(f"Drift detection failed for cycle {cycle_id}: {e}")

    def _log_agent_decision(
        self,
        agent: str,
        action: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log agent decision to cognition feed for UI visibility.

        Args:
            agent: Agent name (Director, Supervisor, etc.)
            action: Action type (strategy_decision, veto_proposal, etc.)
            message: Human-readable message
            metadata: Additional context
        """
        agent_decisions_path = self.memory_path / "agent_decisions.json"

        # Load existing decisions
        if agent_decisions_path.exists():
            try:
                with open(agent_decisions_path, 'r') as f:
                    data = json.load(f)
            except:
                data = {"decisions": []}
        else:
            data = {"decisions": []}

        # Add new decision
        decision = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "message": message,
            "metadata": metadata or {}
        }
        data["decisions"].append(decision)

        # Keep only last 500 decisions to prevent file bloat
        data["decisions"] = data["decisions"][-500:]

        # Write back
        with open(agent_decisions_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return self.registry.get_health_report()

    def wait_for_training_completion(
        self,
        experiment_ids: List[str],
        timeout: Optional[int] = 3600
    ) -> Dict[str, JobStatus]:
        """
        Wait for submitted training jobs to complete.

        Args:
            experiment_ids: List of experiment IDs to wait for
            timeout: Maximum seconds to wait (default 1 hour)

        Returns:
            Dict mapping experiment_id to final JobStatus
        """
        if not self.training_executor:
            logger.warning("Training executor not available - cannot wait for completion")
            return {}

        logger.info(f"Waiting for {len(experiment_ids)} training jobs to complete...")
        completion_status = self.training_executor.wait_for_completion(
            experiment_ids=experiment_ids,
            timeout=timeout
        )

        completed_count = sum(1 for status in completion_status.values() if status == JobStatus.COMPLETED)
        failed_count = sum(1 for status in completion_status.values() if status == JobStatus.FAILED)

        logger.info(f"Training completion: {completed_count} succeeded, {failed_count} failed")

        return completion_status

    def collect_and_integrate_results(
        self,
        experiment_ids: List[str],
        cycle_id: int
    ) -> Dict[str, Any]:
        """
        Collect experiment results and feed them to Historian.

        This is the key feedback loop for autonomous learning.

        Args:
            experiment_ids: List of completed experiment IDs
            cycle_id: Current cycle ID

        Returns:
            Integration summary from Historian
        """
        if not self.training_executor:
            logger.warning("Training executor not available - cannot collect results")
            return {"status": "executor_unavailable"}

        # Collect results from all experiments
        logger.info(f"Collecting results from {len(experiment_ids)} experiments...")
        experiment_results = self.training_executor.collect_batch_results(experiment_ids)

        # Feed results to Historian for learning
        historian = self.registry.get_agent("historian_001")
        if historian:
            logger.info(f"Integrating results into history for learning...")
            integration_summary = historian.integrate_experiment_results(
                experiment_results=experiment_results,
                cycle_id=cycle_id
            )

            logger.info(f"Results integrated: {integration_summary['successful']} successful, "
                       f"{integration_summary['failed']} failed")

            return integration_summary
        else:
            logger.error("Historian agent not found - cannot integrate results")
            return {"status": "historian_unavailable", "results": experiment_results}

    def run_autonomous_cycle(
        self,
        cycle_id: int,
        wait_for_completion: bool = True,
        timeout: Optional[int] = 3600
    ) -> Dict[str, Any]:
        """
        Run a complete autonomous research cycle with training execution and results feedback.

        This method enables true autonomous operation:
        1. Run multi-agent research cycle
        2. Submit approved proposals for training
        3. Wait for training completion
        4. Collect results
        5. Feed results to Historian for next cycle

        Args:
            cycle_id: Current cycle ID
            wait_for_completion: Whether to wait for training jobs to complete
            timeout: Maximum seconds to wait for training

        Returns:
            Complete cycle results including training outcomes
        """
        # Run multi-agent research cycle
        cycle_results = self.run_research_cycle(cycle_id)

        # Check if experiments were submitted
        execution_stage = cycle_results.get("stages", {}).get("execution", {})
        job_ids = execution_stage.get("job_ids", [])

        if not job_ids:
            logger.info("No training jobs submitted - cycle complete")
            cycle_results["training_status"] = "no_jobs_submitted"
            return cycle_results

        logger.info(f"Submitted {len(job_ids)} training jobs")

        # Wait for training completion if requested
        if wait_for_completion and self.training_executor:
            completion_status = self.wait_for_training_completion(
                experiment_ids=job_ids,
                timeout=timeout
            )
            cycle_results["training_completion"] = completion_status

            # Collect and integrate results
            completed_ids = [
                exp_id for exp_id, status in completion_status.items()
                if status == JobStatus.COMPLETED
            ]

            if completed_ids:
                integration_summary = self.collect_and_integrate_results(
                    experiment_ids=completed_ids,
                    cycle_id=cycle_id
                )
                cycle_results["results_integration"] = integration_summary
                logger.info(f"Autonomous cycle {cycle_id} complete with results integration")
            else:
                logger.warning("No experiments completed successfully")
                cycle_results["results_integration"] = {"status": "no_completed_experiments"}
        else:
            logger.info("Not waiting for training completion - cycle complete")
            cycle_results["training_status"] = "submitted_not_waiting"

        return cycle_results


def main():
    """
    Run a single multi-agent research cycle.

    Usage:
        python api/multi_agent_orchestrator.py [cycle_id] [--offline]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Research Orchestrator")
    parser.add_argument("cycle_id", type=int, nargs="?", default=1, help="Cycle ID")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (MockLLMClient)")
    parser.add_argument("--profile", default="development", choices=["development", "staging", "production"])

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        offline_mode=args.offline,
        config_profile=args.profile
    )

    # Run cycle
    results = orchestrator.run_research_cycle(args.cycle_id)

    # Print summary
    print("\n" + "="*60)
    print(f"Multi-Agent Research Cycle {args.cycle_id} Complete")
    print("="*60)
    print(f"Status: {results.get('status', 'success')}")
    print(f"Duration: {results.get('metrics', {}).get('cycle_duration_seconds', 0):.2f}s")
    print(f"Proposals Generated: {results.get('metrics', {}).get('total_proposals', 0)}")
    print(f"Proposals Approved: {results.get('metrics', {}).get('approved_proposals', 0)}")
    print(f"Consensus Rate: {results.get('metrics', {}).get('consensus_rate', 0):.1%}")
    print("="*60)

    # Print agent status
    status = orchestrator.get_agent_status()
    print(f"\nAgent Status: {status['healthy_agents']}/{status['total_agents']} healthy")

    return 0 if results.get("status", "success") != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
