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

        # Initialize agents
        self._initialize_agents()

        logger.info(f"MultiAgentOrchestrator initialized (offline={offline_mode})")

    def _initialize_agents(self):
        """Initialize and register all 9 agents."""
        logger.info("Initializing agents...")

        # Strategic agent
        self.director = DirectorAgent(
            agent_id="director_001",
            model="claude-sonnet-4.5",
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

            return results

        except Exception as e:
            logger.error(f"Cycle {cycle_id} failed: {e}", exc_info=True)
            results["error"] = str(e)
            results["status"] = "failed"
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
        result = historian.process({"cycle_id": cycle_id})

        return {
            "status": "updated",
            "timestamp": datetime.now().isoformat()
        }

    def _director_planning(self, cycle_id: int) -> Dict[str, Any]:
        """Director sets strategic direction."""
        director = self.registry.get_agent("director_001")

        directive = director.process({"cycle_id": cycle_id})

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

        # Architect proposals
        arch_result = architect.process({"cycle_id": cycle_id})
        all_proposals.extend(arch_result.get("proposals", []))

        # Explorer proposals (if in explore mode)
        directive = json.loads((self.memory_path / "directive.json").read_text())
        if directive.get("mode") in ["explore", "wildcat"]:
            exp_result = explorer.process({"cycle_id": cycle_id})
            all_proposals.extend(exp_result.get("proposals", []))

        # Parameter Scientist proposals (if in exploit mode)
        if directive.get("mode") in ["exploit", "explore"]:
            param_result = param_scientist.process({"cycle_id": cycle_id})
            all_proposals.extend(param_result.get("proposals", []))

        return {
            "proposals": all_proposals,
            "total_count": len(all_proposals),
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

            for agent in self.registry.get_active_agents():
                vote = agent.vote_on_proposal(proposal)
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

            # Log vote
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
                resolution = self.conflict_resolver.resolve_conflict(vote_result)
                resolutions.append(resolution)
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

        # Log supervisor decisions
        for decision in decisions:
            self._log_supervisor_decision(cycle_id, decision)

        return {
            "decisions": decisions,
            "approved_proposals": approved_proposals,
            "override_count": sum(1 for d in decisions if d.get("override_consensus", False)),
            "timestamp": datetime.now().isoformat()
        }

    def _executor_preparation(self, cycle_id: int, approved_proposals: List[str]) -> Dict[str, Any]:
        """Executor prepares approved experiments for execution."""
        if not approved_proposals:
            return {"status": "no_approved_proposals", "executions": []}

        executor = self.registry.get_agent("executor_001")

        # Executor would prepare training jobs here
        # For now, just log what would be executed
        return {
            "status": "prepared",
            "approved_count": len(approved_proposals),
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

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return self.registry.get_health_report()


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
