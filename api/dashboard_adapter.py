"""
Dashboard Data Adapter: Bridge orchestrator state to dashboard UI
==================================================================

Provides real-time data feeds for Streamlit dashboard:
- Agent registry status and metrics
- Voting patterns and consensus quality
- Supervisor decisions and overrides
- Proposal quality trends
- System health metrics

Replaces mock data with real orchestrator telemetry.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from llm.decision_logger import get_decision_logger
from agents.registry import AgentRegistry


class DashboardAdapter:
    """
    Adapter to provide dashboard-friendly data from orchestrator state.

    Bridges:
    - AgentRegistry → agent_status data
    - DecisionLogger → voting/consensus/supervisor data
    - Memory files → system state
    """

    def __init__(
        self,
        memory_path: str = "/workspace/arc/memory",
        log_dir: Optional[str] = None
    ):
        """
        Initialize dashboard adapter.

        Args:
            memory_path: Path to ARC memory directory
            log_dir: Path to decision logs (default: memory_path/logs)
        """
        self.memory_path = Path(memory_path)
        self.log_dir = Path(log_dir) if log_dir else self.memory_path / "logs"

        # Initialize decision logger
        self.decision_logger = get_decision_logger(str(self.log_dir))

    def get_agent_status(self, registry: AgentRegistry) -> List[Dict[str, Any]]:
        """
        Get agent registry status for dashboard.

        Args:
            registry: AgentRegistry instance

        Returns:
            List of agent status dictionaries
        """
        agents = registry.get_all_agents()
        agent_status = []

        for agent in agents:
            status = {
                "agent_id": agent.agent_id,
                "role": agent.role,
                "model": agent.model,
                "state": agent.state.value if hasattr(agent.state, 'value') else str(agent.state),
                "voting_weight": agent.voting_weight,
                "priority": agent.priority,
                "healthy": agent.state.value in ['active', 'idle'] if hasattr(agent.state, 'value') else True,
                "last_activity": datetime.now().isoformat(),
                "metrics": {
                    "total_tasks": agent.metrics.get("total_tasks", 0),
                    "successful_tasks": agent.metrics.get("successful_tasks", 0),
                    "avg_response_time_ms": agent.metrics.get("avg_response_time_ms", 0.0),
                    "total_votes": agent.metrics.get("total_votes", 0),
                    "vote_agreement_rate": agent.metrics.get("vote_agreement_rate", 0.0)
                }
            }
            agent_status.append(status)

        return agent_status

    def get_supervisor_decisions(
        self,
        limit: int = 100,
        cycle_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get supervisor decisions from logs.

        Args:
            limit: Maximum decisions to return
            cycle_id: Filter by cycle ID

        Returns:
            List of supervisor decision dictionaries
        """
        supervisor_log = self.log_dir / "supervisor.jsonl"

        if not supervisor_log.exists():
            return []

        decisions = self.decision_logger._query_log(
            supervisor_log,
            cycle_id=cycle_id,
            limit=limit
        )

        # Format for dashboard
        formatted = []
        for decision in decisions:
            formatted.append({
                "proposal_id": decision.get("proposal_id", "unknown"),
                "decision": decision.get("supervisor_decision", "unknown"),
                "risk_assessment": decision.get("risk_assessment", "unknown"),
                "override_consensus": decision.get("override_consensus", False),
                "confidence": decision.get("confidence", 0.0),
                "reasoning": decision.get("reasoning", ""),
                "constraints_violated": decision.get("constraints_violated", []),
                "timestamp": decision.get("timestamp", "")
            })

        return formatted

    def get_risk_distribution(self) -> Dict[str, int]:
        """
        Get distribution of risk levels from supervisor decisions.

        Returns:
            Dict mapping risk level to count
        """
        decisions = self.get_supervisor_decisions(limit=1000)

        distribution = defaultdict(int)
        for decision in decisions:
            risk = decision.get("risk_assessment", "unknown")
            distribution[risk] += 1

        return dict(distribution)

    def get_consensus_metrics(
        self,
        cycle_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get consensus quality metrics.

        Args:
            cycle_id: Filter by cycle ID

        Returns:
            Consensus metrics dictionary
        """
        consensus_logs = self.decision_logger.query_consensus(
            cycle_id=cycle_id,
            limit=1000
        )

        if not consensus_logs:
            return {
                "total_votes_conducted": 0,
                "consensus_rate": 0.0,
                "avg_confidence": 0.0,
                "controversial_rate": 0.0,
                "decision_breakdown": {}
            }

        total = len(consensus_logs)
        reached = sum(1 for c in consensus_logs if c.get("consensus_reached", False))

        confidences = [c.get("confidence", 0.0) for c in consensus_logs]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Controversial = low confidence or failed consensus
        controversial = sum(
            1 for c in consensus_logs
            if not c.get("consensus_reached", False) or c.get("confidence", 0.0) < 0.6
        )

        # Decision breakdown
        decision_breakdown = defaultdict(int)
        for c in consensus_logs:
            decision = c.get("final_decision", "unknown")
            decision_breakdown[decision] += 1

        return {
            "total_votes_conducted": total,
            "consensus_rate": reached / total if total > 0 else 0.0,
            "avg_confidence": avg_confidence,
            "controversial_rate": controversial / total if total > 0 else 0.0,
            "decision_breakdown": dict(decision_breakdown)
        }

    def get_voting_patterns(self) -> Dict[str, Dict[str, float]]:
        """
        Get agent voting agreement patterns.

        Returns:
            Dict mapping agent pairs to agreement rates
        """
        votes = self.decision_logger.query_votes(limit=1000)

        if not votes:
            return {}

        # Group votes by proposal
        votes_by_proposal = defaultdict(list)
        for vote in votes:
            proposal_id = vote.get("proposal_id", "unknown")
            votes_by_proposal[proposal_id].append(vote)

        # Calculate pairwise agreement
        agreement_matrix = defaultdict(lambda: defaultdict(lambda: {"agree": 0, "total": 0}))

        for proposal_votes in votes_by_proposal.values():
            # Compare all pairs
            for i, vote1 in enumerate(proposal_votes):
                for vote2 in proposal_votes[i+1:]:
                    agent1 = vote1.get("agent_id", "unknown")
                    agent2 = vote2.get("agent_id", "unknown")

                    # Check if they agreed
                    agreed = vote1.get("decision") == vote2.get("decision")

                    agreement_matrix[agent1][agent2]["total"] += 1
                    agreement_matrix[agent2][agent1]["total"] += 1

                    if agreed:
                        agreement_matrix[agent1][agent2]["agree"] += 1
                        agreement_matrix[agent2][agent1]["agree"] += 1

        # Calculate agreement rates
        voting_patterns = {}
        for agent1, agreements in agreement_matrix.items():
            voting_patterns[agent1] = {}
            for agent2, stats in agreements.items():
                if stats["total"] > 0:
                    voting_patterns[agent1][agent2] = stats["agree"] / stats["total"]

        return voting_patterns

    def get_proposal_quality_trends(
        self,
        limit_cycles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get proposal quality trends over recent cycles.

        Args:
            limit_cycles: Number of recent cycles to analyze

        Returns:
            List of cycle quality metrics
        """
        cycle_log = self.log_dir / "cycles.jsonl"

        if not cycle_log.exists():
            return []

        # Get recent cycle completions
        cycles = self.decision_logger._query_log(cycle_log, limit=limit_cycles * 2)

        # Group by cycle and get completed cycles
        cycles_by_id = defaultdict(list)
        for cycle in cycles:
            cycle_id = cycle.get("cycle_id")
            cycles_by_id[cycle_id].append(cycle)

        trends = []

        for cycle_id in sorted(cycles_by_id.keys())[-limit_cycles:]:
            events = cycles_by_id[cycle_id]

            # Find completed event
            completed = next(
                (e for e in events if e.get("event_type") == "cycle_completed"),
                None
            )

            if not completed:
                continue

            meta = completed.get("metadata", {})

            # Get consensus data for this cycle
            consensus_logs = self.decision_logger.query_consensus(cycle_id=cycle_id)

            # Calculate quality metrics
            if consensus_logs:
                avg_confidence = sum(c.get("confidence", 0.0) for c in consensus_logs) / len(consensus_logs)
                consensus_rate = sum(1 for c in consensus_logs if c.get("consensus_reached", False)) / len(consensus_logs)
            else:
                avg_confidence = 0.0
                consensus_rate = 0.0

            total_proposals = meta.get("total_proposals", 0)
            approved = meta.get("approved_proposals", 0)
            rejected = total_proposals - approved if total_proposals > 0 else 0

            trends.append({
                "cycle_id": cycle_id,
                "avg_proposal_quality": avg_confidence,  # Use consensus confidence as proxy
                "consensus_score": consensus_rate,
                "proposals_approved": approved,
                "proposals_rejected": rejected,
                "timestamp": completed.get("timestamp", "")
            })

        return trends

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.

        Returns:
            System health dictionary
        """
        # Read system state if it exists
        system_state_file = self.memory_path / "system_state.json"

        if system_state_file.exists():
            with open(system_state_file, 'r') as f:
                system_state = json.load(f)
        else:
            system_state = {}

        # Get latest cycle info
        cycle_log = self.log_dir / "cycles.jsonl"
        latest_cycle = None

        if cycle_log.exists():
            cycles = self.decision_logger._query_log(cycle_log, limit=5)
            if cycles:
                latest_cycle = max(cycles, key=lambda c: c.get("timestamp", ""))

        return {
            "mode": system_state.get("current_mode", "UNKNOWN"),
            "safety_status": system_state.get("safety_status", "UNKNOWN"),
            "current_cycle": system_state.get("current_cycle", 0),
            "last_cycle_time": latest_cycle.get("timestamp", "") if latest_cycle else "",
            "system_health": system_state.get("system_health", "unknown"),
            "active_constraints": system_state.get("active_constraints", [])
        }

    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent system activity across all logs.

        Args:
            limit: Number of recent events

        Returns:
            List of recent activity events
        """
        activity = []

        # Collect from all log files
        log_files = {
            "vote": self.log_dir / "votes.jsonl",
            "consensus": self.log_dir / "consensus.jsonl",
            "conflict": self.log_dir / "conflicts.jsonl",
            "supervisor": self.log_dir / "supervisor.jsonl",
            "cycle": self.log_dir / "cycles.jsonl"
        }

        for event_type, log_file in log_files.items():
            if not log_file.exists():
                continue

            entries = self.decision_logger._query_log(log_file, limit=limit)

            for entry in entries:
                activity.append({
                    "type": event_type,
                    "timestamp": entry.get("timestamp", ""),
                    "cycle_id": entry.get("cycle_id", 0),
                    "event": entry.get("event_type", "unknown"),
                    "summary": self._format_activity_summary(event_type, entry)
                })

        # Sort by timestamp (most recent first)
        activity.sort(key=lambda x: x["timestamp"], reverse=True)

        return activity[:limit]

    def _format_activity_summary(self, event_type: str, entry: Dict[str, Any]) -> str:
        """Format activity entry for display"""
        if event_type == "vote":
            agent = entry.get("agent_role", "unknown")
            decision = entry.get("decision", "unknown")
            return f"{agent} voted {decision}"

        elif event_type == "consensus":
            decision = entry.get("final_decision", "unknown")
            reached = entry.get("consensus_reached", False)
            return f"Consensus {'reached' if reached else 'failed'}: {decision}"

        elif event_type == "supervisor":
            decision = entry.get("supervisor_decision", "unknown")
            override = entry.get("override_consensus", False)
            return f"Supervisor {'override' if override else 'approved'}: {decision}"

        elif event_type == "conflict":
            strategy = entry.get("resolution_strategy", "unknown")
            return f"Conflict resolved via {strategy}"

        elif event_type == "cycle":
            event = entry.get("event_type", "unknown")
            return f"Cycle {event}"

        return "Activity"


def get_dashboard_adapter(memory_path: Optional[str] = None) -> DashboardAdapter:
    """
    Get dashboard adapter instance.

    Args:
        memory_path: Path to memory directory

    Returns:
        DashboardAdapter instance
    """
    if memory_path is None:
        memory_path = "/workspace/arc/memory"

    return DashboardAdapter(memory_path=memory_path)
