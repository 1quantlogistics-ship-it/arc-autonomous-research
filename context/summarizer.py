"""
Cross-Cycle Summarizer for ARC Autonomous Research.

Phase G Task 2.2: Intelligent summarization for multi-agent context sharing.

Two summary types:
1. Executive Summary: Strategic view for Director (200K context capacity)
   - Spans 50+ cycles, focuses on high-level patterns
   - Accuracy trajectory, successful patterns, suggested directions

2. Working Summary: Detailed view for DeepSeek agents (64K context)
   - Last 10 cycles, more granular details
   - Agent-specific filtering based on role

Author: ARC Team (Dev 2)
Created: 2025-11-26
Version: 1.0 (Phase G)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleSummary:
    """
    Summary of a single research cycle.

    Captures key outcomes and learnings from one cycle of experimentation.
    """
    cycle_id: int
    timestamp: datetime
    best_accuracy: float
    best_architecture: str
    key_decisions: List[str] = field(default_factory=list)
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    hypotheses_tested: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    num_experiments: int = 0
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp.isoformat(),
            "best_accuracy": self.best_accuracy,
            "best_architecture": self.best_architecture,
            "key_decisions": self.key_decisions,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "hypotheses_tested": self.hypotheses_tested,
            "metrics": self.metrics,
            "num_experiments": self.num_experiments,
            "token_count": self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleSummary":
        """Create from dictionary."""
        return cls(
            cycle_id=data["cycle_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            best_accuracy=data.get("best_accuracy", 0.0),
            best_architecture=data.get("best_architecture", "unknown"),
            key_decisions=data.get("key_decisions", []),
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            hypotheses_tested=data.get("hypotheses_tested", []),
            metrics=data.get("metrics", {}),
            num_experiments=data.get("num_experiments", 0),
            token_count=data.get("token_count", 0)
        )


@dataclass
class ExecutiveSummary:
    """
    High-level summary for Director agent.

    Designed for 200K context capacity, spanning many cycles.
    Focuses on strategic insights and overall trajectory.
    """
    cycles_covered: List[int] = field(default_factory=list)
    accuracy_trajectory: List[float] = field(default_factory=list)
    best_overall_accuracy: float = 0.0
    best_experiment_id: str = ""
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    suggested_directions: List[str] = field(default_factory=list)
    stagnation_detected: bool = False
    stagnation_cycles: int = 0
    total_experiments: int = 0
    success_rate: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycles_covered": self.cycles_covered,
            "accuracy_trajectory": self.accuracy_trajectory,
            "best_overall_accuracy": self.best_overall_accuracy,
            "best_experiment_id": self.best_experiment_id,
            "successful_patterns": self.successful_patterns,
            "failed_patterns": self.failed_patterns,
            "suggested_directions": self.suggested_directions,
            "stagnation_detected": self.stagnation_detected,
            "stagnation_cycles": self.stagnation_cycles,
            "total_experiments": self.total_experiments,
            "success_rate": self.success_rate,
            "generated_at": self.generated_at.isoformat(),
            "token_count": self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutiveSummary":
        """Create from dictionary."""
        return cls(
            cycles_covered=data.get("cycles_covered", []),
            accuracy_trajectory=data.get("accuracy_trajectory", []),
            best_overall_accuracy=data.get("best_overall_accuracy", 0.0),
            best_experiment_id=data.get("best_experiment_id", ""),
            successful_patterns=data.get("successful_patterns", []),
            failed_patterns=data.get("failed_patterns", []),
            suggested_directions=data.get("suggested_directions", []),
            stagnation_detected=data.get("stagnation_detected", False),
            stagnation_cycles=data.get("stagnation_cycles", 0),
            total_experiments=data.get("total_experiments", 0),
            success_rate=data.get("success_rate", 0.0),
            generated_at=datetime.fromisoformat(data["generated_at"]) if "generated_at" in data else datetime.now(),
            token_count=data.get("token_count", 0)
        )


class CrossCycleSummarizer:
    """
    Summarizer for cross-cycle context management.

    Creates two types of summaries:
    1. Executive Summary: For Director (Claude), high-level strategic view
    2. Working Summary: For DeepSeek agents, detailed recent context

    Summaries are designed to fit within token budgets while
    preserving the most important information.
    """

    # Default token budgets (from Phase G spec)
    EXECUTIVE_MAX_TOKENS = 200000  # For Claude Director
    WORKING_MAX_TOKENS = 64000    # For DeepSeek agents
    DEFAULT_WORKING_TOKENS = 8000  # Default for working summary

    # Agent role mappings for filtering
    AGENT_FOCUS_AREAS = {
        "architect": ["architecture", "model", "layers", "blocks"],
        "critic": ["metrics", "validation", "accuracy", "loss"],
        "explorer": ["search", "exploration", "novel", "experiment"],
        "executor": ["training", "execution", "results", "config"],
        "historian": ["history", "patterns", "trends", "memory"],
        "director": ["strategy", "direction", "decisions", "overall"]
    }

    def __init__(
        self,
        storage_path: Optional[str] = None,
        tokens_per_word: float = 1.3  # Approximation
    ):
        """
        Initialize CrossCycleSummarizer.

        Args:
            storage_path: Path to store summaries (default: memory/summaries/)
            tokens_per_word: Approximate tokens per word for estimation
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("memory/summaries")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tokens_per_word = tokens_per_word
        self._cycle_summaries: Dict[int, CycleSummary] = {}
        self._load_existing_summaries()

        logger.info(f"CrossCycleSummarizer initialized, storage: {self.storage_path}")

    def _load_existing_summaries(self) -> None:
        """Load existing cycle summaries from storage."""
        summary_file = self.storage_path / "cycle_summaries.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                for cycle_data in data.get("cycles", []):
                    summary = CycleSummary.from_dict(cycle_data)
                    self._cycle_summaries[summary.cycle_id] = summary
                logger.info(f"Loaded {len(self._cycle_summaries)} existing cycle summaries")
            except Exception as e:
                logger.warning(f"Failed to load existing summaries: {e}")

    def _save_cycle_summaries(self) -> None:
        """Save cycle summaries to storage."""
        summary_file = self.storage_path / "cycle_summaries.json"
        data = {
            "cycles": [s.to_dict() for s in self._cycle_summaries.values()],
            "updated_at": datetime.now().isoformat()
        }
        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        word_count = len(text.split())
        return int(word_count * self.tokens_per_word)

    def summarize_cycle(
        self,
        cycle_id: int,
        cycle_data: Dict[str, Any]
    ) -> CycleSummary:
        """
        Create summary for a single cycle.

        Extracts key learnings, patterns, and outcomes from cycle data.

        Args:
            cycle_id: Cycle identifier
            cycle_data: Full cycle data including experiments and results

        Returns:
            CycleSummary object
        """
        experiments = cycle_data.get("experiments", [])

        # Find best experiment
        best_accuracy = 0.0
        best_architecture = "unknown"
        best_experiment_id = ""

        completed_experiments = [
            exp for exp in experiments
            if exp.get("status") == "completed" and exp.get("metrics")
        ]

        for exp in completed_experiments:
            metrics = exp.get("metrics", {})
            acc = metrics.get("auc", metrics.get("accuracy", 0))
            if acc > best_accuracy:
                best_accuracy = acc
                best_experiment_id = exp.get("experiment_id", "")
                best_architecture = exp.get("config", {}).get("model", "unknown")

        # Extract what worked
        what_worked = []
        for exp in completed_experiments:
            if exp.get("metrics", {}).get("auc", 0) > 0.8:
                config = exp.get("config", {})
                what_worked.append(
                    f"Config with {config.get('model', 'unknown')} model, "
                    f"lr={config.get('learning_rate', 'N/A')}"
                )

        # Extract what failed
        what_failed = []
        failed_experiments = [
            exp for exp in experiments
            if exp.get("status") == "failed" or exp.get("status") == "error"
        ]
        for exp in failed_experiments[:5]:  # Limit to 5
            error = exp.get("error", "Unknown error")
            config = exp.get("config", {})
            what_failed.append(
                f"{config.get('model', 'unknown')}: {error[:100]}"
            )

        # Extract key decisions (from proposals)
        key_decisions = []
        proposals = cycle_data.get("proposals", [])
        for proposal in proposals[:5]:
            decision = proposal.get("reasoning", proposal.get("description", ""))
            if decision:
                key_decisions.append(decision[:150])

        # Extract hypotheses tested
        hypotheses_tested = []
        for exp in experiments[:5]:
            hypothesis = exp.get("hypothesis", exp.get("config", {}).get("hypothesis"))
            if hypothesis:
                hypotheses_tested.append(hypothesis[:150])

        # Compute aggregate metrics
        metrics = {}
        if completed_experiments:
            auc_values = [e.get("metrics", {}).get("auc", 0) for e in completed_experiments if e.get("metrics")]
            if auc_values:
                metrics["avg_auc"] = sum(auc_values) / len(auc_values)
                metrics["max_auc"] = max(auc_values)
                metrics["min_auc"] = min(auc_values)

        # Create summary
        summary = CycleSummary(
            cycle_id=cycle_id,
            timestamp=datetime.now(),
            best_accuracy=best_accuracy,
            best_architecture=best_architecture,
            key_decisions=key_decisions,
            what_worked=what_worked[:5],
            what_failed=what_failed[:5],
            hypotheses_tested=hypotheses_tested,
            metrics=metrics,
            num_experiments=len(experiments)
        )

        # Estimate token count
        summary_text = json.dumps(summary.to_dict())
        summary.token_count = self._estimate_tokens(summary_text)

        # Store and save
        self._cycle_summaries[cycle_id] = summary
        self._save_cycle_summaries()

        logger.info(
            f"Summarized cycle {cycle_id}: {len(experiments)} experiments, "
            f"best_acc={best_accuracy:.4f}, tokens={summary.token_count}"
        )

        return summary

    def create_executive_summary(
        self,
        last_n_cycles: int = 50,
        max_tokens: Optional[int] = None
    ) -> ExecutiveSummary:
        """
        Create executive summary for Director agent.

        High-level strategic view spanning many cycles.
        Designed for Claude's 200K context capacity.

        Args:
            last_n_cycles: Number of recent cycles to include
            max_tokens: Maximum token budget (default: EXECUTIVE_MAX_TOKENS)

        Returns:
            ExecutiveSummary object
        """
        if max_tokens is None:
            max_tokens = self.EXECUTIVE_MAX_TOKENS

        # Get recent cycle summaries
        sorted_cycles = sorted(
            self._cycle_summaries.values(),
            key=lambda s: s.cycle_id,
            reverse=True
        )[:last_n_cycles]

        if not sorted_cycles:
            return ExecutiveSummary(
                generated_at=datetime.now(),
                token_count=100
            )

        # Build executive summary
        cycles_covered = [s.cycle_id for s in sorted_cycles]
        accuracy_trajectory = [s.best_accuracy for s in sorted_cycles]

        # Best overall
        best_cycle = max(sorted_cycles, key=lambda s: s.best_accuracy)
        best_overall_accuracy = best_cycle.best_accuracy
        best_experiment_id = f"cycle_{best_cycle.cycle_id}"

        # Aggregate patterns
        all_worked = []
        all_failed = []
        for cycle in sorted_cycles:
            all_worked.extend(cycle.what_worked)
            all_failed.extend(cycle.what_failed)

        # Count pattern frequencies
        worked_counts = {}
        for pattern in all_worked:
            # Extract key terms
            key = pattern.split(",")[0] if "," in pattern else pattern[:50]
            worked_counts[key] = worked_counts.get(key, 0) + 1

        failed_counts = {}
        for pattern in all_failed:
            key = pattern.split(":")[0] if ":" in pattern else pattern[:50]
            failed_counts[key] = failed_counts.get(key, 0) + 1

        # Top patterns
        successful_patterns = sorted(
            worked_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        successful_patterns = [f"{p[0]} (x{p[1]})" for p in successful_patterns]

        failed_patterns = sorted(
            failed_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        failed_patterns = [f"{p[0]} (x{p[1]})" for p in failed_patterns]

        # Detect stagnation
        stagnation_detected = False
        stagnation_cycles = 0
        if len(accuracy_trajectory) >= 5:
            recent_5 = accuracy_trajectory[:5]
            improvement = max(recent_5) - min(recent_5)
            if improvement < 0.01:
                stagnation_detected = True
                stagnation_cycles = 5

        # Generate suggested directions
        suggested_directions = self._generate_suggestions(
            successful_patterns,
            failed_patterns,
            stagnation_detected,
            best_overall_accuracy
        )

        # Calculate statistics
        total_experiments = sum(s.num_experiments for s in sorted_cycles)
        successful_count = sum(
            1 for s in sorted_cycles
            if s.best_accuracy > 0.5
        )
        success_rate = successful_count / len(sorted_cycles) if sorted_cycles else 0

        summary = ExecutiveSummary(
            cycles_covered=cycles_covered,
            accuracy_trajectory=accuracy_trajectory,
            best_overall_accuracy=best_overall_accuracy,
            best_experiment_id=best_experiment_id,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            suggested_directions=suggested_directions,
            stagnation_detected=stagnation_detected,
            stagnation_cycles=stagnation_cycles,
            total_experiments=total_experiments,
            success_rate=success_rate,
            generated_at=datetime.now()
        )

        # Estimate token count
        summary_text = json.dumps(summary.to_dict())
        summary.token_count = self._estimate_tokens(summary_text)

        # Save executive summary
        exec_file = self.storage_path / "executive_summary.json"
        with open(exec_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        logger.info(
            f"Created executive summary: {len(cycles_covered)} cycles, "
            f"best_acc={best_overall_accuracy:.4f}, tokens={summary.token_count}"
        )

        return summary

    def create_working_summary(
        self,
        agent_name: str,
        max_tokens: int = 8000,
        last_n_cycles: int = 10
    ) -> Dict[str, Any]:
        """
        Create working summary for a specific agent.

        Tailored context based on agent role, with more detail
        for recent cycles.

        Args:
            agent_name: Name/role of the requesting agent
            max_tokens: Maximum token budget
            last_n_cycles: Number of recent cycles to include

        Returns:
            Dict with agent-specific context
        """
        # Get agent focus areas
        agent_role = agent_name.lower().replace("_agent", "").replace("_001", "")
        focus_areas = self.AGENT_FOCUS_AREAS.get(agent_role, [])

        # Get recent cycles
        sorted_cycles = sorted(
            self._cycle_summaries.values(),
            key=lambda s: s.cycle_id,
            reverse=True
        )[:last_n_cycles]

        if not sorted_cycles:
            return {
                "agent_name": agent_name,
                "max_tokens": max_tokens,
                "used_tokens": 100,
                "cycles": [],
                "focus_areas": focus_areas,
                "summary": "No cycle data available yet."
            }

        # Build working context
        cycles_data = []
        used_tokens = 0

        for cycle in sorted_cycles:
            cycle_dict = cycle.to_dict()

            # Filter to focus areas if applicable
            if focus_areas:
                cycle_dict = self._filter_to_focus(cycle_dict, focus_areas)

            cycle_text = json.dumps(cycle_dict)
            cycle_tokens = self._estimate_tokens(cycle_text)

            if used_tokens + cycle_tokens <= max_tokens:
                cycles_data.append(cycle_dict)
                used_tokens += cycle_tokens
            else:
                # Try compressed version
                compressed = self._compress_cycle(cycle_dict)
                compressed_tokens = self._estimate_tokens(json.dumps(compressed))
                if used_tokens + compressed_tokens <= max_tokens:
                    cycles_data.append(compressed)
                    used_tokens += compressed_tokens
                else:
                    break

        # Build summary
        recent_best = max(
            (c.best_accuracy for c in sorted_cycles), default=0.0
        )

        working_summary = {
            "agent_name": agent_name,
            "max_tokens": max_tokens,
            "used_tokens": used_tokens,
            "num_cycles": len(cycles_data),
            "focus_areas": focus_areas,
            "recent_best_accuracy": recent_best,
            "cycles": cycles_data,
            "generated_at": datetime.now().isoformat()
        }

        # Save working summary
        summary_file = self.storage_path / f"working_{agent_role}.json"
        with open(summary_file, 'w') as f:
            json.dump(working_summary, f, indent=2)

        logger.debug(
            f"Created working summary for {agent_name}: "
            f"{len(cycles_data)} cycles, {used_tokens}/{max_tokens} tokens"
        )

        return working_summary

    def _filter_to_focus(
        self,
        cycle_dict: Dict[str, Any],
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Filter cycle data to focus areas relevant to agent role."""
        filtered = {
            "cycle_id": cycle_dict.get("cycle_id"),
            "timestamp": cycle_dict.get("timestamp"),
            "best_accuracy": cycle_dict.get("best_accuracy"),
            "num_experiments": cycle_dict.get("num_experiments")
        }

        # Filter decisions/patterns by focus keywords
        for key in ["key_decisions", "what_worked", "what_failed"]:
            items = cycle_dict.get(key, [])
            filtered_items = []
            for item in items:
                item_lower = item.lower()
                if any(focus in item_lower for focus in focus_areas):
                    filtered_items.append(item)
            if filtered_items:
                filtered[key] = filtered_items

        return filtered

    def _compress_cycle(self, cycle_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create compressed version of cycle data."""
        return {
            "cycle_id": cycle_dict.get("cycle_id"),
            "best_accuracy": cycle_dict.get("best_accuracy"),
            "num_experiments": cycle_dict.get("num_experiments"),
            "key_insights": cycle_dict.get("what_worked", [])[:2] +
                           cycle_dict.get("what_failed", [])[:1]
        }

    def _generate_suggestions(
        self,
        successful_patterns: List[str],
        failed_patterns: List[str],
        stagnation_detected: bool,
        best_accuracy: float
    ) -> List[str]:
        """Generate strategic suggestions based on patterns."""
        suggestions = []

        # If stagnating, suggest exploration
        if stagnation_detected:
            suggestions.append(
                "Performance has stagnated. Consider exploring different "
                "architecture families or hyperparameter ranges."
            )

        # Based on accuracy level
        if best_accuracy < 0.7:
            suggestions.append(
                "Best accuracy is below 0.7. Focus on finding stable "
                "baseline configurations before optimizing."
            )
        elif best_accuracy < 0.85:
            suggestions.append(
                "Good progress (>0.7). Consider fine-tuning successful "
                "configurations with smaller learning rate adjustments."
            )
        else:
            suggestions.append(
                "Strong performance (>0.85). Focus on robustness and "
                "edge case handling to push towards target."
            )

        # Based on failed patterns
        if failed_patterns:
            top_failure = failed_patterns[0].split(" (")[0]
            suggestions.append(
                f"Frequently failing pattern: '{top_failure}'. "
                f"Consider avoiding or modifying this approach."
            )

        # Based on successful patterns
        if successful_patterns:
            top_success = successful_patterns[0].split(" (")[0]
            suggestions.append(
                f"Successful pattern: '{top_success}'. "
                f"Consider building on this approach."
            )

        return suggestions[:5]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get statistics about stored summaries."""
        return {
            "num_cycle_summaries": len(self._cycle_summaries),
            "cycles_covered": sorted(self._cycle_summaries.keys()),
            "storage_path": str(self.storage_path),
            "total_tokens": sum(s.token_count for s in self._cycle_summaries.values())
        }


# Singleton instance
_summarizer_instance: Optional[CrossCycleSummarizer] = None


def get_summarizer(storage_path: Optional[str] = None) -> CrossCycleSummarizer:
    """Get or create singleton CrossCycleSummarizer instance."""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = CrossCycleSummarizer(storage_path=storage_path)
    return _summarizer_instance


def reset_summarizer() -> None:
    """Reset the singleton instance (for testing)."""
    global _summarizer_instance
    _summarizer_instance = None
