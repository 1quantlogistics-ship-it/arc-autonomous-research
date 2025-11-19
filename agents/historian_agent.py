"""
HistorianAgent: Memory management and learning
===============================================

The Historian compresses experiment history, tracks patterns,
and infers constraints from past failures.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class HistorianAgent(BaseAgent):
    """
    Memory and learning agent.

    Responsibilities:
    - Compress experiment history
    - Track winning/failing configurations
    - Infer forbidden parameter ranges
    - Analyze performance trends
    """

    def __init__(
        self,
        agent_id: str = "historian_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.0,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Historian agent."""
        super().__init__(
            agent_id=agent_id,
            role="historian",
            model=model,
            capabilities=[AgentCapability.MEMORY_MANAGEMENT],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update history summary with new experiment results.

        Args:
            input_data: Contains new experiment results

        Returns:
            Updated history summary
        """
        import time
        start_time = time.time()

        try:
            # Read current history
            history = self.read_memory("history_summary.json") or {}
            constraints = self.read_memory("constraints.json") or {}

            # Get experiment results from input
            new_results = input_data.get("experiment_results", [])

            # Try LLM-based update with timeout and fallback
            try:
                # Compress history for LLM prompt to prevent bloat
                compressed_history = self._compress_history_for_prompt(history)
                compressed_constraints = self._compress_constraints_for_prompt(constraints)

                # Build prompt
                prompt = self._build_update_prompt(compressed_history, compressed_constraints, new_results)

                # Get LLM client
                client = self.llm_router.get_client_for_role(self.role)

                # Generate updated history with increased timeout (180s instead of 60s)
                response = client.generate_json(prompt, max_tokens=2500, temperature=0.5, timeout=180)

                # Write to memory
                self.write_memory("history_summary.json", response.get("history", {}))
                self.write_memory("constraints.json", response.get("constraints", {}))

                # Track success
                duration_ms = (time.time() - start_time) * 1000
                self._track_task("update_history", success=True, duration_ms=duration_ms)

                return response

            except Exception as llm_error:
                # Fallback to algorithmic update if LLM times out or fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"LLM-based history update failed ({llm_error}), using algorithmic fallback")

                # Algorithmic update (no LLM required)
                response = self._algorithmic_history_update(history, constraints, new_results)

                # Write to memory
                self.write_memory("history_summary.json", response.get("history", {}))
                self.write_memory("constraints.json", response.get("constraints", {}))

                # Track success with fallback flag
                duration_ms = (time.time() - start_time) * 1000
                self._track_task("update_history_fallback", success=True, duration_ms=duration_ms)

                return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("update_history", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Historians vote based on historical precedent.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision based on history
        """
        # Read history
        history = self.read_memory("history_summary.json")

        # Check if similar config failed before
        failed_configs = history.get("failed_configs", [])
        config_changes = proposal.get("config_changes", {})

        for failed in failed_configs:
            # Simple similarity check (exact match)
            if all(failed.get(k) == v for k, v in config_changes.items() if k in failed):
                return {
                    "decision": "reject",
                    "confidence": 0.85,
                    "reasoning": "Similar configuration failed previously",
                    "suggested_changes": None
                }

        # Default: approve
        return {
            "decision": "approve",
            "confidence": 0.7,
            "reasoning": "No historical evidence of failure"
        }

    def integrate_experiment_results(
        self,
        experiment_results: List[Dict[str, Any]],
        cycle_id: int
    ) -> Dict[str, Any]:
        """
        Integrate completed experiment results into history.

        This is the key feedback loop that enables autonomous learning:
        1. Update training_history.json with new results
        2. Update constraints based on failures
        3. Track performance trends
        4. Identify successful patterns

        Args:
            experiment_results: List of result dicts from training_executor
            cycle_id: Current cycle ID

        Returns:
            Dict with integration summary
        """
        # Load or initialize training history
        training_history_path = Path(self.memory_path) / "training_history.json"
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
        else:
            training_history = {
                "experiments": [],
                "total_experiments": 0,
                "best_metrics": {},
                "cycles": []
            }

        # Process each result
        successful_experiments = []
        failed_experiments = []

        for result in experiment_results:
            experiment_id = result.get("experiment_id")
            metrics = result.get("metrics", {})
            status = result.get("status")
            config = result.get("config", {})

            # Add to history
            history_entry = {
                "experiment_id": experiment_id,
                "cycle_id": cycle_id,
                "status": status,
                "config": config,
                "metrics": metrics,
                "completed_at": result.get("completed_at"),
                "duration_seconds": result.get("duration_seconds"),
                "proposal_type": result.get("proposal_type"),
                "risk_level": result.get("risk_level")
            }
            training_history["experiments"].append(history_entry)

            # Track success/failure
            if status == "completed" and metrics:
                successful_experiments.append(result)
                # Update best metrics
                self._update_best_metrics(training_history, metrics)
            else:
                failed_experiments.append(result)

        # Update cycle summary
        cycle_summary = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(experiment_results),
            "successful": len(successful_experiments),
            "failed": len(failed_experiments),
            "best_metrics_updated": self._check_if_improved(training_history)
        }
        training_history["cycles"].append(cycle_summary)
        training_history["total_experiments"] = len(training_history["experiments"])

        # Write updated history
        with open(training_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Update constraints based on failures
        if failed_experiments:
            self._update_constraints_from_failures(failed_experiments)

        # Update history summary (compressed version)
        self._update_history_summary(training_history, cycle_id)

        return {
            "status": "integrated",
            "cycle_id": cycle_id,
            "total_experiments": len(experiment_results),
            "successful": len(successful_experiments),
            "failed": len(failed_experiments),
            "best_metrics": training_history.get("best_metrics", {}),
            "training_history_updated": True
        }

    def _update_best_metrics(
        self,
        training_history: Dict[str, Any],
        new_metrics: Dict[str, float]
    ) -> None:
        """Update best metrics if new results improved."""
        best_metrics = training_history.setdefault("best_metrics", {})

        for metric_name, metric_value in new_metrics.items():
            if metric_value is None:
                continue

            current_best = best_metrics.get(metric_name)
            if current_best is None or metric_value > current_best:
                best_metrics[metric_name] = metric_value

    def _check_if_improved(self, training_history: Dict[str, Any]) -> bool:
        """Check if best metrics were updated in this integration."""
        # Simple check: compare last cycle's best to current best
        # More sophisticated version would track per-cycle
        return True  # Simplified for now

    def _update_constraints_from_failures(
        self,
        failed_experiments: List[Dict[str, Any]]
    ) -> None:
        """Update constraints based on failed experiments."""
        constraints_path = Path(self.memory_path) / "constraints.json"

        if constraints_path.exists():
            with open(constraints_path, 'r') as f:
                constraints = json.load(f)
        else:
            constraints = {"forbidden_ranges": [], "unstable_configs": []}

        # Analyze failures and add forbidden ranges
        for failure in failed_experiments:
            config = failure.get("config", {})
            error = failure.get("error", "")

            # Extract problematic parameters
            # Simple heuristic: if LR too high and failed, add constraint
            learning_rate = config.get("learning_rate")
            if learning_rate and learning_rate > 0.01:
                # Check if constraint already exists
                existing = next(
                    (c for c in constraints["forbidden_ranges"] if c.get("param") == "learning_rate"),
                    None
                )
                if not existing:
                    constraints["forbidden_ranges"].append({
                        "param": "learning_rate",
                        "min": None,
                        "max": 0.01,
                        "reason": "Training instability observed with high learning rates"
                    })

            # Add to unstable configs
            constraints["unstable_configs"].append({
                "config": config,
                "reason": error or "Training failed",
                "experiment_id": failure.get("experiment_id")
            })

        # Write updated constraints
        with open(constraints_path, 'w') as f:
            json.dump(constraints, f, indent=2)

    def _update_history_summary(
        self,
        training_history: Dict[str, Any],
        cycle_id: int
    ) -> None:
        """Update compressed history summary."""
        history_summary_path = Path(self.memory_path) / "history_summary.json"

        # Extract recent experiments (last 10)
        recent_experiments = training_history["experiments"][-10:]

        # Identify successful patterns
        successful_configs = [
            exp["config"] for exp in training_history["experiments"]
            if exp.get("status") == "completed" and exp.get("metrics")
        ]

        # Identify failed configs
        failed_configs = [
            exp["config"] for exp in training_history["experiments"]
            if exp.get("status") == "failed"
        ]

        history_summary = {
            "total_cycles": len(training_history["cycles"]),
            "total_experiments": training_history["total_experiments"],
            "best_metrics": training_history.get("best_metrics", {}),
            "recent_experiments": recent_experiments,
            "successful_patterns": self._extract_patterns(successful_configs),
            "failed_configs": failed_configs[-10:],  # Keep last 10 failures
            "last_updated": datetime.now().isoformat(),
            "last_cycle_id": cycle_id
        }

        with open(history_summary_path, 'w') as f:
            json.dump(history_summary, f, indent=2)

    def _extract_patterns(self, successful_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common patterns from successful configs."""
        if not successful_configs:
            return []

        # Simple pattern extraction: most common values for each parameter
        patterns = []

        # Group by model
        models = {}
        for config in successful_configs:
            model = config.get("model", "unknown")
            models[model] = models.get(model, 0) + 1

        if models:
            most_common_model = max(models, key=models.get)
            patterns.append({
                "parameter": "model",
                "value": most_common_model,
                "frequency": models[most_common_model] / len(successful_configs)
            })

        # Group by optimizer
        optimizers = {}
        for config in successful_configs:
            optimizer = config.get("optimizer", "unknown")
            optimizers[optimizer] = optimizers.get(optimizer, 0) + 1

        if optimizers:
            most_common_optimizer = max(optimizers, key=optimizers.get)
            patterns.append({
                "parameter": "optimizer",
                "value": most_common_optimizer,
                "frequency": optimizers[most_common_optimizer] / len(successful_configs)
            })

        return patterns

    def get_training_history(self) -> Dict[str, Any]:
        """Get complete training history."""
        training_history_path = Path(self.memory_path) / "training_history.json"
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                return json.load(f)
        return {}

    def get_performance_trend(self, metric: str = "auc", window: int = 10) -> List[float]:
        """
        Get performance trend for a specific metric.

        Args:
            metric: Metric name (e.g., "auc", "sensitivity")
            window: Number of recent experiments to analyze

        Returns:
            List of metric values over time
        """
        history = self.get_training_history()
        experiments = history.get("experiments", [])

        # Extract metric values from recent experiments
        values = []
        for exp in experiments[-window:]:
            metrics = exp.get("metrics", {})
            if metric in metrics and metrics[metric] is not None:
                values.append(metrics[metric])

        return values

    def detect_stagnation(self, metric: str = "auc", threshold: float = 0.01, window: int = 5) -> bool:
        """
        Detect if progress has stagnated.

        Args:
            metric: Metric to check
            threshold: Minimum improvement required
            window: Number of recent experiments to check

        Returns:
            True if stagnated (no improvement > threshold in last N experiments)
        """
        trend = self.get_performance_trend(metric, window)

        if len(trend) < 2:
            return False

        # Check if improvement between first and last is below threshold
        improvement = trend[-1] - trend[0]
        return improvement < threshold

    def _compress_history_for_prompt(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress history to prevent prompt bloat (addresses >180s Historian issue).

        Args:
            history: Full history dictionary

        Returns:
            Compressed history with only essential info
        """
        # Keep only: best metrics, total counts, last 10 experiments
        compressed = {
            "total_cycles": history.get("total_cycles", 0),
            "total_experiments": history.get("total_experiments", 0),
            "best_metrics": history.get("best_metrics", {}),
            "recent_experiments": history.get("recent_experiments", [])[-10:],  # Only last 10
            "successful_patterns": history.get("successful_patterns", [])[:5],  # Top 5 patterns
            "failed_configs": history.get("failed_configs", [])[-5:]  # Last 5 failures
        }
        return compressed

    def _compress_constraints_for_prompt(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress constraints to prevent prompt bloat.

        Args:
            constraints: Full constraints dictionary

        Returns:
            Compressed constraints with only active constraints
        """
        # Keep only: active forbidden ranges (limit to 10), unstable configs (limit to 5)
        compressed = {
            "forbidden_ranges": constraints.get("forbidden_ranges", [])[:10],
            "unstable_configs": constraints.get("unstable_configs", [])[:5],
            "safe_baselines": constraints.get("safe_baselines", [])[:3]
        }
        return compressed

    def _algorithmic_history_update(
        self,
        history: Dict[str, Any],
        constraints: Dict[str, Any],
        new_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Algorithmic fallback when LLM times out (no LLM required).

        Args:
            history: Current history
            constraints: Current constraints
            new_results: New experiment results

        Returns:
            Updated history and constraints
        """
        # Update counts
        updated_history = history.copy()
        updated_history["total_experiments"] = updated_history.get("total_experiments", 0) + len(new_results)

        # Update best metrics
        for result in new_results:
            metrics = result.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    best_metrics = updated_history.setdefault("best_metrics", {})
                    if metric_name not in best_metrics or metric_value > best_metrics[metric_name]:
                        best_metrics[metric_name] = metric_value

        # Add to recent experiments (keep last 20)
        recent = updated_history.setdefault("recent_experiments", [])
        for result in new_results:
            recent.append({
                "experiment_id": result.get("experiment_id"),
                "config": result.get("config", {}),
                "metrics": result.get("metrics", {}),
                "status": result.get("status")
            })
        updated_history["recent_experiments"] = recent[-20:]  # Keep last 20

        # Update failed configs
        failed = updated_history.setdefault("failed_configs", [])
        for result in new_results:
            if result.get("status") != "completed":
                failed.append({
                    "config": result.get("config", {}),
                    "error": result.get("error", "unknown")
                })
        updated_history["failed_configs"] = failed[-10:]  # Keep last 10

        return {
            "history": updated_history,
            "constraints": constraints,  # Keep constraints unchanged in fallback
            "fallback_used": True
        }

    def _build_update_prompt(
        self,
        history: Dict[str, Any],
        constraints: Dict[str, Any],
        new_results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for history update (now receives compressed inputs)."""
        return f"""You are the Historian agent in ARC (Autonomous Research Collective).
Your role is to maintain a compressed history of all experiments and learn from results.

# Current History Summary (Compressed)
{json.dumps(history, indent=2)}

# Current Constraints (Compressed)
{json.dumps(constraints, indent=2)}

# New Experiment Results
{json.dumps(new_results, indent=2)}

# Your Task
Update the history summary and constraints based on new results:
1. Update total cycles and experiments
2. Update best metrics if improved
3. Add recent experiments to history
4. Identify failed configs and update forbidden ranges
5. Detect successful patterns
6. Infer new constraints from failures

Return ONLY a valid JSON object:
{{
  "history": {{
    "total_cycles": N,
    "total_experiments": M,
    "best_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
    "recent_experiments": [...],
    "failed_configs": [...],
    "successful_patterns": [...]
  }},
  "constraints": {{
    "forbidden_ranges": [{{"parameter": "X", "min": A, "max": B, "reason": "..."}}],
    "unstable_configs": [...],
    "safe_baselines": [...]
  }}
}}"""
