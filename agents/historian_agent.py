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
from tools.world_model import get_world_model

# Phase E: Architecture grammar tracking
try:
    from schemas.architecture_grammar import (
        ArchitectureGrammar, classify_architecture_family, ArchitectureFamily
    )
    ARCHITECTURE_GRAMMAR_AVAILABLE = True
except ImportError:
    ARCHITECTURE_GRAMMAR_AVAILABLE = False

# Phase E: Curriculum strategy tracking (Task 2.6)
try:
    from schemas.curriculum_strategy import (
        CurriculumStrategy, CurriculumStage, DifficultyMetric, PacingStrategy
    )
    CURRICULUM_STRATEGY_AVAILABLE = True
except ImportError:
    CURRICULUM_STRATEGY_AVAILABLE = False

# Phase E: Multi-objective optimization tracking (Task 3.2)
try:
    from schemas.multi_objective import (
        ObjectiveSpec, ParetoFront, ParetoSolution, MultiObjectiveMetrics,
        compute_pareto_frontier, compute_hypervolume, is_dominated
    )
    MULTI_OBJECTIVE_AVAILABLE = True
except ImportError:
    MULTI_OBJECTIVE_AVAILABLE = False


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

        # Initialize world model for outcome prediction
        self.world_model = get_world_model(model_path=str(Path(memory_path) / "world_model"))

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

                # Update world model with new experiment results
                self._update_world_model(new_results, input_data.get("cycle_id", 0))

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

                # Update world model with new experiment results
                self._update_world_model(new_results, input_data.get("cycle_id", 0))

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

            # Phase E: Extract architecture family if grammar present
            architecture_family = None
            if ARCHITECTURE_GRAMMAR_AVAILABLE:
                architecture_family = self._extract_architecture_family(config)

            # Phase E Task 2.6: Extract curriculum info if present
            curriculum_info = None
            if CURRICULUM_STRATEGY_AVAILABLE:
                curriculum_info = self._extract_curriculum_info(config, metrics)

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
                "risk_level": result.get("risk_level"),
                "architecture_family": architecture_family,  # Phase E: Track architecture family
                "curriculum_info": curriculum_info  # Phase E Task 2.6: Track curriculum
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

    def _extract_architecture_family(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Extract architecture family from experiment config.

        Args:
            config: Experiment configuration

        Returns:
            Architecture family name or None
        """
        if not ARCHITECTURE_GRAMMAR_AVAILABLE:
            return None

        try:
            # Check if config has architecture grammar
            arch_config = config.get("architecture", {})
            grammar_dict = arch_config.get("grammar")

            if grammar_dict:
                # Parse grammar and classify
                grammar = ArchitectureGrammar(**grammar_dict)
                family = classify_architecture_family(grammar)
                return family.value

            # Fallback: check for architecture_grammar in root config
            if "architecture_grammar" in config:
                grammar = ArchitectureGrammar(**config["architecture_grammar"])
                family = classify_architecture_family(grammar)
                return family.value

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not extract architecture family: {e}")

        return None

    def get_architecture_family_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics grouped by architecture family.

        Returns:
            Dict mapping architecture family to performance stats
        """
        if not ARCHITECTURE_GRAMMAR_AVAILABLE:
            return {}

        training_history_path = Path(self.memory_path) / "training_history.json"
        if not training_history_path.exists():
            return {}

        with open(training_history_path, 'r') as f:
            training_history = json.load(f)

        # Group experiments by architecture family
        family_stats = {}

        for exp in training_history.get("experiments", []):
            family = exp.get("architecture_family")
            if not family:
                continue

            if family not in family_stats:
                family_stats[family] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_auc": [],
                    "best_auc": 0.0,
                    "experiments": []
                }

            stats = family_stats[family]
            stats["count"] += 1

            if exp.get("status") == "completed":
                stats["successful"] += 1
                metrics = exp.get("metrics", {})
                auc = metrics.get("auc")

                if auc:
                    stats["avg_auc"].append(auc)
                    stats["best_auc"] = max(stats["best_auc"], auc)

                stats["experiments"].append({
                    "experiment_id": exp.get("experiment_id"),
                    "auc": auc,
                    "cycle_id": exp.get("cycle_id")
                })
            else:
                stats["failed"] += 1

        # Compute averages
        for family, stats in family_stats.items():
            if stats["avg_auc"]:
                stats["avg_auc"] = sum(stats["avg_auc"]) / len(stats["avg_auc"])
            else:
                stats["avg_auc"] = 0.0

        return family_stats

    def _extract_curriculum_info(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract curriculum learning information from experiment config.

        Phase E Task 2.6: Parse curriculum strategy and stage info.

        Args:
            config: Experiment configuration
            metrics: Experiment metrics

        Returns:
            Dict with curriculum info or None if not a curriculum experiment
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            return None

        try:
            # Check for curriculum_strategy in config
            curriculum_dict = config.get("curriculum_strategy")
            if not curriculum_dict:
                return None

            # Parse curriculum strategy
            curriculum = CurriculumStrategy(**curriculum_dict)

            # Extract current stage info
            current_stage_id = config.get("current_curriculum_stage", 0)
            current_epoch = config.get("current_epoch", 0)

            curriculum_info = {
                "strategy_name": curriculum.name,
                "difficulty_metric": curriculum.difficulty_metric.value,
                "pacing_strategy": curriculum.pacing_strategy.value,
                "current_stage": current_stage_id,
                "current_epoch": current_epoch,
                "total_stages": len(curriculum.stages),
                "stage_metrics": {
                    "auc": metrics.get("auc"),
                    "sensitivity": metrics.get("sensitivity"),
                    "specificity": metrics.get("specificity")
                }
            }

            # Check if this is a stage transition
            if config.get("stage_transition_event"):
                curriculum_info["transition"] = config["stage_transition_event"]

            return curriculum_info

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not extract curriculum info: {e}")
            return None

    def track_curriculum_progression(
        self,
        experiment_id: str,
        curriculum_name: str,
        stage_transition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track curriculum stage transitions and progression.

        Phase E Task 2.6: Log curriculum learning progression events.

        Args:
            experiment_id: Experiment identifier
            curriculum_name: Name of curriculum strategy
            stage_transition: Stage transition event (if applicable)

        Returns:
            Curriculum progression summary
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            return {"status": "curriculum_tracking_unavailable"}

        # Load curriculum history
        curriculum_history_path = Path(self.memory_path) / "curriculum_history.json"

        if curriculum_history_path.exists():
            with open(curriculum_history_path, 'r') as f:
                curriculum_history = json.load(f)
        else:
            curriculum_history = {"curriculum_strategies": {}}

        # Initialize curriculum strategy entry if needed
        if curriculum_name not in curriculum_history["curriculum_strategies"]:
            curriculum_history["curriculum_strategies"][curriculum_name] = {
                "total_uses": 0,
                "successful_completions": 0,
                "avg_final_auc": 0.0,
                "stage_performance": {},
                "transitions": []
            }

        strategy_data = curriculum_history["curriculum_strategies"][curriculum_name]
        strategy_data["total_uses"] += 1

        # Add stage transition if provided
        if stage_transition:
            strategy_data["transitions"].append({
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                **stage_transition
            })

        # Save updated history
        with open(curriculum_history_path, 'w') as f:
            json.dump(curriculum_history, f, indent=2)

        return {
            "status": "tracked",
            "curriculum_name": curriculum_name,
            "total_uses": strategy_data["total_uses"]
        }

    def analyze_curriculum_effectiveness(
        self,
        curriculum_name: str,
        baseline_name: str = "baseline_no_curriculum"
    ) -> Dict[str, Any]:
        """
        Analyze curriculum strategy effectiveness vs baseline.

        Phase E Task 2.6: Compare curriculum performance to baseline.

        Args:
            curriculum_name: Name of curriculum strategy to analyze
            baseline_name: Baseline strategy for comparison

        Returns:
            Effectiveness analysis with improvement metrics
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            return {"status": "curriculum_tracking_unavailable"}

        # Load training history
        training_history_path = Path(self.memory_path) / "training_history.json"
        if not training_history_path.exists():
            return {"status": "no_history"}

        with open(training_history_path, 'r') as f:
            training_history = json.load(f)

        # Find curriculum experiments
        curriculum_exps = []
        baseline_exps = []

        for exp in training_history.get("experiments", []):
            curriculum_info = exp.get("curriculum_info")
            if not curriculum_info:
                continue

            if curriculum_info.get("strategy_name") == curriculum_name:
                curriculum_exps.append(exp)
            elif curriculum_info.get("strategy_name") == baseline_name:
                baseline_exps.append(exp)

        if not curriculum_exps:
            return {"status": "no_curriculum_experiments", "curriculum_name": curriculum_name}

        # Compute curriculum metrics
        curriculum_aucs = [
            exp["metrics"].get("auc", 0) for exp in curriculum_exps
            if exp.get("status") == "completed" and exp.get("metrics")
        ]

        baseline_aucs = [
            exp["metrics"].get("auc", 0) for exp in baseline_exps
            if exp.get("status") == "completed" and exp.get("metrics")
        ] if baseline_exps else [0.75]  # Default baseline

        curriculum_avg_auc = sum(curriculum_aucs) / len(curriculum_aucs) if curriculum_aucs else 0
        baseline_avg_auc = sum(baseline_aucs) / len(baseline_aucs) if baseline_aucs else 0.75

        improvement = curriculum_avg_auc - baseline_avg_auc

        return {
            "status": "analyzed",
            "curriculum_name": curriculum_name,
            "baseline_name": baseline_name,
            "curriculum_avg_auc": curriculum_avg_auc,
            "baseline_avg_auc": baseline_avg_auc,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_avg_auc * 100) if baseline_avg_auc > 0 else 0,
            "num_curriculum_experiments": len(curriculum_exps),
            "num_baseline_experiments": len(baseline_exps)
        }

    def get_curriculum_stage_performance(
        self,
        curriculum_name: str,
        stage_id: int
    ) -> Dict[str, Any]:
        """
        Get performance statistics for a specific curriculum stage.

        Phase E Task 2.6: Analyze stage-wise performance.

        Args:
            curriculum_name: Name of curriculum strategy
            stage_id: Stage identifier (0-indexed)

        Returns:
            Stage performance summary
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            return {"status": "curriculum_tracking_unavailable"}

        # Load curriculum history
        curriculum_history_path = Path(self.memory_path) / "curriculum_history.json"
        if not curriculum_history_path.exists():
            return {"status": "no_curriculum_history"}

        with open(curriculum_history_path, 'r') as f:
            curriculum_history = json.load(f)

        strategy_data = curriculum_history.get("curriculum_strategies", {}).get(curriculum_name)
        if not strategy_data:
            return {"status": "curriculum_not_found", "curriculum_name": curriculum_name}

        # Analyze transitions for this stage
        stage_transitions = [
            t for t in strategy_data.get("transitions", [])
            if t.get("from_stage") == stage_id or t.get("to_stage") == stage_id
        ]

        stage_performance = {
            "status": "found",
            "curriculum_name": curriculum_name,
            "stage_id": stage_id,
            "num_transitions": len(stage_transitions),
            "transitions": stage_transitions
        }

        return stage_performance

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

    def _update_world_model(self, new_results: List[Dict[str, Any]], cycle_id: int) -> None:
        """
        Update world model with new experiment results.

        Args:
            new_results: List of experiment results
            cycle_id: Current research cycle ID
        """
        import logging
        logger = logging.getLogger(__name__)

        for result in new_results:
            try:
                # Extract config and metrics
                config = result.get("config", {})
                metrics = result.get("metrics", {})
                experiment_id = result.get("experiment_id", "unknown")

                # Extract target metric (AUC)
                observed_metric = metrics.get("auc", 0.0)

                # Skip if no valid metric
                if observed_metric == 0.0:
                    continue

                # Update world model
                update_result = self.world_model.update(
                    config=config,
                    observed_metric=observed_metric,
                    cycle_id=cycle_id,
                    experiment_id=experiment_id
                )

                if update_result.success:
                    logger.info(
                        f"World model updated with {experiment_id}: "
                        f"AUC={observed_metric:.4f}, "
                        f"total_samples={update_result.total_samples}"
                    )
                else:
                    logger.warning(f"World model update failed for {experiment_id}")

            except Exception as e:
                logger.warning(f"Failed to update world model with result: {e}")

    def predict_proposal_outcome(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outcome of a proposal using world model.

        Args:
            proposal: Agent proposal with config

        Returns:
            Dict with prediction results
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            config = proposal.get("config", {})

            # Get world model prediction
            prediction = self.world_model.predict(config, optimize_for="auc")

            logger.debug(
                f"World model prediction for {proposal.get('experiment_id')}: "
                f"AUC={prediction.predicted_metric:.4f}, "
                f"confidence={prediction.confidence:.2f}, "
                f"uncertainty={prediction.uncertainty:.4f}"
            )

            return {
                "predicted_auc": prediction.predicted_metric,
                "confidence": prediction.confidence,
                "uncertainty": prediction.uncertainty,
                "confidence_interval": prediction.confidence_interval,
                "expected_improvement": prediction.expected_improvement,
                "details": prediction.details
            }

        except Exception as e:
            logger.warning(f"World model prediction failed: {e}")
            return {
                "predicted_auc": 0.5,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "error": str(e)
            }

    def get_pareto_frontier(
        self,
        objectives: List[ObjectiveSpec],
        filter_completed: bool = True
    ) -> Dict[str, Any]:
        """
        Compute Pareto frontier from experiment history.

        Phase E: Task 3.2 - Extract non-dominated solutions from training history
        based on specified objectives.

        Args:
            objectives: List of objective specifications (metrics to optimize)
            filter_completed: Only consider completed experiments (default: True)

        Returns:
            Dict containing:
            - pareto_front: ParetoFront object with non-dominated solutions
            - all_solutions: All experiments with Pareto rankings
            - hypervolume: Quality metric for Pareto front
        """
        if not MULTI_OBJECTIVE_AVAILABLE:
            raise RuntimeError("Multi-objective schema not available")

        # Read training history
        training_history = self.read_memory("training_history.json") or {}
        experiments = training_history.get("experiments", [])

        if not experiments:
            return {
                "pareto_front": None,
                "all_solutions": [],
                "hypervolume": 0.0,
                "message": "No experiments in history"
            }

        # Filter experiments
        if filter_completed:
            experiments = [
                exp for exp in experiments
                if exp.get("status") == "completed" and exp.get("metrics")
            ]

        if not experiments:
            return {
                "pareto_front": None,
                "all_solutions": [],
                "hypervolume": 0.0,
                "message": "No completed experiments with metrics"
            }

        # Prepare solutions for Pareto computation
        solutions = []
        for exp in experiments:
            exp_id = exp.get("experiment_id", exp.get("name", "unknown"))
            metrics = exp.get("metrics", {})

            # Extract objective values
            obj_values = {}
            for obj in objectives:
                metric_val = metrics.get(obj.metric_name)
                if metric_val is not None:
                    obj_values[obj.metric_name] = metric_val

            # Only include experiments with all objective metrics
            if len(obj_values) == len(objectives):
                solutions.append({
                    "experiment_id": exp_id,
                    "metrics": obj_values
                })

        if not solutions:
            return {
                "pareto_front": None,
                "all_solutions": [],
                "hypervolume": 0.0,
                "message": "No experiments with all required objective metrics"
            }

        # Compute Pareto frontier
        pareto_solutions = compute_pareto_frontier(
            solutions, objectives, return_all_ranks=False
        )
        all_ranked = compute_pareto_frontier(
            solutions, objectives, return_all_ranks=True
        )

        # Build ParetoFront object
        pareto_solution_objs = []
        for sol in pareto_solutions:
            pareto_solution_objs.append(ParetoSolution(
                experiment_id=sol["experiment_id"],
                objective_values=sol["metrics"],
                dominated_by_count=len(sol.get("dominated_by", [])),
                dominates_count=len(sol.get("dominates", []))
            ))

        # Compute hypervolume
        objective_values_list = [sol["metrics"] for sol in pareto_solutions]
        hypervolume = compute_hypervolume(objective_values_list, objectives)

        pareto_front = ParetoFront(
            objectives=objectives,
            solutions=pareto_solution_objs,
            hypervolume=hypervolume,
            reference_point={obj.metric_name: 0.5 for obj in objectives},
            generation=training_history.get("total_cycles", 0)
        )

        return {
            "pareto_front": pareto_front.dict(),
            "all_solutions": all_ranked,
            "hypervolume": hypervolume,
            "num_pareto_optimal": len(pareto_solutions),
            "num_total": len(solutions)
        }

    def track_pareto_evolution(
        self,
        objectives: List[ObjectiveSpec],
        cycle_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Track Pareto frontier evolution over time.

        Phase E: Task 3.2 - Save snapshots of Pareto frontier at each cycle
        to analyze how the frontier evolves during research.

        Args:
            objectives: List of objective specifications
            cycle_id: Current cycle ID (default: latest from history)

        Returns:
            Dict with evolution statistics and updated pareto_history.json
        """
        if not MULTI_OBJECTIVE_AVAILABLE:
            raise RuntimeError("Multi-objective schema not available")

        # Get current Pareto frontier
        current_frontier = self.get_pareto_frontier(objectives)

        if current_frontier["pareto_front"] is None:
            return {
                "status": "no_frontier",
                "message": "Cannot track evolution without Pareto solutions"
            }

        # Read pareto_history.json
        pareto_history_path = Path(self.memory_path) / "pareto_history.json"
        if pareto_history_path.exists():
            with open(pareto_history_path, 'r') as f:
                pareto_history = json.load(f)
        else:
            pareto_history = {
                "objective_names": [obj.metric_name for obj in objectives],
                "snapshots": []
            }

        # Get cycle ID
        if cycle_id is None:
            training_history = self.read_memory("training_history.json") or {}
            cycle_id = training_history.get("total_cycles", 0)

        # Add snapshot
        snapshot = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "hypervolume": current_frontier["hypervolume"],
            "num_pareto_optimal": current_frontier["num_pareto_optimal"],
            "pareto_solutions": [
                {
                    "experiment_id": sol["experiment_id"],
                    "objective_values": sol["objective_values"]
                }
                for sol in current_frontier["pareto_front"]["solutions"]
            ]
        }

        pareto_history["snapshots"].append(snapshot)

        # Write updated history
        with open(pareto_history_path, 'w') as f:
            json.dump(pareto_history, f, indent=2)

        # Compute evolution statistics
        if len(pareto_history["snapshots"]) > 1:
            prev_snapshot = pareto_history["snapshots"][-2]
            hypervolume_improvement = (
                snapshot["hypervolume"] - prev_snapshot["hypervolume"]
            )
            pareto_size_change = (
                snapshot["num_pareto_optimal"] - prev_snapshot["num_pareto_optimal"]
            )
        else:
            hypervolume_improvement = snapshot["hypervolume"]
            pareto_size_change = snapshot["num_pareto_optimal"]

        return {
            "status": "tracked",
            "cycle_id": cycle_id,
            "current_hypervolume": snapshot["hypervolume"],
            "hypervolume_improvement": hypervolume_improvement,
            "pareto_size": snapshot["num_pareto_optimal"],
            "pareto_size_change": pareto_size_change,
            "total_snapshots": len(pareto_history["snapshots"])
        }

    def analyze_objective_tradeoffs(
        self,
        objectives: List[ObjectiveSpec]
    ) -> Dict[str, Any]:
        """
        Analyze trade-offs between objectives.

        Phase E: Task 3.2 - Statistical analysis of metric correlations
        to understand inherent trade-offs (e.g., sensitivity vs specificity).

        Args:
            objectives: List of objective specifications

        Returns:
            Dict with correlation matrix, trade-off insights, and recommendations
        """
        if not MULTI_OBJECTIVE_AVAILABLE:
            raise RuntimeError("Multi-objective schema not available")

        # Read training history
        training_history = self.read_memory("training_history.json") or {}
        experiments = training_history.get("experiments", [])

        # Extract metric values
        metric_data = {obj.metric_name: [] for obj in objectives}
        for exp in experiments:
            if exp.get("status") == "completed" and exp.get("metrics"):
                metrics = exp.get("metrics", {})
                for obj in objectives:
                    val = metrics.get(obj.metric_name)
                    if val is not None:
                        metric_data[obj.metric_name].append(val)

        # Check if we have enough data
        min_samples = min(len(vals) for vals in metric_data.values())
        if min_samples < 3:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 3 experiments, found {min_samples}",
                "metric_counts": {k: len(v) for k, v in metric_data.items()}
            }

        # Truncate all lists to same length
        for metric_name in metric_data:
            metric_data[metric_name] = metric_data[metric_name][:min_samples]

        # Compute correlation matrix
        import numpy as np

        metric_names = list(metric_data.keys())
        n_metrics = len(metric_names)
        correlation_matrix = np.zeros((n_metrics, n_metrics))

        for i, metric_i in enumerate(metric_names):
            for j, metric_j in enumerate(metric_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Pearson correlation
                    corr = np.corrcoef(
                        metric_data[metric_i],
                        metric_data[metric_j]
                    )[0, 1]
                    correlation_matrix[i, j] = corr

        # Identify trade-offs (negative correlations)
        tradeoffs = []
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                corr = correlation_matrix[i, j]
                if corr < -0.3:  # Moderate negative correlation
                    tradeoffs.append({
                        "metric_1": metric_names[i],
                        "metric_2": metric_names[j],
                        "correlation": float(corr),
                        "interpretation": "trade-off" if corr < -0.5 else "weak trade-off"
                    })

        # Identify synergies (positive correlations)
        synergies = []
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                corr = correlation_matrix[i, j]
                if corr > 0.5:
                    synergies.append({
                        "metric_1": metric_names[i],
                        "metric_2": metric_names[j],
                        "correlation": float(corr),
                        "interpretation": "strong synergy" if corr > 0.8 else "moderate synergy"
                    })

        # Compute metric statistics
        metric_stats = {}
        for metric_name, values in metric_data.items():
            metric_stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values))
            }

        return {
            "status": "analyzed",
            "num_experiments": min_samples,
            "correlation_matrix": correlation_matrix.tolist(),
            "metric_names": metric_names,
            "tradeoffs": tradeoffs,
            "synergies": synergies,
            "metric_statistics": metric_stats,
            "recommendations": self._generate_tradeoff_recommendations(
                tradeoffs, synergies, metric_stats
            )
        }

    def _generate_tradeoff_recommendations(
        self,
        tradeoffs: List[Dict[str, Any]],
        synergies: List[Dict[str, Any]],
        metric_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on trade-off analysis."""
        recommendations = []

        # Recommendation 1: Address major trade-offs
        if tradeoffs:
            strong_tradeoffs = [t for t in tradeoffs if t["correlation"] < -0.5]
            if strong_tradeoffs:
                for trade in strong_tradeoffs:
                    recommendations.append(
                        f"Strong trade-off detected between {trade['metric_1']} and {trade['metric_2']} "
                        f"(r={trade['correlation']:.2f}). Consider Pareto optimization to explore this trade-off."
                    )

        # Recommendation 2: Leverage synergies
        if synergies:
            for synergy in synergies[:2]:  # Top 2 synergies
                recommendations.append(
                    f"Synergy between {synergy['metric_1']} and {synergy['metric_2']} "
                    f"(r={synergy['correlation']:.2f}). Optimizing one may improve both."
                )

        # Recommendation 3: Low variance metrics
        low_variance = [
            metric for metric, stats in metric_stats.items()
            if stats["range"] < 0.05
        ]
        if low_variance:
            recommendations.append(
                f"Metrics {low_variance} show low variance. Consider exploring wider parameter ranges."
            )

        if not recommendations:
            recommendations.append(
                "No strong trade-offs or synergies detected. Multi-objective optimization may not be necessary."
            )

        return recommendations
