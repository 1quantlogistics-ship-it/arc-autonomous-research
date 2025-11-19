"""
End-to-End Tests for Multi-Objective Optimization (Phase E, Week 3).

Tests the complete multi-objective optimization pipeline:
- ObjectiveSpec and Pareto frontier computation
- Historian Pareto tracking and evolution
- World model multi-objective predictions
- Config generator integration

Author: ARC Team (Dev 1)
Created: 2025-11-19
Version: 1.0
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Import multi-objective schemas
from schemas.multi_objective import (
    ObjectiveSpec, ParetoFront, ParetoSolution, MultiObjectiveMetrics,
    OptimizationDirection, compute_pareto_frontier, compute_hypervolume,
    is_dominated, validate_multi_objective_safety, MultiObjectiveConfig
)


class TestObjectiveSpec:
    """Test ObjectiveSpec validation and configuration."""

    def test_basic_objective_creation(self):
        """Test creating a basic objective specification."""
        obj = ObjectiveSpec(
            metric_name="auc",
            weight=0.5,
            direction=OptimizationDirection.MAXIMIZE
        )

        assert obj.metric_name == "auc"
        assert obj.weight == 0.5
        assert obj.direction == OptimizationDirection.MAXIMIZE
        assert obj.constraint is None

    def test_constrained_objective(self):
        """Test objective with constraint."""
        obj = ObjectiveSpec(
            metric_name="sensitivity",
            weight=0.3,
            direction=OptimizationDirection.MAXIMIZE,
            constraint={"type": ">=", "value": 0.85}
        )

        assert obj.constraint["type"] == ">="
        assert obj.constraint["value"] == 0.85

    def test_invalid_metric_name(self):
        """Test that invalid metric names are rejected."""
        with pytest.raises(ValueError, match="not in allowed list"):
            ObjectiveSpec(
                metric_name="invalid_metric",
                weight=0.5,
                direction=OptimizationDirection.MAXIMIZE
            )

    def test_invalid_weight(self):
        """Test that weights outside [0, 1] are rejected."""
        with pytest.raises(ValueError):
            ObjectiveSpec(
                metric_name="auc",
                weight=1.5,  # Invalid: > 1.0
                direction=OptimizationDirection.MAXIMIZE
            )

    def test_invalid_constraint_format(self):
        """Test that invalid constraint formats are rejected."""
        with pytest.raises(ValueError, match="must have 'type' and 'value'"):
            ObjectiveSpec(
                metric_name="auc",
                weight=0.5,
                direction=OptimizationDirection.MAXIMIZE,
                constraint={"invalid": "format"}
            )


class TestParetoComputation:
    """Test Pareto frontier computation functions."""

    def test_dominance_basic(self):
        """Test basic dominance relationship."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        # Solution B dominates A (better in both objectives)
        sol_a = {"auc": 0.85, "sensitivity": 0.80}
        sol_b = {"auc": 0.90, "sensitivity": 0.85}

        assert is_dominated(sol_a, sol_b, objectives) is True
        assert is_dominated(sol_b, sol_a, objectives) is False

    def test_non_dominated_solutions(self):
        """Test non-dominated solutions (trade-off)."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        # Neither dominates the other (trade-off)
        sol_a = {"auc": 0.90, "sensitivity": 0.80}  # High AUC, low sensitivity
        sol_b = {"auc": 0.85, "sensitivity": 0.90}  # Low AUC, high sensitivity

        assert is_dominated(sol_a, sol_b, objectives) is False
        assert is_dominated(sol_b, sol_a, objectives) is False

    def test_pareto_frontier_extraction(self):
        """Test extracting Pareto frontier from solutions."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        solutions = [
            {"experiment_id": "exp_001", "metrics": {"auc": 0.92, "sensitivity": 0.85}},  # Pareto-optimal
            {"experiment_id": "exp_002", "metrics": {"auc": 0.88, "sensitivity": 0.90}},  # Pareto-optimal
            {"experiment_id": "exp_003", "metrics": {"auc": 0.85, "sensitivity": 0.80}},  # Dominated
            {"experiment_id": "exp_004", "metrics": {"auc": 0.80, "sensitivity": 0.75}},  # Dominated
        ]

        pareto_solutions = compute_pareto_frontier(solutions, objectives, return_all_ranks=False)

        # Should have 2 Pareto-optimal solutions
        assert len(pareto_solutions) == 2

        pareto_ids = {sol["experiment_id"] for sol in pareto_solutions}
        assert "exp_001" in pareto_ids
        assert "exp_002" in pareto_ids
        assert "exp_003" not in pareto_ids
        assert "exp_004" not in pareto_ids

    def test_pareto_ranking(self):
        """Test Pareto ranking computation."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        solutions = [
            {"experiment_id": "exp_001", "metrics": {"auc": 0.92, "sensitivity": 0.85}},  # Rank 0
            {"experiment_id": "exp_002", "metrics": {"auc": 0.88, "sensitivity": 0.90}},  # Rank 0
            {"experiment_id": "exp_003", "metrics": {"auc": 0.85, "sensitivity": 0.80}},  # Rank 1
        ]

        all_ranked = compute_pareto_frontier(solutions, objectives, return_all_ranks=True)

        ranks = {sol["experiment_id"]: sol["pareto_rank"] for sol in all_ranked}
        assert ranks["exp_001"] == 0  # Pareto-optimal
        assert ranks["exp_002"] == 0  # Pareto-optimal
        assert ranks["exp_003"] == 1  # Dominated

    def test_hypervolume_2d(self):
        """Test 2D hypervolume computation."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        pareto_solutions = [
            {"auc": 0.90, "sensitivity": 0.85},
            {"auc": 0.85, "sensitivity": 0.90}
        ]

        reference_point = {"auc": 0.5, "sensitivity": 0.5}
        hypervolume = compute_hypervolume(pareto_solutions, objectives, reference_point)

        # Hypervolume should be positive
        assert hypervolume > 0.0

        # Better Pareto front should have higher hypervolume
        better_solutions = [
            {"auc": 0.95, "sensitivity": 0.90},
            {"auc": 0.90, "sensitivity": 0.95}
        ]
        better_hypervolume = compute_hypervolume(better_solutions, objectives, reference_point)
        assert better_hypervolume > hypervolume


class TestParetoFront:
    """Test ParetoFront schema validation."""

    def test_valid_pareto_front(self):
        """Test creating a valid Pareto front."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        solutions = [
            ParetoSolution(
                experiment_id="exp_001",
                objective_values={"auc": 0.92, "sensitivity": 0.85},
                dominated_by_count=0,
                dominates_count=2
            ),
            ParetoSolution(
                experiment_id="exp_002",
                objective_values={"auc": 0.88, "sensitivity": 0.90},
                dominated_by_count=0,
                dominates_count=1
            )
        ]

        pareto_front = ParetoFront(
            objectives=objectives,
            solutions=solutions,
            hypervolume=0.75,
            reference_point={"auc": 0.5, "sensitivity": 0.5},
            generation=5
        )

        assert len(pareto_front.solutions) == 2
        assert pareto_front.hypervolume == 0.75
        assert pareto_front.generation == 5

    def test_single_objective_rejected(self):
        """Test that single-objective fronts are rejected."""
        with pytest.raises(ValueError, match="at least 2 objectives"):
            ParetoFront(
                objectives=[
                    ObjectiveSpec(metric_name="auc", weight=1.0, direction=OptimizationDirection.MAXIMIZE)
                ],
                solutions=[]
            )

    def test_dominated_solution_rejected(self):
        """Test that dominated solutions are rejected from Pareto front."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        dominated_solution = ParetoSolution(
            experiment_id="exp_001",
            objective_values={"auc": 0.85, "sensitivity": 0.80},
            dominated_by_count=1,  # This solution is dominated!
            dominates_count=0
        )

        with pytest.raises(ValueError, match="is dominated"):
            ParetoFront(
                objectives=objectives,
                solutions=[dominated_solution]
            )


class TestClinicalSafety:
    """Test clinical safety validation for multi-objective optimization."""

    def test_valid_pareto_front(self):
        """Test Pareto front that meets clinical safety requirements."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.6, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(
                metric_name="sensitivity",
                weight=0.4,
                direction=OptimizationDirection.MAXIMIZE,
                constraint={"type": ">=", "value": 0.85}
            )
        ]

        solutions = [
            ParetoSolution(
                experiment_id="exp_001",
                objective_values={"auc": 0.92, "sensitivity": 0.88},  # Meets constraint
                dominated_by_count=0,
                dominates_count=1
            )
        ]

        pareto_front = ParetoFront(
            objectives=objectives,
            solutions=solutions,
            hypervolume=0.75
        )

        is_valid, error_msg = validate_multi_objective_safety(pareto_front, min_sensitivity=0.85)
        assert is_valid is True
        assert error_msg == ""

    def test_sensitivity_constraint_violation(self):
        """Test Pareto front that violates sensitivity constraint."""
        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.8, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.2, direction=OptimizationDirection.MAXIMIZE)
        ]

        solutions = [
            ParetoSolution(
                experiment_id="exp_001",
                objective_values={"auc": 0.95, "sensitivity": 0.80},  # Violates constraint (< 0.85)
                dominated_by_count=0,
                dominates_count=0
            )
        ]

        pareto_front = ParetoFront(
            objectives=objectives,
            solutions=solutions,
            hypervolume=0.75
        )

        is_valid, error_msg = validate_multi_objective_safety(pareto_front, min_sensitivity=0.85)
        assert is_valid is False
        assert "sensitivity constraint" in error_msg.lower()

    def test_missing_auc_objective(self):
        """Test that multi-objective optimization without AUC is flagged."""
        objectives = [
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="specificity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        pareto_front = ParetoFront(
            objectives=objectives,
            solutions=[],
            hypervolume=0.0
        )

        is_valid, error_msg = validate_multi_objective_safety(pareto_front)
        assert is_valid is False
        assert "auc" in error_msg.lower()


class TestMultiObjectiveConfig:
    """Test factory methods for common multi-objective configurations."""

    def test_auc_sensitivity_tradeoff(self):
        """Test AUC vs Sensitivity trade-off configuration."""
        objectives = MultiObjectiveConfig.auc_sensitivity_tradeoff()

        assert len(objectives) == 2
        assert objectives[0].metric_name == "auc"
        assert objectives[1].metric_name == "sensitivity"
        assert objectives[1].constraint is not None
        assert objectives[1].constraint["value"] == 0.85

    def test_balanced_classification(self):
        """Test balanced AUC, Sensitivity, Specificity configuration."""
        objectives = MultiObjectiveConfig.balanced_classification()

        assert len(objectives) == 3
        metric_names = {obj.metric_name for obj in objectives}
        assert metric_names == {"auc", "sensitivity", "specificity"}

        # Check weights sum to 1.0
        total_weight = sum(obj.weight for obj in objectives)
        assert abs(total_weight - 1.0) < 0.01

    def test_auc_constrained_sensitivity(self):
        """Test AUC maximization with sensitivity constraint."""
        objectives = MultiObjectiveConfig.auc_constrained_sensitivity()

        assert len(objectives) == 2
        auc_obj = next(obj for obj in objectives if obj.metric_name == "auc")
        sens_obj = next(obj for obj in objectives if obj.metric_name == "sensitivity")

        assert auc_obj.weight == 1.0  # Primary objective
        assert sens_obj.weight == 0.0  # Constraint only
        assert sens_obj.constraint is not None


class TestHistorianParetoTracking:
    """Test Historian agent Pareto tracking methods."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create temporary memory directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_training_history(self, temp_memory_dir):
        """Create mock training history for testing."""
        history = {
            "total_experiments": 5,
            "total_cycles": 2,
            "experiments": [
                {
                    "experiment_id": "exp_001",
                    "status": "completed",
                    "metrics": {"auc": 0.92, "sensitivity": 0.85, "specificity": 0.89}
                },
                {
                    "experiment_id": "exp_002",
                    "status": "completed",
                    "metrics": {"auc": 0.88, "sensitivity": 0.90, "specificity": 0.87}
                },
                {
                    "experiment_id": "exp_003",
                    "status": "completed",
                    "metrics": {"auc": 0.85, "sensitivity": 0.80, "specificity": 0.91}
                },
                {
                    "experiment_id": "exp_004",
                    "status": "failed",
                    "metrics": None
                },
                {
                    "experiment_id": "exp_005",
                    "status": "completed",
                    "metrics": {"auc": 0.80, "sensitivity": 0.75, "specificity": 0.85}
                }
            ],
            "cycles": []
        }

        history_path = Path(temp_memory_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)

        return history_path

    def test_pareto_frontier_extraction(self, temp_memory_dir, mock_training_history):
        """Test extracting Pareto frontier from training history."""
        from agents.historian_agent import HistorianAgent

        historian = HistorianAgent(memory_path=temp_memory_dir)

        objectives = [
            ObjectiveSpec(metric_name="auc", weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction=OptimizationDirection.MAXIMIZE)
        ]

        result = historian.get_pareto_frontier(objectives, filter_completed=True)

        assert result["pareto_front"] is not None
        assert result["num_total"] == 4  # 4 completed experiments
        assert result["num_pareto_optimal"] >= 1  # At least one non-dominated
        assert result["hypervolume"] > 0.0


class TestWorldModelMultiObjective:
    """Test world model multi-objective prediction methods."""

    def test_predict_multi_objective(self):
        """Test multi-objective prediction."""
        from tools.world_model import WorldModel

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorldModel(model_path=tmpdir)

            config = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "model": "resnet50"
            }

            predictions = wm.predict_multi_objective(
                config,
                objectives=["auc", "sensitivity", "specificity"]
            )

            assert "auc" in predictions
            assert "sensitivity" in predictions
            assert "specificity" in predictions

            # Check each prediction has required fields
            for metric, pred in predictions.items():
                assert hasattr(pred, "predicted_metric")
                assert hasattr(pred, "confidence")
                assert hasattr(pred, "uncertainty")

    def test_suggest_pareto_optimal_experiments(self):
        """Test Pareto-optimal experiment suggestion."""
        from tools.world_model import WorldModel

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorldModel(model_path=tmpdir)

            candidate_configs = [
                {"learning_rate": 0.001, "batch_size": 32},
                {"learning_rate": 0.0001, "batch_size": 64},
                {"learning_rate": 0.01, "batch_size": 16}
            ]

            suggestions = wm.suggest_pareto_optimal_experiments(
                candidate_configs,
                objectives=["auc", "sensitivity"],
                acquisition="hypervolume"
            )

            assert len(suggestions) == len(candidate_configs)

            # Check suggestions are sorted by score
            scores = [s["pareto_score"] for s in suggestions]
            assert scores == sorted(scores, reverse=True)

            # Check each suggestion has required fields
            for suggestion in suggestions:
                assert "config" in suggestion
                assert "pareto_score" in suggestion
                assert "predicted_objectives" in suggestion
                assert "auc" in suggestion["predicted_objectives"]
                assert "sensitivity" in suggestion["predicted_objectives"]


class TestConfigGeneratorIntegration:
    """Test config generator multi-objective integration."""

    def test_objectives_translation(self):
        """Test translating objectives to training config format."""
        from config.experiment_config_generator import ExperimentConfigGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ExperimentConfigGenerator(experiments_dir=tmpdir, memory_path=tmpdir)

            objectives_list = [
                {"metric_name": "auc", "weight": 0.5, "direction": "maximize"},
                {
                    "metric_name": "sensitivity",
                    "weight": 0.3,
                    "direction": "maximize",
                    "constraint": {"type": ">=", "value": 0.85}
                },
                {"metric_name": "specificity", "weight": 0.2, "direction": "maximize"}
            ]

            translation = generator._translate_objectives(objectives_list)

            assert translation["multi_objective"] is True
            assert translation["optimization_metrics"] == ["auc", "sensitivity", "specificity"]
            assert translation["metric_weights"]["auc"] == 0.5
            assert translation["constraints"]["sensitivity"]["value"] == 0.85
            assert translation["checkpoint_metric"] == "auc"  # Highest weight
            assert translation["pareto_tracking"] is True

    def test_single_objective_passthrough(self):
        """Test that single objective is not treated as multi-objective."""
        from config.experiment_config_generator import ExperimentConfigGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ExperimentConfigGenerator(experiments_dir=tmpdir, memory_path=tmpdir)

            objectives_list = [
                {"metric_name": "auc", "weight": 1.0, "direction": "maximize"}
            ]

            translation = generator._translate_objectives(objectives_list)

            assert translation["multi_objective"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
