"""
Multi-Objective Optimization Schema for ARC.

Defines schemas for multi-objective experiment optimization including:
- Objective specifications (metrics + weights + direction)
- Pareto frontier tracking and dominance relationships
- Multi-objective experiment results
- Hypervolume computation for Pareto front quality

Key Components:
- ObjectiveSpec: Define optimization objectives (AUC, sensitivity, specificity)
- ParetoFront: Track non-dominated solutions
- MultiObjectiveMetrics: Extended metrics with Pareto information
- Dominance checking and hypervolume computation utilities

Clinical Considerations:
- Primary objective: Maximize AUC (overall performance)
- Constraint: Sensitivity ≥ 0.85 (minimize false negatives)
- Secondary: Balance sensitivity-specificity trade-off
- Hypervolume reference point: (0.5, 0.5, 0.5) for (AUC, sens, spec)

Author: ARC Team (Dev 1)
Created: 2025-11-19
Version: 1.0
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np


class OptimizationDirection(str, Enum):
    """
    Direction of optimization for an objective.

    - MAXIMIZE: Higher values are better (e.g., AUC, sensitivity, specificity)
    - MINIMIZE: Lower values are better (e.g., loss, inference time)
    """
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveSpec(BaseModel):
    """
    Specification for a single optimization objective.

    Example:
        {
            "metric_name": "auc",
            "weight": 0.5,
            "direction": "maximize",
            "constraint": None
        }

    For constrained optimization:
        {
            "metric_name": "sensitivity",
            "weight": 0.3,
            "direction": "maximize",
            "constraint": {"type": ">=", "value": 0.85}
        }
    """
    metric_name: str = Field(
        description="Name of metric to optimize (auc, sensitivity, specificity, etc.)"
    )

    weight: float = Field(
        ge=0.0, le=1.0,
        description="Objective weight in multi-objective optimization (0.0 to 1.0)"
    )

    direction: OptimizationDirection = Field(
        default=OptimizationDirection.MAXIMIZE,
        description="Optimization direction (maximize or minimize)"
    )

    constraint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional constraint (e.g., {'type': '>=', 'value': 0.85})"
    )

    @validator('metric_name')
    def validate_metric_name(cls, v):
        """Validate metric name is a recognized metric."""
        allowed_metrics = [
            "auc", "accuracy", "sensitivity", "specificity",
            "precision", "recall", "f1_score", "dice", "iou",
            "loss", "inference_time"
        ]
        if v not in allowed_metrics:
            raise ValueError(
                f"Metric '{v}' not in allowed list: {allowed_metrics}"
            )
        return v

    @validator('constraint')
    def validate_constraint(cls, v):
        """Validate constraint format."""
        if v is None:
            return v

        if 'type' not in v or 'value' not in v:
            raise ValueError("Constraint must have 'type' and 'value' keys")

        allowed_types = ['>=', '<=', '>', '<', '==']
        if v['type'] not in allowed_types:
            raise ValueError(
                f"Constraint type '{v['type']}' not in {allowed_types}"
            )

        return v


class ParetoSolution(BaseModel):
    """
    A single solution in the Pareto frontier.

    Contains experiment ID and its objective values.
    """
    experiment_id: str = Field(
        description="Experiment ID"
    )

    objective_values: Dict[str, float] = Field(
        description="Objective values {metric_name: value}"
    )

    dominated_by_count: int = Field(
        default=0, ge=0,
        description="Number of solutions that dominate this solution (0 = non-dominated)"
    )

    dominates_count: int = Field(
        default=0, ge=0,
        description="Number of solutions this solution dominates"
    )


class ParetoFront(BaseModel):
    """
    Pareto frontier: set of non-dominated solutions.

    A solution is non-dominated if no other solution is better in all objectives.

    Example:
        {
            "objectives": [
                {"metric_name": "auc", "weight": 0.5, "direction": "maximize"},
                {"metric_name": "sensitivity", "weight": 0.3, "direction": "maximize"}
            ],
            "solutions": [
                {
                    "experiment_id": "exp_001",
                    "objective_values": {"auc": 0.92, "sensitivity": 0.88},
                    "dominated_by_count": 0
                }
            ],
            "hypervolume": 0.756,
            "reference_point": {"auc": 0.5, "sensitivity": 0.5}
        }
    """
    objectives: List[ObjectiveSpec] = Field(
        min_items=2,
        description="List of objectives being optimized"
    )

    solutions: List[ParetoSolution] = Field(
        default=[],
        description="Non-dominated solutions in Pareto front"
    )

    hypervolume: Optional[float] = Field(
        default=None, ge=0.0,
        description="Hypervolume indicator (quality metric for Pareto front)"
    )

    reference_point: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reference point for hypervolume computation"
    )

    generation: int = Field(
        default=0, ge=0,
        description="Generation/cycle number when this front was computed"
    )

    @validator('objectives')
    def validate_at_least_two_objectives(cls, v):
        """Multi-objective optimization requires at least 2 objectives."""
        if len(v) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least 2 objectives"
            )
        return v

    @validator('solutions')
    def validate_all_non_dominated(cls, v):
        """All solutions in Pareto front should be non-dominated."""
        for solution in v:
            if solution.dominated_by_count > 0:
                raise ValueError(
                    f"Solution {solution.experiment_id} is dominated "
                    f"(dominated_by_count={solution.dominated_by_count}). "
                    f"Only non-dominated solutions belong in Pareto front."
                )
        return v


class MultiObjectiveMetrics(BaseModel):
    """
    Extended metrics with multi-objective information.

    Augments standard experiment metrics with Pareto frontier information.
    """
    experiment_id: str = Field(
        description="Experiment ID"
    )

    metrics: Dict[str, float] = Field(
        description="All experiment metrics {metric_name: value}"
    )

    is_pareto_optimal: bool = Field(
        default=False,
        description="True if this experiment is on the Pareto frontier"
    )

    dominated_by: List[str] = Field(
        default=[],
        description="List of experiment IDs that dominate this experiment"
    )

    dominates: List[str] = Field(
        default=[],
        description="List of experiment IDs this experiment dominates"
    )

    pareto_rank: int = Field(
        default=0, ge=0,
        description="Pareto rank (0 = Pareto optimal, 1 = dominated by front, etc.)"
    )

    crowding_distance: Optional[float] = Field(
        default=None, ge=0.0,
        description="Crowding distance in objective space (for diversity ranking)"
    )


def is_dominated(
    solution_a: Dict[str, float],
    solution_b: Dict[str, float],
    objectives: List[ObjectiveSpec]
) -> bool:
    """
    Check if solution_a is dominated by solution_b.

    Solution A is dominated by B if:
    - B is at least as good as A in all objectives
    - B is strictly better than A in at least one objective

    Args:
        solution_a: Objective values for solution A {metric: value}
        solution_b: Objective values for solution B {metric: value}
        objectives: List of objective specifications

    Returns:
        True if A is dominated by B, False otherwise

    Example:
        >>> objectives = [
        ...     ObjectiveSpec(metric_name="auc", weight=0.5, direction="maximize"),
        ...     ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction="maximize")
        ... ]
        >>> a = {"auc": 0.85, "sensitivity": 0.80}
        >>> b = {"auc": 0.90, "sensitivity": 0.85}
        >>> is_dominated(a, b, objectives)
        True
    """
    better_in_all = True
    strictly_better_in_one = False

    for obj in objectives:
        metric = obj.metric_name
        val_a = solution_a.get(metric)
        val_b = solution_b.get(metric)

        if val_a is None or val_b is None:
            continue

        if obj.direction == OptimizationDirection.MAXIMIZE:
            # B should be >= A in all objectives
            if val_b < val_a:
                better_in_all = False
            if val_b > val_a:
                strictly_better_in_one = True
        else:  # MINIMIZE
            # B should be <= A in all objectives
            if val_b > val_a:
                better_in_all = False
            if val_b < val_a:
                strictly_better_in_one = True

    return better_in_all and strictly_better_in_one


def compute_pareto_frontier(
    solutions: List[Dict[str, Any]],
    objectives: List[ObjectiveSpec],
    return_all_ranks: bool = False
) -> List[Dict[str, Any]]:
    """
    Compute Pareto frontier from a set of solutions.

    Args:
        solutions: List of solutions, each with "experiment_id" and "metrics" dict
        objectives: List of objective specifications
        return_all_ranks: If True, return all solutions with pareto_rank assigned

    Returns:
        List of non-dominated solutions (or all solutions with ranks if return_all_ranks=True)

    Example:
        >>> solutions = [
        ...     {"experiment_id": "exp_001", "metrics": {"auc": 0.92, "sensitivity": 0.85}},
        ...     {"experiment_id": "exp_002", "metrics": {"auc": 0.88, "sensitivity": 0.90}},
        ...     {"experiment_id": "exp_003", "metrics": {"auc": 0.85, "sensitivity": 0.80}}
        ... ]
        >>> objectives = [
        ...     ObjectiveSpec(metric_name="auc", weight=0.5, direction="maximize"),
        ...     ObjectiveSpec(metric_name="sensitivity", weight=0.5, direction="maximize")
        ... ]
        >>> frontier = compute_pareto_frontier(solutions, objectives)
        >>> len(frontier)  # exp_001 and exp_002 are non-dominated
        2
    """
    if not solutions:
        return []

    # Compute dominance relationships
    n = len(solutions)
    dominated_by = [[] for _ in range(n)]
    dominates = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            sol_i_metrics = solutions[i].get("metrics", {})
            sol_j_metrics = solutions[j].get("metrics", {})

            if is_dominated(sol_i_metrics, sol_j_metrics, objectives):
                dominated_by[i].append(solutions[j]["experiment_id"])
                dominates[j].append(solutions[i]["experiment_id"])

    # Compute Pareto ranks (0 = non-dominated front)
    pareto_ranks = [-1] * n
    current_rank = 0
    remaining = set(range(n))

    while remaining:
        # Find all non-dominated solutions in remaining set
        current_front = []
        for i in remaining:
            # Check if dominated by any solution in remaining set
            is_dominated_in_remaining = any(
                solutions[j]["experiment_id"] in dominated_by[i]
                for j in remaining if j != i
            )
            if not is_dominated_in_remaining:
                current_front.append(i)

        # Assign rank to current front
        for i in current_front:
            pareto_ranks[i] = current_rank
            remaining.remove(i)

        current_rank += 1

    # Add Pareto information to solutions
    enriched_solutions = []
    for i, solution in enumerate(solutions):
        enriched = solution.copy()
        enriched["is_pareto_optimal"] = (pareto_ranks[i] == 0)
        enriched["pareto_rank"] = pareto_ranks[i]
        enriched["dominated_by"] = dominated_by[i]
        enriched["dominates"] = dominates[i]
        enriched_solutions.append(enriched)

    if return_all_ranks:
        return enriched_solutions
    else:
        # Return only Pareto optimal solutions (rank 0)
        return [s for s in enriched_solutions if s["pareto_rank"] == 0]


def compute_hypervolume(
    pareto_solutions: List[Dict[str, float]],
    objectives: List[ObjectiveSpec],
    reference_point: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute hypervolume indicator for Pareto front.

    Hypervolume measures the volume of objective space dominated by the Pareto front.
    Higher hypervolume = better Pareto front.

    Args:
        pareto_solutions: List of objective value dicts {metric: value}
        objectives: List of objective specifications
        reference_point: Reference point for hypervolume (default: all 0.5)

    Returns:
        Hypervolume value (0.0 to 1.0 for normalized objectives)

    Note:
        This is a simplified 2D/3D hypervolume computation.
        For high-dimensional objectives (>3), consider using pygmo or similar.
    """
    if not pareto_solutions:
        return 0.0

    # Set default reference point
    if reference_point is None:
        reference_point = {obj.metric_name: 0.5 for obj in objectives}

    # Extract objective values
    n_objectives = len(objectives)

    if n_objectives == 2:
        # 2D hypervolume (area)
        return _compute_hypervolume_2d(pareto_solutions, objectives, reference_point)
    elif n_objectives == 3:
        # 3D hypervolume (volume)
        return _compute_hypervolume_3d(pareto_solutions, objectives, reference_point)
    else:
        # High-dimensional: use Monte Carlo approximation
        return _compute_hypervolume_mc(pareto_solutions, objectives, reference_point)


def _compute_hypervolume_2d(
    solutions: List[Dict[str, float]],
    objectives: List[ObjectiveSpec],
    reference_point: Dict[str, float]
) -> float:
    """Compute 2D hypervolume using sweep algorithm."""
    obj1_name = objectives[0].metric_name
    obj2_name = objectives[1].metric_name

    ref1 = reference_point[obj1_name]
    ref2 = reference_point[obj2_name]

    # Extract points
    points = []
    for sol in solutions:
        val1 = sol.get(obj1_name, ref1)
        val2 = sol.get(obj2_name, ref2)

        # Handle minimization objectives (flip values)
        if objectives[0].direction == OptimizationDirection.MINIMIZE:
            val1 = 1.0 - val1
            ref1 = 1.0 - ref1
        if objectives[1].direction == OptimizationDirection.MINIMIZE:
            val2 = 1.0 - val2
            ref2 = 1.0 - ref2

        points.append((val1, val2))

    # Sort by first objective (descending)
    points = sorted(points, key=lambda p: p[0], reverse=True)

    # Sweep algorithm
    hypervolume = 0.0
    prev_y = ref2

    for x, y in points:
        if y > prev_y:
            width = x - ref1
            height = y - prev_y
            hypervolume += width * height
            prev_y = y

    return hypervolume


def _compute_hypervolume_3d(
    solutions: List[Dict[str, float]],
    objectives: List[ObjectiveSpec],
    reference_point: Dict[str, float]
) -> float:
    """Compute 3D hypervolume using layer-by-layer approach."""
    # Simplified 3D hypervolume (Monte Carlo approximation for now)
    return _compute_hypervolume_mc(solutions, objectives, reference_point, n_samples=10000)


def _compute_hypervolume_mc(
    solutions: List[Dict[str, float]],
    objectives: List[ObjectiveSpec],
    reference_point: Dict[str, float],
    n_samples: int = 10000
) -> float:
    """
    Compute hypervolume using Monte Carlo sampling.

    Sample random points in objective space and count how many are dominated
    by the Pareto front.
    """
    # Extract bounds
    obj_names = [obj.metric_name for obj in objectives]

    # Find min/max values for each objective
    bounds = {}
    for obj_name in obj_names:
        values = [sol.get(obj_name, 0.5) for sol in solutions]
        ref_val = reference_point.get(obj_name, 0.5)
        bounds[obj_name] = (min(ref_val, min(values)), max(values))

    # Monte Carlo sampling
    dominated_count = 0

    for _ in range(n_samples):
        # Sample random point
        sample_point = {}
        for obj_name in obj_names:
            low, high = bounds[obj_name]
            sample_point[obj_name] = np.random.uniform(low, high)

        # Check if dominated by any Pareto solution
        for solution in solutions:
            if _dominates_point(solution, sample_point, objectives, reference_point):
                dominated_count += 1
                break

    # Hypervolume = fraction of dominated samples × total volume
    volume = 1.0
    for obj_name in obj_names:
        low, high = bounds[obj_name]
        volume *= (high - low)

    hypervolume = (dominated_count / n_samples) * volume
    return hypervolume


def _dominates_point(
    solution: Dict[str, float],
    point: Dict[str, float],
    objectives: List[ObjectiveSpec],
    reference_point: Dict[str, float]
) -> bool:
    """Check if solution dominates a sampled point."""
    for obj in objectives:
        metric = obj.metric_name
        sol_val = solution.get(metric, reference_point.get(metric, 0.5))
        point_val = point.get(metric, reference_point.get(metric, 0.5))

        if obj.direction == OptimizationDirection.MAXIMIZE:
            if sol_val < point_val:
                return False
        else:  # MINIMIZE
            if sol_val > point_val:
                return False

    return True


def validate_multi_objective_safety(
    pareto_front: ParetoFront,
    min_sensitivity: float = 0.85
) -> Tuple[bool, str]:
    """
    Validate multi-objective Pareto front for clinical safety.

    Ensures all Pareto-optimal solutions meet clinical safety constraints.

    Args:
        pareto_front: Pareto frontier to validate
        min_sensitivity: Minimum required sensitivity (default: 0.85)

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: All solutions must meet sensitivity constraint
    sensitivity_violations = []
    for solution in pareto_front.solutions:
        sens = solution.objective_values.get("sensitivity")
        if sens is not None and sens < min_sensitivity:
            sensitivity_violations.append(
                f"{solution.experiment_id}: sensitivity={sens:.3f}"
            )

    if sensitivity_violations:
        return False, (
            f"Pareto front violates sensitivity constraint (≥ {min_sensitivity}). "
            f"Violations: {sensitivity_violations}"
        )

    # Check 2: At least one objective should be AUC
    has_auc = any(obj.metric_name == "auc" for obj in pareto_front.objectives)
    if not has_auc:
        return False, (
            "Multi-objective optimization should include AUC as primary metric"
        )

    # Check 3: Pareto front should not be empty
    if not pareto_front.solutions:
        return False, "Pareto front is empty (no non-dominated solutions found)"

    # All checks passed
    return True, ""


# Example factory methods
class MultiObjectiveConfig:
    """Factory methods for common multi-objective configurations."""

    @staticmethod
    def auc_sensitivity_tradeoff() -> List[ObjectiveSpec]:
        """
        AUC vs Sensitivity trade-off.

        Explores trade-off between overall performance (AUC) and
        recall/sensitivity (minimize false negatives).
        """
        return [
            ObjectiveSpec(
                metric_name="auc",
                weight=0.6,
                direction=OptimizationDirection.MAXIMIZE
            ),
            ObjectiveSpec(
                metric_name="sensitivity",
                weight=0.4,
                direction=OptimizationDirection.MAXIMIZE,
                constraint={"type": ">=", "value": 0.85}
            )
        ]

    @staticmethod
    def balanced_classification() -> List[ObjectiveSpec]:
        """
        Balanced AUC, Sensitivity, Specificity.

        Optimizes all three metrics with equal weighting.
        """
        return [
            ObjectiveSpec(
                metric_name="auc",
                weight=0.4,
                direction=OptimizationDirection.MAXIMIZE
            ),
            ObjectiveSpec(
                metric_name="sensitivity",
                weight=0.3,
                direction=OptimizationDirection.MAXIMIZE,
                constraint={"type": ">=", "value": 0.85}
            ),
            ObjectiveSpec(
                metric_name="specificity",
                weight=0.3,
                direction=OptimizationDirection.MAXIMIZE
            )
        ]

    @staticmethod
    def auc_constrained_sensitivity() -> List[ObjectiveSpec]:
        """
        Maximize AUC with hard sensitivity constraint.

        Primary objective: AUC
        Constraint: Sensitivity ≥ 0.85
        """
        return [
            ObjectiveSpec(
                metric_name="auc",
                weight=1.0,
                direction=OptimizationDirection.MAXIMIZE
            ),
            ObjectiveSpec(
                metric_name="sensitivity",
                weight=0.0,  # Constraint only, not optimized
                direction=OptimizationDirection.MAXIMIZE,
                constraint={"type": ">=", "value": 0.85}
            )
        ]
