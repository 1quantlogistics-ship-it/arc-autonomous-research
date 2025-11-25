"""
High-dimensional Pareto front visualization for ARC.

Supports >3 objectives using parallel coordinates and dimensionality reduction.
Designed for multi-objective optimization in medical imaging experiments.

Key Features:
- ParetoFront class for tracking dominated/non-dominated solutions
- Parallel coordinates plot for >3 objectives
- Dimensionality reduction projections (t-SNE, PCA, UMAP)
- Hypervolume indicator computation
- Interactive objective weight explorer

Author: ARC Team (Dev 2)
Created: 2025-11-24
Version: 1.0 (Phase F)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - visualization disabled")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - t-SNE/PCA disabled")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not available - UMAP projection disabled")


@dataclass
class ParetoPoint:
    """Single point on Pareto front."""
    objectives: Dict[str, float]
    config: Dict[str, Any]  # Hyperparameters that produced this point
    dominated: bool = False
    experiment_id: Optional[str] = None


class ParetoFront:
    """
    Pareto front with high-dimensional visualization support.

    Tracks solutions from multi-objective optimization and provides
    methods for visualization and analysis.
    """

    def __init__(self, objective_names: List[str],
                 minimize: Optional[List[bool]] = None):
        """
        Args:
            objective_names: Names of objectives (e.g., ['loss', 'latency', 'memory'])
            minimize: Whether each objective should be minimized (default: True for all)
        """
        self.objective_names = objective_names
        self.minimize = minimize or [True] * len(objective_names)
        self.points: List[ParetoPoint] = []

        if len(self.objective_names) != len(self.minimize):
            raise ValueError(
                f"objective_names ({len(objective_names)}) and minimize ({len(minimize)}) "
                f"must have the same length"
            )

    def add_point(self, objectives: Dict[str, float], config: Dict[str, Any],
                  experiment_id: Optional[str] = None) -> ParetoPoint:
        """
        Add a point and update domination status.

        Args:
            objectives: Objective values keyed by name
            config: Configuration that produced this result
            experiment_id: Optional experiment identifier

        Returns:
            The added ParetoPoint
        """
        # Validate objectives
        for name in self.objective_names:
            if name not in objectives:
                raise ValueError(f"Missing objective: {name}")

        point = ParetoPoint(
            objectives=objectives,
            config=config,
            experiment_id=experiment_id
        )
        self.points.append(point)
        self._update_domination()
        return point

    def _dominates(self, p1: ParetoPoint, p2: ParetoPoint) -> bool:
        """Check if p1 dominates p2."""
        dominated = False
        for i, name in enumerate(self.objective_names):
            v1, v2 = p1.objectives[name], p2.objectives[name]

            if self.minimize[i]:
                if v1 > v2:
                    return False
                if v1 < v2:
                    dominated = True
            else:
                if v1 < v2:
                    return False
                if v1 > v2:
                    dominated = True

        return dominated

    def _update_domination(self):
        """Update domination status of all points."""
        for p in self.points:
            p.dominated = False

        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i != j and self._dominates(p2, p1):
                    p1.dominated = True
                    break

    @property
    def pareto_optimal(self) -> List[ParetoPoint]:
        """Get non-dominated points."""
        return [p for p in self.points if not p.dominated]

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objective_names)

    def to_array(self, pareto_only: bool = True) -> np.ndarray:
        """
        Convert to numpy array.

        Args:
            pareto_only: Only include Pareto-optimal points

        Returns:
            Array of shape (n_points, n_objectives)
        """
        points = self.pareto_optimal if pareto_only else self.points
        if not points:
            return np.array([]).reshape(0, self.num_objectives)
        return np.array([
            [p.objectives[name] for name in self.objective_names]
            for p in points
        ])

    def compute_hypervolume(self, reference: Optional[Dict[str, float]] = None,
                           n_samples: int = 10000) -> float:
        """
        Compute hypervolume indicator using Monte Carlo estimation.

        For mixed min/max objectives, we normalize to a minimization problem.

        Args:
            reference: Reference point (default: worst values + margin)
            n_samples: Number of Monte Carlo samples

        Returns:
            Hypervolume indicator value (always non-negative)
        """
        if not self.pareto_optimal:
            return 0.0

        points = self.to_array(pareto_only=True)
        n_obj = len(self.objective_names)

        # Normalize: flip sign for maximization objectives so all are minimized
        normalized_points = points.copy()
        for i in range(n_obj):
            if not self.minimize[i]:
                normalized_points[:, i] = -normalized_points[:, i]

        # Compute reference point if not provided
        if reference is None:
            reference = {}
            for i, name in enumerate(self.objective_names):
                values = [p.objectives[name] for p in self.points]
                if self.minimize[i]:
                    # For minimization: reference is worst (max) + margin
                    reference[name] = max(values) * 1.1 if max(values) > 0 else max(values) - abs(max(values)) * 0.1
                else:
                    # For maximization: reference is worst (min) - margin
                    reference[name] = min(values) * 0.9 if min(values) > 0 else min(values) - abs(min(values)) * 0.1

        # Normalize reference point too
        ref = np.array([reference[name] for name in self.objective_names])
        normalized_ref = ref.copy()
        for i in range(n_obj):
            if not self.minimize[i]:
                normalized_ref[i] = -normalized_ref[i]

        # Ensure ref is actually worse than all points (in minimization sense)
        for i in range(n_obj):
            normalized_ref[i] = max(normalized_ref[i], normalized_points[:, i].max() * 1.01)

        # Monte Carlo hypervolume estimation in normalized space
        mins = normalized_points.min(axis=0)

        # Ensure mins < ref for valid sampling
        for i in range(n_obj):
            if mins[i] >= normalized_ref[i]:
                mins[i] = normalized_ref[i] - abs(normalized_ref[i]) * 0.01 - 1e-9

        samples = np.random.uniform(mins, normalized_ref, size=(n_samples, n_obj))

        dominated_count = 0
        for sample in samples:
            for point in normalized_points:
                # In normalized space, all objectives are minimized
                if all(point[i] <= sample[i] for i in range(n_obj)):
                    dominated_count += 1
                    break

        volume = np.prod(normalized_ref - mins)
        return abs(volume) * (dominated_count / n_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'objective_names': self.objective_names,
            'minimize': self.minimize,
            'points': [
                {
                    'objectives': p.objectives,
                    'config': p.config,
                    'dominated': p.dominated,
                    'experiment_id': p.experiment_id
                }
                for p in self.points
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParetoFront':
        """Deserialize from dictionary."""
        front = cls(data['objective_names'], data.get('minimize'))
        for p in data['points']:
            point = ParetoPoint(
                objectives=p['objectives'],
                config=p['config'],
                dominated=p.get('dominated', False),
                experiment_id=p.get('experiment_id')
            )
            front.points.append(point)
        front._update_domination()
        return front


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_parallel_coordinates(
    front: ParetoFront,
    title: str = "Pareto Front - Parallel Coordinates",
    color_by: Optional[str] = None,
    show_dominated: bool = False,
    height: int = 500
) -> Optional[Any]:
    """
    Create parallel coordinates plot for >3 objectives.

    Args:
        front: ParetoFront object
        title: Plot title
        color_by: Objective name to use for coloring
        show_dominated: Include dominated points
        height: Plot height in pixels

    Returns:
        Plotly Figure or None if plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly required for visualization")
        return None

    points = front.points if show_dominated else front.pareto_optimal
    if not points:
        logger.warning("No points to visualize")
        return None

    # Build dimensions
    dimensions = []
    for name in front.objective_names:
        values = [p.objectives[name] for p in points]
        dimensions.append(dict(
            label=name,
            values=values,
            range=[min(values), max(values)]
        ))

    # Color
    if color_by and color_by in front.objective_names:
        color_values = [p.objectives[color_by] for p in points]
        colorbar_title = color_by
    else:
        color_values = list(range(len(points)))
        colorbar_title = "Index"

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title=title,
        height=height
    )

    return fig


def create_projection_plot(
    front: ParetoFront,
    method: str = 'pca',
    title: str = "Pareto Front - 2D Projection",
    color_by: Optional[str] = None,
    height: int = 500
) -> Optional[Any]:
    """
    Create 2D projection of high-dimensional Pareto front.

    Args:
        front: ParetoFront object
        method: 'tsne', 'umap', or 'pca'
        title: Plot title
        color_by: Objective to use for coloring
        height: Plot height in pixels

    Returns:
        Plotly Figure or None if dependencies unavailable
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly required for visualization")
        return None

    if method in ['tsne', 'pca'] and not SKLEARN_AVAILABLE:
        logger.error("scikit-learn required for t-SNE/PCA")
        return None

    if method == 'umap' and not UMAP_AVAILABLE:
        logger.error("umap-learn required for UMAP")
        return None

    points = front.pareto_optimal
    if len(points) < 3:
        logger.warning("Need at least 3 points for projection")
        return None

    X = front.to_array(pareto_only=True)

    # Compute projection
    if method == 'tsne':
        perplexity = min(30, len(points) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = reducer.fit_transform(X)
    elif method == 'umap':
        n_neighbors = min(15, len(points) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        X_2d = reducer.fit_transform(X)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")

    # Color
    if color_by and color_by in front.objective_names:
        colors = [p.objectives[color_by] for p in points]
        colorbar_title = color_by
    else:
        colors = list(range(len(points)))
        colorbar_title = "Index"

    # Build hover text
    hover_text = []
    for p in points:
        text = "<br>".join(f"{k}: {v:.4f}" for k, v in p.objectives.items())
        if p.experiment_id:
            text = f"ID: {p.experiment_id}<br>" + text
        hover_text.append(text)

    fig = go.Figure(data=go.Scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        text=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"{title} ({method.upper()})",
        xaxis_title=f"{method.upper()} 1",
        yaxis_title=f"{method.upper()} 2",
        height=height
    )

    return fig


def create_objective_scatter_matrix(
    front: ParetoFront,
    title: str = "Objective Scatter Matrix",
    show_dominated: bool = False
) -> Optional[Any]:
    """
    Create scatter matrix showing all pairwise objective relationships.

    Args:
        front: ParetoFront object
        title: Plot title
        show_dominated: Include dominated points

    Returns:
        Plotly Figure or None if plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None

    points = front.points if show_dominated else front.pareto_optimal
    if not points:
        return None

    n_obj = len(front.objective_names)
    fig = make_subplots(
        rows=n_obj, cols=n_obj,
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for i, name_i in enumerate(front.objective_names):
        for j, name_j in enumerate(front.objective_names):
            x = [p.objectives[name_j] for p in points]
            y = [p.objectives[name_i] for p in points]

            if i == j:
                # Histogram on diagonal
                fig.add_trace(
                    go.Histogram(x=x, name=name_i, showlegend=False,
                                marker_color='steelblue'),
                    row=i+1, col=j+1
                )
            else:
                # Scatter plot
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode='markers',
                              name=f"{name_i} vs {name_j}", showlegend=False,
                              marker=dict(size=6, color='steelblue')),
                    row=i+1, col=j+1
                )

            # Update axis labels
            if i == n_obj - 1:
                fig.update_xaxes(title_text=name_j, row=i+1, col=j+1)
            if j == 0:
                fig.update_yaxes(title_text=name_i, row=i+1, col=j+1)

    fig.update_layout(
        title=title,
        height=200*n_obj,
        width=200*n_obj
    )
    return fig


def create_radar_chart(
    front: ParetoFront,
    point_indices: Optional[List[int]] = None,
    title: str = "Radar Chart Comparison",
    normalize: bool = True
) -> Optional[Any]:
    """
    Create radar chart comparing selected Pareto-optimal solutions.

    Args:
        front: ParetoFront object
        point_indices: Indices of points to compare (default: all Pareto-optimal)
        title: Plot title
        normalize: Normalize objectives to [0, 1]

    Returns:
        Plotly Figure or None if plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None

    points = front.pareto_optimal
    if not points:
        return None

    if point_indices is not None:
        points = [points[i] for i in point_indices if i < len(points)]

    # Prepare data
    categories = front.objective_names + [front.objective_names[0]]  # Close the radar

    if normalize:
        # Compute min/max for normalization
        all_points = front.points
        mins = {name: min(p.objectives[name] for p in all_points) for name in front.objective_names}
        maxs = {name: max(p.objectives[name] for p in all_points) for name in front.objective_names}

    fig = go.Figure()

    for idx, point in enumerate(points):
        values = []
        for name in front.objective_names:
            val = point.objectives[name]
            if normalize and maxs[name] != mins[name]:
                val = (val - mins[name]) / (maxs[name] - mins[name])
            values.append(val)
        values.append(values[0])  # Close the radar

        name = point.experiment_id or f"Point {idx}"
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            name=name,
            fill='toself',
            opacity=0.6
        ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1] if normalize else None)),
        showlegend=True
    )

    return fig


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_pareto_front_statistics(front: ParetoFront) -> Dict[str, Any]:
    """
    Compute statistics about the Pareto front.

    Args:
        front: ParetoFront object

    Returns:
        Dictionary of statistics
    """
    pareto_points = front.pareto_optimal
    all_points = front.points

    if not all_points:
        return {'error': 'No points in front'}

    stats = {
        'total_points': len(all_points),
        'pareto_optimal_count': len(pareto_points),
        'dominated_count': len(all_points) - len(pareto_points),
        'pareto_ratio': len(pareto_points) / len(all_points) if all_points else 0,
    }

    # Per-objective statistics
    stats['objectives'] = {}
    for name in front.objective_names:
        values = [p.objectives[name] for p in pareto_points]
        if values:
            stats['objectives'][name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
            }

    # Hypervolume
    try:
        stats['hypervolume'] = front.compute_hypervolume()
    except Exception as e:
        stats['hypervolume'] = None
        stats['hypervolume_error'] = str(e)

    return stats


def export_front_to_json(front: ParetoFront, filepath: str) -> None:
    """Export Pareto front to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(front.to_dict(), f, indent=2, default=str)


def import_front_from_json(filepath: str) -> ParetoFront:
    """Import Pareto front from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ParetoFront.from_dict(data)
