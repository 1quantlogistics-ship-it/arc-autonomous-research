"""
REST API endpoints for Pareto visualization.

Phase F addition: provides REST API for high-dimensional Pareto
front visualization and analysis.

Endpoints:
- POST /api/visualization/pareto/parallel-coordinates
- POST /api/visualization/pareto/projection
- POST /api/visualization/pareto/hypervolume
- POST /api/visualization/pareto/statistics
- POST /api/visualization/pareto/scatter-matrix
- POST /api/visualization/pareto/radar

Author: ARC Team (Dev 2)
Created: 2025-11-24
Version: 1.0 (Phase F)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging

from tools.pareto_viz import (
    ParetoFront,
    create_parallel_coordinates,
    create_projection_plot,
    create_objective_scatter_matrix,
    create_radar_chart,
    compute_pareto_front_statistics,
    PLOTLY_AVAILABLE,
    SKLEARN_AVAILABLE,
    UMAP_AVAILABLE
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visualization", tags=["Visualization"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ParetoPointInput(BaseModel):
    """Input model for a single Pareto point."""
    objectives: Dict[str, float] = Field(
        description="Objective values keyed by name"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration that produced this result"
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Optional experiment identifier"
    )


class ParetoFrontInput(BaseModel):
    """Input model for Pareto front data."""
    objective_names: List[str] = Field(
        description="Names of objectives"
    )
    minimize: Optional[List[bool]] = Field(
        default=None,
        description="Whether each objective should be minimized (default: all True)"
    )
    points: List[ParetoPointInput] = Field(
        description="List of points to add to front"
    )


class VisualizationResponse(BaseModel):
    """Response containing visualization data."""
    success: bool
    plot_json: Optional[str] = None
    error: Optional[str] = None


class HypervolumeResponse(BaseModel):
    """Response for hypervolume computation."""
    hypervolume: float
    num_pareto_optimal: int
    num_total: int
    reference_point: Dict[str, float]


class StatisticsResponse(BaseModel):
    """Response for Pareto front statistics."""
    total_points: int
    pareto_optimal_count: int
    dominated_count: int
    pareto_ratio: float
    hypervolume: Optional[float]
    objectives: Dict[str, Dict[str, float]]


# ============================================================================
# Helper Functions
# ============================================================================

def _build_front(data: ParetoFrontInput) -> ParetoFront:
    """Build ParetoFront from input data."""
    front = ParetoFront(data.objective_names, data.minimize)
    for p in data.points:
        front.add_point(p.objectives, p.config, p.experiment_id)
    return front


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status")
async def get_visualization_status():
    """Get status of visualization dependencies."""
    return {
        "plotly_available": PLOTLY_AVAILABLE,
        "sklearn_available": SKLEARN_AVAILABLE,
        "umap_available": UMAP_AVAILABLE,
        "capabilities": {
            "parallel_coordinates": PLOTLY_AVAILABLE,
            "pca_projection": PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
            "tsne_projection": PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
            "umap_projection": PLOTLY_AVAILABLE and UMAP_AVAILABLE,
            "scatter_matrix": PLOTLY_AVAILABLE,
            "radar_chart": PLOTLY_AVAILABLE,
        }
    }


@router.post("/pareto/parallel-coordinates", response_model=VisualizationResponse)
async def get_parallel_coordinates(
    data: ParetoFrontInput,
    color_by: Optional[str] = None,
    show_dominated: bool = False,
    title: str = "Pareto Front - Parallel Coordinates"
):
    """
    Generate parallel coordinates plot for Pareto front.

    Suitable for visualizing >3 objectives simultaneously.
    """
    if not PLOTLY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Plotly not installed. Install with: pip install plotly"
        )

    try:
        front = _build_front(data)
        fig = create_parallel_coordinates(
            front,
            title=title,
            color_by=color_by,
            show_dominated=show_dominated
        )

        if fig is None:
            return VisualizationResponse(
                success=False,
                error="Failed to create visualization - no points?"
            )

        return VisualizationResponse(
            success=True,
            plot_json=fig.to_json()
        )
    except Exception as e:
        logger.error(f"Error creating parallel coordinates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto/projection", response_model=VisualizationResponse)
async def get_projection(
    data: ParetoFrontInput,
    method: str = "pca",
    color_by: Optional[str] = None,
    title: str = "Pareto Front - 2D Projection"
):
    """
    Generate 2D projection plot of Pareto front.

    Methods: 'pca', 'tsne', 'umap'
    """
    if not PLOTLY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Plotly not installed"
        )

    if method in ['pca', 'tsne'] and not SKLEARN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"{method.upper()} requires scikit-learn. Install with: pip install scikit-learn"
        )

    if method == 'umap' and not UMAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="UMAP requires umap-learn. Install with: pip install umap-learn"
        )

    if method not in ['pca', 'tsne', 'umap']:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'"
        )

    try:
        front = _build_front(data)
        fig = create_projection_plot(
            front,
            method=method,
            title=title,
            color_by=color_by
        )

        if fig is None:
            return VisualizationResponse(
                success=False,
                error="Failed to create projection - need at least 3 points"
            )

        return VisualizationResponse(
            success=True,
            plot_json=fig.to_json()
        )
    except Exception as e:
        logger.error(f"Error creating projection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto/hypervolume", response_model=HypervolumeResponse)
async def compute_hypervolume(
    data: ParetoFrontInput,
    reference: Optional[Dict[str, float]] = None,
    n_samples: int = 10000
):
    """
    Compute hypervolume indicator for Pareto front.

    Uses Monte Carlo estimation.
    """
    try:
        front = _build_front(data)

        # Compute reference if not provided
        if reference is None:
            reference = {}
            for i, name in enumerate(front.objective_names):
                values = [p.objectives[name] for p in front.points]
                if front.minimize[i]:
                    reference[name] = max(values) * 1.1
                else:
                    reference[name] = min(values) * 0.9

        hv = front.compute_hypervolume(reference=reference, n_samples=n_samples)

        return HypervolumeResponse(
            hypervolume=hv,
            num_pareto_optimal=len(front.pareto_optimal),
            num_total=len(front.points),
            reference_point=reference
        )
    except Exception as e:
        logger.error(f"Error computing hypervolume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto/statistics", response_model=StatisticsResponse)
async def get_statistics(data: ParetoFrontInput):
    """Get statistics about the Pareto front."""
    try:
        front = _build_front(data)
        stats = compute_pareto_front_statistics(front)

        return StatisticsResponse(
            total_points=stats['total_points'],
            pareto_optimal_count=stats['pareto_optimal_count'],
            dominated_count=stats['dominated_count'],
            pareto_ratio=stats['pareto_ratio'],
            hypervolume=stats.get('hypervolume'),
            objectives=stats.get('objectives', {})
        )
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto/scatter-matrix", response_model=VisualizationResponse)
async def get_scatter_matrix(
    data: ParetoFrontInput,
    show_dominated: bool = False,
    title: str = "Objective Scatter Matrix"
):
    """
    Generate scatter matrix showing all pairwise objective relationships.
    """
    if not PLOTLY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Plotly not installed"
        )

    try:
        front = _build_front(data)
        fig = create_objective_scatter_matrix(
            front,
            title=title,
            show_dominated=show_dominated
        )

        if fig is None:
            return VisualizationResponse(
                success=False,
                error="Failed to create scatter matrix"
            )

        return VisualizationResponse(
            success=True,
            plot_json=fig.to_json()
        )
    except Exception as e:
        logger.error(f"Error creating scatter matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pareto/radar", response_model=VisualizationResponse)
async def get_radar_chart(
    data: ParetoFrontInput,
    point_indices: Optional[List[int]] = None,
    normalize: bool = True,
    title: str = "Radar Chart Comparison"
):
    """
    Generate radar chart comparing selected Pareto-optimal solutions.
    """
    if not PLOTLY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Plotly not installed"
        )

    try:
        front = _build_front(data)
        fig = create_radar_chart(
            front,
            point_indices=point_indices,
            title=title,
            normalize=normalize
        )

        if fig is None:
            return VisualizationResponse(
                success=False,
                error="Failed to create radar chart"
            )

        return VisualizationResponse(
            success=True,
            plot_json=fig.to_json()
        )
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))
