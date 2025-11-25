"""
Tests for Pareto front visualization.

Author: ARC Team (Dev 2)
Created: 2025-11-24
"""

import pytest
import numpy as np
import json
import tempfile
import os

from tools.pareto_viz import (
    ParetoFront,
    ParetoPoint,
    create_parallel_coordinates,
    create_projection_plot,
    create_objective_scatter_matrix,
    create_radar_chart,
    compute_pareto_front_statistics,
    export_front_to_json,
    import_front_from_json,
    PLOTLY_AVAILABLE,
    SKLEARN_AVAILABLE,
)


class TestParetoPoint:
    """Test ParetoPoint dataclass."""

    def test_create_point(self):
        """Test creating a Pareto point."""
        point = ParetoPoint(
            objectives={'loss': 0.1, 'latency': 10},
            config={'lr': 0.01},
            experiment_id='exp_001'
        )

        assert point.objectives['loss'] == 0.1
        assert point.config['lr'] == 0.01
        assert point.experiment_id == 'exp_001'
        assert not point.dominated


class TestParetoFront:
    """Test ParetoFront class."""

    @pytest.fixture
    def simple_front(self):
        """Create simple front with 2 objectives."""
        front = ParetoFront(
            objective_names=['loss', 'latency'],
            minimize=[True, True]
        )
        return front

    @pytest.fixture
    def populated_front(self, simple_front):
        """Create populated front."""
        front = simple_front
        # Pareto-optimal points (neither dominates the other)
        front.add_point({'loss': 0.1, 'latency': 10}, {'lr': 0.01})   # Low loss, higher latency
        front.add_point({'loss': 0.2, 'latency': 5}, {'lr': 0.001})   # Higher loss, low latency
        front.add_point({'loss': 0.05, 'latency': 15}, {'lr': 0.005}) # Lowest loss, highest latency - also Pareto optimal
        # Dominated point (worse in both objectives than point 1)
        front.add_point({'loss': 0.3, 'latency': 20}, {'lr': 0.1})
        return front

    def test_init(self, simple_front):
        """Test initialization."""
        assert simple_front.objective_names == ['loss', 'latency']
        assert simple_front.minimize == [True, True]
        assert len(simple_front.points) == 0

    def test_add_point(self, simple_front):
        """Test adding points."""
        simple_front.add_point({'loss': 0.1, 'latency': 10}, {'lr': 0.01})

        assert len(simple_front.points) == 1
        assert simple_front.points[0].objectives['loss'] == 0.1

    def test_add_point_missing_objective(self, simple_front):
        """Test error on missing objective."""
        with pytest.raises(ValueError, match="Missing objective"):
            simple_front.add_point({'loss': 0.1}, {'lr': 0.01})

    def test_domination(self, populated_front):
        """Test correct domination detection."""
        pareto = populated_front.pareto_optimal

        # 3 Pareto-optimal, 1 dominated
        assert len(pareto) == 3
        assert populated_front.points[3].dominated

    def test_pareto_optimal_property(self, populated_front):
        """Test pareto_optimal property."""
        pareto = populated_front.pareto_optimal

        for p in pareto:
            assert not p.dominated

    def test_to_array(self, populated_front):
        """Test array conversion."""
        arr = populated_front.to_array(pareto_only=True)

        assert arr.shape == (3, 2)  # 3 pareto points, 2 objectives

    def test_to_array_all(self, populated_front):
        """Test array conversion with all points."""
        arr = populated_front.to_array(pareto_only=False)

        assert arr.shape == (4, 2)  # 4 total points, 2 objectives

    def test_hypervolume(self, populated_front):
        """Test hypervolume computation."""
        hv = populated_front.compute_hypervolume()

        assert hv > 0
        assert isinstance(hv, float)

    def test_hypervolume_empty(self, simple_front):
        """Test hypervolume with empty front."""
        hv = simple_front.compute_hypervolume()

        assert hv == 0.0

    def test_to_dict(self, populated_front):
        """Test serialization."""
        data = populated_front.to_dict()

        assert 'objective_names' in data
        assert 'minimize' in data
        assert 'points' in data
        assert len(data['points']) == 4

    def test_from_dict(self, populated_front):
        """Test deserialization."""
        data = populated_front.to_dict()
        restored = ParetoFront.from_dict(data)

        assert restored.objective_names == populated_front.objective_names
        assert len(restored.points) == len(populated_front.points)


class TestParetoFrontHighDim:
    """Test with high-dimensional objectives."""

    @pytest.fixture
    def front_5d(self):
        """Create 5-dimensional front."""
        front = ParetoFront(
            objective_names=['loss', 'acc', 'latency', 'memory', 'flops'],
            minimize=[True, False, True, True, True]
        )

        np.random.seed(42)
        for i in range(20):
            front.add_point({
                'loss': np.random.uniform(0.1, 0.5),
                'acc': np.random.uniform(0.7, 0.95),
                'latency': np.random.uniform(5, 50),
                'memory': np.random.uniform(50, 500),
                'flops': np.random.uniform(1e6, 1e9),
            }, {'config_id': i})

        return front

    def test_domination_5d(self, front_5d):
        """Test domination in 5D."""
        pareto = front_5d.pareto_optimal

        # Some points should be dominated in 5D
        assert len(pareto) < len(front_5d.points)

    def test_hypervolume_5d(self, front_5d):
        """Test hypervolume in 5D."""
        hv = front_5d.compute_hypervolume()

        assert hv > 0


class TestVisualizationFunctions:
    """Test visualization functions."""

    @pytest.fixture
    def front_for_viz(self):
        """Create front suitable for visualization."""
        front = ParetoFront(
            objective_names=['loss', 'acc', 'latency'],
            minimize=[True, False, True]
        )

        np.random.seed(42)
        for i in range(10):
            front.add_point({
                'loss': np.random.uniform(0.1, 0.5),
                'acc': np.random.uniform(0.7, 0.95),
                'latency': np.random.uniform(5, 50),
            }, {'config_id': i}, experiment_id=f'exp_{i}')

        return front

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_parallel_coordinates(self, front_for_viz):
        """Test parallel coordinates plot."""
        fig = create_parallel_coordinates(front_for_viz)

        assert fig is not None
        assert hasattr(fig, 'to_json')

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_parallel_coordinates_with_color(self, front_for_viz):
        """Test parallel coordinates with color."""
        fig = create_parallel_coordinates(front_for_viz, color_by='loss')

        assert fig is not None

    @pytest.mark.skipif(
        not (PLOTLY_AVAILABLE and SKLEARN_AVAILABLE),
        reason="Plotly or sklearn not installed"
    )
    def test_pca_projection(self, front_for_viz):
        """Test PCA projection."""
        fig = create_projection_plot(front_for_viz, method='pca')

        assert fig is not None

    @pytest.mark.skipif(
        not (PLOTLY_AVAILABLE and SKLEARN_AVAILABLE),
        reason="Plotly or sklearn not installed"
    )
    def test_tsne_projection(self, front_for_viz):
        """Test t-SNE projection."""
        fig = create_projection_plot(front_for_viz, method='tsne')

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_scatter_matrix(self, front_for_viz):
        """Test scatter matrix."""
        fig = create_objective_scatter_matrix(front_for_viz)

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_radar_chart(self, front_for_viz):
        """Test radar chart."""
        fig = create_radar_chart(front_for_viz)

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_radar_chart_selected(self, front_for_viz):
        """Test radar chart with selected points."""
        fig = create_radar_chart(front_for_viz, point_indices=[0, 1, 2])

        assert fig is not None


class TestStatistics:
    """Test statistics computation."""

    @pytest.fixture
    def front_with_data(self):
        """Create front with data for statistics."""
        front = ParetoFront(
            objective_names=['loss', 'latency'],
            minimize=[True, True]
        )

        for i in range(10):
            front.add_point({
                'loss': 0.1 + i * 0.05,
                'latency': 10 - i * 0.5
            }, {'idx': i})

        return front

    def test_compute_statistics(self, front_with_data):
        """Test statistics computation."""
        stats = compute_pareto_front_statistics(front_with_data)

        assert 'total_points' in stats
        assert 'pareto_optimal_count' in stats
        assert 'pareto_ratio' in stats
        assert 'objectives' in stats

        assert stats['total_points'] == 10
        assert stats['pareto_ratio'] <= 1.0


class TestExportImport:
    """Test export/import functions."""

    @pytest.fixture
    def front_for_export(self):
        """Create front for export testing."""
        front = ParetoFront(
            objective_names=['loss', 'latency'],
            minimize=[True, True]
        )
        front.add_point({'loss': 0.1, 'latency': 10}, {'lr': 0.01})
        front.add_point({'loss': 0.2, 'latency': 5}, {'lr': 0.001})
        return front

    def test_export_import_json(self, front_for_export):
        """Test JSON export and import."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            export_front_to_json(front_for_export, filepath)

            restored = import_front_from_json(filepath)

            assert restored.objective_names == front_for_export.objective_names
            assert len(restored.points) == len(front_for_export.points)
        finally:
            os.unlink(filepath)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_front(self):
        """Test operations on empty front."""
        front = ParetoFront(['a', 'b'])

        assert len(front.pareto_optimal) == 0
        assert front.compute_hypervolume() == 0.0
        assert front.to_array().shape == (0, 2)

    def test_single_point(self):
        """Test front with single point."""
        front = ParetoFront(['loss'])
        front.add_point({'loss': 0.5}, {})

        assert len(front.pareto_optimal) == 1
        assert not front.points[0].dominated

    def test_all_dominated_except_one(self):
        """Test when all but one point is dominated."""
        front = ParetoFront(['loss', 'latency'], minimize=[True, True])

        # One clearly dominating point
        front.add_point({'loss': 0.1, 'latency': 1}, {})
        # Multiple dominated points
        front.add_point({'loss': 0.2, 'latency': 2}, {})
        front.add_point({'loss': 0.3, 'latency': 3}, {})
        front.add_point({'loss': 0.4, 'latency': 4}, {})

        assert len(front.pareto_optimal) == 1

    def test_maximize_objectives(self):
        """Test with maximization objectives."""
        front = ParetoFront(['acc', 'score'], minimize=[False, False])

        front.add_point({'acc': 0.9, 'score': 0.8}, {})
        front.add_point({'acc': 0.8, 'score': 0.9}, {})
        front.add_point({'acc': 0.7, 'score': 0.7}, {})  # Dominated

        assert len(front.pareto_optimal) == 2
        assert front.points[2].dominated

    def test_mixed_min_max(self):
        """Test mixed minimization/maximization."""
        front = ParetoFront(
            ['loss', 'acc'],
            minimize=[True, False]  # Minimize loss, maximize acc
        )

        front.add_point({'loss': 0.1, 'acc': 0.9}, {})  # Pareto
        front.add_point({'loss': 0.2, 'acc': 0.95}, {})  # Pareto
        front.add_point({'loss': 0.3, 'acc': 0.8}, {})  # Dominated

        assert len(front.pareto_optimal) == 2
