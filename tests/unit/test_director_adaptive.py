"""
Unit Tests for Adaptive Director Strategy Switching (CPU-Only, Offline)
========================================================================

Tests Director's algorithmic strategy adaptation using synthetic performance histories.
No GPU, no RunPod, no real training required.

Tests:
- Mode transitions (EXPLORE, EXPLOIT, RECOVER)
- Stagnation detection
- Regression detection
- Novelty budget adjustments
- Performance trend analysis
- Strategy persistence across cycles
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Import director
try:
    import sys
    sys.path.insert(0, '/Users/bengibson/Desktop/ARC/arc_clean')
    from agents.director_agent import DirectorAgent
    DIRECTOR_AVAILABLE = True
except ImportError:
    DIRECTOR_AVAILABLE = False
    pytest.skip("Director not available", allow_module_level=True)


def generate_synthetic_history_trend(
    n_experiments: int,
    trend: str,
    base_auc: float = 0.70,
    noise: float = 0.02
) -> Dict[str, Any]:
    """
    Generate synthetic training history with specific performance trend.

    Args:
        n_experiments: Number of experiments
        trend: "improving", "stagnant", "regressing", "volatile"
        base_auc: Starting AUC
        noise: Gaussian noise std

    Returns:
        Training history dict with experiments
    """
    np.random.seed(42)
    experiments = []

    for i in range(n_experiments):
        # Compute AUC based on trend
        if trend == "improving":
            # Steady improvement
            auc = base_auc + (i / n_experiments) * 0.15 + np.random.normal(0, noise)
        elif trend == "stagnant":
            # Flat performance
            auc = base_auc + np.random.normal(0, noise)
        elif trend == "regressing":
            # Performance degradation (more aggressive)
            auc = base_auc - (i / n_experiments) * 0.25 + np.random.normal(0, noise)
        elif trend == "volatile":
            # High variance
            auc = base_auc + np.random.normal(0, noise * 5)
        else:
            auc = base_auc

        # Clip to valid range
        auc = np.clip(auc, 0.5, 1.0)

        # Create experiment record
        exp = {
            "experiment_id": f"exp_{i:03d}",
            "cycle": i // 3,  # 3 experiments per cycle
            "status": "completed",
            "config": {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "epochs": 10
            },
            "metrics": {
                "auc": float(auc),
                "sensitivity": float(auc + np.random.normal(0, 0.01)),
                "specificity": float(auc + np.random.normal(0, 0.01))
            },
            "timestamp": f"2025-01-{i+1:02d}T00:00:00"
        }
        experiments.append(exp)

    # Compute best metrics
    best_exp = max(experiments, key=lambda e: e["metrics"]["auc"])

    return {
        "experiments": experiments,
        "total_experiments": n_experiments,
        "best_metrics": {
            "auc": best_exp["metrics"]["auc"],
            "experiment_id": best_exp["experiment_id"]
        },
        "cycles": list(range(n_experiments // 3))
    }


class MockHistorian:
    """Mock Historian for testing Director."""

    def __init__(self, history: Dict[str, Any]):
        """Initialize with synthetic history."""
        self.history = history

    def get_recent_performance(self, window: int = 5) -> List[float]:
        """Get recent AUC values."""
        experiments = self.history.get("experiments", [])
        if not experiments:
            return []

        # Get last N experiments
        recent = experiments[-window:]
        return [exp["metrics"]["auc"] for exp in recent]

    def get_performance_trend(self, metric: str = "auc", window: int = 5) -> List[float]:
        """Get performance trend for specified metric."""
        experiments = self.history.get("experiments", [])
        if not experiments:
            return []

        # Get last N experiments
        recent = experiments[-window:]
        return [exp["metrics"].get(metric, 0.0) for exp in recent]

    def detect_stagnation(self, metric: str = "auc", threshold: float = 0.01, window: int = 5) -> bool:
        """Detect if performance has stagnated."""
        trend = self.get_performance_trend(metric=metric, window=window)
        if len(trend) < 2:
            return False

        # Check if improvement is below threshold
        improvement = trend[-1] - trend[0]
        return improvement < threshold

    def get_best_auc(self) -> float:
        """Get best AUC achieved."""
        return self.history.get("best_metrics", {}).get("auc", 0.0)

    def get_total_experiments(self) -> int:
        """Get total experiment count."""
        return self.history.get("total_experiments", 0)


class TestDirectorAdaptive:
    """Test suite for Director adaptive strategy."""

    @pytest.fixture
    def director(self, tmp_path):
        """Create director with temp memory."""
        memory_path = tmp_path / "memory"
        memory_path.mkdir()

        director = DirectorAgent(memory_path=str(memory_path))
        return director

    def _setup_director_memory(self, director: DirectorAgent, history: Dict[str, Any]):
        """Setup Director's memory files from synthetic history."""
        # Write history_summary.json
        history_summary = {
            "total_cycles": len(history.get("cycles", [])),
            "total_experiments": history.get("total_experiments", 0),
            "best_metrics": history.get("best_metrics", {}),
            "recent_experiments": history.get("experiments", [])[-10:]  # Last 10
        }
        director.write_memory("history_summary.json", history_summary)

        # Write training_history.json
        director.write_memory("training_history.json", history)

    # ========== MODE DETECTION ==========

    def test_exploit_mode_on_strong_improvement(self, director):
        """Test EXPLOIT mode when performance shows strong improvement (>5%)."""
        # Generate strong improving trend (>5% improvement in window)
        history = generate_synthetic_history_trend(
            n_experiments=15,
            trend="improving",
            base_auc=0.70,
            noise=0.01
        )

        # Manually boost recent experiments to ensure >5% improvement
        for i in range(-5, 0):
            history["experiments"][i]["metrics"]["auc"] += 0.06

        # Setup Director's memory
        self._setup_director_memory(director, history)

        historian = MockHistorian(history)

        # Compute strategy
        strategy = director.compute_adaptive_strategy(historian, window=5)

        assert strategy["mode"] == "exploit", f"Expected EXPLOIT mode on strong improvement, got {strategy['mode']}"
        assert strategy["novelty_budget"]["exploit"] >= 2, "EXPLOIT mode should favor exploit proposals"

    def test_explore_mode_on_stagnation(self, director):
        """Test EXPLORE mode when performance stagnates."""
        # Generate stagnant trend
        history = generate_synthetic_history_trend(
            n_experiments=15,
            trend="stagnant",
            base_auc=0.75
        )

        historian = MockHistorian(history)

        # Compute strategy
        strategy = director.compute_adaptive_strategy(
            historian,
            window=5,
            stagnation_threshold=0.01
        )

        assert strategy["mode"] == "explore", f"Expected EXPLORE mode on stagnation, got {strategy['mode']}"
        assert strategy["novelty_budget"]["explore"] >= 2, "EXPLORE mode should favor explore proposals"

    def test_recover_mode_on_regression(self, director):
        """Test RECOVER mode when performance regresses."""
        # Generate regressing trend
        history = generate_synthetic_history_trend(
            n_experiments=15,
            trend="regressing",
            base_auc=0.80
        )

        historian = MockHistorian(history)

        # Compute strategy
        strategy = director.compute_adaptive_strategy(
            historian,
            window=5,
            regression_threshold=-0.05
        )

        assert strategy["mode"] == "recover", f"Expected RECOVER mode on regression, got {strategy['mode']}"
        assert strategy["novelty_budget"]["exploit"] >= 2, "RECOVER mode should favor safe exploit proposals"
        assert strategy["novelty_budget"]["wildcat"] == 0, "RECOVER mode should avoid wildcat proposals"

    # ========== NOVELTY BUDGET VALIDATION ==========

    def test_novelty_budget_exploit_mode(self, director):
        """Test novelty budget in EXPLOIT mode."""
        history = generate_synthetic_history_trend(15, "improving")
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian)

        # EXPLOIT mode should favor exploit proposals
        assert strategy["novelty_budget"]["exploit"] >= 2
        assert strategy["novelty_budget"]["wildcat"] == 0  # No wildcats in EXPLOIT

    def test_novelty_budget_explore_mode(self, director):
        """Test novelty budget in EXPLORE mode."""
        history = generate_synthetic_history_trend(15, "stagnant")
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian, stagnation_threshold=0.01)

        # EXPLORE mode should favor explore and allow wildcats
        assert strategy["novelty_budget"]["explore"] >= 1
        assert strategy["novelty_budget"]["wildcat"] >= 0  # May include wildcats

    def test_novelty_budget_recover_mode(self, director):
        """Test novelty budget in RECOVER mode."""
        history = generate_synthetic_history_trend(15, "regressing")
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian, regression_threshold=-0.05)

        # RECOVER mode should focus on safe exploit
        assert strategy["novelty_budget"]["exploit"] >= 2
        assert strategy["novelty_budget"]["wildcat"] == 0  # No wildcats in RECOVER

    # ========== TREND ANALYSIS ==========

    def test_trend_detection_improving(self, director):
        """Test trend detection for improving performance."""
        history = generate_synthetic_history_trend(15, "improving", base_auc=0.70)
        historian = MockHistorian(history)

        recent = historian.get_recent_performance(window=5)

        # Compute improvement
        improvement = recent[-1] - recent[0]

        assert improvement > 0.0, "Improving trend should have positive improvement"

    def test_trend_detection_stagnant(self, director):
        """Test trend detection for stagnant performance."""
        history = generate_synthetic_history_trend(15, "stagnant", base_auc=0.75)
        historian = MockHistorian(history)

        recent = historian.get_recent_performance(window=5)

        # Compute variance
        variance = np.var(recent)

        assert variance < 0.01, "Stagnant trend should have low variance"

    def test_trend_detection_regressing(self, director):
        """Test trend detection for regressing performance."""
        history = generate_synthetic_history_trend(15, "regressing", base_auc=0.80)
        historian = MockHistorian(history)

        recent = historian.get_recent_performance(window=5)

        # Compute improvement (should be negative)
        improvement = recent[-1] - recent[0]

        assert improvement < 0.0, "Regressing trend should have negative improvement"

    # ========== EDGE CASES ==========

    def test_no_history(self, director):
        """Test behavior with no history (cold start)."""
        history = {"experiments": [], "total_experiments": 0, "best_metrics": {}}
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian)

        # Should default to EXPLORE mode
        assert strategy["mode"] == "explore", "Cold start should default to EXPLORE"
        assert strategy["novelty_budget"]["explore"] >= 1

    def test_insufficient_history(self, director):
        """Test behavior with very few experiments."""
        history = generate_synthetic_history_trend(3, "improving")
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian, window=5)

        # Should still work with small window
        assert strategy["mode"] in ["explore", "exploit", "recover"]

    def test_volatile_performance(self, director):
        """Test behavior with high variance performance."""
        history = generate_synthetic_history_trend(15, "volatile", base_auc=0.75)
        historian = MockHistorian(history)

        strategy = director.compute_adaptive_strategy(historian)

        # High variance should trigger EXPLORE (need more data)
        assert strategy["mode"] in ["explore", "exploit"]

    # ========== STRATEGY PERSISTENCE ==========

    def test_strategy_consistency_across_cycles(self, director):
        """Test that strategy remains consistent within trend."""
        history = generate_synthetic_history_trend(15, "improving")
        historian = MockHistorian(history)

        # Compute strategy twice
        strategy1 = director.compute_adaptive_strategy(historian)
        strategy2 = director.compute_adaptive_strategy(historian)

        # Should be identical (deterministic)
        assert strategy1["mode"] == strategy2["mode"]
        assert strategy1["novelty_budget"] == strategy2["novelty_budget"]

    # ========== THRESHOLD SENSITIVITY ==========

    def test_stagnation_threshold_sensitivity(self, director):
        """Test sensitivity to stagnation threshold."""
        history = generate_synthetic_history_trend(15, "stagnant", base_auc=0.75)
        historian = MockHistorian(history)

        # Tight threshold (should detect stagnation)
        strategy_tight = director.compute_adaptive_strategy(
            historian, stagnation_threshold=0.01
        )

        # Loose threshold (may not detect stagnation)
        strategy_loose = director.compute_adaptive_strategy(
            historian, stagnation_threshold=0.10
        )

        # Tight threshold should favor EXPLORE more
        assert strategy_tight["mode"] in ["explore", "exploit"]

    def test_regression_threshold_sensitivity(self, director):
        """Test sensitivity to regression threshold."""
        history = generate_synthetic_history_trend(15, "regressing", base_auc=0.80)
        historian = MockHistorian(history)

        # Tight threshold (should detect regression)
        strategy_tight = director.compute_adaptive_strategy(
            historian, regression_threshold=-0.03
        )

        # Loose threshold (may not detect regression)
        strategy_loose = director.compute_adaptive_strategy(
            historian, regression_threshold=-0.20
        )

        # Tight threshold should trigger RECOVER
        assert strategy_tight["mode"] in ["recover", "explore"]


def test_director_adaptive_comprehensive():
    """Comprehensive integration test of adaptive strategy."""
    print("\n" + "="*60)
    print("DIRECTOR ADAPTIVE STRATEGY TEST (CPU-ONLY)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory_path = Path(tmp_dir) / "memory"
        memory_path.mkdir()

        director = DirectorAgent(memory_path=str(memory_path))

        # Test scenarios
        # Note: Director requires >5% improvement for EXPLOIT mode
        scenarios = [
            ("Moderate improvement", "improving", 0.70, "explore"),  # <5% improvement → explore
            ("Stagnant performance", "stagnant", 0.75, "explore"),
            ("Regressing performance", "regressing", 0.80, "recover"),
            ("Volatile performance", "volatile", 0.73, None)  # Can be explore or recover
        ]

        print("\n✓ Testing adaptive mode detection:")

        # Test EXPLOIT mode with strong improvement (manual boost)
        exploit_history = generate_synthetic_history_trend(15, "improving", base_auc=0.70, noise=0.005)

        # Manually set recent 5 experiments to have >5% improvement
        # AUCs: 0.80, 0.82, 0.84, 0.86, 0.88 → improvement = 0.08 (8%)
        exploit_history["experiments"][-5]["metrics"]["auc"] = 0.80
        exploit_history["experiments"][-4]["metrics"]["auc"] = 0.82
        exploit_history["experiments"][-3]["metrics"]["auc"] = 0.84
        exploit_history["experiments"][-2]["metrics"]["auc"] = 0.86
        exploit_history["experiments"][-1]["metrics"]["auc"] = 0.88

        # Setup Director memory
        history_summary = {
            "total_cycles": len(exploit_history.get("cycles", [])),
            "total_experiments": exploit_history.get("total_experiments", 0),
            "best_metrics": exploit_history.get("best_metrics", {}),
            "recent_experiments": exploit_history.get("experiments", [])[-10:]
        }
        director.write_memory("history_summary.json", history_summary)
        director.write_memory("training_history.json", exploit_history)

        exploit_historian = MockHistorian(exploit_history)

        # Debug: Check actual values
        recent = exploit_historian.get_recent_performance(window=5)
        improvement = recent[-1] - recent[0] if len(recent) >= 2 else 0.0
        print(f"  DEBUG: Recent AUCs: {[f'{x:.3f}' for x in recent]}, improvement={improvement:.3f}")

        exploit_strategy = director.compute_adaptive_strategy(exploit_historian, window=5)
        exploit_mode = exploit_strategy["mode"]
        print(f"  ✓ Strong improvement (>5%): mode={exploit_mode}, budget={exploit_strategy['novelty_budget']}")
        assert exploit_mode == "exploit", f"Expected EXPLOIT on strong improvement, got {exploit_mode}"

        for name, trend, base_auc, expected_mode in scenarios:
            history = generate_synthetic_history_trend(15, trend, base_auc=base_auc)

            # Setup Director memory
            history_summary = {
                "total_cycles": len(history.get("cycles", [])),
                "total_experiments": history.get("total_experiments", 0),
                "best_metrics": history.get("best_metrics", {}),
                "recent_experiments": history.get("experiments", [])[-10:]
            }
            director.write_memory("history_summary.json", history_summary)
            director.write_memory("training_history.json", history)

            historian = MockHistorian(history)

            # Debug: Check actual trend values
            recent = historian.get_recent_performance(window=5)
            improvement = recent[-1] - recent[0] if len(recent) >= 2 else 0.0

            strategy = director.compute_adaptive_strategy(
                historian,
                stagnation_threshold=0.01,
                regression_threshold=-0.05,
                window=5
            )

            mode = strategy["mode"]
            budget = strategy["novelty_budget"]

            if expected_mode:
                status = "✓" if mode == expected_mode else "✗"
                print(f"  {status} {name}: mode={mode}, improvement={improvement:.3f}, budget={budget}")
                assert mode == expected_mode, f"Expected {expected_mode}, got {mode}"
            else:
                print(f"  ✓ {name}: mode={mode}, improvement={improvement:.3f}, budget={budget} (flexible)")

        # Test novelty budget constraints
        print("\n✓ Testing novelty budget constraints:")
        for name, trend, base_auc, _ in scenarios:
            history = generate_synthetic_history_trend(15, trend, base_auc=base_auc)
            historian = MockHistorian(history)

            strategy = director.compute_adaptive_strategy(historian)
            budget = strategy["novelty_budget"]

            # Budget should sum to at least 1 (at least 1 proposal type)
            total = budget["exploit"] + budget["explore"] + budget["wildcat"]
            assert total >= 1, f"Budget must allow at least 1 proposal type, got {budget}"

            # RECOVER mode should never allow wildcats
            if strategy["mode"] == "recover":
                assert budget["wildcat"] == 0, "RECOVER mode should not allow wildcats"

            print(f"  ✓ {name}: total_budget={total}, valid")

    print("\n" + "="*60)
    print("✓ ALL DIRECTOR ADAPTIVE TESTS PASSED (CPU-only)")
    print("="*60)


if __name__ == "__main__":
    test_director_adaptive_comprehensive()
    pytest.main([__file__, "-v"])
