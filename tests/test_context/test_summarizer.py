"""
Tests for Cross-Cycle Summarizer.

Phase G Task 2.2: Tests for CrossCycleSummarizer.

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from context.summarizer import (
    CycleSummary,
    ExecutiveSummary,
    CrossCycleSummarizer,
    get_summarizer,
    reset_summarizer,
)


class TestCycleSummary:
    """Test CycleSummary dataclass."""

    def test_create_summary(self):
        """Test creating a cycle summary."""
        now = datetime.now()
        summary = CycleSummary(
            cycle_id=1,
            timestamp=now,
            best_accuracy=0.85,
            best_architecture="resnet",
            key_decisions=["Use ResNet", "Increase batch size"],
            what_worked=["Higher learning rate"],
            what_failed=["Lower batch size"]
        )

        assert summary.cycle_id == 1
        assert summary.best_accuracy == 0.85
        assert summary.best_architecture == "resnet"
        assert len(summary.key_decisions) == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now()
        summary = CycleSummary(
            cycle_id=1,
            timestamp=now,
            best_accuracy=0.85,
            best_architecture="resnet",
            metrics={"auc": 0.85, "sensitivity": 0.90}
        )

        data = summary.to_dict()

        assert data["cycle_id"] == 1
        assert data["best_accuracy"] == 0.85
        assert "timestamp" in data
        assert data["metrics"]["auc"] == 0.85

    def test_from_dict(self):
        """Test deserialization from dict."""
        now = datetime.now()
        data = {
            "cycle_id": 1,
            "timestamp": now.isoformat(),
            "best_accuracy": 0.85,
            "best_architecture": "resnet",
            "key_decisions": ["Decision 1"],
            "what_worked": ["Success 1"],
            "what_failed": ["Failure 1"],
            "hypotheses_tested": ["Hypothesis 1"],
            "metrics": {"auc": 0.85},
            "num_experiments": 10,
            "token_count": 500
        }

        summary = CycleSummary.from_dict(data)

        assert summary.cycle_id == 1
        assert summary.best_accuracy == 0.85
        assert summary.num_experiments == 10


class TestExecutiveSummary:
    """Test ExecutiveSummary dataclass."""

    def test_create_summary(self):
        """Test creating an executive summary."""
        summary = ExecutiveSummary(
            cycles_covered=[1, 2, 3],
            accuracy_trajectory=[0.70, 0.75, 0.80],
            best_overall_accuracy=0.80,
            successful_patterns=["Pattern A", "Pattern B"],
            failed_patterns=["Failed pattern"],
            suggested_directions=["Try X", "Explore Y"]
        )

        assert len(summary.cycles_covered) == 3
        assert summary.best_overall_accuracy == 0.80
        assert len(summary.successful_patterns) == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        summary = ExecutiveSummary(
            cycles_covered=[1, 2],
            accuracy_trajectory=[0.70, 0.80],
            best_overall_accuracy=0.80,
            stagnation_detected=True,
            stagnation_cycles=5
        )

        data = summary.to_dict()

        assert data["cycles_covered"] == [1, 2]
        assert data["stagnation_detected"] is True
        assert "generated_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        now = datetime.now()
        data = {
            "cycles_covered": [1, 2, 3],
            "accuracy_trajectory": [0.70, 0.75, 0.80],
            "best_overall_accuracy": 0.80,
            "best_experiment_id": "exp_123",
            "successful_patterns": ["Pattern A"],
            "failed_patterns": ["Failed B"],
            "suggested_directions": ["Direction C"],
            "stagnation_detected": False,
            "stagnation_cycles": 0,
            "total_experiments": 30,
            "success_rate": 0.85,
            "generated_at": now.isoformat(),
            "token_count": 1000
        }

        summary = ExecutiveSummary.from_dict(data)

        assert len(summary.cycles_covered) == 3
        assert summary.total_experiments == 30
        assert summary.success_rate == 0.85


class TestCrossCycleSummarizer:
    """Test CrossCycleSummarizer class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def summarizer(self, temp_storage):
        """Create summarizer with temp storage."""
        reset_summarizer()
        return CrossCycleSummarizer(storage_path=temp_storage)

    @pytest.fixture
    def sample_cycle_data(self):
        """Create sample cycle data."""
        return {
            "cycle_id": 1,
            "experiments": [
                {
                    "experiment_id": "exp_001",
                    "status": "completed",
                    "config": {"model": "resnet", "learning_rate": 0.001},
                    "metrics": {"auc": 0.85, "sensitivity": 0.90}
                },
                {
                    "experiment_id": "exp_002",
                    "status": "completed",
                    "config": {"model": "efficientnet", "learning_rate": 0.01},
                    "metrics": {"auc": 0.82, "sensitivity": 0.88}
                },
                {
                    "experiment_id": "exp_003",
                    "status": "failed",
                    "config": {"model": "vit", "learning_rate": 0.1},
                    "error": "Training diverged"
                }
            ],
            "proposals": [
                {"reasoning": "Try larger batch size"},
                {"reasoning": "Reduce learning rate"}
            ]
        }

    def test_summarize_cycle(self, summarizer, sample_cycle_data):
        """Test summarizing a cycle."""
        summary = summarizer.summarize_cycle(1, sample_cycle_data)

        assert summary.cycle_id == 1
        assert summary.best_accuracy == 0.85
        assert summary.best_architecture == "resnet"
        assert summary.num_experiments == 3
        assert len(summary.what_failed) > 0

    def test_summarize_empty_cycle(self, summarizer):
        """Test summarizing empty cycle."""
        summary = summarizer.summarize_cycle(1, {"cycle_id": 1, "experiments": []})

        assert summary.cycle_id == 1
        assert summary.best_accuracy == 0.0
        assert summary.num_experiments == 0

    def test_create_executive_summary(self, summarizer, sample_cycle_data):
        """Test creating executive summary."""
        # Add multiple cycles
        for i in range(5):
            cycle_data = sample_cycle_data.copy()
            cycle_data["cycle_id"] = i
            # Increase accuracy over cycles
            cycle_data["experiments"][0]["metrics"]["auc"] = 0.80 + i * 0.02
            summarizer.summarize_cycle(i, cycle_data)

        executive = summarizer.create_executive_summary(last_n_cycles=5)

        assert len(executive.cycles_covered) == 5
        assert len(executive.accuracy_trajectory) == 5
        assert executive.best_overall_accuracy >= 0.80

    def test_create_executive_summary_empty(self, summarizer):
        """Test executive summary with no cycles."""
        executive = summarizer.create_executive_summary()

        assert executive.cycles_covered == []
        assert executive.best_overall_accuracy == 0.0

    def test_create_working_summary(self, summarizer, sample_cycle_data):
        """Test creating working summary for agent."""
        # Add some cycles
        for i in range(3):
            cycle_data = sample_cycle_data.copy()
            cycle_data["cycle_id"] = i
            summarizer.summarize_cycle(i, cycle_data)

        working = summarizer.create_working_summary(
            agent_name="architect",
            max_tokens=5000,
            last_n_cycles=10
        )

        assert working["agent_name"] == "architect"
        assert working["max_tokens"] == 5000
        assert working["used_tokens"] <= 5000
        assert len(working["cycles"]) > 0

    def test_working_summary_respects_budget(self, summarizer, sample_cycle_data):
        """Test that working summary respects token budget."""
        # Add many cycles
        for i in range(20):
            cycle_data = sample_cycle_data.copy()
            cycle_data["cycle_id"] = i
            summarizer.summarize_cycle(i, cycle_data)

        # Request very small budget
        working = summarizer.create_working_summary(
            agent_name="test",
            max_tokens=500,
            last_n_cycles=20
        )

        assert working["used_tokens"] <= 500

    def test_working_summary_agent_filtering(self, summarizer):
        """Test that working summary filters by agent role."""
        # Add cycle with relevant and irrelevant content
        cycle_data = {
            "cycle_id": 1,
            "experiments": [{
                "experiment_id": "exp_001",
                "status": "completed",
                "config": {"model": "resnet"},
                "metrics": {"auc": 0.85}
            }],
            "proposals": []
        }
        summarizer.summarize_cycle(1, cycle_data)

        # Architect should get architecture-related content
        architect_summary = summarizer.create_working_summary(
            agent_name="architect",
            max_tokens=5000
        )

        assert architect_summary["focus_areas"] == ["architecture", "model", "layers", "blocks"]

    def test_stagnation_detection(self, summarizer):
        """Test stagnation detection in executive summary."""
        # Add cycles with no improvement
        for i in range(10):
            cycle_data = {
                "cycle_id": i,
                "experiments": [{
                    "experiment_id": f"exp_{i}",
                    "status": "completed",
                    "config": {"model": "resnet"},
                    "metrics": {"auc": 0.80}  # Same accuracy
                }],
                "proposals": []
            }
            summarizer.summarize_cycle(i, cycle_data)

        executive = summarizer.create_executive_summary(last_n_cycles=10)

        assert executive.stagnation_detected is True

    def test_suggestions_generated(self, summarizer, sample_cycle_data):
        """Test that suggestions are generated."""
        for i in range(3):
            cycle_data = sample_cycle_data.copy()
            cycle_data["cycle_id"] = i
            summarizer.summarize_cycle(i, cycle_data)

        executive = summarizer.create_executive_summary()

        assert len(executive.suggested_directions) > 0

    def test_persistence(self, temp_storage, sample_cycle_data):
        """Test that summaries are persisted and loaded."""
        # Create and populate summarizer
        summarizer1 = CrossCycleSummarizer(storage_path=temp_storage)
        summarizer1.summarize_cycle(1, sample_cycle_data)
        summarizer1.summarize_cycle(2, sample_cycle_data)

        # Create new summarizer with same storage
        summarizer2 = CrossCycleSummarizer(storage_path=temp_storage)

        # Should have loaded existing summaries
        stats = summarizer2.get_summary_stats()
        assert stats["num_cycle_summaries"] == 2

    def test_get_summary_stats(self, summarizer, sample_cycle_data):
        """Test getting summary statistics."""
        summarizer.summarize_cycle(1, sample_cycle_data)
        summarizer.summarize_cycle(2, sample_cycle_data)

        stats = summarizer.get_summary_stats()

        assert stats["num_cycle_summaries"] == 2
        assert 1 in stats["cycles_covered"]
        assert 2 in stats["cycles_covered"]


class TestSummarizerSingleton:
    """Test singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_summarizer()
        yield
        reset_summarizer()

    def test_get_same_instance(self):
        """Test that get_summarizer returns same instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            instance1 = get_summarizer(storage_path=temp_dir)
            instance2 = get_summarizer()

            assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test that reset creates new instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            instance1 = get_summarizer(storage_path=temp_dir)
            reset_summarizer()

            with tempfile.TemporaryDirectory() as temp_dir2:
                instance2 = get_summarizer(storage_path=temp_dir2)

                assert instance1 is not instance2


class TestTokenEstimation:
    """Test token estimation."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_token_count_in_cycle_summary(self, temp_storage):
        """Test that token count is estimated in cycle summary."""
        summarizer = CrossCycleSummarizer(storage_path=temp_storage)

        cycle_data = {
            "cycle_id": 1,
            "experiments": [{
                "experiment_id": "exp_001",
                "status": "completed",
                "config": {"model": "resnet"},
                "metrics": {"auc": 0.85}
            }],
            "proposals": []
        }

        summary = summarizer.summarize_cycle(1, cycle_data)

        assert summary.token_count > 0

    def test_token_count_in_executive_summary(self, temp_storage):
        """Test that token count is estimated in executive summary."""
        summarizer = CrossCycleSummarizer(storage_path=temp_storage)

        for i in range(3):
            summarizer.summarize_cycle(i, {
                "cycle_id": i,
                "experiments": [],
                "proposals": []
            })

        executive = summarizer.create_executive_summary()

        assert executive.token_count > 0


class TestAgentFocusAreas:
    """Test agent-specific focus area filtering."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_architect_focus(self, temp_storage):
        """Test architect agent focus areas."""
        summarizer = CrossCycleSummarizer(storage_path=temp_storage)

        # Add cycle with architecture-related content
        summarizer.summarize_cycle(1, {
            "cycle_id": 1,
            "experiments": [{
                "experiment_id": "exp_001",
                "status": "completed",
                "config": {"model": "architecture_test"},
                "metrics": {"auc": 0.85}
            }],
            "proposals": []
        })

        working = summarizer.create_working_summary("architect", max_tokens=5000)

        assert "architecture" in working["focus_areas"]

    def test_executor_focus(self, temp_storage):
        """Test executor agent focus areas."""
        summarizer = CrossCycleSummarizer(storage_path=temp_storage)

        summarizer.summarize_cycle(1, {
            "cycle_id": 1,
            "experiments": [{
                "experiment_id": "exp_001",
                "status": "completed",
                "config": {},
                "metrics": {"auc": 0.85}
            }],
            "proposals": []
        })

        working = summarizer.create_working_summary("executor", max_tokens=5000)

        assert "training" in working["focus_areas"]
        assert "execution" in working["focus_areas"]

    def test_unknown_agent_no_focus(self, temp_storage):
        """Test that unknown agent gets no specific focus."""
        summarizer = CrossCycleSummarizer(storage_path=temp_storage)

        summarizer.summarize_cycle(1, {
            "cycle_id": 1,
            "experiments": [],
            "proposals": []
        })

        working = summarizer.create_working_summary("unknown_agent", max_tokens=5000)

        assert working["focus_areas"] == []
