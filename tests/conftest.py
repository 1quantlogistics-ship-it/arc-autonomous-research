"""
Pytest configuration and shared fixtures for ARC test suite.

This module provides fixtures for:
- Test settings and configuration
- Temporary directory structures
- Mock memory files
- LLM client mocks
- Common test data
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from datetime import datetime

# Import ARC modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ARCSettings, reset_settings_cache  # noqa: E402 - imports from root config.py
from schemas import (
    Directive, DirectiveMode, Objective, NoveltyBudget,
    HistorySummary, BestMetrics, ExperimentRecord, PerformanceTrends,
    Constraints, ForbiddenRange,
    SystemState, OperatingMode,
    Proposals, Proposal, NoveltyClass, ExpectedImpact, ResourceCost,
    Reviews, Review, ReviewDecision,
    TrendDirection
)


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


# ============================================================================
# Function-level Fixtures - Test Environment
# ============================================================================

@pytest.fixture
def temp_arc_home(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary ARC home directory structure.

    Yields:
        Path to temporary ARC home directory with full structure
    """
    arc_home = tmp_path / "arc"

    # Create directory structure
    (arc_home / "memory").mkdir(parents=True, exist_ok=True)
    (arc_home / "experiments").mkdir(parents=True, exist_ok=True)
    (arc_home / "logs").mkdir(parents=True, exist_ok=True)
    (arc_home / "checkpoints").mkdir(parents=True, exist_ok=True)
    (arc_home / "snapshots").mkdir(parents=True, exist_ok=True)

    yield arc_home

    # Cleanup
    if arc_home.exists():
        shutil.rmtree(arc_home)


@pytest.fixture
def test_settings(temp_arc_home: Path) -> Generator[ARCSettings, None, None]:
    """
    Create test settings with temporary directories.

    Args:
        temp_arc_home: Temporary ARC home directory

    Yields:
        ARCSettings configured for testing
    """
    reset_settings_cache()

    settings = ARCSettings(
        environment="test",
        home=temp_arc_home,
        llm_endpoint="http://localhost:8000/v1",
        mode="SEMI",
        log_level="DEBUG"
    )

    settings.ensure_directories()

    yield settings

    reset_settings_cache()


# ============================================================================
# Memory File Fixtures - Default Valid Data
# ============================================================================

@pytest.fixture
def default_directive() -> Directive:
    """Create a valid default directive."""
    return Directive(
        cycle_id=1,
        mode=DirectiveMode.EXPLORE,
        objective=Objective.IMPROVE_AUC,
        novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1),
        focus_areas=["learning_rate", "architecture"],
        forbidden_axes=["dataset"],
        encouraged_axes=["optimizer", "loss_function"],
        notes="Test directive for exploration",
        timestamp=datetime.utcnow().isoformat()
    )


@pytest.fixture
def default_history_summary() -> HistorySummary:
    """Create a valid default history summary."""
    return HistorySummary(
        total_cycles=5,
        total_experiments=15,
        best_metrics=BestMetrics(
            auc=0.85,
            sensitivity=0.82,
            specificity=0.88
        ),
        recent_experiments=[
            ExperimentRecord(
                experiment_id="exp_5_1",
                auc=0.85,
                sensitivity=0.82,
                specificity=0.88,
                training_time=120.5,
                timestamp=datetime.utcnow().isoformat(),
                success=True
            ),
            ExperimentRecord(
                experiment_id="exp_5_2",
                auc=0.83,
                sensitivity=0.80,
                specificity=0.86,
                training_time=115.2,
                timestamp=datetime.utcnow().isoformat(),
                success=True
            )
        ],
        failed_configs=[],
        successful_patterns=[{"learning_rate": 0.001, "batch_size": 32}],
        performance_trends=PerformanceTrends(
            auc_trend=TrendDirection.IMPROVING,
            sensitivity_trend=TrendDirection.STABLE,
            specificity_trend=TrendDirection.IMPROVING,
            cycles_without_improvement=0,
            consecutive_regressions=0
        ),
        last_updated=datetime.utcnow().isoformat()
    )


@pytest.fixture
def default_constraints() -> Constraints:
    """Create valid default constraints."""
    return Constraints(
        forbidden_ranges=[
            ForbiddenRange(
                param="learning_rate",
                min=0.1,
                max=1.0,
                reason="Causes training instability"
            )
        ],
        unstable_configs=[],
        safe_baselines=[{"learning_rate": 0.001, "batch_size": 32}],
        max_learning_rate=0.1,
        min_batch_size=8,
        max_batch_size=256,
        last_updated=datetime.utcnow().isoformat()
    )


@pytest.fixture
def default_system_state() -> SystemState:
    """Create valid default system state."""
    return SystemState(
        mode=OperatingMode.SEMI,
        arc_version="1.1.0",
        llm_endpoint="http://localhost:8000/v1",
        last_cycle_id=5,
        last_cycle_timestamp=datetime.utcnow().isoformat(),
        status="idle",
        active_experiments=[],
        error_count=0
    )


@pytest.fixture
def default_proposals() -> Proposals:
    """Create valid default proposals."""
    return Proposals(
        cycle_id=6,
        proposals=[
            Proposal(
                experiment_id="exp_6_1",
                novelty_class=NoveltyClass.EXPLOIT,
                hypothesis="Reducing learning rate will improve convergence",
                changes={"learning_rate": 0.0005},
                expected_impact=ExpectedImpact(auc="up", sensitivity="same", specificity="same"),
                resource_cost=ResourceCost.LOW,
                rationale="Historical data shows this range performs well"
            ),
            Proposal(
                experiment_id="exp_6_2",
                novelty_class=NoveltyClass.EXPLORE,
                hypothesis="Adding dropout will reduce overfitting",
                changes={"dropout": 0.3, "learning_rate": 0.001},
                expected_impact=ExpectedImpact(auc="up", sensitivity="up", specificity="same"),
                resource_cost=ResourceCost.MEDIUM,
                rationale="Literature suggests dropout helps with medical imaging"
            )
        ],
        timestamp=datetime.utcnow().isoformat()
    )


@pytest.fixture
def default_reviews() -> Reviews:
    """Create valid default reviews."""
    return Reviews(
        cycle_id=6,
        reviews=[
            Review(
                proposal_id="exp_6_1",
                decision=ReviewDecision.APPROVE,
                issues=[],
                reasoning="Safe exploit experiment with proven track record",
                risk_level="low"
            ),
            Review(
                proposal_id="exp_6_2",
                decision=ReviewDecision.APPROVE,
                issues=["Slightly higher resource cost"],
                reasoning="Good hypothesis backed by literature, worth trying",
                risk_level="medium"
            )
        ],
        approved=["exp_6_1", "exp_6_2"],
        rejected=[],
        revise=[],
        timestamp=datetime.utcnow().isoformat()
    )


# ============================================================================
# Memory File Fixtures - With Files on Disk
# ============================================================================

@pytest.fixture
def memory_files(test_settings: ARCSettings, default_directive: Directive,
                default_history_summary: HistorySummary, default_constraints: Constraints,
                default_system_state: SystemState) -> Dict[str, Path]:
    """
    Create all memory files on disk with valid defaults.

    Returns:
        Dictionary mapping memory file type to file path
    """
    from schemas import save_memory_file

    memory_dir = test_settings.memory_dir
    files = {}

    # Save all default memory files
    files['directive'] = memory_dir / "directive.json"
    save_memory_file(str(files['directive']), default_directive)

    files['history_summary'] = memory_dir / "history_summary.json"
    save_memory_file(str(files['history_summary']), default_history_summary)

    files['constraints'] = memory_dir / "constraints.json"
    save_memory_file(str(files['constraints']), default_constraints)

    files['system_state'] = memory_dir / "system_state.json"
    save_memory_file(str(files['system_state']), default_system_state)

    return files


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_experiment_result() -> Dict[str, Any]:
    """Sample experiment result data."""
    return {
        "experiment_id": "exp_test_1",
        "status": "completed",
        "metrics": {
            "auc": 0.87,
            "sensitivity": 0.84,
            "specificity": 0.90,
            "loss": 0.23
        },
        "training_time": 125.7,
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """Sample training configuration."""
    return {
        "model": {
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 2
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "optimizer": "adam"
        },
        "data": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1
        }
    }


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Helper Functions for Tests
# ============================================================================

@pytest.fixture
def assert_valid_json():
    """Helper to validate JSON files."""
    def _validator(file_path: Path) -> Dict[str, Any]:
        """Load and validate JSON file."""
        assert file_path.exists(), f"File does not exist: {file_path}"
        with open(file_path) as f:
            data = json.load(f)
        assert isinstance(data, dict), "JSON root must be an object"
        return data
    return _validator


@pytest.fixture
def assert_schema_valid():
    """Helper to validate data against Pydantic schemas."""
    from pydantic import ValidationError

    def _validator(data: Dict[str, Any], schema_class):
        """Validate data against schema."""
        try:
            instance = schema_class(**data)
            return instance
        except ValidationError as e:
            pytest.fail(f"Schema validation failed: {e}")

    return _validator


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "llm: Tests requiring LLM mock")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
