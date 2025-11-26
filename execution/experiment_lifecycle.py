"""
Experiment Lifecycle Management

Dev 2 - Bulletproof Execution: Tracks experiment state through full lifecycle.

Features:
- ExperimentState enum with valid transitions
- ExperimentRecord for persistent tracking
- ExperimentRegistry for centralized experiment management
- Recovery of incomplete experiments on startup
- State machine enforcement for valid transitions

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentState(str, Enum):
    """
    Experiment lifecycle states.

    State machine transitions:
    PENDING -> QUEUED -> RUNNING -> COMPLETED | FAILED | CRASHED | TIMEOUT
    PENDING -> CANCELLED (can cancel before running)
    RUNNING -> PAUSED -> RUNNING (optional pause/resume)
    """
    PENDING = "pending"       # Created, not yet queued
    QUEUED = "queued"         # Queued for execution
    RUNNING = "running"       # Currently executing
    PAUSED = "paused"         # Temporarily paused
    COMPLETED = "completed"   # Successfully finished
    FAILED = "failed"         # Failed with error
    CRASHED = "crashed"       # Process crashed unexpectedly
    TIMEOUT = "timeout"       # Exceeded time limit
    CANCELLED = "cancelled"   # Cancelled by user/system

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (no further transitions)."""
        return self in {
            ExperimentState.COMPLETED,
            ExperimentState.FAILED,
            ExperimentState.CRASHED,
            ExperimentState.TIMEOUT,
            ExperimentState.CANCELLED,
        }

    @property
    def is_active(self) -> bool:
        """Check if experiment is actively running or waiting."""
        return self in {
            ExperimentState.PENDING,
            ExperimentState.QUEUED,
            ExperimentState.RUNNING,
            ExperimentState.PAUSED,
        }


# Valid state transitions
VALID_TRANSITIONS: Dict[ExperimentState, List[ExperimentState]] = {
    ExperimentState.PENDING: [
        ExperimentState.QUEUED,
        ExperimentState.CANCELLED,
    ],
    ExperimentState.QUEUED: [
        ExperimentState.RUNNING,
        ExperimentState.CANCELLED,
    ],
    ExperimentState.RUNNING: [
        ExperimentState.COMPLETED,
        ExperimentState.FAILED,
        ExperimentState.CRASHED,
        ExperimentState.TIMEOUT,
        ExperimentState.PAUSED,
        ExperimentState.CANCELLED,
    ],
    ExperimentState.PAUSED: [
        ExperimentState.RUNNING,
        ExperimentState.CANCELLED,
    ],
    # Terminal states have no valid transitions
    ExperimentState.COMPLETED: [],
    ExperimentState.FAILED: [],
    ExperimentState.CRASHED: [],
    ExperimentState.TIMEOUT: [],
    ExperimentState.CANCELLED: [],
}


@dataclass
class ExperimentRecord:
    """
    Complete record of an experiment's lifecycle.

    Persisted to JSON for recovery across ARC restarts.
    """
    experiment_id: str
    cycle_id: int
    proposal_id: str

    # State tracking
    state: ExperimentState = ExperimentState.PENDING
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_seconds: Optional[int] = None
    timeout_seconds: Optional[int] = None

    # Process info
    pid: Optional[int] = None
    gpu_id: Optional[int] = None

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_state: ExperimentState, reason: str = "") -> bool:
        """
        Transition to a new state if valid.

        Args:
            new_state: Target state
            reason: Optional reason for transition

        Returns:
            True if transition successful, False otherwise
        """
        if new_state not in VALID_TRANSITIONS.get(self.state, []):
            logger.warning(
                f"Invalid transition: {self.experiment_id} cannot go from "
                f"{self.state.value} to {new_state.value}"
            )
            return False

        old_state = self.state
        self.state = new_state

        # Record transition
        now = datetime.now().isoformat()
        self.state_history.append({
            "from_state": old_state.value,
            "to_state": new_state.value,
            "timestamp": now,
            "reason": reason,
        })

        # Update timing fields
        if new_state == ExperimentState.QUEUED:
            self.queued_at = now
        elif new_state == ExperimentState.RUNNING:
            self.started_at = now
        elif new_state.is_terminal:
            self.completed_at = now

        logger.info(
            f"Experiment {self.experiment_id}: {old_state.value} -> {new_state.value}"
            f"{' (' + reason + ')' if reason else ''}"
        )

        return True

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate experiment duration if applicable."""
        if not self.started_at:
            return None

        start = datetime.fromisoformat(self.started_at)

        if self.completed_at:
            end = datetime.fromisoformat(self.completed_at)
        else:
            end = datetime.now()

        return (end - start).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enum to string
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Create from dictionary."""
        # Convert state string back to enum
        if isinstance(data.get("state"), str):
            data["state"] = ExperimentState(data["state"])
        return cls(**data)


class ExperimentRegistry:
    """
    Centralized registry for all experiments.

    Features:
    - Thread-safe operations
    - Persistent storage with atomic writes
    - Recovery of incomplete experiments on startup
    - Query experiments by state, cycle, etc.
    - Callbacks for state changes
    """

    def __init__(
        self,
        storage_path: str = "/workspace/arc/experiments",
        auto_persist: bool = True,
    ):
        """
        Initialize the experiment registry.

        Args:
            storage_path: Directory for persistent storage
            auto_persist: Whether to auto-save on changes
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.auto_persist = auto_persist
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._lock = Lock()
        self._state_callbacks: List[Callable[[ExperimentRecord, ExperimentState, ExperimentState], None]] = []

        # Load existing experiments
        self._load_from_disk()

        # Check for incomplete experiments
        self._recover_incomplete()

    def register(self, record: ExperimentRecord) -> ExperimentRecord:
        """
        Register a new experiment.

        Args:
            record: Experiment record to register

        Returns:
            Registered experiment record
        """
        with self._lock:
            if record.experiment_id in self._experiments:
                raise ValueError(f"Experiment {record.experiment_id} already registered")

            self._experiments[record.experiment_id] = record

            if self.auto_persist:
                self._persist_experiment(record)

            logger.info(f"Registered experiment: {record.experiment_id}")
            return record

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        with self._lock:
            return self._experiments.get(experiment_id)

    def update_state(
        self,
        experiment_id: str,
        new_state: ExperimentState,
        reason: str = "",
        **kwargs,
    ) -> bool:
        """
        Update experiment state.

        Args:
            experiment_id: Experiment to update
            new_state: Target state
            reason: Reason for transition
            **kwargs: Additional fields to update (metrics, error_message, etc.)

        Returns:
            True if successful
        """
        with self._lock:
            record = self._experiments.get(experiment_id)
            if not record:
                logger.error(f"Experiment not found: {experiment_id}")
                return False

            old_state = record.state

            if not record.transition_to(new_state, reason):
                return False

            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            if self.auto_persist:
                self._persist_experiment(record)

            # Fire callbacks
            self._fire_callbacks(record, old_state, new_state)

            return True

    def update_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        merge: bool = True,
    ) -> bool:
        """
        Update experiment metrics.

        Args:
            experiment_id: Experiment to update
            metrics: New metrics to add
            merge: If True, merge with existing; if False, replace

        Returns:
            True if successful
        """
        with self._lock:
            record = self._experiments.get(experiment_id)
            if not record:
                return False

            if merge:
                record.metrics.update(metrics)
            else:
                record.metrics = metrics

            if self.auto_persist:
                self._persist_experiment(record)

            return True

    def get_by_state(self, state: ExperimentState) -> List[ExperimentRecord]:
        """Get all experiments in a given state."""
        with self._lock:
            return [
                exp for exp in self._experiments.values()
                if exp.state == state
            ]

    def get_by_cycle(self, cycle_id: int) -> List[ExperimentRecord]:
        """Get all experiments for a cycle."""
        with self._lock:
            return [
                exp for exp in self._experiments.values()
                if exp.cycle_id == cycle_id
            ]

    def get_active(self) -> List[ExperimentRecord]:
        """Get all active (non-terminal) experiments."""
        with self._lock:
            return [
                exp for exp in self._experiments.values()
                if exp.state.is_active
            ]

    def get_incomplete(self) -> List[ExperimentRecord]:
        """Get experiments that were interrupted (RUNNING/QUEUED)."""
        with self._lock:
            return [
                exp for exp in self._experiments.values()
                if exp.state in {ExperimentState.RUNNING, ExperimentState.QUEUED}
            ]

    def add_state_callback(
        self,
        callback: Callable[[ExperimentRecord, ExperimentState, ExperimentState], None],
    ) -> None:
        """
        Add callback for state changes.

        Args:
            callback: Function(record, old_state, new_state)
        """
        self._state_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            state_counts = {}
            for state in ExperimentState:
                count = sum(1 for exp in self._experiments.values() if exp.state == state)
                if count > 0:
                    state_counts[state.value] = count

            return {
                "total_experiments": len(self._experiments),
                "active_experiments": sum(1 for exp in self._experiments.values() if exp.state.is_active),
                "completed_experiments": sum(1 for exp in self._experiments.values() if exp.state == ExperimentState.COMPLETED),
                "failed_experiments": sum(
                    1 for exp in self._experiments.values()
                    if exp.state in {ExperimentState.FAILED, ExperimentState.CRASHED, ExperimentState.TIMEOUT}
                ),
                "state_distribution": state_counts,
            }

    def persist_all(self) -> None:
        """Persist all experiments to disk."""
        with self._lock:
            for record in self._experiments.values():
                self._persist_experiment(record)

    def _persist_experiment(self, record: ExperimentRecord) -> None:
        """Persist single experiment with atomic write."""
        filepath = self.storage_path / f"{record.experiment_id}.json"
        temp_path = filepath.with_suffix(".tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(record.to_dict(), f, indent=2)

            temp_path.replace(filepath)

        except Exception as e:
            logger.error(f"Failed to persist experiment {record.experiment_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _load_from_disk(self) -> None:
        """Load all experiments from disk."""
        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                record = ExperimentRecord.from_dict(data)
                self._experiments[record.experiment_id] = record

            except Exception as e:
                logger.error(f"Failed to load experiment from {filepath}: {e}")

        logger.info(f"Loaded {len(self._experiments)} experiments from disk")

    def _recover_incomplete(self) -> None:
        """Mark interrupted experiments as crashed."""
        incomplete = self.get_incomplete()

        for record in incomplete:
            logger.warning(
                f"Recovering incomplete experiment: {record.experiment_id} "
                f"(was {record.state.value})"
            )

            # Mark as crashed since we don't know what happened
            record.transition_to(
                ExperimentState.CRASHED,
                reason="Recovered on startup - process was interrupted"
            )

            if self.auto_persist:
                self._persist_experiment(record)

    def _fire_callbacks(
        self,
        record: ExperimentRecord,
        old_state: ExperimentState,
        new_state: ExperimentState,
    ) -> None:
        """Fire state change callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(record, old_state, new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")


# Singleton pattern
_registry: Optional[ExperimentRegistry] = None
_registry_lock = Lock()


def get_experiment_registry(
    storage_path: str = "/workspace/arc/experiments",
    **kwargs,
) -> ExperimentRegistry:
    """
    Get or create the global ExperimentRegistry instance.

    Args:
        storage_path: Directory for persistent storage
        **kwargs: Additional arguments for first initialization

    Returns:
        The singleton ExperimentRegistry instance
    """
    global _registry

    with _registry_lock:
        if _registry is None:
            _registry = ExperimentRegistry(storage_path=storage_path, **kwargs)
        return _registry


def reset_experiment_registry() -> None:
    """Reset the global ExperimentRegistry instance (for testing)."""
    global _registry
    with _registry_lock:
        _registry = None
