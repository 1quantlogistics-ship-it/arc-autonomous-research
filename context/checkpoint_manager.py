"""
Checkpoint Manager for Crash Recovery

Phase G - Critical path for crash recovery. User explicitly stated
crash recovery is "very painful" - this addresses that pain point.

Key features:
- Atomic writes using write-then-rename pattern
- Checkpoint after each agent completion
- Fast recovery from latest checkpoint
- Retain last 10 checkpoints (configurable)
"""

import json
import os
import shutil
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CycleCheckpoint:
    """
    Represents a checkpoint of cycle state at a specific phase.

    Attributes:
        cycle_id: The current cycle number
        phase: Current phase ('planning', 'voting', 'executing', 'reviewing')
        agent_states: State of each agent at checkpoint time
        memory_snapshot: Snapshot of memory files relevant to recovery
        timestamp: ISO format timestamp of checkpoint creation
        checkpoint_id: Unique identifier for this checkpoint
        metadata: Additional metadata (version, etc.)
    """
    cycle_id: int
    phase: str  # 'planning', 'voting', 'executing', 'reviewing'
    agent_states: Dict[str, Any]
    memory_snapshot: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checkpoint_id: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate checkpoint_id if not provided."""
        if not self.checkpoint_id:
            self.checkpoint_id = f"ckpt_{self.cycle_id}_{self.phase}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleCheckpoint":
        """Create CycleCheckpoint from dictionary."""
        return cls(**data)

    @property
    def age_seconds(self) -> float:
        """Get age of checkpoint in seconds."""
        created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds()


class CheckpointManager:
    """
    Manages checkpoint persistence and recovery for cycle state.

    Uses atomic writes (write-then-rename) to prevent corruption.
    Maintains a configurable number of recent checkpoints.

    Attributes:
        checkpoint_dir: Directory for storing checkpoints
        max_checkpoints: Maximum number of checkpoints to retain
    """

    VALID_PHASES = frozenset(['planning', 'voting', 'executing', 'reviewing'])

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 10
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints (default: ./checkpoints)
            max_checkpoints: Maximum checkpoints to retain (default: 10)
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.max_checkpoints = max_checkpoints
        self._lock = Lock()

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: dir={self.checkpoint_dir}, max={max_checkpoints}")

    def save_checkpoint(
        self,
        cycle_id: int,
        phase: str,
        agent_states: Dict[str, Any],
        memory_snapshot: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint atomically using write-then-rename.

        Args:
            cycle_id: Current cycle number
            phase: Current phase ('planning', 'voting', 'executing', 'reviewing')
            agent_states: State of each agent
            memory_snapshot: Snapshot of memory state
            metadata: Optional additional metadata

        Returns:
            Path to the saved checkpoint file

        Raises:
            ValueError: If phase is not valid
            IOError: If write fails
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of: {self.VALID_PHASES}")

        checkpoint = CycleCheckpoint(
            cycle_id=cycle_id,
            phase=phase,
            agent_states=agent_states,
            memory_snapshot=memory_snapshot,
            metadata=metadata or {"version": "1.0"}
        )

        return self._write_checkpoint(checkpoint)

    def _write_checkpoint(self, checkpoint: CycleCheckpoint) -> Path:
        """
        Write checkpoint atomically using write-then-rename pattern.

        This prevents partial/corrupt checkpoints from incomplete writes.
        """
        final_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        temp_path = self.checkpoint_dir / f".tmp_{checkpoint.checkpoint_id}.json"

        with self._lock:
            try:
                # Write to temporary file
                with open(temp_path, 'w') as f:
                    json.dump(checkpoint.to_dict(), f, indent=2)
                    # Ensure data is flushed to disk
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (on POSIX systems)
                shutil.move(str(temp_path), str(final_path))

                logger.info(f"Checkpoint saved: {checkpoint.checkpoint_id}")

                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()

                return final_path

            except Exception as e:
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                logger.error(f"Failed to save checkpoint: {e}")
                raise IOError(f"Failed to save checkpoint: {e}") from e

    def _cleanup_old_checkpoints(self) -> int:
        """
        Remove old checkpoints beyond max_checkpoints limit.

        Returns:
            Number of checkpoints removed
        """
        checkpoints = self._list_checkpoint_files()

        if len(checkpoints) <= self.max_checkpoints:
            return 0

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.max_checkpoints]
        removed = 0

        for ckpt_path in to_remove:
            try:
                ckpt_path.unlink()
                removed += 1
                logger.debug(f"Removed old checkpoint: {ckpt_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {ckpt_path}: {e}")

        return removed

    def _list_checkpoint_files(self) -> List[Path]:
        """List all checkpoint files (excluding temp files)."""
        return [
            p for p in self.checkpoint_dir.glob("ckpt_*.json")
            if not p.name.startswith(".tmp_")
        ]

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CycleCheckpoint]:
        """
        Load a specific checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint identifier

        Returns:
            CycleCheckpoint if found, None otherwise
        """
        # Handle both with and without .json extension
        if not checkpoint_id.endswith('.json'):
            checkpoint_id = f"{checkpoint_id}.json"

        ckpt_path = self.checkpoint_dir / checkpoint_id

        if not ckpt_path.exists():
            # Try without .json in case it was passed with .json.json
            ckpt_path = self.checkpoint_dir / checkpoint_id.replace('.json', '')
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return None

        return self._read_checkpoint(ckpt_path)

    def load_latest_checkpoint(self) -> Optional[CycleCheckpoint]:
        """
        Load the most recent checkpoint.

        Returns:
            Most recent CycleCheckpoint, or None if no checkpoints exist
        """
        checkpoints = self._list_checkpoint_files()

        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return self._read_checkpoint(checkpoints[0])

    def get_recovery_point(self, cycle_id: int) -> Optional[CycleCheckpoint]:
        """
        Get the best recovery point for a specific cycle.

        Returns the most recent checkpoint for the given cycle,
        or the latest checkpoint before that cycle if none exists.

        Args:
            cycle_id: The cycle to recover

        Returns:
            Best CycleCheckpoint for recovery, or None if none available
        """
        checkpoints = self._list_checkpoint_files()

        if not checkpoints:
            return None

        # Load all checkpoints and filter
        cycle_checkpoints = []
        fallback_checkpoint = None
        fallback_mtime = 0

        for ckpt_path in checkpoints:
            try:
                ckpt = self._read_checkpoint(ckpt_path)
                if ckpt:
                    if ckpt.cycle_id == cycle_id:
                        cycle_checkpoints.append((ckpt_path, ckpt))
                    elif ckpt.cycle_id < cycle_id:
                        mtime = ckpt_path.stat().st_mtime
                        if mtime > fallback_mtime:
                            fallback_mtime = mtime
                            fallback_checkpoint = ckpt
            except Exception as e:
                logger.warning(f"Error reading checkpoint {ckpt_path}: {e}")
                continue

        # Return most recent checkpoint for this cycle
        if cycle_checkpoints:
            cycle_checkpoints.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            return cycle_checkpoints[0][1]

        # Fall back to most recent checkpoint before this cycle
        return fallback_checkpoint

    def _read_checkpoint(self, path: Path) -> Optional[CycleCheckpoint]:
        """Read and parse a checkpoint file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return CycleCheckpoint.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Corrupt checkpoint file {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading checkpoint {path}: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of checkpoint info dicts (id, cycle_id, phase, timestamp)
        """
        checkpoints = self._list_checkpoint_files()
        results = []

        for ckpt_path in checkpoints:
            ckpt = self._read_checkpoint(ckpt_path)
            if ckpt:
                results.append({
                    "checkpoint_id": ckpt.checkpoint_id,
                    "cycle_id": ckpt.cycle_id,
                    "phase": ckpt.phase,
                    "timestamp": ckpt.timestamp,
                    "age_seconds": ckpt.age_seconds,
                    "file_size": ckpt_path.stat().st_size
                })

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        if not checkpoint_id.endswith('.json'):
            checkpoint_id = f"{checkpoint_id}.json"

        ckpt_path = self.checkpoint_dir / checkpoint_id

        if not ckpt_path.exists():
            return False

        with self._lock:
            try:
                ckpt_path.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                return False

    def clear_all_checkpoints(self) -> int:
        """
        Delete all checkpoints. Use with caution.

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self._list_checkpoint_files()
        deleted = 0

        with self._lock:
            for ckpt_path in checkpoints:
                try:
                    ckpt_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {ckpt_path}: {e}")

        logger.info(f"Cleared {deleted} checkpoints")
        return deleted

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get statistics about checkpoints.

        Returns:
            Dict with count, total_size, oldest, newest info
        """
        checkpoints = self._list_checkpoint_files()

        if not checkpoints:
            return {
                "count": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
                "cycles_covered": []
            }

        total_size = sum(p.stat().st_size for p in checkpoints)
        mtimes = [(p, p.stat().st_mtime) for p in checkpoints]
        mtimes.sort(key=lambda x: x[1])

        # Get unique cycles
        cycles = set()
        for ckpt_path in checkpoints:
            ckpt = self._read_checkpoint(ckpt_path)
            if ckpt:
                cycles.add(ckpt.cycle_id)

        oldest_ckpt = self._read_checkpoint(mtimes[0][0])
        newest_ckpt = self._read_checkpoint(mtimes[-1][0])

        return {
            "count": len(checkpoints),
            "total_size_bytes": total_size,
            "max_checkpoints": self.max_checkpoints,
            "oldest": oldest_ckpt.checkpoint_id if oldest_ckpt else None,
            "newest": newest_ckpt.checkpoint_id if newest_ckpt else None,
            "cycles_covered": sorted(cycles)
        }


# Singleton pattern for global access
_checkpoint_manager: Optional[CheckpointManager] = None
_manager_lock = Lock()


def get_checkpoint_manager(
    checkpoint_dir: Optional[Path] = None,
    max_checkpoints: int = 10
) -> CheckpointManager:
    """
    Get or create the global CheckpointManager instance.

    Args:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum checkpoints to retain

    Returns:
        The singleton CheckpointManager instance
    """
    global _checkpoint_manager

    with _manager_lock:
        if _checkpoint_manager is None:
            _checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=max_checkpoints
            )
        return _checkpoint_manager


def reset_checkpoint_manager() -> None:
    """Reset the global CheckpointManager instance (for testing)."""
    global _checkpoint_manager
    with _manager_lock:
        _checkpoint_manager = None
