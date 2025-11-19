"""
Unified Memory Handler

Provides a consistent, validated interface for all memory file operations.
Acts as the bridge between the schema/config layer and the multi-agent system.

All memory I/O goes through this handler to ensure:
- Schema validation on every read/write
- Atomic writes with rollback capability
- Consistent error handling
- Audit trail logging
- Thread-safe operations
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, List
from datetime import datetime
from contextlib import contextmanager
import threading

from pydantic import BaseModel, ValidationError

# Import all schemas
from schemas import (
    Directive, HistorySummary, Constraints, SystemState,
    Proposals, Reviews, ExperimentRecord,
    validate_memory_file, save_memory_file, create_default_memory_files
)

# Import configuration
from config import get_settings, ARCSettings


# Type variable for generic schema operations
T = TypeVar('T', bound=BaseModel)


# Configure logging
logger = logging.getLogger(__name__)


class MemoryHandlerError(Exception):
    """Base exception for memory handler errors."""
    pass


class ValidationFailedError(MemoryHandlerError):
    """Raised when schema validation fails."""
    pass


class AtomicWriteError(MemoryHandlerError):
    """Raised when atomic write operation fails."""
    pass


class MemoryHandler:
    """
    Unified handler for all ARC memory file operations.

    Provides schema-validated, atomic read/write operations with
    thread safety and audit logging.

    Example:
        handler = MemoryHandler()

        # Read with validation
        directive = handler.load_directive()

        # Update and save atomically
        directive.cycle_id += 1
        handler.save_directive(directive)

        # Or use context manager for automatic rollback
        with handler.transaction():
            state = handler.load_system_state()
            state.status = "running"
            handler.save_system_state(state)
    """

    # Schema mapping for each memory file
    SCHEMA_MAP = {
        'directive.json': Directive,
        'history_summary.json': HistorySummary,
        'constraints.json': Constraints,
        'system_state.json': SystemState,
        'proposals.json': Proposals,
        'reviews.json': Reviews,
    }

    def __init__(self, settings: Optional[ARCSettings] = None):
        """
        Initialize memory handler.

        Args:
            settings: Optional ARC settings. If None, uses get_settings()
        """
        self.settings = settings or get_settings()
        self.memory_dir = self.settings.memory_dir

        # Ensure memory directory exists
        self.settings.ensure_directories()

        # Thread safety
        self._lock = threading.Lock()

        # Transaction support
        self._in_transaction = False
        self._transaction_backups: Dict[str, Any] = {}

        logger.info(f"MemoryHandler initialized with memory_dir={self.memory_dir}")

    # ========================================================================
    # Generic Load/Save Operations
    # ========================================================================

    def load(self, filename: str, schema_class: Type[T]) -> T:
        """
        Load and validate a memory file.

        Args:
            filename: Memory file name (e.g., 'directive.json')
            schema_class: Pydantic model class for validation

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationFailedError: If file doesn't match schema
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.memory_dir / filename

        with self._lock:
            try:
                logger.debug(f"Loading {filename} from {file_path}")
                model = validate_memory_file(str(file_path), schema_class)
                logger.debug(f"Successfully loaded and validated {filename}")
                return model
            except ValidationError as e:
                logger.error(f"Schema validation failed for {filename}: {e}")
                raise ValidationFailedError(f"Invalid schema in {filename}: {e}")
            except FileNotFoundError:
                logger.warning(f"Memory file not found: {filename}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {filename}: {e}")
                raise ValidationFailedError(f"Malformed JSON in {filename}: {e}")

    def save(self, filename: str, model: BaseModel, atomic: bool = True) -> None:
        """
        Save a Pydantic model to a memory file.

        Args:
            filename: Memory file name (e.g., 'directive.json')
            model: Pydantic model instance to save
            atomic: Use atomic write (recommended)

        Raises:
            AtomicWriteError: If atomic write fails
        """
        file_path = self.memory_dir / filename

        with self._lock:
            try:
                logger.debug(f"Saving {filename} to {file_path} (atomic={atomic})")

                # Backup for transaction support
                if self._in_transaction and file_path.exists():
                    with open(file_path, 'r') as f:
                        self._transaction_backups[filename] = f.read()

                save_memory_file(str(file_path), model, atomic=atomic)
                logger.debug(f"Successfully saved {filename}")

            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")
                raise AtomicWriteError(f"Save failed for {filename}: {e}")

    # ========================================================================
    # Convenience Methods for Specific Memory Files
    # ========================================================================

    def load_directive(self) -> Directive:
        """Load and validate directive.json."""
        return self.load('directive.json', Directive)

    def save_directive(self, directive: Directive) -> None:
        """Save validated directive.json."""
        self.save('directive.json', directive)

    def load_history_summary(self) -> HistorySummary:
        """Load and validate history_summary.json."""
        return self.load('history_summary.json', HistorySummary)

    def save_history_summary(self, history: HistorySummary) -> None:
        """Save validated history_summary.json."""
        self.save('history_summary.json', history)

    def load_constraints(self) -> Constraints:
        """Load and validate constraints.json."""
        return self.load('constraints.json', Constraints)

    def save_constraints(self, constraints: Constraints) -> None:
        """Save validated constraints.json."""
        self.save('constraints.json', constraints)

    def load_system_state(self) -> SystemState:
        """Load and validate system_state.json."""
        return self.load('system_state.json', SystemState)

    def save_system_state(self, state: SystemState) -> None:
        """Save validated system_state.json."""
        self.save('system_state.json', state)

    def load_proposals(self) -> Proposals:
        """Load and validate proposals.json."""
        return self.load('proposals.json', Proposals)

    def save_proposals(self, proposals: Proposals) -> None:
        """Save validated proposals.json."""
        self.save('proposals.json', proposals)

    def load_reviews(self) -> Reviews:
        """Load and validate reviews.json."""
        return self.load('reviews.json', Reviews)

    def save_reviews(self, reviews: Reviews) -> None:
        """Save validated reviews.json."""
        self.save('reviews.json', reviews)

    # ========================================================================
    # Transaction Support
    # ========================================================================

    @contextmanager
    def transaction(self):
        """
        Context manager for transactional memory operations.

        Automatically rolls back changes if an exception occurs.

        Example:
            with handler.transaction():
                state = handler.load_system_state()
                state.status = "running"
                handler.save_system_state(state)
                # If any error occurs, changes are rolled back
        """
        self._in_transaction = True
        self._transaction_backups = {}

        try:
            yield self
            logger.debug("Transaction committed successfully")
        except Exception as e:
            logger.error(f"Transaction failed, rolling back: {e}")
            self._rollback()
            raise
        finally:
            self._in_transaction = False
            self._transaction_backups = {}

    def _rollback(self) -> None:
        """Rollback transaction by restoring backups."""
        for filename, backup_content in self._transaction_backups.items():
            file_path = self.memory_dir / filename
            try:
                with open(file_path, 'w') as f:
                    f.write(backup_content)
                logger.info(f"Rolled back {filename}")
            except Exception as e:
                logger.error(f"Failed to rollback {filename}: {e}")

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def load_all_memory(self) -> Dict[str, BaseModel]:
        """
        Load all memory files at once.

        Returns:
            Dictionary mapping filename to validated model

        Example:
            memory = handler.load_all_memory()
            directive = memory['directive.json']
            history = memory['history_summary.json']
        """
        result = {}

        for filename, schema_class in self.SCHEMA_MAP.items():
            file_path = self.memory_dir / filename
            if file_path.exists():
                try:
                    result[filename] = self.load(filename, schema_class)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")

        return result

    def validate_all_memory(self) -> tuple[bool, List[str]]:
        """
        Validate all existing memory files.

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []

        for filename, schema_class in self.SCHEMA_MAP.items():
            file_path = self.memory_dir / filename

            if not file_path.exists():
                errors.append(f"{filename}: File does not exist")
                continue

            try:
                self.load(filename, schema_class)
            except ValidationFailedError as e:
                errors.append(f"{filename}: {e}")
            except Exception as e:
                errors.append(f"{filename}: Unexpected error: {e}")

        return len(errors) == 0, errors

    # ========================================================================
    # Initialization & Maintenance
    # ========================================================================

    def initialize_memory(self, force: bool = False) -> None:
        """
        Initialize memory directory with default files.

        Args:
            force: If True, overwrite existing files
        """
        logger.info(f"Initializing memory at {self.memory_dir}")

        if not force:
            # Use schemas.py function which only creates missing files
            create_default_memory_files(str(self.memory_dir))
        else:
            # Force re-initialize all files
            from schemas import (
                Directive, DirectiveMode, Objective, NoveltyBudget,
                HistorySummary, Constraints, SystemState, OperatingMode
            )

            defaults = {
                'directive.json': Directive(
                    cycle_id=0,
                    mode=DirectiveMode.EXPLORE,
                    objective=Objective.IMPROVE_AUC,
                    novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=0),
                    notes="Initial directive - exploration phase"
                ),
                'history_summary.json': HistorySummary(),
                'constraints.json': Constraints(),
                'system_state.json': SystemState(
                    llm_endpoint=self.settings.llm_endpoint,
                    mode=OperatingMode.SEMI,
                    status="idle"
                ),
            }

            for filename, model in defaults.items():
                self.save(filename, model)
                logger.info(f"Initialized {filename}")

    def backup_memory(self, backup_dir: Optional[Path] = None) -> Path:
        """
        Create a backup of all memory files.

        Args:
            backup_dir: Optional backup directory. If None, uses snapshots_dir

        Returns:
            Path to backup directory
        """
        if backup_dir is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.settings.snapshots_dir / f"backup_{timestamp}"

        backup_dir.mkdir(parents=True, exist_ok=True)

        for filename in self.SCHEMA_MAP.keys():
            source = self.memory_dir / filename
            if source.exists():
                dest = backup_dir / filename
                import shutil
                shutil.copy2(source, dest)

        logger.info(f"Memory backed up to {backup_dir}")
        return backup_dir

    def restore_memory(self, backup_dir: Path) -> None:
        """
        Restore memory from a backup.

        Args:
            backup_dir: Directory containing backup files
        """
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_dir}")

        for filename in self.SCHEMA_MAP.keys():
            source = backup_dir / filename
            if source.exists():
                dest = self.memory_dir / filename
                import shutil
                shutil.copy2(source, dest)

        logger.info(f"Memory restored from {backup_dir}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_file_path(self, filename: str) -> Path:
        """Get full path for a memory file."""
        return self.memory_dir / filename

    def file_exists(self, filename: str) -> bool:
        """Check if a memory file exists."""
        return self.get_file_path(filename).exists()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory files.

        Returns:
            Dictionary with file sizes, modification times, etc.
        """
        stats = {}

        for filename in self.SCHEMA_MAP.keys():
            file_path = self.get_file_path(filename)
            if file_path.exists():
                stat = file_path.stat()
                stats[filename] = {
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'exists': True
                }
            else:
                stats[filename] = {'exists': False}

        return stats


# ============================================================================
# Global Instance (Singleton Pattern)
# ============================================================================

_global_handler: Optional[MemoryHandler] = None


def get_memory_handler(settings: Optional[ARCSettings] = None) -> MemoryHandler:
    """
    Get global memory handler instance (singleton).

    Args:
        settings: Optional settings override

    Returns:
        MemoryHandler instance
    """
    global _global_handler

    if _global_handler is None or settings is not None:
        _global_handler = MemoryHandler(settings)

    return _global_handler


def reset_memory_handler() -> None:
    """Reset global memory handler (useful for testing)."""
    global _global_handler
    _global_handler = None
