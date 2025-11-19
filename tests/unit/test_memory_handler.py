"""
Unit tests for Unified Memory Handler.

Tests the memory handler's ability to provide schema-validated,
atomic, thread-safe memory operations.
"""

import pytest
import json
import shutil
from pathlib import Path
from pydantic import ValidationError

from memory_handler import (
    MemoryHandler, get_memory_handler, reset_memory_handler,
    MemoryHandlerError, ValidationFailedError, AtomicWriteError
)
from schemas import (
    Directive, DirectiveMode, Objective, NoveltyBudget,
    HistorySummary, Constraints, SystemState, OperatingMode,
    Proposals, Proposal, NoveltyClass, ExpectedImpact,
    Reviews, Review, ReviewDecision
)
from config import ARCSettings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_memory_handler(tmp_path):
    """Create a memory handler with temporary directory."""
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    handler = MemoryHandler(settings)
    handler.initialize_memory(force=True)

    return handler


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerInit:
    """Test memory handler initialization."""

    def test_create_handler(self, temp_memory_handler):
        """Test creating a memory handler."""
        assert temp_memory_handler is not None
        assert temp_memory_handler.memory_dir.exists()

    def test_initialize_memory_creates_defaults(self, temp_memory_handler):
        """Test that initialize_memory creates default files."""
        assert (temp_memory_handler.memory_dir / "directive.json").exists()
        assert (temp_memory_handler.memory_dir / "history_summary.json").exists()
        assert (temp_memory_handler.memory_dir / "constraints.json").exists()
        assert (temp_memory_handler.memory_dir / "system_state.json").exists()

    def test_file_exists_check(self, temp_memory_handler):
        """Test file_exists utility."""
        assert temp_memory_handler.file_exists("directive.json")
        assert not temp_memory_handler.file_exists("nonexistent.json")


# ============================================================================
# Load/Save Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerLoadSave:
    """Test basic load/save operations."""

    def test_load_directive(self, temp_memory_handler):
        """Test loading directive with validation."""
        directive = temp_memory_handler.load_directive()

        assert isinstance(directive, Directive)
        assert directive.cycle_id == 0
        assert directive.mode == DirectiveMode.EXPLORE

    def test_save_directive(self, temp_memory_handler):
        """Test saving directive."""
        directive = temp_memory_handler.load_directive()
        directive.cycle_id = 5
        directive.notes = "Updated directive"

        temp_memory_handler.save_directive(directive)

        # Reload and verify
        loaded = temp_memory_handler.load_directive()
        assert loaded.cycle_id == 5
        assert loaded.notes == "Updated directive"

    def test_load_history_summary(self, temp_memory_handler):
        """Test loading history summary."""
        history = temp_memory_handler.load_history_summary()

        assert isinstance(history, HistorySummary)
        assert history.total_cycles == 0
        assert history.total_experiments == 0

    def test_load_constraints(self, temp_memory_handler):
        """Test loading constraints."""
        constraints = temp_memory_handler.load_constraints()

        assert isinstance(constraints, Constraints)
        assert len(constraints.forbidden_ranges) == 0

    def test_load_system_state(self, temp_memory_handler):
        """Test loading system state."""
        state = temp_memory_handler.load_system_state()

        assert isinstance(state, SystemState)
        assert state.mode == OperatingMode.SEMI
        assert state.status == "idle"

    def test_generic_load_save(self, temp_memory_handler):
        """Test generic load/save methods."""
        # Load using generic method
        directive = temp_memory_handler.load('directive.json', Directive)
        assert isinstance(directive, Directive)

        # Modify and save
        directive.cycle_id = 10
        temp_memory_handler.save('directive.json', directive)

        # Reload
        loaded = temp_memory_handler.load('directive.json', Directive)
        assert loaded.cycle_id == 10


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerValidation:
    """Test schema validation during load/save."""

    def test_load_invalid_json_fails(self, temp_memory_handler):
        """Test that loading malformed JSON raises error."""
        # Write invalid JSON
        bad_file = temp_memory_handler.memory_dir / "directive.json"
        bad_file.write_text('{"invalid": json}')

        with pytest.raises(ValidationFailedError, match="Malformed JSON"):
            temp_memory_handler.load_directive()

    def test_load_wrong_schema_fails(self, temp_memory_handler):
        """Test that loading wrong schema raises error."""
        # Write valid JSON but wrong schema
        bad_file = temp_memory_handler.memory_dir / "directive.json"
        bad_file.write_text('{"wrong": "schema", "fields": "here"}')

        with pytest.raises(ValidationFailedError, match="Invalid schema"):
            temp_memory_handler.load_directive()

    def test_validate_all_memory(self, temp_memory_handler):
        """Test validating all memory files."""
        is_valid, errors = temp_memory_handler.validate_all_memory()

        assert is_valid
        assert len(errors) == 0

    def test_validate_all_memory_with_errors(self, temp_memory_handler):
        """Test validation catches errors in memory files."""
        # Corrupt one file
        bad_file = temp_memory_handler.memory_dir / "constraints.json"
        bad_file.write_text('{"broken": "data"}')

        is_valid, errors = temp_memory_handler.validate_all_memory()

        assert not is_valid
        assert any("constraints.json" in error for error in errors)


# ============================================================================
# Transaction Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerTransactions:
    """Test transactional memory operations."""

    def test_transaction_success(self, temp_memory_handler):
        """Test successful transaction commits changes."""
        with temp_memory_handler.transaction():
            state = temp_memory_handler.load_system_state()
            state.status = "running"
            temp_memory_handler.save_system_state(state)

        # Changes should be committed
        loaded = temp_memory_handler.load_system_state()
        assert loaded.status == "running"

    def test_transaction_rollback_on_error(self, temp_memory_handler):
        """Test transaction rolls back on error."""
        # Get initial state
        initial_state = temp_memory_handler.load_system_state()
        initial_status = initial_state.status

        # Try to modify in transaction that fails
        with pytest.raises(Exception):
            with temp_memory_handler.transaction():
                state = temp_memory_handler.load_system_state()
                state.status = "running"
                temp_memory_handler.save_system_state(state)
                raise Exception("Simulated error")

        # Changes should be rolled back
        loaded = temp_memory_handler.load_system_state()
        assert loaded.status == initial_status

    def test_nested_operations_in_transaction(self, temp_memory_handler):
        """Test multiple operations in one transaction."""
        with temp_memory_handler.transaction():
            # Modify multiple files
            state = temp_memory_handler.load_system_state()
            state.status = "running"
            temp_memory_handler.save_system_state(state)

            directive = temp_memory_handler.load_directive()
            directive.cycle_id = 99
            temp_memory_handler.save_directive(directive)

        # Both changes should be committed
        assert temp_memory_handler.load_system_state().status == "running"
        assert temp_memory_handler.load_directive().cycle_id == 99


# ============================================================================
# Batch Operation Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerBatch:
    """Test batch memory operations."""

    def test_load_all_memory(self, temp_memory_handler):
        """Test loading all memory files at once."""
        memory = temp_memory_handler.load_all_memory()

        assert 'directive.json' in memory
        assert 'history_summary.json' in memory
        assert 'constraints.json' in memory
        assert 'system_state.json' in memory

        assert isinstance(memory['directive.json'], Directive)
        assert isinstance(memory['history_summary.json'], HistorySummary)

    def test_get_memory_stats(self, temp_memory_handler):
        """Test getting memory file statistics."""
        stats = temp_memory_handler.get_memory_stats()

        assert 'directive.json' in stats
        assert stats['directive.json']['exists']
        assert 'size_bytes' in stats['directive.json']
        assert 'modified' in stats['directive.json']


# ============================================================================
# Backup/Restore Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerBackup:
    """Test backup and restore operations."""

    def test_backup_memory(self, temp_memory_handler, tmp_path):
        """Test creating a memory backup."""
        backup_dir = temp_memory_handler.backup_memory()

        assert backup_dir.exists()
        assert (backup_dir / "directive.json").exists()
        assert (backup_dir / "history_summary.json").exists()

    def test_restore_memory(self, temp_memory_handler, tmp_path):
        """Test restoring from backup."""
        # Modify current state
        directive = temp_memory_handler.load_directive()
        directive.cycle_id = 42
        temp_memory_handler.save_directive(directive)

        # Create backup
        backup_dir = temp_memory_handler.backup_memory()

        # Modify again
        directive.cycle_id = 100
        temp_memory_handler.save_directive(directive)
        assert temp_memory_handler.load_directive().cycle_id == 100

        # Restore from backup
        temp_memory_handler.restore_memory(backup_dir)

        # Should be back to backup state
        loaded = temp_memory_handler.load_directive()
        assert loaded.cycle_id == 42


# ============================================================================
# Thread Safety Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_reads(self, temp_memory_handler):
        """Test concurrent reads don't conflict."""
        import threading

        results = []

        def read_directive():
            directive = temp_memory_handler.load_directive()
            results.append(directive.cycle_id)

        # Spawn multiple threads
        threads = [threading.Thread(target=read_directive) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should have read successfully
        assert len(results) == 10
        assert all(result == 0 for result in results)

    def test_concurrent_writes_are_serialized(self, temp_memory_handler):
        """Test concurrent writes are properly serialized."""
        import threading
        import time

        def increment_cycle():
            directive = temp_memory_handler.load_directive()
            directive.cycle_id += 1
            time.sleep(0.001)  # Small delay to increase chance of conflicts
            temp_memory_handler.save_directive(directive)

        # Spawn multiple threads
        threads = [threading.Thread(target=increment_cycle) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Final value should be 10 (all increments applied)
        final_directive = temp_memory_handler.load_directive()
        assert final_directive.cycle_id == 10


# ============================================================================
# Global Singleton Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerSingleton:
    """Test global singleton pattern."""

    def test_get_memory_handler_returns_singleton(self):
        """Test that get_memory_handler returns same instance."""
        reset_memory_handler()

        handler1 = get_memory_handler()
        handler2 = get_memory_handler()

        assert handler1 is handler2

    def test_reset_memory_handler(self):
        """Test that reset creates new instance."""
        reset_memory_handler()

        handler1 = get_memory_handler()
        reset_memory_handler()
        handler2 = get_memory_handler()

        assert handler1 is not handler2


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestMemoryHandlerErrors:
    """Test error handling."""

    def test_load_nonexistent_file_raises_error(self, temp_memory_handler):
        """Test loading nonexistent file raises FileNotFoundError."""
        # Remove file
        (temp_memory_handler.memory_dir / "directive.json").unlink()

        with pytest.raises(FileNotFoundError):
            temp_memory_handler.load_directive()

    def test_atomic_write_failure_raises_error(self, temp_memory_handler):
        """Test that write failures raise AtomicWriteError."""
        # Make directory read-only to force write failure
        import stat
        temp_memory_handler.memory_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        directive = temp_memory_handler.load_directive()

        try:
            with pytest.raises(AtomicWriteError):
                temp_memory_handler.save_directive(directive)
        finally:
            # Restore permissions
            temp_memory_handler.memory_dir.chmod(stat.S_IRWXU)
