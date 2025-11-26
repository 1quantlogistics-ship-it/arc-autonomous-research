"""
Tests for Tiered Memory System.

Phase G Task 2.1: Tests for TieredMemorySystem.

Author: ARC Team (Dev 2)
Created: 2025-11-26
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from context.tiered_memory import (
    MemoryEntry,
    TieredMemorySystem,
    get_tiered_memory,
    reset_tiered_memory,
)


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_create_entry(self):
        """Test creating a memory entry."""
        now = datetime.now()
        entry = MemoryEntry(
            key="test_key",
            content={"data": "test"},
            tokens=100,
            created_at=now,
            accessed_at=now,
            tier=1
        )

        assert entry.key == "test_key"
        assert entry.content == {"data": "test"}
        assert entry.tokens == 100
        assert entry.tier == 1
        assert entry.access_count == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now()
        entry = MemoryEntry(
            key="test_key",
            content={"data": "test"},
            tokens=100,
            created_at=now,
            accessed_at=now,
            metadata={"extra": "info"}
        )

        data = entry.to_dict()

        assert data["key"] == "test_key"
        assert data["tokens"] == 100
        assert data["metadata"] == {"extra": "info"}
        assert "created_at" in data
        assert "accessed_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        now = datetime.now()
        data = {
            "key": "test_key",
            "content": {"data": "test"},
            "tokens": 100,
            "created_at": now.isoformat(),
            "accessed_at": now.isoformat(),
            "tier": 2,
            "access_count": 5,
            "metadata": {"extra": "info"}
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.key == "test_key"
        assert entry.tokens == 100
        assert entry.tier == 2
        assert entry.access_count == 5


class TestTieredMemorySystem:
    """Test TieredMemorySystem class."""

    @pytest.fixture
    def temp_archive(self):
        """Create temporary archive directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def memory_system(self, temp_archive):
        """Create memory system with small limits for testing."""
        reset_tiered_memory()
        return TieredMemorySystem(
            tier1_max_tokens=1000,
            tier2_max_tokens=2000,
            archive_path=temp_archive,
            compression_enabled=False  # Disable compression for tests
        )

    def test_put_basic(self, memory_system):
        """Test basic put operation."""
        entry = memory_system.put("key1", {"data": "test"}, tokens=100)

        assert entry.key == "key1"
        assert entry.tier == 1
        assert entry.tokens == 100

        # Check stats
        stats = memory_system.get_stats()
        assert stats["tier1"]["entries"] == 1
        assert stats["tier1"]["tokens"] == 100

    def test_get_basic(self, memory_system):
        """Test basic get operation."""
        memory_system.put("key1", {"data": "test_value"}, tokens=100)

        content = memory_system.get("key1")

        assert content == {"data": "test_value"}

    def test_get_nonexistent(self, memory_system):
        """Test getting nonexistent key."""
        content = memory_system.get("nonexistent")

        assert content is None

    def test_put_update(self, memory_system):
        """Test updating existing entry."""
        memory_system.put("key1", {"data": "original"}, tokens=100)
        memory_system.put("key1", {"data": "updated"}, tokens=150)

        content = memory_system.get("key1")

        assert content == {"data": "updated"}

        # Should still be only 1 entry
        stats = memory_system.get_stats()
        assert stats["tier1"]["entries"] == 1
        assert stats["tier1"]["tokens"] == 150

    def test_delete(self, memory_system):
        """Test delete operation."""
        memory_system.put("key1", {"data": "test"}, tokens=100)

        result = memory_system.delete("key1")

        assert result is True
        assert memory_system.get("key1") is None

    def test_delete_nonexistent(self, memory_system):
        """Test deleting nonexistent key."""
        result = memory_system.delete("nonexistent")

        assert result is False

    def test_eviction_tier1_to_tier2(self, memory_system):
        """Test eviction from Tier 1 to Tier 2."""
        # Fill Tier 1 (1000 tokens max)
        memory_system.put("key1", {"data": 1}, tokens=400)
        memory_system.put("key2", {"data": 2}, tokens=400)

        # This should trigger eviction of key1
        memory_system.put("key3", {"data": 3}, tokens=400)

        stats = memory_system.get_stats()

        # key1 should be in Tier 2
        entry = memory_system.get_entry("key1")
        assert entry.tier == 2

        # key2 and key3 should be in Tier 1
        assert stats["tier1"]["entries"] == 2
        assert stats["tier2"]["entries"] == 1

    def test_eviction_tier2_to_cold(self, memory_system):
        """Test eviction from Tier 2 to cold storage."""
        # Tier 1: 1000 tokens max, Tier 2: 2000 tokens max
        # We need to fill tier2 (2000) and trigger eviction
        # With 400 token entries:
        # - Tier 1 can hold 2 entries (800 tokens, leaves 200 unused)
        # - Tier 2 can hold 5 entries (2000 tokens exactly)
        # So we need 8 entries to start pushing to cold storage

        for i in range(8):
            memory_system.put(f"key{i}", {"data": i}, tokens=400)

        # This should push oldest to cold storage
        memory_system.put("key8", {"data": 8}, tokens=400)

        stats = memory_system.get_stats()

        # Should have entries in cold storage
        assert stats["tier3"]["entries"] > 0

        # Oldest keys should still be accessible from cold
        content = memory_system.get("key0", promote=False)
        assert content is not None
        assert content["data"] == 0

    def test_promotion_on_access(self, memory_system):
        """Test that entries are promoted on access."""
        # Fill Tier 1 to push entry to Tier 2
        memory_system.put("key1", {"data": 1}, tokens=400)
        memory_system.put("key2", {"data": 2}, tokens=400)
        memory_system.put("key3", {"data": 3}, tokens=400)

        # key1 should now be in Tier 2
        entry = memory_system.get_entry("key1")
        assert entry.tier == 2

        # Access key1 with promotion
        content = memory_system.get("key1", promote=True)

        # Should be promoted back to Tier 1
        entry = memory_system.get_entry("key1")
        assert entry.tier == 1

    def test_no_promotion_when_disabled(self, memory_system):
        """Test that promotion can be disabled."""
        # Fill Tier 1 to push entry to Tier 2
        memory_system.put("key1", {"data": 1}, tokens=400)
        memory_system.put("key2", {"data": 2}, tokens=400)
        memory_system.put("key3", {"data": 3}, tokens=400)

        # key1 should be in Tier 2
        entry_before = memory_system.get_entry("key1")
        assert entry_before.tier == 2

        # Access without promotion
        content = memory_system.get("key1", promote=False)

        # Should still be in Tier 2
        entry_after = memory_system.get_entry("key1")
        assert entry_after.tier == 2

    def test_get_context_for_agent(self, memory_system):
        """Test getting context for an agent."""
        # Add some entries
        memory_system.put("key1", {"data": 1}, tokens=100)
        memory_system.put("key2", {"data": 2}, tokens=100)
        memory_system.put("key3", {"data": 3}, tokens=100)

        context = memory_system.get_context_for_agent(
            agent_name="architect",
            max_tokens=250
        )

        assert context["agent_name"] == "architect"
        assert context["max_tokens"] == 250
        assert context["used_tokens"] <= 250
        assert context["num_entries"] > 0
        assert len(context["entries"]) > 0

    def test_get_context_respects_budget(self, memory_system):
        """Test that get_context respects token budget."""
        # Add entries totaling 300 tokens
        memory_system.put("key1", {"data": 1}, tokens=100)
        memory_system.put("key2", {"data": 2}, tokens=100)
        memory_system.put("key3", {"data": 3}, tokens=100)

        # Request only 150 tokens
        context = memory_system.get_context_for_agent(
            agent_name="test",
            max_tokens=150
        )

        assert context["used_tokens"] <= 150

    def test_stats(self, memory_system):
        """Test statistics gathering."""
        # Perform some operations
        memory_system.put("key1", {"data": 1}, tokens=100)
        memory_system.get("key1")
        memory_system.get("nonexistent")

        stats = memory_system.get_stats()

        assert stats["operations"]["puts"] == 1
        assert stats["operations"]["gets"] == 2
        assert stats["operations"]["misses"] == 1
        assert stats["tier1"]["entries"] == 1

    def test_clear_all(self, memory_system):
        """Test clearing all tiers."""
        memory_system.put("key1", {"data": 1}, tokens=100)
        memory_system.put("key2", {"data": 2}, tokens=100)

        cleared = memory_system.clear()

        assert cleared["tier1"] == 2
        stats = memory_system.get_stats()
        assert stats["tier1"]["entries"] == 0
        assert stats["tier1"]["tokens"] == 0

    def test_clear_specific_tier(self, memory_system):
        """Test clearing specific tier."""
        # Add entries to both tiers
        for i in range(4):
            memory_system.put(f"key{i}", {"data": i}, tokens=300)

        # Clear only Tier 1
        cleared = memory_system.clear(tier=1)

        assert cleared["tier1"] > 0
        stats = memory_system.get_stats()
        assert stats["tier1"]["entries"] == 0
        # Tier 2 should still have entries
        assert stats["tier2"]["entries"] > 0


class TestTieredMemorySingleton:
    """Test singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_tiered_memory()
        yield
        reset_tiered_memory()

    def test_get_same_instance(self):
        """Test that get_tiered_memory returns same instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            instance1 = get_tiered_memory(archive_path=temp_dir)
            instance2 = get_tiered_memory()

            assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test that reset creates new instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            instance1 = get_tiered_memory(archive_path=temp_dir)
            reset_tiered_memory()

            with tempfile.TemporaryDirectory() as temp_dir2:
                instance2 = get_tiered_memory(archive_path=temp_dir2)

                assert instance1 is not instance2


class TestColdStoragePersistence:
    """Test cold storage file persistence."""

    @pytest.fixture
    def temp_archive(self):
        """Create temporary archive directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cold_storage_creates_files(self, temp_archive):
        """Test that cold storage creates files."""
        memory = TieredMemorySystem(
            tier1_max_tokens=100,
            tier2_max_tokens=100,
            archive_path=temp_archive,
            compression_enabled=False
        )

        # Force eviction to cold storage
        memory.put("key1", {"data": 1}, tokens=50)
        memory.put("key2", {"data": 2}, tokens=50)
        memory.put("key3", {"data": 3}, tokens=50)
        memory.put("key4", {"data": 4}, tokens=50)
        memory.put("key5", {"data": 5}, tokens=50)

        # Check files were created
        archive_path = Path(temp_archive)
        cold_files = list(archive_path.glob("*.json"))

        assert len(cold_files) > 0

    def test_cold_storage_data_recovery(self, temp_archive):
        """Test that data can be recovered from cold storage."""
        memory = TieredMemorySystem(
            tier1_max_tokens=100,
            tier2_max_tokens=100,
            archive_path=temp_archive,
            compression_enabled=False
        )

        # Force eviction to cold storage
        memory.put("important_key", {"important": "data"}, tokens=50)
        memory.put("key2", {"data": 2}, tokens=50)
        memory.put("key3", {"data": 3}, tokens=50)
        memory.put("key4", {"data": 4}, tokens=50)
        memory.put("key5", {"data": 5}, tokens=50)

        # important_key should be in cold storage
        content = memory.get("important_key", promote=False)

        assert content is not None
        assert content["important"] == "data"


class TestLRUBehavior:
    """Test LRU eviction behavior."""

    @pytest.fixture
    def temp_archive(self):
        """Create temporary archive directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_access_updates_lru_order(self, temp_archive):
        """Test that access updates LRU order."""
        memory = TieredMemorySystem(
            tier1_max_tokens=300,
            tier2_max_tokens=300,
            archive_path=temp_archive
        )

        # Add entries
        memory.put("key1", {"data": 1}, tokens=100)
        memory.put("key2", {"data": 2}, tokens=100)
        memory.put("key3", {"data": 3}, tokens=100)

        # Access key1 to make it recently used
        memory.get("key1")

        # Add new entry to trigger eviction
        memory.put("key4", {"data": 4}, tokens=100)

        # key1 should NOT be evicted (recently accessed)
        # key2 should be evicted (oldest not accessed)
        entry_key1 = memory.get_entry("key1")
        entry_key2 = memory.get_entry("key2")

        assert entry_key1.tier == 1  # Still in Tier 1
        assert entry_key2.tier == 2  # Evicted to Tier 2

    def test_access_count_tracking(self, temp_archive):
        """Test that access count is tracked."""
        memory = TieredMemorySystem(
            tier1_max_tokens=1000,
            tier2_max_tokens=1000,
            archive_path=temp_archive
        )

        memory.put("key1", {"data": 1}, tokens=100)

        # Access multiple times
        memory.get("key1")
        memory.get("key1")
        memory.get("key1")

        entry = memory.get_entry("key1")

        # 1 from put + 3 from get
        assert entry.access_count == 4
