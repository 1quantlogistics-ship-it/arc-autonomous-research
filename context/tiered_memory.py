"""
Tiered Memory System for ARC Autonomous Research.

Phase G Task 2.1: Three-tier memory hierarchy with LRU eviction.

Tiers:
- Tier 1 (Hot): <4K tokens, in-memory, recent active data
- Tier 2 (Warm): <16K tokens, in-memory, less recent data
- Tier 3 (Cold): Unlimited, file-backed at memory/archive/

Data flows: put() -> Tier 1, evict to Tier 2 -> Tier 3
Access promotes entries back up the hierarchy.

Author: ARC Team (Dev 2)
Created: 2025-11-26
Version: 1.0 (Phase G)
"""

import json
import logging
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Optional compression from Dev 1
try:
    from context.compression import CompressionEngine, get_compression_engine
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    logger.debug("CompressionEngine not available, cold storage will use uncompressed JSON")


@dataclass
class MemoryEntry:
    """
    A single entry in the tiered memory system.

    Attributes:
        key: Unique identifier for this entry
        content: The actual data (any JSON-serializable type)
        tokens: Estimated token count for this entry
        created_at: When the entry was first created
        accessed_at: Last access timestamp (for LRU)
        tier: Current tier (1=hot, 2=warm, 3=cold)
        access_count: Number of times this entry has been accessed
        metadata: Optional additional metadata
    """
    key: str
    content: Any
    tokens: int
    created_at: datetime
    accessed_at: datetime
    tier: int = 1
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "content": self.content,
            "tokens": self.tokens,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "tier": self.tier,
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            content=data["content"],
            tokens=data["tokens"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            tier=data.get("tier", 1),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {})
        )


class TieredMemorySystem:
    """
    Three-tier memory system with LRU eviction.

    Tier 1 (Hot): Fast access, limited to tier1_max_tokens
    Tier 2 (Warm): Medium access, limited to tier2_max_tokens
    Tier 3 (Cold): File-backed, unlimited size

    LRU eviction: When a tier is full, oldest accessed entries
    are evicted to the next tier down.

    Access promotion: When an entry is accessed, it can be
    promoted back to a higher tier if promote=True.
    """

    def __init__(
        self,
        tier1_max_tokens: int = 4000,
        tier2_max_tokens: int = 16000,
        archive_path: Optional[str] = None,
        compression_enabled: bool = True
    ):
        """
        Initialize tiered memory system.

        Args:
            tier1_max_tokens: Max tokens for hot tier (default 4K)
            tier2_max_tokens: Max tokens for warm tier (default 16K)
            archive_path: Path for cold storage files (default: memory/archive/)
            compression_enabled: Use compression for cold storage if available
        """
        self.tier1_max_tokens = tier1_max_tokens
        self.tier2_max_tokens = tier2_max_tokens

        # Use OrderedDict to maintain insertion order (for LRU)
        self.tier1: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.tier2: OrderedDict[str, MemoryEntry] = OrderedDict()

        # Cold storage path
        if archive_path:
            self.archive_path = Path(archive_path)
        else:
            self.archive_path = Path("memory/archive")
        self.archive_path.mkdir(parents=True, exist_ok=True)

        # Token counters
        self.tier1_tokens = 0
        self.tier2_tokens = 0

        # Compression
        self.compression_enabled = compression_enabled and COMPRESSION_AVAILABLE
        self._compression_engine = None
        if self.compression_enabled:
            try:
                self._compression_engine = get_compression_engine()
            except Exception as e:
                logger.warning(f"Failed to get compression engine: {e}")
                self.compression_enabled = False

        # Stats
        self._stats = {
            "puts": 0,
            "gets": 0,
            "hits": {"tier1": 0, "tier2": 0, "tier3": 0},
            "misses": 0,
            "evictions": {"tier1_to_tier2": 0, "tier2_to_tier3": 0},
            "promotions": {"tier3_to_tier1": 0, "tier2_to_tier1": 0}
        }

        logger.info(
            f"TieredMemorySystem initialized: "
            f"Tier1={tier1_max_tokens}, Tier2={tier2_max_tokens}, "
            f"Archive={self.archive_path}, Compression={self.compression_enabled}"
        )

    def put(
        self,
        key: str,
        content: Any,
        tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Add or update an entry in memory.

        New entries always go to Tier 1. If Tier 1 is full,
        oldest entries are evicted to Tier 2 (and so on).

        Args:
            key: Unique key for this entry
            content: The data to store
            tokens: Token count for this entry
            metadata: Optional metadata

        Returns:
            The created/updated MemoryEntry
        """
        self._stats["puts"] += 1
        now = datetime.now()

        # Check if key exists in any tier
        existing = self._find_entry(key)
        if existing:
            entry, tier = existing
            old_tokens = entry.tokens

            # Update existing entry
            entry.content = content
            entry.tokens = tokens
            entry.accessed_at = now
            entry.access_count += 1
            if metadata:
                entry.metadata.update(metadata)

            # Update token count for the tier
            token_diff = tokens - old_tokens
            if tier == 1:
                self.tier1_tokens += token_diff
                self.tier1.move_to_end(key)
            elif tier == 2:
                self.tier2_tokens += token_diff
                self.tier2.move_to_end(key)
            # For tier 3, we need to re-save
            elif tier == 3:
                self._save_cold_entry(entry)

            return entry

        # Create new entry
        entry = MemoryEntry(
            key=key,
            content=content,
            tokens=tokens,
            created_at=now,
            accessed_at=now,
            tier=1,
            access_count=1,
            metadata=metadata or {}
        )

        # Ensure space in Tier 1
        self._ensure_tier1_space(tokens)

        # Add to Tier 1
        self.tier1[key] = entry
        self.tier1_tokens += tokens

        logger.debug(f"Put entry '{key}' ({tokens} tokens) to Tier 1")
        return entry

    def get(
        self,
        key: str,
        promote: bool = True
    ) -> Optional[Any]:
        """
        Retrieve content by key.

        Args:
            key: Key to look up
            promote: If True, promote entry to higher tier on access

        Returns:
            Content if found, None otherwise
        """
        self._stats["gets"] += 1

        entry_result = self._find_entry(key)
        if not entry_result:
            self._stats["misses"] += 1
            return None

        entry, tier = entry_result
        entry.accessed_at = datetime.now()
        entry.access_count += 1

        # Track hit
        tier_name = f"tier{tier}"
        self._stats["hits"][tier_name] = self._stats["hits"].get(tier_name, 0) + 1

        # Promote if requested and not already in Tier 1
        if promote and tier > 1:
            self._promote_entry(entry, tier)
        elif tier == 1:
            # Move to end (most recently used)
            self.tier1.move_to_end(key)
        elif tier == 2:
            self.tier2.move_to_end(key)

        return entry.content

    def get_entry(self, key: str) -> Optional[MemoryEntry]:
        """Get the full MemoryEntry object (not just content)."""
        result = self._find_entry(key)
        if result:
            return result[0]
        return None

    def delete(self, key: str) -> bool:
        """
        Delete an entry from memory.

        Args:
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        # Check Tier 1
        if key in self.tier1:
            entry = self.tier1.pop(key)
            self.tier1_tokens -= entry.tokens
            return True

        # Check Tier 2
        if key in self.tier2:
            entry = self.tier2.pop(key)
            self.tier2_tokens -= entry.tokens
            return True

        # Check Tier 3 (cold storage)
        cold_path = self._get_cold_path(key)
        if cold_path.exists():
            cold_path.unlink()
            return True

        return False

    def get_context_for_agent(
        self,
        agent_name: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Get relevant context for a specific agent within token budget.

        Prioritizes hot tier entries, then warm, then selectively
        from cold storage based on recency and relevance.

        Args:
            agent_name: Name of the requesting agent
            max_tokens: Maximum tokens to return

        Returns:
            Dict with entries and metadata
        """
        result_entries = []
        used_tokens = 0

        # Helper to add entries
        def add_entries_from_tier(tier_dict: OrderedDict, tier_num: int):
            nonlocal used_tokens
            # Iterate in reverse (most recently used first)
            for key in reversed(list(tier_dict.keys())):
                if used_tokens >= max_tokens:
                    break
                entry = tier_dict[key]
                if used_tokens + entry.tokens <= max_tokens:
                    result_entries.append({
                        "key": entry.key,
                        "content": entry.content,
                        "tokens": entry.tokens,
                        "tier": tier_num,
                        "accessed_at": entry.accessed_at.isoformat()
                    })
                    used_tokens += entry.tokens

        # Add from Tier 1 first (hot)
        add_entries_from_tier(self.tier1, 1)

        # Then Tier 2 (warm)
        if used_tokens < max_tokens:
            add_entries_from_tier(self.tier2, 2)

        # Then selectively from Tier 3 (cold) if still under budget
        if used_tokens < max_tokens:
            cold_entries = self._get_recent_cold_entries(max_tokens - used_tokens)
            for entry in cold_entries:
                if used_tokens + entry.tokens <= max_tokens:
                    result_entries.append({
                        "key": entry.key,
                        "content": entry.content,
                        "tokens": entry.tokens,
                        "tier": 3,
                        "accessed_at": entry.accessed_at.isoformat()
                    })
                    used_tokens += entry.tokens

        return {
            "agent_name": agent_name,
            "max_tokens": max_tokens,
            "used_tokens": used_tokens,
            "num_entries": len(result_entries),
            "entries": result_entries,
            "tier_breakdown": {
                "tier1": sum(1 for e in result_entries if e["tier"] == 1),
                "tier2": sum(1 for e in result_entries if e["tier"] == 2),
                "tier3": sum(1 for e in result_entries if e["tier"] == 3)
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict with usage stats, hit rates, etc.
        """
        total_gets = self._stats["gets"]
        total_hits = sum(self._stats["hits"].values())

        # Count cold storage entries
        cold_count = len(list(self.archive_path.glob("*.json")))

        return {
            "tier1": {
                "entries": len(self.tier1),
                "tokens": self.tier1_tokens,
                "max_tokens": self.tier1_max_tokens,
                "utilization": self.tier1_tokens / self.tier1_max_tokens if self.tier1_max_tokens > 0 else 0
            },
            "tier2": {
                "entries": len(self.tier2),
                "tokens": self.tier2_tokens,
                "max_tokens": self.tier2_max_tokens,
                "utilization": self.tier2_tokens / self.tier2_max_tokens if self.tier2_max_tokens > 0 else 0
            },
            "tier3": {
                "entries": cold_count,
                "path": str(self.archive_path)
            },
            "operations": {
                "puts": self._stats["puts"],
                "gets": self._stats["gets"],
                "hit_rate": total_hits / total_gets if total_gets > 0 else 0,
                "hits_by_tier": self._stats["hits"].copy(),
                "misses": self._stats["misses"]
            },
            "evictions": self._stats["evictions"].copy(),
            "promotions": self._stats["promotions"].copy(),
            "compression_enabled": self.compression_enabled
        }

    def clear(self, tier: Optional[int] = None) -> Dict[str, int]:
        """
        Clear entries from memory.

        Args:
            tier: Specific tier to clear (1, 2, or 3), or None for all

        Returns:
            Dict with counts of cleared entries per tier
        """
        cleared = {"tier1": 0, "tier2": 0, "tier3": 0}

        if tier is None or tier == 1:
            cleared["tier1"] = len(self.tier1)
            self.tier1.clear()
            self.tier1_tokens = 0

        if tier is None or tier == 2:
            cleared["tier2"] = len(self.tier2)
            self.tier2.clear()
            self.tier2_tokens = 0

        if tier is None or tier == 3:
            for path in self.archive_path.glob("*.json"):
                path.unlink()
                cleared["tier3"] += 1

        return cleared

    def _find_entry(self, key: str) -> Optional[Tuple[MemoryEntry, int]]:
        """Find entry across all tiers. Returns (entry, tier_number)."""
        # Check Tier 1
        if key in self.tier1:
            return (self.tier1[key], 1)

        # Check Tier 2
        if key in self.tier2:
            return (self.tier2[key], 2)

        # Check Tier 3 (cold storage)
        cold_path = self._get_cold_path(key)
        if cold_path.exists():
            try:
                entry = self._load_cold_entry(cold_path)
                if entry:
                    return (entry, 3)
            except Exception as e:
                logger.warning(f"Failed to load cold entry '{key}': {e}")

        return None

    def _ensure_tier1_space(self, needed_tokens: int) -> None:
        """Ensure Tier 1 has space for new entry by evicting to Tier 2."""
        while self.tier1_tokens + needed_tokens > self.tier1_max_tokens and self.tier1:
            # Evict oldest (first in OrderedDict)
            oldest_key = next(iter(self.tier1))
            self._evict_to_tier2(oldest_key)

    def _evict_to_tier2(self, key: str) -> None:
        """Evict entry from Tier 1 to Tier 2."""
        if key not in self.tier1:
            return

        entry = self.tier1.pop(key)
        self.tier1_tokens -= entry.tokens
        entry.tier = 2

        # Ensure space in Tier 2
        self._ensure_tier2_space(entry.tokens)

        # Add to Tier 2
        self.tier2[key] = entry
        self.tier2_tokens += entry.tokens

        self._stats["evictions"]["tier1_to_tier2"] += 1
        logger.debug(f"Evicted '{key}' from Tier 1 to Tier 2")

    def _ensure_tier2_space(self, needed_tokens: int) -> None:
        """Ensure Tier 2 has space by evicting to Tier 3 (cold)."""
        while self.tier2_tokens + needed_tokens > self.tier2_max_tokens and self.tier2:
            oldest_key = next(iter(self.tier2))
            self._evict_to_cold(oldest_key)

    def _evict_to_cold(self, key: str) -> None:
        """Evict entry from Tier 2 to cold storage."""
        if key not in self.tier2:
            return

        entry = self.tier2.pop(key)
        self.tier2_tokens -= entry.tokens
        entry.tier = 3

        # Save to cold storage
        self._save_cold_entry(entry)

        self._stats["evictions"]["tier2_to_tier3"] += 1
        logger.debug(f"Evicted '{key}' from Tier 2 to cold storage")

    def _promote_entry(self, entry: MemoryEntry, from_tier: int) -> None:
        """Promote entry to Tier 1."""
        # Remove from current tier
        if from_tier == 2:
            if entry.key in self.tier2:
                self.tier2.pop(entry.key)
                self.tier2_tokens -= entry.tokens
            self._stats["promotions"]["tier2_to_tier1"] += 1
        elif from_tier == 3:
            # Remove from cold storage
            cold_path = self._get_cold_path(entry.key)
            if cold_path.exists():
                cold_path.unlink()
            self._stats["promotions"]["tier3_to_tier1"] += 1

        # Ensure space in Tier 1
        self._ensure_tier1_space(entry.tokens)

        # Add to Tier 1
        entry.tier = 1
        self.tier1[entry.key] = entry
        self.tier1_tokens += entry.tokens

        logger.debug(f"Promoted '{entry.key}' from Tier {from_tier} to Tier 1")

    def _get_cold_path(self, key: str) -> Path:
        """Get file path for cold storage entry."""
        # Use hash to avoid filesystem issues with special characters
        safe_name = hashlib.md5(key.encode()).hexdigest()
        return self.archive_path / f"{safe_name}.json"

    def _save_cold_entry(self, entry: MemoryEntry) -> None:
        """Save entry to cold storage."""
        cold_path = self._get_cold_path(entry.key)
        data = entry.to_dict()

        if self.compression_enabled and self._compression_engine:
            try:
                # Use compression engine
                compressed = self._compression_engine.compress(data)
                with open(cold_path, 'w') as f:
                    json.dump({"compressed": True, "data": compressed}, f)
            except Exception as e:
                logger.warning(f"Compression failed, saving uncompressed: {e}")
                with open(cold_path, 'w') as f:
                    json.dump({"compressed": False, "data": data}, f)
        else:
            with open(cold_path, 'w') as f:
                json.dump({"compressed": False, "data": data}, f)

    def _load_cold_entry(self, path: Path) -> Optional[MemoryEntry]:
        """Load entry from cold storage."""
        try:
            with open(path, 'r') as f:
                wrapper = json.load(f)

            if wrapper.get("compressed") and self.compression_enabled and self._compression_engine:
                data = self._compression_engine.decompress(wrapper["data"])
            else:
                data = wrapper["data"]

            return MemoryEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load cold entry from {path}: {e}")
            return None

    def _get_recent_cold_entries(self, max_tokens: int) -> List[MemoryEntry]:
        """Get most recently accessed entries from cold storage within token budget."""
        entries = []

        # Load all cold entries and sort by accessed_at
        for path in self.archive_path.glob("*.json"):
            entry = self._load_cold_entry(path)
            if entry:
                entries.append(entry)

        # Sort by most recently accessed
        entries.sort(key=lambda e: e.accessed_at, reverse=True)

        # Filter to token budget
        result = []
        used_tokens = 0
        for entry in entries:
            if used_tokens + entry.tokens <= max_tokens:
                result.append(entry)
                used_tokens += entry.tokens

        return result


# Singleton instance
_tiered_memory_instance: Optional[TieredMemorySystem] = None


def get_tiered_memory(
    tier1_max_tokens: int = 4000,
    tier2_max_tokens: int = 16000,
    archive_path: Optional[str] = None
) -> TieredMemorySystem:
    """
    Get or create singleton TieredMemorySystem instance.

    Args:
        tier1_max_tokens: Max tokens for hot tier
        tier2_max_tokens: Max tokens for warm tier
        archive_path: Path for cold storage

    Returns:
        TieredMemorySystem instance
    """
    global _tiered_memory_instance
    if _tiered_memory_instance is None:
        _tiered_memory_instance = TieredMemorySystem(
            tier1_max_tokens=tier1_max_tokens,
            tier2_max_tokens=tier2_max_tokens,
            archive_path=archive_path
        )
    return _tiered_memory_instance


def reset_tiered_memory() -> None:
    """Reset the singleton instance (for testing)."""
    global _tiered_memory_instance
    _tiered_memory_instance = None
