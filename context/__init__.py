"""
Context Management Package

Phase G - Checkpoint & Token Management for ARC autonomous research.

This package provides:
- Checkpoint management for crash recovery (Dev 1)
- Token budget enforcement across models (Dev 1)
- Context compression strategies (Dev 1)
- Tiered memory system (Dev 2)
- Cross-cycle summarization (Dev 2)
- Integration with the monitoring dashboard
"""

# Dev 1 modules (conditionally imported - may not exist yet)
try:
    from context.checkpoint_manager import (
        CycleCheckpoint,
        CheckpointManager,
        get_checkpoint_manager,
        reset_checkpoint_manager,
    )
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False
    CycleCheckpoint = None
    CheckpointManager = None
    get_checkpoint_manager = None
    reset_checkpoint_manager = None

try:
    from context.token_budget import (
        ModelFamily,
        TokenCounter,
        TokenBudgetEnforcer,
        get_token_counter,
    )
    TOKEN_BUDGET_AVAILABLE = True
except ImportError:
    TOKEN_BUDGET_AVAILABLE = False
    ModelFamily = None
    TokenCounter = None
    TokenBudgetEnforcer = None
    get_token_counter = None

try:
    from context.compression import (
        CompressionStrategy,
        CompressionResult,
        CompressionEngine,
        get_compression_engine,
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    CompressionStrategy = None
    CompressionResult = None
    CompressionEngine = None
    get_compression_engine = None

# Dev 2 modules (Tiered Memory and Summarizer)
from context.tiered_memory import (
    MemoryEntry,
    TieredMemorySystem,
    get_tiered_memory,
    reset_tiered_memory,
)

from context.summarizer import (
    CycleSummary,
    ExecutiveSummary,
    CrossCycleSummarizer,
    get_summarizer,
    reset_summarizer,
)

__all__ = [
    # Dev 1: Checkpoint management
    "CycleCheckpoint",
    "CheckpointManager",
    "get_checkpoint_manager",
    "reset_checkpoint_manager",
    "CHECKPOINT_MANAGER_AVAILABLE",
    # Dev 1: Token budget
    "ModelFamily",
    "TokenCounter",
    "TokenBudgetEnforcer",
    "get_token_counter",
    "TOKEN_BUDGET_AVAILABLE",
    # Dev 1: Compression
    "CompressionStrategy",
    "CompressionResult",
    "CompressionEngine",
    "get_compression_engine",
    "COMPRESSION_AVAILABLE",
    # Dev 2: Tiered Memory
    "MemoryEntry",
    "TieredMemorySystem",
    "get_tiered_memory",
    "reset_tiered_memory",
    # Dev 2: Summarizer
    "CycleSummary",
    "ExecutiveSummary",
    "CrossCycleSummarizer",
    "get_summarizer",
    "reset_summarizer",
]
