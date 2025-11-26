"""
Context Compression Engine

Phase G - Context compression strategies for bounded context growth.

Three strategies:
- Aggressive (>95% usage): Keep only results/decisions/metrics
- Moderate (>80% usage): Keep head 20%, compress middle, keep tail 20%
- Light (>60% usage): Remove only DEBUG/TRACE lines
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock

from context.token_budget import TokenCounter, get_token_counter

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies."""
    NONE = "none"
    LIGHT = "light"          # Remove DEBUG/TRACE, verbose logs
    MODERATE = "moderate"    # Keep head/tail, compress middle
    AGGRESSIVE = "aggressive"  # Keep only critical info


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    strategy_used: CompressionStrategy
    compression_ratio: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved by compression."""
        return self.original_tokens - self.compressed_tokens

    @property
    def reduction_percent(self) -> float:
        """Percentage reduction in tokens."""
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 4),
            "reduction_percent": round(self.reduction_percent, 2),
            "strategy_used": self.strategy_used.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class CompressionEngine:
    """
    Engine for compressing context content based on strategy.

    Supports multiple compression strategies with configurable
    thresholds and behaviors.
    """

    # Patterns for light compression (verbose/debug content)
    VERBOSE_PATTERNS = [
        r"^\s*DEBUG:.*$",
        r"^\s*TRACE:.*$",
        r"^\s*\[debug\].*$",
        r"^\s*\[trace\].*$",
        r"^\s*#\s*TODO:.*$",
        r"^\s*#\s*FIXME:.*$",
        r"^\s*>>>.*$",  # Python REPL prompts
        r"^\s*\.\.\..*$",  # Continuation prompts
        r"^\s*$",  # Empty lines
    ]

    # Patterns for content to always keep (even in aggressive mode)
    CRITICAL_PATTERNS = [
        r"(?i)accuracy[:=]\s*[\d.]+",
        r"(?i)auc[:=]\s*[\d.]+",
        r"(?i)loss[:=]\s*[\d.]+",
        r"(?i)error[:=]",
        r"(?i)result[:=]",
        r"(?i)decision[:=]",
        r"(?i)approved",
        r"(?i)rejected",
        r"(?i)experiment_id",
        r"(?i)cycle_id",
        r"(?i)best_.*[:=]",
    ]

    # Section markers for moderate compression
    SECTION_MARKERS = [
        r"^#+\s+",  # Markdown headers
        r"^##.*$",
        r"^---+$",  # Dividers
        r"^===+$",
        r"^Phase\s+\d",
        r"^Cycle\s+\d",
        r"^Step\s+\d",
    ]

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize the compression engine.

        Args:
            token_counter: TokenCounter instance (creates one if not provided)
        """
        self.counter = token_counter or get_token_counter()
        self._compiled_verbose = [re.compile(p, re.MULTILINE) for p in self.VERBOSE_PATTERNS]
        self._compiled_critical = [re.compile(p) for p in self.CRITICAL_PATTERNS]
        self._compiled_sections = [re.compile(p, re.MULTILINE) for p in self.SECTION_MARKERS]

    def compress(
        self,
        content: str,
        strategy: CompressionStrategy,
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """
        Compress content using the specified strategy.

        Args:
            content: Content to compress
            strategy: Compression strategy to use
            target_tokens: Optional target token count

        Returns:
            CompressionResult with compressed content
        """
        original_tokens = self.counter.count(content)

        if strategy == CompressionStrategy.NONE:
            return CompressionResult(
                original_content=content,
                compressed_content=content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=strategy,
                compression_ratio=1.0
            )

        # Apply compression based on strategy
        if strategy == CompressionStrategy.LIGHT:
            compressed = self._compress_light(content)
        elif strategy == CompressionStrategy.MODERATE:
            compressed = self._compress_moderate(content, target_tokens)
        elif strategy == CompressionStrategy.AGGRESSIVE:
            compressed = self._compress_aggressive(content, target_tokens)
        else:
            compressed = content

        compressed_tokens = self.counter.count(compressed)

        # If we have a target and didn't reach it, try harder
        if target_tokens and compressed_tokens > target_tokens:
            # Escalate compression strategy
            if strategy == CompressionStrategy.LIGHT:
                compressed = self._compress_moderate(content, target_tokens)
                compressed_tokens = self.counter.count(compressed)

            if compressed_tokens > target_tokens and strategy != CompressionStrategy.AGGRESSIVE:
                compressed = self._compress_aggressive(content, target_tokens)
                compressed_tokens = self.counter.count(compressed)

        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressionResult(
            original_content=content,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=strategy,
            compression_ratio=compression_ratio
        )

    def auto_compress(
        self,
        content: str,
        current_tokens: int,
        target_tokens: int
    ) -> CompressionResult:
        """
        Automatically determine and apply compression to reach target.

        Args:
            content: Content to compress
            current_tokens: Current total token usage
            target_tokens: Target token count

        Returns:
            CompressionResult
        """
        tokens_to_save = current_tokens - target_tokens

        if tokens_to_save <= 0:
            return self.compress(content, CompressionStrategy.NONE)

        content_tokens = self.counter.count(content)

        # Determine strategy based on how much we need to save
        save_ratio = tokens_to_save / content_tokens if content_tokens > 0 else 0

        if save_ratio > 0.5:
            strategy = CompressionStrategy.AGGRESSIVE
        elif save_ratio > 0.2:
            strategy = CompressionStrategy.MODERATE
        else:
            strategy = CompressionStrategy.LIGHT

        # Target for this content (proportional reduction)
        content_target = max(100, content_tokens - tokens_to_save)

        return self.compress(content, strategy, content_target)

    def _compress_light(self, content: str) -> str:
        """
        Light compression: Remove DEBUG/TRACE and verbose content.

        Preserves structure and most content, only removing
        clearly verbose or debug information.
        """
        lines = content.split('\n')
        kept_lines = []

        for line in lines:
            keep = True
            for pattern in self._compiled_verbose:
                if pattern.match(line):
                    keep = False
                    break

            if keep:
                kept_lines.append(line)

        # Remove consecutive empty lines
        result_lines = []
        prev_empty = False
        for line in kept_lines:
            is_empty = line.strip() == ''
            if is_empty and prev_empty:
                continue
            result_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(result_lines)

    def _compress_moderate(self, content: str, target_tokens: Optional[int] = None) -> str:
        """
        Moderate compression: Keep head 20%, compress middle, keep tail 20%.

        Preserves context at beginning and end while summarizing middle.
        """
        lines = content.split('\n')
        total_lines = len(lines)

        if total_lines < 10:
            # Too short to compress moderately
            return self._compress_light(content)

        # Calculate head/tail sizes
        head_size = max(5, total_lines // 5)  # 20%
        tail_size = max(5, total_lines // 5)  # 20%

        head = lines[:head_size]
        middle = lines[head_size:-tail_size]
        tail = lines[-tail_size:]

        # Compress middle section
        compressed_middle = self._extract_key_lines(middle)

        # Add marker
        if compressed_middle:
            compressed_middle = [
                "",
                "... [compressed: {} lines -> {}] ...".format(
                    len(middle), len(compressed_middle)
                ),
                ""
            ] + compressed_middle

        result = head + compressed_middle + tail
        return '\n'.join(result)

    def _compress_aggressive(self, content: str, target_tokens: Optional[int] = None) -> str:
        """
        Aggressive compression: Keep only results/decisions/metrics.

        Extracts only critical information, discarding everything else.
        """
        lines = content.split('\n')
        critical_lines = []

        for line in lines:
            # Check if line contains critical information
            is_critical = False
            for pattern in self._compiled_critical:
                if pattern.search(line):
                    is_critical = True
                    break

            # Also keep section headers
            if not is_critical:
                for pattern in self._compiled_sections:
                    if pattern.search(line):
                        is_critical = True
                        break

            if is_critical:
                critical_lines.append(line)

        if not critical_lines:
            # Fallback: keep first and last few lines
            critical_lines = lines[:3] + ["... [aggressively compressed] ..."] + lines[-3:]

        # Add summary header
        result = [
            "[COMPRESSED - Aggressive Mode]",
            f"Original: {len(lines)} lines",
            f"Kept: {len(critical_lines)} critical lines",
            "---",
        ] + critical_lines

        return '\n'.join(result)

    def _extract_key_lines(self, lines: List[str]) -> List[str]:
        """
        Extract key lines from a section (for moderate compression).

        Keeps lines with metrics, decisions, and section markers.
        """
        key_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check for critical patterns
            for pattern in self._compiled_critical:
                if pattern.search(line):
                    key_lines.append(line)
                    break
            else:
                # Check for section markers
                for pattern in self._compiled_sections:
                    if pattern.search(line):
                        key_lines.append(line)
                        break

        return key_lines

    def compress_messages(
        self,
        messages: List[Dict[str, Any]],
        strategy: CompressionStrategy,
        target_tokens: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], CompressionResult]:
        """
        Compress a list of messages (chat format).

        Args:
            messages: List of message dicts
            strategy: Compression strategy
            target_tokens: Optional target token count

        Returns:
            Tuple of (compressed_messages, CompressionResult)
        """
        if strategy == CompressionStrategy.NONE:
            total_tokens = sum(
                self.counter.count(str(m.get("content", "")))
                for m in messages
            )
            return messages, CompressionResult(
                original_content="",
                compressed_content="",
                original_tokens=total_tokens,
                compressed_tokens=total_tokens,
                strategy_used=strategy,
                compression_ratio=1.0,
                metadata={"message_count": len(messages)}
            )

        compressed_messages = []
        original_tokens = 0
        compressed_tokens = 0

        # Keep system messages and recent messages uncompressed
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                content_tokens = self.counter.count(content)
                original_tokens += content_tokens

                # Don't compress system messages or last few messages
                should_compress = (
                    role != "system" and
                    i < len(messages) - 3 and  # Keep last 3 messages
                    content_tokens > 100  # Only compress substantial content
                )

                if should_compress:
                    result = self.compress(content, strategy, target_tokens)
                    compressed_messages.append({
                        **msg,
                        "content": result.compressed_content
                    })
                    compressed_tokens += result.compressed_tokens
                else:
                    compressed_messages.append(msg)
                    compressed_tokens += content_tokens
            else:
                compressed_messages.append(msg)

        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return compressed_messages, CompressionResult(
            original_content="",
            compressed_content="",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=strategy,
            compression_ratio=compression_ratio,
            metadata={"message_count": len(messages)}
        )

    def get_compression_stats(
        self,
        content: str,
        all_strategies: bool = False
    ) -> Dict[str, Any]:
        """
        Get compression statistics for content.

        Args:
            content: Content to analyze
            all_strategies: If True, show results for all strategies

        Returns:
            Dict with compression statistics
        """
        original_tokens = self.counter.count(content)

        if not all_strategies:
            return {
                "original_tokens": original_tokens,
                "content_length": len(content),
                "line_count": len(content.split('\n'))
            }

        stats = {
            "original_tokens": original_tokens,
            "content_length": len(content),
            "line_count": len(content.split('\n')),
            "strategies": {}
        }

        for strategy in [CompressionStrategy.LIGHT, CompressionStrategy.MODERATE, CompressionStrategy.AGGRESSIVE]:
            result = self.compress(content, strategy)
            stats["strategies"][strategy.value] = {
                "compressed_tokens": result.compressed_tokens,
                "tokens_saved": result.tokens_saved,
                "compression_ratio": round(result.compression_ratio, 4),
                "reduction_percent": round(result.reduction_percent, 2)
            }

        return stats


# Singleton pattern for global access
_compression_engine: Optional[CompressionEngine] = None
_engine_lock = Lock()


def get_compression_engine() -> CompressionEngine:
    """
    Get or create the global CompressionEngine instance.

    Returns:
        The singleton CompressionEngine instance
    """
    global _compression_engine

    with _engine_lock:
        if _compression_engine is None:
            _compression_engine = CompressionEngine()
        return _compression_engine


def reset_compression_engine() -> None:
    """Reset the global CompressionEngine instance (for testing)."""
    global _compression_engine
    with _engine_lock:
        _compression_engine = None
