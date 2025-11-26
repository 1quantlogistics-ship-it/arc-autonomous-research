"""
Tests for Compression Engine.

Phase G - Context compression strategies.
"""

import pytest
from context.compression import (
    CompressionStrategy,
    CompressionResult,
    CompressionEngine,
    get_compression_engine,
    reset_compression_engine,
)
from context.token_budget import reset_token_counter


@pytest.fixture
def engine():
    """Create a CompressionEngine instance."""
    reset_compression_engine()
    reset_token_counter()
    return CompressionEngine()


@pytest.fixture
def sample_content():
    """Sample content with various elements for compression testing."""
    return """# Experiment Results
## Cycle 5

DEBUG: Starting experiment run
TRACE: Loading model weights
[debug] Initializing optimizer

accuracy=0.85
loss=0.23
auc=0.87

Step 1: Data loading complete
Step 2: Training started

DEBUG: Batch 1 complete
DEBUG: Batch 2 complete
DEBUG: Batch 3 complete
TRACE: Memory usage: 4GB

result=success
decision=approved

# Summary
The experiment completed successfully.
Best accuracy achieved: 0.85

DEBUG: Cleanup complete
"""


@pytest.fixture
def long_content():
    """Long content for testing moderate compression."""
    header = "# Experiment Log\n\n"
    # 100 lines of content
    middle_lines = [f"Line {i}: Processing data batch {i}\n" for i in range(100)]
    footer = "\n# Results\naccuracy=0.90\nFinal decision=approved\n"
    return header + "".join(middle_lines) + footer


class TestCompressionStrategy:
    """Tests for CompressionStrategy enum."""

    def test_strategy_values(self):
        """Strategies should have correct values."""
        assert CompressionStrategy.NONE.value == "none"
        assert CompressionStrategy.LIGHT.value == "light"
        assert CompressionStrategy.MODERATE.value == "moderate"
        assert CompressionStrategy.AGGRESSIVE.value == "aggressive"


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_result_creation(self):
        """CompressionResult should store all fields."""
        result = CompressionResult(
            original_content="Hello world",
            compressed_content="Hello",
            original_tokens=10,
            compressed_tokens=5,
            strategy_used=CompressionStrategy.LIGHT,
            compression_ratio=0.5
        )

        assert result.original_content == "Hello world"
        assert result.compressed_content == "Hello"
        assert result.original_tokens == 10
        assert result.compressed_tokens == 5
        assert result.strategy_used == CompressionStrategy.LIGHT
        assert result.compression_ratio == 0.5

    def test_tokens_saved(self):
        """tokens_saved should calculate correctly."""
        result = CompressionResult(
            original_content="",
            compressed_content="",
            original_tokens=100,
            compressed_tokens=60,
            strategy_used=CompressionStrategy.MODERATE,
            compression_ratio=0.6
        )

        assert result.tokens_saved == 40

    def test_reduction_percent(self):
        """reduction_percent should calculate correctly."""
        result = CompressionResult(
            original_content="",
            compressed_content="",
            original_tokens=100,
            compressed_tokens=60,
            strategy_used=CompressionStrategy.MODERATE,
            compression_ratio=0.6
        )

        assert result.reduction_percent == 40.0

    def test_reduction_percent_zero_original(self):
        """reduction_percent should handle zero original tokens."""
        result = CompressionResult(
            original_content="",
            compressed_content="",
            original_tokens=0,
            compressed_tokens=0,
            strategy_used=CompressionStrategy.NONE,
            compression_ratio=1.0
        )

        assert result.reduction_percent == 0.0

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        result = CompressionResult(
            original_content="original",
            compressed_content="compressed",
            original_tokens=100,
            compressed_tokens=50,
            strategy_used=CompressionStrategy.AGGRESSIVE,
            compression_ratio=0.5
        )

        d = result.to_dict()
        assert d["original_tokens"] == 100
        assert d["compressed_tokens"] == 50
        assert d["tokens_saved"] == 50
        assert d["reduction_percent"] == 50.0
        assert d["strategy_used"] == "aggressive"


class TestCompressionEngineNone:
    """Tests for no compression strategy."""

    def test_no_compression(self, engine, sample_content):
        """NONE strategy should not modify content."""
        result = engine.compress(sample_content, CompressionStrategy.NONE)

        assert result.compressed_content == sample_content
        assert result.original_tokens == result.compressed_tokens
        assert result.compression_ratio == 1.0
        assert result.tokens_saved == 0


class TestCompressionEngineLight:
    """Tests for light compression strategy."""

    def test_light_removes_debug(self, engine, sample_content):
        """Light compression should remove DEBUG lines."""
        result = engine.compress(sample_content, CompressionStrategy.LIGHT)

        assert "DEBUG:" not in result.compressed_content
        assert "TRACE:" not in result.compressed_content
        assert "[debug]" not in result.compressed_content

    def test_light_preserves_content(self, engine, sample_content):
        """Light compression should preserve important content."""
        result = engine.compress(sample_content, CompressionStrategy.LIGHT)

        assert "accuracy=0.85" in result.compressed_content
        assert "result=success" in result.compressed_content
        assert "# Experiment Results" in result.compressed_content

    def test_light_reduces_tokens(self, engine, sample_content):
        """Light compression should reduce token count."""
        result = engine.compress(sample_content, CompressionStrategy.LIGHT)

        assert result.compressed_tokens < result.original_tokens
        assert result.compression_ratio < 1.0

    def test_light_removes_empty_lines(self, engine):
        """Light compression should consolidate empty lines."""
        content = "Line 1\n\n\n\n\nLine 2\n\n\nLine 3"
        result = engine.compress(content, CompressionStrategy.LIGHT)

        # Should not have multiple consecutive empty lines
        assert "\n\n\n" not in result.compressed_content


class TestCompressionEngineModerate:
    """Tests for moderate compression strategy."""

    def test_moderate_keeps_head_tail(self, engine, long_content):
        """Moderate should keep head and tail sections."""
        result = engine.compress(long_content, CompressionStrategy.MODERATE)

        # Should keep header
        assert "# Experiment Log" in result.compressed_content
        # Should keep footer
        assert "accuracy=0.90" in result.compressed_content
        assert "decision=approved" in result.compressed_content

    def test_moderate_compresses_middle(self, engine, long_content):
        """Moderate should compress middle section."""
        result = engine.compress(long_content, CompressionStrategy.MODERATE)

        # Should be shorter (middle section removed)
        assert result.compressed_tokens < result.original_tokens
        # Should achieve significant reduction
        assert result.compression_ratio < 0.6

    def test_moderate_short_content_falls_back(self, engine):
        """Moderate should fall back to light for short content."""
        short_content = "Line 1\nLine 2\nLine 3"
        result = engine.compress(short_content, CompressionStrategy.MODERATE)

        # Short content shouldn't have compression marker
        # It should just get light treatment
        assert result.compressed_tokens <= result.original_tokens


class TestCompressionEngineAggressive:
    """Tests for aggressive compression strategy."""

    def test_aggressive_keeps_metrics(self, engine, sample_content):
        """Aggressive should keep metrics and results."""
        result = engine.compress(sample_content, CompressionStrategy.AGGRESSIVE)

        assert "accuracy=0.85" in result.compressed_content
        assert "auc=0.87" in result.compressed_content
        assert "result=success" in result.compressed_content

    def test_aggressive_keeps_decisions(self, engine, sample_content):
        """Aggressive should keep decisions."""
        result = engine.compress(sample_content, CompressionStrategy.AGGRESSIVE)

        assert "decision=approved" in result.compressed_content

    def test_aggressive_removes_verbose(self, engine, sample_content):
        """Aggressive should remove verbose content."""
        result = engine.compress(sample_content, CompressionStrategy.AGGRESSIVE)

        assert "DEBUG:" not in result.compressed_content
        assert "TRACE:" not in result.compressed_content
        # Non-critical lines should be removed
        assert "Processing" not in result.compressed_content or "[COMPRESSED" in result.compressed_content

    def test_aggressive_significantly_reduces(self, engine, sample_content):
        """Aggressive should significantly reduce tokens."""
        result = engine.compress(sample_content, CompressionStrategy.AGGRESSIVE)

        # Should be much smaller
        assert result.compression_ratio < 0.7  # At least 30% reduction

    def test_aggressive_adds_header(self, engine, sample_content):
        """Aggressive should add compression header."""
        result = engine.compress(sample_content, CompressionStrategy.AGGRESSIVE)

        assert "[COMPRESSED" in result.compressed_content


class TestAutoCompress:
    """Tests for automatic compression selection."""

    def test_auto_compress_no_need(self, engine):
        """auto_compress should skip if no compression needed."""
        content = "Short content"
        result = engine.auto_compress(content, 100, 1000)

        # Current < target, no compression needed
        assert result.strategy_used == CompressionStrategy.NONE

    def test_auto_compress_selects_appropriate(self, engine, long_content):
        """auto_compress should select appropriate strategy."""
        original_tokens = engine.counter.count(long_content)

        # Request 50% reduction
        target = int(original_tokens * 0.5)
        result = engine.auto_compress(
            long_content,
            original_tokens,
            target
        )

        # Should have selected a compression strategy
        assert result.strategy_used != CompressionStrategy.NONE
        # Should have reduced tokens
        assert result.compressed_tokens < result.original_tokens


class TestCompressMessages:
    """Tests for message list compression."""

    def test_compress_messages_preserves_system(self, engine):
        """System messages should not be compressed."""
        messages = [
            {"role": "system", "content": "DEBUG: You are helpful. DEBUG: Be nice."},
            {"role": "user", "content": "DEBUG: Hello there!"},
        ]

        compressed, result = engine.compress_messages(
            messages, CompressionStrategy.LIGHT
        )

        # System message preserved
        assert "DEBUG:" in compressed[0]["content"]
        # User message may be compressed
        assert len(compressed) == 2

    def test_compress_messages_preserves_recent(self, engine):
        """Recent messages should not be compressed."""
        messages = [
            {"role": "user", "content": "DEBUG: Old message " + "x" * 200},
            {"role": "assistant", "content": "DEBUG: Old response " + "x" * 200},
            {"role": "user", "content": "DEBUG: Second to last"},
            {"role": "assistant", "content": "DEBUG: Last assistant"},
            {"role": "user", "content": "DEBUG: Current message"},
        ]

        compressed, result = engine.compress_messages(
            messages, CompressionStrategy.LIGHT
        )

        # Last 3 messages should be preserved
        assert "DEBUG:" in compressed[-1]["content"]
        assert "DEBUG:" in compressed[-2]["content"]
        assert "DEBUG:" in compressed[-3]["content"]


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_get_compression_stats_basic(self, engine, sample_content):
        """get_compression_stats should return basic info."""
        stats = engine.get_compression_stats(sample_content)

        assert stats["original_tokens"] > 0
        assert stats["content_length"] == len(sample_content)
        assert stats["line_count"] > 0

    def test_get_compression_stats_all_strategies(self, engine, sample_content):
        """get_compression_stats should compare all strategies."""
        stats = engine.get_compression_stats(sample_content, all_strategies=True)

        assert "strategies" in stats
        assert "light" in stats["strategies"]
        assert "moderate" in stats["strategies"]
        assert "aggressive" in stats["strategies"]

        # Each strategy should have stats
        for strategy_stats in stats["strategies"].values():
            assert "compressed_tokens" in strategy_stats
            assert "tokens_saved" in strategy_stats
            assert "compression_ratio" in strategy_stats


class TestTargetTokens:
    """Tests for target token behavior."""

    def test_compress_with_target(self, engine, long_content):
        """Compression should try to reach target."""
        original_tokens = engine.counter.count(long_content)
        target = original_tokens // 2  # 50% target

        result = engine.compress(
            long_content,
            CompressionStrategy.MODERATE,
            target_tokens=target
        )

        # Should get close to target
        assert result.compressed_tokens < original_tokens

    def test_escalate_if_target_not_reached(self, engine, sample_content):
        """Should escalate strategy if target not reached."""
        original_tokens = engine.counter.count(sample_content)
        very_low_target = 10  # Very aggressive target

        result = engine.compress(
            sample_content,
            CompressionStrategy.LIGHT,
            target_tokens=very_low_target
        )

        # Should have tried harder compression
        assert result.compressed_tokens < original_tokens


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_compression_engine_singleton(self):
        """get_compression_engine should return same instance."""
        reset_compression_engine()

        engine1 = get_compression_engine()
        engine2 = get_compression_engine()

        assert engine1 is engine2

    def test_reset_compression_engine(self):
        """reset_compression_engine should clear singleton."""
        engine1 = get_compression_engine()
        reset_compression_engine()
        engine2 = get_compression_engine()

        assert engine1 is not engine2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content(self, engine):
        """Empty content should handle gracefully."""
        result = engine.compress("", CompressionStrategy.AGGRESSIVE)
        # Aggressive mode may add header, but original tokens should be 0
        assert result.original_tokens == 0

    def test_whitespace_only(self, engine):
        """Whitespace-only content should handle gracefully."""
        result = engine.compress("   \n\n   \n", CompressionStrategy.LIGHT)
        # Light compression removes empty lines
        assert len(result.compressed_content.strip()) == 0

    def test_no_critical_content(self, engine):
        """Content with no critical patterns should still work."""
        content = "Just some random text\nNo metrics here\nJust words"
        result = engine.compress(content, CompressionStrategy.AGGRESSIVE)

        # Should still produce output (fallback behavior)
        assert len(result.compressed_content) > 0

    def test_unicode_content(self, engine):
        """Unicode content should work correctly."""
        content = "accuracy=0.85\n日本語テスト\n🎉 Success! 🎉"
        result = engine.compress(content, CompressionStrategy.LIGHT)

        # Unicode should be preserved
        assert "日本語テスト" in result.compressed_content or "[COMPRESSED" in result.compressed_content
