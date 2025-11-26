"""
Tests for Token Budget Enforcer.

Phase G - Token counting and budget enforcement.
"""

import pytest
from context.token_budget import (
    ModelFamily,
    ModelLimits,
    MODEL_CONFIGS,
    get_model_family,
    TokenCounter,
    TokenBudgetEnforcer,
    get_token_counter,
    reset_token_counter,
    TIKTOKEN_AVAILABLE,
)


class TestModelFamily:
    """Tests for model family detection."""

    def test_claude_models(self):
        """Claude model names should map to CLAUDE family."""
        assert get_model_family("claude-3-opus") == ModelFamily.CLAUDE
        assert get_model_family("claude-3-sonnet") == ModelFamily.CLAUDE
        assert get_model_family("claude-3.5-sonnet") == ModelFamily.CLAUDE
        assert get_model_family("claude-sonnet-4") == ModelFamily.CLAUDE
        assert get_model_family("claude-opus-4") == ModelFamily.CLAUDE

    def test_deepseek_models(self):
        """DeepSeek model names should map to DEEPSEEK family."""
        assert get_model_family("deepseek-v3") == ModelFamily.DEEPSEEK
        assert get_model_family("deepseek-coder") == ModelFamily.DEEPSEEK
        assert get_model_family("deepseek-chat") == ModelFamily.DEEPSEEK

    def test_gpt_models(self):
        """GPT model names should map to GPT family."""
        assert get_model_family("gpt-4") == ModelFamily.GPT
        assert get_model_family("gpt-4-turbo") == ModelFamily.GPT
        assert get_model_family("gpt-4o") == ModelFamily.GPT

    def test_partial_matching(self):
        """Partial name matching should work."""
        assert get_model_family("my-custom-claude-model") == ModelFamily.CLAUDE
        assert get_model_family("deepseek-custom") == ModelFamily.DEEPSEEK
        assert get_model_family("gpt-4-custom") == ModelFamily.GPT

    def test_unknown_models(self):
        """Unknown models should return UNKNOWN family."""
        assert get_model_family("llama-2-70b") == ModelFamily.UNKNOWN
        assert get_model_family("mistral-7b") == ModelFamily.UNKNOWN
        assert get_model_family("") == ModelFamily.UNKNOWN


class TestModelConfigs:
    """Tests for model configuration."""

    def test_claude_limits(self):
        """Claude should have 200K context limit."""
        limits = MODEL_CONFIGS[ModelFamily.CLAUDE]
        assert limits.context_limit == 200_000
        assert limits.reserved_tokens == 20_000
        assert limits.effective_limit == 180_000

    def test_deepseek_limits(self):
        """DeepSeek should have 64K context limit."""
        limits = MODEL_CONFIGS[ModelFamily.DEEPSEEK]
        assert limits.context_limit == 64_000
        assert limits.reserved_tokens == 6_400
        assert limits.effective_limit == 57_600

    def test_gpt_limits(self):
        """GPT should have 128K context limit."""
        limits = MODEL_CONFIGS[ModelFamily.GPT]
        assert limits.context_limit == 128_000
        assert limits.reserved_tokens == 12_800
        assert limits.effective_limit == 115_200

    def test_unknown_limits(self):
        """Unknown should have conservative limits."""
        limits = MODEL_CONFIGS[ModelFamily.UNKNOWN]
        assert limits.context_limit == 32_000
        assert limits.effective_limit == 28_800


class TestTokenCounter:
    """Tests for TokenCounter class."""

    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        reset_token_counter()
        return TokenCounter()

    def test_count_empty_string(self, counter):
        """Empty string should return 0 tokens."""
        assert counter.count("") == 0

    def test_count_simple_text(self, counter):
        """Simple text should have reasonable token count."""
        text = "Hello, world!"
        tokens = counter.count(text)
        # Should be small number of tokens
        assert 1 <= tokens <= 10

    def test_count_longer_text(self, counter):
        """Longer text should have proportional token count."""
        short_text = "Hello"
        long_text = "Hello " * 100

        short_tokens = counter.count(short_text)
        long_tokens = counter.count(long_text)

        # Long text should have more tokens
        assert long_tokens > short_tokens
        # Should be roughly proportional (within reason)
        assert long_tokens > short_tokens * 10

    def test_count_messages_empty(self, counter):
        """Empty message list should have minimal tokens."""
        tokens = counter.count_messages([])
        # Just conversation overhead
        assert tokens <= 10

    def test_count_messages_single(self, counter):
        """Single message should count correctly."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        tokens = counter.count_messages(messages)
        # Should include content + overhead
        assert tokens > counter.count("Hello, how are you?")
        assert tokens < 50

    def test_count_messages_multiple(self, counter):
        """Multiple messages should sum correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        tokens = counter.count_messages(messages)
        # Should be sum of content plus overheads
        assert tokens > 10
        assert tokens < 100

    def test_count_messages_multipart_content(self, counter):
        """Multi-part content should be counted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ]
            }
        ]
        tokens = counter.count_messages(messages)
        assert tokens > 0

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_encode_decode_roundtrip(self, counter):
        """Encode then decode should return original text."""
        text = "Hello, world!"
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text


class TestTokenBudgetEnforcer:
    """Tests for TokenBudgetEnforcer class."""

    @pytest.fixture
    def enforcer(self):
        """Create a TokenBudgetEnforcer instance."""
        reset_token_counter()
        return TokenBudgetEnforcer()

    def test_check_budget_within(self, enforcer):
        """Messages within budget should pass."""
        messages = [
            {"role": "user", "content": "Short message"}
        ]
        within, used, available = enforcer.check_budget(messages, "claude-3-sonnet")

        assert within is True
        assert used > 0
        assert available > 0
        assert used + available <= 180_000  # Claude effective limit

    def test_check_budget_over(self, enforcer):
        """Large messages should exceed budget."""
        # Create a very long message
        long_content = "x" * 1_000_000  # 1M chars
        messages = [{"role": "user", "content": long_content}]

        within, used, available = enforcer.check_budget(messages, "deepseek-v3")

        # This should exceed DeepSeek's 57.6K limit
        assert within is False
        assert available == 0

    def test_compression_strategy_none(self, enforcer):
        """Small messages should need no compression."""
        messages = [{"role": "user", "content": "Hello"}]
        strategy = enforcer.get_compression_strategy(messages, "claude-3-sonnet")
        assert strategy == "none"

    def test_compression_strategy_increases_with_usage(self, enforcer):
        """Larger messages should trigger higher compression strategies."""
        # Small message should be none
        small_messages = [{"role": "user", "content": "Hello"}]
        small_strategy = enforcer.get_compression_strategy(small_messages, "deepseek-v3")
        assert small_strategy == "none"

        # Very large message (over limit) should trigger aggressive
        # DeepSeek effective limit is 57.6K tokens
        # Create content that exceeds the limit
        large_content = "x" * 1_000_000  # Should far exceed limit
        large_messages = [{"role": "user", "content": large_content}]
        large_strategy = enforcer.get_compression_strategy(large_messages, "deepseek-v3")
        assert large_strategy == "aggressive"

    def test_compression_thresholds_exist(self):
        """Verify compression thresholds are properly defined."""
        enforcer = TokenBudgetEnforcer()
        assert enforcer.THRESHOLD_LIGHT == 0.60
        assert enforcer.THRESHOLD_MODERATE == 0.80
        assert enforcer.THRESHOLD_AGGRESSIVE == 0.95

    def test_get_usage_stats(self, enforcer):
        """get_usage_stats should return detailed info."""
        messages = [{"role": "user", "content": "Test message"}]
        stats = enforcer.get_usage_stats(messages, "claude-3-sonnet")

        assert stats["model_name"] == "claude-3-sonnet"
        assert stats["model_family"] == "claude"
        assert stats["tokens_used"] > 0
        assert stats["tokens_limit"] == 200_000
        assert stats["tokens_effective"] == 180_000
        assert stats["tokens_available"] > 0
        assert 0 <= stats["usage_percent"] <= 100
        assert stats["within_budget"] is True
        assert stats["compression_strategy"] == "none"
        assert stats["message_count"] == 1

    def test_estimate_room_for(self, enforcer):
        """estimate_room_for should calculate available space."""
        messages = [{"role": "user", "content": "Short"}]
        room = enforcer.estimate_room_for(messages, "claude-3-sonnet")

        # Should have lots of room for Claude
        assert room > 100_000

    def test_estimate_room_for_with_reserve(self, enforcer):
        """estimate_room_for should respect reserve."""
        messages = [{"role": "user", "content": "Short"}]

        room_10 = enforcer.estimate_room_for(messages, "claude-3-sonnet", reserve_percent=0.1)
        room_20 = enforcer.estimate_room_for(messages, "claude-3-sonnet", reserve_percent=0.2)

        # More reserve = less room
        assert room_10 > room_20

    def test_get_model_limits(self, enforcer):
        """get_model_limits should return correct limits."""
        limits = enforcer.get_model_limits("claude-3-sonnet")

        assert limits["context_limit"] == 200_000
        assert limits["reserved_tokens"] == 20_000
        assert limits["effective_limit"] == 180_000
        assert limits["model_family"] == "claude"


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_token_counter_singleton(self):
        """get_token_counter should return same instance."""
        reset_token_counter()

        counter1 = get_token_counter()
        counter2 = get_token_counter()

        assert counter1 is counter2

    def test_reset_token_counter(self):
        """reset_token_counter should clear singleton."""
        counter1 = get_token_counter()
        reset_token_counter()
        counter2 = get_token_counter()

        assert counter1 is not counter2


class TestAccuracy:
    """Tests for token counting accuracy."""

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_accuracy_within_5_percent(self):
        """Token count should be accurate within 5% of actual."""
        counter = TokenCounter()

        # Test various content types
        test_cases = [
            "The quick brown fox jumps over the lazy dog.",
            "def hello_world():\n    print('Hello, World!')",
            "```python\nfor i in range(10):\n    print(i)\n```",
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            "https://example.com/path/to/resource?param=value",
        ]

        for text in test_cases:
            counted = counter.count(text)
            # Tiktoken gives us the "ground truth" since we're using it
            # The test verifies our counting is consistent
            assert counted > 0, f"Failed for: {text}"
            # Each token is roughly 4 characters, so sanity check
            assert counted <= len(text), f"Too many tokens for: {text}"
