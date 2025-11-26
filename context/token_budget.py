"""
Token Budget Enforcer

Phase G - Token counting and budget enforcement across different models.

Model limits:
- Claude Sonnet 4.5: 200K context, 20K reserved = 180K effective
- DeepSeek V3: 64K context, 6.4K reserved = 57.6K effective

Uses tiktoken cl100k_base encoding for accurate token counting.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Try to import tiktoken, gracefully handle if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using approximate token counting")


class ModelFamily(Enum):
    """Model family for token counting and budget enforcement."""
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GPT = "gpt"
    UNKNOWN = "unknown"


@dataclass
class ModelLimits:
    """Token limits for a model family."""
    context_limit: int
    reserved_tokens: int
    encoding_name: str = "cl100k_base"

    @property
    def effective_limit(self) -> int:
        """Get effective token limit (context - reserved)."""
        return self.context_limit - self.reserved_tokens


# Model configurations
MODEL_CONFIGS: Dict[ModelFamily, ModelLimits] = {
    ModelFamily.CLAUDE: ModelLimits(
        context_limit=200_000,
        reserved_tokens=20_000,
        encoding_name="cl100k_base"
    ),
    ModelFamily.DEEPSEEK: ModelLimits(
        context_limit=64_000,
        reserved_tokens=6_400,
        encoding_name="cl100k_base"
    ),
    ModelFamily.GPT: ModelLimits(
        context_limit=128_000,
        reserved_tokens=12_800,
        encoding_name="cl100k_base"
    ),
    ModelFamily.UNKNOWN: ModelLimits(
        context_limit=32_000,
        reserved_tokens=3_200,
        encoding_name="cl100k_base"
    ),
}

# Model name to family mapping
MODEL_FAMILY_MAP: Dict[str, ModelFamily] = {
    # Claude models
    "claude-3-opus": ModelFamily.CLAUDE,
    "claude-3-sonnet": ModelFamily.CLAUDE,
    "claude-3-haiku": ModelFamily.CLAUDE,
    "claude-3.5-sonnet": ModelFamily.CLAUDE,
    "claude-sonnet-4": ModelFamily.CLAUDE,
    "claude-opus-4": ModelFamily.CLAUDE,
    "claude-4.5-sonnet": ModelFamily.CLAUDE,
    # DeepSeek models
    "deepseek-v3": ModelFamily.DEEPSEEK,
    "deepseek-coder": ModelFamily.DEEPSEEK,
    "deepseek-chat": ModelFamily.DEEPSEEK,
    # GPT models
    "gpt-4": ModelFamily.GPT,
    "gpt-4-turbo": ModelFamily.GPT,
    "gpt-4o": ModelFamily.GPT,
}


def get_model_family(model_name: str) -> ModelFamily:
    """
    Determine model family from model name.

    Args:
        model_name: Full model name or identifier

    Returns:
        ModelFamily enum value
    """
    model_lower = model_name.lower()

    # Direct lookup
    if model_lower in MODEL_FAMILY_MAP:
        return MODEL_FAMILY_MAP[model_lower]

    # Partial matching
    if "claude" in model_lower:
        return ModelFamily.CLAUDE
    if "deepseek" in model_lower:
        return ModelFamily.DEEPSEEK
    if "gpt" in model_lower:
        return ModelFamily.GPT

    return ModelFamily.UNKNOWN


class TokenCounter:
    """
    Counts tokens using tiktoken encoding.

    Uses cl100k_base encoding by default, which is accurate for
    most modern LLMs including Claude and GPT-4.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token counter.

        Args:
            encoding_name: Tiktoken encoding to use
        """
        self.encoding_name = encoding_name
        self._encoder = None
        self._lock = Lock()

        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
                logger.debug(f"TokenCounter initialized with {encoding_name}")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")

    def count(self, text: str, model_family: Optional[ModelFamily] = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for
            model_family: Optional model family for encoding selection

        Returns:
            Token count
        """
        if not text:
            return 0

        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}")

        # Fallback: approximate count (4 chars per token)
        return self._approximate_count(text)

    def _approximate_count(self, text: str) -> int:
        """
        Approximate token count when tiktoken unavailable.

        Uses ~4 characters per token as a rough estimate.
        """
        # More accurate approximation based on word/character ratio
        words = len(text.split())
        chars = len(text)

        # Blend word-based and char-based estimates
        word_estimate = int(words * 1.3)  # ~1.3 tokens per word
        char_estimate = chars // 4  # ~4 chars per token

        return max(word_estimate, char_estimate)

    def count_messages(
        self,
        messages: List[Dict[str, Any]],
        model_family: Optional[ModelFamily] = None
    ) -> int:
        """
        Count tokens in a list of messages (chat format).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_family: Optional model family

        Returns:
            Total token count including message overhead
        """
        total = 0

        # Base overhead per message (role, separators, etc.)
        MESSAGE_OVERHEAD = 4

        for msg in messages:
            # Count role
            role = msg.get("role", "")
            total += self.count(role, model_family)

            # Count content
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content, model_family)
            elif isinstance(content, list):
                # Handle multi-part content (e.g., images, text blocks)
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        total += self.count(text, model_family)
                    elif isinstance(part, str):
                        total += self.count(part, model_family)

            # Message overhead
            total += MESSAGE_OVERHEAD

        # Conversation overhead
        total += 3

        return total

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if self._encoder:
            return self._encoder.encode(text)
        return []

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if self._encoder:
            return self._encoder.decode(tokens)
        return ""


class TokenBudgetEnforcer:
    """
    Enforces token budgets across different models.

    Monitors token usage and determines when compression is needed.
    """

    # Compression thresholds
    THRESHOLD_AGGRESSIVE = 0.95  # >95% usage
    THRESHOLD_MODERATE = 0.80    # >80% usage
    THRESHOLD_LIGHT = 0.60      # >60% usage

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize the budget enforcer.

        Args:
            token_counter: TokenCounter instance (creates one if not provided)
        """
        self.counter = token_counter or TokenCounter()

    def check_budget(
        self,
        messages: List[Dict[str, Any]],
        model_name: str
    ) -> Tuple[bool, int, int]:
        """
        Check if messages are within token budget.

        Args:
            messages: List of messages to check
            model_name: Model name for determining limits

        Returns:
            Tuple of (within_budget, tokens_used, tokens_available)
        """
        model_family = get_model_family(model_name)
        limits = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS[ModelFamily.UNKNOWN])

        tokens_used = self.counter.count_messages(messages, model_family)
        effective_limit = limits.effective_limit

        within_budget = tokens_used <= effective_limit
        tokens_available = max(0, effective_limit - tokens_used)

        logger.debug(
            f"Budget check: {tokens_used}/{effective_limit} tokens "
            f"({tokens_used/effective_limit*100:.1f}%) - "
            f"{'OK' if within_budget else 'OVER BUDGET'}"
        )

        return within_budget, tokens_used, tokens_available

    def get_compression_strategy(
        self,
        messages: List[Dict[str, Any]],
        model_name: str
    ) -> str:
        """
        Determine compression strategy based on current usage.

        Args:
            messages: Current messages
            model_name: Model name

        Returns:
            Strategy name: 'aggressive', 'moderate', 'light', or 'none'
        """
        model_family = get_model_family(model_name)
        limits = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS[ModelFamily.UNKNOWN])

        tokens_used = self.counter.count_messages(messages, model_family)
        usage_ratio = tokens_used / limits.effective_limit

        if usage_ratio > self.THRESHOLD_AGGRESSIVE:
            strategy = "aggressive"
        elif usage_ratio > self.THRESHOLD_MODERATE:
            strategy = "moderate"
        elif usage_ratio > self.THRESHOLD_LIGHT:
            strategy = "light"
        else:
            strategy = "none"

        logger.debug(
            f"Compression strategy: {strategy} "
            f"(usage: {usage_ratio*100:.1f}%)"
        )

        return strategy

    def get_usage_stats(
        self,
        messages: List[Dict[str, Any]],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Get detailed token usage statistics.

        Args:
            messages: Current messages
            model_name: Model name

        Returns:
            Dict with usage statistics
        """
        model_family = get_model_family(model_name)
        limits = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS[ModelFamily.UNKNOWN])

        tokens_used = self.counter.count_messages(messages, model_family)
        effective_limit = limits.effective_limit
        usage_ratio = tokens_used / effective_limit

        return {
            "model_name": model_name,
            "model_family": model_family.value,
            "tokens_used": tokens_used,
            "tokens_limit": limits.context_limit,
            "tokens_reserved": limits.reserved_tokens,
            "tokens_effective": effective_limit,
            "tokens_available": max(0, effective_limit - tokens_used),
            "usage_percent": round(usage_ratio * 100, 2),
            "within_budget": tokens_used <= effective_limit,
            "compression_strategy": self.get_compression_strategy(messages, model_name),
            "message_count": len(messages)
        }

    def estimate_room_for(
        self,
        current_messages: List[Dict[str, Any]],
        model_name: str,
        reserve_percent: float = 0.1
    ) -> int:
        """
        Estimate how many tokens are available for new content.

        Args:
            current_messages: Current message history
            model_name: Model name
            reserve_percent: Additional reserve (default 10%)

        Returns:
            Available tokens for new content
        """
        _, tokens_used, tokens_available = self.check_budget(
            current_messages, model_name
        )

        model_family = get_model_family(model_name)
        limits = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS[ModelFamily.UNKNOWN])

        additional_reserve = int(limits.effective_limit * reserve_percent)
        available = max(0, tokens_available - additional_reserve)

        return available

    def get_model_limits(self, model_name: str) -> Dict[str, int]:
        """
        Get token limits for a model.

        Args:
            model_name: Model name

        Returns:
            Dict with context_limit, reserved_tokens, effective_limit
        """
        model_family = get_model_family(model_name)
        limits = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS[ModelFamily.UNKNOWN])

        return {
            "context_limit": limits.context_limit,
            "reserved_tokens": limits.reserved_tokens,
            "effective_limit": limits.effective_limit,
            "model_family": model_family.value
        }


# Singleton pattern for global access
_token_counter: Optional[TokenCounter] = None
_counter_lock = Lock()


def get_token_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """
    Get or create the global TokenCounter instance.

    Args:
        encoding_name: Tiktoken encoding to use

    Returns:
        The singleton TokenCounter instance
    """
    global _token_counter

    with _counter_lock:
        if _token_counter is None:
            _token_counter = TokenCounter(encoding_name)
        return _token_counter


def reset_token_counter() -> None:
    """Reset the global TokenCounter instance (for testing)."""
    global _token_counter
    with _counter_lock:
        _token_counter = None
