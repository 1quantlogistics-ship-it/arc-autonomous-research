"""
Tests for retry logic and retry configuration.

Phase F - Infrastructure & Stability Track
"""
import pytest
import asyncio
from llm.retry import (
    RetryPolicy, with_retry, RetryExhaustedError,
    retry_decorator, RetryContext, sync_with_retry
)
from config.retry_config import get_retry_policy, get_provider_from_model


@pytest.mark.asyncio
async def test_successful_first_attempt():
    """Should return immediately on success."""
    call_count = 0

    async def succeed():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await with_retry(succeed, RetryPolicy(max_retries=3))
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_on_failure():
    """Should retry on transient failure."""
    call_count = 0

    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("Simulated timeout")
        return "success"

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
    result = await with_retry(fail_then_succeed, policy)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    """Should raise RetryExhaustedError when all attempts fail."""
    async def always_fail():
        raise ConnectionError("Always fails")

    policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)

    with pytest.raises(RetryExhaustedError) as exc_info:
        await with_retry(always_fail, policy)

    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, ConnectionError)


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Verify delays increase exponentially."""
    policy = RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0,
        jitter=False
    )

    assert policy.get_delay(0) == 1.0
    assert policy.get_delay(1) == 2.0
    assert policy.get_delay(2) == 4.0


@pytest.mark.asyncio
async def test_max_delay_cap():
    """Delay should not exceed max_delay."""
    policy = RetryPolicy(
        max_retries=10,
        base_delay=1.0,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=False
    )

    assert policy.get_delay(10) == 5.0  # Would be 1024 without cap


@pytest.mark.asyncio
async def test_decorator():
    """Test retry_decorator works correctly."""
    call_count = 0

    @retry_decorator(RetryPolicy(max_retries=2, base_delay=0.01, jitter=False))
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError()
        return "done"

    result = await flaky_function()
    assert result == "done"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_context_tracking():
    """RetryContext should track failures."""
    ctx = RetryContext(RetryPolicy(max_retries=1, base_delay=0.01, jitter=False))

    async def always_fail():
        raise TimeoutError("fail")

    with pytest.raises(RetryExhaustedError):
        await ctx.execute("test_op", always_fail)

    assert ctx.has_failures
    assert len(ctx.failed_operations) == 1
    assert ctx.failed_operations[0]['operation'] == "test_op"


@pytest.mark.asyncio
async def test_retry_context_success():
    """RetryContext should track successful operations."""
    ctx = RetryContext(RetryPolicy(max_retries=1, base_delay=0.01))

    async def succeed():
        return "ok"

    result = await ctx.execute("test_op", succeed)
    assert result == "ok"
    assert not ctx.has_failures


@pytest.mark.asyncio
async def test_on_retry_callback():
    """on_retry callback should be called on each retry."""
    retries_logged = []

    def log_retry(exc, attempt):
        retries_logged.append((type(exc).__name__, attempt))

    call_count = 0

    async def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("timeout")
        return "ok"

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
    result = await with_retry(fail_twice, policy, log_retry)

    assert result == "ok"
    assert len(retries_logged) == 2
    assert retries_logged[0] == ("TimeoutError", 1)
    assert retries_logged[1] == ("TimeoutError", 2)


def test_sync_retry_success():
    """Sync retry should work for synchronous functions."""
    call_count = 0

    def succeed():
        nonlocal call_count
        call_count += 1
        return "success"

    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    result = sync_with_retry(succeed, policy)
    assert result == "success"
    assert call_count == 1


def test_sync_retry_with_failures():
    """Sync retry should retry on failures."""
    call_count = 0

    def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("network error")
        return "recovered"

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
    result = sync_with_retry(fail_then_succeed, policy)
    assert result == "recovered"
    assert call_count == 2


# Tests for retry configuration
def test_get_retry_policy_for_agent():
    """Should return agent-specific policy."""
    policy = get_retry_policy(agent_name="supervisor")
    assert policy.max_retries == 5

    policy = get_retry_policy(agent_name="executor")
    assert policy.max_retries == 2


def test_get_retry_policy_for_provider():
    """Should return provider-specific policy."""
    policy = get_retry_policy(provider="deepseek")
    assert policy.max_retries == 4


def test_get_retry_policy_default():
    """Should return default policy for unknown agent/provider."""
    policy = get_retry_policy(agent_name="unknown_agent")
    assert policy.max_retries == 3  # Default


def test_get_provider_from_model():
    """Should extract provider from model ID."""
    assert get_provider_from_model("claude-sonnet-4.5") == "claude"
    assert get_provider_from_model("deepseek-r1") == "deepseek"
    assert get_provider_from_model("qwen2.5-32b") == "qwen"
    assert get_provider_from_model("llama-3-8b-local") == "llama"
    assert get_provider_from_model("unknown-model") == "unknown"


@pytest.mark.asyncio
async def test_jitter_adds_randomness():
    """Jitter should make delays non-deterministic."""
    policy_with_jitter = RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0,
        jitter=True
    )

    # Get multiple delays and check they vary
    delays = [policy_with_jitter.get_delay(1) for _ in range(10)]

    # With jitter, delays should vary between 0.5x and 1.5x base
    # At attempt 1, base delay = 2.0, so range is 1.0 to 3.0
    assert all(1.0 <= d <= 3.0 for d in delays)
    # Check there's actual variation
    assert len(set(delays)) > 1
