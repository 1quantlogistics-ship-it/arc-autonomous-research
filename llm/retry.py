"""
Retry utilities with exponential backoff for LLM calls.
Handles transient failures from Claude, DeepSeek, Qwen, Llama APIs.

Phase F - Infrastructure & Stability Track
"""
import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Any, Optional, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: Tuple[Type[Exception], ...] = field(default_factory=lambda: (
        TimeoutError,
        ConnectionError,
        asyncio.TimeoutError,
    ))

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())  # 50-150% of calculated delay
        return delay


# Default policies per use case
DEFAULT_LLM_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
)

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
)

QUICK_FAIL_POLICY = RetryPolicy(
    max_retries=1,
    base_delay=0.5,
    max_delay=5.0,
)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


async def with_retry(
    func: Callable[..., T],
    policy: RetryPolicy = DEFAULT_LLM_POLICY,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    *args,
    **kwargs
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        policy: RetryPolicy configuration
        on_retry: Optional callback when retry occurs (exception, attempt_number)
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result of successful function call

    Raises:
        RetryExhaustedError: When all retries are exhausted
    """
    last_exception = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except policy.retryable_exceptions as e:
            last_exception = e

            if attempt < policy.max_retries:
                delay = policy.get_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{policy.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt + 1)

                await asyncio.sleep(delay)
            else:
                logger.error(f"All {policy.max_retries + 1} attempts exhausted: {e}")

    raise RetryExhaustedError(last_exception, policy.max_retries + 1)


def retry_decorator(policy: RetryPolicy = DEFAULT_LLM_POLICY):
    """Decorator version of retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await with_retry(func, policy, None, *args, **kwargs)
        return wrapper
    return decorator


class RetryContext:
    """Context manager for tracking retry state across multiple operations."""

    def __init__(self, policy: RetryPolicy = DEFAULT_LLM_POLICY):
        self.policy = policy
        self.total_retries = 0
        self.failed_operations = []

    async def execute(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute with retry, tracking failures."""
        try:
            return await with_retry(func, self.policy, None, *args, **kwargs)
        except RetryExhaustedError as e:
            self.failed_operations.append({
                'operation': operation_name,
                'error': str(e.last_exception),
                'attempts': e.attempts
            })
            raise

    @property
    def has_failures(self) -> bool:
        return len(self.failed_operations) > 0

    def get_summary(self) -> dict:
        """Get summary of retry operations."""
        return {
            'total_retries': self.total_retries,
            'failed_operations': self.failed_operations,
            'has_failures': self.has_failures
        }


# Synchronous retry for compatibility with existing sync code
def sync_with_retry(
    func: Callable[..., T],
    policy: RetryPolicy = DEFAULT_LLM_POLICY,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    *args,
    **kwargs
) -> T:
    """
    Execute a synchronous function with retry logic.

    Args:
        func: Function to execute
        policy: RetryPolicy configuration
        on_retry: Optional callback when retry occurs (exception, attempt_number)
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result of successful function call

    Raises:
        RetryExhaustedError: When all retries are exhausted
    """
    import time
    last_exception = None

    for attempt in range(policy.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except policy.retryable_exceptions as e:
            last_exception = e

            if attempt < policy.max_retries:
                delay = policy.get_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{policy.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt + 1)

                time.sleep(delay)
            else:
                logger.error(f"All {policy.max_retries + 1} attempts exhausted: {e}")

    raise RetryExhaustedError(last_exception, policy.max_retries + 1)
