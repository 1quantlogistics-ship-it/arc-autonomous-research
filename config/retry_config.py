"""
Agent-specific retry configurations.
Different agents may need different retry behaviors based on their criticality.

Phase F - Infrastructure & Stability Track
"""
from llm.retry import RetryPolicy

# Per-agent retry policies
AGENT_RETRY_POLICIES = {
    # Critical agents - more retries
    "supervisor": RetryPolicy(max_retries=5, base_delay=2.0, max_delay=60.0),
    "director": RetryPolicy(max_retries=4, base_delay=2.0, max_delay=45.0),

    # Standard agents
    "architect": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
    "explorer": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
    "parameter_scientist": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
    "instructor": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
    "critic": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
    "historian": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),

    # Executor - may have longer operations
    "executor": RetryPolicy(max_retries=2, base_delay=5.0, max_delay=120.0),
}

# Per-LLM provider policies (some are more reliable than others)
LLM_PROVIDER_POLICIES = {
    "claude": RetryPolicy(max_retries=3, base_delay=1.0, max_delay=30.0),
    "deepseek": RetryPolicy(max_retries=4, base_delay=2.0, max_delay=45.0),
    "qwen": RetryPolicy(max_retries=4, base_delay=2.0, max_delay=45.0),
    "llama": RetryPolicy(max_retries=3, base_delay=1.5, max_delay=30.0),
}


def get_retry_policy(agent_name: str = None, provider: str = None) -> RetryPolicy:
    """
    Get appropriate retry policy for agent/provider combination.

    Args:
        agent_name: Name of the agent (e.g., 'director', 'architect')
        provider: LLM provider name (e.g., 'claude', 'deepseek')

    Returns:
        RetryPolicy configured for the agent/provider
    """
    if agent_name and agent_name in AGENT_RETRY_POLICIES:
        return AGENT_RETRY_POLICIES[agent_name]
    if provider and provider in LLM_PROVIDER_POLICIES:
        return LLM_PROVIDER_POLICIES[provider]
    return RetryPolicy()  # Default


def get_provider_from_model(model_id: str) -> str:
    """
    Extract provider name from model ID.

    Args:
        model_id: Model identifier (e.g., 'claude-sonnet-4.5', 'deepseek-r1')

    Returns:
        Provider name
    """
    model_lower = model_id.lower()
    if 'claude' in model_lower:
        return 'claude'
    elif 'deepseek' in model_lower:
        return 'deepseek'
    elif 'qwen' in model_lower:
        return 'qwen'
    elif 'llama' in model_lower:
        return 'llama'
    return 'unknown'
