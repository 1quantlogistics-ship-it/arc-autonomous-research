"""
Model Configurations: Available LLM models and their capabilities
==================================================================

Defines model endpoints, capabilities, and routing rules.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for an LLM model.

    Attributes:
        model_id: Unique model identifier
        model_name: Human-readable name
        endpoint: API endpoint URL
        provider: Provider (vllm, openai, anthropic, local)
        context_window: Maximum context length in tokens
        max_output_tokens: Maximum output tokens
        capabilities: List of capabilities (strategy, coding, safety, etc.)
        cost_per_1k_tokens: Estimated cost per 1000 tokens (for budgeting)
        offline: Whether model runs offline
        api_key_required: Whether API key is needed
    """
    model_id: str
    model_name: str
    endpoint: str
    provider: str
    context_window: int
    max_output_tokens: int
    capabilities: List[str]
    cost_per_1k_tokens: float = 0.0
    offline: bool = False
    api_key_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "provider": self.provider,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "capabilities": self.capabilities,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "offline": self.offline,
            "api_key_required": self.api_key_required
        }


# Predefined model configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "deepseek-r1": ModelConfig(
        model_id="deepseek-r1",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        endpoint="http://localhost:8000/v1/completions",
        provider="vllm",
        context_window=32768,
        max_output_tokens=4096,
        capabilities=["reasoning", "analysis", "constraints", "safety"],
        cost_per_1k_tokens=0.14,
        offline=False,
        api_key_required=False
    ),
    "claude-sonnet-4.5": ModelConfig(
        model_id="claude-sonnet-4.5",
        model_name="Claude Sonnet 4.5",
        endpoint="https://api.anthropic.com/v1/messages",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        capabilities=["strategy", "planning", "reasoning", "coding"],
        cost_per_1k_tokens=3.0,
        offline=False,
        api_key_required=True
    ),
    "qwen2.5-32b": ModelConfig(
        model_id="qwen2.5-32b",
        model_name="Qwen 2.5 32B",
        endpoint="http://localhost:8001/generate",
        provider="vllm",
        context_window=32768,
        max_output_tokens=4096,
        capabilities=["data_tasks", "analysis", "safety_review"],
        cost_per_1k_tokens=0.0,  # Self-hosted
        offline=False,
        api_key_required=False
    ),
    "llama-3-8b-local": ModelConfig(
        model_id="llama-3-8b-local",
        model_name="Llama 3 8B (Local)",
        endpoint="http://localhost:8002/generate",
        provider="local",
        context_window=8192,
        max_output_tokens=2048,
        capabilities=["validation", "sanity_checking", "schema_validation"],
        cost_per_1k_tokens=0.0,  # Local
        offline=True,
        api_key_required=False
    ),
    "mock-llm": ModelConfig(
        model_id="mock-llm",
        model_name="Mock LLM (Offline)",
        endpoint="mock://offline",
        provider="mock",
        context_window=100000,
        max_output_tokens=10000,
        capabilities=["all"],
        cost_per_1k_tokens=0.0,
        offline=True,
        api_key_required=False
    )
}


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """
    Get model configuration by ID.

    Args:
        model_id: Model identifier

    Returns:
        ModelConfig or None if not found
    """
    return MODEL_REGISTRY.get(model_id)


def get_models_by_capability(capability: str) -> List[ModelConfig]:
    """
    Get all models with a specific capability.

    Args:
        capability: Capability to filter by

    Returns:
        List of matching ModelConfig objects
    """
    return [
        config for config in MODEL_REGISTRY.values()
        if capability in config.capabilities or "all" in config.capabilities
    ]


def get_offline_models() -> List[ModelConfig]:
    """
    Get all offline-capable models.

    Returns:
        List of offline ModelConfig objects
    """
    return [
        config for config in MODEL_REGISTRY.values()
        if config.offline
    ]


def list_all_models() -> List[ModelConfig]:
    """
    Get all registered models.

    Returns:
        List of all ModelConfig objects
    """
    return list(MODEL_REGISTRY.values())


def register_model(config: ModelConfig) -> None:
    """
    Register a new model configuration.

    Args:
        config: ModelConfig to register
    """
    MODEL_REGISTRY[config.model_id] = config


def get_model_summary() -> Dict[str, Any]:
    """
    Get summary of all registered models.

    Returns:
        Summary dictionary with model statistics
    """
    total = len(MODEL_REGISTRY)
    offline = len(get_offline_models())
    providers = set(config.provider for config in MODEL_REGISTRY.values())

    return {
        "total_models": total,
        "offline_models": offline,
        "online_models": total - offline,
        "providers": list(providers),
        "models": {
            model_id: {
                "name": config.model_name,
                "provider": config.provider,
                "offline": config.offline
            }
            for model_id, config in MODEL_REGISTRY.items()
        }
    }
