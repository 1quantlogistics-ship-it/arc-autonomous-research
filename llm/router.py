"""
LLMRouter: Route agent roles to specific LLM models
====================================================

Enables multi-model architecture by routing different agent roles
to different LLMs based on capabilities and availability.
"""

from typing import Dict, Optional
from llm.client import LLMClient
from llm.mock_client import MockLLMClient
from llm.models import ModelConfig, get_model_config
from llm.health_monitor import get_health_monitor
from config import get_settings


class LLMRouter:
    """
    Routes agent roles to appropriate LLM models.

    Supports:
    - Role-specific model assignment
    - Fallback to default model
    - Offline mode (uses MockLLMClient)
    - Model capability matching
    """

    def __init__(
        self,
        role_to_model: Optional[Dict[str, str]] = None,
        default_model: str = "mock-llm",
        offline_mode: bool = False,
        enable_health_monitoring: bool = True
    ):
        """
        Initialize LLM router.

        Args:
            role_to_model: Mapping of role -> model_id
            default_model: Default model if role not mapped
            offline_mode: Use mock clients instead of real LLMs
            enable_health_monitoring: Use health monitor for failover
        """
        self.role_to_model = role_to_model or self._get_default_routing()
        self.default_model = default_model
        self.offline_mode = offline_mode
        self.enable_health_monitoring = enable_health_monitoring

        # Cache of initialized clients
        self._client_cache: Dict[str, LLMClient] = {}

        # Health monitor for automatic failover
        if enable_health_monitoring and not offline_mode:
            self.health_monitor = get_health_monitor()
        else:
            self.health_monitor = None

    def get_client_for_role(self, role: str) -> LLMClient:
        """
        Get LLM client for a specific agent role.

        Args:
            role: Agent role (director, architect, critic, etc.)

        Returns:
            LLMClient instance configured for that role
        """
        # Determine model for this role
        model_id = self.role_to_model.get(role, self.default_model)

        # Check cache
        cache_key = f"{role}:{model_id}"
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        # Create new client with role-specific timeout
        client = self._create_client(model_id, role=role)
        self._client_cache[cache_key] = client

        return client

    def _create_client(self, model_id: str, role: Optional[str] = None) -> LLMClient:
        """
        Create LLM client for a specific model with health-based failover.

        Args:
            model_id: Model identifier
            role: Agent role (for role-specific timeout configuration)

        Returns:
            LLMClient or MockLLMClient instance
        """
        # Offline mode: always use mock
        if self.offline_mode:
            return MockLLMClient(model_name=model_id)

        # Health monitoring: check if model is available
        if self.health_monitor:
            # Check model availability and get best available (with fallback)
            best_model_id = self.health_monitor.get_best_available_model(model_id)

            if best_model_id != model_id:
                print(f"Health Monitor: Failing over {model_id} -> {best_model_id}")
                model_id = best_model_id

        # Get model config
        config = get_model_config(model_id)

        if not config:
            # Unknown model: fallback to mock
            print(f"Warning: Unknown model {model_id}, using mock client")
            return MockLLMClient(model_name=model_id)

        # Use mock for offline models or if model is "mock-llm"
        if config.offline or config.model_id == "mock-llm":
            return MockLLMClient(model_name=config.model_name)

        # Get timeout from settings (role-specific for Historian)
        settings = get_settings()
        timeout = settings.historian_timeout if role == "historian" else settings.llm_timeout

        # Create real LLM client
        return LLMClient(
            endpoint=config.endpoint,
            model_name=config.model_name,
            api_key=None,  # TODO: Load from env
            timeout=timeout,
            max_retries=settings.llm_max_retries
        )

    def update_routing(self, role: str, model_id: str) -> None:
        """
        Update routing for a specific role.

        Args:
            role: Agent role
            model_id: New model to assign
        """
        self.role_to_model[role] = model_id

        # Invalidate cache for this role
        cache_keys_to_remove = [
            k for k in self._client_cache.keys()
            if k.startswith(f"{role}:")
        ]
        for key in cache_keys_to_remove:
            del self._client_cache[key]

    def set_offline_mode(self, offline: bool) -> None:
        """
        Toggle offline mode.

        Args:
            offline: True for offline (mock clients)
        """
        if offline != self.offline_mode:
            self.offline_mode = offline
            # Clear cache to force recreation
            self._client_cache.clear()

    def get_routing_summary(self) -> Dict[str, str]:
        """
        Get current routing configuration.

        Returns:
            Dictionary of role -> model_id mappings
        """
        return self.role_to_model.copy()

    @staticmethod
    def _get_default_routing() -> Dict[str, str]:
        """
        Get default role -> model routing.

        Returns:
            Default routing dictionary
        """
        return {
            "director": "deepseek-r1",            # Strategy (local)
            "architect": "deepseek-r1",           # Analysis + proposals
            "critic": "deepseek-r1",              # Safety review
            "critic_secondary": "deepseek-r1",    # Secondary safety
            "historian": "deepseek-r1",           # Memory management
            "executor": "deepseek-r1",            # Execution
            "explorer": "deepseek-r1",            # Parameter exploration
            "parameter_scientist": "deepseek-r1", # Hyperparameter optimization
            "supervisor": "deepseek-r1",          # Validation (local)
            "validator": "deepseek-r1"            # Schema validation (local)
        }

    def __repr__(self) -> str:
        return (
            f"<LLMRouter "
            f"offline={self.offline_mode} "
            f"roles={len(self.role_to_model)} "
            f"cached_clients={len(self._client_cache)}>"
        )
