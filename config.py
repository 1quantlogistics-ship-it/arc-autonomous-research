"""
ARC Configuration Management

Centralized configuration using Pydantic BaseSettings with environment variable support.
Eliminates hard-coded paths and enables easy dev/test/prod environment switching.

Environment Variables:
    ARC_HOME: Root directory for ARC system (default: /workspace/arc)
    ARC_MEMORY_DIR: Memory files directory
    ARC_EXPERIMENTS_DIR: Experiments directory
    ARC_LOGS_DIR: Logs directory
    ARC_MODE: Operating mode (SEMI/AUTO/FULL/OFF)
    ARC_LLM_ENDPOINT: LLM API endpoint URL
    ARC_LLM_TIMEOUT: LLM request timeout in seconds
    ARC_ENV: Environment name (dev/test/prod)

Example Usage:
    from config import get_settings

    settings = get_settings()
    print(settings.memory_dir)  # /workspace/arc/memory

    # Override for testing
    settings = get_settings(environment="test")
    print(settings.memory_dir)  # /tmp/arc_test/memory
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class ARCSettings(BaseSettings):
    """
    ARC system configuration with environment variable support.

    All paths can be overridden via environment variables.
    Supports multiple environment profiles (dev/test/prod).
    """

    model_config = SettingsConfigDict(
        env_prefix='ARC_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # ========================================================================
    # Environment & Version
    # ========================================================================

    environment: Literal["dev", "test", "prod"] = Field(
        default="dev",
        description="Runtime environment (determines defaults)"
    )

    arc_version: str = Field(
        default="1.1.0",
        description="ARC system version"
    )

    # ========================================================================
    # Directory Structure
    # ========================================================================

    home: Path = Field(
        default=Path("/workspace/arc"),
        description="Root directory for ARC system"
    )

    memory_dir: Optional[Path] = Field(
        default=None,
        description="Memory files directory (defaults to {home}/memory)"
    )

    experiments_dir: Optional[Path] = Field(
        default=None,
        description="Experiments directory (defaults to {home}/experiments)"
    )

    logs_dir: Optional[Path] = Field(
        default=None,
        description="Logs directory (defaults to {home}/logs)"
    )

    checkpoints_dir: Optional[Path] = Field(
        default=None,
        description="Model checkpoints directory (defaults to {home}/checkpoints)"
    )

    snapshots_dir: Optional[Path] = Field(
        default=None,
        description="Snapshots directory for rollback (defaults to {home}/snapshots)"
    )

    # ========================================================================
    # LLM Configuration
    # ========================================================================

    llm_endpoint: str = Field(
        default="http://localhost:8000/v1",
        description="LLM API endpoint URL"
    )

    llm_model: str = Field(
        default="kimi",
        description="LLM model identifier"
    )

    llm_timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="LLM request timeout in seconds"
    )

    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for LLM calls"
    )

    llm_retry_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Initial retry delay in seconds (exponential backoff)"
    )

    historian_timeout: int = Field(
        default=600,
        ge=60,
        le=1800,
        description="Historian LLM timeout in seconds (longer for deep reasoning)"
    )

    # ========================================================================
    # Operating Mode & Safety
    # ========================================================================

    mode: Literal["SEMI", "AUTO", "FULL", "OFF"] = Field(
        default="SEMI",
        description="Operating mode (SEMI=safest, FULL=autonomous)"
    )

    require_approval_for_train: bool = Field(
        default=True,
        description="Require human approval before training"
    )

    require_approval_for_commands: bool = Field(
        default=True,
        description="Require human approval for exec commands"
    )

    max_concurrent_experiments: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum concurrent training jobs"
    )

    # ========================================================================
    # Command Execution
    # ========================================================================

    allowed_commands: List[str] = Field(
        default=[
            "python", "python3",
            "git", "dvc",
            "nvidia-smi", "nvcc",
            "ls", "pwd", "cat", "head", "tail",
            "grep", "find", "wc",
            "pip", "conda"
        ],
        description="Allowlist of executable commands"
    )

    command_timeout: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Command execution timeout in seconds"
    )

    # ========================================================================
    # Training Configuration
    # ========================================================================

    default_gpu_id: int = Field(
        default=0,
        ge=0,
        le=7,
        description="Default GPU device ID"
    )

    max_training_time: int = Field(
        default=7200,
        ge=60,
        le=86400,
        description="Maximum training time in seconds"
    )

    # ========================================================================
    # Constraint Defaults
    # ========================================================================

    max_learning_rate: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Maximum allowed learning rate"
    )

    min_learning_rate: float = Field(
        default=1e-7,
        gt=0.0,
        description="Minimum allowed learning rate"
    )

    min_batch_size: int = Field(
        default=1,
        ge=1,
        description="Minimum batch size"
    )

    max_batch_size: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum batch size"
    )

    # ========================================================================
    # Logging
    # ========================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )

    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )

    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )

    log_to_console: bool = Field(
        default=True,
        description="Enable console logging"
    )

    # ========================================================================
    # API Server
    # ========================================================================

    api_host: str = Field(
        default="0.0.0.0",
        description="Control Plane API host"
    )

    api_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Control Plane API port"
    )

    api_debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode"
    )

    # ========================================================================
    # Historian & Memory
    # ========================================================================

    max_recent_experiments: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Maximum recent experiments to keep in history"
    )

    stagnation_threshold: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Cycles without improvement before triggering mode change"
    )

    regression_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive regressions before triggering recovery mode"
    )

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator('home', 'memory_dir', 'experiments_dir', 'logs_dir',
                    'checkpoints_dir', 'snapshots_dir', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v

    def model_post_init(self, __context) -> None:
        """Set derived paths and environment-specific defaults."""
        # Set default subdirectories if not specified
        if self.memory_dir is None:
            self.memory_dir = self.home / "memory"
        if self.experiments_dir is None:
            self.experiments_dir = self.home / "experiments"
        if self.logs_dir is None:
            self.logs_dir = self.home / "logs"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.home / "checkpoints"
        if self.snapshots_dir is None:
            self.snapshots_dir = self.home / "snapshots"

        # Environment-specific overrides
        if self.environment == "test":
            self._apply_test_settings()
        elif self.environment == "prod":
            self._apply_prod_settings()

    def _apply_test_settings(self) -> None:
        """Apply test environment settings."""
        # Use temporary directory for tests
        import tempfile
        test_root = Path(tempfile.gettempdir()) / "arc_test"

        if self.home == Path("/workspace/arc"):  # Only override if using default
            self.home = test_root
            self.memory_dir = test_root / "memory"
            self.experiments_dir = test_root / "experiments"
            self.logs_dir = test_root / "logs"
            self.checkpoints_dir = test_root / "checkpoints"
            self.snapshots_dir = test_root / "snapshots"

        # Test-specific settings
        self.llm_timeout = 10  # Shorter timeouts for tests
        self.command_timeout = 30
        self.max_training_time = 60
        self.log_level = "DEBUG"

    def _apply_prod_settings(self) -> None:
        """Apply production environment settings."""
        # Production should always require approvals
        self.require_approval_for_train = True
        self.require_approval_for_commands = True
        self.mode = "SEMI"  # Start in safest mode
        self.api_debug = False
        self.log_level = "INFO"

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        directories = [
            self.home,
            self.memory_dir,
            self.experiments_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.snapshots_dir,
        ]

        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    def get_memory_file_path(self, filename: str) -> Path:
        """Get full path for a memory file."""
        return self.memory_dir / filename

    def get_experiment_path(self, experiment_id: str) -> Path:
        """Get directory path for an experiment."""
        return self.experiments_dir / experiment_id

    def get_log_file_path(self, log_type: str = "arc") -> Path:
        """Get path for a log file."""
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        return self.logs_dir / f"{log_type}_{timestamp}.log"

    def get_snapshot_path(self, snapshot_id: str) -> Path:
        """Get directory path for a snapshot."""
        return self.snapshots_dir / snapshot_id

    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return self.model_dump()


# ============================================================================
# Settings Factory with Caching
# ============================================================================

@lru_cache()
def get_settings(environment: Optional[str] = None) -> ARCSettings:
    """
    Get ARC settings instance (cached).

    Args:
        environment: Override environment (dev/test/prod).
                    If None, reads from ARC_ENVIRONMENT or defaults to 'dev'.

    Returns:
        Configured ARCSettings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.memory_dir)

        >>> test_settings = get_settings(environment="test")
        >>> print(test_settings.memory_dir)  # Uses temp directory
    """
    env = environment or os.getenv("ARC_ENVIRONMENT", "dev")

    # Clear cache if environment changes
    if environment is not None:
        os.environ["ARC_ENVIRONMENT"] = environment

    return ARCSettings(environment=env)


def reset_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()


# ============================================================================
# Pre-configured Settings Instances
# ============================================================================

def get_dev_settings() -> ARCSettings:
    """Get development environment settings."""
    return get_settings(environment="dev")


def get_test_settings() -> ARCSettings:
    """Get test environment settings."""
    return get_settings(environment="test")


def get_prod_settings() -> ARCSettings:
    """Get production environment settings."""
    return get_settings(environment="prod")


# ============================================================================
# Environment Detection
# ============================================================================

def detect_environment() -> Literal["dev", "test", "prod"]:
    """
    Auto-detect runtime environment.

    Returns:
        Detected environment name
    """
    # Check explicit environment variable
    env = os.getenv("ARC_ENVIRONMENT")
    if env in ["dev", "test", "prod"]:
        return env

    # Check if running in pytest
    if "PYTEST_CURRENT_TEST" in os.environ:
        return "test"

    # Check for production indicators
    if os.path.exists("/workspace/arc") and os.getenv("RUNPOD_POD_ID"):
        return "prod"

    # Default to dev
    return "dev"


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_configuration(settings: ARCSettings) -> tuple[bool, List[str]]:
    """
    Validate configuration and return issues.

    Args:
        settings: Settings to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check directory accessibility
    for dir_name, dir_path in [
        ("home", settings.home),
        ("memory_dir", settings.memory_dir),
        ("experiments_dir", settings.experiments_dir),
        ("logs_dir", settings.logs_dir),
    ]:
        if dir_path and not dir_path.parent.exists():
            issues.append(f"{dir_name} parent directory does not exist: {dir_path.parent}")

    # Check LLM endpoint
    if not settings.llm_endpoint.startswith(("http://", "https://")):
        issues.append(f"Invalid LLM endpoint URL: {settings.llm_endpoint}")

    # Check learning rate constraints
    if settings.min_learning_rate >= settings.max_learning_rate:
        issues.append("min_learning_rate must be less than max_learning_rate")

    # Check batch size constraints
    if settings.min_batch_size > settings.max_batch_size:
        issues.append("min_batch_size must be less than or equal to max_batch_size")

    # Check mode safety
    if settings.environment == "prod" and settings.mode == "FULL":
        issues.append("FULL mode is not recommended for production")

    return len(issues) == 0, issues


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    # Example: Load settings
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"ARC Home: {settings.home}")
    print(f"Memory Dir: {settings.memory_dir}")
    print(f"LLM Endpoint: {settings.llm_endpoint}")
    print(f"Operating Mode: {settings.mode}")

    # Validate
    is_valid, issues = validate_configuration(settings)
    if is_valid:
        print("\n✅ Configuration is valid")
    else:
        print("\n❌ Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")

    # Example: Test environment
    test_settings = get_test_settings()
    print(f"\nTest Memory Dir: {test_settings.memory_dir}")
    print(f"Test Timeout: {test_settings.llm_timeout}s")
