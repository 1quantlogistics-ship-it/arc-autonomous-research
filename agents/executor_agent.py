"""
ExecutorAgent: Training execution and results collection
=========================================================

The Executor translates approved proposals into training jobs
and collects experimental results.

Updated for Phase G Bulletproof Execution:
- Uses SubprocessExecutor for crash-isolated training
- GPU pre-flight checks before experiment start
- IPC-based progress monitoring
- ExperimentRegistry for full lifecycle tracking (Dev 2)
- MetricsStreamer for live metrics updates (Dev 2)
"""

import time
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter

# Bulletproof execution imports (Dev 1)
try:
    from execution.subprocess_executor import SubprocessExecutor, ExecutionResult, ExecutionStatus
    from execution.gpu_manager import GPUManager, get_gpu_manager
    BULLETPROOF_EXECUTION_AVAILABLE = True
except ImportError:
    BULLETPROOF_EXECUTION_AVAILABLE = False
    SubprocessExecutor = None
    GPUManager = None

# Lifecycle and streaming imports (Dev 2)
try:
    from execution.experiment_lifecycle import (
        ExperimentState,
        ExperimentRecord,
        ExperimentRegistry,
        get_experiment_registry,
    )
    from execution.metrics_streamer import (
        MetricsStreamer,
        get_metrics_streamer,
    )
    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    ExperimentRegistry = None
    MetricsStreamer = None


class ExecutorAgent(BaseAgent):
    """
    Training execution agent.

    Responsibilities:
    - Generate safe config diffs
    - Execute training via SubprocessExecutor (crash-isolated)
    - GPU pre-flight checks before experiment start
    - Monitor training progress via IPC
    - Collect and report metrics
    """

    def __init__(
        self,
        agent_id: str = "executor_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.0,
        memory_path: str = "/workspace/arc/memory",
        experiments_dir: str = "/workspace/arc/experiments",
        use_subprocess_execution: bool = True,
        gpu_index: int = 0,
    ):
        """
        Initialize Executor agent.

        Args:
            agent_id: Unique agent identifier
            model: LLM model to use
            llm_router: LLM router instance
            voting_weight: Weight for voting
            memory_path: Path for agent memory
            experiments_dir: Directory for experiment data
            use_subprocess_execution: Use bulletproof subprocess execution
            gpu_index: Default GPU index for training
        """
        super().__init__(
            agent_id=agent_id,
            role="executor",
            model=model,
            capabilities=[AgentCapability.EXECUTION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)
        self.experiments_dir = Path(experiments_dir)

        # Bulletproof execution setup (Dev 1)
        self.use_subprocess_execution = use_subprocess_execution and BULLETPROOF_EXECUTION_AVAILABLE
        self.gpu_index = gpu_index
        self._subprocess_executor: Optional[SubprocessExecutor] = None
        self._gpu_manager: Optional[GPUManager] = None

        if self.use_subprocess_execution:
            self._subprocess_executor = SubprocessExecutor()
            self._gpu_manager = get_gpu_manager()

        # Lifecycle and streaming setup (Dev 2)
        self._registry: Optional[ExperimentRegistry] = None
        self._streamer: Optional[MetricsStreamer] = None

        if LIFECYCLE_AVAILABLE:
            try:
                self._registry = get_experiment_registry(
                    storage_path=str(self.experiments_dir / "registry")
                )
                self._streamer = get_metrics_streamer(poll_interval=1.0)
                self._streamer.add_callback(self._on_metrics_update)
            except Exception:
                pass  # Graceful degradation

    @property
    def subprocess_executor(self) -> Optional[SubprocessExecutor]:
        """Get subprocess executor instance."""
        return self._subprocess_executor

    @property
    def gpu_manager(self) -> Optional[GPUManager]:
        """Get GPU manager instance."""
        return self._gpu_manager

    @property
    def registry(self) -> Optional[ExperimentRegistry]:
        """Get experiment registry instance (Dev 2)."""
        return self._registry

    @property
    def streamer(self) -> Optional[MetricsStreamer]:
        """Get metrics streamer instance (Dev 2)."""
        return self._streamer

    def _on_metrics_update(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int,
        timestamp: str,
    ) -> None:
        """Callback when live metrics are received from training (Dev 2)."""
        if self._registry:
            self._registry.update_metrics(experiment_id, metrics)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute approved experiments.

        Args:
            input_data: Contains approved proposals

        Returns:
            Execution status and commands
        """
        import time
        start_time = time.time()

        try:
            # Read approved proposals
            reviews = self.read_memory("reviews.json")
            proposals = self.read_memory("proposals.json")

            # Filter approved proposals
            approved = self._get_approved_proposals(proposals, reviews)

            # Build execution plan
            prompt = self._build_execution_prompt(approved)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate execution plan
            response = client.generate_json(prompt, max_tokens=2000, temperature=0.5)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("plan_execution", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("plan_execution", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executors check if proposal is technically feasible.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision based on feasibility
        """
        # Check if config changes are valid
        config_changes = proposal.get("config_changes", {})

        # Simple validation: ensure numeric params are reasonable
        for param, value in config_changes.items():
            if isinstance(value, (int, float)):
                if value < 0 or value > 1e6:
                    return {
                        "decision": "reject",
                        "confidence": 0.9,
                        "reasoning": f"Parameter {param}={value} is outside reasonable range",
                        "suggested_changes": None
                    }

        # Default: approve
        return {
            "decision": "approve",
            "confidence": 0.8,
            "reasoning": "Configuration is technically feasible"
        }

    def _get_approved_proposals(
        self,
        proposals: Dict[str, Any],
        reviews: Dict[str, Any]
    ) -> list:
        """Filter proposals to only approved ones."""
        approved = []
        proposal_list = proposals.get("proposals", [])
        review_list = reviews.get("reviews", [])

        # Create review lookup
        review_map = {r["experiment_id"]: r for r in review_list}

        for proposal in proposal_list:
            exp_id = proposal["experiment_id"]
            review = review_map.get(exp_id, {})

            if review.get("decision") == "approve":
                approved.append(proposal)

        return approved

    def _build_execution_prompt(self, approved_proposals: list) -> str:
        """Build prompt for execution planning."""
        return f"""You are the Executor agent in ARC (Autonomous Research Collective).
Your role is to translate approved proposals into executable training commands.

# Approved Proposals
{approved_proposals}

# Your Task
For each approved proposal, generate:
1. Training command (safe, validated)
2. Estimated duration
3. Resource requirements

Return ONLY a valid JSON object:
{{
  "executions": [
    {{
      "experiment_id": "exp_XXX",
      "command": "python train.py --param1 value1 --param2 value2",
      "estimated_duration_minutes": NN,
      "status": "queued",
      "notes": "execution notes"
    }}
  ]
}}"""

    # =========================================================================
    # BULLETPROOF EXECUTION METHODS (Phase G)
    # =========================================================================

    def execute_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        timeout_seconds: float = 3600,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cycle_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute a single experiment with subprocess isolation and full lifecycle tracking.

        This provides crash-isolated training:
        - Training runs in separate subprocess
        - Crashes don't kill ARC system
        - Timeout enforcement with graceful shutdown
        - Emergency checkpointing on crash/timeout
        - Full lifecycle tracking via ExperimentRegistry (Dev 2)
        - Live metrics streaming via MetricsStreamer (Dev 2)

        Args:
            experiment_id: Unique experiment identifier
            config: Training configuration
            timeout_seconds: Maximum runtime
            progress_callback: Called with progress updates
            cycle_id: Research cycle ID for tracking

        Returns:
            Execution result dictionary
        """
        start_time = time.time()

        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Register experiment in registry (Dev 2)
        record = None
        if self._registry and LIFECYCLE_AVAILABLE:
            try:
                record = ExperimentRecord(
                    experiment_id=experiment_id,
                    cycle_id=cycle_id,
                    proposal_id=config.get("proposal_id", experiment_id),
                    config=config,
                    timeout_seconds=int(timeout_seconds),
                )
                self._registry.register(record)
                self._registry.update_state(experiment_id, ExperimentState.PENDING, reason="Created")
            except ValueError:
                # Already registered
                record = self._registry.get(experiment_id)

        # Setup metrics streaming (Dev 2)
        metrics_file = exp_dir / "metrics.jsonl"
        if self._streamer:
            self._streamer.register_experiment(experiment_id, str(metrics_file))

        try:
            if not self.use_subprocess_execution:
                # Fallback to legacy execution (not crash-isolated)
                if self._registry:
                    self._registry.update_state(
                        experiment_id, ExperimentState.FAILED,
                        reason="Subprocess execution not available"
                    )
                return self._legacy_execute(experiment_id, config)

            # GPU pre-flight check
            if self._registry:
                self._registry.update_state(experiment_id, ExperimentState.QUEUED, reason="GPU validation")

            preflight_ok, preflight_msg = self._gpu_manager.preflight_check(
                config,
                gpu_index=self.gpu_index,
            )

            if not preflight_ok:
                if self._registry:
                    self._registry.update_state(
                        experiment_id, ExperimentState.FAILED,
                        reason=f"Pre-flight check failed: {preflight_msg}"
                    )
                return {
                    "experiment_id": experiment_id,
                    "status": "failed",
                    "error": f"Pre-flight check failed: {preflight_msg}",
                    "duration_seconds": time.time() - start_time,
                }

            # Update state to RUNNING
            if self._registry:
                self._registry.update_state(experiment_id, ExperimentState.RUNNING, reason="Starting subprocess")

            # Execute in subprocess
            result = self._subprocess_executor.execute(
                experiment_id=experiment_id,
                config=config,
                timeout_seconds=timeout_seconds,
                progress_callback=progress_callback,
            )

            # Track task
            duration_ms = (time.time() - start_time) * 1000
            success = result.status == ExecutionStatus.COMPLETED
            self._track_task("execute_experiment", success=success, duration_ms=duration_ms)

            # Update final state in registry (Dev 2)
            if self._registry:
                final_state = self._map_status_to_state(result.status)
                self._registry.update_state(
                    experiment_id,
                    final_state,
                    reason="Execution complete",
                    metrics=result.metrics if hasattr(result, "metrics") else {},
                    exit_code=result.exit_code if hasattr(result, "exit_code") else None,
                )

            return result.to_dict()

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("execute_experiment", success=False, duration_ms=duration_ms)

            # Update registry with failure (Dev 2)
            if self._registry:
                self._registry.update_state(
                    experiment_id, ExperimentState.CRASHED,
                    reason=str(e)[:500]
                )

            return {
                "experiment_id": experiment_id,
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time,
            }

        finally:
            # Stop streaming for this experiment (Dev 2)
            if self._streamer:
                self._streamer.unregister_experiment(experiment_id)

    def _map_status_to_state(self, status) -> ExperimentState:
        """Map ExecutionStatus to ExperimentState (Dev 2)."""
        if not LIFECYCLE_AVAILABLE:
            return None

        status_map = {
            "completed": ExperimentState.COMPLETED,
            "failed": ExperimentState.FAILED,
            "timeout": ExperimentState.TIMEOUT,
            "crashed": ExperimentState.CRASHED,
            "cancelled": ExperimentState.CANCELLED,
        }

        # Handle ExecutionStatus enum
        status_str = status.value if hasattr(status, "value") else str(status).lower()
        return status_map.get(status_str, ExperimentState.FAILED)

    def _legacy_execute(
        self,
        experiment_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Legacy execution without subprocess isolation."""
        return {
            "experiment_id": experiment_id,
            "status": "queued",
            "message": "Subprocess execution not available, using legacy mode",
            "config": config,
        }

    def check_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU status.

        Returns:
            Dictionary with GPU information
        """
        if self._gpu_manager is None:
            return {"available": False, "message": "GPU manager not initialized"}

        return self._gpu_manager.get_status_summary()

    def estimate_memory(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate GPU memory requirements for a configuration.

        Args:
            config: Training configuration

        Returns:
            Memory estimate dictionary
        """
        if self._gpu_manager is None:
            return {"error": "GPU manager not initialized"}

        estimate = self._gpu_manager.estimate_memory(config)
        return estimate.to_dict()

    def stop_current_execution(self) -> bool:
        """
        Request graceful stop of current execution.

        Returns:
            True if stop was requested, False if nothing running
        """
        if self._subprocess_executor is None:
            return False

        if not self._subprocess_executor.is_running:
            return False

        self._subprocess_executor.stop()
        return True

    def request_checkpoint(self) -> bool:
        """
        Request the running process to save a checkpoint.

        Returns:
            True if request was sent, False if nothing running
        """
        if self._subprocess_executor is None:
            return False

        if not self._subprocess_executor.is_running:
            return False

        self._subprocess_executor.request_checkpoint()
        return True

    @property
    def is_training(self) -> bool:
        """Check if training is currently running."""
        if self._subprocess_executor is None:
            return False
        return self._subprocess_executor.is_running

    @property
    def latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics from current/last execution."""
        if self._subprocess_executor is None:
            return {}
        return self._subprocess_executor.latest_metrics

    # =========================================================================
    # LIFECYCLE METHODS (Dev 2)
    # =========================================================================

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get status of an experiment from registry (Dev 2).

        Args:
            experiment_id: Experiment to query

        Returns:
            Experiment record as dict, or error
        """
        if not self._registry:
            return {"error": "Registry not available"}

        record = self._registry.get(experiment_id)
        if not record:
            return {"error": f"Experiment {experiment_id} not found"}

        return record.to_dict()

    def list_experiments(
        self,
        state: Optional[str] = None,
        cycle_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List experiments from registry (Dev 2).

        Args:
            state: Filter by state (e.g., "completed", "running")
            cycle_id: Filter by cycle ID
            limit: Maximum number to return

        Returns:
            List of experiment records
        """
        if not self._registry:
            return []

        if state and LIFECYCLE_AVAILABLE:
            try:
                state_enum = ExperimentState(state)
                records = self._registry.get_by_state(state_enum)
            except ValueError:
                records = list(self._registry._experiments.values())
        elif cycle_id is not None:
            records = self._registry.get_by_cycle(cycle_id)
        else:
            records = list(self._registry._experiments.values())

        # Sort by creation time (most recent first) and limit
        records = sorted(records, key=lambda r: r.created_at, reverse=True)[:limit]
        return [r.to_dict() for r in records]

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics (Dev 2).

        Returns:
            Statistics about experiments
        """
        if not self._registry:
            return {"error": "Registry not available"}

        return self._registry.get_stats()

    def recover_incomplete_experiments(self) -> List[Dict[str, Any]]:
        """
        Find and recover incomplete experiments after crash (Dev 2).

        Called on ARC startup to handle experiments that were
        interrupted by a crash or restart.

        Returns:
            List of recovery actions taken
        """
        if not self._registry:
            return []

        incomplete = self._registry.get_incomplete()
        results = []

        for record in incomplete:
            exp_dir = self.experiments_dir / record.experiment_id
            checkpoint_dir = exp_dir / "checkpoints"

            # Check for existing checkpoints
            checkpoints = []
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                checkpoints = [c for c in checkpoints if c.name != "latest.pt"]

            if checkpoints:
                # Has checkpoint - can potentially resume
                latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
                results.append({
                    "experiment_id": record.experiment_id,
                    "action": "can_resume",
                    "checkpoint": str(latest_ckpt),
                    "previous_state": record.state.value if hasattr(record.state, "value") else str(record.state),
                })
            else:
                # No checkpoint - mark as crashed
                if LIFECYCLE_AVAILABLE:
                    self._registry.update_state(
                        record.experiment_id,
                        ExperimentState.CRASHED,
                        reason="Recovered on startup - no checkpoint available"
                    )
                results.append({
                    "experiment_id": record.experiment_id,
                    "action": "marked_crashed",
                    "previous_state": record.state.value if hasattr(record.state, "value") else str(record.state),
                })

        return results
