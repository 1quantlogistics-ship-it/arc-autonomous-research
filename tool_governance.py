"""
Tool Governance Layer

Provides validation, safety checks, and transactional execution for
all tool calls (exec, train, eval) in the Control Plane.

Features:
- Schema validation of tool requests
- Constraint checking against safety boundaries
- Transactional execution with automatic rollback
- Audit trail logging
- Resource limits enforcement
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import json

from memory_handler import MemoryHandler, get_memory_handler, ValidationFailedError
from config import get_settings, ARCSettings
from schemas import (
    Constraints, ForbiddenRange, SystemState, OperatingMode,
    ActiveExperiment
)

logger = logging.getLogger(__name__)


class ToolGovernanceError(Exception):
    """Base exception for tool governance errors."""
    pass


class ToolValidationError(ToolGovernanceError):
    """Raised when tool validation fails."""
    pass


class ToolExecutionError(ToolGovernanceError):
    """Raised when tool execution fails."""
    pass


class ToolGovernance:
    """
    Tool governance layer for Control Plane.

    Ensures all tool calls are:
    - Schema-validated
    - Constraint-checked
    - Transactionally executed
    - Audit-logged
    - Resource-limited

    Example:
        governance = ToolGovernance()

        # Validate tool request
        is_valid, error = governance.validate_tool_request("train", train_config)

        # Execute with automatic rollback
        with governance.tool_transaction("train", cycle_id=10):
            result = execute_training(train_config)
    """

    def __init__(
        self,
        settings: Optional[ARCSettings] = None,
        memory: Optional[MemoryHandler] = None
    ):
        """
        Initialize tool governance.

        Args:
            settings: Optional settings (uses get_settings() if None)
            memory: Optional memory handler (uses get_memory_handler() if None)
        """
        self.settings = settings or get_settings()
        self.memory = memory or get_memory_handler(settings)

        # Audit log path
        self.audit_log_path = self.settings.logs_dir / "tool_governance.jsonl"

        logger.info("ToolGovernance initialized")

    # ========================================================================
    # Tool Request Validation
    # ========================================================================

    def validate_tool_request(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        role: str = "unknown"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a tool request against constraints and safety boundaries.

        Args:
            tool_name: Tool to execute (exec, train, eval)
            tool_args: Tool arguments
            role: Agent role making request

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            logger.info(f"Validating {tool_name} request from {role}")

            # Load constraints
            constraints = self.memory.load_constraints()

            # Tool-specific validation
            if tool_name == "train":
                return self._validate_train_request(tool_args, constraints)
            elif tool_name == "exec":
                return self._validate_exec_request(tool_args, constraints)
            elif tool_name == "eval":
                return self._validate_eval_request(tool_args, constraints)
            else:
                return False, f"Unknown tool: {tool_name}"

        except ValidationFailedError as e:
            logger.error(f"Validation failed: {e}")
            return False, f"Schema validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def _validate_train_request(
        self,
        train_args: Dict[str, Any],
        constraints: Constraints
    ) -> Tuple[bool, Optional[str]]:
        """Validate training request against constraints."""
        config = train_args.get("config", {})
        validation_errors = []

        # Check each parameter against forbidden ranges
        for param, value in config.items():
            for forbidden in constraints.forbidden_ranges:
                if forbidden.param == param:
                    # Check min boundary
                    if forbidden.min is not None and value < forbidden.min:
                        validation_errors.append(
                            f"Parameter {param}={value} below safe range "
                            f"(min={forbidden.min}). Reason: {forbidden.reason}"
                        )

                    # Check max boundary
                    if forbidden.max is not None and value > forbidden.max:
                        validation_errors.append(
                            f"Parameter {param}={value} above safe range "
                            f"(max={forbidden.max}). Reason: {forbidden.reason}"
                        )

        # Check for required parameters
        if "experiment_id" not in train_args:
            validation_errors.append("Missing required field: experiment_id")

        if validation_errors:
            return False, "; ".join(validation_errors)

        return True, None

    def _validate_exec_request(
        self,
        exec_args: Dict[str, Any],
        constraints: Constraints
    ) -> Tuple[bool, Optional[str]]:
        """Validate exec request against command allowlist."""
        command = exec_args.get("command", "")

        # Check command allowlist
        if not command:
            return False, "Missing required field: command"

        # Parse command
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"

        base_cmd = cmd_parts[0]

        # Check against allowlist from config
        if base_cmd not in self.settings.allowed_commands:
            return False, f"Command '{base_cmd}' not in allowlist"

        return True, None

    def _validate_eval_request(
        self,
        eval_args: Dict[str, Any],
        constraints: Constraints
    ) -> Tuple[bool, Optional[str]]:
        """Validate eval request."""
        # Check required fields
        if "experiment_id" not in eval_args:
            return False, "Missing required field: experiment_id"

        if "metrics" not in eval_args:
            return False, "Missing required field: metrics"

        # Check metrics list
        metrics = eval_args.get("metrics", [])
        if not isinstance(metrics, list):
            return False, "Field 'metrics' must be a list"

        if len(metrics) == 0:
            return False, "Field 'metrics' cannot be empty"

        return True, None

    # ========================================================================
    # Mode Permission Checks
    # ========================================================================

    def check_mode_permission(self, tool_name: str, requires_approval: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Check if tool execution is allowed in current mode.

        Args:
            tool_name: Tool to execute
            requires_approval: Whether this request requires human approval

        Returns:
            Tuple of (is_allowed, message)
        """
        try:
            # Load system state
            state = self.memory.load_system_state()
            mode = state.mode

            # SEMI mode: all require approval
            if mode == OperatingMode.SEMI:
                if requires_approval:
                    return False, "Command requires human approval in SEMI mode"
                return True, None

            # AUTO mode: training blocked
            if mode == OperatingMode.AUTO:
                if tool_name == "train":
                    return False, "Training not permitted in AUTO mode"
                return True, None

            # FULL mode: everything allowed
            if mode == OperatingMode.FULL:
                return True, None

            return False, f"Unknown operating mode: {mode}"

        except Exception as e:
            logger.error(f"Mode check failed: {e}")
            return False, f"Mode check error: {str(e)}"

    # ========================================================================
    # Transactional Execution
    # ========================================================================

    def tool_transaction(self, tool_name: str, cycle_id: int):
        """
        Context manager for transactional tool execution.

        Creates a memory backup before execution and rolls back on failure.

        Example:
            with governance.tool_transaction("train", cycle_id=10):
                result = execute_training(config)

        Args:
            tool_name: Tool being executed
            cycle_id: Current cycle ID

        Returns:
            Context manager
        """
        return ToolTransaction(
            governance=self,
            tool_name=tool_name,
            cycle_id=cycle_id
        )

    def execute_with_rollback(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        cycle_id: int,
        role: str = "unknown",
        execution_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Execute tool with automatic rollback on failure.

        Args:
            tool_name: Tool to execute
            tool_args: Tool arguments
            cycle_id: Current cycle ID
            role: Agent role making request
            execution_callback: Callback to execute tool (optional)

        Returns:
            Tool execution result

        Raises:
            ToolValidationError: If validation fails
            ToolExecutionError: If execution fails
        """
        # Validate request
        is_valid, error = self.validate_tool_request(tool_name, tool_args, role)
        if not is_valid:
            raise ToolValidationError(error)

        # Check mode permission
        is_allowed, message = self.check_mode_permission(tool_name)
        if not is_allowed:
            raise ToolValidationError(message)

        # Create backup
        backup_dir = self.memory.backup_memory()
        logger.info(f"Created backup for {tool_name} execution: {backup_dir}")

        try:
            # Log tool execution start
            self._log_tool_execution(
                tool_name=tool_name,
                tool_args=tool_args,
                cycle_id=cycle_id,
                role=role,
                status="started"
            )

            # Execute tool (callback provided by caller)
            if execution_callback:
                result = execution_callback()
            else:
                result = {"status": "no_callback_provided"}

            # Log success
            self._log_tool_execution(
                tool_name=tool_name,
                tool_args=tool_args,
                cycle_id=cycle_id,
                role=role,
                status="success",
                result=result
            )

            logger.info(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")

            # Rollback memory
            logger.info(f"Rolling back memory to {backup_dir}")
            self.memory.restore_memory(backup_dir)

            # Log failure
            self._log_tool_execution(
                tool_name=tool_name,
                tool_args=tool_args,
                cycle_id=cycle_id,
                role=role,
                status="failed",
                error=str(e)
            )

            raise ToolExecutionError(f"Tool execution failed: {str(e)}") from e

    # ========================================================================
    # Audit Logging
    # ========================================================================

    def _log_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        cycle_id: int,
        role: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log tool execution to audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "cycle_id": cycle_id,
            "role": role,
            "status": status,
            "args": tool_args,
            "result": result,
            "error": error
        }

        # Write to audit log
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.debug(f"Logged {tool_name} execution: {status}")

    def get_audit_trail(self, cycle_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail of tool executions.

        Args:
            cycle_id: Optional cycle ID to filter by

        Returns:
            List of audit log entries
        """
        if not self.audit_log_path.exists():
            return []

        audit_trail = []

        with open(self.audit_log_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Filter by cycle if specified
                    if cycle_id is None or entry.get("cycle_id") == cycle_id:
                        audit_trail.append(entry)

        return audit_trail

    # ========================================================================
    # Resource Limits
    # ========================================================================

    def check_resource_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if resource limits would be exceeded.

        Returns:
            Tuple of (within_limits, error_message)
        """
        try:
            # Load system state
            state = self.memory.load_system_state()

            # Check active experiments limit
            max_experiments = 10  # Could be from config
            active_count = len(state.active_experiments)

            if active_count >= max_experiments:
                return False, f"Maximum active experiments limit reached ({max_experiments})"

            return True, None

        except Exception as e:
            logger.error(f"Resource limit check failed: {e}")
            return False, f"Resource check error: {str(e)}"


class ToolTransaction:
    """Context manager for transactional tool execution."""

    def __init__(self, governance: ToolGovernance, tool_name: str, cycle_id: int):
        self.governance = governance
        self.tool_name = tool_name
        self.cycle_id = cycle_id
        self.backup_dir = None

    def __enter__(self):
        """Create backup before execution."""
        self.backup_dir = self.governance.memory.backup_memory()
        logger.info(f"Transaction started for {self.tool_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rollback on error."""
        if exc_type is not None:
            logger.error(f"Transaction failed for {self.tool_name}: {exc_val}")
            if self.backup_dir:
                logger.info("Rolling back transaction")
                self.governance.memory.restore_memory(self.backup_dir)
            return False  # Re-raise exception

        logger.info(f"Transaction completed for {self.tool_name}")
        return True


# ============================================================================
# Convenience Functions
# ============================================================================

_global_governance: Optional[ToolGovernance] = None


def get_tool_governance(
    settings: Optional[ARCSettings] = None,
    memory: Optional[MemoryHandler] = None
) -> ToolGovernance:
    """
    Get global tool governance instance.

    Args:
        settings: Optional settings
        memory: Optional memory handler

    Returns:
        ToolGovernance instance
    """
    global _global_governance

    if _global_governance is None:
        _global_governance = ToolGovernance(settings=settings, memory=memory)

    return _global_governance


def reset_tool_governance():
    """Reset global tool governance (useful for testing)."""
    global _global_governance
    _global_governance = None
