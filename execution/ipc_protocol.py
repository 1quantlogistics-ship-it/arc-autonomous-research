"""
IPC Protocol - JSON File-Based Inter-Process Communication

Provides reliable communication between parent ARC process and
child training subprocesses using JSON files.

Design principles:
- File-based for simplicity and crash resilience
- Atomic writes using write-then-rename
- Polling-based message passing
- Supports structured message types

Author: ARC Team (Dev 1)
Created: 2025-11-26
"""

import json
import os
import shutil
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of IPC messages."""
    # Parent → Child
    START = "start"
    STOP = "stop"
    CHECKPOINT = "checkpoint"

    # Child → Parent
    READY = "ready"
    PROGRESS = "progress"
    METRICS = "metrics"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class IPCMessage:
    """
    Structured IPC message.

    Attributes:
        msg_type: Type of message
        payload: Message data
        timestamp: ISO format timestamp
        experiment_id: Associated experiment ID
        sequence: Message sequence number
    """
    msg_type: MessageType
    payload: Dict[str, Any]
    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "msg_type": self.msg_type.value,
            "payload": self.payload,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IPCMessage":
        """Create IPCMessage from dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            payload=data["payload"],
            experiment_id=data["experiment_id"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            sequence=data.get("sequence", 0),
        )


class IPCProtocol:
    """
    JSON file-based IPC protocol for subprocess communication.

    Communication is done through JSON files in a shared directory:
    - parent_to_child.json: Commands from parent to child
    - child_to_parent.json: Status updates from child to parent

    Uses atomic writes (write-then-rename) to prevent partial reads.
    """

    PARENT_TO_CHILD = "parent_to_child.json"
    CHILD_TO_PARENT = "child_to_parent.json"
    LOCK_SUFFIX = ".lock"

    def __init__(
        self,
        ipc_dir: Path,
        experiment_id: str,
        is_parent: bool = True
    ):
        """
        Initialize IPC protocol.

        Args:
            ipc_dir: Directory for IPC files
            experiment_id: Experiment identifier
            is_parent: True if this is the parent process
        """
        self.ipc_dir = Path(ipc_dir)
        self.experiment_id = experiment_id
        self.is_parent = is_parent
        self._sequence = 0
        self._lock = Lock()
        self._last_read_sequence: Dict[str, int] = {}

        # Ensure IPC directory exists
        self.ipc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"IPCProtocol initialized: dir={ipc_dir}, experiment={experiment_id}, is_parent={is_parent}")

    @property
    def _send_file(self) -> Path:
        """File to write outgoing messages."""
        if self.is_parent:
            return self.ipc_dir / self.PARENT_TO_CHILD
        return self.ipc_dir / self.CHILD_TO_PARENT

    @property
    def _recv_file(self) -> Path:
        """File to read incoming messages."""
        if self.is_parent:
            return self.ipc_dir / self.CHILD_TO_PARENT
        return self.ipc_dir / self.PARENT_TO_CHILD

    def send(self, msg_type: MessageType, payload: Dict[str, Any]) -> None:
        """
        Send a message to the other process.

        Args:
            msg_type: Type of message
            payload: Message data
        """
        with self._lock:
            self._sequence += 1
            message = IPCMessage(
                msg_type=msg_type,
                payload=payload,
                experiment_id=self.experiment_id,
                sequence=self._sequence,
            )
            self._write_atomic(self._send_file, message.to_dict())

            logger.debug(f"IPC sent: {msg_type.value} seq={self._sequence}")

    def receive(self, timeout: float = 0.0) -> Optional[IPCMessage]:
        """
        Receive a message from the other process.

        Args:
            timeout: Max time to wait for message (0 = non-blocking)

        Returns:
            IPCMessage if available, None otherwise
        """
        start_time = time.time()

        while True:
            message = self._read_message()
            if message is not None:
                return message

            if timeout <= 0:
                return None

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None

            time.sleep(0.1)  # Poll interval

    def receive_all(self) -> List[IPCMessage]:
        """
        Receive all pending messages.

        Returns:
            List of messages (may be empty)
        """
        messages = []
        while True:
            msg = self.receive(timeout=0)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def wait_for(
        self,
        msg_type: MessageType,
        timeout: float = 30.0
    ) -> Optional[IPCMessage]:
        """
        Wait for a specific message type.

        Args:
            msg_type: Message type to wait for
            timeout: Maximum wait time

        Returns:
            IPCMessage if received, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            message = self.receive(timeout=0.5)
            if message is not None and message.msg_type == msg_type:
                return message

        logger.warning(f"Timeout waiting for {msg_type.value}")
        return None

    def _read_message(self) -> Optional[IPCMessage]:
        """Read and parse message from file."""
        recv_file = self._recv_file

        if not recv_file.exists():
            return None

        try:
            with open(recv_file, 'r') as f:
                data = json.load(f)

            # Check if this is a new message
            file_key = str(recv_file)
            last_seq = self._last_read_sequence.get(file_key, 0)

            msg_seq = data.get("sequence", 0)
            if msg_seq <= last_seq:
                return None  # Already read this message

            self._last_read_sequence[file_key] = msg_seq
            return IPCMessage.from_dict(data)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in IPC file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading IPC message: {e}")
            return None

    def _write_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data atomically using write-then-rename."""
        temp_path = path.with_suffix(".tmp")

        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            shutil.move(str(temp_path), str(path))

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise IOError(f"Failed to write IPC message: {e}") from e

    def cleanup(self) -> None:
        """Clean up IPC files."""
        for f in [self._send_file, self._recv_file]:
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up {f}: {e}")

    # Convenience methods for common message types

    def send_start(self, config: Dict[str, Any]) -> None:
        """Send start command with training config."""
        self.send(MessageType.START, {"config": config})

    def send_stop(self, reason: str = "requested") -> None:
        """Send stop command."""
        self.send(MessageType.STOP, {"reason": reason})

    def send_checkpoint_request(self) -> None:
        """Request child to save checkpoint."""
        self.send(MessageType.CHECKPOINT, {})

    def send_ready(self) -> None:
        """Signal that child is ready to receive commands."""
        self.send(MessageType.READY, {})

    def send_progress(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        **extra
    ) -> None:
        """Send training progress update."""
        self.send(MessageType.PROGRESS, {
            "epoch": epoch,
            "batch": batch,
            "total_batches": total_batches,
            "loss": loss,
            **extra,
        })

    def send_metrics(self, metrics: Dict[str, float]) -> None:
        """Send training metrics."""
        self.send(MessageType.METRICS, {"metrics": metrics})

    def send_checkpoint_saved(self, path: str) -> None:
        """Signal that checkpoint was saved."""
        self.send(MessageType.CHECKPOINT_SAVED, {"path": path})

    def send_error(self, error: str, traceback: Optional[str] = None) -> None:
        """Send error report."""
        self.send(MessageType.ERROR, {
            "error": error,
            "traceback": traceback,
        })

    def send_complete(self, results: Dict[str, Any]) -> None:
        """Signal training completion with results."""
        self.send(MessageType.COMPLETE, {"results": results})


def create_ipc_pair(
    experiment_id: str,
    base_dir: Optional[Path] = None
) -> tuple:
    """
    Create parent and child IPC protocol instances.

    Args:
        experiment_id: Experiment identifier
        base_dir: Base directory for IPC files

    Returns:
        (parent_ipc, child_ipc) tuple
    """
    if base_dir is None:
        base_dir = Path("/tmp/arc_ipc")

    ipc_dir = base_dir / experiment_id

    parent = IPCProtocol(ipc_dir, experiment_id, is_parent=True)
    child = IPCProtocol(ipc_dir, experiment_id, is_parent=False)

    return parent, child
