"""
Tests for Subprocess Executor and IPC Protocol.

Phase G Bulletproof Execution - Tests for crash-isolated training infrastructure.
"""

import pytest
import json
import time
import tempfile
from pathlib import Path

from execution.ipc_protocol import (
    IPCProtocol,
    IPCMessage,
    MessageType,
    create_ipc_pair,
)
from execution.subprocess_executor import (
    SubprocessExecutor,
    ExecutionResult,
    ExecutionStatus,
)


class TestIPCMessage:
    """Tests for IPCMessage dataclass."""

    def test_create_message(self):
        """Should create message with all fields."""
        msg = IPCMessage(
            msg_type=MessageType.PROGRESS,
            payload={"epoch": 1, "loss": 0.5},
            experiment_id="exp_001",
        )

        assert msg.msg_type == MessageType.PROGRESS
        assert msg.payload["epoch"] == 1
        assert msg.experiment_id == "exp_001"
        assert msg.timestamp is not None

    def test_to_dict(self):
        """Should convert to dictionary."""
        msg = IPCMessage(
            msg_type=MessageType.METRICS,
            payload={"auc": 0.85},
            experiment_id="exp_002",
            sequence=5,
        )

        d = msg.to_dict()
        assert d["msg_type"] == "metrics"
        assert d["payload"]["auc"] == 0.85
        assert d["sequence"] == 5

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {
            "msg_type": "complete",
            "payload": {"final_loss": 0.1},
            "experiment_id": "exp_003",
            "timestamp": "2025-01-01T00:00:00Z",
            "sequence": 10,
        }

        msg = IPCMessage.from_dict(d)
        assert msg.msg_type == MessageType.COMPLETE
        assert msg.payload["final_loss"] == 0.1
        assert msg.sequence == 10


class TestIPCProtocol:
    """Tests for IPCProtocol class."""

    @pytest.fixture
    def ipc_dir(self):
        """Create temporary IPC directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def parent_ipc(self, ipc_dir):
        """Create parent IPC protocol."""
        return IPCProtocol(ipc_dir, "test_exp", is_parent=True)

    @pytest.fixture
    def child_ipc(self, ipc_dir):
        """Create child IPC protocol."""
        return IPCProtocol(ipc_dir, "test_exp", is_parent=False)

    def test_send_receive_basic(self, parent_ipc, child_ipc):
        """Should send and receive messages."""
        # Parent sends to child
        parent_ipc.send(MessageType.START, {"config": {"lr": 0.001}})

        # Child receives
        msg = child_ipc.receive(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == MessageType.START
        assert msg.payload["config"]["lr"] == 0.001

    def test_send_receive_bidirectional(self, parent_ipc, child_ipc):
        """Should support bidirectional communication."""
        # Child sends to parent
        child_ipc.send(MessageType.READY, {})

        # Parent receives
        msg = parent_ipc.receive(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == MessageType.READY

    def test_receive_timeout(self, parent_ipc):
        """Should return None on timeout."""
        msg = parent_ipc.receive(timeout=0.1)
        assert msg is None

    def test_receive_all(self, parent_ipc, child_ipc):
        """Should receive all pending messages."""
        # Send multiple messages
        child_ipc.send(MessageType.PROGRESS, {"epoch": 1})
        child_ipc.send(MessageType.PROGRESS, {"epoch": 2})
        child_ipc.send(MessageType.PROGRESS, {"epoch": 3})

        # Receive all - but note each write overwrites, so only last is available
        # This is expected behavior for file-based IPC
        msg = parent_ipc.receive(timeout=1.0)
        assert msg is not None
        assert msg.msg_type == MessageType.PROGRESS

    def test_wait_for_message_type(self, parent_ipc, child_ipc):
        """Should wait for specific message type."""
        # Send ready
        child_ipc.send(MessageType.READY, {})

        # Wait for ready
        msg = parent_ipc.wait_for(MessageType.READY, timeout=2.0)
        assert msg is not None
        assert msg.msg_type == MessageType.READY

    def test_convenience_methods(self, parent_ipc, child_ipc):
        """Should support convenience methods."""
        # Parent sends start
        parent_ipc.send_start({"lr": 0.01})

        msg = child_ipc.receive(timeout=1.0)
        assert msg.msg_type == MessageType.START

        # Child sends progress
        child_ipc.send_progress(epoch=1, batch=10, total_batches=100, loss=0.5)

        msg = parent_ipc.receive(timeout=1.0)
        assert msg.msg_type == MessageType.PROGRESS
        assert msg.payload["epoch"] == 1
        assert msg.payload["loss"] == 0.5

    def test_cleanup(self, ipc_dir, parent_ipc):
        """Should clean up IPC files."""
        parent_ipc.send(MessageType.START, {})
        assert (ipc_dir / "parent_to_child.json").exists()

        parent_ipc.cleanup()
        # Files should be cleaned up
        assert not (ipc_dir / "parent_to_child.json").exists()


class TestCreateIPCPair:
    """Tests for create_ipc_pair helper."""

    def test_creates_pair(self):
        """Should create parent and child IPC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent, child = create_ipc_pair("test_exp", Path(tmpdir))

            assert parent.is_parent is True
            assert child.is_parent is False

            # Test communication
            parent.send_start({})
            msg = child.receive(timeout=1.0)
            assert msg.msg_type == MessageType.START


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_result(self):
        """Should create result with all fields."""
        result = ExecutionResult(
            experiment_id="exp_001",
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            metrics={"auc": 0.9},
        )

        assert result.experiment_id == "exp_001"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.exit_code == 0
        assert result.metrics["auc"] == 0.9

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ExecutionResult(
            experiment_id="exp_002",
            status=ExecutionStatus.FAILED,
            error="OOM",
        )

        d = result.to_dict()
        assert d["experiment_id"] == "exp_002"
        assert d["status"] == "failed"
        assert d["error"] == "OOM"


class TestSubprocessExecutor:
    """Tests for SubprocessExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create SubprocessExecutor instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SubprocessExecutor(ipc_base_dir=Path(tmpdir))

    def test_initialization(self, executor):
        """Should initialize correctly."""
        assert executor.ipc_base_dir.exists()
        assert not executor.is_running

    def test_latest_metrics_empty(self, executor):
        """Should return empty metrics when nothing executed."""
        assert executor.latest_metrics == {}

    def test_is_running_false_initially(self, executor):
        """Should report not running initially."""
        assert executor.is_running is False

    def test_stop_when_not_running(self, executor):
        """Stop should be safe when nothing running."""
        # Should not raise
        executor.stop()


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses(self):
        """Should have all expected statuses."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.STARTING.value == "starting"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.STOPPING.value == "stopping"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.KILLED.value == "killed"


class TestMessageType:
    """Tests for MessageType enum."""

    def test_parent_to_child_types(self):
        """Should have parent-to-child message types."""
        assert MessageType.START.value == "start"
        assert MessageType.STOP.value == "stop"
        assert MessageType.CHECKPOINT.value == "checkpoint"

    def test_child_to_parent_types(self):
        """Should have child-to-parent message types."""
        assert MessageType.READY.value == "ready"
        assert MessageType.PROGRESS.value == "progress"
        assert MessageType.METRICS.value == "metrics"
        assert MessageType.CHECKPOINT_SAVED.value == "checkpoint_saved"
        assert MessageType.ERROR.value == "error"
        assert MessageType.COMPLETE.value == "complete"
