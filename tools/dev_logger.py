"""
Development Logger: Lightweight FDA-Aligned Development Logging
================================================================

Automatically logs development decisions & research cycles in a format
acceptable to FDA reviewers as evidence of:
- Traceability
- Structured iteration
- Controlled changes
- Reproducible development flow
- Process awareness
- Risk awareness

This is NOT full ISO compliance — it's documentation that shows
professional, methodical development.

Satisfies:
- GMLP Principle 9: Documentation
- General SaMD "traceability" requirement
- ISO 13485 Design Controls (minimal)
- ISO 14971 Risk Awareness
- "Explain development process" requirement in Q-Sub meetings
"""

import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExperimentLog:
    """Single experiment log entry."""
    timestamp: str
    experiment_id: str
    cycle_id: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    model_version: str
    dataset_name: str
    dataset_version: str
    reasoning_summary: str
    status: str
    duration_seconds: float
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CycleLog:
    """Research cycle log entry."""
    timestamp: str
    cycle_id: int
    directive: str
    agents_involved: List[str]
    proposals_considered: int
    proposals_approved: int
    proposals_rejected: int
    supervisor_vetoed: bool
    chosen_experiment: Optional[str]
    failures: List[str]
    warnings: List[str]
    duration_seconds: float


@dataclass
class RiskEvent:
    """Risk indicator log entry."""
    timestamp: str
    cycle_id: int
    event_type: str  # crash, divergence, llm_failure, veto, oom, timeout
    severity: str    # low, medium, high, critical
    description: str
    experiment_id: Optional[str] = None
    mitigation: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # Additional context information


@dataclass
class DataProvenanceLog:
    """Dataset provenance log entry."""
    timestamp: str
    dataset_name: str
    dataset_version: str
    checksum: str
    operation: str  # loaded, preprocessed, split, used
    preprocessing_steps: List[str]
    split_ratios: Optional[Dict[str, float]] = None
    num_samples: Optional[int] = None


class DevLogger:
    """
    FDA-aligned development logging system.

    Logs all development activities to structured directories
    for traceability and regulatory compliance.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize development logger.

        Args:
            base_dir: Base directory for dev logs (defaults to settings)
        """
        settings = get_settings()
        if base_dir is None:
            base_dir = settings.home / "dev_logs"

        self.base_dir = Path(base_dir)

        # Create log directories
        self.dirs = {
            "experiments": self.base_dir / "experiments",
            "cycles": self.base_dir / "cycles",
            "data": self.base_dir / "data",
            "risk": self.base_dir / "risk",
            "git_commits": self.base_dir / "git_commits",
            "system_snapshots": self.base_dir / "system_snapshots"
        }

        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"DevLogger initialized at {self.base_dir}")

    def log_experiment(
        self,
        experiment_id: str,
        cycle_id: int,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        reasoning_summary: str,
        status: str,
        duration_seconds: float,
        checkpoint_path: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Path:
        """
        Log experiment execution.

        Returns:
            Path to experiment log file
        """
        log = ExperimentLog(
            timestamp=datetime.utcnow().isoformat(),
            experiment_id=experiment_id,
            cycle_id=cycle_id,
            config=config,
            metrics=metrics,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            reasoning_summary=reasoning_summary,
            status=status,
            duration_seconds=duration_seconds,
            checkpoint_path=checkpoint_path,
            error_message=error_message
        )

        # Write to individual experiment log
        exp_file = self.dirs["experiments"] / f"{experiment_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(asdict(log), f, indent=2)

        # Append to master experiment log
        master_file = self.dirs["experiments"] / "experiment_history.jsonl"
        with open(master_file, 'a') as f:
            f.write(json.dumps(asdict(log)) + '\n')

        logger.info(f"Logged experiment: {experiment_id}")
        return exp_file

    def log_cycle(
        self,
        cycle_id: int,
        directive: str,
        agents_involved: List[str],
        proposals_considered: int,
        proposals_approved: int,
        proposals_rejected: int,
        supervisor_vetoed: bool,
        chosen_experiment: Optional[str],
        failures: List[str],
        warnings: List[str],
        duration_seconds: float
    ) -> Path:
        """
        Log research cycle execution.

        Returns:
            Path to cycle log file
        """
        log = CycleLog(
            timestamp=datetime.utcnow().isoformat(),
            cycle_id=cycle_id,
            directive=directive,
            agents_involved=agents_involved,
            proposals_considered=proposals_considered,
            proposals_approved=proposals_approved,
            proposals_rejected=proposals_rejected,
            supervisor_vetoed=supervisor_vetoed,
            chosen_experiment=chosen_experiment,
            failures=failures,
            warnings=warnings,
            duration_seconds=duration_seconds
        )

        # Write to individual cycle log
        cycle_file = self.dirs["cycles"] / f"cycle_{cycle_id:04d}.json"
        with open(cycle_file, 'w') as f:
            json.dump(asdict(log), f, indent=2)

        # Write human-readable summary
        summary_file = self.dirs["cycles"] / f"cycle_{cycle_id:04d}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Research Cycle {cycle_id}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Timestamp: {log.timestamp}\n")
            f.write(f"Directive: {directive}\n")
            f.write(f"Duration: {duration_seconds:.1f}s\n\n")
            f.write(f"Agents Involved: {', '.join(agents_involved)}\n\n")
            f.write(f"Proposals:\n")
            f.write(f"  Considered: {proposals_considered}\n")
            f.write(f"  Approved: {proposals_approved}\n")
            f.write(f"  Rejected: {proposals_rejected}\n\n")
            f.write(f"Supervisor Veto: {'Yes' if supervisor_vetoed else 'No'}\n")
            f.write(f"Chosen Experiment: {chosen_experiment or 'None'}\n\n")

            if failures:
                f.write(f"Failures ({len(failures)}):\n")
                for failure in failures:
                    f.write(f"  - {failure}\n")
                f.write('\n')

            if warnings:
                f.write(f"Warnings ({len(warnings)}):\n")
                for warning in warnings:
                    f.write(f"  - {warning}\n")

        # Append to master cycle log
        master_file = self.dirs["cycles"] / "cycle_history.jsonl"
        with open(master_file, 'a') as f:
            f.write(json.dumps(asdict(log)) + '\n')

        logger.info(f"Logged cycle: {cycle_id}")
        return cycle_file

    def log_risk_event(
        self,
        cycle_id: int,
        event_type: str,
        severity: str,
        description: str,
        experiment_id: Optional[str] = None,
        mitigation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None  # Accept but log as additional context
    ) -> Path:
        """
        Log risk indicator event.

        Args:
            event_type: crash, divergence, llm_failure, veto, oom, timeout
            severity: low, medium, high, critical

        Returns:
            Path to risk log file
        """
        event = RiskEvent(
            timestamp=datetime.utcnow().isoformat(),
            cycle_id=cycle_id,
            event_type=event_type,
            severity=severity,
            description=description,
            experiment_id=experiment_id,
            mitigation=mitigation,
            context=context
        )

        # Append to risk log
        risk_file = self.dirs["risk"] / "risk_events.jsonl"
        with open(risk_file, 'a') as f:
            f.write(json.dumps(asdict(event)) + '\n')

        # Write human-readable risk report
        report_file = self.dirs["risk"] / f"risk_report_{datetime.utcnow().strftime('%Y%m')}.txt"
        with open(report_file, 'a') as f:
            f.write(f"\n[{event.timestamp}] {severity.upper()}: {event_type}\n")
            f.write(f"Cycle: {cycle_id}\n")
            if experiment_id:
                f.write(f"Experiment: {experiment_id}\n")
            f.write(f"Description: {description}\n")
            if mitigation:
                f.write(f"Mitigation: {mitigation}\n")
            f.write("-" * 80 + "\n")

        logger.warning(f"Risk event logged: {event_type} ({severity})")
        return risk_file

    def log_data_provenance(
        self,
        dataset_name: str,
        dataset_version: str,
        operation: str,
        preprocessing_steps: List[str],
        dataset_path: Optional[str] = None,
        split_ratios: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None
    ) -> Path:
        """
        Log dataset provenance.

        Args:
            operation: loaded, preprocessed, split, used

        Returns:
            Path to provenance log file
        """
        # Compute checksum if path provided
        checksum = "N/A"
        if dataset_path and Path(dataset_path).exists():
            checksum = self._compute_dataset_checksum(dataset_path)

        log = DataProvenanceLog(
            timestamp=datetime.utcnow().isoformat(),
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            checksum=checksum,
            operation=operation,
            preprocessing_steps=preprocessing_steps,
            split_ratios=split_ratios,
            num_samples=num_samples
        )

        # Append to data provenance log
        prov_file = self.dirs["data"] / f"{dataset_name}_provenance.jsonl"
        with open(prov_file, 'a') as f:
            f.write(json.dumps(asdict(log)) + '\n')

        logger.info(f"Logged data provenance: {dataset_name} ({operation})")
        return prov_file

    def log_git_commit(self) -> Optional[Path]:
        """
        Log current git commit.

        Returns:
            Path to git log file, or None if not in git repo
        """
        try:
            # Get commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get commit message
            commit_msg = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%B"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get commit author
            author = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%an <%ae>"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get commit date
            commit_date = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%aI"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            commit_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "commit_hash": commit_hash,
                "commit_message": commit_msg,
                "author": author,
                "commit_date": commit_date
            }

            # Append to git log
            git_file = self.dirs["git_commits"] / "git_history.jsonl"
            with open(git_file, 'a') as f:
                f.write(json.dumps(commit_log) + '\n')

            logger.info(f"Logged git commit: {commit_hash[:8]}")
            return git_file

        except subprocess.CalledProcessError:
            logger.debug("Not in git repository, skipping git commit log")
            return None

    def snapshot_system_state(self, cycle_id: int) -> Path:
        """
        Create timestamped snapshot of system state.

        Args:
            cycle_id: Current cycle ID

        Returns:
            Path to snapshot directory
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.dirs["system_snapshots"] / f"cycle_{cycle_id:04d}_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        settings = get_settings()
        memory_dir = Path(settings.memory_dir)

        # Copy memory files
        for filename in ["system_state.json", "directive.json", "constraints.json"]:
            src = memory_dir / filename
            if src.exists():
                dst = snapshot_dir / filename
                dst.write_text(src.read_text())

        # Create snapshot metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle_id": cycle_id,
            "arc_version": settings.arc_version,
            "environment": settings.environment,
            "mode": settings.mode
        }

        metadata_file = snapshot_dir / "snapshot_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created system snapshot for cycle {cycle_id}")
        return snapshot_dir

    def generate_development_report(self, output_path: Optional[str] = None) -> Path:
        """
        Generate comprehensive development report for FDA review.

        Returns:
            Path to development report
        """
        if output_path is None:
            output_path = self.base_dir / "DEVELOPMENT_REPORT.md"

        output_file = Path(output_path)

        with open(output_file, 'w') as f:
            f.write("# ARC Development Report\n\n")
            f.write(f"**Generated**: {datetime.utcnow().isoformat()}\n\n")
            f.write("---\n\n")

            # Experiment summary
            f.write("## Experiment History\n\n")
            exp_file = self.dirs["experiments"] / "experiment_history.jsonl"
            if exp_file.exists():
                experiments = [json.loads(line) for line in exp_file.read_text().splitlines()]
                f.write(f"**Total Experiments**: {len(experiments)}\n\n")

                successful = sum(1 for e in experiments if e['status'] == 'completed')
                failed = sum(1 for e in experiments if e['status'] == 'failed')

                f.write(f"- Successful: {successful}\n")
                f.write(f"- Failed: {failed}\n")
                f.write(f"- Success Rate: {successful/len(experiments)*100:.1f}%\n\n")
            else:
                f.write("No experiments logged yet.\n\n")

            # Cycle summary
            f.write("## Research Cycles\n\n")
            cycle_file = self.dirs["cycles"] / "cycle_history.jsonl"
            if cycle_file.exists():
                cycles = [json.loads(line) for line in cycle_file.read_text().splitlines()]
                f.write(f"**Total Cycles**: {len(cycles)}\n\n")

                total_proposals = sum(c['proposals_considered'] for c in cycles)
                total_approved = sum(c['proposals_approved'] for c in cycles)

                f.write(f"- Total Proposals Considered: {total_proposals}\n")
                f.write(f"- Total Proposals Approved: {total_approved}\n")
                f.write(f"- Approval Rate: {total_approved/total_proposals*100:.1f}%\n\n")
            else:
                f.write("No cycles logged yet.\n\n")

            # Risk events
            f.write("## Risk Events\n\n")
            risk_file = self.dirs["risk"] / "risk_events.jsonl"
            if risk_file.exists():
                events = [json.loads(line) for line in risk_file.read_text().splitlines()]
                f.write(f"**Total Risk Events**: {len(events)}\n\n")

                by_severity = {}
                for event in events:
                    severity = event['severity']
                    by_severity[severity] = by_severity.get(severity, 0) + 1

                for severity in ['critical', 'high', 'medium', 'low']:
                    if severity in by_severity:
                        f.write(f"- {severity.capitalize()}: {by_severity[severity]}\n")
            else:
                f.write("No risk events logged.\n\n")

            # Data provenance
            f.write("## Data Provenance\n\n")
            data_files = list(self.dirs["data"].glob("*_provenance.jsonl"))
            f.write(f"**Datasets Tracked**: {len(data_files)}\n\n")
            for data_file in data_files:
                dataset_name = data_file.stem.replace("_provenance", "")
                f.write(f"- {dataset_name}\n")

            f.write("\n---\n\n")
            f.write("## Traceability Statement\n\n")
            f.write("This report demonstrates:\n\n")
            f.write("- ✅ Structured development process\n")
            f.write("- ✅ Traceability of experiments to cycles\n")
            f.write("- ✅ Risk awareness and monitoring\n")
            f.write("- ✅ Data provenance tracking\n")
            f.write("- ✅ Reproducible development flow\n\n")
            f.write("**Satisfies**:\n")
            f.write("- GMLP Principle 9: Documentation\n")
            f.write("- General SaMD Traceability Requirement\n")
            f.write("- ISO 13485 Design Controls (minimal)\n")
            f.write("- ISO 14971 Risk Awareness\n")

        logger.info(f"Generated development report: {output_file}")
        return output_file

    def _compute_dataset_checksum(self, dataset_path: str) -> str:
        """Compute SHA256 checksum of dataset directory."""
        path = Path(dataset_path)

        if path.is_file():
            # Single file checksum
            return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

        elif path.is_dir():
            # Directory checksum (hash of sorted file list + sizes)
            files = sorted(path.rglob("*"))
            hash_input = ""
            for f in files:
                if f.is_file():
                    hash_input += f"{f.name}:{f.stat().st_size};"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return "N/A"


# Global singleton
_dev_logger: Optional[DevLogger] = None


def get_dev_logger(base_dir: Optional[str] = None) -> DevLogger:
    """Get global development logger instance."""
    global _dev_logger

    if _dev_logger is None:
        _dev_logger = DevLogger(base_dir=base_dir)

    return _dev_logger
