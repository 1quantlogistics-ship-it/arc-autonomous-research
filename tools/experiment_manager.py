"""
Experiment Manager: Centralized experiment directory structure and artifact management

Creates and manages standardized experiment directories with all artifacts:
- Hydra configs
- Training logs
- Checkpoints
- Evaluation results
- Visualizations (CAMs, DRIs)
- Metrics history

Standard structure:
/workspace/arc/experiments/<experiment_id>/
    config/
        hydra_config.yaml
        experiment_spec.json
    logs/
        train.log
        eval.log
        stdout.log
        stderr.log
    checkpoints/
        best_model.pt
        final_model.pt
        epoch_010.pt
    results/
        metrics.json
        training_history.json
        test_results.json
    visualizations/
        cam/
        gradcam++/
        dri/
    metadata.json

This enables:
- Historian ingestion
- Dashboard display
- ARC memory access
- Reproducibility
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config import get_settings

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages experiment directory structure and artifacts.

    Provides standardized organization for all experiment outputs.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for experiments (default: workspace/arc/experiments)
        """
        settings = get_settings()

        if base_dir is None:
            base_dir = str(settings.home / "workspace" / "arc" / "experiments")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExperimentManager initialized: {self.base_dir}")

    def create_experiment_dir(
        self,
        experiment_id: str,
        task_type: str,
        cycle_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Create standardized experiment directory structure.

        Args:
            experiment_id: Experiment identifier
            task_type: Task type (segmentation or classification)
            cycle_id: Research cycle ID
            metadata: Optional experiment metadata

        Returns:
            Dict with all directory paths
        """
        logger.info(f"Creating experiment directory: {experiment_id}")

        # Root experiment directory
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        dirs = {
            "root": str(exp_dir),
            "config": str(exp_dir / "config"),
            "logs": str(exp_dir / "logs"),
            "checkpoints": str(exp_dir / "checkpoints"),
            "results": str(exp_dir / "results"),
            "visualizations": str(exp_dir / "visualizations"),
            "visualizations_cam": str(exp_dir / "visualizations" / "cam"),
            "visualizations_gradcam_pp": str(exp_dir / "visualizations" / "gradcam++"),
            "visualizations_dri": str(exp_dir / "visualizations" / "dri")
        }

        # Create all subdirectories
        for dir_path in dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata_file = exp_dir / "metadata.json"
        metadata_content = {
            "experiment_id": experiment_id,
            "task_type": task_type,
            "cycle_id": cycle_id,
            "created_at": datetime.utcnow().isoformat(),
            "directory_structure": dirs,
            **(metadata or {})
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata_content, f, indent=2)

        logger.info(f"Created experiment directory: {exp_dir}")

        return dirs

    def get_experiment_dir(self, experiment_id: str) -> Optional[Dict[str, str]]:
        """
        Get experiment directory paths if it exists.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dict with directory paths or None if not found
        """
        exp_dir = self.base_dir / experiment_id

        if not exp_dir.exists():
            return None

        # Load metadata
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get("directory_structure")

        # Fallback: reconstruct paths
        return {
            "root": str(exp_dir),
            "config": str(exp_dir / "config"),
            "logs": str(exp_dir / "logs"),
            "checkpoints": str(exp_dir / "checkpoints"),
            "results": str(exp_dir / "results"),
            "visualizations": str(exp_dir / "visualizations")
        }

    def save_config(
        self,
        experiment_id: str,
        config_name: str,
        config_data: Dict[str, Any]
    ):
        """
        Save configuration file.

        Args:
            experiment_id: Experiment identifier
            config_name: Config filename (e.g., 'hydra_config.yaml')
            config_data: Configuration data
        """
        exp_dir = self.base_dir / experiment_id / "config"
        config_file = exp_dir / config_name

        # Determine format from extension
        if config_name.endswith('.json'):
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        elif config_name.endswith('.yaml') or config_name.endswith('.yml'):
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
        else:
            raise ValueError(f"Unsupported config format: {config_name}")

        logger.info(f"Saved config: {config_file}")

    def save_results(
        self,
        experiment_id: str,
        results_name: str,
        results_data: Dict[str, Any]
    ):
        """
        Save results file.

        Args:
            experiment_id: Experiment identifier
            results_name: Results filename (e.g., 'metrics.json')
            results_data: Results data
        """
        exp_dir = self.base_dir / experiment_id / "results"
        results_file = exp_dir / results_name

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved results: {results_file}")

    def save_checkpoint(
        self,
        experiment_id: str,
        checkpoint_name: str,
        source_path: str
    ):
        """
        Copy checkpoint to experiment directory.

        Args:
            experiment_id: Experiment identifier
            checkpoint_name: Checkpoint filename (e.g., 'best_model.pt')
            source_path: Source checkpoint path
        """
        exp_dir = self.base_dir / experiment_id / "checkpoints"
        dest_path = exp_dir / checkpoint_name

        shutil.copy2(source_path, dest_path)

        logger.info(f"Saved checkpoint: {dest_path}")

    def list_experiments(
        self,
        cycle_id: Optional[int] = None,
        task_type: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """
        List all experiments, optionally filtered.

        Args:
            cycle_id: Filter by cycle ID
            task_type: Filter by task type

        Returns:
            List of experiment metadata
        """
        experiments = []

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            metadata_file = exp_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Apply filters
                if cycle_id is not None and metadata.get("cycle_id") != cycle_id:
                    continue

                if task_type is not None and metadata.get("task_type") != task_type:
                    continue

                experiments.append(metadata)

            except Exception as e:
                logger.error(f"Failed to load metadata for {exp_dir}: {e}")

        # Sort by created_at descending
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return experiments

    def get_experiment_metadata(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment metadata.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Metadata dict or None if not found
        """
        metadata_file = self.base_dir / experiment_id / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete experiment directory and all artifacts.

        Args:
            experiment_id: Experiment to delete

        Returns:
            True if deleted successfully
        """
        logger.warning(f"Deleting experiment: {experiment_id}")

        exp_dir = self.base_dir / experiment_id

        if not exp_dir.exists():
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        try:
            shutil.rmtree(exp_dir)
            logger.info(f"Deleted experiment: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}")
            return False

    def archive_experiment(self, experiment_id: str, archive_dir: Optional[str] = None) -> bool:
        """
        Archive experiment to separate location.

        Args:
            experiment_id: Experiment to archive
            archive_dir: Archive directory (default: workspace/arc/archive)

        Returns:
            True if archived successfully
        """
        logger.info(f"Archiving experiment: {experiment_id}")

        exp_dir = self.base_dir / experiment_id

        if not exp_dir.exists():
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        # Default archive directory
        if archive_dir is None:
            settings = get_settings()
            archive_dir = str(settings.home / "workspace" / "arc" / "archive")

        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)

        # Move experiment
        dest_path = archive_path / experiment_id

        try:
            shutil.move(str(exp_dir), str(dest_path))
            logger.info(f"Archived experiment to: {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to archive experiment: {e}")
            return False


# Global singleton instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager(base_dir: Optional[str] = None) -> ExperimentManager:
    """
    Get global experiment manager instance (singleton).

    Args:
        base_dir: Base directory for experiments

    Returns:
        ExperimentManager instance
    """
    global _experiment_manager

    if _experiment_manager is None:
        _experiment_manager = ExperimentManager(base_dir=base_dir)

    return _experiment_manager
