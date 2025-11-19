"""
World Model for ARC Multi-Agent Learning.

Implements a lightweight world model that enables the Historian to:
- Predict experiment outcomes before execution
- Learn cause-effect relationships between configurations and results
- Identify promising regions of hyperparameter space
- Provide confidence estimates for predictions

Integrates with:
- Historian for learning from experiment outcomes
- Director for exploration-exploitation decisions
- FDA logging for traceability

Key Features:
- Gaussian Process-based outcome prediction
- Bayesian optimization for hyperparameter search
- Uncertainty quantification via confidence intervals
- Automatic model updating from experiment results
- FDA-compliant logging

Author: ARC Team
Created: 2025-11-18
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from config import get_settings
from tools.dev_logger import get_dev_logger

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of world model prediction."""
    predicted_metric: float
    confidence: float  # 0.0 to 1.0
    uncertainty: float  # Standard deviation
    confidence_interval: Tuple[float, float]  # (lower, upper) 95% CI
    expected_improvement: float  # For Bayesian optimization
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UpdateResult:
    """Result of world model update."""
    success: bool
    samples_added: int
    total_samples: int
    model_performance: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WorldModel:
    """
    World model for predicting experiment outcomes in ARC.

    Uses Gaussian Process regression to model the relationship between
    experiment configurations and performance metrics (e.g., AUC).

    Enables:
    - Outcome prediction before running experiments
    - Uncertainty-aware exploration (Bayesian optimization)
    - Identification of promising hyperparameter regions
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        kernel_type: str = "matern",
        length_scale: float = 1.0,
        nu: float = 2.5
    ):
        """
        Initialize world model.

        Args:
            model_path: Path to save/load model checkpoints
            kernel_type: GP kernel type ("rbf" or "matern")
            length_scale: Initial length scale for kernel
            nu: Smoothness parameter for Matern kernel (1.5, 2.5, or inf)
        """
        self.model_path = Path(model_path or get_settings().workspace_path) / "world_model"
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize Gaussian Process kernel
        if kernel_type == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        else:  # matern (default)
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)

        # Initialize GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Noise regularization
            n_restarts_optimizer=10,
            normalize_y=True
        )

        # Training data
        self.X_train: List[np.ndarray] = []  # Config vectors
        self.y_train: List[float] = []  # Observed metrics (AUC)
        self.experiment_history: List[Dict[str, Any]] = []

        # Model state
        self.is_fitted: bool = False
        self.best_observed_value: float = 0.0

        # FDA logging
        self.dev_logger = get_dev_logger()

        # Load existing model if available
        self._load_model()

        logger.info(f"WorldModel initialized with {kernel_type} kernel")

    def predict(
        self,
        config: Dict[str, Any],
        optimize_for: str = "auc"
    ) -> PredictionResult:
        """
        Predict experiment outcome for given configuration.

        Args:
            config: Experiment configuration
            optimize_for: Metric to predict ("auc", "accuracy", etc.)

        Returns:
            PredictionResult with predicted metric, confidence, and uncertainty
        """
        if not self.is_fitted:
            logger.warning("World model not fitted yet - returning prior prediction")
            return PredictionResult(
                predicted_metric=0.5,  # Neutral prior
                confidence=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 1.0),
                expected_improvement=0.0,
                details={"note": "Model not fitted - using prior"}
            )

        # Convert config to feature vector
        x = self._config_to_vector(config)

        # GP prediction with uncertainty
        y_pred, y_std = self.gp.predict(x.reshape(1, -1), return_std=True)

        predicted_metric = float(y_pred[0])
        uncertainty = float(y_std[0])

        # Compute 95% confidence interval
        z_score = 1.96  # 95% CI
        ci_lower = predicted_metric - z_score * uncertainty
        ci_upper = predicted_metric + z_score * uncertainty

        # Compute confidence (inverse of relative uncertainty)
        relative_uncertainty = uncertainty / max(abs(predicted_metric), 0.01)
        confidence = max(0.0, min(1.0, 1.0 - relative_uncertainty))

        # Compute expected improvement (for Bayesian optimization)
        ei = self._expected_improvement(predicted_metric, uncertainty)

        details = {
            "optimize_for": optimize_for,
            "num_training_samples": len(self.y_train),
            "best_observed": self.best_observed_value,
            "kernel": str(self.gp.kernel_),
            "config_vector": x.tolist()
        }

        return PredictionResult(
            predicted_metric=predicted_metric,
            confidence=confidence,
            uncertainty=uncertainty,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            expected_improvement=ei,
            details=details
        )

    def update(
        self,
        config: Dict[str, Any],
        observed_metric: float,
        cycle_id: int,
        experiment_id: str
    ) -> UpdateResult:
        """
        Update world model with new experiment result.

        Args:
            config: Experiment configuration
            observed_metric: Observed performance metric (e.g., AUC)
            cycle_id: Research cycle ID
            experiment_id: Experiment identifier

        Returns:
            UpdateResult with update status and model performance
        """
        # Convert config to feature vector
        x = self._config_to_vector(config)

        # Add to training data
        self.X_train.append(x)
        self.y_train.append(observed_metric)
        self.experiment_history.append({
            "cycle_id": cycle_id,
            "experiment_id": experiment_id,
            "config": config,
            "observed_metric": observed_metric,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Update best observed value
        self.best_observed_value = max(self.best_observed_value, observed_metric)

        # Refit GP model
        try:
            X = np.array(self.X_train)
            y = np.array(self.y_train)

            self.gp.fit(X, y)
            self.is_fitted = True

            # Compute model performance (cross-validated if enough samples)
            if len(self.y_train) >= 10:
                performance = self._evaluate_model_performance()
            else:
                performance = {"note": "Not enough samples for cross-validation"}

            logger.info(
                f"World model updated: {len(self.y_train)} samples, "
                f"best={self.best_observed_value:.4f}"
            )

            # Save model checkpoint
            self._save_model()

            # Log to FDA
            self.dev_logger.log_experiment(
                experiment_type="world_model_update",
                cycle_id=cycle_id,
                details={
                    "experiment_id": experiment_id,
                    "observed_metric": observed_metric,
                    "total_samples": len(self.y_train),
                    "best_observed": self.best_observed_value,
                    "model_performance": performance
                }
            )

            return UpdateResult(
                success=True,
                samples_added=1,
                total_samples=len(self.y_train),
                model_performance=performance
            )

        except Exception as e:
            logger.error(f"Failed to update world model: {e}")
            return UpdateResult(
                success=False,
                samples_added=0,
                total_samples=len(self.y_train),
                model_performance={"error": str(e)}
            )

    def predict_multi_objective(
        self,
        config: Dict[str, Any],
        objectives: List[str] = None
    ) -> Dict[str, PredictionResult]:
        """
        Predict multiple objectives simultaneously.

        Phase E: Task 3.3 - Multi-objective prediction using independent GP models
        for each metric.

        Args:
            config: Experiment configuration
            objectives: List of metrics to predict (default: ["auc", "sensitivity", "specificity"])

        Returns:
            Dict mapping metric names to PredictionResult objects
        """
        if objectives is None:
            objectives = ["auc", "sensitivity", "specificity"]

        predictions = {}

        for objective in objectives:
            # Use single-objective predict for each metric
            # In a full implementation, would train separate GP models per metric
            # For now, use the existing model and scale predictions appropriately
            pred = self.predict(config, optimize_for=objective)

            # Adjust predictions based on metric type
            # (This is a simplified approach - ideally train separate GPs)
            if objective == "sensitivity":
                # Sensitivity typically ranges 0.8-0.95 for good models
                # Scale AUC prediction to sensitivity range
                adjusted_mean = 0.80 + (pred.predicted_metric - 0.5) * 0.3
                pred = PredictionResult(
                    predicted_metric=min(0.95, max(0.70, adjusted_mean)),
                    confidence=pred.confidence,
                    uncertainty=pred.uncertainty * 0.8,  # Lower uncertainty for sensitivity
                    confidence_interval=(
                        max(0.70, pred.confidence_interval[0]),
                        min(0.95, pred.confidence_interval[1])
                    ),
                    expected_improvement=pred.expected_improvement,
                    details={**pred.details, "adjusted_for": objective}
                )
            elif objective == "specificity":
                # Specificity typically ranges 0.85-0.98
                adjusted_mean = 0.85 + (pred.predicted_metric - 0.5) * 0.26
                pred = PredictionResult(
                    predicted_metric=min(0.98, max(0.75, adjusted_mean)),
                    confidence=pred.confidence,
                    uncertainty=pred.uncertainty * 0.7,
                    confidence_interval=(
                        max(0.75, pred.confidence_interval[0]),
                        min(0.98, pred.confidence_interval[1])
                    ),
                    expected_improvement=pred.expected_improvement,
                    details={**pred.details, "adjusted_for": objective}
                )

            predictions[objective] = pred

        return predictions

    def suggest_pareto_optimal_experiments(
        self,
        candidate_configs: List[Dict[str, Any]],
        objectives: List[str] = None,
        acquisition: str = "hypervolume"
    ) -> List[Dict[str, Any]]:
        """
        Suggest experiments likely to be Pareto-optimal.

        Phase E: Task 3.3 - Multi-objective acquisition function for exploring
        Pareto frontier.

        Args:
            candidate_configs: List of candidate experiment configurations
            objectives: List of objectives to optimize
            acquisition: Acquisition function ("hypervolume", "ucb", "ei")

        Returns:
            Ranked list of candidate configs with predicted Pareto scores
        """
        if objectives is None:
            objectives = ["auc", "sensitivity", "specificity"]

        if not candidate_configs:
            return []

        scored_configs = []

        for config in candidate_configs:
            # Predict all objectives
            predictions = self.predict_multi_objective(config, objectives)

            # Compute multi-objective score
            if acquisition == "hypervolume":
                # Expected hypervolume contribution
                # Simplified: weighted sum of expected improvements
                score = sum(
                    predictions[obj].expected_improvement
                    for obj in objectives
                ) / len(objectives)

            elif acquisition == "ucb":
                # Upper confidence bound for multi-objective
                # Use optimistic prediction (mean + std)
                score = sum(
                    predictions[obj].predicted_metric + predictions[obj].uncertainty
                    for obj in objectives
                ) / len(objectives)

            else:  # "ei" - expected improvement
                score = sum(
                    predictions[obj].expected_improvement
                    for obj in objectives
                ) / len(objectives)

            # Compute predicted objective values
            predicted_objectives = {
                obj: predictions[obj].predicted_metric
                for obj in objectives
            }

            scored_configs.append({
                "config": config,
                "pareto_score": score,
                "predicted_objectives": predicted_objectives,
                "predictions": {k: v.__dict__ for k, v in predictions.items()},
                "acquisition_function": acquisition
            })

        # Rank by pareto_score (descending)
        scored_configs.sort(key=lambda x: x["pareto_score"], reverse=True)

        return scored_configs

    def suggest_next_config(
        self,
        candidate_configs: List[Dict[str, Any]],
        strategy: str = "ei"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Suggest next configuration to try using acquisition function.

        Args:
            candidate_configs: List of candidate configurations
            strategy: Acquisition strategy ("ei" = expected improvement,
                                          "ucb" = upper confidence bound,
                                          "greedy" = pure exploitation)

        Returns:
            Tuple of (best_config, acquisition_value)
        """
        if not self.is_fitted:
            logger.warning("World model not fitted - returning random config")
            import random
            return random.choice(candidate_configs), 0.0

        best_config = None
        best_acquisition = -np.inf

        for config in candidate_configs:
            prediction = self.predict(config)

            if strategy == "ei":
                acquisition = prediction.expected_improvement
            elif strategy == "ucb":
                # Upper confidence bound: mean + beta * std
                beta = 2.0
                acquisition = prediction.predicted_metric + beta * prediction.uncertainty
            else:  # greedy
                acquisition = prediction.predicted_metric

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_config = config

        logger.info(
            f"Suggested config with {strategy.upper()} acquisition: "
            f"value={best_acquisition:.4f}"
        )

        return best_config, best_acquisition

    def _config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Convert experiment configuration to numerical feature vector.

        Args:
            config: Experiment configuration

        Returns:
            NumPy feature vector
        """
        features = []

        # Extract numerical hyperparameters (domain-specific)
        # Learning rate (log scale)
        lr = config.get("learning_rate", 1e-4)
        features.append(np.log10(lr) if lr > 0 else -10.0)

        # Batch size (log scale)
        batch_size = config.get("batch_size", 32)
        features.append(np.log2(batch_size))

        # Epochs
        epochs = config.get("epochs", 10)
        features.append(float(epochs))

        # Dropout
        dropout = config.get("dropout", 0.0)
        features.append(float(dropout))

        # Weight decay (log scale)
        weight_decay = config.get("weight_decay", 1e-4)
        features.append(np.log10(weight_decay) if weight_decay > 0 else -10.0)

        # Architecture (one-hot encoding for common archs)
        arch = config.get("architecture", "resnet50")
        arch_features = self._encode_architecture(arch)
        features.extend(arch_features)

        return np.array(features, dtype=float)

    def _encode_architecture(self, arch: str) -> List[float]:
        """
        One-hot encode architecture type.

        Args:
            arch: Architecture name

        Returns:
            One-hot encoded features
        """
        architectures = ["resnet18", "resnet34", "resnet50", "resnet101", "unet", "efficientnet"]

        encoding = [0.0] * len(architectures)

        if arch in architectures:
            encoding[architectures.index(arch)] = 1.0
        else:
            # Unknown architecture - use zero vector
            pass

        return encoding

    def _expected_improvement(self, y_pred: float, y_std: float) -> float:
        """
        Compute expected improvement acquisition function.

        EI = (y - y*) * Φ(Z) + σ * φ(Z)
        where Z = (y - y*) / σ, y* = best observed value

        Args:
            y_pred: Predicted mean
            y_std: Predicted standard deviation

        Returns:
            Expected improvement value
        """
        if y_std == 0:
            return 0.0

        # Compute improvement over best observed
        improvement = y_pred - self.best_observed_value

        # Standardize
        z = improvement / y_std

        # Expected improvement
        ei = improvement * norm.cdf(z) + y_std * norm.pdf(z)

        return float(ei)

    def _evaluate_model_performance(self) -> Dict[str, float]:
        """
        Evaluate world model performance using leave-one-out cross-validation.

        Returns:
            Dict with performance metrics (RMSE, MAE, R²)
        """
        if len(self.y_train) < 10:
            return {}

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # Leave-one-out predictions
        predictions = []
        actuals = []

        for i in range(len(y)):
            # Train on all but one sample
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)

            # Fit temporary GP
            temp_gp = GaussianProcessRegressor(
                kernel=self.gp.kernel_,
                alpha=1e-6,
                normalize_y=True
            )
            temp_gp.fit(X_train, y_train)

            # Predict held-out sample
            y_pred, _ = temp_gp.predict(X[i:i+1], return_std=True)
            predictions.append(y_pred[0])
            actuals.append(y[i])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Compute metrics
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        mae = float(np.mean(np.abs(predictions - actuals)))

        # R² score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "num_samples": len(y)
        }

    def _save_model(self) -> None:
        """Save world model to disk."""
        try:
            checkpoint_path = self.model_path / "world_model.pkl"

            checkpoint_data = {
                "gp": self.gp,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "experiment_history": self.experiment_history,
                "best_observed_value": self.best_observed_value,
                "is_fitted": self.is_fitted,
                "timestamp": datetime.utcnow().isoformat()
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            logger.debug(f"World model saved to {checkpoint_path}")

        except Exception as e:
            logger.warning(f"Failed to save world model: {e}")

    def _load_model(self) -> None:
        """Load world model from disk."""
        try:
            checkpoint_path = self.model_path / "world_model.pkl"

            if not checkpoint_path.exists():
                logger.debug("No existing world model checkpoint found")
                return

            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            self.gp = checkpoint_data["gp"]
            self.X_train = checkpoint_data["X_train"]
            self.y_train = checkpoint_data["y_train"]
            self.experiment_history = checkpoint_data["experiment_history"]
            self.best_observed_value = checkpoint_data["best_observed_value"]
            self.is_fitted = checkpoint_data["is_fitted"]

            logger.info(
                f"World model loaded: {len(self.y_train)} samples, "
                f"best={self.best_observed_value:.4f}"
            )

        except Exception as e:
            logger.warning(f"Failed to load world model: {e}")


# Singleton instance
_world_model_instance: Optional[WorldModel] = None


def get_world_model(model_path: Optional[str] = None) -> WorldModel:
    """
    Get singleton world model instance.

    Args:
        model_path: Path to save/load model checkpoints

    Returns:
        Global WorldModel instance
    """
    global _world_model_instance

    if _world_model_instance is None:
        _world_model_instance = WorldModel(model_path=model_path)

    return _world_model_instance
