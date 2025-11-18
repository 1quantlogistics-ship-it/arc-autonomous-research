"""
World-Model: Predictive Intelligence for Experiment Outcomes
============================================================

Uses Gaussian Process surrogate modeling to predict experiment outcomes
before execution, enabling intelligent exploration and reducing wasted experiments.

Key capabilities:
- Train on {config} → {metrics} history
- Predict outcomes with uncertainty estimates
- Suggest next experiments via acquisition functions
- Guide exploration intelligently
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import get_settings

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. World-model will use simplified predictions.")


@dataclass
class Prediction:
    """Prediction with uncertainty."""
    mean: float
    std: float
    confidence: float

    def __repr__(self):
        return f"{self.mean:.3f} ± {self.std:.3f} (conf: {self.confidence:.2%})"


class WorldModel:
    """
    Gaussian Process surrogate model for experiment outcome prediction.

    Learns from training history to predict:
    - Expected metrics (AUC, sensitivity, specificity)
    - Uncertainty estimates
    - Acquisition function values for smart exploration

    This enables:
    - Filtering bad proposals before execution
    - Focusing exploration on promising regions
    - Reducing wasted experiments
    """

    def __init__(
        self,
        memory_path: Optional[str] = None,
        target_metric: str = "auc",
        kernel_type: str = "matern"
    ):
        """
        Initialize world-model.

        Args:
            memory_path: Path to memory directory (defaults to settings)
            target_metric: Primary metric to predict (auc, sensitivity, specificity)
            kernel_type: GP kernel type (rbf, matern)
        """
        settings = get_settings()
        self.memory_path = Path(memory_path or settings.memory_dir)
        self.target_metric = target_metric
        self.kernel_type = kernel_type

        # Model components
        self.gp_model = None
        self.scaler_X = StandardScaler() if SKLEARN_AVAILABLE else None
        self.scaler_y = StandardScaler() if SKLEARN_AVAILABLE else None

        # Training data
        self.X_train = None  # Configs as feature vectors
        self.y_train = None  # Metrics as targets
        self.feature_names = []
        self.is_trained = False

        # Simplified model (fallback when sklearn unavailable)
        self.simple_history = []
        self.simple_baseline = 0.5

    def train_on_history(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train GP model on training history.

        Args:
            force_retrain: Force retraining even if model exists

        Returns:
            Training summary with model performance
        """
        # Load training history
        history_path = self.memory_path / "training_history.json"
        if not history_path.exists():
            return {
                "status": "no_history",
                "message": "No training history available yet"
            }

        with open(history_path, 'r') as f:
            history = json.load(f)

        experiments = history.get("experiments", [])
        if len(experiments) < 3:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 3 experiments, have {len(experiments)}"
            }

        # Extract successful experiments with metrics
        successful = [
            exp for exp in experiments
            if exp.get("status") == "completed" and exp.get("metrics", {}).get(self.target_metric) is not None
        ]

        if len(successful) < 3:
            return {
                "status": "insufficient_successful",
                "message": f"Need at least 3 successful experiments, have {len(successful)}"
            }

        # Convert configs to feature vectors
        X, y = self._prepare_training_data(successful)

        if X is None or len(X) < 3:
            return {
                "status": "feature_extraction_failed",
                "message": "Could not extract features from experiments"
            }

        if not SKLEARN_AVAILABLE:
            # Simple fallback: just store history
            self.simple_history = successful
            self.simple_baseline = np.mean(y)
            self.is_trained = True
            return {
                "status": "trained_simple",
                "n_experiments": len(successful),
                "baseline_metric": float(self.simple_baseline),
                "message": "Using simple baseline predictor (sklearn unavailable)"
            }

        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Create GP model
        if self.kernel_type == "matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:  # rbf
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )

        # Train model
        self.gp_model.fit(X_scaled, y_scaled)

        self.X_train = X_scaled
        self.y_train = y_scaled
        self.is_trained = True

        # Compute training metrics
        y_pred, y_std = self.gp_model.predict(X_scaled, return_std=True)
        mse = np.mean((y_scaled - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return {
            "status": "trained_gp",
            "n_experiments": len(successful),
            "n_features": X.shape[1],
            "kernel": str(self.gp_model.kernel_),
            "training_rmse": float(rmse),
            "model_type": "GaussianProcess",
            "message": f"GP model trained on {len(successful)} experiments"
        }

    def _prepare_training_data(
        self,
        experiments: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert experiment configs to feature vectors.

        Args:
            experiments: List of experiment dicts with configs and metrics

        Returns:
            (X, y) where X is features, y is target metric
        """
        # Extract features from configs
        features = []
        targets = []

        for exp in experiments:
            config = exp.get("config", {})
            metrics = exp.get("metrics", {})

            # Extract numeric features
            feature_dict = self._config_to_features(config)
            if feature_dict is None:
                continue

            features.append(feature_dict)
            targets.append(metrics.get(self.target_metric, 0.0))

        if not features:
            return None, None

        # Convert to numpy arrays
        if not self.feature_names:
            self.feature_names = list(features[0].keys())

        X = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in features])
        y = np.array(targets)

        return X, y

    def _config_to_features(self, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Convert config dict to feature vector.

        Args:
            config: Experiment config

        Returns:
            Dict of feature_name → value
        """
        features = {}

        # Numeric features
        features["learning_rate"] = float(config.get("learning_rate", 0.0001))
        features["batch_size"] = float(config.get("batch_size", 8))
        features["epochs"] = float(config.get("epochs", 10))
        features["dropout"] = float(config.get("dropout", 0.2))
        features["weight_decay"] = float(config.get("weight_decay", 0.0001))
        features["input_size"] = float(config.get("input_size", 512))

        # Categorical features (one-hot encoding)
        model = config.get("model", "efficientnet_b3")
        features["model_efficientnet_b0"] = 1.0 if model == "efficientnet_b0" else 0.0
        features["model_efficientnet_b3"] = 1.0 if model == "efficientnet_b3" else 0.0
        features["model_efficientnet_b5"] = 1.0 if model == "efficientnet_b5" else 0.0
        features["model_resnet50"] = 1.0 if model == "resnet50" else 0.0

        optimizer = config.get("optimizer", "adam")
        features["optimizer_adam"] = 1.0 if optimizer == "adam" else 0.0
        features["optimizer_adamw"] = 1.0 if optimizer == "adamw" else 0.0
        features["optimizer_sgd"] = 1.0 if optimizer == "sgd" else 0.0

        loss = config.get("loss", "focal")
        features["loss_focal"] = 1.0 if loss == "focal" else 0.0
        features["loss_cross_entropy"] = 1.0 if loss == "cross_entropy" else 0.0

        return features

    def predict(self, config: Dict[str, Any]) -> Prediction:
        """
        Predict experiment outcome for a given config.

        Args:
            config: Experiment configuration

        Returns:
            Prediction with mean, std, confidence
        """
        if not self.is_trained:
            # Not trained yet, return baseline
            return Prediction(mean=0.5, std=0.3, confidence=0.1)

        # Convert config to features
        feature_dict = self._config_to_features(config)
        if feature_dict is None:
            return Prediction(mean=0.5, std=0.3, confidence=0.1)

        X = np.array([[feature_dict.get(name, 0.0) for name in self.feature_names]])

        if not SKLEARN_AVAILABLE:
            # Simple fallback: return baseline with high uncertainty
            return Prediction(
                mean=float(self.simple_baseline),
                std=0.2,
                confidence=0.3
            )

        # Scale features
        X_scaled = self.scaler_X.transform(X)

        # Predict
        y_pred_scaled, y_std_scaled = self.gp_model.predict(X_scaled, return_std=True)

        # Unscale predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        y_std = y_std_scaled[0] * self.scaler_y.scale_[0]

        # Compute confidence (inverse of normalized std)
        confidence = 1.0 / (1.0 + y_std)

        return Prediction(
            mean=float(y_pred),
            std=float(y_std),
            confidence=float(confidence)
        )

    def predict_batch(self, configs: List[Dict[str, Any]]) -> List[Prediction]:
        """Predict outcomes for multiple configs."""
        return [self.predict(config) for config in configs]

    def suggest_next_experiments(
        self,
        candidate_configs: List[Dict[str, Any]],
        n_suggestions: int = 3,
        acquisition: str = "ucb"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Suggest next experiments using acquisition function.

        Args:
            candidate_configs: List of candidate experiment configs
            n_suggestions: Number of experiments to suggest
            acquisition: Acquisition function (ucb, ei, poi)

        Returns:
            List of (config, acquisition_value) sorted by value
        """
        if not candidate_configs:
            return []

        # Predict outcomes for all candidates
        predictions = self.predict_batch(candidate_configs)

        # Compute acquisition function values
        acquisition_values = []
        for pred in predictions:
            if acquisition == "ucb":
                # Upper Confidence Bound: mean + beta * std
                value = pred.mean + 2.0 * pred.std
            elif acquisition == "ei":
                # Expected Improvement (simplified)
                best_observed = float(np.max(self.y_train)) if self.y_train is not None else 0.5
                value = max(0, pred.mean - best_observed) + pred.std
            else:  # poi (Probability of Improvement)
                value = pred.mean + 0.5 * pred.std

            acquisition_values.append(value)

        # Sort by acquisition value
        sorted_indices = np.argsort(acquisition_values)[::-1]
        suggestions = [
            (candidate_configs[i], acquisition_values[i])
            for i in sorted_indices[:n_suggestions]
        ]

        return suggestions

    def filter_proposals(
        self,
        proposals: List[Dict[str, Any]],
        min_predicted_metric: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Filter proposals based on predicted outcomes.

        Args:
            proposals: List of proposals with configs
            min_predicted_metric: Minimum predicted metric to pass

        Returns:
            Filtered list of promising proposals
        """
        if not self.is_trained:
            return proposals  # No filtering if not trained

        filtered = []
        for proposal in proposals:
            config = proposal.get("changes", {})
            prediction = self.predict(config)

            # Keep if predicted outcome is above threshold
            if prediction.mean >= min_predicted_metric:
                # Add prediction info to proposal
                proposal["predicted_metric"] = prediction.mean
                proposal["prediction_confidence"] = prediction.confidence
                filtered.append(proposal)

        return filtered

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of current model state."""
        return {
            "is_trained": self.is_trained,
            "target_metric": self.target_metric,
            "n_training_examples": len(self.X_train) if self.X_train is not None else 0,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "sklearn_available": SKLEARN_AVAILABLE,
            "model_type": "GaussianProcess" if SKLEARN_AVAILABLE and self.is_trained else "Baseline"
        }

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save trained model to disk.

        Args:
            filepath: Optional custom filepath

        Returns:
            Path where model was saved
        """
        if filepath is None:
            filepath = str(self.memory_path / "world_model.json")

        model_data = {
            "is_trained": self.is_trained,
            "target_metric": self.target_metric,
            "feature_names": self.feature_names,
            "simple_baseline": float(self.simple_baseline) if self.simple_baseline else None,
            "sklearn_available": SKLEARN_AVAILABLE,
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        return filepath

    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load trained model from disk.

        Args:
            filepath: Optional custom filepath

        Returns:
            True if loaded successfully
        """
        if filepath is None:
            filepath = str(self.memory_path / "world_model.json")

        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            return False

        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)

            self.is_trained = model_data.get("is_trained", False)
            self.target_metric = model_data.get("target_metric", "auc")
            self.feature_names = model_data.get("feature_names", [])
            self.simple_baseline = model_data.get("simple_baseline")

            # Note: GP model itself not serialized (would need pickle/joblib)
            # User should call train_on_history() after loading

            return True
        except Exception:
            return False


def get_world_model(
    memory_path: Optional[str] = None,
    target_metric: str = "auc",
    auto_train: bool = True
) -> WorldModel:
    """
    Factory function to get world-model instance.

    Args:
        memory_path: Optional custom memory path
        target_metric: Metric to predict
        auto_train: Automatically train on existing history

    Returns:
        WorldModel instance
    """
    kwargs = {"target_metric": target_metric}
    if memory_path:
        kwargs["memory_path"] = memory_path

    model = WorldModel(**kwargs)

    if auto_train:
        model.train_on_history()

    return model
