"""Hyperparameter tuning utilities (Optuna + Ray)."""

from typing import Dict, Any, Optional
import copy

try:
    import optuna
except Exception:  # pragma: no cover - optional
    optuna = None

try:
    from ray import tune
except Exception:  # pragma: no cover - optional
    tune = None

from .trainer import Trainer
from ..utils.config import Config


class OptunaTuner:
    def __init__(
        self,
        base_config: Config,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Initialize Optuna tuner for hyperparameter optimization.
        
        Args:
            base_config: Base configuration to modify
            n_trials: Number of trials to run (default 50 as per requirements)
            study_name: Optional name for the study
            storage: Optional storage URL for persistence
        """
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage

    def _objective(self, trial: "optuna.trial.Trial"):
        """
        Objective function for Optuna optimization.
        
        Tunes:
        - learning_rate: 1e-5 to 5e-4
        - hidden_dim: 128, 256, 384
        - num_layers: 3 to 8
        - dropout: 0.05 to 0.3
        
        Target: minimize validation loss
        """
        cfg = copy.deepcopy(self.base_config)
        
        # Hyperparameters to tune
        cfg.model.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [128, 256, 384]
        )
        cfg.model.num_layers = trial.suggest_int("num_layers", 3, 8)
        cfg.model.num_heads = trial.suggest_categorical("num_heads", [4, 8])
        cfg.model.dropout = trial.suggest_float("dropout", 0.05, 0.3)
        cfg.training.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-5, 5e-4
        )
        
        # Short runs for tuning
        cfg.training.num_epochs = 10
        
        # Train model
        trainer = Trainer(cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        
        return metrics.get("loss", 1e9)

    def run(self):
        """
        Run hyperparameter optimization.
        
        Returns:
            Optuna study object with best parameters
        """
        if optuna is None:
            raise RuntimeError("Optuna not installed")
        
        # Create study with optional persistence
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
        )
        
        # Run optimization
        study.optimize(self._objective, n_trials=self.n_trials)
        
        return study
    
    def save_best_config(self, study, output_path: str):
        """
        Save best hyperparameters to config file.
        
        Args:
            study: Optuna study object
            output_path: Path to save config YAML
        """
        import yaml
        
        best_params = study.best_params
        best_config = copy.deepcopy(self.base_config)
        
        # Update with best parameters
        best_config.model.hidden_dim = best_params['hidden_dim']
        best_config.model.num_layers = best_params['num_layers']
        best_config.model.num_heads = best_params['num_heads']
        best_config.model.dropout = best_params['dropout']
        best_config.training.learning_rate = best_params['learning_rate']
        
        # Save to YAML
        config_dict = best_config.to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"✓ Saved best config to {output_path}")
        print(f"  Best loss: {study.best_value:.6f}")
        print(f"  Best params: {best_params}")


def ray_trainable(config: Dict[str, Any]):  # Ray Tune trainable
    base_cfg_path = config.get("config_path", "config.yaml")
    cfg = Config.from_yaml(base_cfg_path)
    cfg.model.hidden_dim = config["hidden_dim"]
    cfg.model.num_layers = config["num_layers"]
    cfg.model.num_heads = config["num_heads"]
    cfg.model.dropout = config["dropout"]
    cfg.training.learning_rate = config["learning_rate"]
    cfg.training.num_epochs = 10
    trainer = Trainer(cfg)
    trainer.fit()
    metrics = trainer.evaluate_test()
    loss = metrics.get("loss", 1e9)
    bit_sec = metrics.get("bit_security", 0.0)
    if tune:
        tune.report(loss=loss, bit_security=bit_sec)


__all__ = ["OptunaTuner", "ray_trainable"]
