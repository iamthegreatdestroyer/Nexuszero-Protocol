"""Hyperparameter tuning utilities (Optuna + Ray)."""

from typing import Dict, Any
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
    def __init__(self, base_config: Config, n_trials: int = 10):
        self.base_config = base_config
        self.n_trials = n_trials

    def _objective(self, trial: "optuna.trial.Trial"):
        cfg = copy.deepcopy(self.base_config)
        cfg.model.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [128, 256, 384]
        )
        cfg.model.num_layers = trial.suggest_int("num_layers", 3, 8)
        cfg.model.num_heads = trial.suggest_categorical("num_heads", [4, 8])
        cfg.model.dropout = trial.suggest_float("dropout", 0.05, 0.3)
        cfg.training.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-5, 5e-4
        )
        cfg.training.num_epochs = 10  # short runs for tuning
        trainer = Trainer(cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        return metrics.get("loss", 1e9)

    def run(self):
        if optuna is None:
            raise RuntimeError("Optuna not installed")
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.n_trials)
        return study


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
