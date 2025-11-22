"""Combined hyperparameter tuning with WandB logging.

Runs Optuna trials, logging metrics to WandB per trial. Optionally runs a
follow-up Ray Tune search using the best Optuna params as a seed.

Requirements:
    - optuna>=3.3.0 (required)
    - wandb>=0.15.0 (required)
    - ray[tune]>=2.7.0 (optional, for --ray flag)
"""

import argparse
import copy

from nexuszero_optimizer import Config, Trainer
from nexuszero_optimizer.training.tuner import ray_trainable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument(
        "--ray", action="store_true", help="Run Ray Tune after Optuna"
    )
    p.add_argument("--epochs", type=int, default=8, help="Epochs per trial")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.training.num_epochs = args.epochs
    cfg.training.wandb_enabled = True
    cfg.training.wandb_run_name = "optuna-master"
    cfg.training.wandb_tags = ["tuning", "optuna"]

    try:
        import wandb
    except ImportError:
        raise RuntimeError(
            "WandB is required. Install with: pip install wandb>=0.15.0"
        )

    # Optuna Tuning
    try:
        from nexuszero_optimizer.training.tuner import optuna
    except ImportError:
        raise RuntimeError(
            "Optuna is required. Install with: pip install optuna>=3.3.0"
        )
    if optuna is None:
        raise RuntimeError(
            "Optuna is required. Install with: pip install optuna>=3.3.0"
        )

    def objective(trial):
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.model.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [128, 256, 384]
        )
        trial_cfg.model.num_layers = trial.suggest_int("num_layers", 3, 8)
        trial_cfg.model.num_heads = trial.suggest_categorical(
            "num_heads", [4, 8]
        )
        trial_cfg.model.dropout = trial.suggest_float("dropout", 0.05, 0.3)
        trial_cfg.training.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-5, 5e-4
        )
        trial_cfg.training.wandb_run_name = f"trial-{trial.number}"
        trial_cfg.training.wandb_tags = ["optuna", f"trial-{trial.number}"]
        trainer = Trainer(trial_cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        if wandb:
            wandb.log(
                {"trial_loss": metrics.get("loss", 0.0)}, step=trial.number
            )
        return metrics.get("loss", 1e9)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best trial params:", study.best_trial.params)
    print("Best trial loss:", study.best_value)

    if args.ray:
        try:
            from ray import tune
        except ImportError:
            print(
                "Ray Tune not available. "
                "Install with: pip install 'ray[tune]>=2.7.0'"
            )
            print("Skipping Ray Tune phase.")
            return

        search_space = {
            "config_path": args.config,
            "hidden_dim": tune.choice(
                [study.best_trial.params.get("hidden_dim", 256)]
            ),
            "num_layers": tune.choice(
                [study.best_trial.params.get("num_layers", 6)]
            ),
            "num_heads": tune.choice(
                [study.best_trial.params.get("num_heads", 8)]
            ),
            "dropout": tune.uniform(0.05, 0.3),
            "learning_rate": tune.loguniform(1e-5, 5e-4),
        }
        tuner = tune.Tuner(
            ray_trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=4, metric="loss", mode="min"
            ),
        )
        results = tuner.fit()
        best = results.get_best_result(metric="loss", mode="min")
        print("Ray best config:", best.config)
        print("Ray best loss:", best.metrics.get("loss"))


if __name__ == "__main__":
    main()
