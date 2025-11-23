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
import os
from pathlib import Path
import logging

from nexuszero_optimizer import Config, Trainer
from nexuszero_optimizer.training.tuner import ray_trainable
from nexuszero_optimizer.training.dataset import ProofCircuitGenerator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL (e.g. sqlite:///optuna_study.db) "
            "for persistence/resume"
        ),
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="nexuszero_tuning",
        help="Optuna study name when using persistent storage",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampler reproducibility",
    )
    p.add_argument(
        "--ray", action="store_true", help="Run Ray Tune after Optuna"
    )
    p.add_argument("--epochs", type=int, default=8, help="Epochs per trial")
    p.add_argument(
        "--generate",
        action="store_true",
        help="Auto-generate dataset if missing",
    )
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--val-samples", type=int, default=100)
    p.add_argument("--test-samples", type=int, default=100)
    p.add_argument(
        "--batch-cap",
        type=int,
        default=None,
        help="Maximum training batches per epoch (speed-up)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    # Resolve relative data_dir against config file location
    config_root = Path(args.config).resolve().parent
    if not os.path.isabs(cfg.data_dir):
        cfg.data_dir = str((config_root / cfg.data_dir).resolve())
    cfg.training.num_epochs = args.epochs
    cfg.training.wandb_enabled = True
    cfg.training.wandb_run_name = "optuna-master"
    cfg.training.wandb_tags = ["tuning", "optuna"]
    # Optional batch cap
    if args.batch_cap is not None:
        cfg.training.max_batches_per_epoch = args.batch_cap

    def dataset_exists(path: str) -> bool:
        return all(
            (Path(path) / split / "index.h5").exists()
            for split in ("train", "val", "test")
        )

    if not dataset_exists(cfg.data_dir):
        if args.generate:
            logger.info(
                "Dataset missing; generating synthetic dataset for tuning"
            )
            gen = ProofCircuitGenerator(
                min_nodes=10,
                max_nodes=120,
                seed=cfg.seed,
            )
            gen.generate_dataset(
                args.train_samples, cfg.data_dir, split="train"
            )
            gen.generate_dataset(
                args.val_samples, cfg.data_dir, split="val"
            )
            gen.generate_dataset(
                args.test_samples, cfg.data_dir, split="test"
            )
        else:
            raise FileNotFoundError(
                "Dataset not found under '"
                + str(cfg.data_dir)
                + "'. Run with --generate or precompute."
            )

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
        # Initialize a dedicated WandB run per trial (offline respected)
        wandb.init(
            project=cfg.training.wandb_project,
            name=f"trial-{trial.number}",
            reinit=True,
            config=cfg.to_dict(),
            tags=["optuna", f"trial-{trial.number}"],
        )
        trial_cfg = copy.deepcopy(cfg)
        # Disable internal Trainer wandb/tensorboard to avoid double init
        # and premature finish of the outer trial-level run.
        trial_cfg.training.wandb_enabled = False
        if hasattr(trial_cfg.training, "tensorboard_enabled"):
            trial_cfg.training.tensorboard_enabled = False
        # Ensure per-trial absolute data_dir resolution (defensive)
        if not os.path.isabs(trial_cfg.data_dir):
            trial_cfg.data_dir = str(
                (config_root / trial_cfg.data_dir).resolve()
            )
        trial_cfg.model.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [128, 256, 384]
        )
        trial_cfg.model.num_layers = trial.suggest_int("num_layers", 3, 8)
        trial_cfg.model.num_heads = trial.suggest_categorical(
            "num_heads", [4, 8]
        )
        trial_cfg.model.dropout = trial.suggest_float(
            "dropout", 0.05, 0.3
        )
        trial_cfg.training.learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True
        )
        trial_cfg.training.wandb_run_name = f"trial-{trial.number}"
        trial_cfg.training.wandb_tags = ["optuna", f"trial-{trial.number}"]
        if args.batch_cap is not None:
            trial_cfg.training.max_batches_per_epoch = args.batch_cap
        trainer = Trainer(trial_cfg)
        logger.info(
            "Starting trial %d: hidden_dim=%s layers=%s heads=%s" % (
                trial.number,
                trial_cfg.model.hidden_dim,
                trial_cfg.model.num_layers,
                trial_cfg.model.num_heads,
            )
        )
        logger.info(
            "dropout=%.3f lr=%.2e" % (
                trial_cfg.model.dropout,
                trial_cfg.training.learning_rate,
            )
        )
        trainer.fit()
        metrics = trainer.evaluate_test()
        wandb.log({"trial_loss": metrics.get("loss", 0.0)}, step=trial.number)
        wandb.finish()
        return metrics.get("loss", 1e9)

    # Create or load persistent study if storage provided
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.storage:
        try:
            study = optuna.load_study(
                study_name=args.study_name, storage=args.storage
            )
            logger.info(
                f"Loaded existing study '{args.study_name}' from storage "
                f"{args.storage}"
            )
        except KeyError:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                direction="minimize",
                sampler=sampler,
            )
            logger.info(
                f"Created new study '{args.study_name}' at storage "
                f"{args.storage}"
            )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
        logger.info(
            "Created in-memory study (no persistent storage specified)"
        )

    study.optimize(objective, n_trials=args.trials)
    print("Best trial params:", study.best_trial.params)
    print("Best trial loss:", study.best_value)

    # Summary table of completed trials for documentation
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        print("\nCompleted Trials Summary:")
        for t in completed:
            print(
                f"Trial {t.number}: value={t.value:.6f} params={t.params}"
            )

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
