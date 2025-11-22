"""Run Ray Tune hyperparameter search."""

import argparse
from ray import tune
from nexuszero_optimizer.training.tuner import ray_trainable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--samples", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    search_space = {
        "config_path": args.config,
        "hidden_dim": tune.choice([128, 256, 384]),
        "num_layers": tune.randint(3, 9),
        "num_heads": tune.choice([4, 8]),
        "dropout": tune.uniform(0.05, 0.3),
        "learning_rate": tune.loguniform(1e-5, 5e-4),
    }
    tuner = tune.Tuner(
        ray_trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.samples, metric="loss", mode="min"
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    print("Best config:", best.config)
    print("Best loss:", best.metrics.get("loss"))


if __name__ == "__main__":
    main()
