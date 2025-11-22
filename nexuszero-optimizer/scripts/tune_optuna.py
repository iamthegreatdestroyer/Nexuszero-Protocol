"""Run Optuna hyperparameter search."""

import argparse
from nexuszero_optimizer import Config
from nexuszero_optimizer.training.tuner import OptunaTuner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--trials", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    tuner = OptunaTuner(cfg, n_trials=args.trials)
    study = tuner.run()
    print("Best trial:")
    print(study.best_trial.params)
    print("Best loss:", study.best_value)


if __name__ == "__main__":
    main()
