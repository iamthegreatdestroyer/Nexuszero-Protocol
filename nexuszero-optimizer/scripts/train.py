"""CLI script to train Nexuszero Optimizer model."""

import argparse
from nexuszero_optimizer import Config, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    trainer = Trainer(config)
    trainer.fit()
    test_metrics = trainer.evaluate_test()
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
