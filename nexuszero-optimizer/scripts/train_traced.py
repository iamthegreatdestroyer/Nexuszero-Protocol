"""Traced training runner using OpenTelemetry for Nexuszero Optimizer."""

from nexuszero_optimizer import Config, Trainer
from nexuszero_optimizer.utils.tracing import init_tracer, get_tracer
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    return p.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)

    # Initialize tracer and get a tracer
    init_tracer(service_name="nexuszero-optimizer")
    tracer = get_tracer(__name__)

    trainer = Trainer(config)

    with tracer.start_as_current_span("training_run") as run_span:
        run_span.set_attribute("num_epochs", config.training.num_epochs)
        run_span.set_attribute("lr", config.training.learning_rate)

        for epoch in range(1, config.training.num_epochs + 1):
            with tracer.start_as_current_span(f"epoch.{epoch}") as epoch_span:
                epoch_span.set_attribute("epoch", epoch)
                train_metrics = trainer.train_epoch(epoch)
                val_metrics = trainer.validate(epoch)
                epoch_span.set_attribute("train_loss", train_metrics.get("loss", 0.0))
                epoch_span.set_attribute("val_loss", val_metrics.get("loss", 0.0))

            trainer._maybe_checkpoint(val_metrics, epoch)

    test_metrics = trainer.evaluate_test()
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
