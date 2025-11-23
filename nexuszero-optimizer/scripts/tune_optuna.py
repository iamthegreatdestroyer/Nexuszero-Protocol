"""Run Optuna hyperparameter search with Bayesian optimization.

This script performs hyperparameter tuning with:
- 50 trials minimum (configurable)
- Bayesian optimization using Optuna
- Tunes: learning_rate, hidden_dim, num_layers, dropout
- Target: minimize validation loss
- Saves best hyperparameters to config file
"""

import argparse
from pathlib import Path
from nexuszero_optimizer import Config
from nexuszero_optimizer.training.tuner import OptunaTuner


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to base config file"
    )
    p.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials (default: 50)"
    )
    p.add_argument(
        "--output",
        type=str,
        default="best_config.yaml",
        help="Path to save best config"
    )
    p.add_argument(
        "--study_name",
        type=str,
        default="nexuszero_tuning",
        help="Name for the Optuna study"
    )
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="Storage URL for Optuna study"
    )
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading config from {args.config}...")
    cfg = Config.from_yaml(args.config)
    
    print(f"Starting hyperparameter tuning with {args.trials} trials...")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    
    # Create tuner with 50+ trials
    tuner = OptunaTuner(
        cfg,
        n_trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
    )
    
    # Run optimization
    study = tuner.run()
    
    # Print results
    print("\n" + "="*60)
    print("Hyperparameter Tuning Complete!")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_trial.params.items():
        print(f"  {param}: {value}")
    
    # Save best config
    output_path = Path(args.output)
    tuner.save_best_config(study, str(output_path))
    
    print(f"\n✓ Best configuration saved to {output_path.absolute()}")
    print(f"✓ Study database saved to {args.storage}")


if __name__ == "__main__":
    main()
