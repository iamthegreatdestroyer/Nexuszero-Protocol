"""
Script to create a default configuration file.

Usage:
    python create_config.py --output config.yaml
"""

import argparse
from pathlib import Path

from nexuszero_optimizer.utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Create default configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Output path for configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="GNN hidden dimension"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of GNN layers"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create config with custom values
    config = Config()
    config.device = args.device
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Save to file
    output_path = Path(args.output)
    config.to_yaml(str(output_path))
    
    print(f"✅ Created configuration file: {output_path.absolute()}")
    print(f"   Device: {config.device}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    print(f"   Num layers: {config.model.num_layers}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")


if __name__ == "__main__":
    main()
