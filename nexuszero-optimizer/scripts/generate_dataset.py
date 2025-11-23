"""
Script to generate training/validation/test datasets.

Usage:
    python generate_dataset.py --output_dir data --train_samples 10000
"""

import argparse
import logging
from pathlib import Path

from nexuszero_optimizer.training.dataset import ProofCircuitGenerator
from nexuszero_optimizer.utils.crypto_bridge import get_crypto_bridge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate proof circuit datasets for training"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=10000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=2000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=2000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=10,
        help="Minimum nodes per circuit"
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=1000,
        help="Maximum nodes per circuit"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_crypto_bridge",
        action="store_true",
        help="Use actual Rust crypto library (if available)"
    )
    
    args = parser.parse_args()
    
    # Initialize crypto bridge if requested
    crypto_bridge = None
    if args.use_crypto_bridge:
        crypto_bridge = get_crypto_bridge()
        if crypto_bridge.is_available():
            logger.info("✓ Using Rust crypto library for parameter evaluation")
        else:
            logger.warning("⚠ Rust crypto library not available, using simulation")
    
    # Initialize generator
    generator = ProofCircuitGenerator(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        crypto_bridge=crypto_bridge,
        seed=args.seed,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training set
    logger.info(f"Generating {args.train_samples} training samples...")
    generator.generate_dataset(
        num_samples=args.train_samples,
        output_dir=args.output_dir,
        split="train",
    )
    
    # Generate validation set
    logger.info(f"Generating {args.val_samples} validation samples...")
    generator.generate_dataset(
        num_samples=args.val_samples,
        output_dir=args.output_dir,
        split="val",
    )
    
    # Generate test set
    logger.info(f"Generating {args.test_samples} test samples...")
    generator.generate_dataset(
        num_samples=args.test_samples,
        output_dir=args.output_dir,
        split="test",
    )
    
    logger.info("✅ Dataset generation complete!")
    logger.info(f"   Training samples: {args.train_samples}")
    logger.info(f"   Validation samples: {args.val_samples}")
    logger.info(f"   Test samples: {args.test_samples}")
    logger.info(f"   Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
