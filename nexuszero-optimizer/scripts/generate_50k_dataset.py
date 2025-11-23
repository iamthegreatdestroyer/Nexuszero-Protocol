"""
Script to generate 50,000 training examples for neural optimizer.

This script generates a diverse dataset with:
- n: 64-2048 (powers of 2)
- q: varying primes (4096-131072)
- sigma: 1.0-8.0
- All 3 security levels (128/192/256-bit)
- Edge cases and boundary conditions

Output format: JSON file with training examples
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from nexuszero_optimizer.training.dataset import ProofCircuitGenerator
from nexuszero_optimizer.utils.crypto_bridge import get_crypto_bridge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def denormalize_params(params_norm: np.ndarray) -> Dict[str, float]:
    """
    Denormalize parameters from [0, 1] to actual ranges.
    
    Args:
        params_norm: Normalized parameters [n, q, sigma] in [0, 1]
    
    Returns:
        Dictionary with denormalized parameters
    """
    n = int(64 + params_norm[0] * (2048 - 64))
    q = int(4096 + params_norm[1] * (131072 - 4096))
    sigma = 1.0 + params_norm[2] * (8.0 - 1.0)
    
    # Ensure n is power of 2
    n = 2 ** int(np.round(np.log2(n)))
    
    return {
        'n': n,
        'q': q,
        'sigma': float(sigma),
    }


def denormalize_metrics(metrics_norm: np.ndarray) -> Dict[str, float]:
    """
    Denormalize metrics from [0, 1] to actual ranges.
    
    Args:
        metrics_norm: Normalized metrics [proof_size, prove_time, verify_time] in [0, 1]
    
    Returns:
        Dictionary with denormalized metrics
    """
    proof_size = int(metrics_norm[0] * 100000)  # bytes
    prove_time = metrics_norm[1] * 1000  # ms
    verify_time = metrics_norm[2] * 1000  # ms
    
    return {
        'proof_size': proof_size,
        'prove_time': float(prove_time),
        'verify_time': float(verify_time),
    }


def calculate_security_bits(n: int, q: int, sigma: float) -> int:
    """
    Estimate security bits based on parameters.
    
    Args:
        n: Lattice dimension
        q: Modulus
        sigma: Error distribution parameter
    
    Returns:
        Estimated security level in bits
    """
    # Simplified security estimation
    # In practice, this would use proper cryptographic analysis
    log_q = np.log2(q)
    security = min(256, int((n * log_q) / (4 * sigma)))
    
    # Round to nearest security level
    if security < 160:
        return 128
    elif security < 224:
        return 192
    else:
        return 256


def generate_training_example(
    generator: ProofCircuitGenerator,
    example_id: int,
) -> Dict:
    """
    Generate a single training example.
    
    Args:
        generator: Circuit generator
        example_id: Example identifier
    
    Returns:
        Dictionary with training example in the format:
        {
            'id': int,
            'params': {'n': int, 'q': int, 'sigma': float},
            'metrics': {
                'proof_size': int,
                'prove_time': float,
                'verify_time': float,
                'security_bits': int
            }
        }
    """
    # Generate random circuit
    circuit = generator.generate_random_circuit()
    
    # Find optimal parameters (normalized)
    params_norm, metrics_norm, security_level = generator.find_optimal_parameters(circuit)
    
    # Denormalize to actual values
    params = denormalize_params(params_norm)
    metrics = denormalize_metrics(metrics_norm)
    
    # Calculate security bits
    security_bits = calculate_security_bits(params['n'], params['q'], params['sigma'])
    metrics['security_bits'] = security_bits
    
    return {
        'id': example_id,
        'params': params,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate 50,000 training examples for neural optimizer"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/training_data_50k.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of training samples to generate"
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
    
    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate training examples
    logger.info(f"Generating {args.num_samples} training examples...")
    training_data = []
    
    for i in tqdm(range(args.num_samples), desc="Generating examples"):
        example = generate_training_example(generator, i)
        training_data.append(example)
    
    # Compute statistics
    security_levels = [ex['metrics']['security_bits'] for ex in training_data]
    n_values = [ex['params']['n'] for ex in training_data]
    sigma_values = [ex['params']['sigma'] for ex in training_data]
    
    logger.info("✅ Dataset generation complete!")
    logger.info(f"   Total samples: {len(training_data)}")
    logger.info(f"   Security levels: 128-bit: {security_levels.count(128)}, "
                f"192-bit: {security_levels.count(192)}, "
                f"256-bit: {security_levels.count(256)}")
    logger.info(f"   n range: {min(n_values)} - {max(n_values)}")
    logger.info(f"   σ range: {min(sigma_values):.2f} - {max(sigma_values):.2f}")
    
    # Save to JSON
    logger.info(f"Saving to {output_path.absolute()}...")
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'num_samples': len(training_data),
                'min_nodes': args.min_nodes,
                'max_nodes': args.max_nodes,
                'seed': args.seed,
                'security_levels': {
                    '128': security_levels.count(128),
                    '192': security_levels.count(192),
                    '256': security_levels.count(256),
                },
                'parameter_ranges': {
                    'n': {'min': min(n_values), 'max': max(n_values)},
                    'q': {'min': 4096, 'max': 131072},
                    'sigma': {'min': min(sigma_values), 'max': max(sigma_values)},
                }
            },
            'data': training_data,
        }, f, indent=2)
    
    logger.info(f"✓ Saved {len(training_data)} examples to {output_path.absolute()}")


if __name__ == "__main__":
    main()
