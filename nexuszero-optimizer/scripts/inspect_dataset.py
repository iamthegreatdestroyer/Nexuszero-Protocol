"""
Script to inspect dataset statistics.

Usage:
    python inspect_dataset.py --data_dir data --split train
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import Counter

from nexuszero_optimizer.training.dataset import ProofCircuitDataset


def main():
    parser = argparse.ArgumentParser(
        description="Inspect proof circuit dataset"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to analyze (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        dataset = ProofCircuitDataset(args.data_dir, split=args.split)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    num_samples = args.num_samples or len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.split}")
    print(f"Total samples: {len(dataset)}")
    print(f"Analyzing: {num_samples} samples")
    print(f"{'='*60}\n")
    
    # Collect statistics
    num_nodes_list = []
    num_edges_list = []
    params_list = []
    metrics_list = []
    
    for i in range(num_samples):
        data = dataset[i]
        
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)
        params_list.append(data.y.numpy())
        metrics_list.append(data.metrics.numpy())
    
    num_nodes_arr = np.array(num_nodes_list)
    num_edges_arr = np.array(num_edges_list)
    params_arr = np.array(params_list)
    metrics_arr = np.array(metrics_list)
    
    # Print statistics
    print("Circuit Statistics:")
    print(f"  Nodes per circuit:")
    print(f"    Mean: {num_nodes_arr.mean():.1f}")
    print(f"    Std:  {num_nodes_arr.std():.1f}")
    print(f"    Min:  {num_nodes_arr.min()}")
    print(f"    Max:  {num_nodes_arr.max()}")
    
    print(f"\n  Edges per circuit:")
    print(f"    Mean: {num_edges_arr.mean():.1f}")
    print(f"    Std:  {num_edges_arr.std():.1f}")
    print(f"    Min:  {num_edges_arr.min()}")
    print(f"    Max:  {num_edges_arr.max()}")
    
    print(f"\n  Avg edges per node: {(num_edges_arr / num_nodes_arr).mean():.2f}")
    
    print(f"\nParameter Statistics (normalized):")
    print(f"  n (dimension):")
    print(f"    Mean: {params_arr[:, 0].mean():.3f}")
    print(f"    Std:  {params_arr[:, 0].std():.3f}")
    
    print(f"  q (modulus):")
    print(f"    Mean: {params_arr[:, 1].mean():.3f}")
    print(f"    Std:  {params_arr[:, 1].std():.3f}")
    
    print(f"  σ (sigma):")
    print(f"    Mean: {params_arr[:, 2].mean():.3f}")
    print(f"    Std:  {params_arr[:, 2].std():.3f}")
    
    print(f"\nMetrics Statistics (normalized):")
    print(f"  Proof size:")
    print(f"    Mean: {metrics_arr[:, 0].mean():.3f}")
    print(f"    Std:  {metrics_arr[:, 0].std():.3f}")
    
    print(f"  Prove time:")
    print(f"    Mean: {metrics_arr[:, 1].mean():.3f}")
    print(f"    Std:  {metrics_arr[:, 1].std():.3f}")
    
    print(f"  Verify time:")
    print(f"    Mean: {metrics_arr[:, 2].mean():.3f}")
    print(f"    Std:  {metrics_arr[:, 2].std():.3f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
