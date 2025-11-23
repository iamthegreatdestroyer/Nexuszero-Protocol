"""
Dataset and data generation for proof circuit optimization.

This module provides:
- ProofCircuitDataset: PyTorch Geometric dataset for loading circuits
- ProofCircuitGenerator: Generator for synthetic training data
- Collate functions for batching
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import numpy as np
import h5py
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProofCircuitDataset(Dataset):
    """
    Dataset of proof circuits with optimal parameters.
    
    Each sample contains:
    - Circuit graph (nodes = gates, edges = connections)
    - Node features (gate type, fanin, fanout, depth)
    - Edge features (connection type, weight)
    - Target parameters (n, q, sigma) - normalized
    - Performance metrics (proof_size, prove_time, verify_time) - normalized
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset files
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load dataset index
        self.index_file = self.data_dir / split / "index.h5"
        
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Dataset index not found at {self.index_file}. "
                f"Please generate dataset first using ProofCircuitGenerator."
            )
        
        with h5py.File(self.index_file, 'r') as f:
            self.num_samples = len(f['circuit_ids'])
            self.circuit_ids = f['circuit_ids'][:]
        
        logger.info(f"Loaded {self.split} dataset with {self.num_samples} samples")
    
    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
        
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [num_nodes, node_feat_dim]
            - edge_index: Edge indices [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_feat_dim]
            - y: Target parameters [3] (n, q, sigma) - normalized
            - metrics: Performance metrics [3] (size, prove_time, verify_time)
            - circuit_id: Circuit identifier
        """
        circuit_id = self.circuit_ids[idx]
        
        # Load circuit graph
        graph_file = self.data_dir / self.split / f"circuit_{circuit_id}.h5"
        
        with h5py.File(graph_file, 'r') as f:
            # Node features: [gate_type_onehot, fanin, fanout, depth]
            x = torch.tensor(f['node_features'][:], dtype=torch.float)
            
            # Edge indices: [2, num_edges]
            edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
            
            # Edge features: [connection_type_onehot, weight]
            edge_attr = torch.tensor(f['edge_attr'][:], dtype=torch.float)
            
            # Target parameters (normalized to [0, 1])
            params = f['optimal_params'][:]
            y = torch.tensor(params, dtype=torch.float)
            
            # Performance metrics (normalized to [0, 1])
            metrics = f['metrics'][:]
            metrics_tensor = torch.tensor(metrics, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            metrics=metrics_tensor,
            circuit_id=circuit_id,
        )
        
        if self.transform:
            data = self.transform(data)
        
        return data


class ProofCircuitGenerator:
    """Generate synthetic proof circuits for training."""
    
    # Gate types for circuit construction
    GATE_TYPES = ['AND', 'OR', 'NOT', 'XOR', 'MUX', 'ADD', 'MUL']
    
    # Connection types for edges
    CONN_TYPES = ['DATA', 'CONTROL', 'FEEDBACK']
    
    def __init__(
        self,
        min_nodes: int = 10,
        max_nodes: int = 1000,
        crypto_bridge=None,
        seed: Optional[int] = None,
    ):
        """
        Initialize circuit generator.
        
        Args:
            min_nodes: Minimum number of nodes in circuit
            max_nodes: Maximum number of nodes in circuit
            crypto_bridge: Optional CryptoBridge for actual parameter evaluation
            seed: Random seed for reproducibility
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.crypto_bridge = crypto_bridge
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_circuit(self) -> Dict:
        """
        Generate random proof circuit as a DAG.
        
        Returns:
            Dictionary with circuit graph and characteristics:
            - num_nodes: Number of gates in circuit
            - node_features: Node feature matrix
            - edge_index: Edge connectivity
            - edge_attr: Edge features
            - gate_types: List of gate type names
        """
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        
        # Generate node features
        node_features = []
        
        for i in range(num_nodes):
            # One-hot encode gate type
            gate_type_idx = np.random.choice(len(self.GATE_TYPES))
            gate_onehot = np.zeros(len(self.GATE_TYPES))
            gate_onehot[gate_type_idx] = 1
            
            # Fanin/fanout (0-4)
            fanin = np.random.randint(0, 5)
            fanout = np.random.randint(0, 5)
            
            # Depth in circuit (normalized)
            depth = i / num_nodes
            
            # Concatenate features: [gate_type_onehot (7), fanin (1), fanout (1), depth (1)]
            # Total: 10 features
            features = np.concatenate([gate_onehot, [fanin, fanout, depth]])
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Generate edges (random DAG - only forward connections)
        edge_index = []
        edge_attr = []
        
        for i in range(num_nodes):
            # Connect to 0-3 previous nodes (ensures DAG structure)
            if i == 0:
                continue
            
            num_connections = np.random.randint(0, min(4, i + 1))
            if num_connections == 0:
                continue
            
            targets = np.random.choice(i, size=num_connections, replace=False)
            
            for target in targets:
                edge_index.append([i, target])
                
                # Edge features: connection type (one-hot) + weight
                conn_type_idx = np.random.choice(len(self.CONN_TYPES))
                conn_onehot = np.zeros(len(self.CONN_TYPES))
                conn_onehot[conn_type_idx] = 1
                weight = np.random.random()
                
                # Total: 4 features (3 for type + 1 for weight)
                edge_attr.append(np.concatenate([conn_onehot, [weight]]))
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 4), dtype=np.float32)
        
        return {
            'num_nodes': num_nodes,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'gate_types': self.GATE_TYPES,
        }
    
    def find_optimal_parameters(
        self,
        circuit: Dict,
        security_level: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find optimal parameters for circuit with diverse parameter combinations.
        
        Generates diverse parameters covering:
        - n: 64-2048 (powers of 2)
        - q: varying primes (4096-131072)
        - sigma: 1.0-8.0
        - Security levels: 128/192/256-bit
        - Edge cases and boundary conditions
        
        Args:
            circuit: Circuit dictionary from generate_random_circuit
            security_level: Optional security level override (128, 192, or 256)
        
        Returns:
            Tuple of (optimal_params, metrics, security_level)
            - optimal_params: [n, q, sigma] (normalized to [0, 1])
            - metrics: [proof_size, prove_time, verify_time] (normalized to [0, 1])
            - security_level: Assigned security level in bits
        """
        num_nodes = circuit['num_nodes']
        
        # Randomly assign security level if not specified (covering all 3 levels)
        if security_level is None:
            security_level = np.random.choice([128, 192, 256])
        
        # Define parameter ranges based on security level and circuit complexity
        # Extended ranges per issue requirements: n=64-2048, sigma=1.0-8.0
        if security_level == 128:
            # 128-bit security: smaller parameters
            n_options = [64, 128, 256, 512]
            q_options = [4096, 8192, 12289, 16384, 20480]
            sigma_range = (1.0, 4.0)
        elif security_level == 192:
            # 192-bit security: medium parameters
            n_options = [256, 512, 1024]
            q_options = [12289, 20480, 40961, 65537]
            sigma_range = (2.0, 6.0)
        else:  # 256-bit security
            # 256-bit security: larger parameters
            n_options = [512, 1024, 2048]
            q_options = [40961, 65537, 98304, 131072]
            sigma_range = (3.0, 8.0)
        
        # Select parameters based on circuit size
        if num_nodes < 50:
            n = np.random.choice(n_options[:2]) if len(n_options) >= 2 else n_options[0]
        elif num_nodes < 200:
            n = np.random.choice(n_options[1:3]) if len(n_options) >= 3 else n_options[-1]
        elif num_nodes < 500:
            n = np.random.choice(n_options[2:]) if len(n_options) >= 3 else n_options[-1]
        else:
            n = n_options[-1]
        
        # Select q (prime modulus) with diversity
        q = np.random.choice(q_options)
        
        # Select sigma with full range diversity
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        
        # Add edge cases randomly (5% probability)
        if np.random.random() < 0.05:
            # Boundary conditions
            edge_case = np.random.choice(['min_n', 'max_n', 'min_sigma', 'max_sigma', 'min_q', 'max_q'])
            if edge_case == 'min_n':
                n = 64
            elif edge_case == 'max_n':
                n = 2048
            elif edge_case == 'min_sigma':
                sigma = 1.0
            elif edge_case == 'max_sigma':
                sigma = 8.0
            elif edge_case == 'min_q':
                q = 4096
            elif edge_case == 'max_q':
                q = 131072
        
        # Add small randomness for diversity (±5%)
        n = int(n * (0.95 + 0.1 * np.random.random()))
        q = int(q * (0.98 + 0.04 * np.random.random()))
        sigma = sigma * (0.95 + 0.1 * np.random.random())
        
        # Clamp to valid ranges
        n = max(64, min(2048, n))
        q = max(4096, min(131072, q))
        sigma = max(1.0, min(8.0, sigma))
        
        # Ensure n is power of 2
        n = 2 ** int(np.round(np.log2(n)))
        
        # Normalize parameters for neural network
        # n: 64-2048 -> 0-1
        # q: 4096-131072 -> 0-1
        # sigma: 1.0-8.0 -> 0-1
        n_norm = (n - 64) / (2048 - 64)
        q_norm = (q - 4096) / (131072 - 4096)
        sigma_norm = (sigma - 1.0) / (8.0 - 1.0)
        
        params = np.array([n_norm, q_norm, sigma_norm], dtype=np.float32)
        
        # Simulate metrics based on parameters and security level
        # Higher security level = larger proof size and time
        security_factor = security_level / 128.0
        
        proof_size = num_nodes * (n / 100) * 16 * security_factor  # bytes
        prove_time = num_nodes * (n / 1000) * 0.1 * security_factor  # ms
        verify_time = num_nodes * (n / 1000) * 0.05 * security_factor  # ms
        
        # Add noise (±10%)
        proof_size *= (0.9 + 0.2 * np.random.random())
        prove_time *= (0.9 + 0.2 * np.random.random())
        verify_time *= (0.9 + 0.2 * np.random.random())
        
        # Normalize metrics
        # size: 0-100KB -> 0-1
        # times: 0-1000ms -> 0-1
        size_norm = min(proof_size / 100000, 1.0)
        prove_norm = min(prove_time / 1000, 1.0)
        verify_norm = min(verify_time / 1000, 1.0)
        
        metrics = np.array([size_norm, prove_norm, verify_norm], dtype=np.float32)
        
        return params, metrics, security_level
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        split: str = "train",
        show_progress: bool = True,
    ):
        """
        Generate full dataset and save to disk.
        
        Args:
            num_samples: Number of circuits to generate
            output_dir: Output directory
            split: Dataset split name ('train', 'val', 'test')
            show_progress: Show progress bar
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_samples} circuits for {split} split...")
        
        # Create index file
        with h5py.File(output_path / "index.h5", 'w') as f:
            f.create_dataset('circuit_ids', data=np.arange(num_samples))
        
        # Generate circuits
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating {split} circuits")
        
        for i in iterator:
            circuit = self.generate_random_circuit()
            params, metrics, security_level = self.find_optimal_parameters(circuit)
            
            # Save circuit
            with h5py.File(output_path / f"circuit_{i}.h5", 'w') as f:
                f.create_dataset('node_features', data=circuit['node_features'])
                f.create_dataset('edge_index', data=circuit['edge_index'])
                f.create_dataset('edge_attr', data=circuit['edge_attr'])
                f.create_dataset('optimal_params', data=params)
                f.create_dataset('metrics', data=metrics)
                f.create_dataset('security_level', data=security_level)
        
        logger.info(f"✓ Generated {num_samples} circuits in {output_path}")


def collate_fn(batch: List[Data]) -> Batch:
    """
    Custom collate function for batching circuits.
    
    Args:
        batch: List of PyG Data objects
    
    Returns:
        Batched Data object with proper graph batching
    """
    return Batch.from_data_list(batch)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory containing datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    
    train_dataset = ProofCircuitDataset(data_dir, split="train")
    val_dataset = ProofCircuitDataset(data_dir, split="val")
    test_dataset = ProofCircuitDataset(data_dir, split="test")
    
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset generation
    generator = ProofCircuitGenerator(min_nodes=10, max_nodes=100, seed=42)
    
    print("Testing circuit generation...")
    circuit = generator.generate_random_circuit()
    print(f"✓ Generated circuit with {circuit['num_nodes']} nodes")
    print(f"  Node features shape: {circuit['node_features'].shape}")
    print(f"  Edge index shape: {circuit['edge_index'].shape}")
    
    params, metrics, security_level = generator.find_optimal_parameters(circuit)
    print(f"✓ Optimal parameters: {params}")
    print(f"  Metrics: {metrics}")
    print(f"  Security level: {security_level}-bit")
