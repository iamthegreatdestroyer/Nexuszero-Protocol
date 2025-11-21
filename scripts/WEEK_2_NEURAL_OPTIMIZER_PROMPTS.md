# Week 2: Neural Optimizer Foundation - Copilot Prompts

**Project:** Nexuszero Protocol  
**Phase:** Week 2 - AI-Powered Proof Optimization  
**Duration:** 7 days  
**Goal:** Build GNN-based optimizer for automatic proof parameter tuning

---

## 📋 DAILY BREAKDOWN

### Day 1-2: PyTorch Project Setup & Data Pipeline
### Day 3-4: GNN Architecture for Proof Optimization
### Day 5-6: Soundness Verifier Integration
### Day 7: Training Loop & Initial Benchmarks

---

## 🧠 DAY 1-2: PYTORCH PROJECT SETUP

### Prompt 1.1: Project Structure & Environment

```
Create a PyTorch-based neural optimizer project for zero-knowledge proof parameter optimization.

## Project Requirements
- **Name:** nexuszero-optimizer
- **Type:** Python package with PyTorch
- **Purpose:** Train GNN to optimize proof generation parameters
- **Integration:** Links with nexuszero-crypto Rust library via PyO3

## Structure to Create

```
nexuszero-optimizer/
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
├── src/
│   └── nexuszero_optimizer/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gnn.py              # Graph Neural Network architecture
│       │   ├── attention.py        # Attention mechanisms
│       │   └── encoder.py          # Proof circuit encoder
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # Training loop
│       │   ├── dataset.py          # Proof dataset
│       │   └── metrics.py          # Evaluation metrics
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── optimizer.py        # Parameter optimizer
│       │   └── scheduler.py        # Learning rate scheduling
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── soundness.py        # Soundness checker
│       │   └── validator.py        # Parameter validator
│       └── utils/
│           ├── __init__.py
│           ├── crypto_bridge.py    # Bridge to Rust crypto
│           └── config.py           # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_optimization.py
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
└── notebooks/
    ├── data_exploration.ipynb
    └── model_evaluation.ipynb
```

## Dependencies (requirements.txt)

```txt
# Core ML
torch>=2.1.0
torch-geometric>=2.4.0
torch-scatter>=2.1.2
torch-sparse>=0.6.18

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
networkx>=3.1

# Optimization
optuna>=3.3.0
ray[tune]>=2.7.0

# Rust Bridge
maturin>=1.3.0
pyo3>=0.20.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.14.0
wandb>=0.15.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
h5py>=3.9.0
```

## Initial Setup Code

### 1. Configuration (src/nexuszero_optimizer/utils/config.py)

```python
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class ModelConfig:
    \"\"\"GNN model configuration\"\"\"
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"

@dataclass
class TrainingConfig:
    \"\"\"Training configuration\"\"\"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

@dataclass
class OptimizationConfig:
    \"\"\"Proof optimization configuration\"\"\"
    security_level: int = 128  # bits
    max_proof_size: int = 10000  # bytes
    target_verify_time: float = 50.0  # milliseconds

@dataclass
class Config:
    \"\"\"Main configuration\"\"\"
    model: ModelConfig
    training: TrainingConfig
    optimization: OptimizationConfig
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Hardware
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            **{k: v for k, v in data.items() if k not in ["model", "training", "optimization"]}
        )
    
    def to_yaml(self, path: str):
        data = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "optimization": self.optimization.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)
```

### 2. Rust Bridge (src/nexuszero_optimizer/utils/crypto_bridge.py)

```python
import ctypes
from typing import Dict, List, Tuple
import numpy as np

class CryptoBridge:
    \"\"\"
    Bridge to nexuszero-crypto Rust library.
    
    This class provides Python bindings to the Rust cryptography
    library for proof generation and verification.
    \"\"\"
    
    def __init__(self, lib_path: str = "./libnexuszero_crypto.so"):
        \"\"\"
        Initialize bridge to Rust library.
        
        Args:
            lib_path: Path to compiled Rust library
        \"\"\"
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        \"\"\"Setup ctypes function signatures\"\"\"
        # Example: Proof generation function
        self.lib.generate_proof.argtypes = [
            ctypes.c_void_p,  # statement
            ctypes.c_void_p,  # witness
            ctypes.c_void_p,  # parameters
        ]
        self.lib.generate_proof.restype = ctypes.c_void_p
        
        # Verification function
        self.lib.verify_proof.argtypes = [
            ctypes.c_void_p,  # statement
            ctypes.c_void_p,  # proof
            ctypes.c_void_p,  # parameters
        ]
        self.lib.verify_proof.restype = ctypes.c_bool
    
    def generate_proof(
        self,
        statement: Dict,
        witness: Dict,
        parameters: Dict,
    ) -> Tuple[bytes, float, float]:
        \"\"\"
        Generate zero-knowledge proof.
        
        Args:
            statement: Public statement
            witness: Secret witness
            parameters: Cryptographic parameters
        
        Returns:
            Tuple of (proof_bytes, generation_time_ms, proof_size_bytes)
        \"\"\"
        # Convert Python dicts to C structures
        # Call Rust function
        # Return results
        pass  # TODO: Implement
    
    def verify_proof(
        self,
        statement: Dict,
        proof: bytes,
        parameters: Dict,
    ) -> Tuple[bool, float]:
        \"\"\"
        Verify zero-knowledge proof.
        
        Args:
            statement: Public statement
            proof: Proof bytes
            parameters: Cryptographic parameters
        
        Returns:
            Tuple of (is_valid, verification_time_ms)
        \"\"\"
        pass  # TODO: Implement
    
    def estimate_parameters(
        self,
        security_level: int,
    ) -> Dict:
        \"\"\"
        Get estimated parameters for security level.
        
        Args:
            security_level: Security level in bits (128, 192, 256)
        
        Returns:
            Dictionary with n, q, sigma, estimated sizes/times
        \"\"\"
        pass  # TODO: Implement
```

### 3. Package Init (src/nexuszero_optimizer/__init__.py)

```python
\"\"\"
Nexuszero Optimizer

Neural network-based optimizer for zero-knowledge proof parameters.
Uses Graph Neural Networks to learn optimal parameter configurations
that balance security, proof size, and verification time.
\"\"\"

__version__ = "0.1.0"

from .models.gnn import ProofOptimizationGNN
from .training.trainer import ProofTrainer
from .optimization.optimizer import ParameterOptimizer
from .verification.soundness import SoundnessVerifier

__all__ = [
    "ProofOptimizationGNN",
    "ProofTrainer",
    "ParameterOptimizer",
    "SoundnessVerifier",
]
```

### 4. Setup Configuration (pyproject.toml)

```toml
[build-system]
requires = ["maturin>=1.3,<2.0"]
build-backend = "maturin"

[project]
name = "nexuszero-optimizer"
version = "0.1.0"
description = "Neural optimizer for zero-knowledge proof parameters"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.1.0",
    "torch-geometric>=2.4.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "ruff>=0.0.285",
]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
```

## Initialization Tasks

1. Create all directories
2. Install dependencies: `pip install -r requirements.txt`
3. Build Rust bridge: `maturin develop` (in crypto project)
4. Verify imports: `python -c "import nexuszero_optimizer"`
5. Create default config: `config.yaml`

## Verification Steps

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test PyTorch Geometric
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"

# Test project imports
python -c "from nexuszero_optimizer import ProofOptimizationGNN"

# Run tests
pytest tests/ -v
```

Generate complete PyTorch project with proper structure and dependencies.
```

---

### Prompt 1.2: Training Data Pipeline

```
Implement the data pipeline for generating and loading proof optimization training data.

## Background

The GNN needs training data consisting of:
- **Input:** Proof circuits represented as graphs
- **Features:** Circuit characteristics (depth, width, gate types)
- **Labels:** Optimal parameters (n, q, σ) for each circuit
- **Metrics:** Proof size, generation time, verification time

## Implementation Requirements

### 1. Dataset Definition (src/nexuszero_optimizer/training/dataset.py)

```python
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple
import numpy as np
import h5py
from pathlib import Path

class ProofCircuitDataset(Dataset):
    \"\"\"
    Dataset of proof circuits with optimal parameters.
    
    Each sample contains:
    - Circuit graph (nodes = gates, edges = connections)
    - Node features (gate type, fanin, fanout)
    - Edge features (connection type, weight)
    - Target parameters (n, q, sigma)
    - Performance metrics (proof_size, prove_time, verify_time)
    \"\"\"
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
    ):
        \"\"\"
        Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset files
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply
        \"\"\"
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load dataset index
        self.index_file = self.data_dir / split / "index.h5"
        with h5py.File(self.index_file, 'r') as f:
            self.num_samples = len(f['circuit_ids'])
            self.circuit_ids = f['circuit_ids'][:]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Data:
        \"\"\"
        Get a single training sample.
        
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [num_nodes, node_feat_dim]
            - edge_index: Edge indices [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_feat_dim]
            - y: Target parameters [3] (n, q, sigma)
            - metrics: Performance metrics [3] (size, prove_time, verify_time)
        \"\"\"
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
            
            # Target parameters (normalized)
            params = f['optimal_params'][:]
            y = torch.tensor(params, dtype=torch.float)
            
            # Performance metrics (normalized)
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
    \"\"\"Generate synthetic proof circuits for training.\"\"\"
    
    def __init__(
        self,
        min_nodes: int = 10,
        max_nodes: int = 1000,
        crypto_bridge=None,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.crypto_bridge = crypto_bridge
    
    def generate_random_circuit(self) -> Dict:
        \"\"\"
        Generate random proof circuit.
        
        Returns:
            Dictionary with circuit graph and characteristics
        \"\"\"
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        
        # Generate random DAG (Directed Acyclic Graph)
        # Nodes = gates (AND, OR, NOT, XOR, etc.)
        # Edges = wire connections
        
        # Node features
        gate_types = ['AND', 'OR', 'NOT', 'XOR', 'MUX', 'ADD', 'MUL']
        node_features = []
        
        for i in range(num_nodes):
            # One-hot encode gate type
            gate_type = np.random.choice(len(gate_types))
            gate_onehot = np.zeros(len(gate_types))
            gate_onehot[gate_type] = 1
            
            # Fanin/fanout (0-4)
            fanin = np.random.randint(0, 5)
            fanout = np.random.randint(0, 5)
            
            # Depth in circuit (normalized)
            depth = i / num_nodes
            
            features = np.concatenate([gate_onehot, [fanin, fanout, depth]])
            node_features.append(features)
        
        node_features = np.array(node_features)
        
        # Generate edges (random DAG)
        edge_index = []
        edge_attr = []
        
        for i in range(num_nodes):
            # Connect to 1-3 previous nodes
            num_connections = np.random.randint(0, min(4, i + 1))
            targets = np.random.choice(i, size=num_connections, replace=False)
            
            for target in targets:
                edge_index.append([i, target])
                
                # Edge features: connection type + weight
                conn_type = np.random.randint(0, 3)
                conn_onehot = np.zeros(3)
                conn_onehot[conn_type] = 1
                weight = np.random.random()
                
                edge_attr.append(np.concatenate([conn_onehot, [weight]]))
        
        edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))
        edge_attr = np.array(edge_attr) if edge_attr else np.zeros((0, 4))
        
        return {
            'num_nodes': num_nodes,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'gate_types': gate_types,
        }
    
    def find_optimal_parameters(self, circuit: Dict) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"
        Find optimal parameters for circuit using Rust crypto library.
        
        Args:
            circuit: Circuit dictionary from generate_random_circuit
        
        Returns:
            Tuple of (optimal_params, metrics)
            - optimal_params: [n, q, sigma] (normalized)
            - metrics: [proof_size, prove_time, verify_time] (normalized)
        \"\"\"
        # Try different parameter combinations
        # Use Rust library to generate proofs
        # Measure size and time
        # Return best parameters
        
        # For now, use heuristic based on circuit size
        num_nodes = circuit['num_nodes']
        
        # Heuristic: larger circuits need larger parameters
        if num_nodes < 50:
            n, q, sigma = 512, 12289, 3.2
        elif num_nodes < 200:
            n, q, sigma = 1024, 40961, 3.2
        else:
            n, q, sigma = 2048, 65537, 3.2
        
        # Normalize parameters for neural network
        # n: 256-4096 -> 0-1
        # q: 4096-131072 -> 0-1
        # sigma: 2.0-5.0 -> 0-1
        n_norm = (n - 256) / (4096 - 256)
        q_norm = (q - 4096) / (131072 - 4096)
        sigma_norm = (sigma - 2.0) / (5.0 - 2.0)
        
        params = np.array([n_norm, q_norm, sigma_norm])
        
        # Simulate metrics (would come from actual proof generation)
        proof_size = num_nodes * 16  # bytes
        prove_time = num_nodes * 0.1  # ms
        verify_time = num_nodes * 0.05  # ms
        
        # Normalize metrics
        # size: 0-100KB -> 0-1
        # times: 0-1000ms -> 0-1
        size_norm = proof_size / 100000
        prove_norm = prove_time / 1000
        verify_norm = verify_time / 1000
        
        metrics = np.array([size_norm, prove_norm, verify_norm])
        
        return params, metrics
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        split: str = "train",
    ):
        \"\"\"
        Generate full dataset and save to disk.
        
        Args:
            num_samples: Number of circuits to generate
            output_dir: Output directory
            split: Dataset split name
        \"\"\"
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file
        with h5py.File(output_path / "index.h5", 'w') as f:
            f.create_dataset('circuit_ids', data=np.arange(num_samples))
        
        # Generate circuits
        for i in range(num_samples):
            circuit = self.generate_random_circuit()
            params, metrics = self.find_optimal_parameters(circuit)
            
            # Save circuit
            with h5py.File(output_path / f"circuit_{i}.h5", 'w') as f:
                f.create_dataset('node_features', data=circuit['node_features'])
                f.create_dataset('edge_index', data=circuit['edge_index'])
                f.create_dataset('edge_attr', data=circuit['edge_attr'])
                f.create_dataset('optimal_params', data=params)
                f.create_dataset('metrics', data=metrics)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} circuits")

def collate_fn(batch: List[Data]) -> Batch:
    \"\"\"
    Custom collate function for batching circuits.
    
    Args:
        batch: List of PyG Data objects
    
    Returns:
        Batched Data object
    \"\"\"
    return Batch.from_data_list(batch)
```

### 2. Data Generation Script

```python
# scripts/generate_dataset.py

from nexuszero_optimizer.training.dataset import ProofCircuitGenerator
from nexuszero_optimizer.utils.crypto_bridge import CryptoBridge

def main():
    # Initialize generator
    crypto_bridge = CryptoBridge()
    generator = ProofCircuitGenerator(
        min_nodes=10,
        max_nodes=1000,
        crypto_bridge=crypto_bridge,
    )
    
    # Generate datasets
    print("Generating training set...")
    generator.generate_dataset(
        num_samples=10000,
        output_dir="data",
        split="train",
    )
    
    print("Generating validation set...")
    generator.generate_dataset(
        num_samples=2000,
        output_dir="data",
        split="val",
    )
    
    print("Generating test set...")
    generator.generate_dataset(
        num_samples=2000,
        output_dir="data",
        split="test",
    )
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
```

### 3. DataLoader Setup

```python
# src/nexuszero_optimizer/training/dataloader.py

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    \"\"\"Create train/val/test dataloaders.\"\"\"
    
    train_dataset = ProofCircuitDataset(data_dir, split="train")
    val_dataset = ProofCircuitDataset(data_dir, split="val")
    test_dataset = ProofCircuitDataset(data_dir, split="test")
    
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader
```

### 4. Unit Tests

```python
# tests/test_dataset.py

import pytest
from nexuszero_optimizer.training.dataset import (
    ProofCircuitGenerator,
    ProofCircuitDataset,
)

def test_circuit_generation():
    generator = ProofCircuitGenerator(min_nodes=10, max_nodes=100)
    circuit = generator.generate_random_circuit()
    
    assert 'num_nodes' in circuit
    assert 10 <= circuit['num_nodes'] <= 100
    assert circuit['node_features'].shape[0] == circuit['num_nodes']

def test_parameter_finding():
    generator = ProofCircuitGenerator()
    circuit = generator.generate_random_circuit()
    
    params, metrics = generator.find_optimal_parameters(circuit)
    
    assert params.shape == (3,)  # n, q, sigma
    assert metrics.shape == (3,)  # size, prove_time, verify_time
    assert 0 <= params.min() and params.max() <= 1  # Normalized

def test_dataset_loading():
    # Generate small test dataset
    generator = ProofCircuitGenerator(min_nodes=5, max_nodes=20)
    generator.generate_dataset(num_samples=10, output_dir="test_data", split="test")
    
    # Load dataset
    dataset = ProofCircuitDataset("test_data", split="test")
    
    assert len(dataset) == 10
    
    # Test __getitem__
    data = dataset[0]
    assert hasattr(data, 'x')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'y')
    assert hasattr(data, 'metrics')
```

## Usage

```python
# Generate training data
python scripts/generate_dataset.py

# Load and inspect
from nexuszero_optimizer.training.dataset import ProofCircuitDataset
dataset = ProofCircuitDataset("data", split="train")
sample = dataset[0]
print(f"Circuit with {sample.x.shape[0]} nodes")
print(f"Target parameters: {sample.y}")
print(f"Metrics: {sample.metrics}")
```

Implement complete data pipeline with dataset generation and loading.
```

---

## 🧠 DAY 3-4: GNN ARCHITECTURE

### Prompt 2.1: Graph Neural Network Model

```
Implement a Graph Neural Network for learning optimal proof parameters from circuit structure.

## Architecture Overview

The GNN takes a proof circuit as input and predicts optimal cryptographic parameters:
- **Input:** Circuit graph (gates as nodes, wires as edges)
- **Processing:** Multiple GNN layers with attention
- **Output:** Parameters (n, q, σ) + performance estimates

## Implementation

### 1. Base GNN Model (src/nexuszero_optimizer/models/gnn.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from typing import Tuple

class ProofOptimizationGNN(nn.Module):
    \"\"\"
    Graph Neural Network for proof parameter optimization.
    
    Architecture:
    1. Node feature embedding
    2. Multiple GAT (Graph Attention) layers
    3. Global pooling
    4. MLP decoder for parameters
    5. Separate heads for params and metrics
    \"\"\"
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Use edge features in attention
            conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True if i < num_layers - 1 else False,
            )
            self.convs.append(conv)
            
            # Layer normalization
            norm = nn.LayerNorm(hidden_dim)
            self.norms.append(norm)
        
        # Global pooling
        # Combine mean and add pooling
        self.pool_dim = hidden_dim * 2
        
        # Parameter prediction head
        self.param_mlp = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # n, q, sigma
            nn.Sigmoid(),  # Output in [0, 1] (normalized)
        )
        
        # Performance metrics prediction head
        self.metrics_mlp = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # proof_size, prove_time, verify_time
            nn.Sigmoid(),  # Output in [0, 1] (normalized)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Tuple of (parameters, metrics)
            - parameters: [batch_size, 3] - predicted (n, q, sigma)
            - metrics: [batch_size, 3] - predicted (size, prove_time, verify_time)
        \"\"\"
        # Embed inputs
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        # Apply GAT layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.gelu(x)
            
            # Residual connection (if dimensions match)
            if x_in.shape == x.shape:
                x = x + x_in
        
        # Global pooling (combine mean and add)
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_global = torch.cat([x_mean, x_add], dim=1)
        
        # Predict parameters and metrics
        params = self.param_mlp(x_global)
        metrics = self.metrics_mlp(x_global)
        
        return params, metrics
    
    def predict_parameters(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        \"\"\"
        Predict parameters and denormalize to actual values.
        
        Returns:
            Dictionary with:
            - n: Lattice dimension
            - q: Modulus
            - sigma: Error distribution parameter
            - estimated_proof_size: bytes
            - estimated_prove_time: milliseconds
            - estimated_verify_time: milliseconds
        \"\"\"
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        self.eval()
        with torch.no_grad():
            params_norm, metrics_norm = self.forward(x, edge_index, edge_attr, batch)
        
        # Denormalize parameters
        params = params_norm[0].cpu().numpy()
        n = int(256 + params[0] * (4096 - 256))
        q = int(4096 + params[1] * (131072 - 4096))
        sigma = 2.0 + params[2] * (5.0 - 2.0)
        
        # Denormalize metrics
        metrics = metrics_norm[0].cpu().numpy()
        proof_size = int(metrics[0] * 100000)  # bytes
        prove_time = metrics[1] * 1000  # ms
        verify_time = metrics[2] * 1000  # ms
        
        return {
            'n': n,
            'q': q,
            'sigma': sigma,
            'estimated_proof_size': proof_size,
            'estimated_prove_time': prove_time,
            'estimated_verify_time': verify_time,
        }
```

### 2. Attention Mechanisms (src/nexuszero_optimizer/models/attention.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EdgeAwareGATConv(MessagePassing):
    \"\"\"
    Graph Attention Network layer with explicit edge features.
    
    Standard GAT only uses node features for attention.
    This variant also considers edge features.
    \"\"\"
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations
        self.lin_node = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        
        # Attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        Forward pass with edge-aware attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels * heads]
        \"\"\"
        # Transform node features
        x = self.lin_node(x).view(-1, self.heads, self.out_channels)
        
        # Transform edge features
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        Compute messages with edge-aware attention.
        
        Args:
            x_i: Target node features
            x_j: Source node features
            edge_attr: Edge features
            index: Target node indices
        
        Returns:
            Weighted messages
        \"\"\"
        # Combine node and edge features for attention
        # attention = f(x_i, x_j, edge_attr)
        
        # Attention coefficients
        alpha = (torch.cat([x_i, x_j + edge_attr], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted messages
        return (x_j + edge_attr) * alpha.unsqueeze(-1)
```

### 3. Model Tests

```python
# tests/test_models.py

import pytest
import torch
from torch_geometric.data import Data, Batch
from nexuszero_optimizer.models.gnn import ProofOptimizationGNN

def test_gnn_forward():
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
    )
    
    # Create sample data
    x = torch.randn(20, 10)  # 20 nodes
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 4)
    batch = torch.zeros(20, dtype=torch.long)
    
    # Forward pass
    params, metrics = model(x, edge_index, edge_attr, batch)
    
    assert params.shape == (1, 3)  # batch_size=1, 3 parameters
    assert metrics.shape == (1, 3)  # batch_size=1, 3 metrics
    
    # Check outputs are in [0, 1] (normalized)
    assert (params >= 0).all() and (params <= 1).all()
    assert (metrics >= 0).all() and (metrics <= 1).all()

def test_gnn_batching():
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=64,
    )
    
    # Create batch of 3 graphs
    data_list = []
    for _ in range(3):
        x = torch.randn(15, 10)
        edge_index = torch.randint(0, 15, (2, 20))
        edge_attr = torch.randn(20, 4)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    
    batch = Batch.from_data_list(data_list)
    
    # Forward pass
    params, metrics = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
    assert params.shape == (3, 3)  # batch_size=3
    assert metrics.shape == (3, 3)

def test_parameter_prediction():
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=64,
    )
    
    x = torch.randn(20, 10)
    edge_index = torch.randint(0, 20, (2, 30))
    edge_attr = torch.randn(30, 4)
    
    result = model.predict_parameters(x, edge_index, edge_attr)
    
    assert 'n' in result
    assert 'q' in result
    assert 'sigma' in result
    assert 256 <= result['n'] <= 4096
    assert 4096 <= result['q'] <= 131072
    assert 2.0 <= result['sigma'] <= 5.0
```

Implement complete GNN architecture with attention and parameter prediction.
```

[Due to length, continuing in next message with Days 5-7...]

---

**Created:** November 20, 2024  
**Purpose:** Complete Copilot prompts for Week 2 Neural Optimizer (Part 1)  
**Status:** In progress - Days 1-4 complete
