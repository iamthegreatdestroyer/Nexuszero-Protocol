"""
Graph Neural Network for proof parameter optimization.

This module implements a GNN that takes proof circuit graphs as input
and predicts optimal cryptographic parameters (n, q, sigma).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from typing import Tuple, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProofOptimizationGNN(nn.Module):
    """
    Graph Neural Network for proof parameter optimization.
    
    Architecture:
    1. Node and edge feature embedding
    2. Multiple GAT (Graph Attention) layers with residual connections
    3. Global graph pooling (mean + add)
    4. Separate MLP heads for parameters and metrics prediction
    
    The model predicts:
    - Parameters: (n, q, sigma) normalized to [0, 1]
    - Metrics: (proof_size, prove_time, verify_time) normalized to [0, 1]
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize GNN model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input embedding layers
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)
        
        # GAT layers with layer normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Graph Attention Convolution
            conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True,  # Concatenate heads (except last layer)
            )
            self.convs.append(conv)
            
            # Layer normalization for stability
            norm = nn.LayerNorm(hidden_dim)
            self.norms.append(norm)
        
        # Global pooling combines mean and sum
        self.pool_dim = hidden_dim * 2
        
        # Parameter prediction head (n, q, sigma)
        self.param_mlp = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # n, q, sigma
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Performance metrics prediction head (size, prove_time, verify_time)
        self.metrics_mlp = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # proof_size, prove_time, verify_time
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes] (which graph each node belongs to)
        
        Returns:
            Tuple of (parameters, metrics):
            - parameters: [batch_size, 3] - predicted (n, q, sigma) normalized
            - metrics: [batch_size, 3] - predicted (size, prove_time, verify_time) normalized
        """
        # Embed input features
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        # Apply GAT layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            
            # Graph attention convolution
            x = conv(x, edge_index, edge_attr=edge_attr)
            
            # Layer normalization
            x = norm(x)
            
            # Activation
            x = F.gelu(x)
            
            # Residual connection (if dimensions match)
            if x_in.shape == x.shape and i > 0:
                x = x + x_in
        
        # Global pooling (combine mean and add pooling)
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
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Predict parameters and denormalize to actual values.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Optional batch assignment (defaults to single graph)
        
        Returns:
            Dictionary with:
            - n: Lattice dimension (256-4096)
            - q: Modulus (4096-131072)
            - sigma: Error distribution parameter (2.0-5.0)
            - estimated_proof_size: bytes
            - estimated_prove_time: milliseconds
            - estimated_verify_time: milliseconds
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        self.eval()
        with torch.no_grad():
            params_norm, metrics_norm = self.forward(x, edge_index, edge_attr, batch)
        
        # Denormalize parameters (take first sample if batched)
        params = params_norm[0].cpu().numpy()
        n = int(256 + params[0] * (4096 - 256))
        q = int(4096 + params[1] * (131072 - 4096))
        sigma = 2.0 + params[2] * (5.0 - 2.0)
        
        # Round n to nearest power of 2
        n = 2 ** int(np.round(np.log2(n)))
        
        # Denormalize metrics
        metrics = metrics_norm[0].cpu().numpy()
        proof_size = int(metrics[0] * 100000)  # bytes
        prove_time = metrics[1] * 1000  # ms
        verify_time = metrics[2] * 1000  # ms
        
        return {
            'n': n,
            'q': q,
            'sigma': float(sigma),
            'estimated_proof_size': proof_size,
            'estimated_prove_time': float(prove_time),
            'estimated_verify_time': float(verify_time),
        }
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'node_feat_dim': self.node_feat_dim,
                'edge_feat_dim': self.edge_feat_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
            }
        }, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ProofOptimizationGNN':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {path}")
        return model
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing ProofOptimizationGNN...")
    
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
    )
    
    print(f"✓ Model created with {model.count_parameters():,} parameters")
    
    # Create sample data
    x = torch.randn(20, 10)  # 20 nodes
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    batch = torch.zeros(20, dtype=torch.long)
    
    # Forward pass
    params, metrics = model(x, edge_index, edge_attr, batch)
    
    print(f"✓ Forward pass successful")
    print(f"  Parameters shape: {params.shape}")
    print(f"  Metrics shape: {metrics.shape}")
    print(f"  Parameters range: [{params.min():.3f}, {params.max():.3f}]")
    
    # Test prediction
    result = model.predict_parameters(x, edge_index, edge_attr)
    print(f"✓ Prediction successful:")
    print(f"  n = {result['n']}")
    print(f"  q = {result['q']}")
    print(f"  σ = {result['sigma']:.2f}")
