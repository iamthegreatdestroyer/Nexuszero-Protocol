"""
Advanced attention mechanisms for Graph Neural Networks.

This module provides edge-aware attention that considers both node
and edge features when computing attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional


class EdgeAwareGATConv(MessagePassing):
    """
    Graph Attention Network layer with explicit edge features.
    
    Standard GAT only uses node features for attention computation.
    This variant incorporates edge features into the attention mechanism,
    allowing the model to learn which connections are more important
    based on both the nodes they connect and the edge properties.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        """
        Initialize edge-aware GAT layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            edge_dim: Edge feature dimension
            heads: Number of attention heads
            concat: If True, concatenate head outputs; else average
            dropout: Dropout probability for attention weights
            negative_slope: LeakyReLU negative slope
            bias: Whether to use bias
        """
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations for nodes and edges
        self.lin_node = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        # Attention mechanism
        # Computes attention based on: [node_i || node_j || edge_ij]
        self.att = nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with edge-aware attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels * heads] or
            [num_nodes, out_channels] if concat=False
        """
        # Transform node features: [N, F] -> [N, heads * out_channels]
        x = self.lin_node(x).view(-1, self.heads, self.out_channels)
        
        # Transform edge features: [E, edge_dim] -> [E, heads * out_channels]
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Propagate messages with edge-aware attention
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=None,
        )
        
        # Concatenate or average attention heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        """
        Compute messages with edge-aware attention.
        
        Args:
            x_i: Target node features [E, heads, out_channels]
            x_j: Source node features [E, heads, out_channels]
            edge_attr: Edge features [E, heads, out_channels]
            index: Target node indices [E]
            size_i: Number of target nodes
        
        Returns:
            Weighted messages [E, heads, out_channels]
        """
        # Concatenate node and edge features for attention
        # Shape: [E, heads, 3 * out_channels]
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute attention logits
        # Shape: [E, heads]
        alpha = (features * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights using softmax
        alpha = softmax(alpha, index, num_nodes=size_i)
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Combine source node and edge features with attention
        # This allows the model to learn from both node and edge information
        message = (x_j + edge_attr) * alpha.unsqueeze(-1)
        
        return message
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'edge_dim={self.edge_dim})')


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that captures both local and global patterns.
    
    Combines attention at different scales to capture:
    - Local patterns: Immediate neighbors
    - Global patterns: Long-range dependencies
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-scale attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_scales: Number of attention scales
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Attention at different scales
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_scales)
        ])
        
        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale attention.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
        
        Returns:
            Output features [batch, seq_len, hidden_dim]
        """
        outputs = []
        
        # Apply attention at each scale
        for scale_attn in self.scale_attentions:
            out, _ = scale_attn(x, x, x)
            outputs.append(out)
        
        # Combine scales with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, outputs))
        
        # Project output
        output = self.output_proj(combined)
        
        return output


if __name__ == "__main__":
    # Test edge-aware attention
    print("Testing EdgeAwareGATConv...")
    
    conv = EdgeAwareGATConv(
        in_channels=64,
        out_channels=16,
        edge_dim=4,
        heads=4,
    )
    
    # Create sample data
    x = torch.randn(10, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 4)
    
    # Forward pass
    out = conv(x, edge_index, edge_attr)
    
    print(f"✓ Edge-aware attention successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: [10, 64] (4 heads * 16 out_channels)")
