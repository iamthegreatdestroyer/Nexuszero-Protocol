"""
GNN Integration Layer
Bridges the Supervisor Agent system with the PyTorch GNN optimizer.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger("GNNIntegration")


@dataclass
class CircuitGraph:
    """Represents a ZK circuit as a graph for GNN processing."""
    node_features: torch.Tensor  # [num_nodes, feature_dim]
    edge_index: torch.Tensor      # [2, num_edges]
    edge_features: torch.Tensor   # [num_edges, edge_feature_dim]
    constraint_types: torch.Tensor  # Node type labels
    
    def to(self, device: torch.device) -> 'CircuitGraph':
        return CircuitGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            constraint_types=self.constraint_types.to(device)
        )


class CircuitEncoder(nn.Module):
    """Encodes circuit constraints into graph node features."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ProofOptimizationGNN(nn.Module):
    """
    Graph Neural Network for proof optimization parameter prediction.
    Takes circuit structure and outputs optimal prover parameters.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 32  # Number of optimization parameters
    ):
        super().__init__()
        
        self.node_encoder = CircuitEncoder(node_feature_dim, hidden_dim, hidden_dim)
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, edge_feature_dim) for _ in range(num_layers)
        ])
        
        # Global pooling and output
        self.global_pool = GlobalAttentionPool(hidden_dim)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, graph: CircuitGraph) -> torch.Tensor:
        # Encode node features
        h = self.node_encoder(graph.node_features)
        
        # Message passing
        for layer in self.gnn_layers:
            h = layer(h, graph.edge_index, graph.edge_features)
        
        # Global pooling
        graph_embedding = self.global_pool(h)
        
        # Predict optimization parameters
        params = self.output_head(graph_embedding)
        
        return params


class GNNLayer(nn.Module):
    """Single GNN message passing layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        
        # Compute messages
        src_features = h[src]
        dst_features = h[dst]
        messages = self.message_mlp(
            torch.cat([src_features, dst_features, edge_features], dim=-1)
        )
        
        # Aggregate messages (mean aggregation)
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        counts = torch.zeros(h.size(0), 1, device=h.device)
        counts.index_add_(0, dst, torch.ones(messages.size(0), 1, device=h.device))
        aggregated = aggregated / (counts + 1e-8)
        
        # Update node features
        h_new = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        h_new = self.norm(h + h_new)  # Residual connection
        
        return h_new


class GlobalAttentionPool(nn.Module):
    """Attention-based global graph pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(h), dim=0)
        
        # Weighted sum
        graph_embedding = (attention_weights * h).sum(dim=0)
        
        return graph_embedding


class GNNOptimizationOracle:
    """
    Oracle that uses GNN to predict optimal parameters for proof optimization. 
    Integrates with the Supervisor Agent system.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ProofOptimizationGNN().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
        # Parameter mapping: GNN output indices to parameter names
        self.param_mapping = {
            0: "fft_domain_size_exp",
            1: "num_threads",
            2: "msm_window_size",
            3: "parallel_degree",
            4: "memory_pool_mb",
            5: "batch_size",
            6: "use_cuda",
            7: "enable_caching"
        }
    
    def load_model(self, path: str):
        """Load pretrained model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded GNN model from {path}")
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved GNN model to {path}")
    
    def circuit_to_graph(self, constraints: Dict[str, Any]) -> CircuitGraph:
        """Convert circuit constraints to graph representation."""
        num_constraints = constraints.get("count", 100)
        num_variables = constraints.get("variables", num_constraints * 3)
        
        # Create node features (simplified representation)
        node_features = torch.randn(num_constraints, 64)
        
        # Create sparse connectivity (each constraint connects ~3 variables)
        num_edges = min(num_constraints * 3, 10000)
        src = torch.randint(0, num_constraints, (num_edges,))
        dst = torch.randint(0, num_constraints, (num_edges,))
        edge_index = torch.stack([src, dst])
        
        # Edge features
        edge_features = torch.randn(num_edges, 16)
        
        # Constraint types
        constraint_types = torch.randint(0, 4, (num_constraints,))
        
        return CircuitGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            constraint_types=constraint_types
        )
    
    @torch.no_grad()
    def predict_optimal_params(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal optimization parameters for given constraints."""
        
        graph = self.circuit_to_graph(constraints).to(self.device)
        
        # Get GNN predictions
        raw_output = self.model(graph)
        
        # Convert to parameter dictionary
        params = {}
        
        # FFT domain size (power of 2 between 14 and 24)
        params["fft_domain_size"] = 2 ** int(14 + torch.sigmoid(raw_output[0]) * 10)
        
        # Number of threads (1-32)
        params["num_threads"] = int(1 + torch.sigmoid(raw_output[1]) * 31)
        
        # MSM window size (8-20)
        params["msm_window_size"] = int(8 + torch.sigmoid(raw_output[2]) * 12)
        
        # Parallel degree (1-16)
        params["parallel_degree"] = int(1 + torch.sigmoid(raw_output[3]) * 15)
        
        # Memory pool (256-4096 MB)
        params["memory_pool_mb"] = int(256 + torch.sigmoid(raw_output[4]) * 3840)
        
        # Batch size (1-64)
        params["batch_size"] = int(1 + torch.sigmoid(raw_output[5]) * 63)
        
        # Boolean parameters
        params["use_cuda"] = bool(raw_output[6] > 0)
        params["enable_caching"] = bool(raw_output[7] > 0)
        
        return params
    
    def train_on_experience(
        self,
        experiences: List[Tuple[Dict[str, Any], Dict[str, Any], float]]
    ):
        """
        Train the GNN on optimization experiences.
        experiences: List of (constraints, params_used, improvement) tuples
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for constraints, params_used, improvement in experiences:
            graph = self.circuit_to_graph(constraints).to(self.device)
            
            # Create target tensor from successful parameters
            target = self._params_to_tensor(params_used)
            
            # Weight by improvement (better results = stronger signal)
            weight = improvement / 100.0
            
            # Forward pass
            predicted = self.model(graph)
            
            # MSE loss weighted by improvement
            loss = weight * nn.functional.mse_loss(predicted, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        logger.info(f"Trained on {len(experiences)} experiences")
    
    def _params_to_tensor(self, params: Dict[str, Any]) -> torch.Tensor:
        """Convert parameter dict to target tensor."""
        target = torch.zeros(32, device=self.device)
        
        # Inverse of the conversion in predict_optimal_params
        if "fft_domain_size" in params:
            exp = np.log2(params["fft_domain_size"])
            target[0] = (exp - 14) / 10
        
        if "num_threads" in params:
            target[1] = (params["num_threads"] - 1) / 31
        
        if "msm_window_size" in params:
            target[2] = (params["msm_window_size"] - 8) / 12
        
        if "parallel_degree" in params:
            target[3] = (params["parallel_degree"] - 1) / 15
        
        return target


def create_gnn_oracle(model_path: Optional[str] = None) -> GNNOptimizationOracle:
    """Factory function for GNN oracle."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GNNOptimizationOracle(model_path=model_path, device=device)
