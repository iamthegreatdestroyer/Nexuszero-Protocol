"""
Tests for the Advanced GNN model.
"""

import pytest
import torch
import numpy as np

from nexuszero_optimizer.models.gnn_advanced import AdvancedGNNOptimizer


class TestAdvancedGNNOptimizer:
    """Test suite for Advanced GNN model."""
    
    def setup_method(self):
        """Setup for each test."""
        self.model = AdvancedGNNOptimizer(
            node_feat_dim=10,
            edge_feat_dim=4,
            hidden_dim=128,  # Smaller for testing
            num_layers=4,
            num_heads=4,
            dropout=0.1,
        )
    
    def test_model_creation(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.node_feat_dim == 10
        assert self.model.edge_feat_dim == 4
        assert self.model.hidden_dim == 128
        assert self.model.num_layers == 4
        assert self.model.num_heads == 4
        assert self.model.dropout == 0.1
    
    def test_parameter_count(self):
        """Test parameter counting."""
        num_params = self.model.count_parameters()
        assert num_params > 0
        print(f"Model has {num_params:,} parameters")
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create sample data
        x = torch.randn(20, 10)  # 20 nodes
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        batch = torch.zeros(20, dtype=torch.long)
        
        # Forward pass
        params, metrics = self.model(x, edge_index, edge_attr, batch)
        
        # Check shapes
        assert params.shape == (1, 3)  # 1 graph, 3 parameters
        assert metrics.shape == (1, 3)  # 1 graph, 3 metrics
        
        # Check range [0, 1]
        assert (params >= 0).all() and (params <= 1).all()
        assert (metrics >= 0).all() and (metrics <= 1).all()
    
    def test_batched_forward(self):
        """Test batched forward pass."""
        # Create 2 graphs
        x = torch.randn(30, 10)  # 30 nodes total
        edge_index = torch.tensor([[0, 1, 2, 10, 11], [1, 2, 3, 11, 12]], dtype=torch.long)
        edge_attr = torch.randn(5, 4)
        batch = torch.tensor([0]*15 + [1]*15, dtype=torch.long)  # 2 graphs
        
        params, metrics = self.model(x, edge_index, edge_attr, batch)
        
        assert params.shape == (2, 3)  # 2 graphs
        assert metrics.shape == (2, 3)
    
    def test_predict_parameters(self):
        """Test parameter prediction with denormalization."""
        x = torch.randn(20, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        
        result = self.model.predict_parameters(x, edge_index, edge_attr, denormalize=True)
        
        # Check keys
        assert 'n' in result
        assert 'q' in result
        assert 'sigma' in result
        assert 'estimated_proof_size' in result
        assert 'estimated_prove_time' in result
        assert 'estimated_verify_time' in result
        
        # Check ranges
        assert 64 <= result['n'] <= 2048
        assert 4096 <= result['q'] <= 131072
        assert 1.0 <= result['sigma'] <= 8.0
        
        # Check n is power of 2
        assert np.log2(result['n']) == int(np.log2(result['n']))
    
    def test_predict_parameters_normalized(self):
        """Test parameter prediction without denormalization."""
        x = torch.randn(20, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        
        result = self.model.predict_parameters(x, edge_index, edge_attr, denormalize=False)
        
        # Should return normalized values [0, 1]
        assert 0.0 <= result['n'] <= 1.0
        assert 0.0 <= result['q'] <= 1.0
        assert 0.0 <= result['sigma'] <= 1.0
    
    def test_save_and_load(self, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "model.pt"
        self.model.save(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = AdvancedGNNOptimizer.load(str(save_path))
        
        # Check config matches
        assert loaded_model.node_feat_dim == self.model.node_feat_dim
        assert loaded_model.hidden_dim == self.model.hidden_dim
        assert loaded_model.num_layers == self.model.num_layers
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(20, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        batch = torch.zeros(20, dtype=torch.long)
        
        # Forward pass
        params, metrics = self.model(x, edge_index, edge_attr, batch)
        
        # Compute loss
        target = torch.randn(1, 3)
        loss = torch.nn.functional.mse_loss(params, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model"
    
    def test_default_architecture(self):
        """Test model with default architecture from issue requirements."""
        model = AdvancedGNNOptimizer(
            node_feat_dim=10,
            edge_feat_dim=4,
            hidden_dim=256,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
        )
        
        assert model.hidden_dim == 256
        assert model.num_layers == 8
        assert model.num_heads == 8
        
        # Test forward pass
        x = torch.randn(20, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        batch = torch.zeros(20, dtype=torch.long)
        
        params, metrics = model(x, edge_index, edge_attr, batch)
        
        assert params.shape == (1, 3)
        assert metrics.shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
