"""
Unit tests for GNN models.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from nexuszero_optimizer.models.gnn import ProofOptimizationGNN
from nexuszero_optimizer.models.attention import EdgeAwareGATConv


class TestProofOptimizationGNN:
    """Test suite for ProofOptimizationGNN model."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return ProofOptimizationGNN(
            node_feat_dim=10,
            edge_feat_dim=4,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample graph data."""
        x = torch.randn(20, 10)  # 20 nodes, 10 features
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
            dtype=torch.long
        )
        edge_attr = torch.randn(5, 4)  # 5 edges, 4 features
        batch = torch.zeros(20, dtype=torch.long)
        
        return x, edge_index, edge_attr, batch
    
    def test_model_creation(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.node_feat_dim == 10
        assert model.edge_feat_dim == 4
        assert model.hidden_dim == 64
        assert model.num_layers == 3
        
        # Check parameter count
        param_count = model.count_parameters()
        assert param_count > 0
        print(f"Model has {param_count:,} parameters")
    
    def test_forward_pass(self, model, sample_data):
        """Test forward pass through model."""
        x, edge_index, edge_attr, batch = sample_data
        
        params, metrics = model(x, edge_index, edge_attr, batch)
        
        # Check output shapes
        assert params.shape == (1, 3)  # batch_size=1, 3 parameters
        assert metrics.shape == (1, 3)  # batch_size=1, 3 metrics
        
        # Check output ranges (should be normalized to [0, 1])
        assert (params >= 0).all() and (params <= 1).all()
        assert (metrics >= 0).all() and (metrics <= 1).all()
    
    def test_batch_processing(self, model):
        """Test processing multiple graphs in batch."""
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
        
        # Check batch size
        assert params.shape == (3, 3)  # 3 graphs, 3 parameters each
        assert metrics.shape == (3, 3)  # 3 graphs, 3 metrics each
    
    def test_parameter_prediction(self, model, sample_data):
        """Test parameter prediction with denormalization."""
        x, edge_index, edge_attr, _ = sample_data
        
        result = model.predict_parameters(x, edge_index, edge_attr)
        
        # Check required keys
        assert 'n' in result
        assert 'q' in result
        assert 'sigma' in result
        assert 'estimated_proof_size' in result
        assert 'estimated_prove_time' in result
        assert 'estimated_verify_time' in result
        
        # Check parameter ranges
        assert 256 <= result['n'] <= 4096
        assert 4096 <= result['q'] <= 131072
        assert 2.0 <= result['sigma'] <= 5.0
        
        # Check n is power of 2
        n = result['n']
        assert (n & (n - 1)) == 0  # Power of 2 check
    
    def test_model_save_load(self, model, tmp_path):
        """Test saving and loading model."""
        # Save model
        save_path = tmp_path / "test_model.pt"
        model.save(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = ProofOptimizationGNN.load(str(save_path))
        
        # Check configuration matches
        assert loaded_model.node_feat_dim == model.node_feat_dim
        assert loaded_model.edge_feat_dim == model.edge_feat_dim
        assert loaded_model.hidden_dim == model.hidden_dim
        
        # Check outputs match - both models in eval mode
        model.eval()
        loaded_model.eval()
        
        x = torch.randn(10, 10)
        edge_index = torch.randint(0, 10, (2, 15))
        edge_attr = torch.randn(15, 4)
        batch = torch.zeros(10, dtype=torch.long)
        
        with torch.no_grad():
            params1, metrics1 = model(x, edge_index, edge_attr, batch)
            params2, metrics2 = loaded_model(x, edge_index, edge_attr, batch)
        
        torch.testing.assert_close(params1, params2)
        torch.testing.assert_close(metrics1, metrics2)
    
    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through model."""
        model.train()
        x, edge_index, edge_attr, batch = sample_data
        
        # Forward pass
        params, metrics = model(x, edge_index, edge_attr, batch)
        
        # Compute dummy loss
        loss = params.sum() + metrics.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestEdgeAwareGATConv:
    """Test suite for EdgeAwareGATConv layer."""
    
    def test_conv_layer(self):
        """Test edge-aware convolution layer."""
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
        
        # Check output shape
        # 4 heads * 16 out_channels = 64
        assert out.shape == (10, 64)
    
    def test_conv_no_concat(self):
        """Test convolution without concatenating heads."""
        conv = EdgeAwareGATConv(
            in_channels=64,
            out_channels=16,
            edge_dim=4,
            heads=4,
            concat=False,
        )
        
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 4)
        
        out = conv(x, edge_index, edge_attr)
        
        # When concat=False, output is averaged across heads
        assert out.shape == (10, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
