"""
Unit tests for dataset and data generation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from nexuszero_optimizer.training.dataset import (
    ProofCircuitGenerator,
    ProofCircuitDataset,
    collate_fn,
)


class TestProofCircuitGenerator:
    """Test suite for circuit generation."""
    
    def setup_method(self):
        """Setup for each test."""
        self.generator = ProofCircuitGenerator(
            min_nodes=10,
            max_nodes=100,
            seed=42,
        )
    
    def test_circuit_generation(self):
        """Test basic circuit generation."""
        circuit = self.generator.generate_random_circuit()
        
        assert 'num_nodes' in circuit
        assert 'node_features' in circuit
        assert 'edge_index' in circuit
        assert 'edge_attr' in circuit
        
        # Check bounds
        assert 10 <= circuit['num_nodes'] <= 100
        
        # Check shapes
        num_nodes = circuit['num_nodes']
        assert circuit['node_features'].shape[0] == num_nodes
        assert circuit['node_features'].shape[1] == 10  # 7 gate types + 3 features
        
        # Check edge structure (DAG property)
        if circuit['edge_index'].shape[1] > 0:
            # All edges should go from higher to lower index (DAG)
            edge_from = circuit['edge_index'][0]
            edge_to = circuit['edge_index'][1]
            assert np.all(edge_from > edge_to)
    
    def test_parameter_finding(self):
        """Test optimal parameter finding."""
        circuit = self.generator.generate_random_circuit()
        params, metrics, security_level = self.generator.find_optimal_parameters(circuit)
        
        # Check parameter shape and range
        assert params.shape == (3,)  # n, q, sigma
        assert 0 <= params.min() and params.max() <= 1  # Normalized
        
        # Check metrics shape and range
        assert metrics.shape == (3,)  # size, prove_time, verify_time
        assert 0 <= metrics.min() and metrics.max() <= 1  # Normalized
        
        # Check security level
        assert security_level in [128, 192, 256]
    
    def test_dataset_generation(self):
        """Test full dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate small dataset
            self.generator.generate_dataset(
                num_samples=5,
                output_dir=tmpdir,
                split="test",
                show_progress=False,
            )
            
            # Check files exist
            test_dir = Path(tmpdir) / "test"
            assert (test_dir / "index.h5").exists()
            
            for i in range(5):
                assert (test_dir / f"circuit_{i}.h5").exists()
    
    def test_reproducibility(self):
        """Test that same seed produces reproducible circuits."""
        # Test reproducibility with explicit re-seeding before each generation
        np.random.seed(42)
        gen1 = ProofCircuitGenerator(min_nodes=10, max_nodes=100, seed=42)
        circuit1 = gen1.generate_random_circuit()
        
        np.random.seed(42)
        gen2 = ProofCircuitGenerator(min_nodes=10, max_nodes=100, seed=42)
        circuit2 = gen2.generate_random_circuit()
        
        # With same seed and same state, should produce same circuit
        assert circuit1['num_nodes'] == circuit2['num_nodes']
        np.testing.assert_array_equal(
            circuit1['node_features'],
            circuit2['node_features']
        )


class TestProofCircuitDataset:
    """Test suite for dataset loading."""
    
    @pytest.fixture
    def dataset_dir(self):
        """Create temporary dataset for testing."""
        tmpdir = tempfile.mkdtemp()
        
        generator = ProofCircuitGenerator(min_nodes=5, max_nodes=20, seed=42)
        generator.generate_dataset(
            num_samples=10,
            output_dir=tmpdir,
            split="test",
            show_progress=False,
        )
        
        yield tmpdir
        
        # Cleanup
        shutil.rmtree(tmpdir)
    
    def test_dataset_loading(self, dataset_dir):
        """Test dataset loading."""
        dataset = ProofCircuitDataset(dataset_dir, split="test")
        
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, dataset_dir):
        """Test getting individual samples."""
        dataset = ProofCircuitDataset(dataset_dir, split="test")
        data = dataset[0]
        
        # Check required attributes
        assert hasattr(data, 'x')  # Node features
        assert hasattr(data, 'edge_index')  # Edges
        assert hasattr(data, 'edge_attr')  # Edge features
        assert hasattr(data, 'y')  # Target parameters
        assert hasattr(data, 'metrics')  # Performance metrics
        
        # Check shapes
        assert data.y.shape == (3,)  # n, q, sigma
        assert data.metrics.shape == (3,)  # size, prove_time, verify_time
        
        # Check value ranges (normalized)
        assert 0 <= data.y.min() and data.y.max() <= 1
        assert 0 <= data.metrics.min() and data.metrics.max() <= 1
    
    def test_collate_function(self, dataset_dir):
        """Test batching of circuits."""
        dataset = ProofCircuitDataset(dataset_dir, split="test")
        
        # Get batch of samples
        samples = [dataset[i] for i in range(3)]
        batch = collate_fn(samples)
        
        # Check batch attributes
        assert hasattr(batch, 'batch')  # Batch assignment
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        
        # Check batch size - PyG concatenates target tensors
        # 3 graphs * 3 parameters each = 9 total
        assert batch.y.shape[0] == 9  # 3 graphs * 3 params
        assert batch.metrics.shape[0] == 9  # 3 graphs * 3 metrics


def test_dataset_missing_files():
    """Test error handling for missing dataset."""
    with pytest.raises(FileNotFoundError):
        ProofCircuitDataset("nonexistent_dir", split="train")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
