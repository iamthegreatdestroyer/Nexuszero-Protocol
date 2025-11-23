"""
Unit tests for configuration management.
"""

import pytest
import yaml
from pathlib import Path
import tempfile

from nexuszero_optimizer.utils.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    OptimizationConfig,
)


class TestConfigs:
    """Test configuration dataclasses."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.hidden_dim == 256
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.activation == "gelu"
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 100
        assert config.warmup_steps == 1000
    
    def test_optimization_config_defaults(self):
        """Test OptimizationConfig default values."""
        config = OptimizationConfig()
        
        assert config.security_level == 128
        assert config.max_proof_size == 10000
        assert config.target_verify_time == 50.0
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(hidden_dim=128, num_layers=4)
        d = config.to_dict()
        
        assert d['hidden_dim'] == 128
        assert d['num_layers'] == 4
        assert 'dropout' in d


class TestMainConfig:
    """Test main Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        
        assert config.device == "cuda"
        assert config.num_workers == 4
    
    def test_config_to_yaml(self):
        """Test saving configuration to YAML."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.to_yaml(yaml_path)
            
            # Check file exists
            assert Path(yaml_path).exists()
            
            # Load and verify
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            
            assert 'model' in data
            assert 'training' in data
            assert 'optimization' in data
            assert data['device'] == 'cuda'
        
        finally:
            Path(yaml_path).unlink()
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        # Create test YAML
        yaml_data = """
model:
  hidden_dim: 128
  num_layers: 4
  num_heads: 4

training:
  batch_size: 64
  learning_rate: 0.001

optimization:
  security_level: 256

device: cpu
num_workers: 2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            
            # Check values were loaded correctly
            assert config.model.hidden_dim == 128
            assert config.model.num_layers == 4
            assert config.training.batch_size == 64
            assert config.training.learning_rate == 0.001
            assert config.optimization.security_level == 256
            assert config.device == "cpu"
            assert config.num_workers == 2
        
        finally:
            Path(yaml_path).unlink()
    
    def test_config_roundtrip(self):
        """Test saving and loading config (roundtrip)."""
        original = Config(
            model=ModelConfig(hidden_dim=128),
            training=TrainingConfig(batch_size=64),
            device="cpu",
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save
            original.to_yaml(yaml_path)
            
            # Load
            loaded = Config.from_yaml(yaml_path)
            
            # Compare
            assert loaded.model.hidden_dim == 128
            assert loaded.training.batch_size == 64
            assert loaded.device == "cpu"
        
        finally:
            Path(yaml_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
