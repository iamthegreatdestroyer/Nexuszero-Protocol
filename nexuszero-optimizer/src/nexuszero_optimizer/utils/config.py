"""Configuration management for nexuszero-optimizer."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """GNN model configuration."""
    
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": self.activation,
        }


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 500
    early_stopping_patience: int = 10
    scheduler_type: str = "plateau"  # one of ['none','plateau','cosine']
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    checkpoint_best_only: bool = True
    tensorboard_enabled: bool = False
    aux_metrics_loss_weight: float = 0.0
    # WandB / experiment tracking
    wandb_enabled: bool = False
    wandb_project: str = "nexuszero-optimizer"
    wandb_entity: str = ""
    wandb_run_name: str = "run"
    wandb_tags: Optional[Any] = None  # list-like
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "early_stopping_patience": self.early_stopping_patience,
            "scheduler_type": self.scheduler_type,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "checkpoint_best_only": self.checkpoint_best_only,
            "tensorboard_enabled": self.tensorboard_enabled,
            "aux_metrics_loss_weight": self.aux_metrics_loss_weight,
            "wandb_enabled": self.wandb_enabled,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "wandb_tags": self.wandb_tags,
        }


@dataclass
class OptimizationConfig:
    """Proof optimization configuration."""
    
    security_level: int = 128  # bits
    max_proof_size: int = 10000  # bytes
    target_verify_time: float = 50.0  # milliseconds
    
    # Parameter ranges for normalization
    n_min: int = 256
    n_max: int = 4096
    q_min: int = 4096
    q_max: int = 131072
    sigma_min: float = 2.0
    sigma_max: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "security_level": self.security_level,
            "max_proof_size": self.max_proof_size,
            "target_verify_time": self.target_verify_time,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
        }


@dataclass
class Config:
    """Main configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Hardware
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Config object
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            data_dir=data.get("data_dir", "data"),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints"),
            log_dir=data.get("log_dir", "logs"),
            device=data.get("device", "cuda"),
            num_workers=data.get("num_workers", 4),
            seed=data.get("seed", 42),
        )
    
    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        data = {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "optimization": self.optimization.to_dict(),
            "data_dir": self.data_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "device": self.device,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "optimization": self.optimization.to_dict(),
            "data_dir": self.data_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "device": self.device,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }


def create_default_config(path: str = "config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        path: Path to save configuration file
    """
    config = Config()
    config.to_yaml(path)
    print(f"Created default configuration at {path}")


if __name__ == "__main__":
    # Create default config
    create_default_config()
