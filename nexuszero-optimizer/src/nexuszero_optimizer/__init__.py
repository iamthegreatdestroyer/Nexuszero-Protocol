"""
Nexuszero Optimizer

Neural network-based optimizer for zero-knowledge proof parameters.
Uses Graph Neural Networks to learn optimal parameter configurations
that balance security, proof size, and verification time.
"""

__version__ = "0.1.0"

from .models.gnn import ProofOptimizationGNN
from .utils.config import Config, ModelConfig, TrainingConfig, OptimizationConfig
from .verification.soundness import SoundnessVerifier
from .training.trainer import Trainer

__all__ = [
    "ProofOptimizationGNN",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "OptimizationConfig",
    "SoundnessVerifier",
    "Trainer",
]
