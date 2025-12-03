# Copyright (c) 2025 NexusZero Protocol
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
# Licensed under the GNU Affero General Public License v3.0 or later.
# Commercial licensing available at https://nexuszero.io/licensing
#
# NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
# are trademarks of NexusZero Protocol. All Rights Reserved.

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
