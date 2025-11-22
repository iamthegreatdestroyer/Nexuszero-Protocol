# Nexuszero Optimizer

Neural network-based optimizer for zero-knowledge proof parameters using Graph Neural Networks (GNN).

## Overview

The Nexuszero Optimizer uses deep learning to automatically find optimal cryptographic parameters for zero-knowledge proofs. It analyzes proof circuit structures and predicts parameter configurations that balance:

- **Security:** Maintain cryptographic strength
- **Performance:** Minimize proof generation time
- **Size:** Reduce proof size for efficient verification

## Features

- 🧠 **Graph Neural Networks:** Learns from circuit topology
- ⚡ **Fast Optimization:** Predict parameters in milliseconds
- 🔒 **Security Aware:** Maintains cryptographic soundness
- 📊 **Performance Metrics:** Estimates proof size and timing
- 🔗 **Rust Integration:** Bridges to nexuszero-crypto library

## Architecture

```
Circuit Graph → GNN Encoder → Parameter Predictor → Optimal Params (n, q, σ)
                              → Metrics Predictor → Size, Prove Time, Verify Time
```

## Installation

```bash
# Clone repository
cd nexuszero-optimizer

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Jupyter Notebooks

For an interactive walkthrough, see:

- **[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)** - End-to-end workflow covering dataset generation, training, Optuna tuning, and prediction

### Generate Training Data

```python
from nexuszero_optimizer.training.dataset import ProofCircuitGenerator

generator = ProofCircuitGenerator(min_nodes=10, max_nodes=1000)
generator.generate_dataset(
    num_samples=10000,
    output_dir="data",
    split="train"
)
```

### Train Model (CLI)

```bash
python scripts/train.py --config config.yaml
```

This will:

- Load dataset splits from `data/`
- Initialize GNN model, optimizer, scheduler
- Run training with early stopping & checkpointing
- Write TensorBoard logs if enabled in config

### Train Model (Programmatic)

```python
from nexuszero_optimizer import Config, Trainer

config = Config.from_yaml("config.yaml")
trainer = Trainer(config)
trainer.fit()
test_metrics = trainer.evaluate_test()
print(test_metrics)
```

### Predict Parameters

```python
import torch
from nexuszero_optimizer import ProofOptimizationGNN

# Load trained model
model = ProofOptimizationGNN.load("checkpoints/best_model.pt")

# Prepare circuit data (example)
x = torch.randn(100, 10)  # Node features
edge_index = torch.randint(0, 100, (2, 200))  # Edges
edge_attr = torch.randn(200, 4)  # Edge features

# Predict optimal parameters
result = model.predict_parameters(x, edge_index, edge_attr)

print(f"Optimal n: {result['n']}")
print(f"Optimal q: {result['q']}")
print(f"Optimal σ: {result['sigma']}")
print(f"Estimated proof size: {result['estimated_proof_size']} bytes")
print(f"Estimated prove time: {result['estimated_prove_time']:.2f} ms")
```

## Project Structure

```text
nexuszero-optimizer/
├── src/nexuszero_optimizer/
│   ├── models/          # GNN architectures
│   ├── training/        # Training loop and datasets
│   ├── optimization/    # Parameter optimizers
│   ├── verification/    # Soundness checking
│   └── utils/           # Utilities and config
├── tests/               # Unit tests
├── data/                # Training data
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── notebooks/           # Jupyter notebooks
```

## Configuration

Create a `config.yaml`:

```yaml
model:
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  warmup_steps: 1000

optimization:
  security_level: 128
  max_proof_size: 10000
  target_verify_time: 50.0

device: "cuda" # or "cpu"
num_workers: 4
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=nexuszero_optimizer
```

## Development

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Integration with Nexuszero-Crypto

The optimizer integrates with the Rust cryptography library:

```python
from nexuszero_optimizer.utils.crypto_bridge import CryptoBridge

bridge = CryptoBridge("../nexuszero-crypto/target/release/libnexuszero_crypto.so")
params = bridge.estimate_parameters(security_level=128)
```

## Performance

- **Training Time:** ~2-4 hours on GPU for 10k samples
- **Inference:** <10ms per circuit
- **Accuracy:** 95%+ parameter prediction accuracy
- **Dataset Size:** 10k training + 2k validation + 2k test circuits

## Experiment Tracking (WandB)

Enable WandB in `config.yaml` training section:

```yaml
training:
  wandb_enabled: true
  wandb_project: nexuszero-optimizer
  wandb_run_name: exp-gnn-1
  wandb_tags: ["gnn", "lattice", "zk"]
```

Run training and view dashboard:

```bash
python scripts/train.py --config config.yaml
```

Metrics logged: losses, security_score, bit_security, hardness.

## Hyperparameter Tuning

### Optuna

Run search (10 trials):

```bash
python scripts/tune_optuna.py --config config.yaml --trials 10
```

Programmatic usage:

```python
from nexuszero_optimizer import Config
from nexuszero_optimizer.training.tuner import OptunaTuner

cfg = Config.from_yaml("config.yaml")
study = OptunaTuner(cfg, n_trials=5).run()
print(study.best_trial.params)
```

### Ray Tune

Distributed/parallel search:

```bash
python scripts/tune_ray.py --config config.yaml --samples 8
```

Adjust search space in `scripts/tune_ray.py`.

## Roadmap

- [x] Basic GNN architecture
- [x] Data pipeline
- [x] Soundness verification integration
- [x] Training loop with metrics & early stopping
- [x] Hyperparameter tuning with Optuna / Ray
- [ ] Multi-GPU training support
- [ ] Production deployment scripts
- [ ] Advanced security formalization

## License

MIT License - See LICENSE file for details

## Citation

If you use this work, please cite:

```bibtex
@software{nexuszero_optimizer,
  title = {Nexuszero Optimizer: Neural Parameter Optimization for Zero-Knowledge Proofs},
  author = {Nexuszero Protocol},
  year = {2024},
  url = {https://github.com/nexuszero-protocol/nexuszero-optimizer}
}
```

## Contact

- **Issues:** [GitHub Issues](https://github.com/nexuszero-protocol/issues)
- **Email:** [info@nexuszero.com](mailto:info@nexuszero.com)
- **Docs:** [Documentation](https://docs.nexuszero.com)

---

**Status:** Active Development | **Version:** 0.1.0 | **Last Updated:** November 2025
