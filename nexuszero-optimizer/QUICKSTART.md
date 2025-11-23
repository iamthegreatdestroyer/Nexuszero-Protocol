# Nexuszero Optimizer - Quick Start Guide

# Nexuszero Optimizer - Quick Start Guide

<!-- AUTONOMY LEVEL 2 TEST: Do not worry about this line. This PR is to validate the secure pull_request_target workflow and check-run/comment behavior. -->

## 🚀 Getting Started

### 1. Installation

```bash
# Navigate to optimizer directory
cd nexuszero-optimizer

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Generate Training Data

```bash
# Generate dataset with default settings (10k train, 2k val, 2k test)
python scripts/generate_dataset.py --output_dir data

# Or customize:
python scripts/generate_dataset.py \
  --output_dir data \
  --train_samples 20000 \
  --val_samples 4000 \
  --test_samples 4000 \
  --min_nodes 10 \
  --max_nodes 1000
```

This will create:

```
data/
├── train/
│   ├── index.h5
│   ├── circuit_0.h5
│   ├── circuit_1.h5
│   └── ...
├── val/
└── test/
```

### 3. Create Configuration

```bash
# Create default config
python scripts/create_config.py --output config.yaml

# Or customize:
python scripts/create_config.py \
  --output config.yaml \
  --device cuda \
  --hidden_dim 256 \
  --batch_size 32
```

### 4. Inspect Dataset (Optional)

```bash
# View dataset statistics
python scripts/inspect_dataset.py --data_dir data --split train
```

### 5. Test the Model

```python
# test_model.py
import torch
from nexuszero_optimizer.models.gnn import ProofOptimizationGNN
from nexuszero_optimizer.utils.config import Config

# Load configuration
config = Config.from_yaml("config.yaml")

# Create model
model = ProofOptimizationGNN(
    node_feat_dim=10,  # 7 gate types + 3 features
    edge_feat_dim=4,   # 3 connection types + 1 weight
    hidden_dim=config.model.hidden_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    dropout=config.model.dropout,
)

print(f"Model created with {model.count_parameters():,} parameters")

# Test with random data
x = torch.randn(50, 10)  # 50 nodes
edge_index = torch.randint(0, 50, (2, 80))  # 80 edges
edge_attr = torch.randn(80, 4)

# Predict parameters
result = model.predict_parameters(x, edge_index, edge_attr)
print(f"\nPredicted optimal parameters:")
print(f"  n: {result['n']}")
print(f"  q: {result['q']}")
print(f"  σ: {result['sigma']:.2f}")
print(f"\nEstimated performance:")
print(f"  Proof size: {result['estimated_proof_size']} bytes")
print(f"  Prove time: {result['estimated_prove_time']:.2f} ms")
print(f"  Verify time: {result['estimated_verify_time']:.2f} ms")
```

### 6. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=nexuszero_optimizer --cov-report=html
```

## 📊 Next Steps

### Training (Coming in Days 5-7)

The training loop, metrics, and evaluation will be implemented in the next phase:

- **Day 5-6:** Soundness verifier integration
- **Day 7:** Training loop with TensorBoard logging

### Model Usage Example

```python
from nexuszero_optimizer import ProofOptimizationGNN
from nexuszero_optimizer.training.dataset import ProofCircuitDataset, create_dataloaders

# Load trained model
model = ProofOptimizationGNN.load("checkpoints/best_model.pt", device="cuda")

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data",
    batch_size=32,
    num_workers=4,
)

# Make predictions
for batch in test_loader:
    params, metrics = model(
        batch.x.cuda(),
        batch.edge_index.cuda(),
        batch.edge_attr.cuda(),
        batch.batch.cuda(),
    )
    # ... use predictions
    break
```

## 🛠️ Development Workflow

### Code Formatting

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/

# Linting
ruff check src/ tests/ scripts/
```

### Adding New Features

1. Implement feature in appropriate module
2. Add unit tests in `tests/`
3. Update documentation
4. Run tests to ensure nothing breaks
5. Format code

## 📝 Project Structure

```
nexuszero-optimizer/
├── src/nexuszero_optimizer/
│   ├── models/          # GNN architectures
│   │   ├── gnn.py      # Main ProofOptimizationGNN
│   │   └── attention.py # Edge-aware attention
│   ├── training/        # Training components
│   │   └── dataset.py   # Dataset & data generation
│   ├── optimization/    # Parameter optimizers (TBD)
│   ├── verification/    # Soundness checking (TBD)
│   └── utils/
│       ├── config.py    # Configuration management
│       └── crypto_bridge.py # Rust library bridge
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── data/                # Training data
├── checkpoints/         # Model checkpoints
└── logs/                # Training logs
```

## 🔗 Integration with Nexuszero-Crypto

The optimizer can use the Rust crypto library for actual parameter evaluation:

```python
from nexuszero_optimizer.utils.crypto_bridge import CryptoBridge

# Initialize bridge
bridge = CryptoBridge("../nexuszero-crypto/target/release/libnexuszero_crypto.so")

if bridge.is_available():
    # Use actual crypto library
    params = bridge.estimate_parameters(security_level=128)
else:
    # Falls back to simulation mode
    pass
```

## 📚 Additional Resources

- **Main README:** See `README.md` for detailed documentation
- **Configuration:** See `config.yaml` for all available options
- **Tests:** See `tests/` for usage examples
- **Week 2 Prompts:** See `../scripts/WEEK_2_NEURAL_OPTIMIZER_PROMPTS.md`

## ❓ Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 16 # Reduce from 32
```

### PyTorch Geometric Installation Issues

```bash
# Install from conda-forge
conda install pytorch-geometric -c conda-forge

# Or use pip with specific versions
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Dataset Not Found

Make sure to generate the dataset first:

```bash
python scripts/generate_dataset.py --output_dir data
```

---

**Status:** Days 1-4 Complete ✅  
**Next:** Days 5-7 (Soundness verification & training loop)  
**Last Updated:** November 2024
\n<!-- AUTONOMY LEVEL 2 FORK PR:  -->\n
