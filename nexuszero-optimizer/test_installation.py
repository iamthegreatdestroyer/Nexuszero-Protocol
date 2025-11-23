"""
Complete test script to verify nexuszero-optimizer installation.

This script tests all major components:
1. Configuration loading
2. Dataset generation
3. Model creation and inference
4. Crypto bridge (if available)

Usage:
    python test_installation.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 70)
print("NEXUSZERO OPTIMIZER - INSTALLATION TEST")
print("=" * 70)
print()

# Test 1: Import packages
print("Test 1: Importing packages...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    
    import torch_geometric
    print(f"  ✓ PyTorch Geometric {torch_geometric.__version__}")
    
    from nexuszero_optimizer import ProofOptimizationGNN, Config
    print(f"  ✓ nexuszero_optimizer package")
    
    from nexuszero_optimizer.training.dataset import ProofCircuitGenerator, ProofCircuitDataset
    from nexuszero_optimizer.utils.crypto_bridge import CryptoBridge
    print(f"  ✓ All submodules")
    
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

print()

# Test 2: Configuration
print("Test 2: Configuration management...")
try:
    config = Config()
    print(f"  ✓ Default config created")
    print(f"    - Model hidden_dim: {config.model.hidden_dim}")
    print(f"    - Training batch_size: {config.training.batch_size}")
    print(f"    - Device: {config.device}")
    
    # Test YAML I/O
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config = f.name
    
    config.to_yaml(temp_config)
    loaded = Config.from_yaml(temp_config)
    Path(temp_config).unlink()
    
    print(f"  ✓ YAML save/load works")
    
except Exception as e:
    print(f"  ✗ Configuration test failed: {e}")
    sys.exit(1)

print()

# Test 3: Crypto Bridge
print("Test 3: Crypto bridge...")
try:
    bridge = CryptoBridge()
    
    if bridge.is_available():
        print(f"  ✓ Rust library loaded")
    else:
        print(f"  ⚠ Rust library not available (simulation mode)")
    
    # Test parameter estimation
    params = bridge.estimate_parameters(security_level=128)
    print(f"  ✓ Parameter estimation works")
    print(f"    - n: {params['n']}")
    print(f"    - q: {params['q']}")
    print(f"    - σ: {params['sigma']:.2f}")
    
except Exception as e:
    print(f"  ✗ Crypto bridge test failed: {e}")
    sys.exit(1)

print()

# Test 4: Dataset Generation
print("Test 4: Dataset generation...")
try:
    tmpdir = tempfile.mkdtemp()
    
    generator = ProofCircuitGenerator(min_nodes=5, max_nodes=20, seed=42)
    print(f"  ✓ Generator created")
    
    # Generate small test dataset
    generator.generate_dataset(
        num_samples=5,
        output_dir=tmpdir,
        split="test",
        show_progress=False,
    )
    print(f"  ✓ Generated 5 test circuits")
    
    # Load dataset
    dataset = ProofCircuitDataset(tmpdir, split="test")
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    
    # Test data loading
    data = dataset[0]
    print(f"  ✓ Sample loaded:")
    print(f"    - Nodes: {data.x.shape[0]}")
    print(f"    - Edges: {data.edge_index.shape[1]}")
    print(f"    - Parameters: {data.y.shape}")
    
    # Cleanup
    shutil.rmtree(tmpdir)
    
except Exception as e:
    print(f"  ✗ Dataset test failed: {e}")
    if 'tmpdir' in locals():
        shutil.rmtree(tmpdir, ignore_errors=True)
    sys.exit(1)

print()

# Test 5: Model Creation
print("Test 5: Model creation and inference...")
try:
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
    )
    print(f"  ✓ Model created")
    print(f"    - Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(20, 10)
    edge_index = torch.randint(0, 20, (2, 30))
    edge_attr = torch.randn(30, 4)
    batch = torch.zeros(20, dtype=torch.long)
    
    params, metrics = model(x, edge_index, edge_attr, batch)
    print(f"  ✓ Forward pass successful")
    print(f"    - Output shapes: params={params.shape}, metrics={metrics.shape}")
    
    # Test prediction
    result = model.predict_parameters(x, edge_index, edge_attr)
    print(f"  ✓ Parameter prediction works")
    print(f"    - n: {result['n']}")
    print(f"    - q: {result['q']}")
    print(f"    - σ: {result['sigma']:.2f}")
    
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model_path = f.name
    
    model.save(model_path)
    loaded_model = ProofOptimizationGNN.load(model_path)
    Path(model_path).unlink()
    
    print(f"  ✓ Model save/load works")
    
except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: CUDA availability (informational)
print("Test 6: Hardware check...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"    - Device: {torch.cuda.get_device_name(0)}")
    print(f"    - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"  ⚠ CUDA not available (will use CPU)")

print()

# Summary
print("=" * 70)
print("ALL TESTS PASSED! ✅")
print("=" * 70)
print()
print("Nexuszero Optimizer is ready to use!")
print()
print("Next steps:")
print("  1. Generate full dataset:")
print("     python scripts/generate_dataset.py --output_dir data")
print()
print("  2. Create configuration:")
print("     python scripts/create_config.py --output config.yaml")
print()
print("  3. Inspect dataset:")
print("     python scripts/inspect_dataset.py --data_dir data --split train")
print()
print("  4. Run unit tests:")
print("     pytest tests/ -v")
print()
print("See QUICKSTART.md for more information.")
print()
