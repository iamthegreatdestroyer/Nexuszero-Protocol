# Nexuszero Optimizer - Agentic Supervisor

This module contains the Proof Optimization Supervisor and GNN integration used for optimizing zero-knowledge proof performance.

Quick start:

1. Create and activate a Python virtual environment:

```pwsh
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2. Install dependencies (recommended in project):

```pwsh
pip install -r requirements.txt
pip install -r requirements-agents.txt
```

3. Run import checks and a demo:

```pwsh
python .\nexuszero-optimizer\scripts\test_imports.py
python .\nexuszero-optimizer\scripts\demo_supervisor.py
```

Notes:

- `gnn_integration` requires PyTorch (>=2.0.0) to run GNN oracle. The module includes a fallback stub when PyTorch is not available.
- The Supervisor and agents are designed for simulation and integration; the GNN oracle uses a simplified graph representation.

Contributing:

- Add more sophisticated optimization agents by extending `OptimizationAgent`.
- Integrate GNN training pipelines and add datasets under `nexuszero-optimizer/data` for training.
