#!/usr/bin/env python3
import asyncio
import os
import sys

# Add local src path so modules import by package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from agents import supervisor
from agents import gnn_integration

async def run_demo():
    sup = supervisor.create_proof_optimization_supervisor()
    task = supervisor.ProofOptimizationTask(
        task_id="demo_opt_001",
        proof_type="groth16",
        circuit_hash="demo_hash",
        constraints={"count": 1000, "variables": 3000},
        optimization_target="proving_time",
        priority=supervisor.OptimizationPriority.STANDARD,
    )

    result = await sup.optimize_proof(task)
    print("Demo optimization result:")
    print(result.to_dict())

if __name__ == '__main__':
    asyncio.run(run_demo())
