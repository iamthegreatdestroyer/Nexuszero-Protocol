"""
Nexuszero Protocol - Proof Optimization Agent System
Autonomous agents with recursive self-improvement for ZK proof optimization.
"""

from .supervisor import (
    ProofOptimizationSupervisor,
    ProofOptimizationTask,
    OptimizationResult,
    OptimizationAgent,
    ConstraintReductionAgent,
    WitnessOptimizationAgent,
    ProverParameterAgent,
    RecursiveSelfImprovementEngine,
    OptimizationPriority,
    OptimizationPhase,
    AgentState,
    create_proof_optimization_supervisor
)

from .gnn_integration import (
    GNNOptimizationOracle,
    ProofOptimizationGNN,
    CircuitGraph,
    CircuitEncoder,
    create_gnn_oracle
)

__all__ = [
    "ProofOptimizationSupervisor",
    "ProofOptimizationTask",
    "OptimizationResult",
    "OptimizationAgent",
    "ConstraintReductionAgent",
    "WitnessOptimizationAgent",
    "ProverParameterAgent",
    "RecursiveSelfImprovementEngine",
    "OptimizationPriority",
    "OptimizationPhase",
    "AgentState",
    "create_proof_optimization_supervisor",
    "GNNOptimizationOracle",
    "ProofOptimizationGNN",
    "CircuitGraph",
    "CircuitEncoder",
    "create_gnn_oracle"
]
