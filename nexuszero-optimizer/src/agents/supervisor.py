"""
Proof Optimization Supervisor Agent
Orchestrates ZK proof optimization with recursive self-improvement loops. 
Implements DeepAgent pattern for the nexuszero-optimizer GNN system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import hashlib
import uuid

import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProofOptimizationSupervisor")


class OptimizationPhase(Enum):
    """Phases of proof optimization pipeline."""
    ANALYSIS = auto()
    PARAMETER_SEARCH = auto()
    CONSTRAINT_REDUCTION = auto()
    WITNESS_OPTIMIZATION = auto()
    VERIFICATION = auto()
    SELF_CRITIQUE = auto()
    REFINEMENT = auto()


class AgentState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    AWAITING_CRITIQUE = auto()
    REFINING = auto()
    COMPLETED = auto()
    FAILED = auto()


class OptimizationPriority(Enum):
    CRITICAL = 1      # Real-time verification required
    HIGH = 2          # Time-sensitive proofs
    STANDARD = 3      # Normal optimization
    BACKGROUND = 4    # Batch optimization, learning


@dataclass
class ProofOptimizationTask:
    """Represents a proof optimization task."""
    task_id: str
    proof_type: str  # groth16, plonk, bulletproofs, stark
    circuit_hash: str
    constraints: Dict[str, Any]
    optimization_target: str  # proving_time, verification_time, proof_size, memory
    priority: OptimizationPriority = OptimizationPriority.STANDARD
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    max_iterations: int = 10
    improvement_threshold: float = 0.05  # 5% improvement required
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "proof_type": self.proof_type,
            "circuit_hash": self.circuit_hash,
            "optimization_target": self.optimization_target,
            "priority": self.priority.name,
            "max_iterations": self.max_iterations
        }


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    task_id: str
    success: bool
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_percentage: float
    parameters_used: Dict[str, Any]
    execution_time_ms: float
    iterations_used: int
    agent_id: str
    critique_score: Optional[float] = None
    refinement_applied: bool = False
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "improvement_percentage": self.improvement_percentage,
            "iterations_used": self.iterations_used,
            "critique_score": self.critique_score,
            "refinement_applied": self.refinement_applied
        }


class OptimizationAgent(ABC):
    """Abstract base class for optimization agents."""
    
    def __init__(self, agent_id: str, specialization: str):
        self.agent_id = agent_id
        self.specialization = specialization
        self.state = AgentState.IDLE
        self.performance_history: List[Tuple[float, float]] = []  # (improvement, time)
        self.total_optimizations = 0
        self.successful_optimizations = 0
    
    @abstractmethod
    async def optimize(self, task: ProofOptimizationTask, current_params: Dict[str, Any]) -> OptimizationResult:
        """Execute optimization for the given task."""
        pass
    
    @abstractmethod
    async def critique(self, result: OptimizationResult) -> Tuple[float, List[str]]:
        """Critique an optimization result.  Returns (score, suggestions)."""
        pass
    
    def get_success_rate(self) -> float:
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations
    
    def get_average_improvement(self) -> float:
        if not self.performance_history:
            return 0.0
        return sum(p[0] for p in self.performance_history[-50:]) / len(self.performance_history[-50:])


class ConstraintReductionAgent(OptimizationAgent):
    """Agent specialized in reducing constraint count while preserving soundness."""
    
    def __init__(self):
        super().__init__(
            agent_id="constraint_reduction_agent_001",
            specialization="constraint_reduction"
        )
        self.reduction_strategies = [
            "variable_elimination",
            "constraint_merging",
            "redundancy_removal",
            "linear_combination"
        ]
    
    async def optimize(self, task: ProofOptimizationTask, current_params: Dict[str, Any]) -> OptimizationResult:
        start_time = datetime.utcnow()
        self.state = AgentState.PROCESSING
        self.total_optimizations += 1
        
        try:
            original_constraints = task.constraints.get("count", 1000)
            
            # Simulate constraint reduction analysis
            await asyncio.sleep(0.1)
            
            # Apply reduction strategies
            reduction_factor = 0.15 + (0.1 * np.random.random())  # 15-25% reduction
            optimized_constraints = int(original_constraints * (1 - reduction_factor))
            
            improvement = reduction_factor * 100
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.performance_history.append((improvement, execution_time))
            
            if improvement >= task.improvement_threshold * 100:
                self.successful_optimizations += 1
            
            self.state = AgentState.COMPLETED
            
            return OptimizationResult(
                task_id=task.task_id,
                success=True,
                original_metrics={"constraint_count": original_constraints},
                optimized_metrics={"constraint_count": optimized_constraints},
                improvement_percentage=improvement,
                parameters_used={"strategy": "hybrid", "iterations": 3},
                execution_time_ms=execution_time,
                iterations_used=3,
                agent_id=self.agent_id
            )
            
        except Exception as e:
            self.state = AgentState.FAILED
            return OptimizationResult(
                task_id=task.task_id,
                success=False,
                original_metrics={},
                optimized_metrics={},
                improvement_percentage=0.0,
                parameters_used={},
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                iterations_used=0,
                agent_id=self.agent_id,
                errors=[str(e)]
            )
    
    async def critique(self, result: OptimizationResult) -> Tuple[float, List[str]]:
        """Critique constraint reduction results."""
        suggestions = []
        score = 0.5
        
        if result.success:
            if result.improvement_percentage > 20:
                score = 0.9
            elif result.improvement_percentage > 10:
                score = 0.7
                suggestions.append("Consider applying recursive elimination for additional gains")
            else:
                score = 0.5
                suggestions.append("Try alternative constraint merging strategies")
                suggestions.append("Analyze for hidden redundancies in R1CS structure")
        else:
            score = 0.1
            suggestions.append("Review circuit structure for optimization barriers")
        
        return score, suggestions


class WitnessOptimizationAgent(OptimizationAgent):
    """Agent specialized in optimizing witness generation."""
    
    def __init__(self):
        super().__init__(
            agent_id="witness_optimization_agent_001",
            specialization="witness_generation"
        )
    
    async def optimize(self, task: ProofOptimizationTask, current_params: Dict[str, Any]) -> OptimizationResult:
        start_time = datetime.utcnow()
        self.state = AgentState.PROCESSING
        self.total_optimizations += 1
        
        try:
            original_time_ms = current_params.get("witness_gen_time_ms", 500)
            
            await asyncio.sleep(0.1)
            
            # Simulate witness optimization
            optimization_factor = 0.2 + (0.15 * np.random.random())  # 20-35% improvement
            optimized_time_ms = original_time_ms * (1 - optimization_factor)
            
            improvement = optimization_factor * 100
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.performance_history.append((improvement, execution_time))
            
            if improvement >= task.improvement_threshold * 100:
                self.successful_optimizations += 1
            
            self.state = AgentState.COMPLETED
            
            return OptimizationResult(
                task_id=task.task_id,
                success=True,
                original_metrics={"witness_gen_time_ms": original_time_ms},
                optimized_metrics={"witness_gen_time_ms": optimized_time_ms},
                improvement_percentage=improvement,
                parameters_used={"parallel_degree": 4, "caching": True},
                execution_time_ms=execution_time,
                iterations_used=2,
                agent_id=self.agent_id
            )
            
        except Exception as e:
            self.state = AgentState.FAILED
            return OptimizationResult(
                task_id=task.task_id,
                success=False,
                original_metrics={},
                optimized_metrics={},
                improvement_percentage=0.0,
                parameters_used={},
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                iterations_used=0,
                agent_id=self.agent_id,
                errors=[str(e)]
            )
    
    async def critique(self, result: OptimizationResult) -> Tuple[float, List[str]]:
        suggestions = []
        score = 0.5
        
        if result.success and result.improvement_percentage > 25:
            score = 0.85
        elif result.success and result.improvement_percentage > 15:
            score = 0.7
            suggestions.append("Explore SIMD vectorization for field operations")
        else:
            score = 0.4
            suggestions.append("Consider witness caching for repeated sub-circuits")
            suggestions.append("Analyze memory access patterns for cache optimization")
        
        return score, suggestions


class ProverParameterAgent(OptimizationAgent):
    """Agent specialized in optimizing prover parameters."""
    
    def __init__(self):
        super().__init__(
            agent_id="prover_parameter_agent_001",
            specialization="prover_parameters"
        )
    
    async def optimize(self, task: ProofOptimizationTask, current_params: Dict[str, Any]) -> OptimizationResult:
        start_time = datetime.utcnow()
        self.state = AgentState.PROCESSING
        self.total_optimizations += 1
        
        try:
            original_proving_time = current_params.get("proving_time_ms", 2000)
            
            await asyncio.sleep(0.15)
            
            # Simulate parameter optimization
            optimization_factor = 0.1 + (0.2 * np.random.random())  # 10-30% improvement
            optimized_proving_time = original_proving_time * (1 - optimization_factor)
            
            improvement = optimization_factor * 100
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.performance_history.append((improvement, execution_time))
            
            if improvement >= task.improvement_threshold * 100:
                self.successful_optimizations += 1
            
            self.state = AgentState.COMPLETED
            
            return OptimizationResult(
                task_id=task.task_id,
                success=True,
                original_metrics={"proving_time_ms": original_proving_time},
                optimized_metrics={"proving_time_ms": optimized_proving_time},
                improvement_percentage=improvement,
                parameters_used={
                    "fft_domain_size": 2**18,
                    "num_threads": 8,
                    "msm_window_size": 16
                },
                execution_time_ms=execution_time,
                iterations_used=5,
                agent_id=self.agent_id
            )
            
        except Exception as e:
            self.state = AgentState.FAILED
            return OptimizationResult(
                task_id=task.task_id,
                success=False,
                original_metrics={},
                optimized_metrics={},
                improvement_percentage=0.0,
                parameters_used={},
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                iterations_used=0,
                agent_id=self.agent_id,
                errors=[str(e)]
            )
    
    async def critique(self, result: OptimizationResult) -> Tuple[float, List[str]]:
        suggestions = []
        score = 0.5
        
        if result.success and result.improvement_percentage > 20:
            score = 0.9
        elif result.success:
            score = 0.6
            suggestions.append("Experiment with alternative MSM algorithms (Pippenger vs Bucket)")
            suggestions.append("Profile memory bandwidth for potential bottlenecks")
        else:
            score = 0.2
            suggestions.append("Review hardware capabilities for optimization ceiling")
        
        return score, suggestions


class RecursiveSelfImprovementEngine:
    """
    Implements recursive self-improvement for proof optimization.
    Generates -> Critiques -> Refines in a loop until convergence.
    """
    
    def __init__(self, max_refinement_cycles: int = 3, convergence_threshold: float = 0.02):
        self.max_refinement_cycles = max_refinement_cycles
        self.convergence_threshold = convergence_threshold  # Stop if improvement < 2%
        self.improvement_history: List[Dict[str, Any]] = []
    
    async def improve(
        self,
        agent: OptimizationAgent,
        task: ProofOptimizationTask,
        initial_params: Dict[str, Any]
    ) -> OptimizationResult:
        """Run recursive self-improvement loop."""
        
        current_params = initial_params.copy()
        best_result: Optional[OptimizationResult] = None
        cycle = 0
        
        while cycle < self.max_refinement_cycles:
            cycle += 1
            logger.info(f"Self-improvement cycle {cycle}/{self.max_refinement_cycles}")
            
            # Generate: Run optimization
            result = await agent.optimize(task, current_params)
            
            if not result.success:
                logger.warning(f"Optimization failed in cycle {cycle}")
                if best_result:
                    return best_result
                return result
            
            # Critique: Evaluate the result
            agent.state = AgentState.AWAITING_CRITIQUE
            critique_score, suggestions = await agent.critique(result)
            result.critique_score = critique_score
            
            logger.info(f"Cycle {cycle} critique score: {critique_score:.2f}, suggestions: {len(suggestions)}")
            
            # Track improvement
            self.improvement_history.append({
                "cycle": cycle,
                "improvement": result.improvement_percentage,
                "critique_score": critique_score,
                "suggestions": suggestions
            })
            
            # Update best result
            if best_result is None or result.improvement_percentage > best_result.improvement_percentage:
                best_result = result
            
            # Check convergence
            if cycle > 1:
                prev_improvement = self.improvement_history[-2]["improvement"]
                current_improvement = result.improvement_percentage
                delta = abs(current_improvement - prev_improvement) / max(prev_improvement, 0.01)
                
                if delta < self.convergence_threshold:
                    logger.info(f"Converged after {cycle} cycles (delta: {delta:.4f})")
                    break
            
            # Refine: Apply suggestions if score is below threshold
            if critique_score < 0.8 and suggestions:
                agent.state = AgentState.REFINING
                current_params = self._apply_suggestions(current_params, suggestions)
                result.refinement_applied = True
        
        return best_result or result
    
    def _apply_suggestions(self, params: Dict[str, Any], suggestions: List[str]) -> Dict[str, Any]:
        """Apply critique suggestions to parameters."""
        refined = params.copy()
        
        # Simple heuristic refinements based on suggestion keywords
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            
            if "parallel" in suggestion_lower or "simd" in suggestion_lower:
                refined["parallel_degree"] = refined.get("parallel_degree", 4) * 2
            
            if "cache" in suggestion_lower:
                refined["enable_caching"] = True
                refined["cache_size_mb"] = refined.get("cache_size_mb", 256) * 2
            
            if "memory" in suggestion_lower:
                refined["memory_pool_size"] = refined.get("memory_pool_size", 1024) * 1.5
            
            if "iteration" in suggestion_lower or "recursive" in suggestion_lower:
                refined["max_iterations"] = refined.get("max_iterations", 10) + 5
        
        return refined


class ProofOptimizationSupervisor:
    """
    Master supervisor for proof optimization pipeline.
    Coordinates multiple specialized agents with recursive self-improvement. 
    """
    
    def __init__(self):
        self.agents: Dict[str, OptimizationAgent] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.results_cache: Dict[str, OptimizationResult] = {}
        self.active_tasks: Dict[str, ProofOptimizationTask] = {}
        self.improvement_engine = RecursiveSelfImprovementEngine()
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default optimization agents."""
        self.register_agent(ConstraintReductionAgent())
        self.register_agent(WitnessOptimizationAgent())
        self.register_agent(ProverParameterAgent())
        logger.info(f"Registered {len(self.agents)} optimization agents")
    
    def register_agent(self, agent: OptimizationAgent):
        """Register an optimization agent."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent registered: {agent.agent_id} ({agent.specialization})")
    
    def select_agent(self, task: ProofOptimizationTask) -> OptimizationAgent:
        """Select the best agent for a task based on target and performance."""
        target = task.optimization_target
        
        # Map targets to specializations
        target_map = {
            "constraint_count": "constraint_reduction",
            "witness_time": "witness_generation",
            "proving_time": "prover_parameters",
            "verification_time": "prover_parameters",
            "proof_size": "constraint_reduction"
        }
        
        preferred_specialization = target_map.get(target, "prover_parameters")
        
        # Find agents with matching specialization
        matching_agents = [
            agent for agent in self.agents.values()
            if agent.specialization == preferred_specialization and agent.state == AgentState.IDLE
        ]
        
        if not matching_agents:
            # Fall back to any idle agent
            matching_agents = [a for a in self.agents.values() if a.state == AgentState.IDLE]
        
        if not matching_agents:
            # All agents busy, return the one with best success rate
            return max(self.agents.values(), key=lambda a: a.get_success_rate())
        
        # Return agent with best average improvement
        return max(matching_agents, key=lambda a: a.get_average_improvement())
    
    async def optimize_proof(
        self,
        task: ProofOptimizationTask,
        initial_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Optimize a proof using the best available agent with self-improvement."""
        
        logger.info(f"Starting proof optimization: {task.task_id} (target: {task.optimization_target})")
        
        self.active_tasks[task.task_id] = task
        
        # Select optimal agent
        agent = self.select_agent(task)
        logger.info(f"Selected agent: {agent.agent_id}")
        
        # Prepare initial parameters
        params = initial_params or {
            "proving_time_ms": 2000,
            "witness_gen_time_ms": 500,
            "parallel_degree": 4
        }
        
        # Run with recursive self-improvement
        result = await self.improvement_engine.improve(agent, task, params)
        
        # Cache result
        self.results_cache[task.task_id] = result
        del self.active_tasks[task.task_id]
        
        logger.info(f"Optimization complete: {task.task_id} - "
                   f"Improvement: {result.improvement_percentage:.2f}%, "
                   f"Critique: {result.critique_score:.2f}")
        
        return result
    
    async def batch_optimize(
        self,
        tasks: List[ProofOptimizationTask]
    ) -> Dict[str, OptimizationResult]:
        """Optimize multiple proofs in parallel."""
        
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)
        
        # Group by priority for staged execution
        results = {}
        
        for task in sorted_tasks:
            result = await self.optimize_proof(task)
            results[task.task_id] = result
        
        return results
    
    def get_agent_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all agents."""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                "specialization": agent.specialization,
                "state": agent.state.name,
                "success_rate": agent.get_success_rate(),
                "average_improvement": agent.get_average_improvement(),
                "total_optimizations": agent.total_optimizations
            }
        return stats


def create_proof_optimization_supervisor() -> ProofOptimizationSupervisor:
    """Factory function to create configured supervisor."""
    return ProofOptimizationSupervisor()


async def main():
    """Demonstration of the proof optimization supervisor."""
    supervisor = create_proof_optimization_supervisor()
    
    # Create sample optimization tasks
    tasks = [
        ProofOptimizationTask(
            task_id="opt_001",
            proof_type="groth16",
            circuit_hash="abc123def456",
            constraints={"count": 50000, "public_inputs": 10},
            optimization_target="proving_time",
            priority=OptimizationPriority.HIGH
        ),
        ProofOptimizationTask(
            task_id="opt_002",
            proof_type="plonk",
            circuit_hash="xyz789abc012",
            constraints={"count": 100000, "public_inputs": 5},
            optimization_target="constraint_count",
            priority=OptimizationPriority.STANDARD
        )
    ]
    
    print("=== Proof Optimization Supervisor Demo ===\n")
    
    for task in tasks:
        result = await supervisor.optimize_proof(task)
        
        print(f"\nTask: {task.task_id}")
        print(f"  Proof Type: {task.proof_type}")
        print(f"  Target: {task.optimization_target}")
        print(f"  Success: {result.success}")
        print(f"  Improvement: {result.improvement_percentage:.2f}%")
        print(f"  Critique Score: {result.critique_score:.2f}")
        print(f"  Refinement Applied: {result.refinement_applied}")
        print(f"  Iterations: {result.iterations_used}")
    
    print("\n=== Agent Statistics ===")
    for agent_id, stats in supervisor.get_agent_statistics().items():
        print(f"\n{agent_id}:")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Improvement: {stats['average_improvement']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
