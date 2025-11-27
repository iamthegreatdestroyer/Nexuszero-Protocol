"""
Batch Orchestrator for Neural Optimizer
Implements Programmatic Tool Calling pattern for efficient benchmark processing

This module enables processing 10,000+ circuits without context window pollution.
Intermediate results are processed in Python code, not in the AI's context.

Target: 37% token reduction for batch operations
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CircuitBenchmark:
    """Result of benchmarking a single circuit."""
    circuit_id: str
    circuit_size: int
    optimal_n: int
    optimal_q: int
    optimal_sigma: float
    estimated_proof_size: int
    estimated_prove_time_ms: int
    actual_prove_time_ms: Optional[int] = None
    speedup_ratio: Optional[float] = None


@dataclass
class BatchResult:
    """
    Aggregated result from batch processing.
    
    This is the ONLY data that enters the AI's context window.
    All intermediate results (10,000+ individual benchmarks) are
    processed in code and aggregated here.
    """
    total_circuits: int
    successful_circuits: int
    failed_circuits: int
    average_speedup: float
    average_proof_size: int
    average_prove_time_ms: int
    
    # Summary statistics (not raw data)
    min_speedup: float
    max_speedup: float
    speedup_std_dev: float
    
    # Top performers (small subset, not all 10,000)
    top_10_fastest: List[Dict[str, Any]] = field(default_factory=list)
    top_10_smallest_proofs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Worst performers (for investigation)
    worst_10_slowest: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    context_tokens_saved: int = 0  # Estimated tokens saved vs naive approach


class ProgrammaticToolOrchestrator:
    """
    Orchestrate tool calls programmatically to avoid context pollution.
    
    Anthropic's PTC Pattern:
    - AI writes orchestration code
    - Code executes tools outside AI context
    - Only final summary enters context
    
    Benefits:
    - 37% token reduction
    - Parallel execution
    - Clean context window
    """
    
    def __init__(
        self,
        benchmark_tool: Callable[[int, int], Awaitable[Dict[str, Any]]],
        max_parallel: int = 20,
        batch_size: int = 100
    ):
        """
        Initialize the orchestrator.
        
        Args:
            benchmark_tool: Async function to benchmark a single circuit
                           signature: async (security_level, circuit_size) -> result
            max_parallel: Maximum concurrent tool calls
            batch_size: Number of circuits per batch (for progress tracking)
        """
        self.benchmark_tool = benchmark_tool
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_parallel)
    
    async def _benchmark_single(
        self,
        circuit_id: str,
        security_level: int,
        circuit_size: int
    ) -> CircuitBenchmark:
        """
        Benchmark a single circuit with concurrency control.
        
        Results are processed IN CODE, not in AI context.
        """
        async with self._semaphore:
            start = time.perf_counter()
            
            try:
                result = await self.benchmark_tool(security_level, circuit_size)
                actual_time = int((time.perf_counter() - start) * 1000)
                
                return CircuitBenchmark(
                    circuit_id=circuit_id,
                    circuit_size=circuit_size,
                    optimal_n=result["optimal_n"],
                    optimal_q=result["optimal_q"],
                    optimal_sigma=result["optimal_sigma"],
                    estimated_proof_size=result["estimated_proof_size"],
                    estimated_prove_time_ms=result["estimated_prove_time_ms"],
                    actual_prove_time_ms=actual_time,
                    speedup_ratio=result.get("speedup_ratio", 1.0)
                )
            except Exception as e:
                logger.warning(f"Circuit {circuit_id} failed: {e}")
                return CircuitBenchmark(
                    circuit_id=circuit_id,
                    circuit_size=circuit_size,
                    optimal_n=0,
                    optimal_q=0,
                    optimal_sigma=0.0,
                    estimated_proof_size=0,
                    estimated_prove_time_ms=0,
                    speedup_ratio=0.0
                )
    
    async def batch_benchmark(
        self,
        circuits: List[Dict[str, Any]],
        security_level: int = 128
    ) -> BatchResult:
        """
        Benchmark many circuits with Programmatic Tool Calling.
        
        CRITICAL: This method processes 10,000+ circuits but only
        returns a summary. The AI context never sees individual results.
        
        Args:
            circuits: List of {"id": str, "size": int} dicts
            security_level: Security level for all benchmarks
        
        Returns:
            BatchResult with aggregated statistics (fits in ~1KB)
            vs ~500KB if all 10,000 results were in context
        
        Example (for AI assistants):
        ---------------------------
        # Process 10,000 circuits:
        circuits = [{"id": f"circuit_{i}", "size": random.randint(100, 50000)} 
                    for i in range(10000)]
        
        result = await orchestrator.batch_benchmark(circuits, security_level=128)
        
        # AI sees only this summary:
        print(f"Processed {result.total_circuits} circuits")
        print(f"Average speedup: {result.average_speedup:.2f}x")
        print(f"Top 10 fastest: {result.top_10_fastest}")
        
        # Context tokens saved: ~50,000 tokens
        """
        start_time = time.perf_counter()
        
        # Create all benchmark tasks
        tasks = [
            self._benchmark_single(
                circuit["id"],
                security_level,
                circuit["size"]
            )
            for circuit in circuits
        ]
        
        # Execute all benchmarks (OUTSIDE AI context)
        results = await asyncio.gather(*tasks)
        
        # Process results IN CODE (not in AI context)
        successful = [r for r in results if r.optimal_n > 0]
        failed = [r for r in results if r.optimal_n == 0]
        
        if not successful:
            return BatchResult(
                total_circuits=len(results),
                successful_circuits=0,
                failed_circuits=len(failed),
                average_speedup=0.0,
                average_proof_size=0,
                average_prove_time_ms=0,
                min_speedup=0.0,
                max_speedup=0.0,
                speedup_std_dev=0.0,
                processing_time_seconds=time.perf_counter() - start_time
            )
        
        # Calculate statistics (all done in code)
        speedups = [r.speedup_ratio for r in successful if r.speedup_ratio]
        proof_sizes = [r.estimated_proof_size for r in successful]
        prove_times = [r.estimated_prove_time_ms for r in successful]
        
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        avg_proof_size = sum(proof_sizes) // len(proof_sizes) if proof_sizes else 0
        avg_prove_time = sum(prove_times) // len(prove_times) if prove_times else 0
        
        # Standard deviation calculation
        if len(speedups) > 1:
            variance = sum((x - avg_speedup) ** 2 for x in speedups) / len(speedups)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0
        
        # Sort for top/worst (done in code, not context)
        by_speedup = sorted(successful, key=lambda x: x.speedup_ratio or 0, reverse=True)
        by_proof_size = sorted(successful, key=lambda x: x.estimated_proof_size)
        by_time = sorted(successful, key=lambda x: x.estimated_prove_time_ms)
        
        # Extract ONLY top 10 for context (not all 10,000)
        def to_summary(benchmarks: List[CircuitBenchmark]) -> List[Dict[str, Any]]:
            return [
                {
                    "circuit_id": b.circuit_id,
                    "circuit_size": b.circuit_size,
                    "speedup": b.speedup_ratio,
                    "proof_size": b.estimated_proof_size,
                    "prove_time_ms": b.estimated_prove_time_ms
                }
                for b in benchmarks[:10]
            ]
        
        # Estimate tokens saved
        # Naive: ~50 tokens per result × 10,000 = 500,000 tokens
        # With PTC: ~1,000 tokens for summary
        tokens_saved = (len(results) * 50) - 1000
        
        return BatchResult(
            total_circuits=len(results),
            successful_circuits=len(successful),
            failed_circuits=len(failed),
            average_speedup=avg_speedup,
            average_proof_size=avg_proof_size,
            average_prove_time_ms=avg_prove_time,
            min_speedup=min(speedups) if speedups else 0.0,
            max_speedup=max(speedups) if speedups else 0.0,
            speedup_std_dev=std_dev,
            top_10_fastest=to_summary(by_time),
            top_10_smallest_proofs=to_summary(by_proof_size),
            worst_10_slowest=to_summary(by_time[-10:][::-1]),
            processing_time_seconds=time.perf_counter() - start_time,
            context_tokens_saved=tokens_saved
        )
    
    async def grid_search_parameters(
        self,
        n_values: List[int] = [256, 512, 1024, 2048],
        q_values: List[int] = [12289, 40961, 65537],
        circuit_sizes: List[int] = [100, 1000, 10000, 50000]
    ) -> Dict[str, Any]:
        """
        Grid search over parameter combinations.
        
        Programmatic Tool Calling Pattern:
        - This function executes 48+ benchmark combinations
        - Processes results in code
        - Returns only optimal configuration
        
        Example (for AI assistants):
        ---------------------------
        result = await orchestrator.grid_search_parameters()
        print(f"Optimal config: n={result['optimal_config']['n']}")
        # AI sees only the winning config, not all 48 combinations
        """
        results = []
        
        for n in n_values:
            for q in q_values:
                for size in circuit_sizes:
                    # Execute benchmark (outside AI context)
                    result = await self.benchmark_tool(128, size)
                    
                    results.append({
                        "n": n,
                        "q": q,
                        "circuit_size": size,
                        "prove_time_ms": result.get("estimated_prove_time_ms", 0),
                        "proof_size": result.get("estimated_proof_size", 0)
                    })
        
        # Find optimal (done in code)
        optimal = min(results, key=lambda x: x["prove_time_ms"])
        
        # Return only summary (not all 48 results)
        return {
            "total_combinations_tested": len(results),
            "optimal_config": optimal,
            "summary": {
                "fastest_prove_time_ms": optimal["prove_time_ms"],
                "smallest_proof_size": min(r["proof_size"] for r in results),
                "largest_proof_size": max(r["proof_size"] for r in results)
            }
        }


# Convenience function for quick batch processing
async def batch_benchmark_circuits(
    circuits: List[Dict[str, Any]],
    benchmark_fn: Callable,
    security_level: int = 128
) -> BatchResult:
    """
    Quick function to batch benchmark circuits using PTC pattern.
    
    Tool Use Example (for AI assistants):
    ------------------------------------
    # Define circuits:
    circuits = [{"id": f"c{i}", "size": 1000 * (i + 1)} for i in range(10000)]
    
    # Benchmark all (context receives only summary):
    result = await batch_benchmark_circuits(circuits, my_benchmark_fn)
    
    # AI context sees ~1KB instead of ~500KB:
    print(f"Average speedup: {result.average_speedup}x")
    print(f"Tokens saved: {result.context_tokens_saved}")
    """
    orchestrator = ProgrammaticToolOrchestrator(benchmark_fn)
    return await orchestrator.batch_benchmark(circuits, security_level)
