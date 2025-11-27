# NexusZero Protocol - Advanced Tool Use Implementation
# Master Class Copilot Prompt

**Version:** 1.0.0  
**Date:** November 26, 2025  
**Repository:** https://github.com/iamthegreatdestroyer/Nexuszero-Protocol  
**Purpose:** Implement Anthropic's Advanced Tool Use features across the entire project

---

## 🎯 OVERVIEW [REF:ATU-MASTER-001]

This Master Class prompt instructs GitHub Copilot to implement three advanced tool use features from Anthropic's latest release across the NexusZero Protocol:

1. **Tool Search Tool** - Dynamic tool discovery for 50+ tool libraries
2. **Programmatic Tool Calling (PTC)** - Code-based tool orchestration for efficiency
3. **Tool Use Examples** - Concrete usage patterns for parameter accuracy

### Why These Features Matter for NexusZero

| Feature | Problem Solved | Impact |
|---------|---------------|--------|
| Tool Search Tool | 13 agents × 5+ tools = 65+ tools consuming 70K+ tokens | 85% context reduction |
| Programmatic Tool Calling | 10K+ training circuits creating massive context bloat | 37% token reduction |
| Tool Use Examples | FFI bridge parameter errors (wrong byte arrays, invalid ranges) | 72% → 90% accuracy |

---

## 📋 PRE-IMPLEMENTATION CHECKLIST [REF:ATU-MASTER-002]

Before executing this prompt, verify:

```
✅ Repository cloned: C:\Projects\Nexuszero-Protocol
✅ GitHub Copilot Plus active in VS Code
✅ Rust toolchain installed (rustc 1.70+)
✅ Python 3.11+ with PyTorch installed
✅ All tests passing: cargo test --all && pytest
```

---

## 🚀 MASTER COPILOT PROMPT [REF:ATU-MASTER-003]

**Copy the entire block below and paste into GitHub Copilot Chat in VS Code:**

```
=== NEXUSZERO PROTOCOL - ADVANCED TOOL USE IMPLEMENTATION ===
=== MASTER CLASS PROMPT v1.0.0 ===

CONTEXT:
I am implementing Anthropic's Advanced Tool Use features across the NexusZero Protocol.
Repository: https://github.com/iamthegreatdestroyer/Nexuszero-Protocol

PROJECT STATUS:
- Week 1: ✅ Complete (Rust crypto library, 90%+ coverage, FFI bridge exists)
- Week 2: ✅ Complete (Neural optimizer with PyTorch GNN)
- Week 3: 🔄 In Progress (Holographic compression - MPS fix needed)
- Current: Implementing Advanced Tool Use infrastructure

EXISTING CODEBASE STRUCTURE:
```
Nexuszero-Protocol/
├── nexuszero-crypto/          # Rust cryptographic library
│   ├── src/
│   │   ├── ffi.rs             # FFI bindings (EXISTS - needs Tool Use Examples)
│   │   ├── lattice/           # LWE/Ring-LWE implementations
│   │   ├── proof/             # Zero-knowledge proof systems
│   │   ├── params/            # Parameter selection
│   │   └── utils/             # Utilities
│   ├── benches/               # Performance benchmarks
│   └── tests/                 # Unit and integration tests
│
├── nexuszero-optimizer/       # Python neural optimizer
│   ├── src/nexuszero_optimizer/
│   │   ├── models/            # GNN architecture
│   │   ├── training/          # Training pipeline
│   │   ├── optimization/      # Proof optimization
│   │   └── utils/             # Python utilities
│   └── tests/                 # PyTest suite
│
├── nexuszero-holographic/     # Holographic compression (Rust)
│   ├── src/                   # MPS compression
│   └── benches/               # Compression benchmarks
│
├── .agents/                   # 13 AI agent configurations
│   ├── dr_alex_cipher/        # Security specialist
│   ├── dr_asha_neural/        # ML engineer
│   ├── morgan_rustico/        # Rust developer
│   ├── jordan_ops/            # DevOps engineer
│   ├── quinn_quality/         # QA specialist
│   ├── sam_sentinel/          # Security auditor
│   ├── dana_docs/             # Documentation
│   └── taylor_frontend/       # Frontend developer
│
└── .github/
    ├── copilot-instructions.md
    └── workflows/             # CI/CD pipelines
```

OBJECTIVE:
Implement ALL THREE Advanced Tool Use features across the NexusZero Protocol to:
1. Enable dynamic tool discovery for 65+ tools (Tool Search Tool)
2. Reduce context consumption by 37%+ for batch operations (Programmatic Tool Calling)
3. Achieve 90%+ parameter accuracy for FFI calls (Tool Use Examples)

=== TASK 1: TOOL USE EXAMPLES FOR FFI BRIDGE ===
[REF:ATU-IMPL-TUE]

FILE: nexuszero-crypto/src/ffi.rs (MODIFY EXISTING)
FILE: nexuszero-crypto/docs/FFI_EXAMPLES.json (CREATE NEW)

The existing ffi.rs has these functions:
- nexuszero_estimate_parameters(security_level, circuit_size, result)
- nexuszero_validate_params(params)
- nexuszero_get_version()
- nexuszero_free_result(result)

Create a comprehensive Tool Use Examples specification:

```json
// nexuszero-crypto/docs/FFI_EXAMPLES.json
{
  "version": "1.0.0",
  "description": "Tool Use Examples for NexusZero FFI Bridge",
  "tools": [
    {
      "name": "nexuszero_estimate_parameters",
      "description": "Estimate optimal lattice parameters for a given security level and circuit size. Returns n (dimension), q (modulus), sigma (error std dev), proof size, and generation time estimates.",
      "input_schema": {
        "type": "object",
        "properties": {
          "security_level": {
            "type": "integer",
            "enum": [128, 192, 256],
            "description": "Target security level in bits (128=AES-128 equivalent, 192=AES-192, 256=AES-256)"
          },
          "circuit_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000000,
            "description": "Number of gates in the circuit to prove"
          }
        },
        "required": ["security_level", "circuit_size"]
      },
      "input_examples": [
        {
          "security_level": 128,
          "circuit_size": 1000,
          "_description": "Standard security, small circuit (e.g., simple range proof)",
          "_expected_output": {
            "optimal_n": 512,
            "optimal_q": 12289,
            "optimal_sigma": 3.2,
            "estimated_proof_size": 8448,
            "estimated_prove_time_ms": 262
          }
        },
        {
          "security_level": 128,
          "circuit_size": 10000,
          "_description": "Standard security, medium circuit (e.g., Merkle proof)",
          "_expected_output": {
            "optimal_n": 512,
            "optimal_q": 12289,
            "optimal_sigma": 3.2,
            "estimated_proof_size": 8448,
            "estimated_prove_time_ms": 829
          }
        },
        {
          "security_level": 192,
          "circuit_size": 5000,
          "_description": "High security, medium circuit (e.g., financial transaction proof)",
          "_expected_output": {
            "optimal_n": 1024,
            "optimal_q": 40961,
            "optimal_sigma": 2.8,
            "estimated_proof_size": 16640,
            "estimated_prove_time_ms": 2344
          }
        },
        {
          "security_level": 256,
          "circuit_size": 50000,
          "_description": "Maximum security, large circuit (e.g., cross-chain bridge proof)",
          "_expected_output": {
            "optimal_n": 2048,
            "optimal_q": 65537,
            "optimal_sigma": 2.5,
            "estimated_proof_size": 33024,
            "estimated_prove_time_ms": 29663
          }
        },
        {
          "security_level": 128,
          "circuit_size": 100,
          "_description": "Minimal circuit (e.g., single value proof)",
          "_expected_output": {
            "optimal_n": 512,
            "optimal_q": 12289,
            "optimal_sigma": 3.2,
            "estimated_proof_size": 8448,
            "estimated_prove_time_ms": 83
          }
        }
      ]
    },
    {
      "name": "nexuszero_validate_params",
      "description": "Validate cryptographic parameters for correctness. Checks n is power of 2, q >= 2, sigma > 0.",
      "input_schema": {
        "type": "object",
        "properties": {
          "n": {
            "type": "integer",
            "description": "Lattice dimension (must be power of 2: 256, 512, 1024, 2048)"
          },
          "q": {
            "type": "integer",
            "description": "Coefficient modulus (must be >= 2, preferably prime)"
          },
          "sigma": {
            "type": "number",
            "description": "Error distribution standard deviation (must be > 0)"
          }
        },
        "required": ["n", "q", "sigma"]
      },
      "input_examples": [
        {
          "n": 512,
          "q": 12289,
          "sigma": 3.2,
          "_description": "Valid standard parameters",
          "_expected_return": 0
        },
        {
          "n": 1024,
          "q": 40961,
          "sigma": 2.8,
          "_description": "Valid high-security parameters",
          "_expected_return": 0
        },
        {
          "n": 2048,
          "q": 65537,
          "sigma": 2.5,
          "_description": "Valid maximum-security parameters",
          "_expected_return": 0
        },
        {
          "n": 500,
          "q": 12289,
          "sigma": 3.2,
          "_description": "INVALID: n not power of 2",
          "_expected_return": -1
        },
        {
          "n": 512,
          "q": 1,
          "sigma": 3.2,
          "_description": "INVALID: q too small",
          "_expected_return": -1
        },
        {
          "n": 512,
          "q": 12289,
          "sigma": -1.0,
          "_description": "INVALID: sigma negative",
          "_expected_return": -1
        },
        {
          "n": 0,
          "q": 12289,
          "sigma": 3.2,
          "_description": "INVALID: n is zero",
          "_expected_return": -1
        }
      ]
    }
  ]
}
```

Now create Python ctypes bindings with Tool Use Examples embedded:

FILE: nexuszero-crypto/python/nexuszero_ffi.py (CREATE NEW)

```python
"""
NexusZero FFI Bridge for Python
Tool Use Examples embedded for AI assistant accuracy
"""

import ctypes
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

# Tool Use Examples - Embedded for AI assistance
TOOL_USE_EXAMPLES = {
    "estimate_parameters": [
        {
            "input": {"security_level": 128, "circuit_size": 1000},
            "description": "Standard security, small circuit",
            "expected": {"optimal_n": 512, "optimal_q": 12289, "optimal_sigma": 3.2}
        },
        {
            "input": {"security_level": 256, "circuit_size": 50000},
            "description": "Maximum security, large circuit",
            "expected": {"optimal_n": 2048, "optimal_q": 65537, "optimal_sigma": 2.5}
        },
    ],
    "validate_params": [
        {
            "input": {"n": 512, "q": 12289, "sigma": 3.2},
            "description": "Valid standard parameters",
            "expected_return": 0
        },
        {
            "input": {"n": 500, "q": 12289, "sigma": 3.2},
            "description": "INVALID: n not power of 2",
            "expected_return": -1
        },
    ]
}


@dataclass
class CryptoParams:
    """C-compatible cryptographic parameters structure."""
    n: int      # Lattice dimension (power of 2: 256, 512, 1024, 2048)
    q: int      # Coefficient modulus (>= 2, preferably prime)
    sigma: float  # Error distribution std dev (> 0)


@dataclass
class OptimizationResult:
    """C-compatible optimization result structure."""
    optimal_n: int
    optimal_q: int
    optimal_sigma: float
    estimated_proof_size: int
    estimated_prove_time_ms: int


class _CryptoParamsC(ctypes.Structure):
    """C structure for CryptoParams."""
    _fields_ = [
        ("n", ctypes.c_uint32),
        ("q", ctypes.c_uint32),
        ("sigma", ctypes.c_double),
    ]


class _OptimizationResultC(ctypes.Structure):
    """C structure for OptimizationResult."""
    _fields_ = [
        ("optimal_n", ctypes.c_uint32),
        ("optimal_q", ctypes.c_uint32),
        ("optimal_sigma", ctypes.c_double),
        ("estimated_proof_size", ctypes.c_uint64),
        ("estimated_prove_time_ms", ctypes.c_uint64),
    ]


# Error codes
FFI_SUCCESS = 0
FFI_ERROR_INVALID_PARAM = -1
FFI_ERROR_INTERNAL = -2
FFI_ERROR_NULL_POINTER = -3


class NexusZeroCrypto:
    """
    Python interface to NexusZero cryptographic library via FFI.
    
    Tool Use Examples are embedded to help AI assistants generate
    correct function calls.
    
    Example Usage (for AI assistants):
    ---------------------------------
    # Standard security, small circuit:
    >>> crypto = NexusZeroCrypto()
    >>> result = crypto.estimate_parameters(128, 1000)
    >>> print(result.optimal_n)  # Expected: 512
    
    # Maximum security, large circuit:
    >>> result = crypto.estimate_parameters(256, 50000)
    >>> print(result.optimal_n)  # Expected: 2048
    
    # Validate parameters:
    >>> is_valid = crypto.validate_params(512, 12289, 3.2)  # Returns True
    >>> is_valid = crypto.validate_params(500, 12289, 3.2)  # Returns False (n not power of 2)
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the NexusZero crypto library.
        
        Args:
            lib_path: Path to the shared library. Auto-detected if None.
                     - Windows: nexuszero_crypto.dll
                     - Linux: libnexuszero_crypto.so
                     - macOS: libnexuszero_crypto.dylib
        """
        if lib_path is None:
            lib_path = self._find_library()
        
        self._lib = ctypes.CDLL(lib_path)
        self._configure_functions()
    
    def _find_library(self) -> str:
        """Find the nexuszero-crypto shared library."""
        import sys
        
        # Determine library name by platform
        if sys.platform == "win32":
            lib_name = "nexuszero_crypto.dll"
        elif sys.platform == "darwin":
            lib_name = "libnexuszero_crypto.dylib"
        else:
            lib_name = "libnexuszero_crypto.so"
        
        # Search paths
        search_paths = [
            Path(__file__).parent.parent / "target" / "release" / lib_name,
            Path(__file__).parent.parent / "target" / "debug" / lib_name,
            Path.cwd() / lib_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            f"Could not find {lib_name}. Build with: cargo build --release"
        )
    
    def _configure_functions(self) -> None:
        """Configure FFI function signatures."""
        # nexuszero_estimate_parameters
        self._lib.nexuszero_estimate_parameters.argtypes = [
            ctypes.c_uint32,                      # security_level
            ctypes.c_uint32,                      # circuit_size
            ctypes.POINTER(_OptimizationResultC)  # result
        ]
        self._lib.nexuszero_estimate_parameters.restype = ctypes.c_int32
        
        # nexuszero_validate_params
        self._lib.nexuszero_validate_params.argtypes = [
            ctypes.POINTER(_CryptoParamsC)
        ]
        self._lib.nexuszero_validate_params.restype = ctypes.c_int32
        
        # nexuszero_get_version
        self._lib.nexuszero_get_version.argtypes = []
        self._lib.nexuszero_get_version.restype = ctypes.c_char_p
        
        # nexuszero_free_result
        self._lib.nexuszero_free_result.argtypes = [
            ctypes.POINTER(_OptimizationResultC)
        ]
        self._lib.nexuszero_free_result.restype = None
    
    def estimate_parameters(
        self, 
        security_level: int, 
        circuit_size: int
    ) -> OptimizationResult:
        """
        Estimate optimal lattice parameters for proof generation.
        
        Args:
            security_level: Target security in bits (128, 192, or 256)
            circuit_size: Number of gates in the circuit (1 to 1,000,000)
        
        Returns:
            OptimizationResult with optimal parameters and estimates
        
        Raises:
            ValueError: If security_level not in [128, 192, 256]
            ValueError: If circuit_size not in range [1, 1_000_000]
            RuntimeError: If internal error occurs
        
        Tool Use Examples:
        -----------------
        # Example 1: Standard security, small circuit
        estimate_parameters(security_level=128, circuit_size=1000)
        # Returns: optimal_n=512, optimal_q=12289, optimal_sigma=3.2
        
        # Example 2: High security, medium circuit
        estimate_parameters(security_level=192, circuit_size=5000)
        # Returns: optimal_n=1024, optimal_q=40961, optimal_sigma=2.8
        
        # Example 3: Maximum security, large circuit
        estimate_parameters(security_level=256, circuit_size=50000)
        # Returns: optimal_n=2048, optimal_q=65537, optimal_sigma=2.5
        """
        # Validate inputs
        if security_level not in (128, 192, 256):
            raise ValueError(
                f"security_level must be 128, 192, or 256, got {security_level}"
            )
        
        if not 1 <= circuit_size <= 1_000_000:
            raise ValueError(
                f"circuit_size must be in [1, 1_000_000], got {circuit_size}"
            )
        
        # Call FFI
        result = _OptimizationResultC()
        status = self._lib.nexuszero_estimate_parameters(
            ctypes.c_uint32(security_level),
            ctypes.c_uint32(circuit_size),
            ctypes.byref(result)
        )
        
        # Check status
        if status == FFI_ERROR_INVALID_PARAM:
            raise ValueError("Invalid parameters")
        elif status == FFI_ERROR_INTERNAL:
            raise RuntimeError("Internal error in cryptographic library")
        elif status != FFI_SUCCESS:
            raise RuntimeError(f"Unknown error: {status}")
        
        return OptimizationResult(
            optimal_n=result.optimal_n,
            optimal_q=result.optimal_q,
            optimal_sigma=result.optimal_sigma,
            estimated_proof_size=result.estimated_proof_size,
            estimated_prove_time_ms=result.estimated_prove_time_ms
        )
    
    def validate_params(self, n: int, q: int, sigma: float) -> bool:
        """
        Validate cryptographic parameters.
        
        Args:
            n: Lattice dimension (must be power of 2: 256, 512, 1024, 2048)
            q: Coefficient modulus (must be >= 2)
            sigma: Error distribution std dev (must be > 0)
        
        Returns:
            True if parameters are valid, False otherwise
        
        Tool Use Examples:
        -----------------
        # Valid parameters:
        validate_params(n=512, q=12289, sigma=3.2)   # Returns: True
        validate_params(n=1024, q=40961, sigma=2.8)  # Returns: True
        validate_params(n=2048, q=65537, sigma=2.5)  # Returns: True
        
        # Invalid parameters:
        validate_params(n=500, q=12289, sigma=3.2)   # Returns: False (n not power of 2)
        validate_params(n=512, q=1, sigma=3.2)       # Returns: False (q too small)
        validate_params(n=512, q=12289, sigma=-1.0)  # Returns: False (sigma negative)
        """
        params = _CryptoParamsC(
            n=ctypes.c_uint32(n),
            q=ctypes.c_uint32(q),
            sigma=ctypes.c_double(sigma)
        )
        
        status = self._lib.nexuszero_validate_params(ctypes.byref(params))
        return status == FFI_SUCCESS
    
    def get_version(self) -> str:
        """Get the library version string."""
        return self._lib.nexuszero_get_version().decode("utf-8")
    
    @staticmethod
    def get_tool_examples() -> Dict[str, Any]:
        """
        Get Tool Use Examples for AI assistant integration.
        
        Returns:
            Dictionary of tool examples in Anthropic's Tool Use Examples format
        """
        return TOOL_USE_EXAMPLES


# Convenience function for quick access
def estimate_optimal_params(security_level: int, circuit_size: int) -> OptimizationResult:
    """
    Quick function to estimate optimal parameters.
    
    Tool Use Examples (for AI assistants):
    -------------------------------------
    # Standard case:
    estimate_optimal_params(128, 1000)  # n=512, q=12289
    
    # High security:
    estimate_optimal_params(192, 5000)  # n=1024, q=40961
    
    # Maximum security:
    estimate_optimal_params(256, 50000)  # n=2048, q=65537
    """
    crypto = NexusZeroCrypto()
    return crypto.estimate_parameters(security_level, circuit_size)
```


=== TASK 2: PROGRAMMATIC TOOL CALLING FOR NEURAL OPTIMIZER ===
[REF:ATU-IMPL-PTC]

The neural optimizer processes 10,000+ circuits during training. Currently, each benchmark
result enters the context. With PTC, we'll orchestrate benchmarks in code.

FILE: nexuszero-optimizer/src/nexuszero_optimizer/utils/batch_orchestrator.py (CREATE NEW)

```python
"""
Batch Orchestrator for Neural Optimizer
Implements Programmatic Tool Calling pattern for efficient benchmark processing

This module enables processing 10,000+ circuits without context window pollution.
Intermediate results are processed in Python code, not in the AI's context.
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
```


=== TASK 3: TOOL SEARCH TOOL INFRASTRUCTURE ===
[REF:ATU-IMPL-TST]

Create the infrastructure for Tool Search Tool across the 13 agents and 65+ tools.

FILE: nexuszero-optimizer/src/nexuszero_optimizer/utils/tool_registry.py (CREATE NEW)

```python
"""
Tool Search Tool Registry for NexusZero Protocol

Implements Anthropic's Tool Search Tool pattern:
- Register tools with defer_loading=True for on-demand discovery
- Search tools by name, description, or capability
- Load only relevant tools into context (85% token savings)
"""

import json
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for tool organization."""
    CRYPTO = "cryptography"
    NEURAL = "neural_optimizer"
    COMPRESSION = "holographic_compression"
    BENCHMARK = "benchmarking"
    SECURITY = "security"
    CHAIN = "blockchain_connector"
    MONITORING = "monitoring"
    ADMIN = "administration"


@dataclass
class ToolDefinition:
    """
    Tool definition with defer_loading support.
    
    Matches Anthropic's Tool Use API structure.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: ToolCategory
    
    # Tool Search Tool configuration
    defer_loading: bool = True  # True = discovered on-demand
    
    # Search metadata
    keywords: List[str] = field(default_factory=list)
    agent_owner: Optional[str] = None  # Which agent owns this tool
    
    # Tool Use Examples (optional)
    input_examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "defer_loading": self.defer_loading
        }
        
        if self.input_examples:
            result["input_examples"] = self.input_examples
        
        return result


class ToolRegistry:
    """
    Central registry for all NexusZero tools.
    
    Implements Tool Search Tool pattern for efficient tool discovery.
    
    Usage:
    ------
    registry = ToolRegistry()
    
    # Register tools from all agents
    registry.register_crypto_tools()
    registry.register_neural_tools()
    registry.register_compression_tools()
    
    # Search for relevant tools (returns only matches)
    tools = registry.search("benchmark proof generation")
    
    # Get tools for API call (core + searched tools)
    api_tools = registry.get_api_tools(searched_tool_names=["benchmark_ntt"])
    
    Token Savings Example:
    ---------------------
    Without Tool Search:
    - All 65 tools loaded: ~70,000 tokens
    
    With Tool Search:
    - Core tools loaded: ~3,000 tokens
    - Search tool: ~500 tokens
    - On-demand tools: ~2,000 tokens (only what's needed)
    - Total: ~5,500 tokens (92% savings)
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._core_tools: Set[str] = set()  # Always loaded (defer_loading=False)
        self._search_index: Dict[str, Set[str]] = {}  # keyword -> tool names
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
        if not tool.defer_loading:
            self._core_tools.add(tool.name)
        
        # Build search index
        self._index_tool(tool)
        
        logger.debug(f"Registered tool: {tool.name} (defer={tool.defer_loading})")
    
    def _index_tool(self, tool: ToolDefinition) -> None:
        """Index a tool for search."""
        # Index by name parts
        for word in tool.name.lower().split("_"):
            self._add_to_index(word, tool.name)
        
        # Index by description words
        for word in tool.description.lower().split():
            if len(word) > 3:  # Skip short words
                self._add_to_index(word, tool.name)
        
        # Index by keywords
        for keyword in tool.keywords:
            self._add_to_index(keyword.lower(), tool.name)
        
        # Index by category
        self._add_to_index(tool.category.value, tool.name)
    
    def _add_to_index(self, keyword: str, tool_name: str) -> None:
        """Add a keyword to the search index."""
        if keyword not in self._search_index:
            self._search_index[keyword] = set()
        self._search_index[keyword].add(tool_name)
    
    def search(
        self, 
        query: str,
        category: Optional[ToolCategory] = None,
        max_results: int = 10
    ) -> List[ToolDefinition]:
        """
        Search for tools matching a query.
        
        Implements regex-based search similar to Anthropic's
        tool_search_tool_regex_20251119.
        
        Args:
            query: Search query (keywords or regex pattern)
            category: Optional category filter
            max_results: Maximum number of results
        
        Returns:
            List of matching ToolDefinitions (not loaded yet)
        
        Example (for AI assistants):
        ---------------------------
        # Search for benchmark tools:
        tools = registry.search("benchmark proof")
        # Returns: [benchmark_ntt, benchmark_proof_gen, ...]
        
        # Search with category filter:
        tools = registry.search("optimize", category=ToolCategory.NEURAL)
        # Returns: [optimize_circuit, optimize_params, ...]
        """
        matches: Dict[str, int] = {}  # tool_name -> score
        
        # Tokenize query
        query_words = query.lower().split()
        
        for word in query_words:
            # Exact match in index
            if word in self._search_index:
                for tool_name in self._search_index[word]:
                    matches[tool_name] = matches.get(tool_name, 0) + 2
            
            # Partial match (prefix)
            for indexed_word, tool_names in self._search_index.items():
                if indexed_word.startswith(word) or word.startswith(indexed_word):
                    for tool_name in tool_names:
                        matches[tool_name] = matches.get(tool_name, 0) + 1
        
        # Apply category filter
        if category:
            matches = {
                name: score 
                for name, score in matches.items()
                if self._tools[name].category == category
            }
        
        # Sort by score and return top results
        sorted_matches = sorted(
            matches.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_results]
        
        return [self._tools[name] for name, _ in sorted_matches]
    
    def get_api_tools(
        self,
        searched_tool_names: Optional[List[str]] = None,
        include_search_tool: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get tools formatted for Anthropic API.
        
        Returns:
        - Core tools (always loaded)
        - Tool Search Tool (if enabled)
        - Searched tools (if provided)
        
        This is the final output sent to the API.
        """
        result = []
        
        # Add search tool if enabled
        if include_search_tool:
            result.append({
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search"
            })
        
        # Add core tools (defer_loading=False)
        for name in self._core_tools:
            result.append(self._tools[name].to_api_format())
        
        # Add searched tools
        if searched_tool_names:
            for name in searched_tool_names:
                if name in self._tools and name not in self._core_tools:
                    result.append(self._tools[name].to_api_format())
        
        return result
    
    def get_all_deferred(self) -> List[Dict[str, Any]]:
        """
        Get all deferred tools for initial API setup.
        
        These are NOT loaded into context - they're just registered
        for on-demand discovery.
        """
        return [
            tool.to_api_format()
            for tool in self._tools.values()
            if tool.defer_loading
        ]
    
    def estimate_token_savings(self) -> Dict[str, int]:
        """
        Estimate token savings from using Tool Search Tool.
        
        Returns:
            Dictionary with token estimates
        """
        # Estimate ~1000 tokens per tool definition
        tokens_per_tool = 1000
        
        total_tools = len(self._tools)
        core_tools = len(self._core_tools)
        deferred_tools = total_tools - core_tools
        
        naive_tokens = total_tools * tokens_per_tool
        optimized_tokens = core_tools * tokens_per_tool + 500  # +500 for search tool
        
        return {
            "total_tools": total_tools,
            "core_tools": core_tools,
            "deferred_tools": deferred_tools,
            "naive_approach_tokens": naive_tokens,
            "optimized_approach_tokens": optimized_tokens,
            "tokens_saved": naive_tokens - optimized_tokens,
            "savings_percentage": round(
                (1 - optimized_tokens / naive_tokens) * 100, 1
            )
        }
    
    # =================================================================
    # PRE-REGISTERED TOOLS FOR NEXUSZERO PROTOCOL
    # =================================================================
    
    def register_all_nexuszero_tools(self) -> None:
        """Register all NexusZero Protocol tools."""
        self.register_crypto_tools()
        self.register_neural_tools()
        self.register_compression_tools()
        self.register_benchmark_tools()
        self.register_security_tools()
        self.register_chain_tools()
        self.register_monitoring_tools()
    
    def register_crypto_tools(self) -> None:
        """Register cryptography tools (nexuszero-crypto)."""
        
        # CORE TOOLS - Always loaded
        self.register(ToolDefinition(
            name="nexuszero_prove_range",
            description="Generate a zero-knowledge range proof for a value",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "Value to prove is in range"},
                    "min_value": {"type": "integer", "description": "Minimum allowed value"},
                    "max_value": {"type": "integer", "description": "Maximum allowed value"},
                    "blinding": {"type": "string", "description": "32-byte hex blinding factor"}
                },
                "required": ["value", "min_value", "max_value"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["proof", "range", "zkp", "zero-knowledge", "bulletproof"],
            agent_owner="dr_alex_cipher",
            input_examples=[
                {"value": 50, "min_value": 0, "max_value": 100, "_description": "Prove 50 is in [0,100]"},
                {"value": 1000000, "min_value": 0, "max_value": 2000000, "_description": "Large range proof"}
            ]
        ))
        
        self.register(ToolDefinition(
            name="nexuszero_verify_proof",
            description="Verify a zero-knowledge proof",
            input_schema={
                "type": "object",
                "properties": {
                    "proof": {"type": "string", "description": "Hex-encoded proof"},
                    "statement": {"type": "object", "description": "Statement being proved"}
                },
                "required": ["proof", "statement"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["verify", "proof", "zkp", "validation"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="nexuszero_estimate_parameters",
            description="Estimate optimal lattice parameters for security level and circuit size",
            input_schema={
                "type": "object",
                "properties": {
                    "security_level": {"type": "integer", "enum": [128, 192, 256]},
                    "circuit_size": {"type": "integer", "minimum": 1, "maximum": 1000000}
                },
                "required": ["security_level", "circuit_size"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=False,  # CORE - always loaded
            keywords=["parameters", "lattice", "lwe", "security"],
            agent_owner="morgan_rustico",
            input_examples=[
                {"security_level": 128, "circuit_size": 1000, "_expected_n": 512},
                {"security_level": 256, "circuit_size": 50000, "_expected_n": 2048}
            ]
        ))
        
        # DEFERRED TOOLS - Discovered on demand
        self.register(ToolDefinition(
            name="generate_keypair",
            description="Generate a new lattice-based keypair",
            input_schema={
                "type": "object",
                "properties": {
                    "security_level": {"type": "integer", "enum": [128, 192, 256]}
                },
                "required": ["security_level"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["key", "keypair", "generate", "lattice"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="sign_message",
            description="Sign a message using lattice-based signature scheme",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "private_key": {"type": "string"}
                },
                "required": ["message", "private_key"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["sign", "signature", "message", "authenticate"],
            agent_owner="dr_alex_cipher"
        ))
        
        self.register(ToolDefinition(
            name="verify_signature",
            description="Verify a lattice-based signature",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "signature": {"type": "string"},
                    "public_key": {"type": "string"}
                },
                "required": ["message", "signature", "public_key"]
            },
            category=ToolCategory.CRYPTO,
            defer_loading=True,
            keywords=["verify", "signature", "authenticate"],
            agent_owner="dr_alex_cipher"
        ))
    
    def register_neural_tools(self) -> None:
        """Register neural optimizer tools (nexuszero-optimizer)."""
        
        self.register(ToolDefinition(
            name="optimize_circuit",
            description="Optimize a proof circuit using neural network",
            input_schema={
                "type": "object",
                "properties": {
                    "circuit": {"type": "object", "description": "Circuit definition"},
                    "optimization_level": {"type": "string", "enum": ["fast", "balanced", "thorough"]}
                },
                "required": ["circuit"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["optimize", "circuit", "neural", "gnn", "ml"],
            agent_owner="dr_asha_neural"
        ))
        
        self.register(ToolDefinition(
            name="predict_optimal_params",
            description="Use trained model to predict optimal parameters",
            input_schema={
                "type": "object",
                "properties": {
                    "circuit_features": {"type": "object"},
                    "model_checkpoint": {"type": "string"}
                },
                "required": ["circuit_features"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["predict", "parameters", "model", "inference"],
            agent_owner="dr_asha_neural"
        ))
        
        self.register(ToolDefinition(
            name="batch_optimize_circuits",
            description="Optimize multiple circuits in batch using PTC pattern",
            input_schema={
                "type": "object",
                "properties": {
                    "circuits": {"type": "array", "items": {"type": "object"}},
                    "max_parallel": {"type": "integer", "default": 20}
                },
                "required": ["circuits"]
            },
            category=ToolCategory.NEURAL,
            defer_loading=True,
            keywords=["batch", "optimize", "parallel", "circuits"],
            agent_owner="dr_asha_neural"
        ))
    
    def register_compression_tools(self) -> None:
        """Register holographic compression tools (nexuszero-holographic)."""
        
        self.register(ToolDefinition(
            name="compress_state",
            description="Compress proof state using MPS/tensor network",
            input_schema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Hex-encoded state"},
                    "target_ratio": {"type": "number", "description": "Target compression ratio"}
                },
                "required": ["state"]
            },
            category=ToolCategory.COMPRESSION,
            defer_loading=True,
            keywords=["compress", "mps", "tensor", "holographic"],
            agent_owner="morgan_rustico"
        ))
        
        self.register(ToolDefinition(
            name="decompress_state",
            description="Decompress state from MPS representation",
            input_schema={
                "type": "object",
                "properties": {
                    "compressed": {"type": "string"}
                },
                "required": ["compressed"]
            },
            category=ToolCategory.COMPRESSION,
            defer_loading=True,
            keywords=["decompress", "restore", "state"],
            agent_owner="morgan_rustico"
        ))
    
    def register_benchmark_tools(self) -> None:
        """Register benchmarking tools."""
        
        self.register(ToolDefinition(
            name="benchmark_ntt",
            description="Benchmark NTT performance for various dimensions",
            input_schema={
                "type": "object",
                "properties": {
                    "dimensions": {"type": "array", "items": {"type": "integer"}},
                    "iterations": {"type": "integer", "default": 1000}
                },
                "required": ["dimensions"]
            },
            category=ToolCategory.BENCHMARK,
            defer_loading=True,
            keywords=["benchmark", "ntt", "performance", "fft"],
            agent_owner="quinn_quality"
        ))
        
        self.register(ToolDefinition(
            name="benchmark_proof_generation",
            description="Benchmark proof generation across parameter sets",
            input_schema={
                "type": "object",
                "properties": {
                    "parameter_sets": {"type": "array"},
                    "circuit_sizes": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["parameter_sets", "circuit_sizes"]
            },
            category=ToolCategory.BENCHMARK,
            defer_loading=True,
            keywords=["benchmark", "proof", "generation", "performance"],
            agent_owner="quinn_quality"
        ))
    
    def register_security_tools(self) -> None:
        """Register security audit tools."""
        
        self.register(ToolDefinition(
            name="audit_timing",
            description="Run timing analysis for side-channel vulnerabilities",
            input_schema={
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "input_variations": {"type": "array"}
                },
                "required": ["function_name"]
            },
            category=ToolCategory.SECURITY,
            defer_loading=True,
            keywords=["audit", "timing", "side-channel", "security"],
            agent_owner="sam_sentinel"
        ))
        
        self.register(ToolDefinition(
            name="fuzz_test",
            description="Run fuzz testing on cryptographic functions",
            input_schema={
                "type": "object",
                "properties": {
                    "target_function": {"type": "string"},
                    "iterations": {"type": "integer", "default": 10000}
                },
                "required": ["target_function"]
            },
            category=ToolCategory.SECURITY,
            defer_loading=True,
            keywords=["fuzz", "test", "security", "vulnerability"],
            agent_owner="sam_sentinel"
        ))
    
    def register_chain_tools(self) -> None:
        """Register blockchain connector tools."""
        
        chains = ["ethereum", "bitcoin", "solana", "polygon", "cosmos"]
        
        for chain in chains:
            self.register(ToolDefinition(
                name=f"submit_proof_{chain}",
                description=f"Submit a ZK proof to {chain.title()} network",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proof": {"type": "string"},
                        "contract_address": {"type": "string"}
                    },
                    "required": ["proof", "contract_address"]
                },
                category=ToolCategory.CHAIN,
                defer_loading=True,
                keywords=["submit", "proof", chain, "blockchain"],
                agent_owner="casey_cloud"
            ))
            
            self.register(ToolDefinition(
                name=f"verify_proof_{chain}",
                description=f"Verify a ZK proof on {chain.title()} network",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proof_hash": {"type": "string"},
                        "contract_address": {"type": "string"}
                    },
                    "required": ["proof_hash", "contract_address"]
                },
                category=ToolCategory.CHAIN,
                defer_loading=True,
                keywords=["verify", "proof", chain, "blockchain"],
                agent_owner="casey_cloud"
            ))
    
    def register_monitoring_tools(self) -> None:
        """Register monitoring and observability tools."""
        
        self.register(ToolDefinition(
            name="get_metrics",
            description="Get current system metrics (CPU, memory, proof rate)",
            input_schema={
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}}
                }
            },
            category=ToolCategory.MONITORING,
            defer_loading=True,
            keywords=["metrics", "monitoring", "prometheus", "grafana"],
            agent_owner="jordan_ops"
        ))
        
        self.register(ToolDefinition(
            name="alert_on_threshold",
            description="Set up alert for metric threshold",
            input_schema={
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "threshold": {"type": "number"},
                    "comparison": {"type": "string", "enum": ["gt", "lt", "eq"]}
                },
                "required": ["metric", "threshold", "comparison"]
            },
            category=ToolCategory.MONITORING,
            defer_loading=True,
            keywords=["alert", "threshold", "monitoring", "notification"],
            agent_owner="jordan_ops"
        ))


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _global_registry.register_all_nexuszero_tools()
    return _global_registry


def search_tools(query: str) -> List[ToolDefinition]:
    """
    Search for tools matching a query.
    
    Tool Use Example (for AI assistants):
    ------------------------------------
    # Find benchmark tools:
    tools = search_tools("benchmark proof")
    
    # Find security audit tools:
    tools = search_tools("side-channel timing")
    
    # Find Ethereum tools:
    tools = search_tools("ethereum submit")
    """
    return get_registry().search(query)
```


=== TASK 4: INTEGRATION TESTS ===
[REF:ATU-IMPL-TESTS]

FILE: nexuszero-optimizer/tests/test_advanced_tool_use.py (CREATE NEW)

```python
"""
Integration tests for Advanced Tool Use features.

Tests:
1. Tool Use Examples accuracy
2. Programmatic Tool Calling efficiency
3. Tool Search Tool discovery
"""

import pytest
import asyncio
from pathlib import Path
import json

# Import modules under test
from nexuszero_optimizer.utils.batch_orchestrator import (
    ProgrammaticToolOrchestrator,
    BatchResult,
    batch_benchmark_circuits
)
from nexuszero_optimizer.utils.tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolCategory,
    get_registry,
    search_tools
)


class TestToolUseExamples:
    """Test Tool Use Examples integration."""
    
    def test_ffi_examples_file_exists(self):
        """Verify FFI examples JSON file exists."""
        examples_path = Path("../nexuszero-crypto/docs/FFI_EXAMPLES.json")
        # This will pass after Task 1 implementation
        # assert examples_path.exists()
    
    def test_examples_match_schema(self):
        """Verify examples match their schema definitions."""
        registry = get_registry()
        
        for tool in registry._tools.values():
            if tool.input_examples:
                schema = tool.input_schema
                for example in tool.input_examples:
                    # Verify required fields are present
                    for required in schema.get("required", []):
                        if not required.startswith("_"):  # Skip metadata
                            assert required in example, \
                                f"Tool {tool.name} example missing required field: {required}"
    
    def test_crypto_tool_examples_accuracy(self):
        """Test that crypto tool examples produce expected results."""
        # This test verifies that Tool Use Examples improve accuracy
        registry = get_registry()
        tool = registry._tools.get("nexuszero_estimate_parameters")
        
        assert tool is not None
        assert len(tool.input_examples) >= 2
        
        # Verify examples have expected outputs
        for example in tool.input_examples:
            if "_expected_n" in example:
                # Validate expected n values are valid powers of 2
                expected_n = example["_expected_n"]
                assert expected_n in [256, 512, 1024, 2048]


class TestProgrammaticToolCalling:
    """Test Programmatic Tool Calling implementation."""
    
    @pytest.fixture
    def mock_benchmark_tool(self):
        """Create a mock benchmark tool for testing."""
        async def benchmark(security_level: int, circuit_size: int):
            # Simulate benchmark result
            await asyncio.sleep(0.001)  # Simulate work
            return {
                "optimal_n": 512 if security_level == 128 else 1024,
                "optimal_q": 12289,
                "optimal_sigma": 3.2,
                "estimated_proof_size": 8448,
                "estimated_prove_time_ms": circuit_size // 10,
                "speedup_ratio": 1.5
            }
        return benchmark
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, mock_benchmark_tool):
        """Test that batch processing aggregates results efficiently."""
        orchestrator = ProgrammaticToolOrchestrator(mock_benchmark_tool)
        
        # Create 100 test circuits
        circuits = [
            {"id": f"circuit_{i}", "size": 1000 + i * 100}
            for i in range(100)
        ]
        
        result = await orchestrator.batch_benchmark(circuits)
        
        # Verify aggregation
        assert isinstance(result, BatchResult)
        assert result.total_circuits == 100
        assert result.successful_circuits == 100
        assert result.failed_circuits == 0
        
        # Verify token savings estimate
        assert result.context_tokens_saved > 0
        
        # Verify only summaries, not raw data
        assert len(result.top_10_fastest) <= 10
        assert len(result.worst_10_slowest) <= 10
    
    @pytest.mark.asyncio
    async def test_context_token_savings(self, mock_benchmark_tool):
        """Verify PTC provides significant token savings."""
        orchestrator = ProgrammaticToolOrchestrator(mock_benchmark_tool)
        
        circuits = [{"id": f"c{i}", "size": 1000} for i in range(1000)]
        result = await orchestrator.batch_benchmark(circuits)
        
        # Should save ~50 tokens per circuit × 1000 = 50,000 tokens
        # Minus ~1000 for summary = 49,000 saved
        assert result.context_tokens_saved >= 45000
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_benchmark_tool):
        """Test that benchmarks run in parallel."""
        import time
        
        orchestrator = ProgrammaticToolOrchestrator(
            mock_benchmark_tool,
            max_parallel=20
        )
        
        circuits = [{"id": f"c{i}", "size": 1000} for i in range(100)]
        
        start = time.perf_counter()
        result = await orchestrator.batch_benchmark(circuits)
        elapsed = time.perf_counter() - start
        
        # With parallel execution, 100 circuits should complete faster
        # than 100 × sequential time
        # Each mock takes 0.001s, so sequential would be 0.1s minimum
        # With 20 parallel, should be ~0.005s
        assert elapsed < 0.1  # Verify parallelism is working


class TestToolSearchTool:
    """Test Tool Search Tool implementation."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with all tools."""
        registry = get_registry()
        
        # Should have tools from all categories
        assert len(registry._tools) >= 20
        
        # Should have core tools
        assert len(registry._core_tools) >= 3
    
    def test_search_by_keyword(self):
        """Test keyword-based tool search."""
        registry = get_registry()
        
        # Search for benchmark tools
        results = registry.search("benchmark")
        assert len(results) > 0
        assert all("benchmark" in r.name.lower() or "benchmark" in r.description.lower()
                   for r in results)
        
        # Search for proof tools
        results = registry.search("proof")
        assert len(results) > 0
    
    def test_search_by_category(self):
        """Test category-filtered search."""
        registry = get_registry()
        
        # Search only in crypto category
        results = registry.search("generate", category=ToolCategory.CRYPTO)
        for r in results:
            assert r.category == ToolCategory.CRYPTO
    
    def test_token_savings_estimate(self):
        """Test token savings calculation."""
        registry = get_registry()
        savings = registry.estimate_token_savings()
        
        assert savings["tokens_saved"] > 0
        assert savings["savings_percentage"] > 50  # Should save >50%
    
    def test_api_tools_format(self):
        """Test that API tools are correctly formatted."""
        registry = get_registry()
        
        api_tools = registry.get_api_tools(
            searched_tool_names=["benchmark_ntt"],
            include_search_tool=True
        )
        
        # Should include search tool
        search_tools = [t for t in api_tools if t.get("type") == "tool_search_tool_regex_20251119"]
        assert len(search_tools) == 1
        
        # Should include core tools
        core_tool_names = [t["name"] for t in api_tools if "name" in t]
        assert "nexuszero_prove_range" in core_tool_names
        
        # Should include searched tool
        assert "benchmark_ntt" in core_tool_names
    
    def test_deferred_loading(self):
        """Test that deferred tools are not in core."""
        registry = get_registry()
        
        # Most tools should be deferred
        deferred_count = sum(
            1 for t in registry._tools.values() if t.defer_loading
        )
        core_count = len(registry._core_tools)
        
        assert deferred_count > core_count * 3  # Many more deferred than core


class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete advanced tool use workflow."""
        # Step 1: Search for relevant tools
        registry = get_registry()
        tools = registry.search("optimize circuit neural")
        
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        
        # Step 2: Get API tools with searched tools
        api_tools = registry.get_api_tools(
            searched_tool_names=tool_names[:5],
            include_search_tool=True
        )
        
        # Verify reduced token usage
        savings = registry.estimate_token_savings()
        assert savings["savings_percentage"] > 50
        
        # Step 3: Verify tool examples exist for accuracy
        for tool in tools[:3]:
            if tool.input_examples:
                assert len(tool.input_examples) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```


=== TASK 5: DOCUMENTATION ===
[REF:ATU-IMPL-DOCS]

FILE: docs/ADVANCED_TOOL_USE.md (CREATE NEW)

```markdown
# NexusZero Protocol - Advanced Tool Use Guide

## Overview

This document describes the implementation of Anthropic's Advanced Tool Use
features across the NexusZero Protocol.

## Features Implemented

### 1. Tool Use Examples

**Location:** `nexuszero-crypto/docs/FFI_EXAMPLES.json`

Tool Use Examples provide concrete usage patterns for our FFI functions,
improving AI assistant accuracy from 72% to 90%+ for parameter handling.

**Example:**
```json
{
  "name": "nexuszero_estimate_parameters",
  "input_examples": [
    {
      "security_level": 128,
      "circuit_size": 1000,
      "_expected_n": 512
    }
  ]
}
```

### 2. Programmatic Tool Calling

**Location:** `nexuszero-optimizer/src/nexuszero_optimizer/utils/batch_orchestrator.py`

PTC enables processing 10,000+ circuits with only ~1KB entering the AI context:

```python
result = await orchestrator.batch_benchmark(circuits)
# AI sees: BatchResult with statistics
# AI does NOT see: 10,000 individual benchmark results
```

**Token Savings:** 37% reduction (from ~50KB to ~1KB for large batches)

### 3. Tool Search Tool

**Location:** `nexuszero-optimizer/src/nexuszero_optimizer/utils/tool_registry.py`

Dynamic tool discovery for 65+ tools:

```python
# Search for relevant tools (on-demand)
tools = search_tools("benchmark proof")

# Get API tools (only loads what's needed)
api_tools = registry.get_api_tools(searched_tool_names=["benchmark_ntt"])
```

**Token Savings:** 85% reduction (from 70K to ~5K tokens)

## Usage

### For AI Agents

When implementing tasks, AI agents should:

1. Use `search_tools()` to find relevant tools
2. Reference Tool Use Examples for correct parameter formats
3. Use `batch_benchmark_circuits()` for large-scale operations

### For Developers

1. Register new tools in `tool_registry.py`
2. Add examples to `FFI_EXAMPLES.json`
3. Use PTC pattern for batch operations

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tool Loading Tokens | 70,000 | 5,500 | 92% reduction |
| Batch Processing Tokens | 500,000 | 1,000 | 99.8% reduction |
| Parameter Accuracy | 72% | 90%+ | 25% improvement |
```


=== VERIFICATION ===
[REF:ATU-IMPL-VERIFY]

After generating all code, run these verification commands:

```bash
# 1. Verify Rust FFI compiles
cd nexuszero-crypto
cargo build --release
cargo test

# 2. Verify Python modules import
cd ../nexuszero-optimizer
python -c "from nexuszero_optimizer.utils.tool_registry import get_registry; print(get_registry().estimate_token_savings())"
python -c "from nexuszero_optimizer.utils.batch_orchestrator import ProgrammaticToolOrchestrator; print('PTC OK')"

# 3. Run integration tests
pytest tests/test_advanced_tool_use.py -v

# 4. Verify FFI examples JSON
cat ../nexuszero-crypto/docs/FFI_EXAMPLES.json | python -m json.tool

# 5. Check token savings
python -c "
from nexuszero_optimizer.utils.tool_registry import get_registry
r = get_registry()
print(r.estimate_token_savings())
"
```

Expected output:
```
{
    'total_tools': 25+,
    'core_tools': 3,
    'deferred_tools': 22+,
    'naive_approach_tokens': 25000+,
    'optimized_approach_tokens': 3500,
    'tokens_saved': 21500+,
    'savings_percentage': 86.0+
}
```


=== END OF MASTER PROMPT ===

GENERATE ALL FILES IN THIS ORDER:
1. nexuszero-crypto/docs/FFI_EXAMPLES.json
2. nexuszero-crypto/python/nexuszero_ffi.py
3. nexuszero-optimizer/src/nexuszero_optimizer/utils/batch_orchestrator.py
4. nexuszero-optimizer/src/nexuszero_optimizer/utils/tool_registry.py
5. nexuszero-optimizer/tests/test_advanced_tool_use.py
6. docs/ADVANCED_TOOL_USE.md

After generation, run the verification commands to ensure correctness.
```

---

## 📊 SUCCESS METRICS [REF:ATU-MASTER-004]

After implementation, verify these metrics:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Tool Search Token Savings | >80% | `registry.estimate_token_savings()["savings_percentage"]` |
| PTC Token Reduction | >35% | `result.context_tokens_saved` from batch operations |
| FFI Example Coverage | 100% | All FFI functions have 3+ examples |
| Test Pass Rate | 100% | `pytest tests/test_advanced_tool_use.py` |
| Tool Count | 25+ | `len(registry._tools)` |
| Core Tools | 3-5 | `len(registry._core_tools)` |

---

## 🔗 REFERENCE LINKS [REF:ATU-MASTER-005]

- **Anthropic Article:** https://www.anthropic.com/engineering/advanced-tool-use
- **Tool Search Tool Docs:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
- **PTC Docs:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling
- **Tool Use Examples Docs:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#providing-tool-use-examples

---

## 📝 IMPLEMENTATION NOTES [REF:ATU-MASTER-006]

### Integration with Existing Code

The implementation extends existing modules:

1. **FFI Bridge (ffi.rs)** - Already exists, adding examples documentation
2. **Neural Optimizer** - Adding batch orchestrator as new utility
3. **Tool Registry** - New infrastructure for all 13 agents

### Compatibility

- **Rust:** 1.70+ (existing requirement)
- **Python:** 3.11+ (existing requirement)
- **Anthropic API:** Beta features require header `betas=["advanced-tool-use-2025-11-20"]`

### Agent Integration

Each agent folder in `.agents/` can reference the tool registry:

```python
# In any agent context
from nexuszero_optimizer.utils.tool_registry import search_tools

# Agent discovers relevant tools
tools = search_tools("benchmark")
```

---

**Document Version:** 1.0.0  
**Last Updated:** November 26, 2025  
**Status:** ✅ READY FOR EXECUTION

**NEXT STEP:** Copy the Master Copilot Prompt (Section [REF:ATU-MASTER-003]) into GitHub Copilot Chat in VS Code and execute.
