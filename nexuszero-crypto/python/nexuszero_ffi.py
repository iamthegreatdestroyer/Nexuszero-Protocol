"""
NexusZero FFI Bridge for Python
Tool Use Examples embedded for AI assistant accuracy

This module provides Python bindings to the NexusZero cryptographic library
with embedded Tool Use Examples to improve AI assistant parameter accuracy
from 72% to 90%+.
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
            "input": {"security_level": 192, "circuit_size": 5000},
            "description": "High security, medium circuit",
            "expected": {"optimal_n": 1024, "optimal_q": 40961, "optimal_sigma": 2.8}
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
