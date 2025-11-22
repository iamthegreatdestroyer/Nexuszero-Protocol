"""
Bridge to nexuszero-crypto Rust library.

This module provides Python bindings to the Rust cryptography library
for proof generation and verification.
"""

import ctypes
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CryptoBridge:
    """
    Bridge to nexuszero-crypto Rust library.
    
    This class provides Python bindings to the Rust cryptography
    library for proof generation and verification.
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize bridge to Rust library.
        
        Args:
            lib_path: Path to compiled Rust library.
                     If None, will search in common locations.
        """
        self.lib = None
        self.lib_path = lib_path
        
        if lib_path is None:
            # Search for library in common locations
            possible_paths = [
                "../nexuszero-crypto/target/release/libnexuszero_crypto.so",
                "../nexuszero-crypto/target/release/libnexuszero_crypto.dylib",
                "../nexuszero-crypto/target/release/nexuszero_crypto.dll",
                "../nexuszero-crypto/target/debug/libnexuszero_crypto.so",
                "../nexuszero-crypto/target/debug/libnexuszero_crypto.dylib",
                "../nexuszero-crypto/target/debug/nexuszero_crypto.dll",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    lib_path = path
                    break
        
        if lib_path and Path(lib_path).exists():
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures()
                logger.info(f"Loaded Rust library from {lib_path}")
            except Exception as e:
                logger.warning(f"Failed to load Rust library: {e}")
                logger.warning("Falling back to simulation mode")
        else:
            logger.warning(f"Rust library not found at {lib_path}")
            logger.warning("Running in simulation mode")
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for Rust FFI."""
        if self.lib is None:
            return
        
        try:
            # Proof generation function
            self.lib.generate_proof.argtypes = [
                ctypes.c_void_p,  # statement
                ctypes.c_void_p,  # witness
                ctypes.c_void_p,  # parameters
            ]
            self.lib.generate_proof.restype = ctypes.c_void_p
            
            # Verification function
            self.lib.verify_proof.argtypes = [
                ctypes.c_void_p,  # statement
                ctypes.c_void_p,  # proof
                ctypes.c_void_p,  # parameters
            ]
            self.lib.verify_proof.restype = ctypes.c_bool
            
            # Parameter estimation function
            self.lib.estimate_parameters.argtypes = [ctypes.c_int]
            self.lib.estimate_parameters.restype = ctypes.c_void_p
            
        except AttributeError as e:
            logger.warning(f"Some Rust functions not available: {e}")
    
    def is_available(self) -> bool:
        """Check if Rust library is available."""
        return self.lib is not None
    
    def generate_proof(
        self,
        statement: Dict,
        witness: Dict,
        parameters: Dict,
    ) -> Tuple[bytes, float, float]:
        """
        Generate zero-knowledge proof.
        
        Args:
            statement: Public statement
            witness: Secret witness
            parameters: Cryptographic parameters (n, q, sigma)
        
        Returns:
            Tuple of (proof_bytes, generation_time_ms, proof_size_bytes)
        """
        if not self.is_available():
            # Simulation mode
            return self._simulate_proof_generation(statement, witness, parameters)
        
        # TODO: Implement actual FFI call
        # For now, use simulation
        return self._simulate_proof_generation(statement, witness, parameters)
    
    def verify_proof(
        self,
        statement: Dict,
        proof: bytes,
        parameters: Dict,
    ) -> Tuple[bool, float]:
        """
        Verify zero-knowledge proof.
        
        Args:
            statement: Public statement
            proof: Proof bytes
            parameters: Cryptographic parameters
        
        Returns:
            Tuple of (is_valid, verification_time_ms)
        """
        if not self.is_available():
            # Simulation mode
            return self._simulate_verification(statement, proof, parameters)
        
        # TODO: Implement actual FFI call
        return self._simulate_verification(statement, proof, parameters)
    
    def estimate_parameters(
        self,
        security_level: int,
        circuit_size: Optional[int] = None,
    ) -> Dict:
        """
        Get estimated parameters for security level.
        
        Args:
            security_level: Security level in bits (128, 192, 256)
            circuit_size: Optional circuit size hint
        
        Returns:
            Dictionary with n, q, sigma, estimated sizes/times
        """
        # Parameter selection based on security level
        if security_level >= 256:
            n, q, sigma = 2048, 65537, 3.2
        elif security_level >= 192:
            n, q, sigma = 1024, 40961, 3.2
        else:  # 128-bit security
            n, q, sigma = 512, 12289, 3.2
        
        # Adjust based on circuit size if provided
        if circuit_size:
            if circuit_size > 500:
                n *= 2
            elif circuit_size > 200:
                n = int(n * 1.5)
        
        # Estimate performance metrics
        proof_size = n * 32  # Rough estimate
        prove_time = n * 0.1  # ms
        verify_time = n * 0.05  # ms
        
        return {
            "n": n,
            "q": q,
            "sigma": sigma,
            "estimated_proof_size": proof_size,
            "estimated_prove_time": prove_time,
            "estimated_verify_time": verify_time,
            "security_level": security_level,
        }
    
    def _simulate_proof_generation(
        self,
        statement: Dict,
        witness: Dict,
        parameters: Dict,
    ) -> Tuple[bytes, float, float]:
        """Simulate proof generation for testing."""
        n = parameters.get("n", 512)
        
        # Generate fake proof bytes
        proof_size = n * 32
        proof_bytes = bytes(np.random.randint(0, 256, proof_size, dtype=np.uint8))
        
        # Simulate timing based on parameters
        generation_time = n * 0.1  # ms
        
        return proof_bytes, generation_time, float(proof_size)
    
    def _simulate_verification(
        self,
        statement: Dict,
        proof: bytes,
        parameters: Dict,
    ) -> Tuple[bool, float]:
        """Simulate proof verification for testing."""
        n = parameters.get("n", 512)
        
        # Simulate verification (always passes in simulation)
        is_valid = True
        verification_time = n * 0.05  # ms
        
        return is_valid, verification_time
    
    def normalize_parameters(
        self,
        n: int,
        q: int,
        sigma: float,
    ) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range for neural network.
        
        Args:
            n: Lattice dimension (256-4096)
            q: Modulus (4096-131072)
            sigma: Error distribution parameter (2.0-5.0)
        
        Returns:
            Normalized parameters [n_norm, q_norm, sigma_norm]
        """
        n_norm = (n - 256) / (4096 - 256)
        q_norm = (q - 4096) / (131072 - 4096)
        sigma_norm = (sigma - 2.0) / (5.0 - 2.0)
        
        return np.array([n_norm, q_norm, sigma_norm], dtype=np.float32)
    
    def denormalize_parameters(
        self,
        params_norm: np.ndarray,
    ) -> Tuple[int, int, float]:
        """
        Denormalize parameters from [0, 1] range to actual values.
        
        Args:
            params_norm: Normalized parameters [n_norm, q_norm, sigma_norm]
        
        Returns:
            Tuple of (n, q, sigma)
        """
        n = int(256 + params_norm[0] * (4096 - 256))
        q = int(4096 + params_norm[1] * (131072 - 4096))
        sigma = 2.0 + params_norm[2] * (5.0 - 2.0)
        
        # Round to valid values
        n = 2 ** int(np.log2(n))  # Round to nearest power of 2
        
        return n, q, sigma


# Singleton instance
_crypto_bridge_instance: Optional[CryptoBridge] = None


def get_crypto_bridge(lib_path: Optional[str] = None) -> CryptoBridge:
    """
    Get singleton CryptoBridge instance.
    
    Args:
        lib_path: Optional path to Rust library
    
    Returns:
        CryptoBridge instance
    """
    global _crypto_bridge_instance
    
    if _crypto_bridge_instance is None:
        _crypto_bridge_instance = CryptoBridge(lib_path)
    
    return _crypto_bridge_instance


if __name__ == "__main__":
    # Test the bridge
    bridge = CryptoBridge()
    
    if bridge.is_available():
        print("✓ Rust library loaded successfully")
    else:
        print("⚠ Running in simulation mode")
    
    # Test parameter estimation
    params = bridge.estimate_parameters(security_level=128, circuit_size=100)
    print(f"\nEstimated parameters for 128-bit security:")
    print(f"  n: {params['n']}")
    print(f"  q: {params['q']}")
    print(f"  σ: {params['sigma']:.2f}")
    print(f"  Proof size: {params['estimated_proof_size']} bytes")
    print(f"  Prove time: {params['estimated_prove_time']:.2f} ms")
    print(f"  Verify time: {params['estimated_verify_time']:.2f} ms")
