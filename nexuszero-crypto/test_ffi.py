#!/usr/bin/env python3
"""
Test script for Nexuszero Crypto FFI bindings

This script tests the C FFI interface from Python using ctypes.
It verifies that the Rust cryptographic library can be called from Python.
"""

import ctypes
import os
import sys
from pathlib import Path

# Find the compiled library
def find_library():
    """Locate the compiled cdylib file"""
    repo_root = Path(__file__).parent.parent
    
    # Common library locations
    search_paths = [
        repo_root / "target" / "release",
        repo_root / "target" / "debug",
    ]
    
    # Library names vary by platform
    lib_names = [
        "libnexuszero_crypto.so",      # Linux
        "libnexuszero_crypto.dylib",   # macOS
        "nexuszero_crypto.dll",        # Windows
    ]
    
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)
    
    return None

# Load the library
lib_path = find_library()
if lib_path is None:
    print("ERROR: Could not find nexuszero-crypto library")
    print("Please build the library first with: cargo build --release")
    sys.exit(1)

print(f"Loading library from: {lib_path}")
lib = ctypes.CDLL(lib_path)

# Define C structures matching the Rust definitions

class CryptoParams(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint32),
        ("q", ctypes.c_uint32),
        ("sigma", ctypes.c_double),
    ]

class OptimizationResult(ctypes.Structure):
    _fields_ = [
        ("optimal_n", ctypes.c_uint32),
        ("optimal_q", ctypes.c_uint32),
        ("optimal_sigma", ctypes.c_double),
        ("estimated_proof_size", ctypes.c_uint64),
        ("estimated_prove_time_ms", ctypes.c_uint64),
    ]

# Define function signatures
lib.nexuszero_estimate_parameters.argtypes = [
    ctypes.c_uint32,  # security_level
    ctypes.c_uint32,  # circuit_size
    ctypes.POINTER(OptimizationResult),  # result
]
lib.nexuszero_estimate_parameters.restype = ctypes.c_int32

lib.nexuszero_free_result.argtypes = [ctypes.POINTER(OptimizationResult)]
lib.nexuszero_free_result.restype = None

lib.nexuszero_get_version.argtypes = []
lib.nexuszero_get_version.restype = ctypes.c_char_p

lib.nexuszero_validate_params.argtypes = [ctypes.POINTER(CryptoParams)]
lib.nexuszero_validate_params.restype = ctypes.c_int32

# Error codes
FFI_SUCCESS = 0
FFI_ERROR_INVALID_PARAM = -1
FFI_ERROR_INTERNAL = -2
FFI_ERROR_NULL_POINTER = -3

def test_get_version():
    """Test getting the library version"""
    print("\n=== Test: Get Version ===")
    version = lib.nexuszero_get_version()
    version_str = version.decode('utf-8')
    print(f"Library version: {version_str}")
    assert version_str.startswith("0."), "Version should start with 0."
    print("✓ PASSED")

def test_estimate_parameters_128bit():
    """Test parameter estimation for 128-bit security"""
    print("\n=== Test: Estimate Parameters (128-bit) ===")
    result = OptimizationResult()
    
    status = lib.nexuszero_estimate_parameters(128, 1000, ctypes.byref(result))
    
    print(f"Status: {status}")
    assert status == FFI_SUCCESS, f"Expected FFI_SUCCESS, got {status}"
    
    print(f"Optimal n: {result.optimal_n}")
    print(f"Optimal q: {result.optimal_q}")
    print(f"Optimal sigma: {result.optimal_sigma:.2f}")
    print(f"Estimated proof size: {result.estimated_proof_size} bytes")
    print(f"Estimated prove time: {result.estimated_prove_time_ms} ms")
    
    assert result.optimal_n > 0, "n should be positive"
    assert result.optimal_q > 0, "q should be positive"
    assert result.optimal_sigma > 0, "sigma should be positive"
    assert result.estimated_proof_size > 0, "proof size should be positive"
    
    lib.nexuszero_free_result(ctypes.byref(result))
    print("✓ PASSED")

def test_estimate_parameters_192bit():
    """Test parameter estimation for 192-bit security"""
    print("\n=== Test: Estimate Parameters (192-bit) ===")
    result = OptimizationResult()
    
    status = lib.nexuszero_estimate_parameters(192, 5000, ctypes.byref(result))
    
    assert status == FFI_SUCCESS, f"Expected FFI_SUCCESS, got {status}"
    
    print(f"Optimal n: {result.optimal_n}")
    print(f"Optimal q: {result.optimal_q}")
    assert result.optimal_n >= 512, "n should be at least 512 for 192-bit security"
    
    lib.nexuszero_free_result(ctypes.byref(result))
    print("✓ PASSED")

def test_estimate_parameters_256bit():
    """Test parameter estimation for 256-bit security"""
    print("\n=== Test: Estimate Parameters (256-bit) ===")
    result = OptimizationResult()
    
    status = lib.nexuszero_estimate_parameters(256, 10000, ctypes.byref(result))
    
    assert status == FFI_SUCCESS, f"Expected FFI_SUCCESS, got {status}"
    
    print(f"Optimal n: {result.optimal_n}")
    assert result.optimal_n >= 1024, "n should be at least 1024 for 256-bit security"
    
    lib.nexuszero_free_result(ctypes.byref(result))
    print("✓ PASSED")

def test_invalid_security_level():
    """Test with invalid security level"""
    print("\n=== Test: Invalid Security Level ===")
    result = OptimizationResult()
    
    status = lib.nexuszero_estimate_parameters(99, 1000, ctypes.byref(result))
    
    print(f"Status: {status}")
    assert status == FFI_ERROR_INVALID_PARAM, f"Expected FFI_ERROR_INVALID_PARAM, got {status}"
    print("✓ PASSED")

def test_validate_params_valid():
    """Test parameter validation with valid parameters"""
    print("\n=== Test: Validate Valid Parameters ===")
    params = CryptoParams(n=512, q=12289, sigma=3.2)
    
    status = lib.nexuszero_validate_params(ctypes.byref(params))
    
    print(f"Status: {status}")
    assert status == FFI_SUCCESS, f"Expected FFI_SUCCESS, got {status}"
    print("✓ PASSED")

def test_validate_params_invalid_n():
    """Test parameter validation with invalid n (not power of 2)"""
    print("\n=== Test: Validate Invalid n ===")
    params = CryptoParams(n=500, q=12289, sigma=3.2)
    
    status = lib.nexuszero_validate_params(ctypes.byref(params))
    
    print(f"Status: {status}")
    assert status == FFI_ERROR_INVALID_PARAM, f"Expected FFI_ERROR_INVALID_PARAM, got {status}"
    print("✓ PASSED")

def test_validate_params_invalid_q():
    """Test parameter validation with invalid q"""
    print("\n=== Test: Validate Invalid q ===")
    params = CryptoParams(n=512, q=1, sigma=3.2)
    
    status = lib.nexuszero_validate_params(ctypes.byref(params))
    
    print(f"Status: {status}")
    assert status == FFI_ERROR_INVALID_PARAM, f"Expected FFI_ERROR_INVALID_PARAM, got {status}"
    print("✓ PASSED")

def test_validate_params_invalid_sigma():
    """Test parameter validation with invalid sigma"""
    print("\n=== Test: Validate Invalid sigma ===")
    params = CryptoParams(n=512, q=12289, sigma=-1.0)
    
    status = lib.nexuszero_validate_params(ctypes.byref(params))
    
    print(f"Status: {status}")
    assert status == FFI_ERROR_INVALID_PARAM, f"Expected FFI_ERROR_INVALID_PARAM, got {status}"
    print("✓ PASSED")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Nexuszero Crypto FFI Test Suite")
    print("=" * 60)
    
    tests = [
        test_get_version,
        test_estimate_parameters_128bit,
        test_estimate_parameters_192bit,
        test_estimate_parameters_256bit,
        test_invalid_security_level,
        test_validate_params_valid,
        test_validate_params_invalid_n,
        test_validate_params_invalid_q,
        test_validate_params_invalid_sigma,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
