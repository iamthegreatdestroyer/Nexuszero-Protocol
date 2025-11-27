#!/usr/bin/env python3
"""
Automated fix script for NexusZero integration test compilation errors.
This script addresses the major API mismatches between test assumptions and actual implementation.
"""

import re
import os

def fix_validation_result_enum_usage(content):
    """Fix ValidationResult enum usage - it's actually a struct"""
    # Replace ValidationResult::Valid with ValidationResult { is_valid: true, errors: vec![], warnings: vec![] }
    content = re.sub(
        r'ValidationResult::Valid',
        'ValidationResult { is_valid: true, errors: vec![], warnings: vec![] }',
        content
    )

    # Replace ValidationResult::Invalid(vec![...]) with ValidationResult { is_valid: false, errors: vec![...], warnings: vec![] }
    content = re.sub(
        r'ValidationResult::Invalid\(vec!\[([^\]]+)\]\)',
        r'ValidationResult { is_valid: false, errors: vec![\1], warnings: vec![] }',
        content
    )

    return content

def fix_validation_error_enum_usage(content):
    """Fix ValidationError enum usage - it's actually a struct"""
    # Replace ValidationError::InvalidStatement with ValidationError { code: ValidationErrorCode::InvalidStatementFormat, message: "...".to_string(), field: None }
    replacements = {
        'ValidationError::InvalidStatement': 'ValidationError { code: ValidationErrorCode::InvalidStatementFormat, message: "Invalid statement".to_string(), field: None }',
        'ValidationError::InvalidWitness': 'ValidationError { code: ValidationErrorCode::WitnessMismatch, message: "Invalid witness".to_string(), field: None }',
        'ValidationError::SecurityLevelMismatch': 'ValidationError { code: ValidationErrorCode::InsecureParameters, message: "Security level mismatch".to_string(), field: None }',
        'ValidationError::ParameterOutOfRange': 'ValidationError { code: ValidationErrorCode::SizeExceeded, message: "Parameter out of range".to_string(), field: None }',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content

def fix_protocol_error_variants(content):
    """Fix ProtocolError variant names"""
    replacements = {
        'ProtocolError::InvalidInput': 'ProtocolError::ValidationFailed',
        'ProtocolError::CompressionError': 'ProtocolError::CompressionFailed',
        'ProtocolError::OptimizationError': 'ProtocolError::OptimizationFailed',
        'ProtocolError::CryptoError': 'ProtocolError::ProofGenerationFailed',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content

def fix_batch_proof_result_enum_usage(content):
    """Fix BatchProofResult enum usage - it's actually a struct"""
    # Replace BatchProofResult::Success(proof) with BatchProofResult { id: "...".to_string(), proof: Some(proof), error: None, duration_ms: 0.0 }
    content = re.sub(
        r'BatchProofResult::Success\(([^)]+)\)',
        r'BatchProofResult { id: "test".to_string(), proof: Some(\1), error: None, duration_ms: 0.0 }',
        content
    )

    # Replace BatchProofResult::Error(err) with BatchProofResult { id: "...".to_string(), proof: None, error: Some(err), duration_ms: 0.0 }
    content = re.sub(
        r'BatchProofResult::Error\(([^)]+)\)',
        r'BatchProofResult { id: "test".to_string(), proof: None, error: Some(\1), duration_ms: 0.0 }',
        content
    )

    return content

def fix_proof_metrics_fields(content):
    """Fix ProofMetrics struct field names"""
    # Remove verification_time_ms field
    content = re.sub(r'verification_time_ms:\s*[^,]+,', '', content)

    return content

def fix_comprehensive_proof_metrics_fields(content):
    """Fix ComprehensiveProofMetrics struct field names"""
    # Remove non-existent fields
    content = re.sub(r'proof_metrics:\s*[^,]+,', '', content)
    content = re.sub(r'stage_metrics:\s*[^,]+,', '', content)
    content = re.sub(r'optimization_metrics:\s*[^,]+,', '', content)
    content = re.sub(r'compression_metrics:\s*[^,]+,', '', content)
    content = re.sub(r'cache_metrics:\s*[^,]+,', '', content)

    return content

def fix_optimized_proof_fields(content):
    """Fix OptimizedProof struct - remove proof_data field"""
    content = re.sub(r'proof_data:\s*[^,]+,', '', content)

    return content

def fix_security_level_variants(content):
    """Fix SecurityLevel enum variants"""
    content = content.replace('SecurityLevel::Bit64', 'SecurityLevel::Bit128')

    return content

def fix_hash_function_variants(content):
    """Fix HashFunction enum variants"""
    content = content.replace('HashFunction::BLAKE3', 'HashFunction::Blake3')

    return content

def fix_compression_strategy_variants(content):
    """Fix CompressionStrategy enum variants"""
    replacements = {
        'CompressionStrategy::TT_SVD': 'CompressionStrategy::TensorTrain',
        'CompressionStrategy::Hybrid': 'CompressionStrategy::HybridMps',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content

def fix_method_names(content):
    """Fix incorrect method names"""
    replacements = {
        'record_proof(': 'record_proof_size(',
        'total_proofs()': 'count()',
        'put(': 'insert(',
        'len()': 'stats().size',
        'generate_report()': 'finalize()',
        'is_enabled()': 'is_compression_enabled()',  # Assuming this method exists
        'stats()': 'get_stats()',  # Assuming this method exists
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content

def fix_batch_proof_request_fields(content):
    """Fix BatchProofRequest struct - add missing id and priority fields"""
    # Add id and priority fields to BatchProofRequest initializations
    content = re.sub(
        r'BatchProofRequest\s*\{',
        'BatchProofRequest { id: "test".to_string(), priority: 0,',
        content
    )

    return content

def fix_crypto_parameters_construction(content):
    """Fix CryptoParameters construction - use proper constructors"""
    content = re.sub(
        r'nexuszero_crypto::CryptoParameters::default\(\)',
        'nexuszero_crypto::CryptoParameters::new_128bit_security()',
        content
    )

    return content

def fix_proof_construction(content):
    """Fix Proof construction - use proper constructors instead of default"""
    # This is complex - for now, replace with a placeholder that needs manual fixing
    content = re.sub(
        r'Proof::default\(\)',
        'create_dummy_proof()',  # We'll need to define this helper
        content
    )

    return content

def add_missing_imports(content):
    """Add missing imports"""
    imports_to_add = [
        'use nexuszero_integration::ValidationErrorCode;',
        'use nexuszero_integration::optimization::Optimizer;',
    ]

    # Add after existing imports
    for import_line in imports_to_add:
        if import_line not in content:
            # Find the last use statement
            lines = content.split('\n')
            last_use_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('use ') and ';' in line:
                    last_use_idx = i

            if last_use_idx >= 0:
                lines.insert(last_use_idx + 1, import_line)
                content = '\n'.join(lines)

    return content

def main():
    test_file = r'c:\Users\sgbil\Nexuszero-Protocol\nexuszero-integration\tests\integration_tests.rs'

    print("Reading test file...")
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Applying fixes...")

    # Apply all fixes
    content = fix_validation_result_enum_usage(content)
    content = fix_validation_error_enum_usage(content)
    content = fix_protocol_error_variants(content)
    content = fix_batch_proof_result_enum_usage(content)
    content = fix_proof_metrics_fields(content)
    content = fix_comprehensive_proof_metrics_fields(content)
    content = fix_optimized_proof_fields(content)
    content = fix_security_level_variants(content)
    content = fix_hash_function_variants(content)
    content = fix_compression_strategy_variants(content)
    content = fix_method_names(content)
    content = fix_batch_proof_request_fields(content)
    content = fix_crypto_parameters_construction(content)
    content = fix_proof_construction(content)
    content = add_missing_imports(content)

    print("Writing fixed content...")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Fixes applied. Running compilation check...")

    # Run cargo check
    os.chdir(r'c:\Users\sgbil\Nexuszero-Protocol\nexuszero-integration')
    os.system('cargo check --tests')

if __name__ == '__main__':
    main()