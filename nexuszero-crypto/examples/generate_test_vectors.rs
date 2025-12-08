use nexuszero_crypto::test_vectors::generate_test_vectors;
use std::fs;
use std::path::Path;

/// Generate comprehensive test vectors for independent security audit
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 NexusZero Protocol - Test Vector Generation");
    println!("==============================================");
    println!();

    println!("Generating comprehensive test vectors for security audit...");
    println!("This may take a few moments...");
    println!();

    // Generate test vectors
    let test_vectors = generate_test_vectors()?;

    // Display summary
    println!("✅ Test Vectors Generated Successfully!");
    println!();
    println!("📊 Summary:");
    println!("  • LWE Key Generation Tests: {}", test_vectors.lwe_vectors.keygen_tests.len());
    println!("  • LWE Encrypt/Decrypt Tests: {}", test_vectors.lwe_vectors.encrypt_decrypt_tests.len());
    println!("  • LWE Soundness Tests: {}", test_vectors.lwe_vectors.soundness_tests.len());
    println!("  • Bulletproof Valid Range Proofs: {}", test_vectors.bulletproof_vectors.valid_range_proofs.len());
    println!("  • Bulletproof Invalid Tests: {}", test_vectors.bulletproof_vectors.invalid_range_proofs.len());
    println!("  • Bulletproof Edge Cases: {}", test_vectors.bulletproof_vectors.edge_case_proofs.len());
    println!("  • Schnorr Valid Proofs: {}", test_vectors.schnorr_vectors.valid_proofs.len());
    println!("  • Schnorr Invalid Proofs: {}", test_vectors.schnorr_vectors.invalid_proofs.len());
    println!("  • Schnorr Soundness Tests: {}", test_vectors.schnorr_vectors.soundness_tests.len());
    println!("  • SHA256 Hash Tests: {}", test_vectors.hash_vectors.sha256_tests.len());
    println!("  • Hash Consistency Tests: {}", test_vectors.hash_vectors.consistency_tests.len());
    println!();

    // Calculate total tests
    let total_tests = test_vectors.lwe_vectors.keygen_tests.len()
        + test_vectors.lwe_vectors.encrypt_decrypt_tests.len()
        + test_vectors.lwe_vectors.soundness_tests.len()
        + test_vectors.bulletproof_vectors.valid_range_proofs.len()
        + test_vectors.bulletproof_vectors.invalid_range_proofs.len()
        + test_vectors.bulletproof_vectors.edge_case_proofs.len()
        + test_vectors.schnorr_vectors.valid_proofs.len()
        + test_vectors.schnorr_vectors.invalid_proofs.len()
        + test_vectors.schnorr_vectors.soundness_tests.len()
        + test_vectors.hash_vectors.sha256_tests.len()
        + test_vectors.hash_vectors.consistency_tests.len();

    println!("📈 Total Test Vectors: {}", total_tests);
    println!();

    // Serialize to JSON
    println!("💾 Serializing test vectors to JSON...");
    let json_data = serde_json::to_string_pretty(&test_vectors)?;
    println!("✅ JSON serialization completed!");
    println!();

    // Create output directory if it doesn't exist
    let output_dir = Path::new("audit_materials");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
        println!("📁 Created audit_materials directory");
    }

    // Write to file
    let output_path = output_dir.join("security_test_vectors.json");
    fs::write(&output_path, json_data)?;
    println!("💾 Test vectors saved to: {}", output_path.display());
    println!();

    // Display metadata
    println!("📋 Metadata:");
    println!("  • Version: {}", test_vectors.metadata.version);
    println!("  • Generated: {}", test_vectors.metadata.generated_at);
    println!("  • Security Level: {}", test_vectors.metadata.security_level);
    println!("  • Seed: {}", &test_vectors.metadata.seed[..16]); // First 16 chars
    println!();

    // Instructions for auditors
    println!("🔍 Auditor Instructions:");
    println!("========================");
    println!("1. Verify the deterministic seed: {}", test_vectors.metadata.seed);
    println!("2. Re-run this program to confirm identical outputs");
    println!("3. Use these vectors to validate cryptographic implementations");
    println!("4. Check that all verification results match expected values");
    println!("5. Validate hash consistency and cryptographic properties");
    println!();

    println!("🎯 Next Steps:");
    println!("==============");
    println!("1. Review the security specification document");
    println!("2. Validate test vectors against implementation");
    println!("3. Perform formal verification of critical components");
    println!("4. Conduct penetration testing");
    println!("5. Issue security audit report");
    println!();

    println!("✨ Test vector generation completed successfully!");
    println!("🔒 Ready for independent security audit.");

    Ok(())
}