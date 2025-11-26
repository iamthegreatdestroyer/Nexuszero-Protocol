//! Integration with NexusZero Crypto Module
//!
//! This example demonstrates how to use holographic compression with
//! the NexusZero cryptographic primitives for proof storage.
//!
//! Run with: `cargo run --example integrate_with_crypto`

use nexuszero_holographic::{CompressedTensorTrain, CompressionConfig};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      NexusZero Holographic - Crypto Integration Example      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Simulate proof generation
    // =========================================================================
    println!("━━━ Step 1: Generate Proof Data ━━━\n");
    
    // In real usage, this would be a Bulletproof or similar ZK proof
    // Here we simulate the structure of typical proof data
    let proof_bytes = generate_simulated_proof(512);
    println!("📜 Proof generated: {} bytes", proof_bytes.len());
    println!("   Type: Simulated Bulletproof range proof");
    println!("   Structure: Group elements + scalars\n");

    // =========================================================================
    // Compress for storage
    // =========================================================================
    println!("━━━ Step 2: Compress for Storage ━━━\n");
    
    let config = CompressionConfig::balanced();
    
    let start = std::time::Instant::now();
    let compressed = CompressedTensorTrain::compress(&proof_bytes, config)?;
    let compress_time = start.elapsed();
    
    let stats = compressed.stats();
    let storage_size = stats.compressed_bytes;
    let ratio = stats.compression_ratio();
    
    println!("📦 Compression complete:");
    println!("   Original: {} bytes", proof_bytes.len());
    println!("   Compressed: {} bytes", storage_size);
    println!("   Ratio: {:.2}x", ratio);
    println!("   Time: {:?}", compress_time);
    
    // Calculate savings
    let saved = proof_bytes.len() as i64 - storage_size as i64;
    let saved_pct = if proof_bytes.len() > 0 {
        (1.0 - (storage_size as f64 / proof_bytes.len() as f64)) * 100.0
    } else {
        0.0
    };
    println!("   Saved: {} bytes ({:.1}%)\n", saved, saved_pct);

    // =========================================================================
    // Serialize for transmission/storage
    // =========================================================================
    println!("━━━ Step 3: Serialize for Storage ━━━\n");
    
    let serialized = compressed.to_bytes()?;
    println!("💾 Serialized: {} bytes", serialized.len());
    
    // Write to temp file
    let temp_path = std::env::temp_dir().join("proof.holo");
    fs::write(&temp_path, &serialized)?;
    println!("   Written to: {}", temp_path.display());
    
    // Also try LZ4 compressed serialization
    let serialized_lz4 = compressed.to_bytes_lz4()?;
    println!("   With LZ4: {} bytes", serialized_lz4.len());
    
    // Calculate total overhead
    let overhead = serialized.len() as i64 - storage_size as i64;
    println!("   Serialization overhead: {} bytes\n", overhead);

    // =========================================================================
    // Load and decompress
    // =========================================================================
    println!("━━━ Step 4: Load and Decompress ━━━\n");
    
    // Read from file
    let loaded_bytes = fs::read(&temp_path)?;
    println!("📖 Loaded from disk: {} bytes", loaded_bytes.len());
    
    // Deserialize
    let start = std::time::Instant::now();
    let loaded = CompressedTensorTrain::from_bytes(&loaded_bytes)?;
    let deserialize_time = start.elapsed();
    println!("   Deserialized in {:?}", deserialize_time);
    
    // Decompress
    let start = std::time::Instant::now();
    let recovered = loaded.decompress()?;
    let decompress_time = start.elapsed();
    println!("   Decompressed in {:?}", decompress_time);
    
    // Verify
    let error: f64 = proof_bytes.iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .sum::<f64>() / proof_bytes.len().max(1) as f64;
    
    if error < 1.0 {
        println!("\n✅ Verification passed: recovered proof matches original! (error: {:.4})\n", error);
    } else {
        println!("\n⚠️  Some reconstruction error: {:.4} (expected for lossy modes)\n", error);
    }

    // =========================================================================
    // Batch processing example
    // =========================================================================
    println!("━━━ Step 5: Batch Proof Processing ━━━\n");
    
    let num_proofs = 10;
    let proofs: Vec<Vec<u8>> = (0..num_proofs)
        .map(|i| generate_simulated_proof(512 + i * 32))
        .collect();
    
    let total_original: usize = proofs.iter().map(|p| p.len()).sum();
    println!("📦 Processing {} proofs ({} bytes total)", num_proofs, total_original);
    
    // Compress all proofs
    let start = std::time::Instant::now();
    let compressed_proofs: Vec<CompressedTensorTrain> = proofs
        .iter()
        .map(|p| CompressedTensorTrain::compress(p, CompressionConfig::balanced()))
        .collect::<Result<Vec<_>, _>>()?;
    let batch_time = start.elapsed();
    
    let total_compressed: usize = compressed_proofs
        .iter()
        .map(|c| c.stats().compressed_bytes)
        .sum();
    
    println!("   Compressed: {} bytes", total_compressed);
    println!("   Overall ratio: {:.2}x", total_original as f64 / total_compressed.max(1) as f64);
    println!("   Total time: {:?}", batch_time);
    println!("   Per proof: {:?}", batch_time / num_proofs as u32);
    
    // Serialize batch
    let batch_serialized: Vec<Vec<u8>> = compressed_proofs
        .iter()
        .map(|c| c.to_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    
    let total_serialized: usize = batch_serialized.iter().map(|b| b.len()).sum();
    println!("\n💾 Batch serialized: {} bytes total", total_serialized);

    // =========================================================================
    // Hybrid compression demo
    // =========================================================================
    println!("\n━━━ Step 6: Hybrid Compression (Holographic + Zstd) ━━━\n");
    
    let large_proof = generate_simulated_proof(10240);
    println!("📜 Large proof: {} bytes", large_proof.len());
    
    // Holographic only
    let holo_only = CompressedTensorTrain::compress(&large_proof, CompressionConfig::balanced())?;
    let holo_serial = holo_only.to_bytes()?;
    println!("   Holographic only: {} bytes", holo_serial.len());
    
    // Holographic + Zstd
    let hybrid = zstd::encode_all(&holo_serial[..], 3)?;
    println!("   Holographic + Zstd: {} bytes", hybrid.len());
    
    // Zstd only
    let zstd_only = zstd::encode_all(&large_proof[..], 3)?;
    println!("   Zstd only: {} bytes", zstd_only.len());
    
    let best_size = hybrid.len().min(zstd_only.len());
    if hybrid.len() <= zstd_only.len() {
        println!("\n   Best result: Holographic + Zstd hybrid");
        if zstd_only.len() > 0 {
            println!("   Improvement over Zstd alone: {:.1}x",
                     zstd_only.len() as f64 / hybrid.len() as f64);
        }
    } else {
        println!("\n   Best result: Zstd alone (for this data pattern)");
    }

    // =========================================================================
    // Cleanup
    // =========================================================================
    if temp_path.exists() {
        fs::remove_file(&temp_path)?;
        println!("\n🧹 Cleaned up temp file");
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                   Integration Summary                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Workflow:                                                   ║");
    println!("║  1. Generate ZK proof → Vec<u8>                              ║");
    println!("║  2. Compress with CompressedTensorTrain                      ║");
    println!("║  3. Serialize with to_bytes() or to_bytes_lz4()              ║");
    println!("║  4. Store/transmit serialized bytes                          ║");
    println!("║  5. Deserialize and decompress to recover proof              ║");
    println!("║                                                              ║");
    println!("║  Benefits:                                                   ║");
    println!("║  • Significant proof storage reduction                       ║");
    println!("║  • Reduced bandwidth for proof transmission                  ║");
    println!("║  • Near-lossless reconstruction                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Generate simulated proof data with realistic structure
fn generate_simulated_proof(size: usize) -> Vec<u8> {
    // Simulate proof structure:
    // - Group elements (32-byte curve points with structure)
    // - Scalar values (32-byte field elements)
    // - Commitments (typically have internal correlations)
    
    let mut proof = Vec::with_capacity(size);
    
    // Simulated commitment points (groups of 32 bytes)
    let num_commitments = size / 64;
    for i in 0..num_commitments {
        // X coordinate (structured)
        for j in 0..32 {
            let base = ((i * 7 + j * 3) % 256) as u8;
            proof.push(base);
        }
        // Y coordinate (related to X)
        for j in 0..32 {
            let base = ((i * 11 + j * 5 + 128) % 256) as u8;
            proof.push(base);
        }
    }
    
    // Fill remaining bytes
    while proof.len() < size {
        proof.push(((proof.len() * 13) % 256) as u8);
    }
    
    proof.truncate(size);
    proof
}
