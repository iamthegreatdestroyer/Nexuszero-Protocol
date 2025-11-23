use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexuszero_crypto::lattice::lwe::*;
use nexuszero_crypto::lattice::ring_lwe::*;
use nexuszero_crypto::params::security::SecurityLevel;
use nexuszero_crypto::proof::statement::*;
use nexuszero_crypto::proof::witness::*;
use nexuszero_crypto::proof::proof::*;
use num_bigint::BigUint;
use rand::thread_rng;

// ============================================================================
// LWE Benchmarks
// ============================================================================

fn benchmark_lwe_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("LWE Operations");
    
    for (name, n, m, q, sigma) in &[
        ("128-bit", 256, 512, 12289, 3.2),
        ("192-bit", 384, 768, 12289, 3.2),
        ("256-bit", 512, 1024, 12289, 3.2),
    ] {
        let params = LWEParameters::new(*n, *m, *q, *sigma);
        let mut rng = thread_rng();
        
        // KeyGen
        group.bench_function(
            BenchmarkId::new("KeyGen", name),
            |b| b.iter(|| {
                let mut rng = thread_rng();
                keygen(black_box(&params), &mut rng)
            })
        );
        
        // Encrypt
        let (sk, pk) = keygen(&params, &mut rng).unwrap();
        group.bench_function(
            BenchmarkId::new("Encrypt", name),
            |b| b.iter(|| {
                let mut rng = thread_rng();
                encrypt(black_box(&pk), black_box(true), black_box(&params), &mut rng)
            })
        );
        
        // Decrypt
        let (sk, pk) = keygen(&params, &mut rng).unwrap();
        let ct = encrypt(&pk, true, &params, &mut rng).unwrap();
        group.bench_function(
            BenchmarkId::new("Decrypt", name),
            |b| b.iter(|| decrypt(black_box(&sk), black_box(&ct), black_box(&params)))
        );
    }
    
    group.finish();
}

// ============================================================================
// Ring-LWE Benchmarks
// ============================================================================

fn benchmark_ring_lwe_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ring-LWE Operations");
    
    for (name, params) in &[
        ("128-bit", RingLWEParameters::new_128bit_security()),
        ("192-bit", RingLWEParameters::new_192bit_security()),
        ("256-bit", RingLWEParameters::new_256bit_security()),
    ] {
        // KeyGen
        group.bench_function(
            BenchmarkId::new("KeyGen", name),
            |b| b.iter(|| ring_keygen(black_box(params)))
        );
        
        // Encrypt
        let (sk, pk) = ring_keygen(params).unwrap();
        let message = vec![true, false, true, false, true];  // 5 bits
        group.bench_function(
            BenchmarkId::new("Encrypt", name),
            |b| b.iter(|| ring_encrypt(black_box(&pk), black_box(&message), black_box(params)))
        );
        
        // Decrypt
        let (sk, pk) = ring_keygen(params).unwrap();
        let message = vec![true, false, true, false, true];
        let ct = ring_encrypt(&pk, &message, params).unwrap();
        group.bench_function(
            BenchmarkId::new("Decrypt", name),
            |b| b.iter(|| ring_decrypt(black_box(&sk), black_box(&ct), black_box(params)))
        );
    }
    
    group.finish();
}

// ============================================================================
// Polynomial Operations Benchmarks
// ============================================================================

fn benchmark_polynomial_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial Operations");
    
    let q = 12289u64;
    for degree in &[128, 256, 512, 1024] {
        let poly_a = Polynomial {
            coeffs: vec![1i64; *degree],
            modulus: q,
            degree: *degree,
        };
        let poly_b = Polynomial {
            coeffs: vec![2i64; *degree],
            modulus: q,
            degree: *degree,
        };
        
        // Addition
        group.bench_function(
            BenchmarkId::new("Addition", degree),
            |b| b.iter(|| poly_add(black_box(&poly_a), black_box(&poly_b), q))
        );
        
        // Subtraction
        group.bench_function(
            BenchmarkId::new("Subtraction", degree),
            |b| b.iter(|| poly_sub(black_box(&poly_a), black_box(&poly_b), q))
        );
        
        // Schoolbook Multiplication
        group.bench_function(
            BenchmarkId::new("Mult-Schoolbook", degree),
            |b| b.iter(|| poly_mult_schoolbook(black_box(&poly_a), black_box(&poly_b), q))
        );
        
        // NTT Transform
        let primitive_root = find_primitive_root(*degree, q).unwrap();
        group.bench_function(
            BenchmarkId::new("NTT-Forward", degree),
            |b| b.iter(|| ntt(black_box(&poly_a), q, primitive_root))
        );
    }
    
    group.finish();
}

// ============================================================================
// Proof System Benchmarks
// ============================================================================

fn benchmark_proof_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Proof Operations");
    
    // Discrete Log Proof Generation
    let generator = vec![2u8; 32];
    let secret = vec![42u8; 32];
    let modulus_bytes = vec![0xFF; 32];
    let gen_big = BigUint::from_bytes_be(&generator);
    let secret_big = BigUint::from_bytes_be(&secret);
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
    
    let statement = StatementBuilder::new()
        .discrete_log(generator, public_value)
        .build()
        .unwrap();
    let witness = Witness::discrete_log(secret);
    
    group.bench_function("Discrete-Log-Prove", |b| {
        b.iter(|| prove(black_box(&statement), black_box(&witness)))
    });
    
    // Discrete Log Proof Verification
    let proof = prove(&statement, &witness).unwrap();
    group.bench_function("Discrete-Log-Verify", |b| {
        b.iter(|| verify(black_box(&statement), black_box(&proof)))
    });
    
    // Preimage Proof
    use sha3::{Digest, Sha3_256};
    let preimage = b"benchmark_message".to_vec();
    let mut hasher = Sha3_256::new();
    hasher.update(&preimage);
    let hash = hasher.finalize().to_vec();
    
    let preimage_statement = StatementBuilder::new()
        .preimage(HashFunction::SHA3_256, hash)
        .build()
        .unwrap();
    let preimage_witness = Witness::preimage(preimage);
    
    group.bench_function("Preimage-Prove", |b| {
        b.iter(|| prove(black_box(&preimage_statement), black_box(&preimage_witness)))
    });
    
    let preimage_proof = prove(&preimage_statement, &preimage_witness).unwrap();
    group.bench_function("Preimage-Verify", |b| {
        b.iter(|| verify(black_box(&preimage_statement), black_box(&preimage_proof)))
    });
    
    // Proof Serialization
    group.bench_function("Proof-Serialize", |b| {
        b.iter(|| proof.to_bytes())
    });
    
    // Proof Deserialization
    let serialized = proof.to_bytes().unwrap();
    group.bench_function("Proof-Deserialize", |b| {
        b.iter(|| Proof::from_bytes(black_box(&serialized)))
    });
    
    group.finish();
}

// ============================================================================
// End-to-End Workflow Benchmarks
// ============================================================================

fn benchmark_e2e_workflows(c: &mut Criterion) {
    let mut group = c.benchmark_group("End-to-End Workflows");
    
    // Complete LWE encryption workflow
    group.bench_function("LWE-Full-Workflow", |b| {
        b.iter(|| {
            let params = LWEParameters::new(256, 512, 12289, 3.2);
            let mut rng = thread_rng();
            let (sk, pk) = keygen(&params, &mut rng).unwrap();
            let ct = encrypt(&pk, true, &params, &mut rng).unwrap();
            let _pt = decrypt(&sk, &ct, &params);
        })
    });
    
    // Complete Ring-LWE encryption workflow
    group.bench_function("Ring-LWE-Full-Workflow", |b| {
        b.iter(|| {
            let params = RingLWEParameters::new_128bit_security();
            let (sk, pk) = ring_keygen(&params).unwrap();
            let message = vec![true, false, true, false, true];
            let ct = ring_encrypt(&pk, &message, &params).unwrap();
            let _dec = ring_decrypt(&sk, &ct, &params);
        })
    });
    
    // Complete proof generation and verification workflow
    group.bench_function("Proof-Full-Workflow", |b| {
        b.iter(|| {
            let generator = vec![2u8; 32];
            let secret = vec![42u8; 32];
            let modulus_bytes = vec![0xFF; 32];
            let gen_big = BigUint::from_bytes_be(&generator);
            let secret_big = BigUint::from_bytes_be(&secret);
            let mod_big = BigUint::from_bytes_be(&modulus_bytes);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator, public_value)
                .build()
                .unwrap();
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).unwrap();
            let _result = verify(&statement, &proof);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lwe_operations,
    benchmark_ring_lwe_operations,
    benchmark_polynomial_operations,
    benchmark_proof_operations,
    benchmark_e2e_workflows
);
criterion_main!(benches);
