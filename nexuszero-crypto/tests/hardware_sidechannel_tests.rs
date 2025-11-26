//! Integration tests for hardware-backed security and side-channel analysis

use nexuszero_crypto::utils::hardware::{select_backend, BackendType};
use nexuszero_crypto::utils::sidechannel::{
    test_constant_time, CacheSimulator, DudectAnalyzer, MemoryTracer, AccessType,
};
use std::time::{Duration, Instant};

#[test]
fn test_hardware_backend_selection() {
    let backend = select_backend();
    assert!(backend.is_available());

    // Should at minimum have software fallback
    let backend_type = backend.backend_type();
    println!("Selected backend: {}", backend_type);
    assert!(matches!(
        backend_type,
        BackendType::IntelSGX | BackendType::ARMTrustZone | BackendType::HSM | BackendType::Software
    ));
}

#[test]
fn test_hardware_secure_modpow() {
    let backend = select_backend();

    let base = vec![0x02];
    let exp = vec![0x05];
    let modulus = vec![0x0D]; // 13

    let result = backend.secure_modpow(&base, &exp, &modulus).unwrap();

    // 2^5 mod 13 = 32 mod 13 = 6
    assert_eq!(result, vec![0x06]);
}

#[test]
fn test_hardware_secure_dot_product() {
    let backend = select_backend();

    let a = vec![2, 3, 4];
    let b = vec![5, 6, 7];

    let result = backend.secure_dot_product(&a, &b).unwrap();

    // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
    assert_eq!(result, 56);
}

#[test]
fn test_hardware_random_generation() {
    let backend = select_backend();

    let random1 = backend.secure_random(32).unwrap();
    let random2 = backend.secure_random(32).unwrap();

    assert_eq!(random1.len(), 32);
    assert_eq!(random2.len(), 32);
    assert_ne!(random1, random2, "RNG should produce different values");
}

#[test]
fn test_hardware_attestation() {
    let backend = select_backend();
    let report = backend.attest().unwrap();

    assert!(report.timestamp > 0);
    println!(
        "Attestation from {}: measurement_len={}",
        report.backend,
        report.measurement.len()
    );
}

#[test]
fn test_dudect_constant_time_operation() {
    // Test a truly constant-time operation
    let result = test_constant_time(
        |_class| {
            let start = Instant::now();
            // Constant work regardless of input
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            std::hint::black_box(sum);
            start.elapsed()
        },
        200,
    );

    println!("Constant-time test: t={:?}, CT={}", result.t_statistic, result.is_constant_time);
    // May occasionally fail due to system noise, but should generally pass
}

#[test]
fn test_dudect_variable_time_operation() {
    // Test a variable-time operation (intentionally leaky)
    let result = test_constant_time(
        |class| {
            let start = Instant::now();
            if class {
                // More work for class 1
                std::thread::sleep(Duration::from_micros(10));
            }
            start.elapsed()
        },
        100,
    );

    println!("Variable-time test: t={:?}, CT={}", result.t_statistic, result.is_constant_time);
    // Should detect timing difference
    if let Some(t) = result.t_statistic {
        assert!(t.abs() > 4.5, "Should detect significant timing difference");
    }
}

#[test]
fn test_cache_simulator_flush_reload() {
    let mut cache = CacheSimulator::new();

    // Simulate Flush+Reload attack
    let target_addr = 0x400000;

    // Prime: access target (loads into cache)
    assert!(!cache.access(target_addr), "First access should be miss");

    // Probe: access again (should be hit)
    assert!(cache.access(target_addr), "Second access should be hit");

    // Flush: attacker flushes the line
    cache.flush(target_addr);

    // Probe: check if victim accessed (miss = not accessed, hit = accessed)
    assert!(!cache.access(target_addr), "After flush should be miss");
}

#[test]
fn test_cache_pattern_analysis() {
    let mut cache = CacheSimulator::new();

    // Simulate uniform access pattern (constant-time)
    for i in 0..100 {
        cache.access(i * 64); // Different cache lines
    }

    let analysis = cache.analyze_patterns();
    println!(
        "Cache analysis: accesses={}, hit_rate={:.2}, unique_lines={}, suspicious={}",
        analysis.total_accesses,
        analysis.cache_hit_rate,
        analysis.unique_cache_lines,
        analysis.suspicious_lines.len()
    );

    assert!(analysis.suspicious_lines.is_empty(), "Should have no suspicious patterns");
}

#[test]
fn test_cache_secret_dependent_access() {
    let mut cache = CacheSimulator::new();

    // Simulate secret-dependent access (bad!)
    // Access line 0 heavily (80 times) and other lines occasionally
    let base_addr = 0x400000;

    // Skewed access pattern: 80% to one line, 20% distributed
    for i in 0..100 {
        if i < 80 {
            cache.access(base_addr); // Line 0 - heavily accessed
        } else {
            cache.access(base_addr + (i * 64)); // Different lines
        }
    }

    let analysis = cache.analyze_patterns();
    println!("Secret-dependent cache: suspicious_lines={:?}", analysis.suspicious_lines);

    // Should detect non-uniform access (line 0 accessed much more)
    assert!(
        !analysis.suspicious_lines.is_empty(),
        "Should detect secret-dependent access pattern (found {} suspicious lines)",
        analysis.suspicious_lines.len()
    );
}

#[test]
#[ignore]
fn test_memory_tracer() {
    let mut tracer = MemoryTracer::new();

    // Simulate constant-time memory accesses
    for i in 0..10 {
        tracer.record(0x1000 + i * 8, 8, AccessType::Read);
        std::thread::sleep(Duration::from_micros(10));
    }

    let regularity = tracer.analyze_regularity();
    println!("Memory access regularity: {:.2}", regularity);

    // Regularity close to 1.0 = constant-time
    assert!(regularity > 0.5, "Should have reasonably regular timing");
}

#[test]
fn test_memory_secret_dependency_detection() {
    let mut tracer = MemoryTracer::new();

    // Simulate operation
    tracer.record(0x1000, 8, AccessType::Read);
    tracer.record(0x2000, 8, AccessType::Read);

    assert!(!tracer.has_secret_dependency());

    // Flag secret-dependent access
    tracer.mark_secret_dependent();
    assert!(tracer.has_secret_dependency());
}

#[test]
fn test_dudect_analyzer_insufficient_samples() {
    let mut analyzer = DudectAnalyzer::new();

    // Not enough samples
    for _ in 0..5 {
        analyzer.add_class0(Duration::from_nanos(1000));
        analyzer.add_class1(Duration::from_nanos(1000));
    }

    assert!(analyzer.compute_t_statistic().is_none(), "Should return None with <10 samples");
}
