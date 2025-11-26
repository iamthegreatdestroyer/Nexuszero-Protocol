/// Example: Advanced Parameter Selection
/// Demonstrates the ParameterSelector with various configurations:
/// - Basic parameter generation for different security levels
/// - Custom constraints (dimension, modulus ranges)
/// - Prime modulus selection using Miller-Rabin
/// - Security level estimation
use nexuszero_crypto::{
    params::{ParameterSelector, SecurityLevel, is_prime_miller_rabin, generate_prime},
    CryptoResult,
};

fn main() -> CryptoResult<()> {
    println!("=== NexusZero Parameter Selection Examples ===\n");

    // Example 1: Basic LWE parameter selection
    basic_lwe_selection()?;

    // Example 2: Basic Ring-LWE parameter selection
    basic_ring_lwe_selection()?;

    // Example 3: Custom constraints
    custom_constraints()?;

    // Example 4: Prime modulus selection
    prime_modulus_selection()?;

    // Example 5: Security estimation
    security_estimation();

    // Example 6: Primality testing
    primality_testing();

    // Example 7: Prime generation
    prime_generation()?;

    Ok(())
}

fn basic_lwe_selection() -> CryptoResult<()> {
    println!("=== Example 1: Basic LWE Parameter Selection ===");
    
    for &security_level in &[SecurityLevel::Bit128, SecurityLevel::Bit192, SecurityLevel::Bit256] {
        let selector = ParameterSelector::new()
            .target_security(security_level);
        
        let params = selector.build_lwe()?;
        
        println!("Security Level: {:?}", security_level);
        println!("  n (dimension): {}", params.n);
        println!("  m (samples):   {}", params.m);
        println!("  q (modulus):   {}", params.q);
        println!("  σ (sigma):     {:.2}", params.sigma);
        
        // Estimate actual security
        let estimated_security = ParameterSelector::estimate_security(
            params.n, 
            params.q, 
            params.sigma
        );
        println!("  Estimated security: {} bits\n", estimated_security);
    }
    
    Ok(())
}

fn basic_ring_lwe_selection() -> CryptoResult<()> {
    println!("=== Example 2: Basic Ring-LWE Parameter Selection ===");
    
    for &security_level in &[SecurityLevel::Bit128, SecurityLevel::Bit192, SecurityLevel::Bit256] {
        let selector = ParameterSelector::new()
            .target_security(security_level);
        
        let params = selector.build_ring_lwe()?;
        
        println!("Security Level: {:?}", security_level);
        println!("  n (degree):    {}", params.n);
        println!("  q (modulus):   {}", params.q);
        println!("  σ (sigma):     {:.2}", params.sigma);
        println!("  Power of 2:    {}", params.n.is_power_of_two());
        
        let estimated_security = ParameterSelector::estimate_security(
            params.n, 
            params.q, 
            params.sigma
        );
        println!("  Estimated security: {} bits\n", estimated_security);
    }
    
    Ok(())
}

fn custom_constraints() -> CryptoResult<()> {
    println!("=== Example 3: Custom Constraints ===");
    
    // Scenario: Need at least 512-bit dimension but no more than 1024
    let selector = ParameterSelector::new()
        .target_security(SecurityLevel::Bit128)
        .min_dimension(512)
        .max_dimension(1024);
    
    let params = selector.build_lwe()?;
    
    println!("Constrained LWE Parameters:");
    println!("  n (dimension): {} (within [512, 1024])", params.n);
    println!("  m (samples):   {}", params.m);
    println!("  q (modulus):   {}", params.q);
    println!("  σ (sigma):     {:.2}\n", params.sigma);
    
    // Scenario: Custom m/n ratio
    let selector = ParameterSelector::new()
        .target_security(SecurityLevel::Bit128)
        .custom_ratio(3.0); // More samples for extra security
    
    let params = selector.build_lwe()?;
    
    println!("Custom Ratio (m/n = 3.0):");
    println!("  n: {}", params.n);
    println!("  m: {} (3× dimension)", params.m);
    println!("  Ratio: {:.2}\n", params.m as f64 / params.n as f64);
    
    Ok(())
}

fn prime_modulus_selection() -> CryptoResult<()> {
    println!("=== Example 4: Prime Modulus Selection ===");
    
    // Without prime preference
    let selector = ParameterSelector::new()
        .target_security(SecurityLevel::Bit128);
    
    let params = selector.build_lwe()?;
    
    println!("Standard modulus:");
    println!("  q: {}", params.q);
    println!("  Is prime: {}\n", is_prime_miller_rabin(params.q, 20));
    
    // With prime preference
    let selector = ParameterSelector::new()
        .target_security(SecurityLevel::Bit128)
        .prefer_prime_modulus(true);
    
    let params = selector.build_lwe()?;
    
    println!("Prime modulus:");
    println!("  q: {}", params.q);
    println!("  Is prime: {} (verified with Miller-Rabin)\n", 
        is_prime_miller_rabin(params.q, 20));
    
    Ok(())
}

fn security_estimation() {
    println!("=== Example 5: Security Estimation ===");
    
    // Compare different parameter sets
    let test_cases = vec![
        (256, 12289, 3.2, "Small (128-bit target)"),
        (384, 16411, 3.8, "Medium (192-bit target)"),
        (512, 20483, 4.0, "Large (256-bit target)"),
        (1024, 40961, 4.5, "Extra Large"),
    ];
    
    for (n, q, sigma, description) in test_cases {
        let security = ParameterSelector::estimate_security(n, q, sigma);
        println!("{}", description);
        println!("  Parameters: n={}, q={}, σ={:.1}", n, q, sigma);
        println!("  Estimated security: {} bits\n", security);
    }
}

fn primality_testing() {
    println!("=== Example 6: Primality Testing (Miller-Rabin) ===");
    
    let test_numbers = vec![
        (2, "Small prime"),
        (97, "Two-digit prime"),
        (541, "Three-digit prime"),
        (7919, "Four-digit prime"),
        (12289, "Common in lattice crypto"),
        (100, "Composite"),
        (1001, "Composite (7 × 11 × 13)"),
    ];
    
    for (n, description) in test_numbers {
        let is_prime = is_prime_miller_rabin(n, 20);
        let result = if is_prime { "PRIME" } else { "COMPOSITE" };
        println!("{:5} - {} [{}]", n, description, result);
    }
    println!();
}

fn prime_generation() -> CryptoResult<()> {
    println!("=== Example 7: Prime Generation ===");
    
    for bit_length in &[10, 12, 14, 16] {
        let prime = generate_prime(*bit_length)?;
        let min = 1u64 << (bit_length - 1);
        let max = (1u64 << bit_length) - 1;
        
        println!("{}-bit prime: {}", bit_length, prime);
        println!("  Range: [{}, {}]", min, max);
        println!("  Verified: {}\n", is_prime_miller_rabin(prime, 20));
    }
    
    Ok(())
}
