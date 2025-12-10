// Quick verification of Shamir's trick implementation
use num_bigint::{BigUint, ToBigUint};

fn naive_dual_exp(a: &BigUint, x: &BigUint, b: &BigUint, y: &BigUint, modulus: &BigUint) -> BigUint {
    // Compute a^x
    let mut result_a = BigUint::one();
    for _ in 0..x.to_u64().unwrap_or(0) {
        result_a = (&result_a * a) % modulus;
    }
    
    // Compute b^y
    let mut result_b = BigUint::one();
    for _ in 0..y.to_u64().unwrap_or(0) {
        result_b = (&result_b * b) % modulus;
    }
    
    // Multiply
    (&result_a * &result_b) % modulus
}

fn main() {
    let a = 2u32.to_biguint().unwrap();
    let x = 3u32.to_biguint().unwrap();
    let b = 3u32.to_biguint().unwrap();
    let y = 2u32.to_biguint().unwrap();
    let modulus = 7u32.to_biguint().unwrap();

    let expected = naive_dual_exp(&a, &x, &b, &y, &modulus);
    println!("2^3 * 3^2 mod 7 = {}", expected);
    
    // Let me manually verify:
    // 2^3 = 8 mod 7 = 1
    // 3^2 = 9 mod 7 = 2
    // 1 * 2 = 2
    println!("Manual: 2^3 mod 7 = {}, 3^2 mod 7 = 2, result = 2");
}
