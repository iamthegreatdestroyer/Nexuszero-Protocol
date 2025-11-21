//! Mathematical primitives
//!
//! Common mathematical operations for cryptography.

use crate::{CryptoError, CryptoResult};

/// Modular exponentiation: base^exp mod modulus
pub fn modular_exponentiation(_base: &[u8], _exp: &[u8], _modulus: u64) -> CryptoResult<Vec<u8>> {
    // TODO: Implement efficient modular exponentiation
    // For now, return a placeholder
    Err(CryptoError::MathError(
        "Not yet implemented".to_string(),
    ))
}

/// Modular multiplicative inverse: find x such that (a * x) mod m = 1
pub fn mod_inverse(a: i64, m: i64) -> CryptoResult<i64> {
    // Extended Euclidean algorithm
    let (mut old_r, mut r) = (a, m);
    let (mut old_s, mut s) = (1i64, 0i64);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    if old_r > 1 {
        return Err(CryptoError::MathError(format!(
            "{} has no inverse mod {}",
            a, m
        )));
    }

    if old_s < 0 {
        old_s += m;
    }

    Ok(old_s)
}

/// Greatest common divisor
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(gcd(100, 50), 50);
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 5 = 15 ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7).unwrap();
        assert_eq!((3 * inv) % 7, 1);

        // 2 has no inverse mod 6 (not coprime)
        assert!(mod_inverse(2, 6).is_err());
    }
}
