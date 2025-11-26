//! Secure random number generation

use rand::{RngCore, rngs::OsRng};

/// Generate random bytes
pub fn random_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    OsRng.fill_bytes(&mut bytes);
    bytes
}

/// Generate 32 random bytes
pub fn random_32() -> [u8; 32] {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    bytes
}

/// Generate 64 random bytes  
pub fn random_64() -> [u8; 64] {
    let mut bytes = [0u8; 64];
    OsRng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random u64
pub fn random_u64() -> u64 {
    OsRng.next_u64()
}

/// Generate random u32
pub fn random_u32() -> u32 {
    OsRng.next_u32()
}

/// Generate random hex string
pub fn random_hex(byte_len: usize) -> String {
    hex::encode(random_bytes(byte_len))
}

/// Generate secure nonce (12 bytes for GCM)
pub fn generate_nonce() -> [u8; 12] {
    let mut nonce = [0u8; 12];
    OsRng.fill_bytes(&mut nonce);
    nonce
}

/// Generate secure IV (16 bytes)
pub fn generate_iv() -> [u8; 16] {
    let mut iv = [0u8; 16];
    OsRng.fill_bytes(&mut iv);
    iv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_bytes() {
        let bytes1 = random_bytes(32);
        let bytes2 = random_bytes(32);
        assert_ne!(bytes1, bytes2);
    }

    #[test]
    fn test_random_hex() {
        let hex = random_hex(16);
        assert_eq!(hex.len(), 32); // 16 bytes = 32 hex chars
    }
}
