//! NexusZero Crypto Library - Cryptographic primitives
//!
//! This crate provides cryptographic primitives used across NexusZero services:
//! - Hashing (SHA-256, SHA-3, BLAKE3)
//! - Symmetric encryption (AES-GCM, ChaCha20-Poly1305)
//! - Asymmetric cryptography (Ed25519, X25519, secp256k1)
//! - Key derivation
//! - Secure random generation

pub mod hash;
pub mod symmetric;
pub mod asymmetric;
pub mod random;
pub mod error;

pub use error::CryptoError;
pub use hash::*;
pub use symmetric::*;
pub use asymmetric::*;
pub use random::*;

/// Re-export common crypto types
pub mod prelude {
    pub use super::hash::{sha256, sha3_256, blake3_hash};
    pub use super::symmetric::{AesGcm, ChaCha20Poly1305Cipher};
    pub use super::asymmetric::{Ed25519KeyPair, X25519KeyPair};
    pub use super::random::{random_bytes, random_u64};
    pub use super::CryptoError;
}
