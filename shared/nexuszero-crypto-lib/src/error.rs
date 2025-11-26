//! Cryptographic errors

use thiserror::Error;

/// Cryptographic operation errors
#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("Encryption failed")]
    EncryptionFailed,

    #[error("Decryption failed")]
    DecryptionFailed,

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Invalid key length")]
    InvalidKeyLength,

    #[error("Invalid secret key")]
    InvalidSecretKey,

    #[error("Invalid public key")]
    InvalidPublicKey,

    #[error("Signing failed")]
    SigningFailed,

    #[error("Signature invalid")]
    SignatureInvalid,

    #[error("Key derivation failed")]
    KeyDerivationFailed,

    #[error("Hash mismatch")]
    HashMismatch,

    #[error("Random generation failed")]
    RandomFailed,
}
