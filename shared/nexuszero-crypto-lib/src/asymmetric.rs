//! Asymmetric cryptography

use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use x25519_dalek::{StaticSecret, PublicKey};
use rand::rngs::OsRng;
use crate::error::CryptoError;
use serde::{Deserialize, Serialize};

/// Ed25519 key pair for signing
#[derive(Clone)]
pub struct Ed25519KeyPair {
    signing_key: SigningKey,
}

impl Ed25519KeyPair {
    /// Generate new random key pair
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Create from 32-byte secret key
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        Self { signing_key }
    }

    /// Get secret key bytes
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.signing_key.sign(message).to_bytes()
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &[u8; 64]) -> Result<(), CryptoError> {
        let sig = Signature::from_bytes(signature);
        self.signing_key
            .verifying_key()
            .verify(message, &sig)
            .map_err(|_| CryptoError::SignatureInvalid)
    }
}

/// Verify signature with public key only
pub fn verify_ed25519(
    public_key: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), CryptoError> {
    let verifying_key = VerifyingKey::from_bytes(public_key)
        .map_err(|_| CryptoError::InvalidPublicKey)?;
    let sig = Signature::from_bytes(signature);
    verifying_key
        .verify(message, &sig)
        .map_err(|_| CryptoError::SignatureInvalid)
}

/// X25519 key pair for key exchange
pub struct X25519KeyPair {
    secret: StaticSecret,
    public: PublicKey,
}

impl X25519KeyPair {
    /// Generate new random key pair
    pub fn generate() -> Self {
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    /// Create from 32-byte secret key
    pub fn from_secret(bytes: [u8; 32]) -> Self {
        let secret = StaticSecret::from(bytes);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    /// Perform Diffie-Hellman key exchange
    pub fn diffie_hellman(&self, their_public: &[u8; 32]) -> [u8; 32] {
        let their_public = PublicKey::from(*their_public);
        self.secret.diffie_hellman(&their_public).to_bytes()
    }
}

/// Secp256k1 key pair (Ethereum-compatible)
#[derive(Clone)]
pub struct Secp256k1KeyPair {
    secret_key: k256::SecretKey,
}

impl Secp256k1KeyPair {
    /// Generate new random key pair
    pub fn generate() -> Self {
        let secret_key = k256::SecretKey::random(&mut OsRng);
        Self { secret_key }
    }

    /// Create from 32-byte secret key
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, CryptoError> {
        let secret_key = k256::SecretKey::from_bytes(bytes.into())
            .map_err(|_| CryptoError::InvalidSecretKey)?;
        Ok(Self { secret_key })
    }

    /// Get secret key bytes
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.secret_key.to_bytes().into()
    }

    /// Get public key bytes (compressed, 33 bytes)
    pub fn public_key_compressed(&self) -> Vec<u8> {
        use k256::elliptic_curve::sec1::ToEncodedPoint;
        self.secret_key
            .public_key()
            .to_encoded_point(true)
            .as_bytes()
            .to_vec()
    }

    /// Get public key bytes (uncompressed, 65 bytes)
    pub fn public_key_uncompressed(&self) -> Vec<u8> {
        use k256::elliptic_curve::sec1::ToEncodedPoint;
        self.secret_key
            .public_key()
            .to_encoded_point(false)
            .as_bytes()
            .to_vec()
    }

    /// Get Ethereum address (last 20 bytes of keccak256 of uncompressed public key)
    pub fn ethereum_address(&self) -> [u8; 20] {
        use sha3::{Keccak256, Digest};
        let pubkey = self.public_key_uncompressed();
        // Skip the 0x04 prefix
        let mut hasher = Keccak256::new();
        hasher.update(&pubkey[1..]);
        let hash = hasher.finalize();
        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[12..32]);
        address
    }

    /// Sign message hash (32 bytes) - returns recoverable signature
    pub fn sign(&self, message_hash: &[u8; 32]) -> Result<RecoverableSignature, CryptoError> {
        use k256::ecdsa::{SigningKey, signature::hazmat::PrehashSigner};
        
        let signing_key = SigningKey::from(&self.secret_key);
        let (signature, recovery_id) = signing_key
            .sign_prehash_recoverable(message_hash)
            .map_err(|_| CryptoError::SigningFailed)?;
        
        Ok(RecoverableSignature {
            r: signature.r().to_bytes().into(),
            s: signature.s().to_bytes().into(),
            v: recovery_id.to_byte(),
        })
    }
}

/// Recoverable signature (Ethereum-style)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverableSignature {
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub v: u8,
}

impl RecoverableSignature {
    /// Convert to 65-byte format (r || s || v)
    pub fn to_bytes(&self) -> [u8; 65] {
        let mut bytes = [0u8; 65];
        bytes[0..32].copy_from_slice(&self.r);
        bytes[32..64].copy_from_slice(&self.s);
        bytes[64] = self.v;
        bytes
    }

    /// Parse from 65-byte format
    pub fn from_bytes(bytes: &[u8; 65]) -> Self {
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        Self {
            r,
            s,
            v: bytes[64],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_sign_verify() {
        let keypair = Ed25519KeyPair::generate();
        let message = b"Hello, World!";
        
        let signature = keypair.sign(message);
        assert!(keypair.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_x25519_key_exchange() {
        let alice = X25519KeyPair::generate();
        let bob = X25519KeyPair::generate();
        
        let alice_shared = alice.diffie_hellman(&bob.public_key_bytes());
        let bob_shared = bob.diffie_hellman(&alice.public_key_bytes());
        
        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_secp256k1_ethereum_address() {
        let keypair = Secp256k1KeyPair::generate();
        let address = keypair.ethereum_address();
        assert_eq!(address.len(), 20);
    }
}
