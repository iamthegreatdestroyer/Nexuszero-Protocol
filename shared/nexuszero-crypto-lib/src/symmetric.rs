//! Symmetric encryption

use aes_gcm::{
    Aes256Gcm, 
    aead::{Aead, KeyInit, OsRng, generic_array::GenericArray},
};
use chacha20poly1305::ChaCha20Poly1305;
use crate::error::CryptoError;
use rand::RngCore;

/// AES-256-GCM cipher
pub struct AesGcm {
    cipher: Aes256Gcm,
}

impl AesGcm {
    /// Create from 32-byte key
    pub fn new(key: &[u8; 32]) -> Self {
        let key = GenericArray::from_slice(key);
        Self {
            cipher: Aes256Gcm::new(key),
        }
    }

    /// Generate a random key
    pub fn generate_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        key
    }

    /// Encrypt with random nonce
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = GenericArray::from_slice(&nonce_bytes);
        
        let ciphertext = self.cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| CryptoError::EncryptionFailed)?;
        
        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt (nonce prepended to ciphertext)
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if data.len() < 12 {
            return Err(CryptoError::InvalidData("Data too short".into()));
        }
        
        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = GenericArray::from_slice(nonce_bytes);
        
        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::DecryptionFailed)
    }
}

/// ChaCha20-Poly1305 cipher
pub struct ChaCha20Poly1305Cipher {
    cipher: ChaCha20Poly1305,
}

impl ChaCha20Poly1305Cipher {
    /// Create from 32-byte key
    pub fn new(key: &[u8; 32]) -> Self {
        let key = GenericArray::from_slice(key);
        Self {
            cipher: ChaCha20Poly1305::new(key),
        }
    }

    /// Generate a random key
    pub fn generate_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        key
    }

    /// Encrypt with random nonce
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = GenericArray::from_slice(&nonce_bytes);
        
        let ciphertext = self.cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| CryptoError::EncryptionFailed)?;
        
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt (nonce prepended to ciphertext)
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if data.len() < 12 {
            return Err(CryptoError::InvalidData("Data too short".into()));
        }
        
        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = GenericArray::from_slice(nonce_bytes);
        
        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::DecryptionFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_gcm_roundtrip() {
        let key = AesGcm::generate_key();
        let cipher = AesGcm::new(&key);
        
        let plaintext = b"Hello, World!";
        let ciphertext = cipher.encrypt(plaintext).unwrap();
        let decrypted = cipher.decrypt(&ciphertext).unwrap();
        
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_chacha20_roundtrip() {
        let key = ChaCha20Poly1305Cipher::generate_key();
        let cipher = ChaCha20Poly1305Cipher::new(&key);
        
        let plaintext = b"Hello, World!";
        let ciphertext = cipher.encrypt(plaintext).unwrap();
        let decrypted = cipher.decrypt(&ciphertext).unwrap();
        
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }
}
