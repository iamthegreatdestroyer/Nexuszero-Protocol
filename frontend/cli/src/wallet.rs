//! Wallet Management for NexusZero CLI

use bip39::{Language, Mnemonic};
use ethers::signers::{LocalWallet, Signer};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{CliError, CliResult};
use crate::config::Config;

/// Wallet data stored on disk
#[derive(Debug, Serialize, Deserialize)]
pub struct WalletData {
    pub name: String,
    pub address: String,
    pub encrypted_key: String,
    pub created_at: String,
}

/// Active wallet in memory
pub struct Wallet {
    pub name: String,
    pub address: String,
    inner: LocalWallet,
}

impl Wallet {
    /// Create a new wallet with random mnemonic
    pub fn create(name: &str, password: &str) -> CliResult<(Self, String)> {
        // Generate 32 bytes of random entropy for 24-word mnemonic
        let mut entropy = [0u8; 32];
        getrandom::fill(&mut entropy)
            .map_err(|e| CliError::Wallet(format!("Failed to generate entropy: {}", e)))?;
        
        let mnemonic = Mnemonic::from_entropy(&entropy)
            .map_err(|e| CliError::Wallet(format!("Failed to generate mnemonic: {}", e)))?;

        let phrase = mnemonic.to_string();

        let wallet = Self::from_mnemonic(name, &phrase, password)?;
        Ok((wallet, phrase))
    }

    /// Import wallet from mnemonic
    pub fn from_mnemonic(name: &str, phrase: &str, _password: &str) -> CliResult<Self> {
        let mnemonic = Mnemonic::parse_in_normalized(Language::English, phrase)
            .map_err(|e| CliError::Wallet(format!("Invalid mnemonic: {}", e)))?;

        // Derive wallet from mnemonic (first account)
        let seed = mnemonic.to_seed("");
        let wallet = LocalWallet::from_bytes(&seed[..32])
            .map_err(|e| CliError::Wallet(format!("Failed to create wallet: {}", e)))?;

        let address = format!("{:?}", wallet.address());

        Ok(Self {
            name: name.to_string(),
            address,
            inner: wallet,
        })
    }

    /// Import wallet from private key
    pub fn from_private_key(name: &str, key: &str) -> CliResult<Self> {
        let key_bytes = hex::decode(key.trim_start_matches("0x"))
            .map_err(|e| CliError::Wallet(format!("Invalid private key hex: {}", e)))?;

        let wallet = LocalWallet::from_bytes(&key_bytes)
            .map_err(|e| CliError::Wallet(format!("Failed to create wallet: {}", e)))?;

        let address = format!("{:?}", wallet.address());

        Ok(Self {
            name: name.to_string(),
            address,
            inner: wallet,
        })
    }

    /// Get the underlying signer
    pub fn signer(&self) -> &LocalWallet {
        &self.inner
    }

    /// Sign a message
    pub async fn sign_message(&self, message: &[u8]) -> CliResult<String> {
        let signature = self.inner.sign_message(message).await
            .map_err(|e| CliError::Wallet(format!("Failed to sign: {}", e)))?;
        Ok(format!("0x{}", hex::encode(signature.to_vec())))
    }

    /// Save wallet to encrypted file
    pub fn save(&self, path: &PathBuf, password: &str) -> CliResult<()> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        
        // In production, use proper encryption (AES-GCM with Argon2 key derivation)
        // For now, simple base64 encoding (NOT SECURE - placeholder)
        let key_bytes = self.inner.signer().to_bytes();
        let encrypted = STANDARD.encode(&key_bytes);

        let data = WalletData {
            name: self.name.clone(),
            address: self.address.clone(),
            encrypted_key: format!("{}:{}", password.len(), encrypted), // Placeholder
            created_at: chrono_lite_timestamp(),
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load wallet from encrypted file
    pub fn load(path: &PathBuf, password: &str) -> CliResult<Self> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        
        let content = std::fs::read_to_string(path)?;
        let data: WalletData = serde_json::from_str(&content)?;

        // Parse the placeholder encryption (NOT SECURE)
        let parts: Vec<&str> = data.encrypted_key.split(':').collect();
        if parts.len() != 2 {
            return Err(CliError::Wallet("Invalid wallet file".to_string()));
        }

        let expected_len: usize = parts[0].parse()
            .map_err(|_| CliError::Wallet("Invalid wallet file".to_string()))?;

        if password.len() != expected_len {
            return Err(CliError::Wallet("Incorrect password".to_string()));
        }

        let key_bytes = STANDARD.decode(parts[1])
            .map_err(|e| CliError::Wallet(format!("Failed to decode key: {}", e)))?;

        let wallet = LocalWallet::from_bytes(&key_bytes)
            .map_err(|e| CliError::Wallet(format!("Failed to load wallet: {}", e)))?;

        Ok(Self {
            name: data.name,
            address: data.address,
            inner: wallet,
        })
    }
}

/// Get wallet directory
pub fn wallet_dir(config: &Config) -> CliResult<PathBuf> {
    match &config.wallet_path {
        Some(path) => Ok(path.clone()),
        None => {
            let data_dir = Config::default_data_dir()?;
            Ok(data_dir.join("wallets"))
        }
    }
}

/// List all wallets
pub fn list_wallets(config: &Config) -> CliResult<Vec<WalletData>> {
    let dir = wallet_dir(config)?;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut wallets = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            let content = std::fs::read_to_string(&path)?;
            if let Ok(data) = serde_json::from_str::<WalletData>(&content) {
                wallets.push(data);
            }
        }
    }

    Ok(wallets)
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_from_private_key() {
        // Known test private key (DO NOT USE IN PRODUCTION)
        let key = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
        let wallet = Wallet::from_private_key("test", key);
        assert!(wallet.is_ok());
    }

    #[test]
    fn test_wallet_from_mnemonic() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let wallet = Wallet::from_mnemonic("test", phrase, "password");
        assert!(wallet.is_ok());
    }
}
