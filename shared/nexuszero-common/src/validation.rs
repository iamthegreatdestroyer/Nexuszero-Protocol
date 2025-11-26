//! Input validation utilities

use validator::Validate;
use crate::error::AppError;

/// Validate a struct that implements Validate trait
pub fn validate<T: Validate>(input: &T) -> Result<(), AppError> {
    input.validate().map_err(|e| {
        AppError::Validation(format!("{}", e))
    })
}

/// Validate Ethereum address format
pub fn validate_eth_address(address: &str) -> bool {
    if !address.starts_with("0x") {
        return false;
    }
    let hex_part = &address[2..];
    if hex_part.len() != 40 {
        return false;
    }
    hex_part.chars().all(|c| c.is_ascii_hexdigit())
}

/// Validate Bitcoin address format (simplified)
pub fn validate_btc_address(address: &str) -> bool {
    // Simplified validation - checks length and prefix
    let len = address.len();
    (len >= 26 && len <= 35) || // Legacy P2PKH/P2SH
    (address.starts_with("bc1") && len >= 42 && len <= 62) // Bech32
}

/// Validate UUID format
pub fn validate_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

/// Validate positive amount string
pub fn validate_amount(s: &str) -> bool {
    match s.parse::<f64>() {
        Ok(v) => v > 0.0,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eth_address_validation() {
        assert!(validate_eth_address("0x742d35Cc6634C0532925a3b844Bc9e7595f5aAFe"));
        assert!(!validate_eth_address("742d35Cc6634C0532925a3b844Bc9e7595f5aAFe"));
        assert!(!validate_eth_address("0x742d35"));
    }

    #[test]
    fn test_uuid_validation() {
        assert!(validate_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!validate_uuid("not-a-uuid"));
    }

    #[test]
    fn test_amount_validation() {
        assert!(validate_amount("100.5"));
        assert!(!validate_amount("-50"));
        assert!(!validate_amount("abc"));
    }
}
