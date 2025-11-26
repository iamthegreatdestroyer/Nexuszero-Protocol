//! Common utilities

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Generate a new UUID v4
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Get current UTC timestamp
pub fn now() -> DateTime<Utc> {
    Utc::now()
}

/// Encode bytes to hex string
pub fn to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Decode hex string to bytes
pub fn from_hex(s: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(s)
}

/// Encode bytes to base64
pub fn to_base64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode base64 string to bytes
pub fn from_base64(s: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s)
}

/// Truncate string with ellipsis
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Mask sensitive data (show first and last 4 chars)
pub fn mask_sensitive(s: &str) -> String {
    if s.len() <= 8 {
        "*".repeat(s.len())
    } else {
        format!("{}...{}", &s[..4], &s[s.len()-4..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id() {
        let id = generate_id();
        assert_eq!(id.len(), 36);
    }

    #[test]
    fn test_hex_encoding() {
        let bytes = b"hello";
        let hex = to_hex(bytes);
        let decoded = from_hex(&hex).unwrap();
        assert_eq!(bytes.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_mask_sensitive() {
        assert_eq!(mask_sensitive("1234567890abcdef"), "1234...cdef");
        assert_eq!(mask_sensitive("short"), "*****");
    }
}
