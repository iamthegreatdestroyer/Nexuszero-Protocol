//! JWT Authentication module
//!
//! Provides JWT token generation and validation for user authentication

use crate::config::JwtConfig;
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// JWT Claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Not before (Unix timestamp)
    pub nbf: i64,

    /// Issuer
    pub iss: String,

    /// JWT ID (unique identifier for this token)
    pub jti: String,

    /// Token type (access or refresh)
    pub token_type: TokenType,

    /// User roles
    pub roles: Vec<String>,

    /// Custom claims
    #[serde(flatten)]
    pub custom: CustomClaims,
}

/// Token type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Access,
    Refresh,
}

/// Custom claims for NexusZero
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CustomClaims {
    /// User's public key hash
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pubkey_hash: Option<String>,

    /// Default privacy level preference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_privacy_level: Option<u8>,

    /// Session ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// JWT token pair (access + refresh)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
}

/// JWT token manager
pub struct JwtManager {
    config: JwtConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JwtManager {
    /// Create a new JWT manager
    pub fn new(config: &JwtConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());

        Self {
            config: config.clone(),
            encoding_key,
            decoding_key,
        }
    }

    /// Generate a token pair for a user
    pub fn generate_token_pair(
        &self,
        user_id: &str,
        roles: Vec<String>,
        custom: CustomClaims,
    ) -> Result<TokenPair, jsonwebtoken::errors::Error> {
        let access_token = self.generate_access_token(user_id, roles.clone(), custom.clone())?;
        let refresh_token = self.generate_refresh_token(user_id, roles)?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            token_type: "Bearer".to_string(),
            expires_in: self.config.access_token_expiry_secs as i64,
        })
    }

    /// Generate an access token
    pub fn generate_access_token(
        &self,
        user_id: &str,
        roles: Vec<String>,
        custom: CustomClaims,
    ) -> Result<String, jsonwebtoken::errors::Error> {
        let now = Utc::now();
        let expiry = now + Duration::seconds(self.config.access_token_expiry_secs as i64);

        let claims = Claims {
            sub: user_id.to_string(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            nbf: now.timestamp(),
            iss: self.config.issuer.clone(),
            jti: Uuid::new_v4().to_string(),
            token_type: TokenType::Access,
            roles,
            custom,
        };

        encode(&Header::default(), &claims, &self.encoding_key)
    }

    /// Generate a refresh token
    pub fn generate_refresh_token(
        &self,
        user_id: &str,
        roles: Vec<String>,
    ) -> Result<String, jsonwebtoken::errors::Error> {
        let now = Utc::now();
        let expiry = now + Duration::seconds(self.config.refresh_token_expiry_secs as i64);

        let claims = Claims {
            sub: user_id.to_string(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            nbf: now.timestamp(),
            iss: self.config.issuer.clone(),
            jti: Uuid::new_v4().to_string(),
            token_type: TokenType::Refresh,
            roles,
            custom: CustomClaims::default(),
        };

        encode(&Header::default(), &claims, &self.encoding_key)
    }

    /// Validate and decode a token
    pub fn validate_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let mut validation = Validation::default();
        validation.set_issuer(&[&self.config.issuer]);

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)?;
        Ok(token_data.claims)
    }

    /// Validate specifically an access token
    pub fn validate_access_token(&self, token: &str) -> Result<Claims, crate::error::ApiError> {
        let claims = self.validate_token(token)?;

        if claims.token_type != TokenType::Access {
            return Err(crate::error::ApiError::InvalidToken);
        }

        Ok(claims)
    }

    /// Validate specifically a refresh token
    pub fn validate_refresh_token(&self, token: &str) -> Result<Claims, crate::error::ApiError> {
        let claims = self.validate_token(token)?;

        if claims.token_type != TokenType::Refresh {
            return Err(crate::error::ApiError::InvalidToken);
        }

        Ok(claims)
    }

    /// Extract token from Authorization header
    pub fn extract_token(auth_header: &str) -> Option<&str> {
        if auth_header.starts_with("Bearer ") {
            Some(&auth_header[7..])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JwtConfig;

    fn test_config() -> JwtConfig {
        JwtConfig {
            secret: "test_secret_key_for_jwt_testing_32bytes!".to_string(),
            access_token_expiry_secs: 3600,
            refresh_token_expiry_secs: 604800,
            issuer: "test-issuer".to_string(),
        }
    }

    #[test]
    fn test_generate_and_validate_access_token() {
        let config = test_config();
        let manager = JwtManager::new(&config);

        let token = manager
            .generate_access_token(
                "user-123",
                vec!["user".to_string()],
                CustomClaims::default(),
            )
            .unwrap();

        let claims = manager.validate_access_token(&token).unwrap();
        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.token_type, TokenType::Access);
        assert!(claims.roles.contains(&"user".to_string()));
    }

    #[test]
    fn test_generate_and_validate_refresh_token() {
        let config = test_config();
        let manager = JwtManager::new(&config);

        let token = manager
            .generate_refresh_token("user-123", vec!["user".to_string()])
            .unwrap();

        let claims = manager.validate_refresh_token(&token).unwrap();
        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.token_type, TokenType::Refresh);
    }

    #[test]
    fn test_token_pair_generation() {
        let config = test_config();
        let manager = JwtManager::new(&config);

        let pair = manager
            .generate_token_pair("user-123", vec!["user".to_string()], CustomClaims::default())
            .unwrap();

        assert_eq!(pair.token_type, "Bearer");
        assert_eq!(pair.expires_in, 3600);

        // Validate both tokens
        let access_claims = manager.validate_access_token(&pair.access_token).unwrap();
        let refresh_claims = manager.validate_refresh_token(&pair.refresh_token).unwrap();

        assert_eq!(access_claims.sub, "user-123");
        assert_eq!(refresh_claims.sub, "user-123");
    }

    #[test]
    fn test_extract_token() {
        assert_eq!(
            JwtManager::extract_token("Bearer abc123"),
            Some("abc123")
        );
        assert_eq!(JwtManager::extract_token("abc123"), None);
        assert_eq!(JwtManager::extract_token(""), None);
    }

    #[test]
    fn test_invalid_token() {
        let config = test_config();
        let manager = JwtManager::new(&config);

        let result = manager.validate_token("invalid.token.here");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_token_type() {
        let config = test_config();
        let manager = JwtManager::new(&config);

        // Generate refresh token but validate as access
        let refresh = manager
            .generate_refresh_token("user-123", vec![])
            .unwrap();
        let result = manager.validate_access_token(&refresh);
        assert!(result.is_err());

        // Generate access token but validate as refresh
        let access = manager
            .generate_access_token("user-123", vec![], CustomClaims::default())
            .unwrap();
        let result = manager.validate_refresh_token(&access);
        assert!(result.is_err());
    }
}
