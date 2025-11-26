//! Authentication Handlers
//!
//! Handles user authentication, registration, and token management

use crate::auth::{CustomClaims, JwtManager, TokenPair};
use crate::error::{ApiError, ApiResult};
use crate::middleware::auth::{blacklist_token, AuthenticatedUser};
use crate::state::AppState;
use axum::{extract::Extension, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

/// Login request
#[derive(Debug, Deserialize, Validate)]
pub struct LoginRequest {
    #[validate(length(min = 1, max = 255))]
    pub username: String,

    #[validate(length(min = 8, max = 128))]
    pub password: String,
}

/// Login response
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub user: UserInfo,
    pub tokens: TokenPair,
}

/// User information
#[derive(Debug, Serialize)]
pub struct UserInfo {
    pub id: String,
    pub username: String,
    pub roles: Vec<String>,
    pub default_privacy_level: u8,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Registration request
#[derive(Debug, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(length(min = 3, max = 50))]
    pub username: String,

    #[validate(email)]
    pub email: String,

    #[validate(length(min = 8, max = 128))]
    pub password: String,

    /// Optional public key for cryptographic operations
    pub public_key: Option<String>,
}

/// Registration response
#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub user: UserInfo,
    pub message: String,
}

/// Refresh token request
#[derive(Debug, Deserialize)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

/// Current user response
#[derive(Debug, Serialize)]
pub struct CurrentUserResponse {
    pub user: UserInfo,
}

/// User login handler
pub async fn login(
    Extension(state): Extension<Arc<AppState>>,
    Json(payload): Json<LoginRequest>,
) -> ApiResult<Json<LoginResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    // Query user from database
    let user = sqlx::query_as!(
        UserRecord,
        r#"
        SELECT id, username, password_hash, roles, default_privacy_level, created_at
        FROM users
        WHERE username = $1
        "#,
        payload.username
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::InvalidCredentials)?;

    // Verify password (using bcrypt or argon2)
    // In production, use proper password hashing
    let password_valid = verify_password(&payload.password, &user.password_hash)?;
    if !password_valid {
        return Err(ApiError::InvalidCredentials);
    }

    // Generate JWT tokens
    let jwt_manager = JwtManager::new(&state.config.jwt);
    let custom_claims = CustomClaims {
        pubkey_hash: None,
        default_privacy_level: Some(user.default_privacy_level as u8),
        session_id: Some(Uuid::new_v4().to_string()),
    };

    let tokens = jwt_manager
        .generate_token_pair(&user.id.to_string(), user.roles.clone(), custom_claims)
        .map_err(|e| ApiError::InternalError)?;

    // Update last login timestamp
    sqlx::query!(
        "UPDATE users SET last_login = NOW() WHERE id = $1",
        user.id
    )
    .execute(&state.db)
    .await?;

    tracing::info!(user_id = %user.id, "User logged in successfully");

    Ok(Json(LoginResponse {
        user: UserInfo {
            id: user.id.to_string(),
            username: user.username,
            roles: user.roles,
            default_privacy_level: user.default_privacy_level as u8,
            created_at: user.created_at,
        },
        tokens,
    }))
}

/// User registration handler
pub async fn register(
    Extension(state): Extension<Arc<AppState>>,
    Json(payload): Json<RegisterRequest>,
) -> ApiResult<Json<RegisterResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    // Check if username already exists
    let exists = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM users WHERE username = $1)",
        payload.username
    )
    .fetch_one(&state.db)
    .await?
    .unwrap_or(false);

    if exists {
        return Err(ApiError::BadRequest("Username already taken".to_string()));
    }

    // Check if email already exists
    let email_exists = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)",
        payload.email
    )
    .fetch_one(&state.db)
    .await?
    .unwrap_or(false);

    if email_exists {
        return Err(ApiError::BadRequest("Email already registered".to_string()));
    }

    // Hash password
    let password_hash = hash_password(&payload.password)?;

    // Create user
    let user_id = Uuid::new_v4();
    let default_roles = vec!["user".to_string()];

    sqlx::query!(
        r#"
        INSERT INTO users (id, username, email, password_hash, roles, default_privacy_level, public_key, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
        user_id,
        payload.username,
        payload.email,
        password_hash,
        &default_roles,
        3i32, // Default privacy level: Private
        payload.public_key
    )
    .execute(&state.db)
    .await?;

    tracing::info!(user_id = %user_id, username = %payload.username, "New user registered");

    Ok(Json(RegisterResponse {
        user: UserInfo {
            id: user_id.to_string(),
            username: payload.username,
            roles: default_roles,
            default_privacy_level: 3,
            created_at: chrono::Utc::now(),
        },
        message: "Registration successful".to_string(),
    }))
}

/// Refresh token handler
pub async fn refresh_token(
    Extension(state): Extension<Arc<AppState>>,
    Json(payload): Json<RefreshTokenRequest>,
) -> ApiResult<Json<TokenPair>> {
    let jwt_manager = JwtManager::new(&state.config.jwt);

    // Validate refresh token
    let claims = jwt_manager.validate_refresh_token(&payload.refresh_token)?;

    // Get user from database to ensure they still exist and are active
    let user = sqlx::query_as!(
        UserRecord,
        r#"
        SELECT id, username, password_hash, roles, default_privacy_level, created_at
        FROM users
        WHERE id = $1
        "#,
        Uuid::parse_str(&claims.sub).map_err(|_| ApiError::InvalidToken)?
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::InvalidToken)?;

    // Generate new token pair
    let custom_claims = CustomClaims {
        pubkey_hash: claims.custom.pubkey_hash,
        default_privacy_level: Some(user.default_privacy_level as u8),
        session_id: Some(Uuid::new_v4().to_string()),
    };

    let tokens = jwt_manager
        .generate_token_pair(&user.id.to_string(), user.roles, custom_claims)
        .map_err(|_| ApiError::InternalError)?;

    // Blacklist the old refresh token
    let ttl = state.config.jwt.refresh_token_expiry_secs;
    blacklist_token(&state, &claims.jti, ttl).await?;

    tracing::debug!(user_id = %user.id, "Token refreshed");

    Ok(Json(tokens))
}

/// Logout handler
pub async fn logout(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
) -> ApiResult<Json<serde_json::Value>> {
    // Blacklist the current access token
    let ttl = state.config.jwt.access_token_expiry_secs;
    blacklist_token(&state, &user.claims.jti, ttl).await?;

    tracing::info!(user_id = %user.user_id, "User logged out");

    Ok(Json(serde_json::json!({
        "message": "Logged out successfully"
    })))
}

/// Get current user handler
pub async fn get_current_user(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
) -> ApiResult<Json<CurrentUserResponse>> {
    let user_record = sqlx::query_as!(
        UserRecord,
        r#"
        SELECT id, username, password_hash, roles, default_privacy_level, created_at
        FROM users
        WHERE id = $1
        "#,
        Uuid::parse_str(&user.user_id).map_err(|_| ApiError::InvalidToken)?
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("User not found".to_string()))?;

    Ok(Json(CurrentUserResponse {
        user: UserInfo {
            id: user_record.id.to_string(),
            username: user_record.username,
            roles: user_record.roles,
            default_privacy_level: user_record.default_privacy_level as u8,
            created_at: user_record.created_at,
        },
    }))
}

// Helper types and functions

#[derive(Debug)]
struct UserRecord {
    id: Uuid,
    username: String,
    password_hash: String,
    roles: Vec<String>,
    default_privacy_level: i32,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Hash a password using bcrypt
fn hash_password(password: &str) -> Result<String, ApiError> {
    // In production, use bcrypt or argon2
    // For now, using a placeholder
    Ok(format!("hashed:{}", password))
}

/// Verify a password against a hash
fn verify_password(password: &str, hash: &str) -> Result<bool, ApiError> {
    // In production, use bcrypt or argon2
    // For now, using a placeholder
    Ok(hash == format!("hashed:{}", password))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_login_request_validation() {
        let valid = LoginRequest {
            username: "testuser".to_string(),
            password: "password123".to_string(),
        };
        assert!(valid.validate().is_ok());

        let invalid = LoginRequest {
            username: "".to_string(),
            password: "short".to_string(),
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_register_request_validation() {
        let valid = RegisterRequest {
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
            public_key: None,
        };
        assert!(valid.validate().is_ok());

        let invalid_email = RegisterRequest {
            username: "testuser".to_string(),
            email: "not-an-email".to_string(),
            password: "password123".to_string(),
            public_key: None,
        };
        assert!(invalid_email.validate().is_err());
    }
}
