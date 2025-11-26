//! JWT Authentication Middleware
//!
//! Validates JWT tokens and extracts user claims for protected routes

use crate::auth::{Claims, JwtManager};
use crate::error::ApiError;
use crate::state::AppState;
use axum::{
    body::Body,
    extract::State,
    http::{header, Request},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

/// Authenticated user information extracted from JWT
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub user_id: String,
    pub roles: Vec<String>,
    pub claims: Claims,
}

/// JWT authentication middleware
pub async fn jwt_auth(
    State(state): State<Arc<AppState>>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, ApiError> {
    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(header) => JwtManager::extract_token(header).ok_or(ApiError::Unauthorized)?,
        None => return Err(ApiError::Unauthorized),
    };

    // Create JWT manager and validate token
    let jwt_manager = JwtManager::new(&state.config.jwt);
    let claims = jwt_manager.validate_access_token(token)?;

    // Check if token is blacklisted in Redis
    if is_token_blacklisted(&state, &claims.jti).await? {
        return Err(ApiError::InvalidToken);
    }

    // Create authenticated user
    let user = AuthenticatedUser {
        user_id: claims.sub.clone(),
        roles: claims.roles.clone(),
        claims,
    };

    // Insert user into request extensions
    request.extensions_mut().insert(user);

    Ok(next.run(request).await)
}

/// Check if a token (by JTI) is blacklisted
async fn is_token_blacklisted(state: &AppState, jti: &str) -> Result<bool, ApiError> {
    let mut conn = state.redis_conn().await?;
    let key = format!("token:blacklist:{}", jti);
    let exists: bool = redis::cmd("EXISTS")
        .arg(&key)
        .query_async(&mut conn)
        .await?;
    Ok(exists)
}

/// Blacklist a token (on logout)
pub async fn blacklist_token(state: &AppState, jti: &str, ttl_secs: u64) -> Result<(), ApiError> {
    let mut conn = state.redis_conn().await?;
    let key = format!("token:blacklist:{}", jti);
    redis::cmd("SETEX")
        .arg(&key)
        .arg(ttl_secs)
        .arg("1")
        .query_async(&mut conn)
        .await?;
    Ok(())
}

/// Role-based access control helper
impl AuthenticatedUser {
    /// Check if user has a specific role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Check if user has any of the specified roles
    pub fn has_any_role(&self, roles: &[&str]) -> bool {
        roles.iter().any(|r| self.has_role(r))
    }

    /// Check if user has all of the specified roles
    pub fn has_all_roles(&self, roles: &[&str]) -> bool {
        roles.iter().all(|r| self.has_role(r))
    }

    /// Check if user is admin
    pub fn is_admin(&self) -> bool {
        self.has_role("admin")
    }
}

/// Macro for requiring specific roles
#[macro_export]
macro_rules! require_role {
    ($user:expr, $role:expr) => {
        if !$user.has_role($role) {
            return Err(crate::error::ApiError::Forbidden(format!(
                "Role '{}' required",
                $role
            )));
        }
    };
}

/// Macro for requiring any of specified roles
#[macro_export]
macro_rules! require_any_role {
    ($user:expr, $($role:expr),+) => {
        let roles = vec![$($role),+];
        if !$user.has_any_role(&roles) {
            return Err(crate::error::ApiError::Forbidden(format!(
                "One of roles {:?} required",
                roles
            )));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{CustomClaims, TokenType};

    fn create_test_user(roles: Vec<String>) -> AuthenticatedUser {
        AuthenticatedUser {
            user_id: "test-user".to_string(),
            roles: roles.clone(),
            claims: Claims {
                sub: "test-user".to_string(),
                exp: 9999999999,
                iat: 0,
                nbf: 0,
                iss: "test".to_string(),
                jti: "test-jti".to_string(),
                token_type: TokenType::Access,
                roles,
                custom: CustomClaims::default(),
            },
        }
    }

    #[test]
    fn test_has_role() {
        let user = create_test_user(vec!["user".to_string(), "admin".to_string()]);
        assert!(user.has_role("user"));
        assert!(user.has_role("admin"));
        assert!(!user.has_role("superadmin"));
    }

    #[test]
    fn test_has_any_role() {
        let user = create_test_user(vec!["user".to_string()]);
        assert!(user.has_any_role(&["user", "admin"]));
        assert!(!user.has_any_role(&["admin", "superadmin"]));
    }

    #[test]
    fn test_has_all_roles() {
        let user = create_test_user(vec!["user".to_string(), "admin".to_string()]);
        assert!(user.has_all_roles(&["user", "admin"]));
        assert!(!user.has_all_roles(&["user", "superadmin"]));
    }

    #[test]
    fn test_is_admin() {
        let admin_user = create_test_user(vec!["admin".to_string()]);
        let regular_user = create_test_user(vec!["user".to_string()]);

        assert!(admin_user.is_admin());
        assert!(!regular_user.is_admin());
    }
}
