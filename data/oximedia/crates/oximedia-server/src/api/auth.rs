//! Authentication API endpoints.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    models::user::{User, UserRole},
    AppState,
};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Register request.
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    /// Username
    pub username: String,
    /// Email
    pub email: String,
    /// Password
    pub password: String,
}

/// Register response.
#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    /// User ID
    pub user_id: String,
    /// JWT token
    pub token: String,
}

/// Login request.
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    /// Username or email
    pub username: String,
    /// Password
    pub password: String,
}

/// Login response.
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    /// User ID
    pub user_id: String,
    /// Username
    pub username: String,
    /// JWT token
    pub token: String,
}

/// Refresh token request.
#[derive(Debug, Deserialize)]
pub struct RefreshTokenRequest {
    /// Current token
    pub token: String,
}

/// Refresh token response.
#[derive(Debug, Serialize)]
pub struct RefreshTokenResponse {
    /// New JWT token
    pub token: String,
}

/// Registers a new user.
pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> ServerResult<impl IntoResponse> {
    // Validate input
    if req.username.is_empty() || req.email.is_empty() || req.password.is_empty() {
        return Err(ServerError::BadRequest(
            "Username, email, and password are required".to_string(),
        ));
    }

    if req.password.len() < 8 {
        return Err(ServerError::BadRequest(
            "Password must be at least 8 characters".to_string(),
        ));
    }

    // Check if username or email already exists
    let exists = crate::db::query_scalar::<i64, _>(
        "SELECT COUNT(*) FROM users WHERE username = ? OR email = ?",
    )
    .bind(&req.username)
    .bind(&req.email)
    .fetch_one(state.db.pool())
    .await?;

    if exists > 0 {
        return Err(ServerError::Conflict(
            "Username or email already exists".to_string(),
        ));
    }

    // Hash password
    let password_hash = state.auth.hash_password(&req.password)?;

    // Create user
    let user = User::new(req.username, req.email, password_hash);

    // Save to database
    crate::db::query(
        r"
        INSERT INTO users (id, username, email, password_hash, role, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(&user.id)
    .bind(&user.username)
    .bind(&user.email)
    .bind(&user.password_hash)
    .bind(user.role.to_string())
    .bind(user.created_at)
    .bind(user.updated_at)
    .execute(state.db.pool())
    .await?;

    // Generate token
    let token = state.auth.generate_token(
        user.id.clone(),
        user.username,
        user.role,
        state.config.jwt_expiration as i64,
    )?;

    Ok((
        StatusCode::CREATED,
        Json(RegisterResponse {
            user_id: user.id,
            token,
        }),
    ))
}

/// Logs in a user.
pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoginRequest>,
) -> ServerResult<impl IntoResponse> {
    // Find user by username or email
    let row = crate::db::query(
        r"
        SELECT id, username, email, password_hash, role
        FROM users
        WHERE username = ? OR email = ?
        ",
    )
    .bind(&req.username)
    .bind(&req.username)
    .fetch_optional(state.db.pool())
    .await?
    .ok_or_else(|| ServerError::Unauthorized("Invalid credentials".to_string()))?;

    let user_id: String = row.get("id");
    let username: String = row.get("username");
    let password_hash: String = row.get("password_hash");
    let role_str: String = row.get("role");
    let role = role_str
        .parse::<UserRole>()
        .map_err(|e| ServerError::Internal(e))?;

    // Verify password
    let valid = state.auth.verify_password(&req.password, &password_hash)?;
    if !valid {
        return Err(ServerError::Unauthorized("Invalid credentials".to_string()));
    }

    // Update last login
    crate::db::query("UPDATE users SET last_login = ? WHERE id = ?")
        .bind(chrono::Utc::now().timestamp())
        .bind(&user_id)
        .execute(state.db.pool())
        .await?;

    // Generate token
    let token = state.auth.generate_token(
        user_id.clone(),
        username.clone(),
        role,
        state.config.jwt_expiration as i64,
    )?;

    Ok(Json(LoginResponse {
        user_id,
        username,
        token,
    }))
}

/// Refreshes an authentication token.
pub async fn refresh_token(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RefreshTokenRequest>,
) -> ServerResult<impl IntoResponse> {
    // Validate current token
    let claims = state.auth.validate_token(&req.token)?;

    // Generate new token
    let token = state.auth.generate_token(
        claims.sub,
        claims.username,
        claims.role,
        state.config.jwt_expiration as i64,
    )?;

    Ok(Json(RefreshTokenResponse { token }))
}

/// Logs out a user.
pub async fn logout(_auth_user: AuthUser) -> ServerResult<impl IntoResponse> {
    // In a stateless JWT system, logout is handled client-side by discarding the token
    // For enhanced security, you could maintain a token blacklist here
    Ok(StatusCode::NO_CONTENT)
}
