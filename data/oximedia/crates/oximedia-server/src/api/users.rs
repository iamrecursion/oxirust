//! User management API endpoints.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    models::user::ApiKey,
    AppState,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// User profile response.
#[derive(Debug, Serialize)]
pub struct UserProfile {
    /// User ID
    pub id: String,
    /// Username
    pub username: String,
    /// Email
    pub email: String,
    /// User role
    pub role: String,
    /// Creation timestamp
    pub created_at: i64,
}

/// Update user request.
#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    /// New email (optional)
    pub email: Option<String>,
}

/// Change password request.
#[derive(Debug, Deserialize)]
pub struct ChangePasswordRequest {
    /// Current password
    pub current_password: String,
    /// New password
    pub new_password: String,
}

/// Create API key request.
#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    /// Key name
    pub name: String,
    /// Expiration in days (optional)
    pub expires_in_days: Option<i64>,
}

/// Create API key response.
#[derive(Debug, Serialize)]
pub struct CreateApiKeyResponse {
    /// Key ID
    pub key_id: String,
    /// API key (only returned once)
    pub api_key: String,
}

/// Gets the current user's profile.
pub async fn get_current_user(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    let row = crate::db::query(
        r"
        SELECT id, username, email, role, created_at
        FROM users
        WHERE id = ?
        ",
    )
    .bind(&auth_user.user_id)
    .fetch_one(state.db.pool())
    .await?;

    Ok(Json(UserProfile {
        id: row.get("id"),
        username: row.get("username"),
        email: row.get("email"),
        role: row.get("role"),
        created_at: row.get("created_at"),
    }))
}

/// Updates the current user's profile.
pub async fn update_current_user(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<UpdateUserRequest>,
) -> ServerResult<impl IntoResponse> {
    if let Some(email) = &req.email {
        // Check if email is already taken
        let exists = crate::db::query_scalar::<i64, _>(
            "SELECT COUNT(*) FROM users WHERE email = ? AND id != ?",
        )
        .bind(email)
        .bind(&auth_user.user_id)
        .fetch_one(state.db.pool())
        .await?;

        if exists > 0 {
            return Err(ServerError::Conflict("Email already in use".to_string()));
        }

        // Update email
        crate::db::query("UPDATE users SET email = ?, updated_at = ? WHERE id = ?")
            .bind(email)
            .bind(chrono::Utc::now().timestamp())
            .bind(&auth_user.user_id)
            .execute(state.db.pool())
            .await?;
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Changes the current user's password.
pub async fn change_password(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<ChangePasswordRequest>,
) -> ServerResult<impl IntoResponse> {
    // Validate new password
    if req.new_password.len() < 8 {
        return Err(ServerError::BadRequest(
            "Password must be at least 8 characters".to_string(),
        ));
    }

    // Get current password hash
    let password_hash: String =
        crate::db::query_scalar("SELECT password_hash FROM users WHERE id = ?")
            .bind(&auth_user.user_id)
            .fetch_one(state.db.pool())
            .await?;

    // Verify current password
    let valid = state
        .auth
        .verify_password(&req.current_password, &password_hash)?;
    if !valid {
        return Err(ServerError::Unauthorized(
            "Current password is incorrect".to_string(),
        ));
    }

    // Hash new password
    let new_hash = state.auth.hash_password(&req.new_password)?;

    // Update password
    crate::db::query("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?")
        .bind(&new_hash)
        .bind(chrono::Utc::now().timestamp())
        .bind(&auth_user.user_id)
        .execute(state.db.pool())
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Lists the current user's API keys.
pub async fn list_api_keys(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    let rows = crate::db::query(
        r"
        SELECT id, name, created_at, expires_at, last_used
        FROM api_keys
        WHERE user_id = ?
        ORDER BY created_at DESC
        ",
    )
    .bind(&auth_user.user_id)
    .fetch_all(state.db.pool())
    .await?;

    let keys: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            serde_json::json!({
                "id": row.get::<String>("id"),
                "name": row.get::<String>("name"),
                "created_at": row.get::<i64>("created_at"),
                "expires_at": row.get::<Option<i64>>("expires_at"),
                "last_used": row.get::<Option<i64>>("last_used"),
            })
        })
        .collect();

    Ok(Json(keys))
}

/// Creates a new API key.
pub async fn create_api_key(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<CreateApiKeyRequest>,
) -> ServerResult<impl IntoResponse> {
    // Generate API key
    let api_key = crate::auth::AuthManager::generate_api_key();
    let key_hash = crate::auth::AuthManager::hash_api_key(&api_key);

    // Calculate expiration
    let expires_at = req
        .expires_in_days
        .map(|days| chrono::Utc::now().timestamp() + days * 86400);

    // Create API key
    let key = ApiKey::new(auth_user.user_id, key_hash, req.name, expires_at);

    // Save to database
    crate::db::query(
        r"
        INSERT INTO api_keys (id, user_id, key_hash, name, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(&key.id)
    .bind(&key.user_id)
    .bind(&key.key_hash)
    .bind(&key.name)
    .bind(key.created_at)
    .bind(key.expires_at)
    .execute(state.db.pool())
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(CreateApiKeyResponse {
            key_id: key.id,
            api_key,
        }),
    ))
}

/// Revokes an API key.
pub async fn revoke_api_key(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(key_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    // Verify the key belongs to the user
    let count = crate::db::query_scalar::<i64, _>(
        "SELECT COUNT(*) FROM api_keys WHERE id = ? AND user_id = ?",
    )
    .bind(&key_id)
    .bind(&auth_user.user_id)
    .fetch_one(state.db.pool())
    .await?;

    if count == 0 {
        return Err(ServerError::NotFound("API key not found".to_string()));
    }

    // Delete the key
    crate::db::query("DELETE FROM api_keys WHERE id = ?")
        .bind(&key_id)
        .execute(state.db.pool())
        .await?;

    Ok(StatusCode::NO_CONTENT)
}
