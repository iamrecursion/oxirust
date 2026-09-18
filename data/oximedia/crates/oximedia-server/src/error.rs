//! Error types for the media server.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Result type for server operations.
pub type ServerResult<T> = Result<T, ServerError>;

/// Errors that can occur in the media server.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] crate::db::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// OxiMedia core error
    #[error("Media processing error: {0}")]
    Media(#[from] oximedia_core::error::OxiError),

    /// Authentication error
    #[error("Authentication failed: {0}")]
    Unauthorized(String),

    /// Forbidden resource
    #[error("Forbidden: {0}")]
    Forbidden(String),

    /// Resource not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid request
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// Conflict (e.g., duplicate resource)
    #[error("Conflict: {0}")]
    Conflict(String),

    /// JWT error
    #[error("JWT error: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),

    /// Password hashing error
    #[error("Password hashing error: {0}")]
    PasswordHash(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Insufficient storage
    #[error("Insufficient storage: {0}")]
    InsufficientStorage(String),

    /// Unsupported media type
    #[error("Unsupported media type: {0}")]
    UnsupportedMediaType(String),

    /// Transcoding error
    #[error("Transcoding failed: {0}")]
    TranscodingFailed(String),

    /// Upload error
    #[error("Upload failed: {0}")]
    UploadFailed(String),

    /// Internal server error
    #[error("Internal server error: {0}")]
    Internal(String),
}

impl ServerError {
    /// Returns the HTTP status code for this error.
    #[must_use]
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Forbidden(_) => StatusCode::FORBIDDEN,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::RateLimitExceeded => StatusCode::TOO_MANY_REQUESTS,
            Self::InsufficientStorage(_) => StatusCode::INSUFFICIENT_STORAGE,
            Self::UnsupportedMediaType(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Database(_)
            | Self::Io(_)
            | Self::Media(_)
            | Self::Jwt(_)
            | Self::PasswordHash(_)
            | Self::TranscodingFailed(_)
            | Self::UploadFailed(_)
            | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Returns a user-friendly error message.
    #[must_use]
    pub fn user_message(&self) -> String {
        match self {
            Self::Unauthorized(msg) => format!("Unauthorized: {msg}"),
            Self::Forbidden(msg) => format!("Forbidden: {msg}"),
            Self::NotFound(msg) => format!("Not found: {msg}"),
            Self::BadRequest(msg) => format!("Bad request: {msg}"),
            Self::Conflict(msg) => format!("Conflict: {msg}"),
            Self::RateLimitExceeded => "Rate limit exceeded. Please try again later.".to_string(),
            Self::InsufficientStorage(msg) => format!("Insufficient storage: {msg}"),
            Self::UnsupportedMediaType(msg) => format!("Unsupported media type: {msg}"),
            Self::TranscodingFailed(msg) => format!("Transcoding failed: {msg}"),
            Self::UploadFailed(msg) => format!("Upload failed: {msg}"),
            Self::Database(_)
            | Self::Io(_)
            | Self::Media(_)
            | Self::Jwt(_)
            | Self::PasswordHash(_)
            | Self::Internal(_) => "Internal server error. Please contact support.".to_string(),
        }
    }
}

impl From<argon2::password_hash::Error> for ServerError {
    fn from(err: argon2::password_hash::Error) -> Self {
        Self::PasswordHash(err.to_string())
    }
}

impl From<oximedia_net::error::NetError> for ServerError {
    fn from(err: oximedia_net::error::NetError) -> Self {
        Self::Internal(format!("Network/streaming error: {err}"))
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.user_message();

        // Log internal errors
        if matches!(self, Self::Database(_) | Self::Io(_) | Self::Internal(_)) {
            tracing::error!("Internal error: {}", self);
        }

        let body = Json(json!({
            "error": {
                "code": status.as_u16(),
                "message": message,
            }
        }));

        (status, body).into_response()
    }
}
