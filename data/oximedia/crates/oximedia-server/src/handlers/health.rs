//! Health check handlers.

use crate::{error::ServerResult, AppState};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;
use std::sync::Arc;

/// Health check endpoint.
pub async fn health_check() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "version": env!("CARGO_PKG_VERSION"),
        })),
    )
}

/// Readiness check endpoint (checks database connectivity).
pub async fn readiness_check(
    State(state): State<Arc<AppState>>,
) -> ServerResult<impl IntoResponse> {
    // Check database health
    let db_healthy = state.db.health_check().await.unwrap_or(false);

    if db_healthy {
        Ok((
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "database": "ok",
            })),
        ))
    } else {
        Ok((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "database": "unavailable",
            })),
        ))
    }
}
