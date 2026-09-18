//! Search API endpoints.

use crate::{auth::AuthUser, error::ServerResult, models::media::Media, AppState};
use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Search query parameters.
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    /// Search query string
    pub q: String,
    /// Maximum results
    #[serde(default = "default_limit")]
    pub limit: i64,
}

const fn default_limit() -> i64 {
    50
}

/// Search suggestion query.
#[derive(Debug, Deserialize)]
pub struct SuggestQuery {
    /// Query prefix
    pub q: String,
    /// Maximum suggestions
    #[serde(default = "default_suggest_limit")]
    pub limit: i64,
}

const fn default_suggest_limit() -> i64 {
    10
}

/// Suggestion result.
#[derive(Debug, Serialize)]
pub struct Suggestion {
    /// Suggestion text
    pub text: String,
    /// Media ID
    pub media_id: String,
}

/// Searches media files.
pub async fn search_media(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Query(query): Query<SearchQuery>,
) -> ServerResult<impl IntoResponse> {
    if query.q.is_empty() {
        return Ok(Json(Vec::<Media>::new()));
    }

    let media = state
        .library
        .search_media(&auth_user.user_id, &query.q, query.limit)
        .await?;

    Ok(Json(media))
}

/// Provides search suggestions.
pub async fn suggest(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Query(query): Query<SuggestQuery>,
) -> ServerResult<impl IntoResponse> {
    if query.q.is_empty() {
        return Ok(Json(Vec::<Suggestion>::new()));
    }

    let media = state
        .library
        .search_media(&auth_user.user_id, &query.q, query.limit)
        .await?;

    let suggestions: Vec<Suggestion> = media
        .into_iter()
        .map(|m| Suggestion {
            text: m.original_filename,
            media_id: m.id,
        })
        .collect();

    Ok(Json(suggestions))
}
