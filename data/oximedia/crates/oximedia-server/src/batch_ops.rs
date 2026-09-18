//! Batch operations on media items.
//!
//! Provides endpoints that accept multiple media IDs in a single request and
//! perform the operation on each item, collecting per-item successes and
//! failures into a `BatchResult` rather than aborting the entire request on
//! the first error.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    AppState,
};
use axum::{extract::State, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

// ── Domain types ──────────────────────────────────────────────────────────────

/// A single per-item error within a batch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    /// The media ID that failed.
    pub id: String,
    /// Human-readable error description.
    pub error: String,
}

/// Aggregated result of a batch operation.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchResult {
    /// Media IDs that were processed successfully.
    pub succeeded: Vec<String>,
    /// Items that failed, with per-item error details.
    pub failed: Vec<BatchError>,
}

impl BatchResult {
    /// Creates an empty `BatchResult`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            succeeded: Vec::new(),
            failed: Vec::new(),
        }
    }

    /// Records a success.
    pub fn add_success(&mut self, id: impl Into<String>) {
        self.succeeded.push(id.into());
    }

    /// Records a failure.
    pub fn add_failure(&mut self, id: impl Into<String>, error: impl Into<String>) {
        self.failed.push(BatchError {
            id: id.into(),
            error: error.into(),
        });
    }

    /// Returns `true` when every item succeeded.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failed.is_empty()
    }
}

impl Default for BatchResult {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request types ─────────────────────────────────────────────────────────────

/// Request body for `POST /api/v1/media/batch/delete`.
#[derive(Debug, Deserialize)]
pub struct BatchDeleteRequest {
    /// Media IDs to delete (at most 100 per call).
    pub media_ids: Vec<String>,
}

/// Request body for `POST /api/v1/media/batch/transcode`.
#[derive(Debug, Deserialize)]
pub struct BatchTranscodeRequest {
    /// Media IDs to transcode (at most 100 per call).
    pub media_ids: Vec<String>,
    /// Transcode preset identifier (e.g. "1080p", "720p", "audio-opus").
    pub preset: String,
}

/// A single metadata update: a map of key → value pairs.
#[derive(Debug, Deserialize)]
pub struct MetadataUpdate {
    /// Media ID.
    pub media_id: String,
    /// Key-value metadata fields to set (existing keys are overwritten).
    pub fields: std::collections::HashMap<String, String>,
}

/// Request body for `POST /api/v1/media/batch/update-metadata`.
#[derive(Debug, Deserialize)]
pub struct BatchUpdateMetadataRequest {
    /// Per-item metadata update instructions.
    pub updates: Vec<MetadataUpdate>,
}

// ── Limits ────────────────────────────────────────────────────────────────────

/// Maximum number of items accepted in a single batch request.
const BATCH_LIMIT: usize = 100;

// ── Preset resolution ─────────────────────────────────────────────────────────

/// Parameters derived from a named transcode preset.
struct PresetParams {
    output_format: String,
    codec_video: Option<String>,
    codec_audio: Option<String>,
    width: Option<i32>,
    height: Option<i32>,
    bitrate: Option<i32>,
}

/// Resolves a preset name into concrete encoding parameters.
///
/// Returns `None` for unknown presets.
fn resolve_preset(name: &str) -> Option<PresetParams> {
    match name {
        "2160p" | "4k" => Some(PresetParams {
            output_format: "webm".to_string(),
            codec_video: Some("av1".to_string()),
            codec_audio: Some("opus".to_string()),
            width: Some(3840),
            height: Some(2160),
            bitrate: Some(20_000_000),
        }),
        "1080p" => Some(PresetParams {
            output_format: "webm".to_string(),
            codec_video: Some("av1".to_string()),
            codec_audio: Some("opus".to_string()),
            width: Some(1920),
            height: Some(1080),
            bitrate: Some(8_000_000),
        }),
        "720p" => Some(PresetParams {
            output_format: "webm".to_string(),
            codec_video: Some("vp9".to_string()),
            codec_audio: Some("opus".to_string()),
            width: Some(1280),
            height: Some(720),
            bitrate: Some(4_000_000),
        }),
        "480p" => Some(PresetParams {
            output_format: "webm".to_string(),
            codec_video: Some("vp9".to_string()),
            codec_audio: Some("opus".to_string()),
            width: Some(854),
            height: Some(480),
            bitrate: Some(1_500_000),
        }),
        "audio-opus" => Some(PresetParams {
            output_format: "ogg".to_string(),
            codec_video: None,
            codec_audio: Some("opus".to_string()),
            width: None,
            height: None,
            bitrate: Some(192_000),
        }),
        "audio-flac" => Some(PresetParams {
            output_format: "flac".to_string(),
            codec_video: None,
            codec_audio: Some("flac".to_string()),
            width: None,
            height: None,
            bitrate: None,
        }),
        _ => None,
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `POST /api/v1/media/batch/delete` — delete multiple media items.
///
/// Each item is deleted independently; failures do not abort the batch.
pub async fn batch_delete(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(body): Json<BatchDeleteRequest>,
) -> ServerResult<impl IntoResponse> {
    if body.media_ids.is_empty() {
        return Err(ServerError::BadRequest(
            "media_ids must not be empty".to_string(),
        ));
    }
    if body.media_ids.len() > BATCH_LIMIT {
        return Err(ServerError::BadRequest(format!(
            "Batch size exceeds maximum of {BATCH_LIMIT}"
        )));
    }

    let mut result = BatchResult::new();

    for media_id in &body.media_ids {
        match delete_single_media(&state, &auth_user, media_id).await {
            Ok(()) => result.add_success(media_id),
            Err(e) => result.add_failure(media_id, e.to_string()),
        }
    }

    Ok(Json(result))
}

/// `POST /api/v1/media/batch/transcode` — submit transcode jobs for multiple items.
pub async fn batch_transcode(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(body): Json<BatchTranscodeRequest>,
) -> ServerResult<impl IntoResponse> {
    if body.media_ids.is_empty() {
        return Err(ServerError::BadRequest(
            "media_ids must not be empty".to_string(),
        ));
    }
    if body.media_ids.len() > BATCH_LIMIT {
        return Err(ServerError::BadRequest(format!(
            "Batch size exceeds maximum of {BATCH_LIMIT}"
        )));
    }

    let preset = resolve_preset(&body.preset).ok_or_else(|| {
        ServerError::BadRequest(format!(
            "Unknown preset '{}'. Valid presets: 2160p, 1080p, 720p, 480p, audio-opus, audio-flac",
            body.preset
        ))
    })?;

    let mut result = BatchResult::new();
    let now = chrono::Utc::now().timestamp();

    for media_id in &body.media_ids {
        match submit_transcode_job(&state, &auth_user, media_id, &preset, now).await {
            Ok(job_id) => {
                result.succeeded.push(job_id);
            }
            Err(e) => result.add_failure(media_id, e.to_string()),
        }
    }

    Ok(Json(result))
}

/// `POST /api/v1/media/batch/update-metadata` — update metadata on multiple items.
pub async fn batch_update_metadata(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(body): Json<BatchUpdateMetadataRequest>,
) -> ServerResult<impl IntoResponse> {
    if body.updates.is_empty() {
        return Err(ServerError::BadRequest(
            "updates must not be empty".to_string(),
        ));
    }
    if body.updates.len() > BATCH_LIMIT {
        return Err(ServerError::BadRequest(format!(
            "Batch size exceeds maximum of {BATCH_LIMIT}"
        )));
    }

    let mut result = BatchResult::new();

    for update in &body.updates {
        match apply_metadata_update(&state, &auth_user, update).await {
            Ok(()) => result.add_success(&update.media_id),
            Err(e) => result.add_failure(&update.media_id, e.to_string()),
        }
    }

    Ok(Json(result))
}

// ── Per-item helpers ──────────────────────────────────────────────────────────

/// Deletes a single media item after ownership/permission checks.
async fn delete_single_media(
    state: &Arc<AppState>,
    auth_user: &AuthUser,
    media_id: &str,
) -> ServerResult<()> {
    // Fetch the media row to get the file path and check ownership.
    let row = crate::db::query("SELECT user_id, filename FROM media WHERE id = ?")
        .bind(media_id)
        .fetch_optional(state.db.pool())
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("Media '{}' not found", media_id)))?;

    let owner_id: String = row.get("user_id");
    if owner_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(format!(
            "Not authorized to delete media '{}'",
            media_id
        )));
    }

    // Delete from the database (cascades to metadata, collection_items, jobs).
    crate::db::query("DELETE FROM media WHERE id = ?")
        .bind(media_id)
        .execute(state.db.pool())
        .await?;

    // Best-effort removal of the stored file.
    let filename: String = row.get("filename");
    let path = state.config.media_dir.join(&filename);
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!("batch_delete: could not remove file {:?}: {}", path, e);
        }
    }

    Ok(())
}

/// Inserts a transcode job for one media item, returning the new job ID.
async fn submit_transcode_job(
    state: &Arc<AppState>,
    auth_user: &AuthUser,
    media_id: &str,
    preset: &PresetParams,
    now: i64,
) -> ServerResult<String> {
    // Verify the media item exists and is accessible.
    let row = crate::db::query("SELECT user_id, status FROM media WHERE id = ?")
        .bind(media_id)
        .fetch_optional(state.db.pool())
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("Media '{}' not found", media_id)))?;

    let owner_id: String = row.get("user_id");
    if owner_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(format!(
            "Not authorized to transcode media '{}'",
            media_id
        )));
    }

    let status: String = row.get("status");
    if status == "processing" {
        return Err(ServerError::Conflict(format!(
            "Media '{}' is already being processed",
            media_id
        )));
    }

    let job_id = Uuid::new_v4().to_string();

    crate::db::query(
        r"
        INSERT INTO transcode_jobs
            (id, user_id, media_id, status, progress,
             output_format, output_codec_video, output_codec_audio,
             output_width, output_height, output_bitrate, created_at)
        VALUES (?, ?, ?, 'queued', 0.0, ?, ?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(&job_id)
    .bind(&auth_user.user_id)
    .bind(media_id)
    .bind(&preset.output_format)
    .bind(&preset.codec_video)
    .bind(&preset.codec_audio)
    .bind(preset.width)
    .bind(preset.height)
    .bind(preset.bitrate)
    .bind(now)
    .execute(state.db.pool())
    .await?;

    // Add the new job to the in-process queue.
    {
        let mut queue = state.job_queue.write().await;
        queue.enqueue(job_id.clone());
    }

    Ok(job_id)
}

/// Upserts key-value metadata fields for a single media item.
async fn apply_metadata_update(
    state: &Arc<AppState>,
    auth_user: &AuthUser,
    update: &MetadataUpdate,
) -> ServerResult<()> {
    // Ownership check.
    let row = crate::db::query("SELECT user_id FROM media WHERE id = ?")
        .bind(&update.media_id)
        .fetch_optional(state.db.pool())
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("Media '{}' not found", update.media_id)))?;

    let owner_id: String = row.get("user_id");
    if owner_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(format!(
            "Not authorized to update metadata for media '{}'",
            update.media_id
        )));
    }

    // Upsert each key-value pair.
    for (key, value) in &update.fields {
        crate::db::query(
            r"
            INSERT INTO media_metadata (media_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT (media_id, key) DO UPDATE SET value = excluded.value
            ",
        )
        .bind(&update.media_id)
        .bind(key)
        .bind(value)
        .execute(state.db.pool())
        .await?;
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BatchResult unit tests ────────────────────────────────────────────────

    #[test]
    fn test_batch_result_starts_empty() {
        let r = BatchResult::new();
        assert!(r.succeeded.is_empty());
        assert!(r.failed.is_empty());
        assert!(r.all_succeeded());
    }

    #[test]
    fn test_batch_result_add_success() {
        let mut r = BatchResult::new();
        r.add_success("m1");
        assert_eq!(r.succeeded, vec!["m1"]);
        assert!(r.all_succeeded());
    }

    #[test]
    fn test_batch_result_add_failure() {
        let mut r = BatchResult::new();
        r.add_failure("m2", "not found");
        assert!(!r.all_succeeded());
        assert_eq!(r.failed[0].id, "m2");
        assert_eq!(r.failed[0].error, "not found");
    }

    #[test]
    fn test_batch_result_mixed() {
        let mut r = BatchResult::new();
        r.add_success("m1");
        r.add_failure("m2", "denied");
        assert!(!r.all_succeeded());
        assert_eq!(r.succeeded.len(), 1);
        assert_eq!(r.failed.len(), 1);
    }

    #[test]
    fn test_batch_result_serializes() {
        let mut r = BatchResult::new();
        r.add_success("a");
        r.add_failure("b", "oops");
        let j = serde_json::to_value(&r).expect("serialize");
        assert_eq!(j["succeeded"][0], "a");
        assert_eq!(j["failed"][0]["id"], "b");
        assert_eq!(j["failed"][0]["error"], "oops");
    }

    // ── Preset resolution tests ───────────────────────────────────────────────

    #[test]
    fn test_resolve_preset_1080p() {
        let p = resolve_preset("1080p").expect("preset");
        assert_eq!(p.width, Some(1920));
        assert_eq!(p.height, Some(1080));
        assert_eq!(p.output_format, "webm");
    }

    #[test]
    fn test_resolve_preset_720p() {
        let p = resolve_preset("720p").expect("preset");
        assert_eq!(p.width, Some(1280));
        assert_eq!(p.height, Some(720));
    }

    #[test]
    fn test_resolve_preset_audio_opus() {
        let p = resolve_preset("audio-opus").expect("preset");
        assert_eq!(p.codec_audio.as_deref(), Some("opus"));
        assert!(p.codec_video.is_none());
        assert_eq!(p.output_format, "ogg");
    }

    #[test]
    fn test_resolve_preset_audio_flac() {
        let p = resolve_preset("audio-flac").expect("preset");
        assert_eq!(p.codec_audio.as_deref(), Some("flac"));
        assert!(p.bitrate.is_none());
    }

    #[test]
    fn test_resolve_preset_4k_alias() {
        let p = resolve_preset("4k").expect("4k alias");
        assert_eq!(p.height, Some(2160));
    }

    #[test]
    fn test_resolve_preset_unknown_returns_none() {
        assert!(resolve_preset("320x240-xvid").is_none());
    }

    // ── Batch limit validation (logic-level, no HTTP stack) ───────────────────

    #[test]
    fn test_batch_error_serializes() {
        let e = BatchError {
            id: "m1".to_string(),
            error: "forbidden".to_string(),
        };
        let j = serde_json::to_value(&e).expect("serialize");
        assert_eq!(j["id"], "m1");
        assert_eq!(j["error"], "forbidden");
    }

    #[test]
    fn test_batch_limit_constant() {
        assert_eq!(BATCH_LIMIT, 100);
    }

    // ── Partial-completion / multi-status semantics ───────────────────────────

    /// Drives a mixed batch through the same accumulate-per-item control flow
    /// that the route handlers use (`Ok => add_success`, `Err => add_failure`),
    /// without the DB/AppState layer. The fake per-item op fails for a fixed set
    /// of IDs so we can assert exact partial state.
    fn run_mixed_batch(ids: &[String], fails: &[&str]) -> BatchResult {
        let mut result = BatchResult::new();
        for id in ids {
            // Simulated per-item operation: error for IDs in `fails`.
            let outcome: Result<(), String> = if fails.contains(&id.as_str()) {
                Err(format!("Media '{id}' not found"))
            } else {
                Ok(())
            };
            match outcome {
                Ok(()) => result.add_success(id),
                Err(e) => result.add_failure(id, e),
            }
        }
        result
    }

    #[test]
    fn test_batch_partial_completion_ten_items_three_fail() {
        let ids: Vec<String> = (0..10).map(|i| format!("m{i}")).collect();
        // Fail items 2, 5, 8 — a non-contiguous mix, including last region.
        let fails = ["m2", "m5", "m8"];

        let result = run_mixed_batch(&ids, &fails);

        // Counts are exactly right.
        assert_eq!(result.succeeded.len(), 7);
        assert_eq!(result.failed.len(), 3);
        assert!(!result.all_succeeded());

        // The batch is NOT aborted on the first failure: every input item is
        // accounted for in exactly one bucket (multi-status coherence).
        assert_eq!(result.succeeded.len() + result.failed.len(), ids.len());

        // Failed bucket holds precisely the intended IDs (order preserved).
        let failed_ids: Vec<&str> = result.failed.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(failed_ids, vec!["m2", "m5", "m8"]);

        // Succeeded bucket holds the complement, in original order.
        assert_eq!(
            result.succeeded,
            vec!["m0", "m1", "m3", "m4", "m6", "m7", "m9"]
        );

        // No ID appears in both buckets.
        for f in &result.failed {
            assert!(
                !result.succeeded.contains(&f.id),
                "id {} must not be in both buckets",
                f.id
            );
        }

        // Each failure carries a descriptive, item-specific message.
        assert!(result.failed.iter().all(|e| e.error.contains(&e.id)));
    }

    #[test]
    fn test_batch_all_fail_yields_empty_success_set() {
        let ids: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let result = run_mixed_batch(&ids, &["a", "b", "c"]);
        assert!(result.succeeded.is_empty());
        assert_eq!(result.failed.len(), 3);
        assert!(!result.all_succeeded());
    }

    #[test]
    fn test_batch_partial_serializes_to_multi_status_shape() {
        let ids: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
        let result = run_mixed_batch(&ids, &["y"]);
        let j = serde_json::to_value(&result).expect("serialize");

        // Multi-status JSON: a succeeded array and a failed array of {id,error}.
        assert_eq!(
            j["succeeded"],
            serde_json::json!(["x", "z"]),
            "succeeded list mismatch"
        );
        assert_eq!(j["failed"][0]["id"], "y");
        assert!(j["failed"][0]["error"].as_str().is_some());
        assert_eq!(j["failed"].as_array().expect("failed array").len(), 1);
    }
}
