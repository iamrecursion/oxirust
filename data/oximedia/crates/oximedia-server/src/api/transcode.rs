//! Transcoding API endpoints.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    models::transcode::{TranscodeJob, TranscodeStatus},
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

/// Submit transcoding job request.
#[derive(Debug, Deserialize)]
pub struct SubmitJobRequest {
    /// Source media ID
    pub media_id: String,
    /// Output format (e.g., "webm", "mp4")
    pub output_format: String,
    /// Output video codec (e.g., "av1", "vp9")
    pub output_codec_video: Option<String>,
    /// Output audio codec (e.g., "opus", "vorbis")
    pub output_codec_audio: Option<String>,
    /// Output width
    pub output_width: Option<i32>,
    /// Output height
    pub output_height: Option<i32>,
    /// Output bitrate
    pub output_bitrate: Option<i64>,
}

/// Submit job response.
#[derive(Debug, Serialize)]
pub struct SubmitJobResponse {
    /// Job ID
    pub job_id: String,
}

/// Job status response.
#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    /// Job ID
    pub job_id: String,
    /// Status
    pub status: String,
    /// Progress (0-100)
    pub progress: f64,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Submits a new transcoding job.
pub async fn submit_job(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<SubmitJobRequest>,
) -> ServerResult<impl IntoResponse> {
    // Verify media exists and user has access
    let media = state.library.get_media(&req.media_id).await?;
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    // Validate codecs are patent-free
    if let Some(ref codec) = req.output_codec_video {
        let valid_codecs = ["av1", "vp9", "vp8", "theora"];
        if !valid_codecs.contains(&codec.to_lowercase().as_str()) {
            return Err(ServerError::BadRequest(format!(
                "Unsupported video codec: {codec}. Only patent-free codecs allowed (AV1, VP9, VP8, Theora)"
            )));
        }
    }

    if let Some(ref codec) = req.output_codec_audio {
        let valid_codecs = ["opus", "vorbis", "flac", "pcm"];
        if !valid_codecs.contains(&codec.to_lowercase().as_str()) {
            return Err(ServerError::BadRequest(format!(
                "Unsupported audio codec: {codec}. Only patent-free codecs allowed (Opus, Vorbis, FLAC, PCM)"
            )));
        }
    }

    // Create job
    let job = TranscodeJob::new(
        auth_user.user_id,
        req.media_id,
        req.output_format,
        req.output_codec_video,
        req.output_codec_audio,
        req.output_width,
        req.output_height,
        req.output_bitrate,
    );

    // Save to database
    crate::db::query(
        r"
        INSERT INTO transcode_jobs (
            id, user_id, media_id, status, progress,
            output_format, output_codec_video, output_codec_audio,
            output_width, output_height, output_bitrate,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(&job.id)
    .bind(&job.user_id)
    .bind(&job.media_id)
    .bind(job.status.to_string())
    .bind(job.progress)
    .bind(&job.output_format)
    .bind(&job.output_codec_video)
    .bind(&job.output_codec_audio)
    .bind(job.output_width)
    .bind(job.output_height)
    .bind(job.output_bitrate)
    .bind(job.created_at)
    .execute(state.db.pool())
    .await?;

    // Enqueue job for processing
    {
        let mut queue = state.job_queue.write().await;
        queue.enqueue(job.id.clone());
    }
    tracing::info!("Enqueued transcode job {}", job.id);

    Ok((
        StatusCode::CREATED,
        Json(SubmitJobResponse { job_id: job.id }),
    ))
}

/// Gets a transcoding job by ID.
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(job_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let row = crate::db::query(
        r"
        SELECT * FROM transcode_jobs WHERE id = ?
        ",
    )
    .bind(&job_id)
    .fetch_one(state.db.pool())
    .await
    .map_err(|_| ServerError::NotFound(format!("Job not found: {job_id}")))?;

    let user_id: String = row.get("user_id");

    // Verify ownership
    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this job".to_string(),
        ));
    }

    let job = row_to_job(&row)?;

    Ok(Json(job))
}

/// Cancels a transcoding job.
pub async fn cancel_job(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(job_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let row = crate::db::query("SELECT user_id, status FROM transcode_jobs WHERE id = ?")
        .bind(&job_id)
        .fetch_one(state.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Job not found: {job_id}")))?;

    let user_id: String = row.get("user_id");
    let status_str: String = row.get("status");

    // Verify ownership
    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this job".to_string(),
        ));
    }

    // Check if job can be cancelled
    if status_str == "completed" || status_str == "cancelled" || status_str == "failed" {
        return Err(ServerError::BadRequest(format!(
            "Cannot cancel job in {status_str} state"
        )));
    }

    // Update status
    crate::db::query("UPDATE transcode_jobs SET status = ? WHERE id = ?")
        .bind(TranscodeStatus::Cancelled.to_string())
        .bind(&job_id)
        .execute(state.db.pool())
        .await?;

    // Send cancellation signal to worker: remove from pending queue or
    // mark as cancelled so any in-progress worker will stop.
    {
        let mut queue = state.job_queue.write().await;
        queue.cancel(&job_id);
    }
    tracing::info!("Cancelled transcode job {}", job_id);

    Ok(StatusCode::NO_CONTENT)
}

/// Gets the status of a transcoding job.
pub async fn get_job_status(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(job_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let row = crate::db::query(
        r"
        SELECT user_id, status, progress, error_message
        FROM transcode_jobs
        WHERE id = ?
        ",
    )
    .bind(&job_id)
    .fetch_one(state.db.pool())
    .await
    .map_err(|_| ServerError::NotFound(format!("Job not found: {job_id}")))?;

    let user_id: String = row.get("user_id");

    // Verify ownership
    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this job".to_string(),
        ));
    }

    Ok(Json(JobStatusResponse {
        job_id,
        status: row.get("status"),
        progress: row.get("progress"),
        error: row.get("error_message"),
    }))
}

/// Lists transcoding jobs for the current user.
pub async fn list_jobs(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    let rows = crate::db::query(
        r"
        SELECT * FROM transcode_jobs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        ",
    )
    .bind(&auth_user.user_id)
    .fetch_all(state.db.pool())
    .await?;

    let jobs: Result<Vec<_>, _> = rows.iter().map(row_to_job).collect();
    let jobs = jobs?;

    Ok(Json(jobs))
}

/// Helper to convert database row to [`TranscodeJob`].
fn row_to_job(row: &crate::db::Row) -> ServerResult<TranscodeJob> {
    let status_str: String = row.get("status");
    let status = status_str
        .parse::<TranscodeStatus>()
        .map_err(ServerError::Internal)?;

    Ok(TranscodeJob {
        id: row.get("id"),
        user_id: row.get("user_id"),
        media_id: row.get("media_id"),
        status,
        progress: row.get("progress"),
        output_format: row.get("output_format"),
        output_codec_video: row.get("output_codec_video"),
        output_codec_audio: row.get("output_codec_audio"),
        output_width: row.get("output_width"),
        output_height: row.get("output_height"),
        output_bitrate: row.get("output_bitrate"),
        output_path: row.get("output_path"),
        error_message: row.get("error_message"),
        created_at: row.get("created_at"),
        started_at: row.get("started_at"),
        completed_at: row.get("completed_at"),
    })
}
