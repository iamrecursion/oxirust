//! Media management API endpoints.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    models::media::Media,
    upload::UploadManager,
    AppState,
};
use axum::{
    body::{Body, Bytes},
    extract::{Multipart, Path, Query, State},
    http::{
        header::{HeaderName, HeaderValue, CONTENT_TYPE},
        Response, StatusCode,
    },
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Upload file response.
#[derive(Debug, Serialize)]
pub struct UploadResponse {
    /// Media ID
    pub media_id: String,
    /// Filename
    pub filename: String,
}

/// List media query parameters.
#[derive(Debug, Deserialize)]
pub struct ListMediaQuery {
    /// Limit
    #[serde(default = "default_limit")]
    pub limit: i64,
    /// Offset
    #[serde(default)]
    pub offset: i64,
}

const fn default_limit() -> i64 {
    50
}

/// Update media request.
#[derive(Debug, Deserialize)]
pub struct UpdateMediaRequest {
    /// New filename (optional)
    pub filename: Option<String>,
}

/// Init multipart upload request.
#[derive(Debug, Deserialize)]
pub struct InitMultipartRequest {
    /// Filename
    pub filename: String,
    /// Total file size
    pub total_size: i64,
    /// Chunk size (default: 5MB)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: i64,
}

const fn default_chunk_size() -> i64 {
    5 * 1024 * 1024
}

/// Init multipart upload response.
#[derive(Debug, Serialize)]
pub struct InitMultipartResponse {
    /// Upload ID
    pub upload_id: String,
    /// Total chunks
    pub total_chunks: i32,
}

/// Upload part response.
#[derive(Debug, Serialize)]
pub struct UploadPartResponse {
    /// Chunk number
    pub chunk_number: i32,
    /// Progress percentage
    pub progress: f64,
}

/// Complete multipart upload response.
#[derive(Debug, Serialize)]
pub struct CompleteMultipartResponse {
    /// Media ID
    pub media_id: String,
}

/// Uploads a media file.
pub async fn upload_file(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    mut multipart: Multipart,
) -> ServerResult<impl IntoResponse> {
    let mut filename = String::new();
    let mut data = Vec::new();

    // Extract file from multipart form
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ServerError::UploadFailed(format!("Multipart error: {e}")))?
    {
        let name = field.name().unwrap_or_default().to_string();

        if name == "file" {
            filename = field.file_name().unwrap_or("unnamed").to_string();
            data = field
                .bytes()
                .await
                .map_err(|e| ServerError::UploadFailed(format!("Failed to read file: {e}")))?
                .to_vec();
        }
    }

    if filename.is_empty() || data.is_empty() {
        return Err(ServerError::BadRequest("No file provided".to_string()));
    }

    // Detect MIME type
    let mime_type = mime_guess::from_path(&filename)
        .first_or_octet_stream()
        .to_string();

    // Generate unique filename
    let ext = std::path::Path::new(&filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let unique_filename = format!("{}_{}.{}", Uuid::new_v4(), filename, ext);

    // Save file
    let file_path = state.config.media_dir.join(&unique_filename);
    tokio::fs::write(&file_path, &data).await?;

    // Create media entry
    let media = Media::new(
        auth_user.user_id,
        unique_filename,
        filename,
        mime_type,
        i64::try_from(data.len()).unwrap_or(i64::MAX),
    );

    state.library.add_media(&media).await?;

    Ok((
        StatusCode::CREATED,
        Json(UploadResponse {
            media_id: media.id,
            filename: media.original_filename,
        }),
    ))
}

/// Initializes a multipart upload.
pub async fn init_multipart_upload(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<InitMultipartRequest>,
) -> ServerResult<impl IntoResponse> {
    let upload_manager = UploadManager::new(state.db.clone(), state.config.clone());
    let upload = upload_manager
        .init_upload(
            &auth_user.user_id,
            &req.filename,
            req.total_size,
            req.chunk_size,
        )
        .await?;

    Ok((
        StatusCode::CREATED,
        Json(InitMultipartResponse {
            upload_id: upload.id,
            total_chunks: upload.total_chunks,
        }),
    ))
}

/// Uploads a part of a multipart upload.
pub async fn upload_part(
    State(state): State<Arc<AppState>>,
    _auth_user: AuthUser,
    Path(upload_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    body: Bytes,
) -> ServerResult<impl IntoResponse> {
    let chunk_number = params
        .get("chunk_number")
        .and_then(|s| s.parse::<i32>().ok())
        .ok_or_else(|| ServerError::BadRequest("Missing chunk_number parameter".to_string()))?;

    let upload_manager = UploadManager::new(state.db.clone(), state.config.clone());
    upload_manager
        .upload_chunk(&upload_id, chunk_number, body.to_vec())
        .await?;

    let upload = upload_manager.get_upload(&upload_id).await?;
    let progress = (f64::from(upload.completed_chunks) / f64::from(upload.total_chunks)) * 100.0;

    Ok(Json(UploadPartResponse {
        chunk_number,
        progress,
    }))
}

/// Completes a multipart upload.
pub async fn complete_multipart_upload(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(upload_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let upload_manager = UploadManager::new(state.db.clone(), state.config.clone());
    let file_path = upload_manager.complete_upload(&upload_id).await?;

    // Get upload info
    let upload = upload_manager.get_upload(&upload_id).await?;

    // Detect MIME type
    let mime_type = mime_guess::from_path(&upload.filename)
        .first_or_octet_stream()
        .to_string();

    // Move file to media directory
    let ext = std::path::Path::new(&upload.filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let unique_filename = format!("{}_{}.{}", Uuid::new_v4(), upload.filename, ext);
    let final_path = state.config.media_dir.join(&unique_filename);
    tokio::fs::rename(&file_path, &final_path).await?;

    // Create media entry
    let media = Media::new(
        auth_user.user_id,
        unique_filename,
        upload.filename,
        mime_type,
        upload.total_size,
    );

    state.library.add_media(&media).await?;

    Ok(Json(CompleteMultipartResponse { media_id: media.id }))
}

/// Aborts a multipart upload.
pub async fn abort_multipart_upload(
    State(state): State<Arc<AppState>>,
    _auth_user: AuthUser,
    Path(upload_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let upload_manager = UploadManager::new(state.db.clone(), state.config.clone());
    upload_manager.abort_upload(&upload_id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Lists media files for the current user.
pub async fn list_media(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Query(query): Query<ListMediaQuery>,
) -> ServerResult<impl IntoResponse> {
    let media = state
        .library
        .list_media(&auth_user.user_id, query.limit, query.offset)
        .await?;

    Ok(Json(media))
}

/// Gets media metadata by ID.
pub async fn get_media(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    // Verify ownership
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    Ok(Json(media))
}

/// Updates media metadata.
pub async fn update_media(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(media_id): Path<String>,
    Json(req): Json<UpdateMediaRequest>,
) -> ServerResult<impl IntoResponse> {
    let mut media = state.library.get_media(&media_id).await?;

    // Verify ownership
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    if let Some(filename) = req.filename {
        media.original_filename = filename;
    }

    state.library.update_media(&media).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a media file.
pub async fn delete_media(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    // Verify ownership
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    state.library.delete_media(&media_id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Gets metadata for a media file.
pub async fn get_metadata(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    // Verify ownership
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    let metadata = state.library.get_metadata(&media_id).await?;

    Ok(Json(metadata))
}

/// Gets a thumbnail for a media file.
pub async fn get_thumbnail(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    let thumbnail_path = media
        .thumbnail_path
        .ok_or_else(|| ServerError::NotFound("Thumbnail not available".to_string()))?;

    let full_path = state.config.thumbnail_dir.join(thumbnail_path);
    let data = tokio::fs::read(full_path).await?;

    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
        data,
    ))
}

/// Generates a preview for a media file.
pub async fn generate_preview(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    // Verify ownership
    if media.user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this media".to_string(),
        ));
    }

    generate_rgba_preview(state, media).await
}

/// Preview dimensions and hash constants.
const PREVIEW_WIDTH: u32 = 64;
const PREVIEW_HEIGHT: u32 = 36;
const PREVIEW_HEADER_BYTES: usize = 512;
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Generates a deterministic 64×36 RGBA gradient preview for a media file.
///
/// Takes the first 512 bytes of the file and uses an FNV-1a hash to seed a
/// gradient pattern.  The result is raw RGBA pixels (4 bytes per pixel).
///
/// # Errors
///
/// Returns an error if the HTTP response builder fails.
#[allow(clippy::cast_possible_truncation)]
async fn generate_rgba_preview(
    state: Arc<AppState>,
    media: crate::models::media::Media,
) -> ServerResult<impl IntoResponse> {
    let file_path = state.config.media_dir.join(&media.filename);
    let header = match tokio::fs::read(&file_path).await {
        Ok(data) => data[..data.len().min(PREVIEW_HEADER_BYTES)].to_vec(),
        Err(_) => Vec::new(),
    };

    // FNV-1a hash seed.  `i as u64` is a widening cast on all supported
    // platforms (usize ≤ 64 bits), so no precision is lost.
    let seed: u64 = header.iter().enumerate().fold(FNV_OFFSET, |h, (i, &b)| {
        h.wrapping_mul(FNV_PRIME)
            .wrapping_add(u64::from(b))
            .wrapping_add(i as u64)
    });

    // Build raw RGBA pixels: gradient pattern seeded by `seed`.
    // The truncating casts (u64→u32→u8) are intentional for wrapping colours.
    let pixel_count = (PREVIEW_WIDTH * PREVIEW_HEIGHT) as usize;
    let mut pixels: Vec<u8> = Vec::with_capacity(pixel_count * 4);
    for y in 0..PREVIEW_HEIGHT {
        for x in 0..PREVIEW_WIDTH {
            let r = ((seed & 0xff) as u32)
                .wrapping_add(x * 3)
                .wrapping_add(y * 2) as u8;
            let g = (((seed >> 8) & 0xff) as u32)
                .wrapping_add(x * 2)
                .wrapping_add(y * 3) as u8;
            let b_val = (((seed >> 16) & 0xff) as u32)
                .wrapping_add(x)
                .wrapping_add(y * 4) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b_val);
            pixels.push(0xff); // alpha = opaque
        }
    }

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/octet-stream")
        .header(
            HeaderName::from_static("x-preview-width"),
            HeaderValue::from_static("64"),
        )
        .header(
            HeaderName::from_static("x-preview-height"),
            HeaderValue::from_static("36"),
        )
        .header(
            HeaderName::from_static("x-preview-format"),
            HeaderValue::from_static("rgba"),
        )
        .body(Body::from(pixels))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(response)
}

/// Gets a thumbnail sprite sheet.
pub async fn get_thumbnail_sprite(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    let media = state.library.get_media(&media_id).await?;

    let sprite_path = media
        .sprite_path
        .ok_or_else(|| ServerError::NotFound("Thumbnail sprite not available".to_string()))?;

    let full_path = state.config.thumbnail_dir.join(sprite_path);
    let data = tokio::fs::read(full_path).await?;

    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
        data,
    ))
}
