//! HTTP handlers for streaming endpoints.

use crate::{
    error::{ServerError, ServerResult},
    mmap_segments::{MmapError, SegmentKey, SegmentStore},
    AppState,
};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tokio::fs::File;
use tokio_util::io::ReaderStream;

/// Serves HLS master playlist.
pub async fn serve_hls_master(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> ServerResult<Response> {
    let media = state.library.get_media(&media_id).await?;
    let hls = super::hls::HlsGenerator::new(state.config.clone());
    let playlist = hls.generate_master_playlist(&media)?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")],
        playlist,
    )
        .into_response())
}

/// Serves HLS media playlist for a variant.
pub async fn serve_hls_playlist(
    State(state): State<Arc<AppState>>,
    Path((media_id, variant)): Path<(String, String)>,
) -> ServerResult<Response> {
    let media = state.library.get_media(&media_id).await?;
    let hls = super::hls::HlsGenerator::new(state.config.clone());
    let playlist = hls.generate_media_playlist(&media, &variant)?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")],
        playlist,
    )
        .into_response())
}

/// Serves HLS segment via the memory-mapped segment store.
pub async fn serve_hls_segment(
    State(state): State<Arc<AppState>>,
    Path((media_id, variant, segment)): Path<(String, String, usize)>,
    headers: HeaderMap,
) -> ServerResult<Response> {
    let hls = super::hls::HlsGenerator::new(state.config.clone());
    let segment_path = hls.get_segment_path(&media_id, &variant, segment);

    let key = SegmentKey::new(&media_id, &variant, segment as u32);
    serve_segment_mmap(
        &state.segment_store,
        key,
        &segment_path,
        &headers,
        "video/mp2t",
    )
}

/// Serves DASH manifest.
pub async fn serve_dash_manifest(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> ServerResult<Response> {
    let media = state.library.get_media(&media_id).await?;
    let dash = super::dash::DashGenerator::new(state.config.clone());
    let manifest = dash.generate_manifest(&media)?;

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/dash+xml")],
        manifest,
    )
        .into_response())
}

/// Serves DASH initialization segment via the memory-mapped segment store.
pub async fn serve_dash_init(
    State(state): State<Arc<AppState>>,
    Path((media_id, variant)): Path<(String, String)>,
    headers: HeaderMap,
) -> ServerResult<Response> {
    let dash = super::dash::DashGenerator::new(state.config.clone());
    let init_path = dash.get_init_path(&media_id, &variant);

    let key = SegmentKey::init(&media_id, &variant);
    serve_segment_mmap(&state.segment_store, key, &init_path, &headers, "video/mp4")
}

/// Serves DASH media segment via the memory-mapped segment store.
pub async fn serve_dash_segment(
    State(state): State<Arc<AppState>>,
    Path((media_id, variant, segment)): Path<(String, String, usize)>,
    headers: HeaderMap,
) -> ServerResult<Response> {
    let dash = super::dash::DashGenerator::new(state.config.clone());
    let segment_path = dash.get_segment_path(&media_id, &variant, segment);

    let key = SegmentKey::new(&media_id, &variant, segment as u32);
    serve_segment_mmap(
        &state.segment_store,
        key,
        &segment_path,
        &headers,
        "video/mp4",
    )
}

/// Serves progressive download (full file streaming).
pub async fn serve_progressive(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
    headers: HeaderMap,
) -> ServerResult<Response> {
    let media = state.library.get_media(&media_id).await?;
    let file_path = state.config.media_dir.join(&media.filename);

    if !file_path.exists() {
        return Err(ServerError::NotFound("Media file not found".to_string()));
    }

    serve_file_with_range(file_path, headers, &media.mime_type).await
}

/// Serves file download (with Content-Disposition attachment).
pub async fn serve_download(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> ServerResult<Response> {
    let media = state.library.get_media(&media_id).await?;
    let file_path = state.config.media_dir.join(&media.filename);

    if !file_path.exists() {
        return Err(ServerError::NotFound("Media file not found".to_string()));
    }

    let file = File::open(&file_path).await?;
    let metadata = file.metadata().await?;
    let file_size = metadata.len();

    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, media.mime_type.as_str()),
            (header::CONTENT_LENGTH, &file_size.to_string()),
            (
                header::CONTENT_DISPOSITION,
                &format!("attachment; filename=\"{}\"", media.original_filename),
            ),
        ],
        body,
    )
        .into_response())
}

// ── mmap segment helper ───────────────────────────────────────────────────────

/// Serves a media segment from the memory-mapped [`SegmentStore`].
///
/// Resolves the segment from the cache (or maps it from disk on first access),
/// then builds an axum `Response` with range-request support:
/// - Returns `206 Partial Content` with `Content-Range` when a valid `Range`
///   header is present.
/// - Returns `200 OK` with the full segment otherwise.
///
/// # Errors
/// - `404 Not Found` if the file does not exist on disk.
/// - `500 Internal Server Error` for I/O or oversized-file errors.
pub fn serve_segment_mmap(
    store: &SegmentStore,
    key: SegmentKey,
    path: &std::path::Path,
    headers: &HeaderMap,
    content_type: &str,
) -> ServerResult<Response> {
    let segment = store.get_or_map(key, path).map_err(|e| match e {
        MmapError::NotFound(p) => {
            ServerError::NotFound(format!("segment not found: {}", p.display()))
        }
        MmapError::Io(io_err) => ServerError::Io(io_err),
        MmapError::TooLarge { size, limit } => {
            ServerError::Internal(format!("segment {size} B exceeds limit {limit} B"))
        }
        MmapError::CacheFull => ServerError::Internal("segment cache is full".to_string()),
    })?;

    let total_len = segment.len as u64;

    // Check for a Range header and serve a partial response if valid.
    if let Some(range_header) = headers.get(header::RANGE) {
        if let Ok(range_str) = range_header.to_str() {
            if let Some((start, end)) = crate::mmap_segments::parse_byte_range(range_str, total_len)
            {
                let length = end - start + 1;
                let body_bytes: Vec<u8> = segment
                    .slice(start as usize, (end + 1) as usize)
                    .unwrap_or_default()
                    .to_vec();

                return Ok((
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type),
                        (header::CONTENT_LENGTH, &length.to_string()),
                        (
                            header::CONTENT_RANGE,
                            &format!("bytes {start}-{end}/{total_len}"),
                        ),
                        (header::ACCEPT_RANGES, "bytes"),
                    ],
                    body_bytes,
                )
                    .into_response());
            }
        }
    }

    // Full segment response.
    let body_bytes = segment.bytes().to_vec();
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CONTENT_LENGTH, &total_len.to_string()),
            (header::ACCEPT_RANGES, "bytes"),
        ],
        body_bytes,
    )
        .into_response())
}

/// Helper function to serve a file with range request support.
async fn serve_file_with_range(
    file_path: std::path::PathBuf,
    headers: HeaderMap,
    content_type: &str,
) -> ServerResult<Response> {
    let file = File::open(&file_path).await?;
    let metadata = file.metadata().await?;
    let file_size = metadata.len();

    // Check for Range header
    if let Some(range_header) = headers.get(header::RANGE) {
        if let Ok(range_str) = range_header.to_str() {
            if let Some(range) = parse_range(range_str, file_size) {
                let (start, end) = range;
                let length = end - start + 1;

                // Seek to start position
                use tokio::io::{AsyncReadExt, AsyncSeekExt};
                let mut file = file;
                file.seek(std::io::SeekFrom::Start(start)).await?;

                // Read the range
                let mut buffer = vec![0u8; length as usize];
                file.read_exact(&mut buffer).await?;

                return Ok((
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type),
                        (header::CONTENT_LENGTH, &length.to_string()),
                        (
                            header::CONTENT_RANGE,
                            &format!("bytes {}-{}/{}", start, end, file_size),
                        ),
                        (header::ACCEPT_RANGES, "bytes"),
                    ],
                    buffer,
                )
                    .into_response());
            }
        }
    }

    // No range request, serve full file
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CONTENT_LENGTH, &file_size.to_string()),
            (header::ACCEPT_RANGES, "bytes"),
        ],
        body,
    )
        .into_response())
}

/// Parses a Range header.
fn parse_range(range_str: &str, file_size: u64) -> Option<(u64, u64)> {
    let range_str = range_str.strip_prefix("bytes=")?;
    let parts: Vec<&str> = range_str.split('-').collect();

    if parts.len() != 2 {
        return None;
    }

    let start = if parts[0].is_empty() {
        // Suffix range: -500 means last 500 bytes
        let suffix = parts[1].parse::<u64>().ok()?;
        file_size.saturating_sub(suffix)
    } else {
        parts[0].parse::<u64>().ok()?
    };

    let end = if parts[1].is_empty() {
        file_size - 1
    } else {
        parts[1].parse::<u64>().ok()?.min(file_size - 1)
    };

    if start > end || start >= file_size {
        return None;
    }

    Some((start, end))
}
