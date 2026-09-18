//! Streaming management API routes.

use crate::{
    error::{ServerError, ServerResult},
    streaming_server::StreamingServer,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Stream information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    /// Stream key.
    pub stream_key: String,

    /// Application name.
    pub app_name: String,

    /// Is active.
    pub active: bool,

    /// Viewer count.
    pub viewer_count: u64,

    /// Bytes ingested.
    pub bytes_ingested: u64,

    /// Duration in seconds.
    pub duration: u64,
}

/// List all active streams.
pub async fn list_streams(
    State(server): State<Arc<StreamingServer>>,
) -> ServerResult<Json<Vec<StreamInfo>>> {
    let streams = server.rtmp_server().list_streams();

    let stream_infos: Vec<StreamInfo> = streams
        .iter()
        .map(|stream| StreamInfo {
            stream_key: stream.stream_key.clone(),
            app_name: stream.app_name.clone(),
            active: true,
            viewer_count: 0,
            bytes_ingested: *stream.bytes_received.read(),
            duration: stream.start_time.elapsed().as_secs(),
        })
        .collect();

    Ok(Json(stream_infos))
}

/// Get stream details.
pub async fn get_stream(
    State(server): State<Arc<StreamingServer>>,
    Path((app_name, stream_key)): Path<(String, String)>,
) -> ServerResult<Json<StreamInfo>> {
    let stream = server
        .rtmp_server()
        .get_stream(&app_name, &stream_key)
        .ok_or_else(|| ServerError::NotFound("Stream not found".to_string()))?;

    let info = StreamInfo {
        stream_key: stream.stream_key.clone(),
        app_name: stream.app_name.clone(),
        active: true,
        viewer_count: 0,
        bytes_ingested: *stream.bytes_received.read(),
        duration: stream.start_time.elapsed().as_secs(),
    };

    Ok(Json(info))
}

/// Stop a stream.
pub async fn stop_stream(
    State(server): State<Arc<StreamingServer>>,
    Path((app_name, stream_key)): Path<(String, String)>,
) -> ServerResult<impl IntoResponse> {
    server.stop_stream(&app_name, &stream_key).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get server metrics.
pub async fn get_metrics(
    State(server): State<Arc<StreamingServer>>,
) -> ServerResult<impl IntoResponse> {
    let metrics = server.metrics();
    let exporter = crate::metrics::PrometheusExporter::new(Arc::clone(metrics));
    let content = exporter.export();

    Ok((
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        content,
    ))
}

/// Get server stats.
#[derive(Debug, Clone, Serialize)]
pub struct ServerStats {
    /// Active stream count.
    pub active_streams: usize,

    /// Total bytes received.
    pub bytes_received: f64,

    /// Total bytes sent.
    pub bytes_sent: f64,

    /// Uptime in seconds.
    pub uptime_seconds: u64,
}

/// Gets server statistics.
pub async fn get_stats(
    State(server): State<Arc<StreamingServer>>,
) -> ServerResult<Json<ServerStats>> {
    let metrics = server.metrics();

    let stats = ServerStats {
        active_streams: server.active_stream_count(),
        bytes_received: metrics
            .get_metric("bytes_received_total")
            .map(|m| m.value)
            .unwrap_or(0.0),
        bytes_sent: metrics
            .get_metric("bytes_sent_total")
            .map(|m| m.value)
            .unwrap_or(0.0),
        uptime_seconds: metrics.uptime().as_secs(),
    };

    Ok(Json(stats))
}
