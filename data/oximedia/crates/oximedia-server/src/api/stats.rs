//! Statistics API endpoints.

use crate::{auth::AuthUser, error::ServerResult, AppState};
use axum::{extract::State, response::IntoResponse, Json};
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

// ── BandwidthTracker ─────────────────────────────────────────────────────────

/// Window duration for bandwidth calculations (60 seconds).
const BANDWIDTH_WINDOW_SECS: u64 = 60;

/// A single timestamped byte-count sample.
struct BandwidthSample {
    /// When the transfer was recorded.
    at: Instant,
    /// Number of bytes transferred.
    bytes: u64,
}

/// Rolling-window bandwidth tracker.
///
/// Maintains two separate deques of `(timestamp, bytes)` samples for upload
/// and download traffic. Samples older than `BANDWIDTH_WINDOW_SECS` are
/// discarded automatically whenever [`BandwidthTracker::reset_window`] is called or when
/// computing the current bandwidth.
pub struct BandwidthTracker {
    upload_samples: VecDeque<BandwidthSample>,
    download_samples: VecDeque<BandwidthSample>,
}

impl BandwidthTracker {
    /// Creates a new, empty bandwidth tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            upload_samples: VecDeque::new(),
            download_samples: VecDeque::new(),
        }
    }

    /// Records `bytes` bytes of upload traffic at the current instant.
    pub fn add_upload(&mut self, bytes: u64) {
        self.upload_samples.push_back(BandwidthSample {
            at: Instant::now(),
            bytes,
        });
    }

    /// Records `bytes` bytes of download traffic at the current instant.
    pub fn add_download(&mut self, bytes: u64) {
        self.download_samples.push_back(BandwidthSample {
            at: Instant::now(),
            bytes,
        });
    }

    /// Removes samples older than the rolling window from both queues.
    pub fn reset_window(&mut self) {
        let window = Duration::from_secs(BANDWIDTH_WINDOW_SECS);
        // Use checked_sub to avoid potential overflow on platforms where
        // Instant is not monotonic (e.g., very early in process lifetime).
        if let Some(cutoff) = Instant::now().checked_sub(window) {
            while self.upload_samples.front().is_some_and(|s| s.at < cutoff) {
                self.upload_samples.pop_front();
            }
            while self.download_samples.front().is_some_and(|s| s.at < cutoff) {
                self.download_samples.pop_front();
            }
        }
    }

    /// Returns the current upload bandwidth in bytes per second.
    ///
    /// Computed as total bytes in the rolling window divided by the actual
    /// elapsed window duration.
    #[must_use]
    pub fn upload_bps(&self) -> f64 {
        Self::compute_bps(&self.upload_samples)
    }

    /// Returns the current download bandwidth in bytes per second.
    #[must_use]
    pub fn download_bps(&self) -> f64 {
        Self::compute_bps(&self.download_samples)
    }

    fn compute_bps(samples: &VecDeque<BandwidthSample>) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let window = Duration::from_secs(BANDWIDTH_WINDOW_SECS);
        let Some(cutoff) = Instant::now().checked_sub(window) else {
            return 0.0;
        };
        let total_bytes: u64 = samples
            .iter()
            .filter(|s| s.at >= cutoff)
            .map(|s| s.bytes)
            .sum();
        // Use f64::from for the u32-range constant to avoid precision warnings.
        // BANDWIDTH_WINDOW_SECS is 60, well within f64 precision bounds.
        #[allow(clippy::cast_precision_loss)]
        let elapsed = BANDWIDTH_WINDOW_SECS as f64;
        #[allow(clippy::cast_precision_loss)]
        let result = total_bytes as f64 / elapsed;
        result
    }
}

impl Default for BandwidthTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Module-level bandwidth tracker shared across requests.
///
/// Uses a `Mutex` so it can be updated from multiple async tasks.
static BANDWIDTH: std::sync::OnceLock<Mutex<BandwidthTracker>> = std::sync::OnceLock::new();

/// Returns a reference to the global [`BandwidthTracker`] mutex.
fn bandwidth() -> &'static Mutex<BandwidthTracker> {
    BANDWIDTH.get_or_init(|| Mutex::new(BandwidthTracker::new()))
}

/// Records upload bytes for the current request.
///
/// Exposed so that upload handlers can call it after writing data.
#[allow(dead_code)]
pub fn record_upload(bytes: u64) {
    if let Ok(mut tracker) = bandwidth().lock() {
        tracker.reset_window();
        tracker.add_upload(bytes);
    }
}

/// Records download bytes for the current request.
///
/// Exposed so that streaming/download handlers can call it after sending data.
#[allow(dead_code)]
pub fn record_download(bytes: u64) {
    if let Ok(mut tracker) = bandwidth().lock() {
        tracker.reset_window();
        tracker.add_download(bytes);
    }
}

/// Server statistics.
#[derive(Debug, Serialize)]
pub struct ServerStats {
    /// Total users
    pub total_users: i64,
    /// Total media files
    pub total_media: i64,
    /// Total storage used (bytes)
    pub total_storage: i64,
    /// Active transcoding jobs
    pub active_jobs: i64,
    /// Total collections
    pub total_collections: i64,
}

/// Storage statistics.
#[derive(Debug, Serialize)]
pub struct StorageStats {
    /// Total files
    pub total_files: u64,
    /// Total size in bytes
    pub total_size: u64,
    /// Average file size
    pub avg_size: f64,
    /// Largest file size
    pub max_size: u64,
    /// Smallest file size
    pub min_size: u64,
    /// Storage by media type
    pub by_type: Vec<StorageByType>,
}

/// Storage statistics by media type.
#[derive(Debug, Serialize)]
pub struct StorageByType {
    /// Media type (video, audio, image)
    pub media_type: String,
    /// File count
    pub count: i64,
    /// Total size
    pub size: i64,
}

/// Bandwidth statistics.
#[derive(Debug, Serialize)]
pub struct BandwidthStats {
    /// Total streams served
    pub total_streams: i64,
    /// Total bytes transferred
    pub total_bytes: i64,
    /// Average bitrate
    pub avg_bitrate: f64,
}

/// Gets server statistics.
pub async fn get_stats(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    // Only admins can view server stats
    if !auth_user.is_admin() {
        // Return user-specific stats
        let total_media: i64 =
            crate::db::query_scalar("SELECT COUNT(*) FROM media WHERE user_id = ?")
                .bind(&auth_user.user_id)
                .fetch_one(state.db.pool())
                .await?;

        let total_storage: Option<i64> =
            crate::db::query_scalar("SELECT SUM(file_size) FROM media WHERE user_id = ?")
                .bind(&auth_user.user_id)
                .fetch_one(state.db.pool())
                .await?;

        let total_collections: i64 =
            crate::db::query_scalar("SELECT COUNT(*) FROM collections WHERE user_id = ?")
                .bind(&auth_user.user_id)
                .fetch_one(state.db.pool())
                .await?;

        return Ok(Json(serde_json::json!({
            "total_media": total_media,
            "total_storage": total_storage.unwrap_or(0),
            "total_collections": total_collections,
        })));
    }

    // Admin stats
    let total_users: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM users")
        .fetch_one(state.db.pool())
        .await?;

    let total_media: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM media")
        .fetch_one(state.db.pool())
        .await?;

    let total_storage: Option<i64> = crate::db::query_scalar("SELECT SUM(file_size) FROM media")
        .fetch_one(state.db.pool())
        .await?;

    let active_jobs: i64 = crate::db::query_scalar(
        "SELECT COUNT(*) FROM transcode_jobs WHERE status IN ('queued', 'processing')",
    )
    .fetch_one(state.db.pool())
    .await?;

    let total_collections: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM collections")
        .fetch_one(state.db.pool())
        .await?;

    Ok(Json(serde_json::json!({
        "total_users": total_users,
        "total_media": total_media,
        "total_storage": total_storage.unwrap_or(0),
        "active_jobs": active_jobs,
        "total_collections": total_collections,
    })))
}

/// Gets storage statistics.
pub async fn get_storage_stats(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    // Get overall storage stats
    let db_stats = state.db.get_storage_stats().await?;

    // Get stats by media type
    let type_rows = crate::db::query(
        r"
        SELECT
            CASE
                WHEN mime_type LIKE 'video/%' THEN 'video'
                WHEN mime_type LIKE 'audio/%' THEN 'audio'
                WHEN mime_type LIKE 'image/%' THEN 'image'
                ELSE 'other'
            END as media_type,
            COUNT(*) as count,
            SUM(file_size) as size
        FROM media
        WHERE user_id = ? AND status = 'ready'
        GROUP BY media_type
        ",
    )
    .bind(if auth_user.is_admin() {
        ""
    } else {
        &auth_user.user_id
    })
    .fetch_all(state.db.pool())
    .await?;

    let by_type: Vec<StorageByType> = type_rows
        .iter()
        .map(|row| StorageByType {
            media_type: row.get("media_type"),
            count: row.get("count"),
            size: row.get::<Option<i64>>("size").unwrap_or(0),
        })
        .collect();

    Ok(Json(StorageStats {
        total_files: db_stats.total_files,
        total_size: db_stats.total_size,
        avg_size: db_stats.avg_size,
        max_size: db_stats.max_size,
        min_size: db_stats.min_size,
        by_type,
    }))
}

/// Gets bandwidth statistics.
pub async fn get_bandwidth_stats(
    State(_state): State<Arc<AppState>>,
    _auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    // Read current bandwidth from the rolling-window tracker.
    let (upload_bps, download_bps) = if let Ok(mut tracker) = bandwidth().lock() {
        tracker.reset_window();
        (tracker.upload_bps(), tracker.download_bps())
    } else {
        (0.0, 0.0)
    };

    let total_bps = upload_bps + download_bps;
    // avg_bitrate is the combined upload+download bandwidth in bits per second.
    let avg_bitrate = total_bps * 8.0;

    Ok(Json(BandwidthStats {
        total_streams: 0, // Stream count tracking requires integration with the streaming layer.
        total_bytes: 0, // Cumulative totals are not persisted across restarts in this implementation.
        avg_bitrate,
    }))
}
