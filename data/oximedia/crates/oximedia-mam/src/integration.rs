//! Integration with other oximedia crates
//!
//! Provides integration for:
//! - oximedia-container for media analysis
//! - oximedia-transcode for proxy generation
//! - oximedia-qc for quality control
//! - oximedia-cloud for cloud storage
//! - oximedia-workflow for automation
//! - oximedia-metadata for metadata extraction
//! - oximedia-proxy for proxy management
//! - oximedia-search for full-text search

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{MamError, Result};

/// Integration manager for oximedia crates
pub struct IntegrationManager {
    /// Enable container analysis
    pub enable_container: bool,
    /// Enable transcoding
    pub enable_transcode: bool,
    /// Enable quality control
    pub enable_qc: bool,
    /// Enable cloud storage
    pub enable_cloud: bool,
}

impl IntegrationManager {
    /// Create a new integration manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_container: true,
            enable_transcode: true,
            enable_qc: true,
            enable_cloud: false,
        }
    }

    /// Analyze media file using oximedia-container
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails
    pub async fn analyze_container(&self, file_path: &Path) -> Result<ContainerAnalysis> {
        if !self.enable_container {
            return Err(MamError::Internal(
                "Container analysis not enabled".to_string(),
            ));
        }

        // Placeholder: In production, would use oximedia-container
        tracing::info!("Analyzing container: {:?}", file_path);

        Ok(ContainerAnalysis {
            format: Some("mp4".to_string()),
            duration_ms: Some(60000),
            bitrate: Some(5_000_000),
            video_tracks: vec![VideoTrack {
                codec: "h264".to_string(),
                width: 1920,
                height: 1080,
                frame_rate: 29.97,
                bitrate: Some(4_500_000),
            }],
            audio_tracks: vec![AudioTrack {
                codec: "aac".to_string(),
                sample_rate: 48000,
                channels: 2,
                bitrate: Some(128_000),
            }],
        })
    }

    /// Generate proxy using oximedia-transcode
    ///
    /// # Errors
    ///
    /// Returns an error if transcoding fails
    pub async fn generate_proxy_transcode(
        &self,
        source_path: &Path,
        dest_path: &Path,
        config: TranscodeConfig,
    ) -> Result<()> {
        if !self.enable_transcode {
            return Err(MamError::Internal("Transcoding not enabled".to_string()));
        }

        // Placeholder: In production, would use oximedia-transcode
        tracing::info!(
            "Transcoding: {:?} -> {:?} with config: {:?}",
            source_path,
            dest_path,
            config
        );

        Ok(())
    }

    /// Run quality control using oximedia-qc
    ///
    /// # Errors
    ///
    /// Returns an error if QC fails
    pub async fn run_quality_control(&self, file_path: &Path) -> Result<QualityControlReport> {
        if !self.enable_qc {
            return Err(MamError::Internal(
                "Quality control not enabled".to_string(),
            ));
        }

        // Placeholder: In production, would use oximedia-qc
        tracing::info!("Running quality control on: {:?}", file_path);

        Ok(QualityControlReport {
            overall_score: 9.5,
            video_quality: Some(9.8),
            audio_quality: Some(9.2),
            issues: vec![],
            warnings: vec![],
            passed: true,
        })
    }

    /// Upload to cloud using oximedia-cloud
    ///
    /// # Errors
    ///
    /// Returns an error if upload fails
    pub async fn upload_to_cloud(
        &self,
        file_path: &Path,
        destination: &str,
        provider: CloudProvider,
    ) -> Result<String> {
        if !self.enable_cloud {
            return Err(MamError::Internal("Cloud storage not enabled".to_string()));
        }

        // Placeholder: In production, would use oximedia-cloud
        tracing::info!(
            "Uploading to cloud: {:?} -> {} (provider: {:?})",
            file_path,
            destination,
            provider
        );

        Ok(format!("{}/{}", provider.as_str(), destination))
    }

    /// Extract metadata using oximedia-metadata
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails
    pub async fn extract_metadata(&self, file_path: &Path) -> Result<MediaMetadata> {
        // Placeholder: In production, would use oximedia-metadata
        tracing::info!("Extracting metadata from: {:?}", file_path);

        Ok(MediaMetadata {
            title: None,
            description: None,
            keywords: vec![],
            creator: None,
            creation_date: None,
            camera: None,
            location: None,
            exif: None,
            iptc: None,
            xmp: None,
        })
    }

    /// Detect scenes using oximedia-scene
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails
    pub async fn detect_scenes(&self, file_path: &Path) -> Result<Vec<Scene>> {
        // Placeholder: In production, would use oximedia-scene
        tracing::info!("Detecting scenes in: {:?}", file_path);

        Ok(vec![Scene {
            start_ms: 0,
            end_ms: 5000,
            score: 0.95,
            frame_count: 150,
        }])
    }

    /// Generate thumbnails using oximedia-proxy
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails
    pub async fn generate_thumbnails(
        &self,
        file_path: &Path,
        output_dir: &Path,
        count: usize,
    ) -> Result<Vec<String>> {
        // Placeholder: In production, would use oximedia-proxy
        tracing::info!(
            "Generating {} thumbnails from: {:?} to {:?}",
            count,
            file_path,
            output_dir
        );

        let thumbnails: Vec<String> = (0..count)
            .map(|i| format!("{}/thumb_{:04}.jpg", output_dir.display(), i))
            .collect();

        Ok(thumbnails)
    }
}

impl Default for IntegrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Container analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerAnalysis {
    pub format: Option<String>,
    pub duration_ms: Option<i64>,
    pub bitrate: Option<i64>,
    pub video_tracks: Vec<VideoTrack>,
    pub audio_tracks: Vec<AudioTrack>,
}

/// Video track information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTrack {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub bitrate: Option<i64>,
}

/// Audio track information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTrack {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub bitrate: Option<i64>,
}

/// Transcode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeConfig {
    pub video_codec: String,
    pub audio_codec: String,
    pub width: u32,
    pub height: u32,
    pub bitrate: i64,
    pub preset: String,
}

/// Quality control report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlReport {
    pub overall_score: f64,
    pub video_quality: Option<f64>,
    pub audio_quality: Option<f64>,
    pub issues: Vec<QualityIssue>,
    pub warnings: Vec<String>,
    pub passed: bool,
}

/// Quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub severity: IssueSeverity,
    pub category: String,
    pub description: String,
    pub timecode_ms: Option<i64>,
}

/// Issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical issue
    Critical,
    /// Major issue
    Major,
    /// Minor issue
    Minor,
    /// Warning
    Warning,
}

/// Cloud provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon S3
    S3,
    /// Azure Blob Storage
    Azure,
    /// Google Cloud Storage
    GCS,
}

impl CloudProvider {
    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::S3 => "s3",
            Self::Azure => "azure",
            Self::GCS => "gcs",
        }
    }
}

/// Media metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub creator: Option<String>,
    pub creation_date: Option<String>,
    pub camera: Option<String>,
    pub location: Option<Location>,
    pub exif: Option<serde_json::Value>,
    pub iptc: Option<serde_json::Value>,
    pub xmp: Option<serde_json::Value>,
}

/// Location metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub name: Option<String>,
}

/// Scene detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub start_ms: i64,
    pub end_ms: i64,
    pub score: f64,
    pub frame_count: u32,
}

// ── MamEventBus: broadcast-channel pub/sub hub ──────────────────────────────

use crate::event_bus::MamEvent;
use tokio::sync::broadcast;

/// Error returned when broadcasting an event fails.
///
/// This wraps `tokio::sync::broadcast::error::SendError` without exposing the
/// generic parameter, making it easier to use across crate boundaries.
#[derive(Debug, thiserror::Error)]
#[error("Broadcast send error: no active receivers — {0}")]
pub struct BroadcastError(pub String);

impl<T: std::fmt::Debug> From<broadcast::error::SendError<T>> for BroadcastError {
    fn from(e: broadcast::error::SendError<T>) -> Self {
        Self(format!("{e:?}"))
    }
}

/// Central publish/subscribe hub for MAM domain events.
///
/// Multiple subsystems subscribe via [`MamEventBus::subscribe`].  The ingest
/// pipeline (and folder sync) publish via [`MamEventBus::publish`] or by
/// cloning [`MamEventBus::sender`].
///
/// # Notes on slow subscribers
///
/// `tokio::sync::broadcast` uses a fixed-capacity ring buffer.  Slow receivers
/// that fall more than `capacity` messages behind will experience
/// [`broadcast::error::RecvError::Lagged`] on the next `recv()` call — they
/// are **not** blocked and do **not** slow down the sender.
pub struct MamEventBus {
    tx: broadcast::Sender<MamEvent>,
}

impl MamEventBus {
    /// Create a new bus with the given ring-buffer `capacity`.
    ///
    /// A capacity of 16–64 is sufficient for typical usage.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to all events on this bus.
    ///
    /// Subscribers receive events in the order they are published.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<MamEvent> {
        self.tx.subscribe()
    }

    /// Publish an event to all active subscribers.
    ///
    /// Returns the number of receivers that will receive the event.
    ///
    /// # Errors
    ///
    /// Returns [`BroadcastError`] if there are no active receivers (the message
    /// could not be delivered to anyone).
    pub fn publish(&self, event: MamEvent) -> std::result::Result<usize, BroadcastError> {
        self.tx.send(event).map_err(BroadcastError::from)
    }

    /// Clone the underlying sender so that other components can publish events
    /// without holding a reference to the bus itself (e.g. `FolderSync`).
    #[must_use]
    pub fn sender(&self) -> broadcast::Sender<MamEvent> {
        self.tx.clone()
    }

    /// Return the number of active receivers.
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_manager_new() {
        let manager = IntegrationManager::new();
        assert!(manager.enable_container);
        assert!(manager.enable_transcode);
        assert!(manager.enable_qc);
        assert!(!manager.enable_cloud);
    }

    #[test]
    fn test_cloud_provider_as_str() {
        assert_eq!(CloudProvider::S3.as_str(), "s3");
        assert_eq!(CloudProvider::Azure.as_str(), "azure");
        assert_eq!(CloudProvider::GCS.as_str(), "gcs");
    }

    #[test]
    fn test_container_analysis() {
        let analysis = ContainerAnalysis {
            format: Some("mp4".to_string()),
            duration_ms: Some(60000),
            bitrate: Some(5_000_000),
            video_tracks: vec![VideoTrack {
                codec: "h264".to_string(),
                width: 1920,
                height: 1080,
                frame_rate: 29.97,
                bitrate: Some(4_500_000),
            }],
            audio_tracks: vec![AudioTrack {
                codec: "aac".to_string(),
                sample_rate: 48000,
                channels: 2,
                bitrate: Some(128_000),
            }],
        };

        assert_eq!(analysis.format, Some("mp4".to_string()));
        assert_eq!(analysis.video_tracks.len(), 1);
        assert_eq!(analysis.audio_tracks.len(), 1);
    }

    #[test]
    fn test_transcode_config() {
        let config = TranscodeConfig {
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            width: 1280,
            height: 720,
            bitrate: 3_000_000,
            preset: "medium".to_string(),
        };

        assert_eq!(config.video_codec, "h264");
        assert_eq!(config.width, 1280);
    }

    #[test]
    fn test_quality_control_report() {
        let report = QualityControlReport {
            overall_score: 9.5,
            video_quality: Some(9.8),
            audio_quality: Some(9.2),
            issues: vec![],
            warnings: vec![],
            passed: true,
        };

        assert_eq!(report.overall_score, 9.5);
        assert!(report.passed);
    }

    #[test]
    fn test_scene() {
        let scene = Scene {
            start_ms: 0,
            end_ms: 5000,
            score: 0.95,
            frame_count: 150,
        };

        assert_eq!(scene.start_ms, 0);
        assert_eq!(scene.end_ms, 5000);
        assert_eq!(scene.frame_count, 150);
    }
}
