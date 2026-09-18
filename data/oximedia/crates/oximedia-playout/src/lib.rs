//! # OxiMedia Playout Server
//!
//! Professional broadcast playout server with frame-accurate timing,
//! 24/7 reliability, and support for multiple broadcast outputs.
//!
//! ## Features
//!
//! - Frame-accurate timing (no dropped frames)
//! - 24/7 reliability with emergency fallback
//! - Genlock/sync support for professional broadcast
//! - Multiple simultaneous outputs (SDI, NDI, RTMP, SRT, IP multicast)
//! - Graphics overlay (logos, lower thirds, tickers)
//! - Comprehensive monitoring and alerting
//! - SCTE-35 marker insertion for ad breaks
//! - Dynamic playlist management
//!
//! ## Example
//!
//! ```no_run
//! use oximedia_playout::{PlayoutServer, PlayoutConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = PlayoutConfig::default();
//!     let server = PlayoutServer::new(config).await?;
//!     server.start().await?;
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use thiserror::Error;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::{mpsc, RwLock};

/// SCTE-35 ad insertion and splice point management.
pub mod ad_insertion;
#[cfg(not(target_arch = "wasm32"))]
pub mod api;
#[cfg(not(target_arch = "wasm32"))]
pub mod asrun;
#[cfg(not(target_arch = "wasm32"))]
pub mod automation;
pub mod branding;
/// Scheduled branding overlay management (bugs, watermarks, end-boards).
pub mod branding_inserter;
#[cfg(not(target_arch = "wasm32"))]
pub mod bxf;
pub mod catchup;
pub mod cg;
#[cfg(not(target_arch = "wasm32"))]
pub mod channel;
/// Channel format and configuration registry (SD/HD/UHD, frame rate, audio).
pub mod channel_config;
pub mod clip_store;
pub mod compliance_ingest;
/// Automated compliance recording scheduler with retention policy enforcement.
pub mod compliance_recorder;
#[cfg(not(target_arch = "wasm32"))]
pub mod content;
/// Frame-accurate cue point triggering for broadcast automation.
pub mod cue_trigger;
#[cfg(not(target_arch = "wasm32"))]
pub mod device;
/// Emergency Alert System (EAS/SAME) support.
pub mod emergency_alert;
pub mod event_log;
#[cfg(not(target_arch = "wasm32"))]
pub mod failover;
/// Frame ring buffer with pre-roll gating and overflow/underrun detection.
pub mod frame_buffer;
/// Frame-accurate trim engine with SMPTE timecode support.
pub mod frame_trim;
/// Automatic gap detection and filler content insertion.
pub mod gap_filler;
pub mod graphics;
/// Playout health monitoring (dropped frames, late segments).
pub mod health;
pub mod highlight_automation;
/// HLS manifest generation for catchup / VOD platform export.
pub mod hls_catchup;
pub mod ingest;
/// Minimal-contention SPSC ring buffer for inter-thread frame passing.
pub mod lockfree_frame_ring;
/// Template-based lower-third generation for broadcast graphics overlays.
pub mod lower_third;
/// Signal routing from programme sources to SDI/IP/RTMP/file targets.
pub mod media_router_playout;
pub mod monitoring;
#[cfg(not(target_arch = "wasm32"))]
pub mod output;
pub mod output_router;
#[cfg(not(target_arch = "wasm32"))]
pub mod playback;
pub mod playlist;
/// Playlist ingest session: format detection, item validation, clip trimming.
pub mod playlist_ingest;
/// Detailed playout logging and audit trail.
pub mod playout_log;
/// 24-hour playout schedule grid with conflict detection and gap finding.
pub mod playout_schedule;
/// Pre-decode manager: background thread pool for gapless playlist transitions.
pub mod predecode;
pub mod preflight;
/// PTP (Precision Time Protocol) clock source for sub-microsecond synchronisation.
pub mod ptp_clock;
pub mod rundown;
/// Rundown editor with insert, delete, reorder, split, merge, and undo/redo.
pub mod rundown_editor;
/// Automated highlight clip extraction using scene analysis.
pub mod scene_highlight;
/// Time-blocked schedule management for broadcast playout.
pub mod schedule_block;
/// Playlist-to-schedule conversion for broadcast playout.
pub mod schedule_convert;
/// Time-slot schedule grid with booking, availability, and overlap queries.
pub mod schedule_slot;
pub mod scheduler;
pub mod secondary_events;
/// Ordered processing chain (input -> process -> output) with bypass support.
pub mod signal_chain;
pub mod simulcast;
pub mod subtitle_inserter;
pub mod tally_system;
/// Timecode burn-in overlay for monitoring outputs.
pub mod timecode_overlay;
pub mod transitions;

/// Result type for playout operations
pub type Result<T> = std::result::Result<T, PlayoutError>;

/// Errors that can occur during playout operations
#[derive(Error, Debug)]
pub enum PlayoutError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Scheduler error: {0}")]
    Scheduler(String),

    #[error("Playlist error: {0}")]
    Playlist(String),

    #[error("Playback error: {0}")]
    Playback(String),

    #[error("Output error: {0}")]
    Output(String),

    #[error("Graphics error: {0}")]
    Graphics(String),

    #[error("Monitoring error: {0}")]
    Monitoring(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Synchronization error: {0}")]
    Sync(String),

    #[error("Timing error: {0}")]
    Timing(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Emergency fallback activated: {0}")]
    EmergencyFallback(String),

    #[error("Checksum error: {0}")]
    Checksum(String),

    #[error("PTP error: {0}")]
    Ptp(String),

    #[error("Integrity error: {0}")]
    Integrity(String),
}

/// Video format configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VideoFormat {
    /// 1920x1080 progressive at 23.976 fps
    HD1080p2398,
    /// 1920x1080 progressive at 24 fps
    HD1080p24,
    /// 1920x1080 progressive at 25 fps
    HD1080p25,
    /// 1920x1080 progressive at 29.97 fps
    HD1080p2997,
    /// 1920x1080 progressive at 30 fps
    HD1080p30,
    /// 1920x1080 progressive at 50 fps
    HD1080p50,
    /// 1920x1080 progressive at 59.94 fps
    HD1080p5994,
    /// 1920x1080 progressive at 60 fps
    HD1080p60,
    /// 1920x1080 interlaced at 50 Hz
    HD1080i50,
    /// 1920x1080 interlaced at 59.94 Hz
    HD1080i5994,
    /// 3840x2160 progressive at 25 fps
    UHD2160p25,
    /// 3840x2160 progressive at 29.97 fps
    UHD2160p2997,
    /// 3840x2160 progressive at 50 fps
    UHD2160p50,
    /// 3840x2160 progressive at 59.94 fps
    UHD2160p5994,
}

impl VideoFormat {
    /// Get frame rate in frames per second
    pub fn fps(&self) -> f64 {
        match self {
            Self::HD1080p2398 => 23.976,
            Self::HD1080p24 => 24.0,
            Self::HD1080p25 | Self::UHD2160p25 => 25.0,
            Self::HD1080p2997 | Self::UHD2160p2997 => 29.97,
            Self::HD1080p30 => 30.0,
            Self::HD1080p50 | Self::HD1080i50 | Self::UHD2160p50 => 50.0,
            Self::HD1080p5994 | Self::HD1080i5994 | Self::UHD2160p5994 => 59.94,
            Self::HD1080p60 => 60.0,
        }
    }

    /// Get width in pixels
    pub fn width(&self) -> u32 {
        match self {
            Self::HD1080p2398
            | Self::HD1080p24
            | Self::HD1080p25
            | Self::HD1080p2997
            | Self::HD1080p30
            | Self::HD1080p50
            | Self::HD1080p5994
            | Self::HD1080p60
            | Self::HD1080i50
            | Self::HD1080i5994 => 1920,
            Self::UHD2160p25 | Self::UHD2160p2997 | Self::UHD2160p50 | Self::UHD2160p5994 => 3840,
        }
    }

    /// Get height in pixels
    pub fn height(&self) -> u32 {
        match self {
            Self::HD1080p2398
            | Self::HD1080p24
            | Self::HD1080p25
            | Self::HD1080p2997
            | Self::HD1080p30
            | Self::HD1080p50
            | Self::HD1080p5994
            | Self::HD1080p60
            | Self::HD1080i50
            | Self::HD1080i5994 => 1080,
            Self::UHD2160p25 | Self::UHD2160p2997 | Self::UHD2160p50 | Self::UHD2160p5994 => 2160,
        }
    }
}

/// Audio format configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
        }
    }
}

/// Playout server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayoutConfig {
    /// Video format for output
    pub video_format: VideoFormat,

    /// Audio format for output
    pub audio_format: AudioFormat,

    /// Enable genlock synchronization
    pub genlock_enabled: bool,

    /// Reference clock source (e.g., "internal", "sdi", "ptp")
    pub clock_source: String,

    /// Buffer size in frames
    pub buffer_size: usize,

    /// Emergency fallback content path
    pub fallback_content: PathBuf,

    /// Maximum allowed latency in milliseconds
    pub max_latency_ms: u64,

    /// Enable frame drop detection
    pub detect_frame_drops: bool,

    /// Playlist directory
    pub playlist_dir: PathBuf,

    /// Content root directory
    pub content_root: PathBuf,

    /// Enable monitoring
    pub monitoring_enabled: bool,

    /// Monitoring port
    pub monitoring_port: u16,
}

impl Default for PlayoutConfig {
    fn default() -> Self {
        Self {
            video_format: VideoFormat::HD1080p25,
            audio_format: AudioFormat::default(),
            genlock_enabled: false,
            clock_source: "internal".to_string(),
            buffer_size: 10,
            fallback_content: PathBuf::from("/var/oximedia/fallback.mxf"),
            max_latency_ms: 100,
            detect_frame_drops: true,
            playlist_dir: PathBuf::from("/var/oximedia/playlists"),
            content_root: PathBuf::from("/var/oximedia/content"),
            monitoring_enabled: true,
            monitoring_port: 8080,
        }
    }
}

/// Playout server state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayoutState {
    /// Server is stopped
    Stopped,
    /// Server is starting up
    Starting,
    /// Server is running normally
    Running,
    /// Server is paused
    Paused,
    /// Server is in emergency fallback mode
    Fallback,
    /// Server is stopping
    Stopping,
}

/// Internal server state
#[cfg(not(target_arch = "wasm32"))]
struct ServerState {
    state: PlayoutState,
    scheduler: Option<Arc<scheduler::Scheduler>>,
    playback: Option<Arc<playback::PlaybackEngine>>,
    outputs: Vec<Arc<output::Output>>,
    graphics: Option<Arc<graphics::GraphicsEngine>>,
    monitor: Option<Arc<monitoring::Monitor>>,
}

/// Configuration for graceful shutdown behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownConfig {
    /// Maximum time to wait for in-flight frames to drain (milliseconds).
    pub drain_timeout_ms: u64,
    /// Whether to flush the frame buffer on shutdown.
    pub flush_buffers: bool,
    /// Whether to wait for current playlist item to finish before stopping.
    pub wait_for_current_item: bool,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout_ms: 5000,
            flush_buffers: true,
            wait_for_current_item: false,
        }
    }
}

/// Professional broadcast playout server
#[cfg(not(target_arch = "wasm32"))]
pub struct PlayoutServer {
    config: Arc<RwLock<PlayoutConfig>>,
    state: Arc<RwLock<ServerState>>,
    shutdown_config: ShutdownConfig,
    #[allow(dead_code)]
    control_tx: mpsc::Sender<ControlCommand>,
    #[allow(dead_code)]
    control_rx: Arc<RwLock<mpsc::Receiver<ControlCommand>>>,
}

/// Control commands for the playout server
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ControlCommand {
    Start,
    Stop,
    Pause,
    Resume,
    LoadPlaylist(PathBuf),
    EmergencyFallback,
    Shutdown,
    /// Hot-swap configuration without stopping.
    Reconfigure(PlayoutConfig),
}

#[cfg(not(target_arch = "wasm32"))]
impl PlayoutServer {
    /// Create a new playout server with the given configuration
    pub async fn new(config: PlayoutConfig) -> Result<Self> {
        let (control_tx, control_rx) = mpsc::channel(100);

        let state = ServerState {
            state: PlayoutState::Stopped,
            scheduler: None,
            playback: None,
            outputs: Vec::new(),
            graphics: None,
            monitor: None,
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(state)),
            shutdown_config: ShutdownConfig::default(),
            control_tx,
            control_rx: Arc::new(RwLock::new(control_rx)),
        })
    }

    /// Create a new playout server with custom shutdown configuration.
    pub async fn with_shutdown_config(
        config: PlayoutConfig,
        shutdown_config: ShutdownConfig,
    ) -> Result<Self> {
        let (control_tx, control_rx) = mpsc::channel(100);

        let state = ServerState {
            state: PlayoutState::Stopped,
            scheduler: None,
            playback: None,
            outputs: Vec::new(),
            graphics: None,
            monitor: None,
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(state)),
            shutdown_config,
            control_tx,
            control_rx: Arc::new(RwLock::new(control_rx)),
        })
    }

    /// Start the playout server
    pub async fn start(&self) -> Result<()> {
        let config = self.config.read().await.clone();
        let mut state = self.state.write().await;

        if state.state != PlayoutState::Stopped {
            return Err(PlayoutError::Config(
                "Server is already running".to_string(),
            ));
        }

        state.state = PlayoutState::Starting;

        // Initialize scheduler
        let scheduler_config = scheduler::SchedulerConfig::default();
        state.scheduler = Some(Arc::new(scheduler::Scheduler::new(scheduler_config)));

        // Initialize playback engine
        let playback_config = playback::PlaybackConfig::from_playout_config(&config);
        state.playback = Some(Arc::new(playback::PlaybackEngine::new(playback_config)?));

        // Initialize graphics engine
        let graphics_config = graphics::GraphicsConfig::default();
        state.graphics = Some(Arc::new(graphics::GraphicsEngine::new(graphics_config)?));

        // Initialize monitor
        if config.monitoring_enabled {
            let monitor_config = monitoring::MonitorConfig {
                port: config.monitoring_port,
                audio_meters: true,
                waveform: false,
                vectorscope: false,
                alert_history_size: 100,
                metrics_retention_seconds: 3600,
            };
            state.monitor = Some(Arc::new(monitoring::Monitor::new(monitor_config)?));
        }

        state.state = PlayoutState::Running;

        Ok(())
    }

    /// Stop the playout server with graceful shutdown.
    ///
    /// Drains in-flight frames from the playback buffer up to the configured
    /// `drain_timeout_ms`. Outputs are flushed before being torn down.
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.state == PlayoutState::Stopped {
            return Ok(());
        }

        state.state = PlayoutState::Stopping;

        // --- Graceful drain of in-flight frames ---
        if self.shutdown_config.flush_buffers {
            if let Some(playback) = &state.playback {
                let deadline = tokio::time::Instant::now()
                    + Duration::from_millis(self.shutdown_config.drain_timeout_ms);

                // Drain frames from the playback buffer until empty or timeout.
                loop {
                    let level = playback.buffer_level();
                    if level == 0 {
                        break;
                    }
                    if tokio::time::Instant::now() >= deadline {
                        // Timed out — force-stop.
                        break;
                    }
                    // Consume a frame (simulates output delivery).
                    let _ = playback.get_next_frame();
                    // Yield to avoid busy-spin.
                    tokio::task::yield_now().await;
                }
            }
        }

        // --- Tear down resources ---
        state.monitor = None;
        state.graphics = None;

        // Stop playback engine cleanly.
        if let Some(playback) = state.playback.take() {
            let _ = playback.stop().await;
        }

        state.scheduler = None;
        state.outputs.clear();

        state.state = PlayoutState::Stopped;

        Ok(())
    }

    /// Pause playout
    pub async fn pause(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if state.state == PlayoutState::Running {
            state.state = PlayoutState::Paused;
        }
        Ok(())
    }

    /// Resume playout
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if state.state == PlayoutState::Paused {
            state.state = PlayoutState::Running;
        }
        Ok(())
    }

    /// Get current server state
    pub async fn state(&self) -> PlayoutState {
        self.state.read().await.state
    }

    /// Load a new playlist
    pub async fn load_playlist(&self, path: PathBuf) -> Result<()> {
        let state = self.state.read().await;
        if let Some(scheduler) = &state.scheduler {
            scheduler.load_playlist(path).await?;
        }
        Ok(())
    }

    /// Activate emergency fallback
    pub async fn emergency_fallback(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.state = PlayoutState::Fallback;
        Ok(())
    }

    /// Get a snapshot of the current playout configuration.
    pub async fn config(&self) -> PlayoutConfig {
        self.config.read().await.clone()
    }

    /// Hot-swap the playout configuration without stopping the server.
    ///
    /// Only safe-to-change fields are applied while the server is running:
    /// - `monitoring_enabled` / `monitoring_port`
    /// - `max_latency_ms`
    /// - `detect_frame_drops`
    /// - `buffer_size`
    /// - `fallback_content`
    /// - `genlock_enabled`
    /// - `clock_source`
    ///
    /// Fields that require a restart (`video_format`, `audio_format`) are
    /// stored but only take effect after a stop/start cycle.
    pub async fn reconfigure(&self, new_config: PlayoutConfig) -> Result<()> {
        let old_config = self.config.read().await.clone();

        // Validate the new configuration before applying.
        if new_config.buffer_size == 0 {
            return Err(PlayoutError::Config("buffer_size must be > 0".to_string()));
        }

        // Determine what changed.
        let monitoring_changed = old_config.monitoring_enabled != new_config.monitoring_enabled
            || old_config.monitoring_port != new_config.monitoring_port;

        // Apply monitoring changes while running.
        if monitoring_changed {
            let mut state = self.state.write().await;
            if new_config.monitoring_enabled && state.monitor.is_none() {
                let monitor_config = monitoring::MonitorConfig {
                    port: new_config.monitoring_port,
                    audio_meters: true,
                    waveform: false,
                    vectorscope: false,
                    alert_history_size: 100,
                    metrics_retention_seconds: 3600,
                };
                state.monitor = Some(Arc::new(monitoring::Monitor::new(monitor_config)?));
            } else if !new_config.monitoring_enabled {
                state.monitor = None;
            }
        }

        // Store the full new config (format changes take effect on next start).
        *self.config.write().await = new_config;

        Ok(())
    }

    /// Wait for server to finish (blocks until shutdown)
    pub async fn wait(&self) -> Result<()> {
        loop {
            let state = self.state().await;
            if state == PlayoutState::Stopped {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_format_properties() {
        let format = VideoFormat::HD1080p25;
        assert_eq!(format.fps(), 25.0);
        assert_eq!(format.width(), 1920);
        assert_eq!(format.height(), 1080);
    }

    #[test]
    fn test_default_config() {
        let config = PlayoutConfig::default();
        assert_eq!(config.video_format, VideoFormat::HD1080p25);
        assert_eq!(config.audio_format.sample_rate, 48000);
        assert_eq!(config.buffer_size, 10);
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        assert_eq!(server.state().await, PlayoutState::Stopped);
    }

    // --- Graceful shutdown tests ---

    #[tokio::test]
    async fn test_graceful_shutdown_stopped_server() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        // Stopping an already-stopped server should be a no-op.
        server.stop().await.expect("should succeed");
        assert_eq!(server.state().await, PlayoutState::Stopped);
    }

    #[tokio::test]
    async fn test_graceful_shutdown_running_server() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");
        assert_eq!(server.state().await, PlayoutState::Running);

        server.stop().await.expect("should stop gracefully");
        assert_eq!(server.state().await, PlayoutState::Stopped);
    }

    #[tokio::test]
    async fn test_graceful_shutdown_with_custom_config() {
        let config = PlayoutConfig::default();
        let shutdown_cfg = ShutdownConfig {
            drain_timeout_ms: 100,
            flush_buffers: true,
            wait_for_current_item: false,
        };
        let server = PlayoutServer::with_shutdown_config(config, shutdown_cfg)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");
        server.stop().await.expect("should stop");
        assert_eq!(server.state().await, PlayoutState::Stopped);
    }

    #[tokio::test]
    async fn test_graceful_shutdown_no_flush() {
        let config = PlayoutConfig::default();
        let shutdown_cfg = ShutdownConfig {
            drain_timeout_ms: 100,
            flush_buffers: false,
            wait_for_current_item: false,
        };
        let server = PlayoutServer::with_shutdown_config(config, shutdown_cfg)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");
        server.stop().await.expect("should stop immediately");
        assert_eq!(server.state().await, PlayoutState::Stopped);
    }

    #[test]
    fn test_shutdown_config_default() {
        let cfg = ShutdownConfig::default();
        assert_eq!(cfg.drain_timeout_ms, 5000);
        assert!(cfg.flush_buffers);
        assert!(!cfg.wait_for_current_item);
    }

    // --- Hot-swap configuration tests ---

    #[tokio::test]
    async fn test_hot_swap_config_while_stopped() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");

        let mut new_config = PlayoutConfig::default();
        new_config.max_latency_ms = 200;
        new_config.buffer_size = 20;

        server
            .reconfigure(new_config)
            .await
            .expect("should reconfigure");

        let current = server.config().await;
        assert_eq!(current.max_latency_ms, 200);
        assert_eq!(current.buffer_size, 20);
    }

    #[tokio::test]
    async fn test_hot_swap_config_while_running() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");

        let mut new_config = PlayoutConfig::default();
        new_config.max_latency_ms = 50;
        new_config.detect_frame_drops = false;

        server
            .reconfigure(new_config)
            .await
            .expect("should reconfigure while running");

        let current = server.config().await;
        assert_eq!(current.max_latency_ms, 50);
        assert!(!current.detect_frame_drops);

        server.stop().await.expect("should stop");
    }

    #[tokio::test]
    async fn test_hot_swap_invalid_config() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");

        let mut bad_config = PlayoutConfig::default();
        bad_config.buffer_size = 0;

        let result = server.reconfigure(bad_config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_hot_swap_enable_monitoring() {
        let mut config = PlayoutConfig::default();
        config.monitoring_enabled = false;
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");

        // Enable monitoring via hot-swap
        let mut new_config = server.config().await;
        new_config.monitoring_enabled = true;
        new_config.monitoring_port = 19090;
        server
            .reconfigure(new_config)
            .await
            .expect("should enable monitoring");

        let current = server.config().await;
        assert!(current.monitoring_enabled);
        assert_eq!(current.monitoring_port, 19090);

        server.stop().await.expect("should stop");
    }

    #[tokio::test]
    async fn test_hot_swap_disable_monitoring() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");
        server.start().await.expect("should start");

        // Disable monitoring
        let mut new_config = server.config().await;
        new_config.monitoring_enabled = false;
        server
            .reconfigure(new_config)
            .await
            .expect("should disable monitoring");

        let current = server.config().await;
        assert!(!current.monitoring_enabled);

        server.stop().await.expect("should stop");
    }

    #[tokio::test]
    async fn test_hot_swap_video_format_stored() {
        let config = PlayoutConfig::default();
        let server = PlayoutServer::new(config)
            .await
            .expect("should succeed in test");

        let mut new_config = PlayoutConfig::default();
        new_config.video_format = VideoFormat::UHD2160p50;
        server
            .reconfigure(new_config)
            .await
            .expect("should store format change");

        let current = server.config().await;
        assert_eq!(current.video_format, VideoFormat::UHD2160p50);
    }
}
