//! Multi-platform simultaneous streaming.
//!
//! Broadcasts encoded media to multiple streaming platforms (Twitch, YouTube,
//! Facebook, etc.) concurrently, with independent per-platform configuration
//! and error handling so that a failure on one platform does not disrupt others.

use crate::{GamingError, GamingResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// StreamPlatform
// ---------------------------------------------------------------------------

/// Supported streaming platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPlatform {
    /// Twitch.tv
    Twitch,
    /// YouTube Gaming / YouTube Live
    YouTube,
    /// Facebook Gaming / Facebook Live
    Facebook,
    /// A custom / self-hosted RTMP target.
    Custom,
}

impl StreamPlatform {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Twitch => "Twitch",
            Self::YouTube => "YouTube",
            Self::Facebook => "Facebook",
            Self::Custom => "Custom",
        }
    }

    /// Default ingest URL template for the platform.
    #[must_use]
    pub fn default_ingest_url(self) -> &'static str {
        match self {
            Self::Twitch => "rtmp://live.twitch.tv/app",
            Self::YouTube => "rtmp://a.rtmp.youtube.com/live2",
            Self::Facebook => "rtmps://live-api-s.facebook.com:443/rtmp",
            Self::Custom => "",
        }
    }
}

// ---------------------------------------------------------------------------
// PlatformStreamConfig
// ---------------------------------------------------------------------------

/// Per-platform stream configuration.
#[derive(Debug, Clone)]
pub struct PlatformStreamConfig {
    /// The platform to stream to.
    pub platform: StreamPlatform,
    /// RTMP / RTMPS ingest URL.
    pub ingest_url: String,
    /// Stream key (secret).
    pub stream_key: String,
    /// Video bitrate in kbps for this platform.
    pub video_bitrate_kbps: u32,
    /// Audio bitrate in kbps for this platform.
    pub audio_bitrate_kbps: u32,
    /// Maximum resolution for this platform.
    pub max_resolution: (u32, u32),
    /// Whether to re-encode for this platform (or pass-through).
    pub re_encode: bool,
    /// Whether this platform target is enabled.
    pub enabled: bool,
}

impl PlatformStreamConfig {
    /// Create a new platform config with sensible defaults for the given platform.
    #[must_use]
    pub fn new(platform: StreamPlatform, stream_key: &str) -> Self {
        let (bitrate, audio_br, max_res) = match platform {
            StreamPlatform::Twitch => (6000, 160, (1920, 1080)),
            StreamPlatform::YouTube => (8000, 128, (2560, 1440)),
            StreamPlatform::Facebook => (4000, 128, (1920, 1080)),
            StreamPlatform::Custom => (6000, 128, (1920, 1080)),
        };

        Self {
            platform,
            ingest_url: platform.default_ingest_url().to_string(),
            stream_key: stream_key.to_string(),
            video_bitrate_kbps: bitrate,
            audio_bitrate_kbps: audio_br,
            max_resolution: max_res,
            re_encode: false,
            enabled: true,
        }
    }

    /// Validate this platform config.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.ingest_url.is_empty() {
            issues.push(format!("{}: ingest URL is empty", self.platform.label()));
        }
        if self.stream_key.is_empty() {
            issues.push(format!("{}: stream key is empty", self.platform.label()));
        }
        if self.video_bitrate_kbps < 500 {
            issues.push(format!(
                "{}: video bitrate must be >= 500 kbps",
                self.platform.label()
            ));
        }
        if self.max_resolution.0 == 0 || self.max_resolution.1 == 0 {
            issues.push(format!(
                "{}: resolution must be non-zero",
                self.platform.label()
            ));
        }
        issues
    }

    /// Whether the config passes validation.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

// ---------------------------------------------------------------------------
// PlatformStreamState
// ---------------------------------------------------------------------------

/// Runtime state of a single platform stream.
#[derive(Debug, Clone)]
pub struct PlatformStreamState {
    /// The platform.
    pub platform: StreamPlatform,
    /// Current connection status.
    pub status: PlatformStatus,
    /// Total frames sent to this platform.
    pub frames_sent: u64,
    /// Total bytes sent to this platform.
    pub bytes_sent: u64,
    /// Number of errors encountered.
    pub error_count: u32,
    /// Last error message, if any.
    pub last_error: Option<String>,
    /// When this platform stream was started.
    pub started_at: Option<Instant>,
    /// Number of reconnection attempts.
    pub reconnect_attempts: u32,
}

/// Connection status for a platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformStatus {
    /// Not connected.
    Disconnected,
    /// Connection in progress.
    Connecting,
    /// Connected and streaming.
    Live,
    /// Encountered an error (may auto-reconnect).
    Error,
    /// Deliberately stopped.
    Stopped,
}

impl PlatformStreamState {
    fn new(platform: StreamPlatform) -> Self {
        Self {
            platform,
            status: PlatformStatus::Disconnected,
            frames_sent: 0,
            bytes_sent: 0,
            error_count: 0,
            last_error: None,
            started_at: None,
            reconnect_attempts: 0,
        }
    }

    /// Uptime since the stream started, or zero if not started.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.started_at
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Whether this platform is currently live.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.status == PlatformStatus::Live
    }
}

// ---------------------------------------------------------------------------
// MultiStreamManager
// ---------------------------------------------------------------------------

/// Manages simultaneous streaming to multiple platforms.
///
/// Each platform target is independently configured, started, and tracked.
/// Errors on one platform do not affect the others.
pub struct MultiStreamManager {
    configs: HashMap<StreamPlatform, PlatformStreamConfig>,
    states: HashMap<StreamPlatform, PlatformStreamState>,
    /// Maximum reconnection attempts before giving up on a platform.
    pub max_reconnect_attempts: u32,
}

impl MultiStreamManager {
    /// Create a new multi-stream manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            states: HashMap::new(),
            max_reconnect_attempts: 5,
        }
    }

    /// Add a platform target configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the config is invalid.
    pub fn add_platform(&mut self, config: PlatformStreamConfig) -> GamingResult<()> {
        let issues = config.validate();
        if !issues.is_empty() {
            return Err(GamingError::PlatformError(issues.join("; ")));
        }
        let platform = config.platform;
        self.configs.insert(platform, config);
        self.states
            .insert(platform, PlatformStreamState::new(platform));
        Ok(())
    }

    /// Remove a platform target.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is currently live.
    pub fn remove_platform(&mut self, platform: StreamPlatform) -> GamingResult<()> {
        if let Some(state) = self.states.get(&platform) {
            if state.status == PlatformStatus::Live {
                return Err(GamingError::PlatformError(format!(
                    "Cannot remove {}: currently live. Stop first.",
                    platform.label()
                )));
            }
        }
        self.configs.remove(&platform);
        self.states.remove(&platform);
        Ok(())
    }

    /// Start streaming on a specific platform.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is not configured or already live.
    pub fn start_platform(&mut self, platform: StreamPlatform) -> GamingResult<()> {
        if !self.configs.contains_key(&platform) {
            return Err(GamingError::PlatformError(format!(
                "{} is not configured",
                platform.label()
            )));
        }

        let state = self
            .states
            .entry(platform)
            .or_insert_with(|| PlatformStreamState::new(platform));

        if state.status == PlatformStatus::Live {
            return Err(GamingError::PlatformError(format!(
                "{} is already live",
                platform.label()
            )));
        }

        state.status = PlatformStatus::Live;
        state.started_at = Some(Instant::now());
        state.error_count = 0;
        state.last_error = None;
        state.reconnect_attempts = 0;
        Ok(())
    }

    /// Stop streaming on a specific platform.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is not configured.
    pub fn stop_platform(&mut self, platform: StreamPlatform) -> GamingResult<()> {
        let state = self.states.get_mut(&platform).ok_or_else(|| {
            GamingError::PlatformError(format!("{} is not configured", platform.label()))
        })?;

        state.status = PlatformStatus::Stopped;
        Ok(())
    }

    /// Start all configured and enabled platforms.
    ///
    /// Returns a list of (platform, error) pairs for platforms that failed to start.
    pub fn start_all(&mut self) -> Vec<(StreamPlatform, GamingError)> {
        let platforms: Vec<StreamPlatform> = self
            .configs
            .iter()
            .filter(|(_, c)| c.enabled)
            .map(|(p, _)| *p)
            .collect();

        let mut errors = Vec::new();
        for p in platforms {
            if let Err(e) = self.start_platform(p) {
                errors.push((p, e));
            }
        }
        errors
    }

    /// Stop all platforms.
    pub fn stop_all(&mut self) {
        for state in self.states.values_mut() {
            state.status = PlatformStatus::Stopped;
        }
    }

    /// Broadcast a frame to all live platforms.
    ///
    /// Returns per-platform byte counts. In a real implementation, this would
    /// push the encoded data to each platform's RTMP connection.
    pub fn broadcast_frame(&mut self, data: &[u8]) -> HashMap<StreamPlatform, u64> {
        let mut sent = HashMap::new();
        let byte_count = data.len() as u64;

        for (platform, state) in &mut self.states {
            if state.status == PlatformStatus::Live {
                state.frames_sent += 1;
                state.bytes_sent += byte_count;
                sent.insert(*platform, byte_count);
            }
        }

        sent
    }

    /// Record an error on a specific platform.
    ///
    /// If the number of reconnection attempts exceeds `max_reconnect_attempts`,
    /// the platform is moved to `Stopped` status.
    pub fn record_error(&mut self, platform: StreamPlatform, error_msg: &str) {
        if let Some(state) = self.states.get_mut(&platform) {
            state.error_count += 1;
            state.last_error = Some(error_msg.to_string());
            state.reconnect_attempts += 1;

            if state.reconnect_attempts > self.max_reconnect_attempts {
                state.status = PlatformStatus::Stopped;
            } else {
                state.status = PlatformStatus::Error;
            }
        }
    }

    /// Get the state of a specific platform.
    #[must_use]
    pub fn platform_state(&self, platform: StreamPlatform) -> Option<&PlatformStreamState> {
        self.states.get(&platform)
    }

    /// Get all platform states.
    #[must_use]
    pub fn all_states(&self) -> Vec<&PlatformStreamState> {
        self.states.values().collect()
    }

    /// Number of platforms currently live.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.states.values().filter(|s| s.is_live()).count()
    }

    /// Number of configured platforms.
    #[must_use]
    pub fn platform_count(&self) -> usize {
        self.configs.len()
    }

    /// Whether any platform is currently live.
    #[must_use]
    pub fn is_any_live(&self) -> bool {
        self.states.values().any(|s| s.is_live())
    }

    /// Get the config for a platform.
    #[must_use]
    pub fn platform_config(&self, platform: StreamPlatform) -> Option<&PlatformStreamConfig> {
        self.configs.get(&platform)
    }

    /// Get a mutable reference to a platform config for live updates.
    pub fn platform_config_mut(
        &mut self,
        platform: StreamPlatform,
    ) -> Option<&mut PlatformStreamConfig> {
        self.configs.get_mut(&platform)
    }

    /// Attempt to reconnect a platform that is in `Error` status.
    ///
    /// Uses exponential backoff: the delay doubles with each attempt
    /// (`base_delay * 2^(attempt-1)`), capped at `max_delay`.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is not in `Error` status or not configured.
    pub fn attempt_reconnect(&mut self, platform: StreamPlatform) -> GamingResult<ReconnectResult> {
        // Check state without holding mutable borrow
        {
            let state = self.states.get(&platform).ok_or_else(|| {
                GamingError::PlatformError(format!("{} is not configured", platform.label()))
            })?;

            if state.status != PlatformStatus::Error {
                return Err(GamingError::PlatformError(format!(
                    "{} is not in error state (current: {:?})",
                    platform.label(),
                    state.status
                )));
            }

            if state.reconnect_attempts >= self.max_reconnect_attempts {
                // Need mutable access below
            }
        }

        let max_attempts = self.max_reconnect_attempts;
        let state = self.states.get_mut(&platform).ok_or_else(|| {
            GamingError::PlatformError(format!("{} is not configured", platform.label()))
        })?;

        if state.reconnect_attempts >= max_attempts {
            state.status = PlatformStatus::Stopped;
            return Ok(ReconnectResult {
                platform,
                attempt: state.reconnect_attempts,
                backoff: Duration::ZERO,
                gave_up: true,
            });
        }

        let attempt = state.reconnect_attempts;
        let base_ms: u64 = 1000;
        let max_ms: u64 = 30_000;
        let delay_ms = base_ms.saturating_mul(1u64 << attempt.min(15)).min(max_ms);
        let backoff = Duration::from_millis(delay_ms);

        // Simulate successful reconnection
        state.status = PlatformStatus::Live;
        state.last_error = None;

        Ok(ReconnectResult {
            platform,
            attempt,
            backoff,
            gave_up: false,
        })
    }

    /// Get stream health metrics for a specific platform.
    #[must_use]
    pub fn stream_health(&self, platform: StreamPlatform) -> Option<StreamHealthMetrics> {
        let state = self.states.get(&platform)?;
        let config = self.configs.get(&platform)?;

        let uptime = state.uptime();
        let uptime_secs = uptime.as_secs_f64();

        let avg_bitrate_kbps = if uptime_secs > 0.0 {
            ((state.bytes_sent as f64 * 8.0) / (uptime_secs * 1000.0)) as u32
        } else {
            0
        };

        let fps = if uptime_secs > 0.0 {
            state.frames_sent as f64 / uptime_secs
        } else {
            0.0
        };

        let bitrate_stability = if config.video_bitrate_kbps > 0 && avg_bitrate_kbps > 0 {
            let ratio = avg_bitrate_kbps as f64 / config.video_bitrate_kbps as f64;
            ratio.min(1.0)
        } else {
            0.0
        };

        Some(StreamHealthMetrics {
            platform,
            is_live: state.is_live(),
            uptime,
            frames_sent: state.frames_sent,
            bytes_sent: state.bytes_sent,
            avg_bitrate_kbps,
            target_bitrate_kbps: config.video_bitrate_kbps,
            bitrate_stability,
            effective_fps: fps,
            error_count: state.error_count,
            reconnect_attempts: state.reconnect_attempts,
            health_score: Self::compute_health_score(state, bitrate_stability),
        })
    }

    /// Compute a 0.0-1.0 health score for a platform.
    fn compute_health_score(state: &PlatformStreamState, bitrate_stability: f64) -> f64 {
        if !state.is_live() {
            return 0.0;
        }

        let mut score = 1.0;

        // Deduct for errors
        let error_penalty = (state.error_count as f64) * 0.1;
        score -= error_penalty.min(0.5);

        // Factor in bitrate stability
        score *= bitrate_stability.max(0.1);

        // Deduct for reconnections
        let reconnect_penalty = (state.reconnect_attempts as f64) * 0.05;
        score -= reconnect_penalty.min(0.3);

        score.clamp(0.0, 1.0)
    }

    /// Get health metrics for all configured platforms.
    #[must_use]
    pub fn all_health_metrics(&self) -> Vec<StreamHealthMetrics> {
        self.configs
            .keys()
            .filter_map(|p| self.stream_health(*p))
            .collect()
    }

    /// Summary of all platform statuses.
    #[must_use]
    pub fn status_summary(&self) -> HashMap<StreamPlatform, PlatformStatus> {
        self.states.iter().map(|(p, s)| (*p, s.status)).collect()
    }

    /// Update the quality settings for a live platform.
    ///
    /// Changes take effect on next frame. Does not interrupt the stream.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is not configured.
    pub fn update_quality(
        &mut self,
        platform: StreamPlatform,
        video_bitrate_kbps: Option<u32>,
        audio_bitrate_kbps: Option<u32>,
        max_resolution: Option<(u32, u32)>,
    ) -> GamingResult<()> {
        let config = self.configs.get_mut(&platform).ok_or_else(|| {
            GamingError::PlatformError(format!("{} is not configured", platform.label()))
        })?;

        if let Some(vbr) = video_bitrate_kbps {
            if vbr < 500 {
                return Err(GamingError::InvalidConfig(
                    "Video bitrate must be >= 500 kbps".into(),
                ));
            }
            config.video_bitrate_kbps = vbr;
        }
        if let Some(abr) = audio_bitrate_kbps {
            config.audio_bitrate_kbps = abr;
        }
        if let Some(res) = max_resolution {
            if res.0 == 0 || res.1 == 0 {
                return Err(GamingError::InvalidConfig(
                    "Resolution must be non-zero".into(),
                ));
            }
            config.max_resolution = res;
        }
        Ok(())
    }

    /// Broadcast a frame to a specific platform only.
    ///
    /// Returns bytes sent, or 0 if the platform is not live.
    pub fn send_frame_to(&mut self, platform: StreamPlatform, data: &[u8]) -> u64 {
        if let Some(state) = self.states.get_mut(&platform) {
            if state.status == PlatformStatus::Live {
                let byte_count = data.len() as u64;
                state.frames_sent += 1;
                state.bytes_sent += byte_count;
                return byte_count;
            }
        }
        0
    }

    /// Reset error state for a platform, moving it back to `Disconnected`.
    ///
    /// # Errors
    ///
    /// Returns error if the platform is not configured.
    pub fn reset_error(&mut self, platform: StreamPlatform) -> GamingResult<()> {
        let state = self.states.get_mut(&platform).ok_or_else(|| {
            GamingError::PlatformError(format!("{} is not configured", platform.label()))
        })?;
        state.status = PlatformStatus::Disconnected;
        state.error_count = 0;
        state.reconnect_attempts = 0;
        state.last_error = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ReconnectResult
// ---------------------------------------------------------------------------

/// Result of a reconnection attempt.
#[derive(Debug)]
pub struct ReconnectResult {
    /// The platform that was reconnected.
    pub platform: StreamPlatform,
    /// Which attempt number this was.
    pub attempt: u32,
    /// The backoff duration that should be waited.
    pub backoff: Duration,
    /// Whether the platform gave up after too many attempts.
    pub gave_up: bool,
}

// ---------------------------------------------------------------------------
// StreamHealthMetrics
// ---------------------------------------------------------------------------

/// Health metrics for a single platform stream.
#[derive(Debug, Clone)]
pub struct StreamHealthMetrics {
    /// The platform.
    pub platform: StreamPlatform,
    /// Whether the platform is currently live.
    pub is_live: bool,
    /// How long the stream has been running.
    pub uptime: Duration,
    /// Total frames sent.
    pub frames_sent: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Average bitrate in kbps.
    pub avg_bitrate_kbps: u32,
    /// Target bitrate from config.
    pub target_bitrate_kbps: u32,
    /// Stability of bitrate (0.0 - 1.0).
    pub bitrate_stability: f64,
    /// Effective frames per second.
    pub effective_fps: f64,
    /// Total error count.
    pub error_count: u32,
    /// Total reconnection attempts.
    pub reconnect_attempts: u32,
    /// Overall health score (0.0 - 1.0).
    pub health_score: f64,
}

// ---------------------------------------------------------------------------
// ChatOverlay
// ---------------------------------------------------------------------------

/// Configuration for per-platform chat overlay integration.
#[derive(Debug, Clone)]
pub struct ChatOverlay {
    /// Which platform's chat to display.
    pub platform: StreamPlatform,
    /// Whether the overlay is visible.
    pub visible: bool,
    /// Position on screen (x, y) in pixels from top-left.
    pub position: (u32, u32),
    /// Size of the overlay (width, height) in pixels.
    pub size: (u32, u32),
    /// Opacity (0.0 fully transparent, 1.0 fully opaque).
    pub opacity: f32,
    /// Maximum number of chat messages to show.
    pub max_messages: usize,
    /// Font size in points.
    pub font_size: u32,
    /// Background color as RGBA.
    pub background_rgba: [u8; 4],
}

impl ChatOverlay {
    /// Create a new chat overlay with default settings.
    #[must_use]
    pub fn new(platform: StreamPlatform) -> Self {
        Self {
            platform,
            visible: true,
            position: (20, 400),
            size: (360, 500),
            opacity: 0.8,
            max_messages: 15,
            font_size: 14,
            background_rgba: [0, 0, 0, 180],
        }
    }

    /// Area of the overlay in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.size.0) * u64::from(self.size.1)
    }

    /// Set opacity, clamped to [0.0, 1.0].
    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity.clamp(0.0, 1.0);
    }
}

impl Default for MultiStreamManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn twitch_config() -> PlatformStreamConfig {
        PlatformStreamConfig::new(StreamPlatform::Twitch, "live_abc123")
    }

    fn youtube_config() -> PlatformStreamConfig {
        PlatformStreamConfig::new(StreamPlatform::YouTube, "yt_key_456")
    }

    fn facebook_config() -> PlatformStreamConfig {
        PlatformStreamConfig::new(StreamPlatform::Facebook, "fb_key_789")
    }

    // -- StreamPlatform --

    #[test]
    fn test_platform_labels() {
        assert_eq!(StreamPlatform::Twitch.label(), "Twitch");
        assert_eq!(StreamPlatform::YouTube.label(), "YouTube");
        assert_eq!(StreamPlatform::Facebook.label(), "Facebook");
        assert_eq!(StreamPlatform::Custom.label(), "Custom");
    }

    #[test]
    fn test_platform_default_ingest_urls() {
        assert!(!StreamPlatform::Twitch.default_ingest_url().is_empty());
        assert!(!StreamPlatform::YouTube.default_ingest_url().is_empty());
        assert!(!StreamPlatform::Facebook.default_ingest_url().is_empty());
        assert!(StreamPlatform::Custom.default_ingest_url().is_empty());
    }

    // -- PlatformStreamConfig --

    #[test]
    fn test_config_defaults_per_platform() {
        let tc = twitch_config();
        assert_eq!(tc.video_bitrate_kbps, 6000);
        assert_eq!(tc.max_resolution, (1920, 1080));
        assert!(tc.enabled);

        let yc = youtube_config();
        assert_eq!(yc.video_bitrate_kbps, 8000);
        assert_eq!(yc.max_resolution, (2560, 1440));
    }

    #[test]
    fn test_config_validation_valid() {
        assert!(twitch_config().is_valid());
        assert!(youtube_config().is_valid());
        assert!(facebook_config().is_valid());
    }

    #[test]
    fn test_config_validation_empty_url() {
        let mut cfg = twitch_config();
        cfg.ingest_url = String::new();
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_validation_empty_key() {
        let mut cfg = twitch_config();
        cfg.stream_key = String::new();
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_validation_low_bitrate() {
        let mut cfg = twitch_config();
        cfg.video_bitrate_kbps = 100;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_validation_zero_resolution() {
        let mut cfg = twitch_config();
        cfg.max_resolution = (0, 1080);
        assert!(!cfg.is_valid());
    }

    // -- MultiStreamManager --

    #[test]
    fn test_manager_creation() {
        let mgr = MultiStreamManager::new();
        assert_eq!(mgr.platform_count(), 0);
        assert_eq!(mgr.live_count(), 0);
        assert!(!mgr.is_any_live());
    }

    #[test]
    fn test_add_platform() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add twitch");
        assert_eq!(mgr.platform_count(), 1);
    }

    #[test]
    fn test_add_invalid_platform() {
        let mut mgr = MultiStreamManager::new();
        let mut cfg = twitch_config();
        cfg.stream_key = String::new();
        assert!(mgr.add_platform(cfg).is_err());
    }

    #[test]
    fn test_start_platform() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        assert_eq!(mgr.live_count(), 1);
        assert!(mgr.is_any_live());
    }

    #[test]
    fn test_start_unconfigured_platform() {
        let mut mgr = MultiStreamManager::new();
        assert!(mgr.start_platform(StreamPlatform::Twitch).is_err());
    }

    #[test]
    fn test_start_already_live() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        assert!(mgr.start_platform(StreamPlatform::Twitch).is_err());
    }

    #[test]
    fn test_stop_platform() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        mgr.stop_platform(StreamPlatform::Twitch).expect("stop");
        assert_eq!(mgr.live_count(), 0);
    }

    #[test]
    fn test_stop_unconfigured() {
        let mut mgr = MultiStreamManager::new();
        assert!(mgr.stop_platform(StreamPlatform::Twitch).is_err());
    }

    #[test]
    fn test_remove_platform() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.remove_platform(StreamPlatform::Twitch).expect("remove");
        assert_eq!(mgr.platform_count(), 0);
    }

    #[test]
    fn test_remove_live_platform_fails() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        assert!(mgr.remove_platform(StreamPlatform::Twitch).is_err());
    }

    #[test]
    fn test_start_all() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add twitch");
        mgr.add_platform(youtube_config()).expect("add youtube");
        mgr.add_platform(facebook_config()).expect("add facebook");

        let errors = mgr.start_all();
        assert!(errors.is_empty());
        assert_eq!(mgr.live_count(), 3);
    }

    #[test]
    fn test_start_all_skips_disabled() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        let mut yt = youtube_config();
        yt.enabled = false;
        mgr.add_platform(yt).expect("add");

        let errors = mgr.start_all();
        assert!(errors.is_empty());
        assert_eq!(mgr.live_count(), 1);
    }

    #[test]
    fn test_stop_all() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_all();
        mgr.stop_all();
        assert_eq!(mgr.live_count(), 0);
    }

    #[test]
    fn test_broadcast_frame() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_all();

        let data = vec![0u8; 5000];
        let sent = mgr.broadcast_frame(&data);
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[&StreamPlatform::Twitch], 5000);
        assert_eq!(sent[&StreamPlatform::YouTube], 5000);

        let twitch_state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(twitch_state.frames_sent, 1);
        assert_eq!(twitch_state.bytes_sent, 5000);
    }

    #[test]
    fn test_broadcast_frame_only_live() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        // YouTube not started

        let sent = mgr.broadcast_frame(&[0u8; 100]);
        assert_eq!(sent.len(), 1);
        assert!(sent.contains_key(&StreamPlatform::Twitch));
    }

    #[test]
    fn test_record_error() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        mgr.record_error(StreamPlatform::Twitch, "connection reset");
        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.error_count, 1);
        assert_eq!(state.status, PlatformStatus::Error);
        assert_eq!(state.last_error.as_deref(), Some("connection reset"));
    }

    #[test]
    fn test_record_error_exceeds_max_reconnects() {
        let mut mgr = MultiStreamManager::new();
        mgr.max_reconnect_attempts = 2;
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        mgr.record_error(StreamPlatform::Twitch, "err1");
        mgr.record_error(StreamPlatform::Twitch, "err2");
        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.status, PlatformStatus::Error);

        mgr.record_error(StreamPlatform::Twitch, "err3");
        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.status, PlatformStatus::Stopped);
    }

    #[test]
    fn test_platform_state_uptime() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.uptime(), Duration::ZERO);
        assert!(!state.is_live());
    }

    #[test]
    fn test_all_states() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        assert_eq!(mgr.all_states().len(), 2);
    }

    #[test]
    fn test_platform_config_accessor() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        let cfg = mgr.platform_config(StreamPlatform::Twitch).expect("config");
        assert_eq!(cfg.platform, StreamPlatform::Twitch);
        assert!(mgr.platform_config(StreamPlatform::YouTube).is_none());
    }

    #[test]
    fn test_default_impl() {
        let mgr = MultiStreamManager::default();
        assert_eq!(mgr.platform_count(), 0);
    }

    #[test]
    fn test_multiple_broadcasts() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        for _ in 0..100 {
            mgr.broadcast_frame(&[0u8; 1000]);
        }

        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.frames_sent, 100);
        assert_eq!(state.bytes_sent, 100_000);
    }

    #[test]
    fn test_independent_error_handling() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_all();

        // Error on Twitch should not affect YouTube
        mgr.record_error(StreamPlatform::Twitch, "connection lost");

        let twitch = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(twitch.status, PlatformStatus::Error);

        let yt = mgr.platform_state(StreamPlatform::YouTube).expect("state");
        assert_eq!(yt.status, PlatformStatus::Live);
    }

    // -- reconnect --

    #[test]
    fn test_attempt_reconnect_success() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        mgr.record_error(StreamPlatform::Twitch, "network error");

        let result = mgr
            .attempt_reconnect(StreamPlatform::Twitch)
            .expect("reconnect");
        assert!(!result.gave_up);
        assert_eq!(result.platform, StreamPlatform::Twitch);

        let state = mgr.platform_state(StreamPlatform::Twitch).expect("state");
        assert_eq!(state.status, PlatformStatus::Live);
    }

    #[test]
    fn test_attempt_reconnect_not_in_error() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        assert!(mgr.attempt_reconnect(StreamPlatform::Twitch).is_err());
    }

    #[test]
    fn test_attempt_reconnect_gives_up() {
        let mut mgr = MultiStreamManager::new();
        mgr.max_reconnect_attempts = 1;
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        // Record one error (reconnect_attempts = 1, equals max)
        mgr.record_error(StreamPlatform::Twitch, "err");
        // Now attempt >= max, so it gives up
        let result = mgr
            .attempt_reconnect(StreamPlatform::Twitch)
            .expect("reconnect");
        assert!(result.gave_up);
    }

    #[test]
    fn test_attempt_reconnect_unconfigured() {
        let mut mgr = MultiStreamManager::new();
        assert!(mgr.attempt_reconnect(StreamPlatform::Twitch).is_err());
    }

    // -- stream_health --

    #[test]
    fn test_stream_health_not_live() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        let health = mgr.stream_health(StreamPlatform::Twitch).expect("health");
        assert!(!health.is_live);
        assert!((health.health_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stream_health_live_with_frames() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        for _ in 0..100 {
            mgr.broadcast_frame(&[0u8; 1000]);
        }

        let health = mgr.stream_health(StreamPlatform::Twitch).expect("health");
        assert!(health.is_live);
        assert_eq!(health.frames_sent, 100);
        assert_eq!(health.bytes_sent, 100_000);
        assert!(health.health_score > 0.0);
    }

    #[test]
    fn test_stream_health_degraded_by_errors() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        mgr.broadcast_frame(&[0u8; 1000]);

        let healthy = mgr.stream_health(StreamPlatform::Twitch).expect("h");
        let healthy_score = healthy.health_score;

        mgr.record_error(StreamPlatform::Twitch, "glitch");
        // Re-connect to make it live again
        let _ = mgr.attempt_reconnect(StreamPlatform::Twitch);
        mgr.broadcast_frame(&[0u8; 1000]);

        let degraded = mgr.stream_health(StreamPlatform::Twitch).expect("h");
        assert!(degraded.health_score <= healthy_score);
    }

    #[test]
    fn test_all_health_metrics() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        let metrics = mgr.all_health_metrics();
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_stream_health_unconfigured() {
        let mgr = MultiStreamManager::new();
        assert!(mgr.stream_health(StreamPlatform::Twitch).is_none());
    }

    // -- update_quality --

    #[test]
    fn test_update_quality_bitrate() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.update_quality(StreamPlatform::Twitch, Some(3000), None, None)
            .expect("update");
        let cfg = mgr.platform_config(StreamPlatform::Twitch).expect("cfg");
        assert_eq!(cfg.video_bitrate_kbps, 3000);
    }

    #[test]
    fn test_update_quality_resolution() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.update_quality(StreamPlatform::Twitch, None, None, Some((1280, 720)))
            .expect("update");
        let cfg = mgr.platform_config(StreamPlatform::Twitch).expect("cfg");
        assert_eq!(cfg.max_resolution, (1280, 720));
    }

    #[test]
    fn test_update_quality_invalid_bitrate() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        assert!(mgr
            .update_quality(StreamPlatform::Twitch, Some(100), None, None)
            .is_err());
    }

    #[test]
    fn test_update_quality_invalid_resolution() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        assert!(mgr
            .update_quality(StreamPlatform::Twitch, None, None, Some((0, 720)))
            .is_err());
    }

    #[test]
    fn test_update_quality_unconfigured() {
        let mut mgr = MultiStreamManager::new();
        assert!(mgr
            .update_quality(StreamPlatform::Twitch, Some(3000), None, None)
            .is_err());
    }

    // -- send_frame_to --

    #[test]
    fn test_send_frame_to_single_platform() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_all();

        let sent = mgr.send_frame_to(StreamPlatform::Twitch, &[0u8; 2000]);
        assert_eq!(sent, 2000);

        // Only Twitch should have the frame
        let tw = mgr.platform_state(StreamPlatform::Twitch).expect("s");
        assert_eq!(tw.frames_sent, 1);
        let yt = mgr.platform_state(StreamPlatform::YouTube).expect("s");
        assert_eq!(yt.frames_sent, 0);
    }

    #[test]
    fn test_send_frame_to_not_live() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        let sent = mgr.send_frame_to(StreamPlatform::Twitch, &[0u8; 100]);
        assert_eq!(sent, 0);
    }

    // -- reset_error --

    #[test]
    fn test_reset_error() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");
        mgr.record_error(StreamPlatform::Twitch, "fail");
        mgr.reset_error(StreamPlatform::Twitch).expect("reset");

        let state = mgr.platform_state(StreamPlatform::Twitch).expect("s");
        assert_eq!(state.status, PlatformStatus::Disconnected);
        assert_eq!(state.error_count, 0);
        assert_eq!(state.reconnect_attempts, 0);
        assert!(state.last_error.is_none());
    }

    // -- status_summary --

    #[test]
    fn test_status_summary() {
        let mut mgr = MultiStreamManager::new();
        mgr.add_platform(twitch_config()).expect("add");
        mgr.add_platform(youtube_config()).expect("add");
        mgr.start_platform(StreamPlatform::Twitch).expect("start");

        let summary = mgr.status_summary();
        assert_eq!(summary[&StreamPlatform::Twitch], PlatformStatus::Live);
        assert_eq!(
            summary[&StreamPlatform::YouTube],
            PlatformStatus::Disconnected
        );
    }

    // -- ChatOverlay --

    #[test]
    fn test_chat_overlay_defaults() {
        let overlay = ChatOverlay::new(StreamPlatform::Twitch);
        assert_eq!(overlay.platform, StreamPlatform::Twitch);
        assert!(overlay.visible);
        assert_eq!(overlay.max_messages, 15);
        assert!(overlay.area() > 0);
    }

    #[test]
    fn test_chat_overlay_set_opacity() {
        let mut overlay = ChatOverlay::new(StreamPlatform::YouTube);
        overlay.set_opacity(1.5);
        assert!((overlay.opacity - 1.0).abs() < f32::EPSILON);
        overlay.set_opacity(-0.5);
        assert!((overlay.opacity - 0.0).abs() < f32::EPSILON);
        overlay.set_opacity(0.5);
        assert!((overlay.opacity - 0.5).abs() < f32::EPSILON);
    }
}
