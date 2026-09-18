//! Stream quality configuration and management.
//!
//! Provides [`StreamQuality`] presets, the [`StreamingConfig`] parameter set,
//! and [`StreamConfigManager`] for storing and retrieving named configurations.

#![allow(dead_code)]

use std::collections::HashMap;

/// Predefined quality tiers for a game stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamQuality {
    /// 480p at 30 fps — suitable for low-bandwidth connections.
    Low,
    /// 720p at 30 fps — standard definition for most viewers.
    Medium,
    /// 720p at 60 fps — smooth motion for competitive games.
    High,
    /// 1080p at 60 fps — recommended for general streaming.
    Ultra,
    /// 1440p at 60 fps — for high-end hardware and fast internet.
    Source,
}

impl StreamQuality {
    /// Recommended video bitrate in kbps for each quality tier.
    #[must_use]
    pub fn bitrate_kbps(self) -> u32 {
        match self {
            Self::Low => 2_500,
            Self::Medium => 4_000,
            Self::High => 6_000,
            Self::Ultra => 8_000,
            Self::Source => 12_000,
        }
    }

    /// Output resolution `(width, height)` in pixels.
    #[must_use]
    pub fn resolution(self) -> (u32, u32) {
        match self {
            Self::Low => (854, 480),
            Self::Medium => (1280, 720),
            Self::High => (1280, 720),
            Self::Ultra => (1920, 1080),
            Self::Source => (2560, 1440),
        }
    }

    /// Target frames per second.
    #[must_use]
    pub fn fps(self) -> u32 {
        match self {
            Self::Low | Self::Medium => 30,
            Self::High | Self::Ultra | Self::Source => 60,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low (480p30)",
            Self::Medium => "Medium (720p30)",
            Self::High => "High (720p60)",
            Self::Ultra => "Ultra (1080p60)",
            Self::Source => "Source (1440p60)",
        }
    }
}

/// Full set of streaming parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingConfig {
    /// Quality preset applied when this config was created.
    pub quality: StreamQuality,
    /// Resolved video bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Resolved output width in pixels.
    pub width: u32,
    /// Resolved output height in pixels.
    pub height: u32,
    /// Resolved frames per second.
    pub fps: u32,
    /// Audio bitrate in kbps.
    pub audio_bitrate_kbps: u32,
    /// RTMP ingest URL (empty = not configured).
    pub ingest_url: String,
    /// Optional stream title.
    pub title: Option<String>,
    /// Whether low-latency mode is requested.
    pub low_latency: bool,
}

impl StreamingConfig {
    /// Construct a `StreamingConfig` from a quality preset.
    #[must_use]
    pub fn from_quality(quality: StreamQuality) -> Self {
        let (width, height) = quality.resolution();
        Self {
            quality,
            bitrate_kbps: quality.bitrate_kbps(),
            width,
            height,
            fps: quality.fps(),
            audio_bitrate_kbps: 160,
            ingest_url: String::new(),
            title: None,
            low_latency: false,
        }
    }

    /// Override the bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Set the RTMP ingest URL.
    #[must_use]
    pub fn with_ingest_url(mut self, url: impl Into<String>) -> Self {
        self.ingest_url = url.into();
        self
    }

    /// Set a stream title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Enable or disable low-latency mode.
    #[must_use]
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Total bitrate (video + audio) in kbps.
    #[must_use]
    pub fn total_bitrate_kbps(&self) -> u32 {
        self.bitrate_kbps.saturating_add(self.audio_bitrate_kbps)
    }

    /// Whether the config has a valid ingest URL configured.
    #[must_use]
    pub fn is_ready_to_stream(&self) -> bool {
        !self.ingest_url.is_empty()
    }
}

/// Manages named [`StreamingConfig`] profiles.
///
/// Allows users to save, load, update, and delete named stream configurations
/// so they can quickly switch between different streaming setups.
pub struct StreamConfigManager {
    configs: HashMap<String, StreamingConfig>,
    /// Name of the currently active profile (if any).
    active: Option<String>,
}

impl StreamConfigManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            active: None,
        }
    }

    /// Save a named configuration, replacing any existing entry with that name.
    pub fn save(&mut self, name: impl Into<String>, config: StreamingConfig) {
        self.configs.insert(name.into(), config);
    }

    /// Retrieve a reference to a named configuration.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StreamingConfig> {
        self.configs.get(name)
    }

    /// Remove a named configuration and return it, if it existed.
    pub fn remove(&mut self, name: &str) -> Option<StreamingConfig> {
        let removed = self.configs.remove(name);
        if self.active.as_deref() == Some(name) {
            self.active = None;
        }
        removed
    }

    /// Set the active profile by name.  Returns `false` if the name is unknown.
    pub fn set_active(&mut self, name: &str) -> bool {
        if self.configs.contains_key(name) {
            self.active = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// Reference to the currently active configuration.
    #[must_use]
    pub fn active_config(&self) -> Option<&StreamingConfig> {
        self.active.as_deref().and_then(|n| self.configs.get(n))
    }

    /// Name of the active profile.
    #[must_use]
    pub fn active_name(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Number of stored profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether no profiles are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Names of all stored profiles.
    #[must_use]
    pub fn profile_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.configs.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

impl Default for StreamConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- StreamQuality ---

    #[test]
    fn test_quality_bitrates() {
        assert_eq!(StreamQuality::Low.bitrate_kbps(), 2_500);
        assert_eq!(StreamQuality::Ultra.bitrate_kbps(), 8_000);
        assert_eq!(StreamQuality::Source.bitrate_kbps(), 12_000);
    }

    #[test]
    fn test_quality_resolutions() {
        assert_eq!(StreamQuality::Low.resolution(), (854, 480));
        assert_eq!(StreamQuality::Ultra.resolution(), (1920, 1080));
        assert_eq!(StreamQuality::Source.resolution(), (2560, 1440));
    }

    #[test]
    fn test_quality_fps() {
        assert_eq!(StreamQuality::Low.fps(), 30);
        assert_eq!(StreamQuality::Medium.fps(), 30);
        assert_eq!(StreamQuality::High.fps(), 60);
        assert_eq!(StreamQuality::Ultra.fps(), 60);
        assert_eq!(StreamQuality::Source.fps(), 60);
    }

    #[test]
    fn test_quality_labels_non_empty() {
        for q in [
            StreamQuality::Low,
            StreamQuality::Medium,
            StreamQuality::High,
            StreamQuality::Ultra,
            StreamQuality::Source,
        ] {
            assert!(!q.label().is_empty());
        }
    }

    // --- StreamingConfig ---

    #[test]
    fn test_from_quality_populates_fields() {
        let cfg = StreamingConfig::from_quality(StreamQuality::Ultra);
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.fps, 60);
        assert_eq!(cfg.bitrate_kbps, 8_000);
    }

    #[test]
    fn test_with_bitrate_override() {
        let cfg = StreamingConfig::from_quality(StreamQuality::High).with_bitrate(5_500);
        assert_eq!(cfg.bitrate_kbps, 5_500);
    }

    #[test]
    fn test_total_bitrate() {
        let cfg = StreamingConfig::from_quality(StreamQuality::Medium);
        assert_eq!(cfg.total_bitrate_kbps(), 4_000 + 160);
    }

    #[test]
    fn test_is_ready_to_stream_false_without_url() {
        let cfg = StreamingConfig::from_quality(StreamQuality::Low);
        assert!(!cfg.is_ready_to_stream());
    }

    #[test]
    fn test_is_ready_to_stream_true_with_url() {
        let cfg = StreamingConfig::from_quality(StreamQuality::Low)
            .with_ingest_url("rtmp://live.twitch.tv/app/STREAM_KEY");
        assert!(cfg.is_ready_to_stream());
    }

    #[test]
    fn test_with_title() {
        let cfg = StreamingConfig::from_quality(StreamQuality::High).with_title("Evening Grind");
        assert_eq!(cfg.title.as_deref(), Some("Evening Grind"));
    }

    #[test]
    fn test_low_latency_flag() {
        let cfg = StreamingConfig::from_quality(StreamQuality::High).with_low_latency(true);
        assert!(cfg.low_latency);
    }

    // --- StreamConfigManager ---

    #[test]
    fn test_manager_starts_empty() {
        let m = StreamConfigManager::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert!(m.active_config().is_none());
    }

    #[test]
    fn test_save_and_get() {
        let mut m = StreamConfigManager::new();
        let cfg = StreamingConfig::from_quality(StreamQuality::Ultra);
        m.save("main", cfg.clone());
        assert_eq!(m.get("main"), Some(&cfg));
    }

    #[test]
    fn test_remove_profile() {
        let mut m = StreamConfigManager::new();
        m.save("temp", StreamingConfig::from_quality(StreamQuality::Low));
        let removed = m.remove("temp");
        assert!(removed.is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_set_active_valid() {
        let mut m = StreamConfigManager::new();
        m.save("p1", StreamingConfig::from_quality(StreamQuality::High));
        assert!(m.set_active("p1"));
        assert_eq!(m.active_name(), Some("p1"));
    }

    #[test]
    fn test_set_active_invalid_returns_false() {
        let mut m = StreamConfigManager::new();
        assert!(!m.set_active("nonexistent"));
    }

    #[test]
    fn test_remove_active_clears_active() {
        let mut m = StreamConfigManager::new();
        m.save("s", StreamingConfig::from_quality(StreamQuality::Medium));
        m.set_active("s");
        m.remove("s");
        assert!(m.active_name().is_none());
    }

    #[test]
    fn test_profile_names_sorted() {
        let mut m = StreamConfigManager::new();
        m.save(
            "z_profile",
            StreamingConfig::from_quality(StreamQuality::Low),
        );
        m.save(
            "a_profile",
            StreamingConfig::from_quality(StreamQuality::High),
        );
        let names = m.profile_names();
        assert_eq!(names, vec!["a_profile", "z_profile"]);
    }
}
