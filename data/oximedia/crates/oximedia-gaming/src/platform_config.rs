//! Platform-specific configuration and a registry for managing multiple
//! streaming destinations simultaneously.
//!
//! Provides [`PlatformType`] to enumerate known platforms,
//! [`PlatformConfig`] to hold per-platform settings, and
//! [`PlatformRegistry`] to manage a collection of configured destinations.

#![allow(dead_code)]

use std::collections::HashMap;

/// Known streaming / content platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlatformType {
    /// Twitch live-streaming platform.
    Twitch,
    /// `YouTube` live / `YouTube` Gaming.
    YouTube,
    /// Facebook Gaming / Facebook Live.
    Facebook,
    /// `TikTok` LIVE.
    TikTok,
    /// Kick streaming platform.
    Kick,
    /// Custom RTMP/SRT destination.
    CustomRtmp,
}

impl PlatformType {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Twitch => "Twitch",
            Self::YouTube => "YouTube",
            Self::Facebook => "Facebook",
            Self::TikTok => "TikTok",
            Self::Kick => "Kick",
            Self::CustomRtmp => "Custom RTMP",
        }
    }

    /// Default ingest server URL for the platform (empty for custom).
    #[must_use]
    pub fn default_ingest_url(self) -> &'static str {
        match self {
            Self::Twitch => "rtmp://live.twitch.tv/app",
            Self::YouTube => "rtmp://a.rtmp.youtube.com/live2",
            Self::Facebook => "rtmps://live-api-s.facebook.com:443/rtmp/",
            Self::TikTok => "rtmp://push.tiktok.com/live/",
            Self::Kick => "rtmp://fa723fc1b171.global-contribute.live-video.net/app",
            Self::CustomRtmp => "",
        }
    }

    /// Recommended maximum bitrate in kbps for this platform.
    #[must_use]
    pub fn recommended_max_bitrate_kbps(self) -> u32 {
        match self {
            Self::Twitch => 6000,
            Self::YouTube => 12_000,
            Self::Facebook => 4000,
            Self::TikTok => 4000,
            Self::Kick => 8000,
            Self::CustomRtmp => 50_000,
        }
    }

    /// Whether the platform requires a secure (RTMPS/TLS) connection.
    #[must_use]
    pub fn requires_tls(self) -> bool {
        matches!(self, Self::Facebook)
    }
}

/// Per-platform streaming configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformConfig {
    /// Platform type.
    pub platform: PlatformType,
    /// Stream key (kept opaque for security; not logged).
    pub stream_key: String,
    /// Ingest URL override.  `None` means use the platform default.
    pub ingest_url: Option<String>,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Whether this destination is currently enabled.
    pub enabled: bool,
    /// Optional human-readable label (e.g. "Main", "Backup").
    pub label: Option<String>,
}

impl PlatformConfig {
    /// Create a new platform configuration.
    #[must_use]
    pub fn new(platform: PlatformType, stream_key: impl Into<String>) -> Self {
        Self {
            bitrate_kbps: platform.recommended_max_bitrate_kbps(),
            platform,
            stream_key: stream_key.into(),
            ingest_url: None,
            enabled: true,
            label: None,
        }
    }

    /// Override the ingest URL.
    #[must_use]
    pub fn with_ingest_url(mut self, url: impl Into<String>) -> Self {
        self.ingest_url = Some(url.into());
        self
    }

    /// Set the target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Enable or disable this destination.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Effective ingest URL (custom override or platform default).
    #[must_use]
    pub fn effective_ingest_url(&self) -> &str {
        self.ingest_url
            .as_deref()
            .unwrap_or_else(|| self.platform.default_ingest_url())
    }

    /// Whether the configured bitrate exceeds the platform recommendation.
    #[must_use]
    pub fn is_bitrate_exceeded(&self) -> bool {
        self.bitrate_kbps > self.platform.recommended_max_bitrate_kbps()
    }
}

/// Registry of configured streaming destinations.
///
/// Allows multi-streaming to several platforms simultaneously.
#[derive(Debug, Clone)]
pub struct PlatformRegistry {
    /// Configs keyed by a unique label (auto-generated from platform name
    /// when no explicit label is set).
    configs: HashMap<String, PlatformConfig>,
}

impl PlatformRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Register a platform configuration.
    ///
    /// Uses the explicit label if set, otherwise falls back to the platform
    /// name.  Returns `false` if a config with the same key already exists.
    pub fn register(&mut self, config: PlatformConfig) -> bool {
        let key = config
            .label
            .clone()
            .unwrap_or_else(|| config.platform.name().to_owned());
        if self.configs.contains_key(&key) {
            return false;
        }
        self.configs.insert(key, config);
        true
    }

    /// Remove a configuration by key.
    pub fn remove(&mut self, key: &str) -> Option<PlatformConfig> {
        self.configs.remove(key)
    }

    /// Retrieve a configuration by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&PlatformConfig> {
        self.configs.get(key)
    }

    /// All registered keys.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    /// Number of registered destinations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// All enabled destinations.
    #[must_use]
    pub fn enabled(&self) -> Vec<&PlatformConfig> {
        self.configs.values().filter(|c| c.enabled).collect()
    }

    /// Validate all destinations.  Returns a list of keys whose bitrate
    /// exceeds the platform recommendation.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        self.configs
            .iter()
            .filter(|(_, c)| c.is_bitrate_exceeded())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Total combined bitrate of all enabled destinations (kbps).
    #[must_use]
    pub fn total_bitrate_kbps(&self) -> u64 {
        self.configs
            .values()
            .filter(|c| c.enabled)
            .map(|c| u64::from(c.bitrate_kbps))
            .sum()
    }
}

impl Default for PlatformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PlatformType ---

    #[test]
    fn test_platform_names() {
        assert_eq!(PlatformType::Twitch.name(), "Twitch");
        assert_eq!(PlatformType::YouTube.name(), "YouTube");
        assert_eq!(PlatformType::CustomRtmp.name(), "Custom RTMP");
    }

    #[test]
    fn test_default_ingest_urls() {
        assert!(!PlatformType::Twitch.default_ingest_url().is_empty());
        assert!(PlatformType::CustomRtmp.default_ingest_url().is_empty());
    }

    #[test]
    fn test_recommended_bitrates() {
        assert_eq!(PlatformType::Twitch.recommended_max_bitrate_kbps(), 6000);
        assert!(PlatformType::YouTube.recommended_max_bitrate_kbps() > 0);
    }

    #[test]
    fn test_requires_tls() {
        assert!(PlatformType::Facebook.requires_tls());
        assert!(!PlatformType::Twitch.requires_tls());
    }

    // --- PlatformConfig ---

    #[test]
    fn test_config_creation() {
        let cfg = PlatformConfig::new(PlatformType::Twitch, "live_abc123");
        assert_eq!(cfg.platform, PlatformType::Twitch);
        assert_eq!(cfg.stream_key, "live_abc123");
        assert!(cfg.enabled);
        assert_eq!(cfg.bitrate_kbps, 6000);
    }

    #[test]
    fn test_effective_ingest_url_default() {
        let cfg = PlatformConfig::new(PlatformType::YouTube, "key");
        assert_eq!(
            cfg.effective_ingest_url(),
            PlatformType::YouTube.default_ingest_url()
        );
    }

    #[test]
    fn test_effective_ingest_url_override() {
        let cfg = PlatformConfig::new(PlatformType::CustomRtmp, "key")
            .with_ingest_url("rtmp://my.server/live");
        assert_eq!(cfg.effective_ingest_url(), "rtmp://my.server/live");
    }

    #[test]
    fn test_bitrate_exceeded() {
        let cfg = PlatformConfig::new(PlatformType::Twitch, "key").with_bitrate(10_000);
        assert!(cfg.is_bitrate_exceeded());

        let normal = PlatformConfig::new(PlatformType::Twitch, "key");
        assert!(!normal.is_bitrate_exceeded());
    }

    // --- PlatformRegistry ---

    #[test]
    fn test_registry_empty() {
        let reg = PlatformRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = PlatformRegistry::new();
        let cfg = PlatformConfig::new(PlatformType::Twitch, "key123");
        assert!(reg.register(cfg));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("Twitch").is_some());
    }

    #[test]
    fn test_registry_duplicate_key_rejected() {
        let mut reg = PlatformRegistry::new();
        reg.register(PlatformConfig::new(PlatformType::Twitch, "key1"));
        let dup = PlatformConfig::new(PlatformType::Twitch, "key2");
        assert!(!reg.register(dup));
    }

    #[test]
    fn test_registry_label_as_key() {
        let mut reg = PlatformRegistry::new();
        let cfg = PlatformConfig::new(PlatformType::Twitch, "key").with_label("Primary");
        reg.register(cfg);
        assert!(reg.get("Primary").is_some());
        assert!(reg.get("Twitch").is_none());
    }

    #[test]
    fn test_registry_enabled_filter() {
        let mut reg = PlatformRegistry::new();
        reg.register(PlatformConfig::new(PlatformType::Twitch, "k1").with_label("TW"));
        reg.register(
            PlatformConfig::new(PlatformType::YouTube, "k2")
                .with_label("YT")
                .with_enabled(false),
        );
        assert_eq!(reg.enabled().len(), 1);
    }

    #[test]
    fn test_registry_total_bitrate() {
        let mut reg = PlatformRegistry::new();
        reg.register(
            PlatformConfig::new(PlatformType::Twitch, "k1")
                .with_label("TW")
                .with_bitrate(6000),
        );
        reg.register(
            PlatformConfig::new(PlatformType::YouTube, "k2")
                .with_label("YT")
                .with_bitrate(8000),
        );
        assert_eq!(reg.total_bitrate_kbps(), 14_000);
    }

    #[test]
    fn test_registry_validate() {
        let mut reg = PlatformRegistry::new();
        reg.register(
            PlatformConfig::new(PlatformType::Twitch, "k1")
                .with_label("TW")
                .with_bitrate(20_000),
        );
        let issues = reg.validate();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0], "TW");
    }
}
