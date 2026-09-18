//! Channel format and configuration management for the playout server.
//!
//! Provides `ChannelFormat`, `ChannelConfig`, and `ChannelConfigStore` for
//! describing and querying broadcast channel parameters.

#![allow(dead_code)]

use std::collections::HashMap;

// ── ChannelFormat ─────────────────────────────────────────────────────────────

/// Broadcast channel resolution class.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChannelFormat {
    /// Standard definition (720×576 PAL or 720×480 NTSC).
    Sd,
    /// High definition (1920×1080 or 1280×720).
    Hd,
    /// Ultra-high definition 4K (3840×2160).
    Uhd,
}

impl ChannelFormat {
    /// Nominal pixel width for the format.
    pub fn width(&self) -> u32 {
        match self {
            Self::Sd => 720,
            Self::Hd => 1920,
            Self::Uhd => 3840,
        }
    }

    /// Nominal pixel height for the format.
    pub fn height(&self) -> u32 {
        match self {
            Self::Sd => 576,
            Self::Hd => 1080,
            Self::Uhd => 2160,
        }
    }

    /// Total pixel count (width × height).
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width()) * u64::from(self.height())
    }

    /// Returns `true` when this format is high-definition or better.
    pub fn is_hd_or_better(&self) -> bool {
        matches!(self, Self::Hd | Self::Uhd)
    }

    /// Aspect ratio as `(width_parts, height_parts)` in simplified form.
    pub fn aspect_ratio(&self) -> (u32, u32) {
        match self {
            Self::Sd => (4, 3),
            Self::Hd | Self::Uhd => (16, 9),
        }
    }
}

// ── ChannelConfig ─────────────────────────────────────────────────────────────

/// Full configuration for a single playout channel.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Unique channel identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Video format (resolution class).
    pub format: ChannelFormat,
    /// Frame rate in frames per second.
    pub frame_rate: f64,
    /// Number of audio channels.
    pub audio_channels: u8,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Whether this channel is active.
    pub active: bool,
}

impl ChannelConfig {
    /// Create a new channel configuration with required fields.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        format: ChannelFormat,
        frame_rate: f64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            format,
            frame_rate,
            audio_channels: 2,
            sample_rate: 48_000,
            active: true,
        }
    }

    /// Returns `true` when the channel is HD or UHD.
    pub fn is_hd(&self) -> bool {
        self.format.is_hd_or_better()
    }

    /// Pixel width of this channel's format.
    pub fn width(&self) -> u32 {
        self.format.width()
    }

    /// Pixel height of this channel's format.
    pub fn height(&self) -> u32 {
        self.format.height()
    }

    /// Frame interval in milliseconds (1000 / frame_rate).
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_interval_ms(&self) -> f64 {
        if self.frame_rate > 0.0 {
            1000.0 / self.frame_rate
        } else {
            0.0
        }
    }

    /// Enable or disable this channel.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

// ── ChannelConfigStore ────────────────────────────────────────────────────────

/// A registry of named channel configurations.
pub struct ChannelConfigStore {
    channels: HashMap<String, ChannelConfig>,
}

impl ChannelConfigStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Add a channel configuration.  Replaces any existing entry with the same ID.
    pub fn add(&mut self, config: ChannelConfig) {
        self.channels.insert(config.id.clone(), config);
    }

    /// Find a channel by ID, returning a reference.
    pub fn find(&self, id: &str) -> Option<&ChannelConfig> {
        self.channels.get(id)
    }

    /// Find a mutable channel by ID.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut ChannelConfig> {
        self.channels.get_mut(id)
    }

    /// Remove a channel configuration, returning it if present.
    pub fn remove(&mut self, id: &str) -> Option<ChannelConfig> {
        self.channels.remove(id)
    }

    /// Total number of stored channels.
    pub fn count(&self) -> usize {
        self.channels.len()
    }

    /// Returns `true` when no channels are stored.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Return all HD (or better) channels.
    pub fn hd_channels(&self) -> Vec<&ChannelConfig> {
        self.channels.values().filter(|c| c.is_hd()).collect()
    }

    /// Return all active channels.
    pub fn active_channels(&self) -> Vec<&ChannelConfig> {
        self.channels.values().filter(|c| c.active).collect()
    }
}

impl Default for ChannelConfigStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_channel() -> ChannelConfig {
        ChannelConfig::new("ch1", "HD Main", ChannelFormat::Hd, 25.0)
    }

    fn sd_channel() -> ChannelConfig {
        ChannelConfig::new("ch2", "SD Archive", ChannelFormat::Sd, 25.0)
    }

    fn uhd_channel() -> ChannelConfig {
        ChannelConfig::new("ch3", "UHD Premium", ChannelFormat::Uhd, 50.0)
    }

    // ChannelFormat

    #[test]
    fn format_sd_dimensions() {
        assert_eq!(ChannelFormat::Sd.width(), 720);
        assert_eq!(ChannelFormat::Sd.height(), 576);
    }

    #[test]
    fn format_hd_dimensions() {
        assert_eq!(ChannelFormat::Hd.width(), 1920);
        assert_eq!(ChannelFormat::Hd.height(), 1080);
    }

    #[test]
    fn format_uhd_dimensions() {
        assert_eq!(ChannelFormat::Uhd.width(), 3840);
        assert_eq!(ChannelFormat::Uhd.height(), 2160);
    }

    #[test]
    fn format_pixel_count() {
        assert_eq!(ChannelFormat::Hd.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn format_is_hd_or_better() {
        assert!(!ChannelFormat::Sd.is_hd_or_better());
        assert!(ChannelFormat::Hd.is_hd_or_better());
        assert!(ChannelFormat::Uhd.is_hd_or_better());
    }

    #[test]
    fn format_aspect_ratio_sd() {
        assert_eq!(ChannelFormat::Sd.aspect_ratio(), (4, 3));
    }

    #[test]
    fn format_aspect_ratio_hd() {
        assert_eq!(ChannelFormat::Hd.aspect_ratio(), (16, 9));
        assert_eq!(ChannelFormat::Uhd.aspect_ratio(), (16, 9));
    }

    // ChannelConfig

    #[test]
    fn channel_config_is_hd_true() {
        assert!(hd_channel().is_hd());
        assert!(uhd_channel().is_hd());
    }

    #[test]
    fn channel_config_is_hd_false_for_sd() {
        assert!(!sd_channel().is_hd());
    }

    #[test]
    fn channel_config_dimensions() {
        let ch = hd_channel();
        assert_eq!(ch.width(), 1920);
        assert_eq!(ch.height(), 1080);
    }

    #[test]
    fn channel_config_frame_interval() {
        let ch = hd_channel(); // 25 fps → 40 ms
        let interval = ch.frame_interval_ms();
        assert!((interval - 40.0).abs() < 0.001);
    }

    #[test]
    fn channel_config_set_active() {
        let mut ch = hd_channel();
        assert!(ch.active);
        ch.set_active(false);
        assert!(!ch.active);
    }

    // ChannelConfigStore

    #[test]
    fn store_add_and_count() {
        let mut store = ChannelConfigStore::new();
        assert!(store.is_empty());
        store.add(hd_channel());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn store_find() {
        let mut store = ChannelConfigStore::new();
        store.add(hd_channel());
        assert!(store.find("ch1").is_some());
        assert!(store.find("ch99").is_none());
    }

    #[test]
    fn store_remove() {
        let mut store = ChannelConfigStore::new();
        store.add(sd_channel());
        let removed = store.remove("ch2");
        assert!(removed.is_some());
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn store_hd_channels_filter() {
        let mut store = ChannelConfigStore::new();
        store.add(hd_channel());
        store.add(sd_channel());
        store.add(uhd_channel());
        assert_eq!(store.hd_channels().len(), 2);
    }

    #[test]
    fn store_active_channels_filter() {
        let mut store = ChannelConfigStore::new();
        store.add(hd_channel());
        let mut inactive = sd_channel();
        inactive.set_active(false);
        store.add(inactive);
        assert_eq!(store.active_channels().len(), 1);
    }

    #[test]
    fn store_replace_existing() {
        let mut store = ChannelConfigStore::new();
        store.add(hd_channel());
        store.add(hd_channel()); // same id → replace
        assert_eq!(store.count(), 1);
    }
}
