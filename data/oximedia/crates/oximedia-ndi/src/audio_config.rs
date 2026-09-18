#![allow(dead_code)]
//! Audio configuration management for NDI streams.
//!
//! Provides detailed audio configuration including channel layouts,
//! sample format negotiation, and audio routing for NDI connections.

use std::fmt;

/// Supported audio sample formats for NDI transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiSampleFormat {
    /// 16-bit signed integer PCM.
    S16,
    /// 24-bit signed integer PCM (packed in 32 bits).
    S24,
    /// 32-bit signed integer PCM.
    S32,
    /// 32-bit IEEE floating point.
    F32,
}

impl NdiSampleFormat {
    /// Returns the number of bytes per sample for this format.
    #[must_use]
    pub fn bytes_per_sample(self) -> u32 {
        match self {
            Self::S16 => 2,
            Self::S24 | Self::S32 | Self::F32 => 4,
        }
    }

    /// Returns the bit depth of this format.
    #[must_use]
    pub fn bit_depth(self) -> u32 {
        match self {
            Self::S16 => 16,
            Self::S24 => 24,
            Self::S32 => 32,
            Self::F32 => 32,
        }
    }

    /// Returns true if this is a floating-point format.
    #[must_use]
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32)
    }
}

impl fmt::Display for NdiSampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S16 => write!(f, "S16"),
            Self::S24 => write!(f, "S24"),
            Self::S32 => write!(f, "S32"),
            Self::F32 => write!(f, "F32"),
        }
    }
}

/// Standard audio channel layout designations for NDI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Single mono channel.
    Mono,
    /// Stereo left/right.
    Stereo,
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 surround (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    Surround71,
    /// Custom channel count with no predefined assignment.
    Custom(u16),
}

impl ChannelLayout {
    /// Returns the number of channels in this layout.
    #[must_use]
    pub fn channel_count(self) -> u16 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(n) => n,
        }
    }

    /// Determines a channel layout from a raw channel count.
    #[must_use]
    pub fn from_channel_count(count: u16) -> Self {
        match count {
            1 => Self::Mono,
            2 => Self::Stereo,
            6 => Self::Surround51,
            8 => Self::Surround71,
            n => Self::Custom(n),
        }
    }

    /// Returns true if the layout represents surround sound (more than 2 channels).
    #[must_use]
    pub fn is_surround(self) -> bool {
        self.channel_count() > 2
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mono => write!(f, "Mono"),
            Self::Stereo => write!(f, "Stereo"),
            Self::Surround51 => write!(f, "5.1 Surround"),
            Self::Surround71 => write!(f, "7.1 Surround"),
            Self::Custom(n) => write!(f, "Custom({n}ch)"),
        }
    }
}

/// Complete audio configuration for an NDI audio stream.
#[derive(Debug, Clone, PartialEq)]
pub struct NdiAudioConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel layout.
    pub channel_layout: ChannelLayout,
    /// Sample format.
    pub sample_format: NdiSampleFormat,
    /// Number of samples per channel in each NDI audio frame.
    pub samples_per_frame: u32,
    /// Reference audio level in dBFS (typically -20 or -18).
    pub reference_level_dbfs: f64,
    /// Whether to apply dither when converting between formats.
    pub dither_enabled: bool,
}

impl Default for NdiAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channel_layout: ChannelLayout::Stereo,
            sample_format: NdiSampleFormat::F32,
            samples_per_frame: 1602,
            reference_level_dbfs: -20.0,
            dither_enabled: true,
        }
    }
}

impl NdiAudioConfig {
    /// Creates a new audio configuration with specified parameters.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channel_layout: ChannelLayout,
        sample_format: NdiSampleFormat,
    ) -> Self {
        let samples_per_frame = Self::compute_default_frame_size(sample_rate);
        Self {
            sample_rate,
            channel_layout,
            sample_format,
            samples_per_frame,
            reference_level_dbfs: -20.0,
            dither_enabled: true,
        }
    }

    /// Creates a standard broadcast configuration (48kHz, stereo, F32).
    #[must_use]
    pub fn broadcast_stereo() -> Self {
        Self::default()
    }

    /// Creates a 5.1 surround configuration at 48kHz.
    #[must_use]
    pub fn surround_51() -> Self {
        Self::new(48000, ChannelLayout::Surround51, NdiSampleFormat::F32)
    }

    /// Creates a 7.1 surround configuration at 48kHz.
    #[must_use]
    pub fn surround_71() -> Self {
        Self::new(48000, ChannelLayout::Surround71, NdiSampleFormat::F32)
    }

    /// Computes a default frame size (number of samples per channel per frame)
    /// for the given sample rate at approximately 30 fps.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    fn compute_default_frame_size(sample_rate: u32) -> u32 {
        // At ~29.97 fps (NTSC), we need sample_rate / 29.97 samples per frame
        let ideal = sample_rate as f64 / 29.97;
        ideal.ceil() as u32
    }

    /// Returns the total number of bytes required for one frame of audio data.
    #[must_use]
    pub fn frame_byte_size(&self) -> usize {
        let channels = self.channel_layout.channel_count() as usize;
        let bytes_per_sample = self.sample_format.bytes_per_sample() as usize;
        self.samples_per_frame as usize * channels * bytes_per_sample
    }

    /// Returns the duration of one audio frame in microseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_duration_us(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        let seconds = self.samples_per_frame as f64 / self.sample_rate as f64;
        (seconds * 1_000_000.0) as u64
    }

    /// Returns the audio bitrate in bits per second.
    #[must_use]
    pub fn bitrate_bps(&self) -> u64 {
        let channels = u64::from(self.channel_layout.channel_count());
        let bit_depth = u64::from(self.sample_format.bit_depth());
        let rate = u64::from(self.sample_rate);
        channels * bit_depth * rate
    }

    /// Checks whether the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.sample_rate > 0
            && self.channel_layout.channel_count() > 0
            && self.samples_per_frame > 0
    }

    /// Returns true if this config can be directly mixed with another
    /// (same sample rate and sample format).
    #[must_use]
    pub fn is_mix_compatible(&self, other: &Self) -> bool {
        self.sample_rate == other.sample_rate && self.sample_format == other.sample_format
    }

    /// Returns a string description of this configuration.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{}Hz {} {} ({} samples/frame)",
            self.sample_rate, self.channel_layout, self.sample_format, self.samples_per_frame
        )
    }
}

/// Routing assignment for an individual audio channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChannelRoute {
    /// Source channel index.
    pub source_channel: u16,
    /// Destination channel index.
    pub dest_channel: u16,
    /// Gain in millibels (0 = unity, -6000 = -60 dB).
    pub gain_mb: i32,
    /// Whether this route is muted.
    pub muted: bool,
}

impl AudioChannelRoute {
    /// Creates a new 1:1 channel route at unity gain.
    #[must_use]
    pub fn unity(source: u16, dest: u16) -> Self {
        Self {
            source_channel: source,
            dest_channel: dest,
            gain_mb: 0,
            muted: false,
        }
    }

    /// Creates a muted route.
    #[must_use]
    pub fn muted(source: u16, dest: u16) -> Self {
        Self {
            source_channel: source,
            dest_channel: dest,
            gain_mb: 0,
            muted: true,
        }
    }

    /// Converts the gain from millibels to a linear multiplier.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn linear_gain(&self) -> f64 {
        if self.muted {
            return 0.0;
        }
        let db = self.gain_mb as f64 / 100.0;
        10.0_f64.powf(db / 20.0)
    }
}

/// A complete audio routing matrix for mapping source channels to destinations.
#[derive(Debug, Clone)]
pub struct AudioRoutingMatrix {
    /// Ordered list of channel routes.
    pub routes: Vec<AudioChannelRoute>,
    /// Number of source channels.
    pub source_channels: u16,
    /// Number of destination channels.
    pub dest_channels: u16,
}

impl AudioRoutingMatrix {
    /// Creates a pass-through routing matrix (1:1 mapping).
    #[must_use]
    pub fn passthrough(channels: u16) -> Self {
        let routes = (0..channels)
            .map(|ch| AudioChannelRoute::unity(ch, ch))
            .collect();
        Self {
            routes,
            source_channels: channels,
            dest_channels: channels,
        }
    }

    /// Creates a stereo-to-mono downmix routing.
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        Self {
            routes: vec![
                AudioChannelRoute {
                    source_channel: 0,
                    dest_channel: 0,
                    gain_mb: -301, // -3.01 dB for equal-power sum
                    muted: false,
                },
                AudioChannelRoute {
                    source_channel: 1,
                    dest_channel: 0,
                    gain_mb: -301,
                    muted: false,
                },
            ],
            source_channels: 2,
            dest_channels: 1,
        }
    }

    /// Returns the number of routes defined in this matrix.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Returns all routes targeting a specific destination channel.
    #[must_use]
    pub fn routes_for_dest(&self, dest: u16) -> Vec<&AudioChannelRoute> {
        self.routes
            .iter()
            .filter(|r| r.dest_channel == dest)
            .collect()
    }

    /// Returns true if the routing matrix has any muted routes.
    #[must_use]
    pub fn has_muted_routes(&self) -> bool {
        self.routes.iter().any(|r| r.muted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bytes() {
        assert_eq!(NdiSampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(NdiSampleFormat::S24.bytes_per_sample(), 4);
        assert_eq!(NdiSampleFormat::S32.bytes_per_sample(), 4);
        assert_eq!(NdiSampleFormat::F32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_sample_format_bit_depth() {
        assert_eq!(NdiSampleFormat::S16.bit_depth(), 16);
        assert_eq!(NdiSampleFormat::S24.bit_depth(), 24);
        assert_eq!(NdiSampleFormat::S32.bit_depth(), 32);
        assert_eq!(NdiSampleFormat::F32.bit_depth(), 32);
    }

    #[test]
    fn test_sample_format_is_float() {
        assert!(!NdiSampleFormat::S16.is_float());
        assert!(!NdiSampleFormat::S24.is_float());
        assert!(NdiSampleFormat::F32.is_float());
    }

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayout::Custom(4).channel_count(), 4);
    }

    #[test]
    fn test_channel_layout_from_count() {
        assert_eq!(ChannelLayout::from_channel_count(1), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from_channel_count(2), ChannelLayout::Stereo);
        assert_eq!(
            ChannelLayout::from_channel_count(6),
            ChannelLayout::Surround51
        );
        assert_eq!(
            ChannelLayout::from_channel_count(8),
            ChannelLayout::Surround71
        );
        assert_eq!(
            ChannelLayout::from_channel_count(4),
            ChannelLayout::Custom(4)
        );
    }

    #[test]
    fn test_channel_layout_is_surround() {
        assert!(!ChannelLayout::Mono.is_surround());
        assert!(!ChannelLayout::Stereo.is_surround());
        assert!(ChannelLayout::Surround51.is_surround());
        assert!(ChannelLayout::Surround71.is_surround());
    }

    #[test]
    fn test_audio_config_default() {
        let cfg = NdiAudioConfig::default();
        assert_eq!(cfg.sample_rate, 48000);
        assert_eq!(cfg.channel_layout, ChannelLayout::Stereo);
        assert_eq!(cfg.sample_format, NdiSampleFormat::F32);
        assert!(cfg.dither_enabled);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_audio_config_frame_byte_size() {
        let cfg = NdiAudioConfig {
            sample_rate: 48000,
            channel_layout: ChannelLayout::Stereo,
            sample_format: NdiSampleFormat::S16,
            samples_per_frame: 1024,
            reference_level_dbfs: -20.0,
            dither_enabled: false,
        };
        // 1024 samples * 2 channels * 2 bytes = 4096
        assert_eq!(cfg.frame_byte_size(), 4096);
    }

    #[test]
    fn test_audio_config_bitrate() {
        let cfg = NdiAudioConfig::new(48000, ChannelLayout::Stereo, NdiSampleFormat::S16);
        // 2 channels * 16 bits * 48000 = 1_536_000
        assert_eq!(cfg.bitrate_bps(), 1_536_000);
    }

    #[test]
    fn test_audio_config_frame_duration() {
        let cfg = NdiAudioConfig {
            sample_rate: 48000,
            channel_layout: ChannelLayout::Stereo,
            sample_format: NdiSampleFormat::F32,
            samples_per_frame: 48000, // exactly 1 second
            reference_level_dbfs: -20.0,
            dither_enabled: false,
        };
        assert_eq!(cfg.frame_duration_us(), 1_000_000);
    }

    #[test]
    fn test_audio_config_mix_compatible() {
        let a = NdiAudioConfig::new(48000, ChannelLayout::Stereo, NdiSampleFormat::F32);
        let b = NdiAudioConfig::new(48000, ChannelLayout::Surround51, NdiSampleFormat::F32);
        let c = NdiAudioConfig::new(44100, ChannelLayout::Stereo, NdiSampleFormat::F32);
        assert!(a.is_mix_compatible(&b)); // same rate & format, different layout OK
        assert!(!a.is_mix_compatible(&c)); // different rate
    }

    #[test]
    fn test_channel_route_unity() {
        let route = AudioChannelRoute::unity(0, 1);
        assert_eq!(route.source_channel, 0);
        assert_eq!(route.dest_channel, 1);
        assert_eq!(route.gain_mb, 0);
        assert!(!route.muted);
        assert!((route.linear_gain() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_route_muted() {
        let route = AudioChannelRoute::muted(0, 0);
        assert!(route.muted);
        assert!((route.linear_gain()).abs() < 1e-9);
    }

    #[test]
    fn test_routing_matrix_passthrough() {
        let matrix = AudioRoutingMatrix::passthrough(8);
        assert_eq!(matrix.route_count(), 8);
        assert_eq!(matrix.source_channels, 8);
        assert_eq!(matrix.dest_channels, 8);
        assert!(!matrix.has_muted_routes());
    }

    #[test]
    fn test_routing_matrix_stereo_to_mono() {
        let matrix = AudioRoutingMatrix::stereo_to_mono();
        assert_eq!(matrix.route_count(), 2);
        assert_eq!(matrix.source_channels, 2);
        assert_eq!(matrix.dest_channels, 1);
        let dest0 = matrix.routes_for_dest(0);
        assert_eq!(dest0.len(), 2);
    }

    #[test]
    fn test_surround_configs() {
        let s51 = NdiAudioConfig::surround_51();
        assert_eq!(s51.channel_layout, ChannelLayout::Surround51);
        let s71 = NdiAudioConfig::surround_71();
        assert_eq!(s71.channel_layout, ChannelLayout::Surround71);
    }

    #[test]
    fn test_audio_config_description() {
        let cfg = NdiAudioConfig::default();
        let desc = cfg.description();
        assert!(desc.contains("48000"));
        assert!(desc.contains("Stereo"));
        assert!(desc.contains("F32"));
    }

    #[test]
    fn test_invalid_audio_config() {
        let cfg = NdiAudioConfig {
            sample_rate: 0,
            channel_layout: ChannelLayout::Stereo,
            sample_format: NdiSampleFormat::F32,
            samples_per_frame: 1024,
            reference_level_dbfs: -20.0,
            dither_enabled: false,
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_frame_duration_zero_sample_rate() {
        let cfg = NdiAudioConfig {
            sample_rate: 0,
            channel_layout: ChannelLayout::Mono,
            sample_format: NdiSampleFormat::S16,
            samples_per_frame: 100,
            reference_level_dbfs: -20.0,
            dither_enabled: false,
        };
        assert_eq!(cfg.frame_duration_us(), 0);
    }
}
