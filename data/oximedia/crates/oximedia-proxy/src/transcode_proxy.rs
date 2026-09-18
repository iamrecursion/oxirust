//! Proxy transcoding settings for OxiMedia proxy system.
//!
//! Provides proxy codec selection, bitrate ladders, and quality presets
//! for efficient proxy generation workflows.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Codec choices for proxy generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyCodecChoice {
    /// H.264 / AVC — widely compatible, CPU-efficient.
    H264,
    /// H.265 / HEVC — better compression, higher CPU cost.
    H265,
    /// VP9 — open format, good for web delivery proxies.
    Vp9,
    /// AV1 — best compression ratio, highest CPU cost.
    Av1,
    /// Apple ProRes Proxy — fast decode on Apple hardware.
    ProResProxy,
    /// DNxHD/DNxHR LB — fast decode on Avid systems.
    DnxhdLb,
}

impl ProxyCodecChoice {
    /// Returns the name of the codec as a string.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "hevc",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
            Self::ProResProxy => "prores_ks",
            Self::DnxhdLb => "dnxhd",
        }
    }

    /// Returns the typical container format for this codec.
    #[must_use]
    pub fn container(&self) -> &'static str {
        match self {
            Self::H264 | Self::H265 | Self::Vp9 | Self::Av1 => "mp4",
            Self::ProResProxy => "mov",
            Self::DnxhdLb => "mxf",
        }
    }

    /// Whether this codec supports hardware acceleration on most platforms.
    #[must_use]
    pub fn hardware_accelerated(&self) -> bool {
        matches!(self, Self::H264 | Self::H265)
    }
}

impl Default for ProxyCodecChoice {
    fn default() -> Self {
        Self::H264
    }
}

/// A single rung of a proxy bitrate ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct BitrateLadderRung {
    /// Label for this rung (e.g., "1080p", "720p").
    pub label: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Target bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Codec for this rung.
    pub codec: ProxyCodecChoice,
}

impl BitrateLadderRung {
    /// Create a new bitrate ladder rung.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        width: u32,
        height: u32,
        bitrate_bps: u64,
        codec: ProxyCodecChoice,
    ) -> Self {
        Self {
            label: label.into(),
            width,
            height,
            bitrate_bps,
            codec,
        }
    }

    /// Pixel count for this rung.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Bits per pixel at this rung's bitrate and 24fps.
    #[must_use]
    pub fn bits_per_pixel_at_24fps(&self) -> f64 {
        let pixels_per_second = self.pixel_count() as f64 * 24.0;
        if pixels_per_second <= 0.0 {
            return 0.0;
        }
        self.bitrate_bps as f64 / pixels_per_second
    }
}

/// A proxy bitrate ladder containing multiple resolution rungs.
#[derive(Debug, Clone, Default)]
pub struct ProxyBitrateLadder {
    /// Rungs from highest to lowest quality.
    rungs: Vec<BitrateLadderRung>,
}

impl ProxyBitrateLadder {
    /// Create an empty bitrate ladder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a standard H.264 proxy ladder for offline editing.
    #[must_use]
    pub fn standard_h264() -> Self {
        let mut ladder = Self::new();
        ladder.add_rung(BitrateLadderRung::new(
            "1080p",
            1920,
            1080,
            8_000_000,
            ProxyCodecChoice::H264,
        ));
        ladder.add_rung(BitrateLadderRung::new(
            "720p",
            1280,
            720,
            4_000_000,
            ProxyCodecChoice::H264,
        ));
        ladder.add_rung(BitrateLadderRung::new(
            "540p",
            960,
            540,
            2_000_000,
            ProxyCodecChoice::H264,
        ));
        ladder.add_rung(BitrateLadderRung::new(
            "quarter",
            480,
            270,
            800_000,
            ProxyCodecChoice::H264,
        ));
        ladder
    }

    /// Add a rung to the ladder.
    pub fn add_rung(&mut self, rung: BitrateLadderRung) {
        self.rungs.push(rung);
    }

    /// Number of rungs in the ladder.
    #[must_use]
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Find the rung with the highest bitrate.
    #[must_use]
    pub fn highest_quality_rung(&self) -> Option<&BitrateLadderRung> {
        self.rungs.iter().max_by_key(|r| r.bitrate_bps)
    }

    /// Find the rung with the lowest bitrate.
    #[must_use]
    pub fn lowest_quality_rung(&self) -> Option<&BitrateLadderRung> {
        self.rungs.iter().min_by_key(|r| r.bitrate_bps)
    }

    /// Find a rung by label.
    #[must_use]
    pub fn find_by_label(&self, label: &str) -> Option<&BitrateLadderRung> {
        self.rungs.iter().find(|r| r.label == label)
    }

    /// All rungs in the ladder.
    #[must_use]
    pub fn rungs(&self) -> &[BitrateLadderRung] {
        &self.rungs
    }
}

/// Named quality preset for proxy transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    /// Ultra-low: smallest files, for mobile / slow network editing.
    UltraLow,
    /// Low: good for most offline editing scenarios.
    Low,
    /// Medium: balanced quality and file size.
    Medium,
    /// High: near-lossless proxy for critical color work.
    High,
}

impl Default for QualityPreset {
    fn default() -> Self {
        Self::Low
    }
}

/// Settings for proxy transcoding derived from a quality preset.
#[derive(Debug, Clone)]
pub struct ProxyTranscodeSettings {
    /// Codec to use.
    pub codec: ProxyCodecChoice,
    /// Target bitrate in bps.
    pub bitrate_bps: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// CRF value (lower = better quality; 0 = lossless for supported codecs).
    pub crf: u8,
    /// Number of encoding threads (0 = auto).
    pub threads: u32,
    /// Whether to copy the audio stream without re-encoding.
    pub copy_audio: bool,
}

impl ProxyTranscodeSettings {
    /// Create settings from a quality preset for a 1080p source.
    #[must_use]
    pub fn from_preset_1080p(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::UltraLow => Self {
                codec: ProxyCodecChoice::H264,
                bitrate_bps: 1_000_000,
                width: 480,
                height: 270,
                crf: 35,
                threads: 0,
                copy_audio: true,
            },
            QualityPreset::Low => Self {
                codec: ProxyCodecChoice::H264,
                bitrate_bps: 3_000_000,
                width: 960,
                height: 540,
                crf: 28,
                threads: 0,
                copy_audio: true,
            },
            QualityPreset::Medium => Self {
                codec: ProxyCodecChoice::H264,
                bitrate_bps: 6_000_000,
                width: 1280,
                height: 720,
                crf: 23,
                threads: 0,
                copy_audio: true,
            },
            QualityPreset::High => Self {
                codec: ProxyCodecChoice::H265,
                bitrate_bps: 12_000_000,
                width: 1920,
                height: 1080,
                crf: 18,
                threads: 0,
                copy_audio: true,
            },
        }
    }

    /// Override the codec.
    #[must_use]
    pub fn with_codec(mut self, codec: ProxyCodecChoice) -> Self {
        self.codec = codec;
        self
    }

    /// Override the thread count.
    #[must_use]
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    /// Override the CRF value.
    #[must_use]
    pub fn with_crf(mut self, crf: u8) -> Self {
        self.crf = crf;
        self
    }

    /// Resolution as a tuple (width, height).
    #[must_use]
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Estimated file size in megabytes for a given duration in seconds.
    #[must_use]
    pub fn estimated_size_mb(&self, duration_secs: f64) -> f64 {
        (self.bitrate_bps as f64 * duration_secs) / (8.0 * 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_name() {
        assert_eq!(ProxyCodecChoice::H264.name(), "h264");
        assert_eq!(ProxyCodecChoice::H265.name(), "hevc");
        assert_eq!(ProxyCodecChoice::Vp9.name(), "vp9");
        assert_eq!(ProxyCodecChoice::ProResProxy.name(), "prores_ks");
    }

    #[test]
    fn test_codec_container() {
        assert_eq!(ProxyCodecChoice::H264.container(), "mp4");
        assert_eq!(ProxyCodecChoice::ProResProxy.container(), "mov");
        assert_eq!(ProxyCodecChoice::DnxhdLb.container(), "mxf");
    }

    #[test]
    fn test_codec_hardware_accelerated() {
        assert!(ProxyCodecChoice::H264.hardware_accelerated());
        assert!(ProxyCodecChoice::H265.hardware_accelerated());
        assert!(!ProxyCodecChoice::Av1.hardware_accelerated());
        assert!(!ProxyCodecChoice::Vp9.hardware_accelerated());
    }

    #[test]
    fn test_codec_default() {
        assert_eq!(ProxyCodecChoice::default(), ProxyCodecChoice::H264);
    }

    #[test]
    fn test_bitrate_ladder_rung_pixel_count() {
        let rung = BitrateLadderRung::new("1080p", 1920, 1080, 8_000_000, ProxyCodecChoice::H264);
        assert_eq!(rung.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_bitrate_ladder_rung_bits_per_pixel() {
        let rung = BitrateLadderRung::new("test", 100, 100, 2_400_000, ProxyCodecChoice::H264);
        // 2_400_000 / (10000 * 24) = 10.0
        let bpp = rung.bits_per_pixel_at_24fps();
        assert!((bpp - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_proxy_bitrate_ladder_standard_h264() {
        let ladder = ProxyBitrateLadder::standard_h264();
        assert_eq!(ladder.rung_count(), 4);
    }

    #[test]
    fn test_proxy_bitrate_ladder_highest_quality() {
        let ladder = ProxyBitrateLadder::standard_h264();
        let rung = ladder
            .highest_quality_rung()
            .expect("should succeed in test");
        assert_eq!(rung.bitrate_bps, 8_000_000);
    }

    #[test]
    fn test_proxy_bitrate_ladder_lowest_quality() {
        let ladder = ProxyBitrateLadder::standard_h264();
        let rung = ladder
            .lowest_quality_rung()
            .expect("should succeed in test");
        assert_eq!(rung.bitrate_bps, 800_000);
    }

    #[test]
    fn test_proxy_bitrate_ladder_find_by_label() {
        let ladder = ProxyBitrateLadder::standard_h264();
        assert!(ladder.find_by_label("720p").is_some());
        assert!(ladder.find_by_label("4k").is_none());
    }

    #[test]
    fn test_proxy_bitrate_ladder_empty() {
        let ladder = ProxyBitrateLadder::new();
        assert_eq!(ladder.rung_count(), 0);
        assert!(ladder.highest_quality_rung().is_none());
        assert!(ladder.lowest_quality_rung().is_none());
    }

    #[test]
    fn test_proxy_transcode_settings_from_preset_low() {
        let settings = ProxyTranscodeSettings::from_preset_1080p(QualityPreset::Low);
        assert_eq!(settings.codec, ProxyCodecChoice::H264);
        assert_eq!(settings.width, 960);
        assert_eq!(settings.height, 540);
    }

    #[test]
    fn test_proxy_transcode_settings_from_preset_high() {
        let settings = ProxyTranscodeSettings::from_preset_1080p(QualityPreset::High);
        assert_eq!(settings.codec, ProxyCodecChoice::H265);
        assert_eq!(settings.width, 1920);
    }

    #[test]
    fn test_proxy_transcode_settings_with_codec() {
        let settings = ProxyTranscodeSettings::from_preset_1080p(QualityPreset::Low)
            .with_codec(ProxyCodecChoice::Vp9);
        assert_eq!(settings.codec, ProxyCodecChoice::Vp9);
    }

    #[test]
    fn test_proxy_transcode_settings_resolution() {
        let settings = ProxyTranscodeSettings::from_preset_1080p(QualityPreset::Medium);
        assert_eq!(settings.resolution(), (1280, 720));
    }

    #[test]
    fn test_proxy_transcode_settings_estimated_size() {
        let settings = ProxyTranscodeSettings::from_preset_1080p(QualityPreset::Low);
        // 3_000_000 bps * 10s / 8 / 1_000_000 = 3.75 MB
        let size = settings.estimated_size_mb(10.0);
        assert!((size - 3.75).abs() < 1e-6);
    }

    #[test]
    fn test_quality_preset_default() {
        assert_eq!(QualityPreset::default(), QualityPreset::Low);
    }
}
