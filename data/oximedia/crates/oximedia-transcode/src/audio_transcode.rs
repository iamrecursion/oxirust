//! Audio-specific transcoding configuration and utilities.
//!
//! This module provides structures and helper functions for configuring
//! audio transcoding operations, including codec selection, bitrate estimation,
//! loudness normalisation, and channel layout naming.

#![allow(dead_code)]

/// Configuration for an audio transcode operation.
#[derive(Debug, Clone)]
pub struct AudioTranscodeConfig {
    /// Name of the input audio codec (e.g., `"aac"`, `"flac"`).
    pub input_codec: String,
    /// Name of the output audio codec (e.g., `"opus"`, `"aac"`).
    pub output_codec: String,
    /// Target bitrate in kilobits per second. Ignored for lossless codecs.
    pub bitrate_kbps: u32,
    /// Output sample rate in Hz (e.g., 48000).
    pub sample_rate: u32,
    /// Number of output channels (1 = mono, 2 = stereo, 6 = 5.1, etc.).
    pub channels: u8,
    /// Whether to apply loudness normalisation.
    pub normalize: bool,
    /// Target loudness in LUFS for normalisation (e.g., -23.0 for EBU R128).
    pub target_lufs: f64,
}

impl AudioTranscodeConfig {
    /// Creates a new config with the given codecs and basic parameters.
    pub fn new(
        input_codec: impl Into<String>,
        output_codec: impl Into<String>,
        bitrate_kbps: u32,
        sample_rate: u32,
        channels: u8,
    ) -> Self {
        Self {
            input_codec: input_codec.into(),
            output_codec: output_codec.into(),
            bitrate_kbps,
            sample_rate,
            channels,
            normalize: false,
            target_lufs: -23.0,
        }
    }

    /// Returns a config for AAC stereo at 256 kbps / 48 kHz.
    #[must_use]
    pub fn aac_stereo_256k() -> Self {
        Self::new("pcm_s24le", "aac", 256, 48_000, 2)
    }

    /// Returns a config for Opus stereo at 128 kbps / 48 kHz.
    #[must_use]
    pub fn opus_stereo_128k() -> Self {
        Self::new("pcm_s24le", "opus", 128, 48_000, 2)
    }

    /// Returns a config for FLAC lossless stereo at 48 kHz.
    #[must_use]
    pub fn flac_lossless() -> Self {
        let mut cfg = Self::new("pcm_s24le", "flac", 0, 48_000, 2);
        cfg.bitrate_kbps = 0; // lossless – bitrate not applicable
        cfg
    }

    /// Enables loudness normalisation with the given LUFS target.
    #[must_use]
    pub fn with_normalization(mut self, target_lufs: f64) -> Self {
        self.normalize = true;
        self.target_lufs = target_lufs;
        self
    }

    /// Returns `true` if the output codec is lossless.
    #[must_use]
    pub fn is_lossless_output(&self) -> bool {
        is_lossless_codec(&self.output_codec)
    }

    /// Returns `true` if the configuration is considered valid.
    ///
    /// A valid config has a non-empty output codec, a positive sample rate,
    /// and at least one channel.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.output_codec.is_empty() && self.sample_rate > 0 && self.channels > 0
    }
}

/// Represents a pending audio transcode job.
#[derive(Debug, Clone)]
pub struct AudioTranscodeJob {
    /// Transcoding configuration.
    pub config: AudioTranscodeConfig,
    /// Path to the input audio file.
    pub input_path: String,
    /// Path to the output audio file.
    pub output_path: String,
}

impl AudioTranscodeJob {
    /// Creates a new audio transcode job.
    pub fn new(
        config: AudioTranscodeConfig,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            config,
            input_path: input_path.into(),
            output_path: output_path.into(),
        }
    }

    /// Estimates the output file size in bytes for this job.
    ///
    /// For lossless codecs the estimate is zero (unknown without actual encoding).
    #[must_use]
    pub fn estimated_output_size_bytes(&self) -> u64 {
        if self.config.is_lossless_output() {
            return 0;
        }
        0 // duration is not stored; see free function for duration-based estimate
    }

    /// Returns a human-readable summary of the job.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} → {} | {} → {} | {}ch @ {}Hz | {} kbps",
            self.input_path,
            self.output_path,
            self.config.input_codec,
            self.config.output_codec,
            self.config.channels,
            self.config.sample_rate,
            self.config.bitrate_kbps,
        )
    }
}

/// Estimates the output file size in bytes for the given duration and bitrate.
///
/// Returns 0 for lossless or if bitrate is zero.
#[must_use]
pub fn estimate_output_size_bytes(duration_ms: u64, bitrate_kbps: u32) -> u64 {
    if bitrate_kbps == 0 || duration_ms == 0 {
        return 0;
    }
    // size = bitrate (bits/s) * duration (s) / 8 bytes/bit
    let bits = u64::from(bitrate_kbps) * 1000 * duration_ms / 1000;
    bits / 8
}

/// Returns the conventional channel layout name for the given channel count.
#[must_use]
pub fn channel_layout_name(channels: u8) -> &'static str {
    match channels {
        1 => "mono",
        2 => "stereo",
        3 => "2.1",
        4 => "quad",
        5 => "4.1",
        6 => "5.1",
        7 => "6.1",
        8 => "7.1",
        _ => "unknown",
    }
}

/// Returns `true` if the codec name is a known lossless audio codec.
#[must_use]
pub fn is_lossless_codec(codec: &str) -> bool {
    matches!(
        codec.to_lowercase().as_str(),
        "flac"
            | "alac"
            | "pcm_s16le"
            | "pcm_s16be"
            | "pcm_s24le"
            | "pcm_s24be"
            | "pcm_s32le"
            | "pcm_s32be"
            | "pcm_f32le"
            | "pcm_f64le"
            | "wavpack"
            | "tta"
            | "mlp"
            | "truehd"
    )
}

/// Returns the typical maximum bitrate in kbps for a given codec at the given channel count.
///
/// These are approximate reference values, not hard limits.
#[must_use]
pub fn typical_max_bitrate_kbps(codec: &str, channels: u8) -> u32 {
    let per_channel: u32 = match codec.to_lowercase().as_str() {
        "opus" => 64,
        "aac" | "aac_lc" | "he_aac" => 128,
        "mp3" => 160,
        "vorbis" => 96,
        "ac3" | "eac3" => 192,
        "dts" => 256,
        _ => 128,
    };
    per_channel * u32::from(channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_stereo_256k_config() {
        let cfg = AudioTranscodeConfig::aac_stereo_256k();
        assert_eq!(cfg.output_codec, "aac");
        assert_eq!(cfg.bitrate_kbps, 256);
        assert_eq!(cfg.sample_rate, 48_000);
        assert_eq!(cfg.channels, 2);
        assert!(!cfg.is_lossless_output());
    }

    #[test]
    fn test_opus_stereo_128k_config() {
        let cfg = AudioTranscodeConfig::opus_stereo_128k();
        assert_eq!(cfg.output_codec, "opus");
        assert_eq!(cfg.bitrate_kbps, 128);
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn test_flac_lossless_config() {
        let cfg = AudioTranscodeConfig::flac_lossless();
        assert_eq!(cfg.output_codec, "flac");
        assert_eq!(cfg.bitrate_kbps, 0);
        assert!(cfg.is_lossless_output());
    }

    #[test]
    fn test_config_with_normalization() {
        let cfg = AudioTranscodeConfig::aac_stereo_256k().with_normalization(-16.0);
        assert!(cfg.normalize);
        assert!((cfg.target_lufs - -16.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_is_valid() {
        let cfg = AudioTranscodeConfig::aac_stereo_256k();
        assert!(cfg.is_valid());

        let bad = AudioTranscodeConfig::new("pcm", "", 256, 48_000, 2);
        assert!(!bad.is_valid());

        let bad_rate = AudioTranscodeConfig::new("pcm", "aac", 256, 0, 2);
        assert!(!bad_rate.is_valid());

        let bad_ch = AudioTranscodeConfig::new("pcm", "aac", 256, 48_000, 0);
        assert!(!bad_ch.is_valid());
    }

    #[test]
    fn test_estimate_output_size_bytes() {
        // 128 kbps for 10 seconds = 128000 * 10 / 8 = 160 000 bytes
        assert_eq!(estimate_output_size_bytes(10_000, 128), 160_000);
    }

    #[test]
    fn test_estimate_output_size_bytes_zero_bitrate() {
        assert_eq!(estimate_output_size_bytes(10_000, 0), 0);
    }

    #[test]
    fn test_estimate_output_size_bytes_zero_duration() {
        assert_eq!(estimate_output_size_bytes(0, 256), 0);
    }

    #[test]
    fn test_channel_layout_name() {
        assert_eq!(channel_layout_name(1), "mono");
        assert_eq!(channel_layout_name(2), "stereo");
        assert_eq!(channel_layout_name(6), "5.1");
        assert_eq!(channel_layout_name(8), "7.1");
        assert_eq!(channel_layout_name(10), "unknown");
    }

    #[test]
    fn test_is_lossless_codec_known_lossless() {
        assert!(is_lossless_codec("flac"));
        assert!(is_lossless_codec("FLAC"));
        assert!(is_lossless_codec("alac"));
        assert!(is_lossless_codec("pcm_s16le"));
        assert!(is_lossless_codec("wavpack"));
        assert!(is_lossless_codec("truehd"));
    }

    #[test]
    fn test_is_lossless_codec_known_lossy() {
        assert!(!is_lossless_codec("aac"));
        assert!(!is_lossless_codec("opus"));
        assert!(!is_lossless_codec("mp3"));
        assert!(!is_lossless_codec("vorbis"));
        assert!(!is_lossless_codec("ac3"));
    }

    #[test]
    fn test_typical_max_bitrate_stereo() {
        let opus_stereo = typical_max_bitrate_kbps("opus", 2);
        assert_eq!(opus_stereo, 128);

        let aac_51 = typical_max_bitrate_kbps("aac", 6);
        assert_eq!(aac_51, 768);
    }

    #[test]
    fn test_audio_transcode_job_summary() {
        let cfg = AudioTranscodeConfig::aac_stereo_256k();
        let job = AudioTranscodeJob::new(cfg, "input.mxf", "output.m4a");
        let summary = job.summary();
        assert!(summary.contains("input.mxf"));
        assert!(summary.contains("output.m4a"));
        assert!(summary.contains("aac"));
    }
}
