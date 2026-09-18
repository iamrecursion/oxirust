//! High-level `TranscodePreset` enum and `TranscodeEstimator` for real-world workflows.
//!
//! `TranscodePreset` encodes industry-standard platform requirements as
//! ready-to-use `TranscodeConfig` values.  `TranscodeEstimator` provides
//! lightweight analytical estimates of output size, encoding speed and
//! perceptual quality (VMAF approximation) without actually running an
//! encoder.

use crate::{QualityMode, TranscodeConfig};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// TranscodePreset
// ─────────────────────────────────────────────────────────────────────────────

/// Common real-world transcoding presets for streaming platforms, archive, and
/// delivery workflows.
///
/// Each variant maps to a concrete `TranscodeConfig` via [`TranscodePreset::into_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscodePreset {
    // ── Streaming platforms ───────────────────────────────────────────────
    /// YouTube 1080p — AV1 4 Mbps, Opus 192 kbps.
    YouTubeHd,
    /// YouTube 4K UHD — AV1 15 Mbps, Opus 192 kbps.
    YouTubeUhd,
    /// Netflix 1080p — AV1 6 Mbps, Opus 256 kbps.
    NetflixHd,
    /// Twitch live 1080p60 — VP9 6 Mbps, Opus 160 kbps.
    TwitchStreamHd,
    // ── Archive ───────────────────────────────────────────────────────────
    /// Lossless archive — FFV1 Level 3 video, FLAC level 8 audio.
    LosslessArchive,
    /// ProRes-like high-bitrate CBR output using VP9 (edit-friendly proxy).
    ProresLt,
    // ── Delivery ─────────────────────────────────────────────────────────
    /// Broadcast HD — AV1 CBR 50 Mbps, PCM 48 kHz.
    BroadcastHd,
    /// Web delivery — VP9 2 Mbps 720p, Opus 128 kbps.
    WebDelivery,
    /// Podcast audio — Opus 64 kbps mono CBR (no video).
    PodcastAudio,
}

impl TranscodePreset {
    /// Converts this preset into a ready-to-use [`TranscodeConfig`].
    ///
    /// The returned config has all codec, bitrate, resolution and frame-rate
    /// fields pre-populated.  `input` and `output` paths are left as `None` so
    /// callers can attach them.
    #[must_use]
    pub fn into_config(self) -> TranscodeConfig {
        match self {
            Self::YouTubeHd => TranscodeConfig {
                video_codec: Some("av1".to_string()),
                audio_codec: Some("opus".to_string()),
                video_bitrate: Some(4_000_000),
                audio_bitrate: Some(192_000),
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::High),
                hw_accel: true,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },

            Self::YouTubeUhd => TranscodeConfig {
                video_codec: Some("av1".to_string()),
                audio_codec: Some("opus".to_string()),
                video_bitrate: Some(15_000_000),
                audio_bitrate: Some(192_000),
                width: Some(3840),
                height: Some(2160),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::VeryHigh),
                hw_accel: true,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },

            Self::NetflixHd => TranscodeConfig {
                video_codec: Some("av1".to_string()),
                audio_codec: Some("opus".to_string()),
                video_bitrate: Some(6_000_000),
                audio_bitrate: Some(256_000),
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((24, 1)),
                quality_mode: Some(QualityMode::VeryHigh),
                hw_accel: true,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },

            Self::TwitchStreamHd => TranscodeConfig {
                video_codec: Some("vp9".to_string()),
                audio_codec: Some("opus".to_string()),
                video_bitrate: Some(6_000_000),
                audio_bitrate: Some(160_000),
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((60, 1)),
                quality_mode: Some(QualityMode::High),
                hw_accel: true,
                preserve_metadata: false,
                ..TranscodeConfig::default()
            },

            Self::LosslessArchive => TranscodeConfig {
                video_codec: Some("ffv1".to_string()),
                audio_codec: Some("flac".to_string()),
                // Lossless — no bitrate targets
                video_bitrate: None,
                audio_bitrate: None,
                // Preserve original resolution and frame rate
                width: None,
                height: None,
                frame_rate: None,
                quality_mode: Some(QualityMode::VeryHigh),
                preserve_metadata: true,
                hw_accel: false, // FFV1 is software-only
                ..TranscodeConfig::default()
            },

            Self::ProresLt => TranscodeConfig {
                video_codec: Some("vp9".to_string()),
                audio_codec: Some("opus".to_string()),
                // High bitrate CBR approximating ProRes LT at 1080p (~45 Mbps)
                video_bitrate: Some(45_000_000),
                audio_bitrate: Some(192_000),
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::VeryHigh),
                hw_accel: true,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },

            Self::BroadcastHd => TranscodeConfig {
                video_codec: Some("av1".to_string()),
                audio_codec: Some("pcm".to_string()),
                video_bitrate: Some(50_000_000),
                audio_bitrate: Some(1_536_000), // PCM 48 kHz 16-bit stereo
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::VeryHigh),
                hw_accel: true,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },

            Self::WebDelivery => TranscodeConfig {
                video_codec: Some("vp9".to_string()),
                audio_codec: Some("opus".to_string()),
                video_bitrate: Some(2_000_000),
                audio_bitrate: Some(128_000),
                width: Some(1280),
                height: Some(720),
                frame_rate: Some((30, 1)),
                quality_mode: Some(QualityMode::Medium),
                hw_accel: true,
                preserve_metadata: false,
                ..TranscodeConfig::default()
            },

            Self::PodcastAudio => TranscodeConfig {
                video_codec: None,
                audio_codec: Some("opus".to_string()),
                video_bitrate: None,
                audio_bitrate: Some(64_000),
                width: None,
                height: None,
                frame_rate: None,
                quality_mode: Some(QualityMode::Medium),
                normalize_audio: true,
                hw_accel: false,
                preserve_metadata: true,
                ..TranscodeConfig::default()
            },
        }
    }

    /// Returns a short human-readable description of this preset.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::YouTubeHd => "YouTube 1080p HD — AV1 4 Mbps video, Opus 192 kbps audio, 30 fps",
            Self::YouTubeUhd => "YouTube 4K UHD — AV1 15 Mbps video, Opus 192 kbps audio, 30 fps",
            Self::NetflixHd => "Netflix 1080p — AV1 6 Mbps video, Opus 256 kbps audio, 24 fps",
            Self::TwitchStreamHd => {
                "Twitch live 1080p60 — VP9 6 Mbps video, Opus 160 kbps audio, 60 fps"
            }
            Self::LosslessArchive => {
                "Lossless archive — FFV1 Level 3 lossless video, FLAC level 8 lossless audio"
            }
            Self::ProresLt => {
                "ProRes LT-equivalent — VP9 CBR 45 Mbps, Opus 192 kbps, edit-friendly proxy"
            }
            Self::BroadcastHd => {
                "Broadcast HD — AV1 CBR 50 Mbps video, PCM 48 kHz uncompressed audio"
            }
            Self::WebDelivery => "Web delivery — VP9 2 Mbps 720p video, Opus 128 kbps audio",
            Self::PodcastAudio => {
                "Podcast audio-only — Opus 64 kbps mono CBR with loudness normalisation"
            }
        }
    }

    /// Returns all available presets in logical order.
    #[must_use]
    pub fn all() -> Vec<TranscodePreset> {
        vec![
            Self::YouTubeHd,
            Self::YouTubeUhd,
            Self::NetflixHd,
            Self::TwitchStreamHd,
            Self::LosslessArchive,
            Self::ProresLt,
            Self::BroadcastHd,
            Self::WebDelivery,
            Self::PodcastAudio,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TranscodeEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Analytical estimator for transcoding resource and quality metrics.
///
/// All methods are pure functions — they perform no I/O and no actual encoding.
/// Results are approximations useful for pre-flight planning (UI display, job
/// scheduling, storage budgeting) rather than precise measurements.
#[derive(Debug, Clone)]
pub struct TranscodeEstimator;

impl TranscodeEstimator {
    /// Estimates the output file size in bytes.
    ///
    /// Uses the combined bitrate of video and audio streams to compute the
    /// expected byte count for a given duration.
    ///
    /// # Arguments
    ///
    /// * `duration_secs` — Content duration in seconds.
    /// * `video_bitrate_kbps` — Video bitrate in kilobits per second (or `None`
    ///   for audio-only output).
    /// * `audio_bitrate_kbps` — Audio bitrate in kilobits per second (or `None`
    ///   for video-only output).
    #[must_use]
    pub fn estimate_size_bytes(
        duration_secs: f64,
        video_bitrate_kbps: Option<u32>,
        audio_bitrate_kbps: Option<u32>,
    ) -> u64 {
        if duration_secs <= 0.0 {
            return 0;
        }
        let total_kbps =
            u64::from(video_bitrate_kbps.unwrap_or(0)) + u64::from(audio_bitrate_kbps.unwrap_or(0));
        // total_kbps * 1000 bits/s / 8 bits/byte * duration_secs
        let bytes_per_second = total_kbps * 1000 / 8;
        (bytes_per_second as f64 * duration_secs) as u64
    }

    /// Estimates the encoding speed factor relative to real-time.
    ///
    /// A factor of `1.0` means encoding takes as long as the source duration.
    /// A factor of `0.5` means encoding takes half the source duration (2× faster
    /// than real-time), while `4.0` means encoding takes four times longer than
    /// the source (0.25× real-time).
    ///
    /// # Arguments
    ///
    /// * `codec` — Codec name (e.g. `"av1"`, `"vp9"`, `"h264"`, `"flac"`, `"opus"`).
    /// * `preset` — Encoder preset string (e.g. `"slow"`, `"fast"`, `"ultrafast"`).
    /// * `resolution_pixels` — Total pixel count (width × height; use `0` for
    ///   audio-only streams).
    #[must_use]
    pub fn estimate_speed_factor(codec: &str, preset: &str, resolution_pixels: u64) -> f32 {
        // Base speed factor per codec family (at 1080p reference)
        let base_factor: f32 = match codec {
            "av1" | "libaom-av1" | "svt-av1" => 5.0,
            "vp9" | "libvpx-vp9" => 2.5,
            "vp8" | "libvpx" => 1.2,
            "h264" | "libx264" | "h265" | "hevc" | "libx265" => 0.8,
            "flac" | "pcm" | "pcm_s16le" | "pcm_s24le" | "pcm_s32le" => 0.05,
            "opus" | "libopus" => 0.1,
            "ffv1" => 1.0,
            _ => 1.5,
        };

        // Preset multiplier: slower presets cost more CPU time
        let preset_multiplier: f32 = match preset {
            "ultrafast" | "superfast" => 0.2,
            "veryfast" => 0.4,
            "faster" | "fast" => 0.6,
            "medium" => 1.0,
            "slow" => 1.8,
            "slower" => 2.5,
            "veryslow" => 4.0,
            "placebo" => 8.0,
            _ => 1.0,
        };

        // Resolution scaling: 1080p (≈2 MP) is the reference
        let reference_pixels: f64 = 1920.0 * 1080.0;
        let resolution_scale: f32 = if resolution_pixels == 0 {
            // Audio-only — resolution does not matter
            0.0
        } else {
            let scale = (resolution_pixels as f64 / reference_pixels).sqrt();
            scale as f32
        };

        // For audio codecs the base factor already incorporates the stream
        // complexity independently of resolution
        let is_audio_only = matches!(
            codec,
            "flac" | "opus" | "libopus" | "pcm" | "pcm_s16le" | "pcm_s24le" | "pcm_s32le"
        );

        if is_audio_only {
            (base_factor * preset_multiplier).max(0.01)
        } else {
            let res_factor = resolution_scale.max(0.25); // minimum scaling
            (base_factor * preset_multiplier * res_factor).max(0.01)
        }
    }

    /// Estimates perceptual video quality as an approximate VMAF score (0–100).
    ///
    /// The estimate is derived from the bits-per-pixel metric and a
    /// logarithmic saturation curve that reflects empirically observed
    /// VMAF behaviour.  It is not a substitute for actual VMAF measurement.
    ///
    /// # Arguments
    ///
    /// * `bitrate_kbps` — Video bitrate in kilobits per second.
    /// * `resolution_pixels` — Total pixel count (width × height).
    ///
    /// # Returns
    ///
    /// An estimated VMAF score in the range `[0.0, 100.0]`.
    #[must_use]
    pub fn estimate_vmaf(bitrate_kbps: u32, resolution_pixels: u64) -> f32 {
        if resolution_pixels == 0 || bitrate_kbps == 0 {
            return 0.0;
        }

        // Bits per pixel per second — a resolution-normalised quality metric.
        // Higher bpp → higher quality.
        let bpp = (bitrate_kbps as f64 * 1000.0) / resolution_pixels as f64;

        // Empirically-derived logarithmic VMAF curve.
        // VMAF ≈ 100 * (1 − exp(−k * bpp))  with shape parameter k=6.
        // At bpp=0.1 → ~45 VMAF, bpp=0.3 → ~84, bpp=0.7 → ~99
        let k = 6.0_f64;
        let vmaf = 100.0 * (1.0 - (-k * bpp).exp());

        // Clamp to valid range
        vmaf.clamp(0.0, 100.0) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TranscodePreset tests ─────────────────────────────────────────────

    #[test]
    fn test_transcode_preset_all_returns_all_variants() {
        let all = TranscodePreset::all();
        assert_eq!(all.len(), 9, "Expected 9 preset variants");
    }

    #[test]
    fn test_youtube_hd_config() {
        let config = TranscodePreset::YouTubeHd.into_config();
        assert_eq!(config.video_codec, Some("av1".to_string()));
        assert_eq!(config.audio_codec, Some("opus".to_string()));
        assert_eq!(config.video_bitrate, Some(4_000_000));
        assert_eq!(config.audio_bitrate, Some(192_000));
        assert_eq!(config.width, Some(1920));
        assert_eq!(config.height, Some(1080));
        assert_eq!(config.frame_rate, Some((30, 1)));
    }

    #[test]
    fn test_youtube_uhd_config() {
        let config = TranscodePreset::YouTubeUhd.into_config();
        assert_eq!(config.video_codec, Some("av1".to_string()));
        assert_eq!(config.width, Some(3840));
        assert_eq!(config.height, Some(2160));
        assert_eq!(config.video_bitrate, Some(15_000_000));
    }

    #[test]
    fn test_netflix_hd_config() {
        let config = TranscodePreset::NetflixHd.into_config();
        assert_eq!(config.video_codec, Some("av1".to_string()));
        assert_eq!(config.audio_bitrate, Some(256_000));
        assert_eq!(config.frame_rate, Some((24, 1)));
    }

    #[test]
    fn test_twitch_stream_hd_config() {
        let config = TranscodePreset::TwitchStreamHd.into_config();
        assert_eq!(config.video_codec, Some("vp9".to_string()));
        assert_eq!(config.frame_rate, Some((60, 1)));
        assert_eq!(config.audio_bitrate, Some(160_000));
    }

    #[test]
    fn test_lossless_archive_config() {
        let config = TranscodePreset::LosslessArchive.into_config();
        assert_eq!(config.video_codec, Some("ffv1".to_string()));
        assert_eq!(config.audio_codec, Some("flac".to_string()));
        // Lossless — no bitrate targets
        assert!(config.video_bitrate.is_none());
        assert!(config.audio_bitrate.is_none());
        // FFV1 is software-only
        assert!(!config.hw_accel);
        assert!(config.preserve_metadata);
    }

    #[test]
    fn test_prores_lt_config() {
        let config = TranscodePreset::ProresLt.into_config();
        assert_eq!(config.video_codec, Some("vp9".to_string()));
        // High bitrate for edit-friendly proxy
        assert!(config.video_bitrate.unwrap_or(0) > 10_000_000);
    }

    #[test]
    fn test_broadcast_hd_config() {
        let config = TranscodePreset::BroadcastHd.into_config();
        assert_eq!(config.video_codec, Some("av1".to_string()));
        assert_eq!(config.audio_codec, Some("pcm".to_string()));
        assert_eq!(config.video_bitrate, Some(50_000_000));
    }

    #[test]
    fn test_web_delivery_config() {
        let config = TranscodePreset::WebDelivery.into_config();
        assert_eq!(config.video_codec, Some("vp9".to_string()));
        assert_eq!(config.width, Some(1280));
        assert_eq!(config.height, Some(720));
        assert_eq!(config.video_bitrate, Some(2_000_000));
        assert_eq!(config.audio_bitrate, Some(128_000));
    }

    #[test]
    fn test_podcast_audio_config() {
        let config = TranscodePreset::PodcastAudio.into_config();
        assert_eq!(config.audio_codec, Some("opus".to_string()));
        assert_eq!(config.audio_bitrate, Some(64_000));
        // No video
        assert!(config.video_codec.is_none());
        assert!(config.video_bitrate.is_none());
        // Loudness normalisation enabled for podcasts
        assert!(config.normalize_audio);
    }

    #[test]
    fn test_description_not_empty() {
        for preset in TranscodePreset::all() {
            let desc = preset.description();
            assert!(
                !desc.is_empty(),
                "Description for {preset:?} should not be empty"
            );
        }
    }

    #[test]
    fn test_all_presets_unique_descriptions() {
        let descs: Vec<&'static str> = TranscodePreset::all()
            .iter()
            .map(|p| p.description())
            .collect();
        let unique: std::collections::HashSet<&&str> = descs.iter().collect();
        assert_eq!(
            unique.len(),
            descs.len(),
            "All preset descriptions should be unique"
        );
    }

    // ── TranscodeEstimator tests ──────────────────────────────────────────

    #[test]
    fn test_estimator_size_bytes_basic() {
        // (5000 + 192) kbps * 1000 / 8 bytes/s * 60 s
        let size = TranscodeEstimator::estimate_size_bytes(60.0, Some(5_000), Some(192));
        assert!(size > 0, "Size should be positive");
        assert!(
            size < 200_000_000,
            "Sanity check: should be < 200 MB for 60s"
        );
    }

    #[test]
    fn test_estimator_size_bytes_zero_duration() {
        let size = TranscodeEstimator::estimate_size_bytes(0.0, Some(5_000), Some(192));
        assert_eq!(size, 0);
    }

    #[test]
    fn test_estimator_size_bytes_negative_duration() {
        let size = TranscodeEstimator::estimate_size_bytes(-10.0, Some(5_000), Some(192));
        assert_eq!(size, 0);
    }

    #[test]
    fn test_estimator_size_bytes_audio_only() {
        let size = TranscodeEstimator::estimate_size_bytes(120.0, None, Some(128));
        // 128 kbps * 1000 / 8 * 120 = 1_920_000
        assert_eq!(size, 1_920_000);
    }

    #[test]
    fn test_estimator_size_bytes_video_only() {
        let size = TranscodeEstimator::estimate_size_bytes(10.0, Some(4_000), None);
        // 4000 kbps * 1000 / 8 * 10 = 5_000_000
        assert_eq!(size, 5_000_000);
    }

    #[test]
    fn test_estimator_speed_factor_av1_slow() {
        let factor = TranscodeEstimator::estimate_speed_factor("av1", "slow", 1920 * 1080);
        assert!(
            factor > 1.0,
            "AV1 slow at 1080p should be slower than real-time"
        );
    }

    #[test]
    fn test_estimator_speed_factor_h264_fast() {
        let factor = TranscodeEstimator::estimate_speed_factor("h264", "fast", 1280 * 720);
        assert!(factor > 0.0, "Speed factor should be positive");
        // h264 fast should be reasonably fast
        assert!(factor < 4.0, "h264 fast should not be extremely slow");
    }

    #[test]
    fn test_estimator_speed_factor_audio_codec() {
        let factor = TranscodeEstimator::estimate_speed_factor("opus", "medium", 0);
        assert!(factor > 0.0);
        // Audio encoding is much faster than real-time
        assert!(
            factor < 1.0,
            "Opus encoding should be faster than real-time"
        );
    }

    #[test]
    fn test_estimator_speed_factor_4k_slower() {
        let factor_1080p = TranscodeEstimator::estimate_speed_factor("av1", "medium", 1920 * 1080);
        let factor_4k = TranscodeEstimator::estimate_speed_factor("av1", "medium", 3840 * 2160);
        assert!(
            factor_4k > factor_1080p,
            "4K should take longer to encode than 1080p"
        );
    }

    #[test]
    fn test_estimator_vmaf_high_bitrate() {
        let vmaf = TranscodeEstimator::estimate_vmaf(10_000, 1920 * 1080);
        assert!(vmaf > 0.0, "VMAF should be positive");
        assert!(vmaf <= 100.0, "VMAF should not exceed 100");
    }

    #[test]
    fn test_estimator_vmaf_low_bitrate() {
        let vmaf_low = TranscodeEstimator::estimate_vmaf(500, 1920 * 1080);
        let vmaf_high = TranscodeEstimator::estimate_vmaf(8_000, 1920 * 1080);
        assert!(
            vmaf_low < vmaf_high,
            "Higher bitrate should produce higher VMAF"
        );
    }

    #[test]
    fn test_estimator_vmaf_clamped_at_100() {
        let vmaf = TranscodeEstimator::estimate_vmaf(100_000, 320 * 240);
        assert!(vmaf <= 100.0, "VMAF must be clamped to 100.0");
    }

    #[test]
    fn test_estimator_vmaf_zero_resolution() {
        let vmaf = TranscodeEstimator::estimate_vmaf(5_000, 0);
        assert_eq!(vmaf, 0.0, "Zero resolution should return VMAF 0");
    }

    #[test]
    fn test_estimator_vmaf_zero_bitrate() {
        let vmaf = TranscodeEstimator::estimate_vmaf(0, 1920 * 1080);
        assert_eq!(vmaf, 0.0, "Zero bitrate should return VMAF 0");
    }
}
