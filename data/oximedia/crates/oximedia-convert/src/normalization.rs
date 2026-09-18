#![allow(dead_code)]
//! Pre-conversion normalization for media files.
//!
//! This module provides normalization routines that prepare media for conversion:
//! - Audio level normalization (peak and RMS-based)
//! - Frame rate normalization via frame duplication/dropping
//! - Resolution normalization with rounding to even dimensions
//! - Channel layout normalization (upmix, downmix)

/// Audio normalization mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioNormMode {
    /// Normalize to a peak amplitude target.
    Peak {
        /// Target peak in dBFS.
        target_dbfs: f64,
    },
    /// Normalize to an RMS target.
    Rms {
        /// Target RMS in dBFS.
        target_dbfs: f64,
    },
    /// Apply EBU R128 loudness normalization.
    Loudness {
        /// Target integrated loudness in LUFS.
        target_lufs: f64,
    },
}

/// Result of audio normalization analysis.
#[derive(Debug, Clone, Copy)]
pub struct AudioNormResult {
    /// Measured peak in dBFS.
    pub measured_peak_dbfs: f64,
    /// Measured RMS in dBFS.
    pub measured_rms_dbfs: f64,
    /// Gain to apply in dB.
    pub gain_db: f64,
    /// Whether clipping would occur without limiting.
    pub would_clip: bool,
}

/// Analyze audio samples and compute the normalization gain.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_audio_norm(samples: &[f32], mode: AudioNormMode) -> AudioNormResult {
    if samples.is_empty() {
        return AudioNormResult {
            measured_peak_dbfs: -300.0,
            measured_rms_dbfs: -300.0,
            gain_db: 0.0,
            would_clip: false,
        };
    }

    let peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    let peak_f64 = f64::from(peak);
    let peak_dbfs = if peak_f64 > 1e-15 {
        20.0 * peak_f64.log10()
    } else {
        -300.0
    };

    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt();
    let rms_dbfs = if rms > 1e-15 {
        20.0 * rms.log10()
    } else {
        -300.0
    };

    let gain_db = match mode {
        AudioNormMode::Peak { target_dbfs } => target_dbfs - peak_dbfs,
        AudioNormMode::Rms { target_dbfs } => target_dbfs - rms_dbfs,
        AudioNormMode::Loudness { target_lufs } => {
            // Simplified: use RMS as a proxy for loudness
            target_lufs - rms_dbfs
        }
    };

    let would_clip = peak_dbfs + gain_db > 0.0;

    AudioNormResult {
        measured_peak_dbfs: peak_dbfs,
        measured_rms_dbfs: rms_dbfs,
        gain_db,
        would_clip,
    }
}

/// Apply a gain in dB to audio samples in place.
#[allow(clippy::cast_precision_loss)]
pub fn apply_gain(samples: &mut [f32], gain_db: f64) {
    let linear = 10.0_f64.powf(gain_db / 20.0);
    #[allow(clippy::cast_possible_truncation)]
    let linear_f32 = linear as f32;
    for s in samples.iter_mut() {
        *s *= linear_f32;
    }
}

/// Apply gain with a hard limiter at 0 dBFS.
#[allow(clippy::cast_precision_loss)]
pub fn apply_gain_with_limit(samples: &mut [f32], gain_db: f64) {
    let linear = 10.0_f64.powf(gain_db / 20.0);
    #[allow(clippy::cast_possible_truncation)]
    let linear_f32 = linear as f32;
    for s in samples.iter_mut() {
        *s = (*s * linear_f32).clamp(-1.0, 1.0);
    }
}

// ── Frame rate normalization ──────────────────────────────────────────

/// Strategy for frame rate conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRateStrategy {
    /// Drop/duplicate frames (nearest neighbor).
    NearestNeighbor,
    /// Blend adjacent frames for smoother output.
    Blend,
}

/// Compute the frame index mapping from source to target frame rate.
///
/// Returns a list of (`target_frame`, `source_frame`) pairs.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn compute_frame_mapping(
    num_source_frames: usize,
    source_fps: f64,
    target_fps: f64,
) -> Vec<(usize, usize)> {
    if source_fps <= 0.0 || target_fps <= 0.0 || num_source_frames == 0 {
        return Vec::new();
    }
    let duration = num_source_frames as f64 / source_fps;
    let num_target = (duration * target_fps).round() as usize;

    (0..num_target)
        .map(|t| {
            let target_time = t as f64 / target_fps;
            let src_frame = (target_time * source_fps).round() as usize;
            (t, src_frame.min(num_source_frames.saturating_sub(1)))
        })
        .collect()
}

/// Compute blend weights for frame rate conversion with blending.
///
/// For each target frame, returns `(src_frame_a, src_frame_b, weight_b)`.
/// Weight for `a` is `1.0 - weight_b`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn compute_blend_mapping(
    num_source_frames: usize,
    source_fps: f64,
    target_fps: f64,
) -> Vec<(usize, usize, f64)> {
    if source_fps <= 0.0 || target_fps <= 0.0 || num_source_frames == 0 {
        return Vec::new();
    }
    let duration = num_source_frames as f64 / source_fps;
    let num_target = (duration * target_fps).round() as usize;
    let max_src = num_source_frames.saturating_sub(1);

    (0..num_target)
        .map(|t| {
            let target_time = t as f64 / target_fps;
            let src_pos = target_time * source_fps;
            let frame_a = (src_pos.floor() as usize).min(max_src);
            let frame_b = (frame_a + 1).min(max_src);
            let weight_b = src_pos - src_pos.floor();
            (frame_a, frame_b, weight_b)
        })
        .collect()
}

// ── Resolution normalization ──────────────────────────────────────────

/// Round a resolution to even dimensions (required by most codecs).
#[must_use]
pub fn round_to_even(width: u32, height: u32) -> (u32, u32) {
    let w = if width % 2 != 0 { width + 1 } else { width };
    let h = if height % 2 != 0 { height + 1 } else { height };
    (w, h)
}

/// Round a resolution to a multiple of a given alignment.
#[must_use]
pub fn round_to_alignment(width: u32, height: u32, alignment: u32) -> (u32, u32) {
    if alignment == 0 {
        return (width, height);
    }
    let w = width.div_ceil(alignment) * alignment;
    let h = height.div_ceil(alignment) * alignment;
    (w, h)
}

// ── Channel layout normalization ──────────────────────────────────────

/// Channel layout for audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels).
    Stereo,
    /// 5.1 surround (6 channels).
    Surround5_1,
    /// 7.1 surround (8 channels).
    Surround7_1,
}

impl ChannelLayout {
    /// Return the number of channels.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround5_1 => 6,
            Self::Surround7_1 => 8,
        }
    }
}

/// Downmix stereo samples to mono.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn downmix_stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    stereo
        .chunks(2)
        .map(|ch| {
            if ch.len() == 2 {
                (ch[0] + ch[1]) * 0.5
            } else {
                ch[0]
            }
        })
        .collect()
}

/// Upmix mono samples to stereo (duplicate to both channels).
#[must_use]
pub fn upmix_mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &s in mono {
        stereo.push(s);
        stereo.push(s);
    }
    stereo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_peak_norm() {
        let samples = vec![0.5_f32, -0.5, 0.25, -0.25];
        let result = analyze_audio_norm(&samples, AudioNormMode::Peak { target_dbfs: 0.0 });
        // peak is 0.5 => ~-6 dBFS, so gain should be ~6 dB
        assert!((result.gain_db - 6.02).abs() < 0.1);
    }

    #[test]
    fn test_analyze_rms_norm() {
        let samples = vec![0.5_f32; 1000];
        let result = analyze_audio_norm(&samples, AudioNormMode::Rms { target_dbfs: 0.0 });
        // RMS of 0.5 => ~-6 dBFS
        assert!((result.gain_db - 6.02).abs() < 0.1);
    }

    #[test]
    fn test_analyze_empty() {
        let result = analyze_audio_norm(&[], AudioNormMode::Peak { target_dbfs: 0.0 });
        assert_eq!(result.gain_db, 0.0);
    }

    #[test]
    fn test_would_clip() {
        let samples = vec![0.9_f32, -0.9];
        let result = analyze_audio_norm(&samples, AudioNormMode::Peak { target_dbfs: 0.0 });
        // After normalization peak would be 0 dBFS, but it won't exceed it
        assert!(!result.would_clip);
        // Now try target above 0
        let result2 = analyze_audio_norm(&samples, AudioNormMode::Peak { target_dbfs: 6.0 });
        assert!(result2.would_clip);
    }

    #[test]
    fn test_apply_gain() {
        let mut samples = vec![0.5_f32, -0.5];
        apply_gain(&mut samples, 6.0);
        // 6 dB ≈ ×2
        assert!((samples[0] - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_apply_gain_with_limit() {
        let mut samples = vec![0.8_f32, -0.8];
        apply_gain_with_limit(&mut samples, 6.0);
        // Gain would push to ~1.6 but limiter clamps to 1.0
        assert!((samples[0] - 1.0).abs() < 0.001);
        assert!((samples[1] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_frame_mapping_same_fps() {
        let mapping = compute_frame_mapping(100, 30.0, 30.0);
        assert_eq!(mapping.len(), 100);
        for (i, (t, s)) in mapping.iter().enumerate() {
            assert_eq!(*t, i);
            assert_eq!(*s, i);
        }
    }

    #[test]
    fn test_frame_mapping_double_fps() {
        let mapping = compute_frame_mapping(30, 30.0, 60.0);
        assert_eq!(mapping.len(), 60);
    }

    #[test]
    fn test_frame_mapping_empty() {
        assert!(compute_frame_mapping(0, 30.0, 60.0).is_empty());
        assert!(compute_frame_mapping(100, 0.0, 60.0).is_empty());
    }

    #[test]
    fn test_blend_mapping() {
        let blends = compute_blend_mapping(30, 30.0, 60.0);
        assert_eq!(blends.len(), 60);
        for &(a, b, w) in &blends {
            assert!(a <= 29);
            assert!(b <= 29);
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_round_to_even() {
        assert_eq!(round_to_even(1920, 1080), (1920, 1080));
        assert_eq!(round_to_even(1921, 1081), (1922, 1082));
    }

    #[test]
    fn test_round_to_alignment() {
        assert_eq!(round_to_alignment(1920, 1080, 16), (1920, 1088));
        assert_eq!(round_to_alignment(100, 100, 8), (104, 104));
    }

    #[test]
    fn test_round_to_alignment_zero() {
        assert_eq!(round_to_alignment(100, 100, 0), (100, 100));
    }

    #[test]
    fn test_downmix_stereo_to_mono() {
        let stereo = vec![0.6_f32, 0.4, 1.0, 0.0];
        let mono = downmix_stereo_to_mono(&stereo);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_upmix_mono_to_stereo() {
        let mono = vec![0.5_f32, -0.3];
        let stereo = upmix_mono_to_stereo(&mono);
        assert_eq!(stereo.len(), 4);
        assert_eq!(stereo[0], 0.5);
        assert_eq!(stereo[1], 0.5);
    }

    #[test]
    fn test_channel_layout_count() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround5_1.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround7_1.channel_count(), 8);
    }
}
