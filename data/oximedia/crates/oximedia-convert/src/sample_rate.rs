// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio sample-rate conversion utilities.
//!
//! Provides nearest-neighbour, linear, and sinc-based resampling,
//! along with a Hamming-windowed sinc kernel generator and a
//! mono ↔ stereo channel converter.

use std::f32::consts::PI;

/// Quality preset for resampling operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleQuality {
    /// Fastest: no interpolation between source samples.
    Low,
    /// Good quality: linear interpolation between adjacent samples.
    Medium,
    /// Best quality: sinc-filter resampling with a Hamming window.
    High,
}

impl ResampleQuality {
    /// Return the sinc-filter length (number of taps) used for this quality.
    #[must_use]
    pub const fn filter_length(self) -> usize {
        match self {
            Self::Low => 1,
            Self::Medium => 8,
            Self::High => 64,
        }
    }
}

/// Resample `src` from `src_rate` to `dst_rate` using nearest-neighbour
/// (no interpolation) — fastest but lowest quality.
///
/// Returns an empty `Vec` if either rate is zero.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn nearest_resample(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == 0 || dst_rate == 0 || src.is_empty() {
        return Vec::new();
    }

    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    let dst_len = ((src.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_idx = ((i as f64 * ratio) as usize).min(src.len() - 1);
        out.push(src[src_idx]);
    }

    out
}

/// Resample `src` from `src_rate` to `dst_rate` using linear interpolation.
///
/// Returns an empty `Vec` if either rate is zero.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn linear_resample(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == 0 || dst_rate == 0 || src.is_empty() {
        return Vec::new();
    }

    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    let dst_len = ((src.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let pos = i as f64 * ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(src.len() - 1);
        let frac = (pos - pos.floor()) as f32;
        out.push(src[lo] * (1.0 - frac) + src[hi] * frac);
    }

    out
}

/// Compute a normalised sinc kernel of `length` taps with a Hamming window
/// and a normalised cut-off frequency of `cutoff` (0.0..0.5 of Nyquist).
///
/// Returns an empty `Vec` if `length` is zero.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_sinc_kernel(cutoff: f32, length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }

    let mut kernel = Vec::with_capacity(length);
    let half = (length as f32 - 1.0) / 2.0;

    for i in 0..length {
        let n = i as f32 - half;

        // Sinc function
        let sinc = if n.abs() < 1e-7 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * n).sin() / (PI * n)
        };

        // Hamming window
        let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (length as f32 - 1.0)).cos();
        kernel.push(sinc * window);
    }

    // Normalise so the DC gain is 1
    let sum: f32 = kernel.iter().sum();
    if sum.abs() > 1e-10 {
        for k in &mut kernel {
            *k /= sum;
        }
    }

    kernel
}

/// Convert between channel counts by mixing or duplicating channels.
///
/// Supported conversions:
/// - mono → stereo: duplicate the single channel to both outputs
/// - stereo → mono: average the two channels
/// - same channel count: pass-through
///
/// Any other combination returns a zeroed buffer of the expected length.
///
/// Returns an empty `Vec` if `src` is empty or either channel count is zero.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn channel_convert(src: &[f32], src_channels: u8, dst_channels: u8) -> Vec<f32> {
    if src.is_empty() || src_channels == 0 || dst_channels == 0 {
        return Vec::new();
    }

    if src_channels == dst_channels {
        return src.to_vec();
    }

    let frame_count = src.len() / src_channels as usize;

    match (src_channels, dst_channels) {
        (1, 2) => {
            // mono → stereo
            let mut out = Vec::with_capacity(frame_count * 2);
            for &s in src {
                out.push(s);
                out.push(s);
            }
            out
        }
        (2, 1) => {
            // stereo → mono
            let mut out = Vec::with_capacity(frame_count);
            for chunk in src.chunks_exact(2) {
                out.push((chunk[0] + chunk[1]) * 0.5);
            }
            out
        }
        _ => {
            // Unsupported: return zeroed buffer
            vec![0.0f32; frame_count * dst_channels as usize]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ResampleQuality ---

    #[test]
    fn test_resample_quality_filter_lengths() {
        assert_eq!(ResampleQuality::Low.filter_length(), 1);
        assert_eq!(ResampleQuality::Medium.filter_length(), 8);
        assert_eq!(ResampleQuality::High.filter_length(), 64);
    }

    // --- nearest_resample ---

    #[test]
    fn test_nearest_resample_same_rate() {
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let out = nearest_resample(&src, 44100, 44100);
        assert_eq!(out.len(), src.len());
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_resample_upsample_doubles_approx() {
        let src: Vec<f32> = vec![1.0, 2.0];
        let out = nearest_resample(&src, 22050, 44100);
        assert!(!out.is_empty());
        assert!(out.len() >= 2);
    }

    #[test]
    fn test_nearest_resample_zero_rate_returns_empty() {
        let src = vec![1.0f32, 2.0];
        assert!(nearest_resample(&src, 0, 44100).is_empty());
        assert!(nearest_resample(&src, 44100, 0).is_empty());
    }

    #[test]
    fn test_nearest_resample_empty_src() {
        assert!(nearest_resample(&[], 44100, 22050).is_empty());
    }

    // --- linear_resample ---

    #[test]
    fn test_linear_resample_same_rate() {
        let src: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0];
        let out = linear_resample(&src, 44100, 44100);
        assert_eq!(out.len(), src.len());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_resample_upsample() {
        let src: Vec<f32> = vec![0.0, 1.0];
        let out = linear_resample(&src, 1, 2);
        assert!(out.len() >= 2);
    }

    #[test]
    fn test_linear_resample_zero_rate_returns_empty() {
        let src = vec![1.0f32];
        assert!(linear_resample(&src, 0, 44100).is_empty());
        assert!(linear_resample(&src, 44100, 0).is_empty());
    }

    #[test]
    fn test_linear_resample_interpolation_midpoint() {
        // Two samples: 0.0 and 2.0; resampled at 2x should give ~1.0 in between
        let src: Vec<f32> = vec![0.0, 2.0];
        let out = linear_resample(&src, 1, 2);
        // First output should be 0.0; second should be ~1.0
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 0.2, "out[1]={}", out[1]);
    }

    // --- compute_sinc_kernel ---

    #[test]
    fn test_sinc_kernel_length() {
        let k = compute_sinc_kernel(0.25, 16);
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn test_sinc_kernel_zero_length() {
        assert!(compute_sinc_kernel(0.25, 0).is_empty());
    }

    #[test]
    fn test_sinc_kernel_normalised() {
        let k = compute_sinc_kernel(0.25, 64);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    #[test]
    fn test_sinc_kernel_symmetric() {
        let k = compute_sinc_kernel(0.25, 17);
        let len = k.len();
        for i in 0..len / 2 {
            assert!(
                (k[i] - k[len - 1 - i]).abs() < 1e-5,
                "asymmetry at {i}: {} vs {}",
                k[i],
                k[len - 1 - i]
            );
        }
    }

    // --- channel_convert ---

    #[test]
    fn test_channel_convert_passthrough() {
        let src = vec![1.0f32, -1.0, 0.5, -0.5];
        let out = channel_convert(&src, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn test_channel_convert_mono_to_stereo() {
        let src = vec![1.0f32, 0.5, -1.0];
        let out = channel_convert(&src, 1, 2);
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_channel_convert_stereo_to_mono() {
        let src = vec![1.0f32, -1.0, 0.5, 0.5];
        let out = channel_convert(&src, 2, 1);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6); // (1 + -1) / 2
        assert!((out[1] - 0.5).abs() < 1e-6); // (0.5 + 0.5) / 2
    }

    #[test]
    fn test_channel_convert_empty_returns_empty() {
        let out = channel_convert(&[], 1, 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_channel_convert_zero_channels_returns_empty() {
        let src = vec![1.0f32];
        assert!(channel_convert(&src, 0, 1).is_empty());
        assert!(channel_convert(&src, 1, 0).is_empty());
    }
}
