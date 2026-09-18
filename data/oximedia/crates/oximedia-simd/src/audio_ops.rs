//! SIMD-optimized audio buffer operations.
//!
//! Provides scalar fallback implementations of common audio DSP operations,
//! designed with SIMD-friendly data layouts and access patterns.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(dead_code)]

/// A contiguous buffer of 32-bit float audio samples.
#[derive(Debug, Clone, Default)]
pub struct SampleBuffer {
    /// Raw PCM samples.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
}

impl SampleBuffer {
    /// Create a new `SampleBuffer` with the given capacity.
    #[must_use]
    pub fn new(capacity: usize, sample_rate: u32, channels: u32) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            sample_rate,
            channels,
        }
    }

    /// Return the number of frames (samples per channel).
    #[must_use]
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
}

/// Blend two audio buffers together.
///
/// Result\[i\] = a\[i\] * (1 - ratio) + b\[i\] * ratio
///
/// `ratio` should be in [0.0, 1.0].  Values outside that range are accepted
/// but may produce clipping.  Both slices must have the same length.
#[must_use]
pub fn mix_buffers(a: &[f32], b: &[f32], ratio: f32) -> Vec<f32> {
    let len = a.len().min(b.len());
    let inv = 1.0_f32 - ratio;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(a[i] * inv + b[i] * ratio);
    }
    out
}

/// Multiply every sample in `samples` by `gain` in-place.
pub fn apply_gain(samples: &mut [f32], gain: f32) {
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// Normalize samples so that the peak amplitude equals 1.0.
///
/// If all samples are zero the buffer is left unchanged.
pub fn normalize(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .copied()
        .fold(0.0_f32, |acc, s| acc.max(s.abs()));
    if peak > 0.0 {
        let inv = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= inv;
        }
    }
}

/// Compute the root-mean-square level of `samples`.
///
/// Returns 0.0 for an empty slice.
#[must_use]
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().copied().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Find the peak (maximum absolute) amplitude in `samples`.
///
/// Returns 0.0 for an empty slice.
#[must_use]
pub fn find_peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .copied()
        .fold(0.0_f32, |acc, s| acc.max(s.abs()))
}

/// Remove the DC offset (mean value) from `samples` in-place.
pub fn dc_offset_remove(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean: f32 = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for s in samples.iter_mut() {
        *s -= mean;
    }
}

/// Interleave separate left and right channel buffers into a stereo buffer.
///
/// Output layout: [L0, R0, L1, R1, …].  If the slices differ in length the
/// shorter one determines the output frame count.
#[must_use]
pub fn channel_interleave(left: &[f32], right: &[f32]) -> Vec<f32> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

/// Split an interleaved stereo buffer into separate left and right channels.
///
/// Input layout must be [L0, R0, L1, R1, …].  Trailing unpaired samples are
/// ignored.
#[must_use]
pub fn channel_deinterleave(interleaved: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let frames = interleaved.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for i in 0..frames {
        left.push(interleaved[i * 2]);
        right.push(interleaved[i * 2 + 1]);
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_mix_buffers_zero_ratio() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let out = mix_buffers(&a, &b, 0.0);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 2.0));
        assert!(approx_eq(out[2], 3.0));
    }

    #[test]
    fn test_mix_buffers_one_ratio() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let out = mix_buffers(&a, &b, 1.0);
        assert!(approx_eq(out[0], 4.0));
        assert!(approx_eq(out[1], 5.0));
        assert!(approx_eq(out[2], 6.0));
    }

    #[test]
    fn test_mix_buffers_half_ratio() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let out = mix_buffers(&a, &b, 0.5);
        assert!(approx_eq(out[0], 0.5));
    }

    #[test]
    fn test_apply_gain_doubles() {
        let mut samples = vec![0.5_f32, 1.0, -0.5];
        apply_gain(&mut samples, 2.0);
        assert!(approx_eq(samples[0], 1.0));
        assert!(approx_eq(samples[1], 2.0));
        assert!(approx_eq(samples[2], -1.0));
    }

    #[test]
    fn test_apply_gain_zero() {
        let mut samples = vec![1.0_f32, -1.0, 0.5];
        apply_gain(&mut samples, 0.0);
        for s in &samples {
            assert!(approx_eq(*s, 0.0));
        }
    }

    #[test]
    fn test_normalize_basic() {
        let mut samples = vec![0.5_f32, -1.0, 0.25];
        normalize(&mut samples);
        assert!(approx_eq(find_peak(&samples), 1.0));
    }

    #[test]
    fn test_normalize_all_zeros() {
        let mut samples = vec![0.0_f32; 4];
        normalize(&mut samples); // must not panic
        for s in &samples {
            assert!(approx_eq(*s, 0.0));
        }
    }

    #[test]
    fn test_compute_rms_empty() {
        assert!(approx_eq(compute_rms(&[]), 0.0));
    }

    #[test]
    fn test_compute_rms_constant() {
        let samples = vec![1.0_f32; 4];
        assert!(approx_eq(compute_rms(&samples), 1.0));
    }

    #[test]
    fn test_find_peak() {
        let samples = vec![0.1_f32, -0.9, 0.3, -0.5];
        assert!(approx_eq(find_peak(&samples), 0.9));
    }

    #[test]
    fn test_dc_offset_remove() {
        let mut samples = vec![1.0_f32, 2.0, 3.0, 4.0]; // mean = 2.5
        dc_offset_remove(&mut samples);
        let mean_after: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(approx_eq(mean_after, 0.0));
    }

    #[test]
    fn test_channel_interleave_roundtrip() {
        let left = vec![1.0_f32, 2.0, 3.0];
        let right = vec![4.0_f32, 5.0, 6.0];
        let interleaved = channel_interleave(&left, &right);
        assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let (l2, r2) = channel_deinterleave(&interleaved);
        assert_eq!(l2, left);
        assert_eq!(r2, right);
    }

    #[test]
    fn test_channel_deinterleave_empty() {
        let (l, r) = channel_deinterleave(&[]);
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn test_sample_buffer_frames() {
        let mut buf = SampleBuffer::new(6, 44100, 2);
        buf.samples = vec![0.0; 6];
        assert_eq!(buf.frames(), 3);
    }
}
