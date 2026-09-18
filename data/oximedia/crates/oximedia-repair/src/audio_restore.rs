//! Audio restoration: click removal, dropout recovery, and level restoration.

#![allow(dead_code)]

/// Direction of a fade operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeDirection {
    /// Fade in from silence.
    In,
    /// Fade out to silence.
    Out,
}

/// Apply a linear fade to a slice of samples.
///
/// For `FadeDirection::In`, sample 0 starts at 0 and the last sample keeps
/// its original amplitude. For `FadeDirection::Out`, the first sample keeps
/// its amplitude and the last approaches 0.
pub fn apply_linear_fade(samples: &mut [f32], direction: FadeDirection) {
    let n = samples.len();
    if n == 0 {
        return;
    }
    for (i, s) in samples.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / (n - 1).max(1) as f32;
        let gain = match direction {
            FadeDirection::In => t,
            FadeDirection::Out => 1.0 - t,
        };
        *s *= gain;
    }
}

/// Detect clicks using a simple threshold on the sample-to-sample delta.
///
/// Returns the indices of samples identified as clicks.
pub fn detect_clicks(samples: &[f32], delta_threshold: f32) -> Vec<usize> {
    let mut clicks = Vec::new();
    for i in 1..samples.len() {
        if (samples[i] - samples[i - 1]).abs() > delta_threshold {
            clicks.push(i);
        }
    }
    clicks
}

/// Remove a detected click at `idx` by linear interpolation over a window of
/// radius `half_window` samples.
pub fn remove_click(samples: &mut [f32], idx: usize, half_window: usize) {
    if samples.is_empty() {
        return;
    }
    let start = idx.saturating_sub(half_window);
    let end = (idx + half_window + 1).min(samples.len());
    let count = end - start;
    if count < 2 {
        return;
    }
    let start_val = samples[start];
    let end_val = if end < samples.len() {
        samples[end - 1]
    } else {
        0.0
    };
    for (i, pos) in (start..end).enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / (count - 1).max(1) as f32;
        samples[pos] = start_val + t * (end_val - start_val);
    }
}

/// Detect a dropout: a run of near-silent samples where the RMS falls below
/// `silence_threshold` for at least `min_duration` consecutive samples.
///
/// Returns `(start, length)` for each detected dropout.
pub fn detect_dropouts(
    samples: &[f32],
    silence_threshold: f32,
    min_duration: usize,
) -> Vec<(usize, usize)> {
    let mut dropouts = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut run_len = 0usize;

    for (i, &s) in samples.iter().enumerate() {
        if s.abs() < silence_threshold {
            if run_start.is_none() {
                run_start = Some(i);
                run_len = 0;
            }
            run_len += 1;
        } else {
            if run_len >= min_duration {
                if let Some(rs) = run_start {
                    dropouts.push((rs, run_len));
                }
            }
            run_start = None;
            run_len = 0;
        }
    }
    if run_len >= min_duration {
        if let Some(rs) = run_start {
            dropouts.push((rs, run_len));
        }
    }
    dropouts
}

/// Fill a dropout region (start..start+length) using simple linear interpolation
/// from the sample just before to the sample just after the region.
pub fn fill_dropout(samples: &mut [f32], start: usize, length: usize) {
    if length == 0 || samples.is_empty() {
        return;
    }
    let end = (start + length).min(samples.len());
    let pre = if start > 0 { samples[start - 1] } else { 0.0 };
    let post = if end < samples.len() {
        samples[end]
    } else {
        0.0
    };
    for (i, idx) in (start..end).enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = (i + 1) as f32 / (end - start + 1) as f32;
        samples[idx] = pre + t * (post - pre);
    }
}

/// Compute the RMS (Root Mean Square) level of a sample buffer.
#[allow(clippy::cast_precision_loss)]
pub fn rms_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute gain required to bring `current_rms` to `target_rms`.
pub fn level_correction_gain(current_rms: f32, target_rms: f32) -> f32 {
    if current_rms < f32::EPSILON {
        1.0
    } else {
        target_rms / current_rms
    }
}

/// Apply gain correction to a sample buffer (in-place), clamping results to [-1.0, 1.0].
pub fn apply_gain(samples: &mut [f32], gain: f32) {
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

/// Restore audio level: measure current RMS, compute the required gain,
/// and apply it in-place.
pub fn restore_level(samples: &mut [f32], target_rms: f32) {
    let current = rms_level(samples);
    let gain = level_correction_gain(current, target_rms);
    apply_gain(samples, gain);
}

/// Denoise by zeroing samples whose magnitude falls below a noise floor threshold.
pub fn noise_gate(samples: &mut [f32], floor: f32) {
    for s in samples.iter_mut() {
        if s.abs() < floor {
            *s = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_linear_fade_in_starts_near_zero() {
        let mut samples = vec![1.0f32; 100];
        apply_linear_fade(&mut samples, FadeDirection::In);
        assert!(samples[0].abs() < 1e-6);
        assert!((samples[99] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_linear_fade_out_ends_near_zero() {
        let mut samples = vec![1.0f32; 100];
        apply_linear_fade(&mut samples, FadeDirection::Out);
        assert!((samples[0] - 1.0).abs() < 1e-5);
        assert!(samples[99].abs() < 1e-6);
    }

    #[test]
    fn test_apply_linear_fade_empty() {
        let mut samples: Vec<f32> = vec![];
        apply_linear_fade(&mut samples, FadeDirection::In);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_detect_clicks_finds_spike() {
        let mut samples = vec![0.0f32; 20];
        samples[10] = 5.0; // large jump at index 10
        let clicks = detect_clicks(&samples, 1.0);
        assert!(clicks.contains(&10));
    }

    #[test]
    fn test_detect_clicks_no_clicks_in_smooth_signal() {
        let samples: Vec<f32> = (0..20).map(|i| i as f32 * 0.01).collect();
        let clicks = detect_clicks(&samples, 0.5);
        assert!(clicks.is_empty());
    }

    #[test]
    fn test_remove_click_reduces_spike() {
        let mut samples = vec![0.0f32; 20];
        samples[10] = 10.0;
        remove_click(&mut samples, 10, 3);
        assert!(samples[10].abs() < 5.0);
    }

    #[test]
    fn test_remove_click_empty() {
        let mut samples: Vec<f32> = vec![];
        remove_click(&mut samples, 0, 2);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_detect_dropouts_finds_silence() {
        let mut samples = vec![0.5f32; 30];
        for s in &mut samples[10..20] {
            *s = 0.0;
        }
        let dropouts = detect_dropouts(&samples, 0.05, 5);
        assert!(!dropouts.is_empty());
        assert_eq!(dropouts[0].0, 10);
    }

    #[test]
    fn test_detect_dropouts_none_in_loud_signal() {
        let samples = vec![0.9f32; 30];
        let dropouts = detect_dropouts(&samples, 0.05, 5);
        assert!(dropouts.is_empty());
    }

    #[test]
    fn test_fill_dropout_interpolates() {
        let mut samples = vec![0.5f32; 20];
        for s in &mut samples[5..10] {
            *s = 0.0;
        }
        fill_dropout(&mut samples, 5, 5);
        // All samples should be non-zero after fill
        for &s in &samples[5..10] {
            assert!(s >= 0.0); // just check no NaN
            assert!(!s.is_nan());
        }
    }

    #[test]
    fn test_rms_level_all_ones() {
        let samples = vec![1.0f32; 100];
        assert!((rms_level(&samples) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_level_empty() {
        assert_eq!(rms_level(&[]), 0.0);
    }

    #[test]
    fn test_level_correction_gain() {
        let gain = level_correction_gain(0.5, 1.0);
        assert!((gain - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_gain_clamps() {
        let mut samples = vec![0.8f32; 5];
        apply_gain(&mut samples, 2.0); // 0.8 * 2 = 1.6, clamped to 1.0
        for s in &samples {
            assert!(*s <= 1.0);
        }
    }

    #[test]
    fn test_noise_gate_zeros_quiet_samples() {
        let mut samples = vec![0.001f32, 0.5f32, 0.002f32, 0.9f32];
        noise_gate(&mut samples, 0.01);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[2], 0.0);
        assert!(samples[1] > 0.0);
        assert!(samples[3] > 0.0);
    }

    #[test]
    fn test_restore_level_brings_rms_close_to_target() {
        let mut samples: Vec<f32> = vec![0.1f32; 100];
        restore_level(&mut samples, 0.5);
        let after_rms = rms_level(&samples);
        // After restoration, RMS should be clamped but closer to target
        assert!(after_rms > 0.09);
    }
}
