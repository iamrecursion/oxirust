#![allow(dead_code)]
//! Stereo field repair and mid/side balance correction.
//!
//! This module provides tools for diagnosing and repairing stereo imaging
//! problems commonly found in archival recordings, including:
//!
//! - **Channel imbalance** -- level difference between left and right
//! - **Mid/Side decomposition** -- separate center content from spatial content
//! - **Stereo width adjustment** -- widen or narrow the stereo image
//! - **Cross-correlation analysis** -- detect phase issues and mono compatibility
//! - **Channel delay compensation** -- fix timing mismatches between L and R

/// Result of stereo field analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoFieldAnalysis {
    /// Level imbalance in dB (positive = right louder).
    pub imbalance_db: f64,
    /// Cross-correlation coefficient (-1.0 to 1.0).
    pub correlation: f64,
    /// Estimated stereo width (0.0 = mono, 1.0 = full stereo, >1.0 = out of phase).
    pub width: f64,
    /// Estimated delay between channels in samples (positive = right leads).
    pub delay_samples: i32,
    /// RMS level of the mid (center) signal.
    pub mid_rms: f64,
    /// RMS level of the side (spatial) signal.
    pub side_rms: f64,
}

/// Analyze the stereo field of a left/right pair.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_stereo_field(left: &[f32], right: &[f32]) -> StereoFieldAnalysis {
    let len = left.len().min(right.len());
    if len == 0 {
        return StereoFieldAnalysis {
            imbalance_db: 0.0,
            correlation: 0.0,
            width: 0.0,
            delay_samples: 0,
            mid_rms: 0.0,
            side_rms: 0.0,
        };
    }

    // RMS levels
    let left_rms = rms(&left[..len]);
    let right_rms = rms(&right[..len]);

    // Imbalance in dB
    let imbalance_db = if left_rms > 1e-10 && right_rms > 1e-10 {
        20.0 * (right_rms / left_rms).log10()
    } else {
        0.0
    };

    // Cross-correlation
    let correlation = cross_correlation(&left[..len], &right[..len]);

    // Mid/Side
    let (mid, side) = encode_mid_side(&left[..len], &right[..len]);
    let mid_rms = rms(&mid);
    let side_rms = rms(&side);

    // Width estimate
    let width = if mid_rms > 1e-10 {
        side_rms / mid_rms
    } else {
        0.0
    };

    // Delay estimation
    let delay_samples = estimate_delay(&left[..len], &right[..len], 64);

    StereoFieldAnalysis {
        imbalance_db,
        correlation,
        width,
        delay_samples,
        mid_rms,
        side_rms,
    }
}

/// Encode left/right to mid/side.
#[allow(clippy::cast_precision_loss)]
pub fn encode_mid_side(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = left.len().min(right.len());
    let mut mid = Vec::with_capacity(len);
    let mut side = Vec::with_capacity(len);
    for i in 0..len {
        mid.push((left[i] + right[i]) * 0.5);
        side.push((left[i] - right[i]) * 0.5);
    }
    (mid, side)
}

/// Decode mid/side back to left/right.
pub fn decode_mid_side(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = mid.len().min(side.len());
    let mut left = Vec::with_capacity(len);
    let mut right = Vec::with_capacity(len);
    for i in 0..len {
        left.push(mid[i] + side[i]);
        right.push(mid[i] - side[i]);
    }
    (left, right)
}

/// Adjust stereo width using mid/side processing.
///
/// `width` of 0.0 = mono, 1.0 = original, 2.0 = double-wide.
pub fn adjust_stereo_width(left: &[f32], right: &[f32], width: f32) -> (Vec<f32>, Vec<f32>) {
    let (mid, side) = encode_mid_side(left, right);
    let scaled_side: Vec<f32> = side.iter().map(|&s| s * width).collect();
    decode_mid_side(&mid, &scaled_side)
}

/// Correct channel level imbalance by applying gain to bring channels to equal RMS.
#[allow(clippy::cast_precision_loss)]
pub fn correct_imbalance(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = left.len().min(right.len());
    let left_rms = rms(&left[..len]);
    let right_rms = rms(&right[..len]);

    if left_rms < 1e-10 || right_rms < 1e-10 {
        return (left[..len].to_vec(), right[..len].to_vec());
    }

    // Target is average of both
    let target = (left_rms + right_rms) * 0.5;
    let left_gain = (target / left_rms) as f32;
    let right_gain = (target / right_rms) as f32;

    let new_left: Vec<f32> = left[..len].iter().map(|&s| s * left_gain).collect();
    let new_right: Vec<f32> = right[..len].iter().map(|&s| s * right_gain).collect();
    (new_left, new_right)
}

/// Compensate for a delay between channels.
///
/// If `delay_samples` > 0, the right channel leads and will be delayed.
/// If `delay_samples` < 0, the left channel leads and will be delayed.
pub fn compensate_delay(left: &[f32], right: &[f32], delay_samples: i32) -> (Vec<f32>, Vec<f32>) {
    let len = left.len().min(right.len());
    if delay_samples == 0 || len == 0 {
        return (left[..len].to_vec(), right[..len].to_vec());
    }

    let abs_delay = delay_samples.unsigned_abs() as usize;
    if abs_delay >= len {
        return (left[..len].to_vec(), right[..len].to_vec());
    }

    if delay_samples > 0 {
        // Right leads: delay right channel
        let mut new_right = vec![0.0_f32; abs_delay];
        new_right.extend_from_slice(&right[..len - abs_delay]);
        (left[..len].to_vec(), new_right)
    } else {
        // Left leads: delay left channel
        let mut new_left = vec![0.0_f32; abs_delay];
        new_left.extend_from_slice(&left[..len - abs_delay]);
        (new_left, right[..len].to_vec())
    }
}

/// Compute RMS level of a signal.
#[allow(clippy::cast_precision_loss)]
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute normalized cross-correlation between two signals.
#[allow(clippy::cast_precision_loss)]
fn cross_correlation(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum_ab = 0.0_f64;
    let mut sum_aa = 0.0_f64;
    let mut sum_bb = 0.0_f64;
    for i in 0..len {
        let fa = f64::from(a[i]);
        let fb = f64::from(b[i]);
        sum_ab += fa * fb;
        sum_aa += fa * fa;
        sum_bb += fb * fb;
    }
    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        sum_ab / denom
    }
}

/// Estimate the delay between left and right channels by finding the lag that
/// maximizes cross-correlation, searching up to `max_lag` samples.
#[allow(clippy::cast_precision_loss)]
fn estimate_delay(left: &[f32], right: &[f32], max_lag: usize) -> i32 {
    let len = left.len().min(right.len());
    if len < 2 {
        return 0;
    }
    let max_search = max_lag.min(len / 2);

    let mut best_lag: i32 = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag_i in 0..=max_search {
        let lag = lag_i;
        // Positive lag: right shifted
        if lag < len {
            let mut corr = 0.0_f64;
            for i in 0..len - lag {
                corr += f64::from(left[i]) * f64::from(right[i + lag]);
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag as i32;
            }
        }
        // Negative lag: left shifted
        if lag > 0 && lag < len {
            let mut corr = 0.0_f64;
            for i in 0..len - lag {
                corr += f64::from(left[i + lag]) * f64::from(right[i]);
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = -(lag as i32);
            }
        }
    }

    best_lag
}

/// Compute the mono compatibility of a stereo signal.
///
/// Returns a value between -1.0 and 1.0 where 1.0 means perfect mono
/// compatibility (in-phase) and -1.0 means completely out of phase.
pub fn mono_compatibility(left: &[f32], right: &[f32]) -> f64 {
    cross_correlation(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_silence() {
        let samples = vec![0.0_f32; 100];
        assert!(rms(&samples) < 1e-10);
    }

    #[test]
    fn test_rms_dc() {
        let samples = vec![0.5_f32; 100];
        let r = rms(&samples);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let left = vec![1.0_f32, 0.5, -0.3, 0.8];
        let right = vec![0.5_f32, 0.5, 0.3, -0.2];
        let (mid, side) = encode_mid_side(&left, &right);
        let (dec_left, dec_right) = decode_mid_side(&mid, &side);
        for i in 0..left.len() {
            assert!((dec_left[i] - left[i]).abs() < 1e-6);
            assert!((dec_right[i] - right[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mid_side_mono_signal() {
        let mono = vec![1.0_f32, 0.5, -0.3];
        let (mid, side) = encode_mid_side(&mono, &mono);
        // Mid should equal input, side should be zero
        for i in 0..mono.len() {
            assert!((mid[i] - mono[i]).abs() < 1e-6);
            assert!(side[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_adjust_width_mono() {
        let left = vec![1.0_f32, -1.0, 0.5];
        let right = vec![0.5_f32, -0.5, 0.25];
        let (new_l, new_r) = adjust_stereo_width(&left, &right, 0.0);
        // With width 0, both channels should be the mid signal
        for i in 0..left.len() {
            assert!((new_l[i] - new_r[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adjust_width_identity() {
        let left = vec![1.0_f32, -1.0, 0.5];
        let right = vec![0.5_f32, -0.5, 0.25];
        let (new_l, new_r) = adjust_stereo_width(&left, &right, 1.0);
        for i in 0..left.len() {
            assert!((new_l[i] - left[i]).abs() < 1e-6);
            assert!((new_r[i] - right[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_correct_imbalance() {
        let left = vec![1.0_f32; 100];
        let right = vec![0.5_f32; 100];
        let (new_l, new_r) = correct_imbalance(&left, &right);
        let new_l_rms = rms(&new_l);
        let new_r_rms = rms(&new_r);
        assert!((new_l_rms - new_r_rms).abs() < 1e-6);
    }

    #[test]
    fn test_correct_imbalance_balanced() {
        let left = vec![0.7_f32; 50];
        let right = vec![0.7_f32; 50];
        let (new_l, new_r) = correct_imbalance(&left, &right);
        for i in 0..50 {
            assert!((new_l[i] - 0.7).abs() < 1e-6);
            assert!((new_r[i] - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cross_correlation_identical() {
        let sig = vec![1.0_f32, -1.0, 0.5, -0.5];
        let c = cross_correlation(&sig, &sig);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_correlation_inverted() {
        let sig = vec![1.0_f32, -1.0, 0.5, -0.5];
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        let c = cross_correlation(&sig, &inv);
        assert!((c - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compensate_delay_zero() {
        let left = vec![1.0_f32, 2.0, 3.0];
        let right = vec![4.0_f32, 5.0, 6.0];
        let (new_l, new_r) = compensate_delay(&left, &right, 0);
        assert_eq!(new_l, left);
        assert_eq!(new_r, right);
    }

    #[test]
    fn test_compensate_delay_positive() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![5.0, 6.0, 7.0, 8.0];
        let (new_l, new_r) = compensate_delay(&left, &right, 1);
        assert_eq!(new_l.len(), 4);
        assert_eq!(new_r.len(), 4);
        // Right should be delayed by 1
        assert!((new_r[0]).abs() < 1e-6);
        assert!((new_r[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_stereo_field_mono() {
        let signal = vec![0.5_f32; 200];
        let analysis = analyze_stereo_field(&signal, &signal);
        assert!(analysis.imbalance_db.abs() < 0.1);
        assert!((analysis.correlation - 1.0).abs() < 0.01);
        assert!(analysis.width < 0.01);
    }

    #[test]
    fn test_mono_compatibility_in_phase() {
        let sig = vec![1.0_f32, 0.0, -1.0, 0.0];
        let compat = mono_compatibility(&sig, &sig);
        assert!((compat - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_delay_no_delay() {
        let sig = vec![0.0_f32, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let delay = estimate_delay(&sig, &sig, 4);
        assert_eq!(delay, 0);
    }
}
