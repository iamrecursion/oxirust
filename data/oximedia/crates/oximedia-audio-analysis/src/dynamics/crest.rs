//! Crest factor computation.

use crate::compute_rms;

/// Compute crest factor (peak-to-RMS ratio).
///
/// Crest factor indicates how "peaky" a signal is. High crest factors
/// indicate transient-rich signals, while low crest factors indicate
/// compressed or sustained sounds.
///
/// # Arguments
/// * `samples` - Audio samples
///
/// # Returns
/// Crest factor (linear, not in dB)
pub fn crest_factor(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let peak = samples.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
    let rms = compute_rms(samples);

    if rms > 0.0 {
        peak / rms
    } else {
        0.0
    }
}

/// Compute crest factor in decibels.
#[must_use]
pub fn crest_factor_db(samples: &[f32]) -> f32 {
    let crest = crest_factor(samples);
    if crest > 0.0 {
        20.0 * crest.log10()
    } else {
        -100.0
    }
}

/// Compute crest factor over time.
pub fn crest_factor_over_time(samples: &[f32], window_size: usize) -> Vec<f32> {
    if window_size == 0 || samples.len() < window_size {
        return vec![];
    }

    samples
        .windows(window_size)
        .step_by(window_size / 2) // 50% overlap
        .map(crest_factor)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crest_factor() {
        // Sine wave has crest factor of sqrt(2) ≈ 1.414
        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 100.0).sin())
            .collect();

        let crest = crest_factor(&samples);
        assert!((crest - 1.414).abs() < 0.1);
    }

    #[test]
    fn test_crest_factor_db() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let crest_db = crest_factor_db(&samples);
        assert!(crest_db >= 0.0); // Square wave has crest factor of 1 (0 dB)
    }

    #[test]
    fn test_crest_over_time() {
        let samples = vec![1.0; 1000];
        let crest_track = crest_factor_over_time(&samples, 100);
        assert!(!crest_track.is_empty());

        // Constant signal should have crest factor of 1
        for &crest in &crest_track {
            assert!((crest - 1.0).abs() < 0.01);
        }
    }
}
