//! Spectral flatness computation.

/// Compute spectral flatness (Wiener entropy).
///
/// Spectral flatness is a measure of how noise-like a sound is, as opposed to
/// being tonal. A high spectral flatness (close to 1) indicates noise-like sound,
/// while a low value indicates a tonal sound.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum
///
/// # Returns
/// Spectral flatness value between 0 and 1
#[must_use]
pub fn spectral_flatness(magnitude: &[f32]) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    // Filter out zero and very small values
    let min_magnitude = 1e-10_f32;
    let valid_magnitudes: Vec<f32> = magnitude.iter().map(|&m| m.max(min_magnitude)).collect();

    // Geometric mean
    let log_sum: f32 = valid_magnitudes.iter().map(|&m| m.ln()).sum();
    let geometric_mean = (log_sum / valid_magnitudes.len() as f32).exp();

    // Arithmetic mean
    let arithmetic_mean: f32 = valid_magnitudes.iter().sum::<f32>() / valid_magnitudes.len() as f32;

    if arithmetic_mean > 0.0 {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}

/// Compute spectral flatness in decibels.
#[must_use]
pub fn spectral_flatness_db(magnitude: &[f32]) -> f32 {
    let flatness = spectral_flatness(magnitude);
    if flatness > 0.0 {
        10.0 * flatness.log10()
    } else {
        -100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_flatness() {
        // White noise (all bins equal) should have flatness near 1
        let noise = vec![1.0; 100];
        let flatness = spectral_flatness(&noise);
        assert!((flatness - 1.0).abs() < 0.01);

        // Pure tone (one dominant bin) should have lower flatness than noise
        let mut tone = vec![0.01; 100];
        tone[50] = 1.0;
        let flatness = spectral_flatness(&tone);
        // With background noise, flatness won't be very low
        assert!(flatness >= 0.0 && flatness < 1.0);
    }

    #[test]
    fn test_flatness_db() {
        let magnitude = vec![1.0; 100];
        let db = spectral_flatness_db(&magnitude);
        assert!((db - 0.0).abs() < 0.1); // Near 0 dB for flat spectrum
    }
}
