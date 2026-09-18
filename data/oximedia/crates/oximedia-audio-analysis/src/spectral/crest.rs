//! Spectral crest factor computation.

/// Compute spectral crest factor.
///
/// The spectral crest factor is the ratio of the peak magnitude to the
/// average magnitude in the spectrum. High values indicate a tonal sound,
/// while low values indicate a noise-like sound.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum
///
/// # Returns
/// Spectral crest factor (linear, not in dB)
pub fn spectral_crest(magnitude: &[f32]) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let max_magnitude = magnitude.iter().copied().fold(0.0_f32, f32::max);
    let mean_magnitude: f32 = magnitude.iter().sum::<f32>() / magnitude.len() as f32;

    if mean_magnitude > 0.0 {
        max_magnitude / mean_magnitude
    } else {
        0.0
    }
}

/// Compute spectral crest factor in decibels.
#[must_use]
pub fn spectral_crest_db(magnitude: &[f32]) -> f32 {
    let crest = spectral_crest(magnitude);
    if crest > 0.0 {
        20.0 * crest.log10()
    } else {
        -100.0
    }
}

/// Compute spectral crest per band.
#[must_use]
pub fn spectral_crest_bands(magnitude: &[f32], num_bands: usize) -> Vec<f32> {
    if magnitude.is_empty() || num_bands == 0 {
        return vec![];
    }

    let band_size = magnitude.len() / num_bands;
    (0..num_bands)
        .map(|band_idx| {
            let start = band_idx * band_size;
            let end = if band_idx == num_bands - 1 {
                magnitude.len()
            } else {
                (band_idx + 1) * band_size
            };
            spectral_crest(&magnitude[start..end])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_crest() {
        // Pure tone should have high crest factor
        let mut tone = vec![0.1; 100];
        tone[50] = 1.0;
        let crest = spectral_crest(&tone);
        assert!(crest > 5.0);

        // Flat spectrum should have low crest factor
        let flat = vec![1.0; 100];
        let crest = spectral_crest(&flat);
        assert!((crest - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_crest_db() {
        let magnitude = vec![0.5; 100];
        let crest_db = spectral_crest_db(&magnitude);
        assert!((crest_db - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_crest_bands() {
        let magnitude = vec![1.0; 100];
        let bands = spectral_crest_bands(&magnitude, 10);
        assert_eq!(bands.len(), 10);
        for crest in bands {
            assert!((crest - 1.0).abs() < 0.01);
        }
    }
}
