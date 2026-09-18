//! Spectral centroid computation.

/// Spectral centroid computer.
pub struct SpectralCentroid {
    sample_rate: f32,
    window_size: usize,
}

impl SpectralCentroid {
    /// Create a new spectral centroid computer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
        }
    }

    /// Compute spectral centroid from magnitude spectrum.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            let freq = i as f32 * self.sample_rate / self.window_size as f32;
            weighted_sum += freq * mag;
            total += mag;
        }

        if total > 0.0 {
            weighted_sum / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_centroid_creation() {
        let centroid = SpectralCentroid::new(44100.0, 2048);
        assert_eq!(centroid.sample_rate, 44100.0);
    }

    #[test]
    fn test_compute_centroid() {
        let centroid = SpectralCentroid::new(44100.0, 2048);
        let spectrum = vec![0.0, 1.0, 0.0, 0.0];
        let result = centroid.compute(&spectrum);
        assert!(result > 0.0);
    }
}
