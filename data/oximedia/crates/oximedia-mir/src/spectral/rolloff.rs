//! Spectral rolloff computation.

/// Spectral rolloff computer.
pub struct SpectralRolloff {
    sample_rate: f32,
    window_size: usize,
    threshold: f32,
}

impl SpectralRolloff {
    /// Create a new spectral rolloff computer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            threshold: 0.85, // 85% of energy
        }
    }

    /// Compute spectral rolloff frequency.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, spectrum: &[f32]) -> f32 {
        let total_energy: f32 = spectrum.iter().map(|m| m * m).sum();
        let threshold_energy = total_energy * self.threshold;

        let mut cumulative_energy = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            cumulative_energy += mag * mag;
            if cumulative_energy >= threshold_energy {
                return i as f32 * self.sample_rate / self.window_size as f32;
            }
        }

        // If not found, return Nyquist frequency
        self.sample_rate / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_rolloff_creation() {
        let rolloff = SpectralRolloff::new(44100.0, 2048);
        assert_eq!(rolloff.sample_rate, 44100.0);
    }

    #[test]
    fn test_compute_rolloff() {
        let rolloff = SpectralRolloff::new(44100.0, 2048);
        let spectrum = vec![1.0, 1.0, 0.1, 0.1];
        let result = rolloff.compute(&spectrum);
        assert!(result > 0.0);
    }
}
