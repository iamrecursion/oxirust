//! Spectral flux computation.

/// Spectral flux computer.
pub struct SpectralFlux;

impl SpectralFlux {
    /// Create a new spectral flux computer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute spectral flux between two frames.
    #[must_use]
    pub fn compute(&self, current: &[f32], previous: &[f32]) -> f32 {
        if current.len() != previous.len() {
            return 0.0;
        }

        current
            .iter()
            .zip(previous)
            .map(|(c, p)| (c - p).max(0.0).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Default for SpectralFlux {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_flux_creation() {
        let _flux = SpectralFlux::new();
    }

    #[test]
    fn test_compute_flux() {
        let flux = SpectralFlux::new();
        let current = vec![1.0, 2.0, 3.0];
        let previous = vec![0.5, 1.5, 2.5];
        let result = flux.compute(&current, &previous);
        assert!(result > 0.0);
    }
}
