//! Spectral contrast computation.

/// Spectral contrast computer.
pub struct SpectralContrast {
    #[allow(dead_code)]
    sample_rate: f32,
    #[allow(dead_code)]
    window_size: usize,
    num_bands: usize,
}

impl SpectralContrast {
    /// Create a new spectral contrast computer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            num_bands: 6,
        }
    }

    /// Compute spectral contrast across frequency bands.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut contrasts = Vec::with_capacity(self.num_bands);

        let band_size = spectrum.len() / self.num_bands;

        for band_idx in 0..self.num_bands {
            let start = band_idx * band_size;
            let end = ((band_idx + 1) * band_size).min(spectrum.len());

            if end <= start {
                contrasts.push(0.0);
                continue;
            }

            let band = &spectrum[start..end];

            // Sort band values
            let mut sorted_band = band.to_vec();
            sorted_band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Peak: average of top 20%
            let peak_start = (sorted_band.len() as f32 * 0.8) as usize;
            let peak_avg = if peak_start < sorted_band.len() {
                sorted_band[peak_start..].iter().sum::<f32>()
                    / (sorted_band.len() - peak_start) as f32
            } else {
                0.0
            };

            // Valley: average of bottom 20%
            let valley_end = (sorted_band.len() as f32 * 0.2) as usize;
            let valley_avg = if valley_end > 0 {
                sorted_band[..valley_end].iter().sum::<f32>() / valley_end as f32
            } else {
                0.0
            };

            // Contrast
            let contrast = if valley_avg > 0.0 {
                (peak_avg / (valley_avg + 1e-10)).log10()
            } else {
                0.0
            };

            contrasts.push(contrast);
        }

        contrasts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_contrast_creation() {
        let contrast = SpectralContrast::new(44100.0, 2048);
        assert_eq!(contrast.sample_rate, 44100.0);
    }

    #[test]
    fn test_compute_contrast() {
        let contrast = SpectralContrast::new(44100.0, 2048);
        let spectrum = vec![1.0; 60];
        let result = contrast.compute(&spectrum);
        assert_eq!(result.len(), 6);
    }
}
