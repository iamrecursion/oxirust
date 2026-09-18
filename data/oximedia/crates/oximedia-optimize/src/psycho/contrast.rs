//! Contrast sensitivity function.

use std::f64::consts::PI;

/// Contrast sensitivity function (CSF) calculator.
pub struct ContrastSensitivity {
    viewing_distance: f64,
    display_ppi: f64,
}

impl Default for ContrastSensitivity {
    fn default() -> Self {
        Self::new(1.0, 96.0) // 1 meter, 96 PPI
    }
}

impl ContrastSensitivity {
    /// Creates a new CSF calculator.
    ///
    /// # Parameters
    /// - `viewing_distance`: Viewing distance in meters
    /// - `display_ppi`: Display pixels per inch
    #[must_use]
    pub fn new(viewing_distance: f64, display_ppi: f64) -> Self {
        Self {
            viewing_distance,
            display_ppi,
        }
    }

    /// Calculates contrast sensitivity for a spatial frequency.
    ///
    /// # Parameters
    /// - `frequency`: Spatial frequency in cycles per degree
    ///
    /// # Returns
    /// Sensitivity value (higher = more sensitive to this frequency)
    #[must_use]
    pub fn calculate(&self, frequency: f64) -> f64 {
        // Barten CSF model (simplified)
        let peak_frequency = 4.0; // cycles per degree
        let peak_sensitivity = 100.0;

        let normalized_freq = frequency / peak_frequency;
        peak_sensitivity * (-normalized_freq.powi(2)).exp()
    }

    /// Converts block size to spatial frequency.
    #[must_use]
    pub fn block_size_to_frequency(&self, block_size: usize) -> f64 {
        let pixels_per_degree = self.pixels_per_degree();
        let cycles_per_block = 1.0; // Fundamental frequency
        cycles_per_block / (block_size as f64 / pixels_per_degree)
    }

    fn pixels_per_degree(&self) -> f64 {
        // Convert display PPI and viewing distance to pixels per degree of visual angle
        let pixels_per_meter = self.display_ppi * 39.37; // inches per meter
        let radians_per_degree = PI / 180.0;
        2.0 * self.viewing_distance * pixels_per_meter * radians_per_degree.tan()
    }

    /// Calculates perceptual weight for a block size.
    ///
    /// Smaller blocks (higher frequencies) get less weight if beyond peak sensitivity.
    #[must_use]
    pub fn perceptual_weight(&self, block_size: usize) -> f64 {
        let frequency = self.block_size_to_frequency(block_size);
        let sensitivity = self.calculate(frequency);
        let max_sensitivity = 100.0;
        sensitivity / max_sensitivity
    }
}

/// Contrast sensitivity function helper.
#[derive(Debug, Clone, Copy)]
pub struct ContrastSensitivityFunction {
    /// Peak frequency in cycles per degree.
    pub peak_frequency: f64,
    /// Peak sensitivity.
    pub peak_sensitivity: f64,
}

impl Default for ContrastSensitivityFunction {
    fn default() -> Self {
        Self {
            peak_frequency: 4.0,
            peak_sensitivity: 100.0,
        }
    }
}

impl ContrastSensitivityFunction {
    /// Evaluates the CSF at a given frequency.
    #[must_use]
    pub fn evaluate(&self, frequency: f64) -> f64 {
        let normalized = frequency / self.peak_frequency;
        self.peak_sensitivity * (-normalized.powi(2)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csf_creation() {
        let csf = ContrastSensitivity::default();
        assert_eq!(csf.viewing_distance, 1.0);
        assert_eq!(csf.display_ppi, 96.0);
    }

    #[test]
    fn test_csf_peak_frequency() {
        let csf = ContrastSensitivity::default();
        let sensitivity_at_peak = csf.calculate(4.0);
        let sensitivity_at_0 = csf.calculate(0.0);
        let sensitivity_at_high = csf.calculate(20.0);

        assert!(sensitivity_at_peak > sensitivity_at_high);
        assert!(sensitivity_at_0 > sensitivity_at_peak);
    }

    #[test]
    fn test_perceptual_weight() {
        let csf = ContrastSensitivity::default();
        let weight_large = csf.perceptual_weight(64);
        let weight_small = csf.perceptual_weight(4);
        assert!(weight_large >= 0.0 && weight_large <= 1.0);
        assert!(weight_small >= 0.0 && weight_small <= 1.0);
    }

    #[test]
    fn test_csf_function() {
        let csf = ContrastSensitivityFunction::default();
        // At frequency == peak_frequency, normalized = 1.0, value = peak * e^(-1) ≈ 36.79
        let sensitivity_at_peak_freq = csf.evaluate(4.0);
        // At frequency 0, normalized = 0, value = peak * e^0 = peak_sensitivity = 100.0
        let sensitivity_at_zero = csf.evaluate(0.0);
        assert!((sensitivity_at_zero - 100.0).abs() < 1.0); // Maximum at zero frequency
        assert!(sensitivity_at_peak_freq < sensitivity_at_zero); // Falls off from zero
        assert!(sensitivity_at_peak_freq > 0.0); // Still positive
    }
}
