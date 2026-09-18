//! Display uniformity testing.
//!
//! This module provides tools for measuring and analyzing display uniformity.

use crate::Rgb;
use serde::{Deserialize, Serialize};

/// Display uniformity test result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniformityReport {
    /// Grid size (number of measurement points per side).
    pub grid_size: usize,
    /// Luminance measurements at each grid point (cd/m²).
    pub luminance_measurements: Vec<f64>,
    /// Color measurements at each grid point (RGB).
    pub color_measurements: Vec<Rgb>,
    /// Maximum luminance deviation (percentage).
    pub max_luminance_deviation: f64,
    /// Average luminance deviation (percentage).
    pub avg_luminance_deviation: f64,
    /// Maximum color deviation (Delta E).
    pub max_color_deviation: f64,
    /// Average color deviation (Delta E).
    pub avg_color_deviation: f64,
    /// Center luminance (cd/m²).
    pub center_luminance: f64,
}

impl UniformityReport {
    /// Create a new uniformity report.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid_size: usize,
        luminance_measurements: Vec<f64>,
        color_measurements: Vec<Rgb>,
        max_luminance_deviation: f64,
        avg_luminance_deviation: f64,
        max_color_deviation: f64,
        avg_color_deviation: f64,
        center_luminance: f64,
    ) -> Self {
        Self {
            grid_size,
            luminance_measurements,
            color_measurements,
            max_luminance_deviation,
            avg_luminance_deviation,
            max_color_deviation,
            avg_color_deviation,
            center_luminance,
        }
    }

    /// Check if uniformity is acceptable.
    ///
    /// # Arguments
    ///
    /// * `max_luminance_deviation` - Maximum acceptable luminance deviation (%)
    /// * `max_color_deviation` - Maximum acceptable color deviation (Delta E)
    #[must_use]
    pub fn is_acceptable(&self, max_luminance_deviation: f64, max_color_deviation: f64) -> bool {
        self.max_luminance_deviation <= max_luminance_deviation
            && self.max_color_deviation <= max_color_deviation
    }

    /// Get the uniformity grade (A-F).
    #[must_use]
    pub fn grade(&self) -> char {
        // Grade based on maximum luminance deviation
        if self.max_luminance_deviation < 5.0 {
            'A'
        } else if self.max_luminance_deviation < 10.0 {
            'B'
        } else if self.max_luminance_deviation < 15.0 {
            'C'
        } else if self.max_luminance_deviation < 20.0 {
            'D'
        } else if self.max_luminance_deviation < 25.0 {
            'E'
        } else {
            'F'
        }
    }
}

/// Display uniformity test.
pub struct UniformityTest {
    grid_size: usize,
}

impl UniformityTest {
    /// Create a new uniformity test.
    #[must_use]
    pub fn new(grid_size: usize) -> Self {
        Self { grid_size }
    }

    /// Perform uniformity test from measurements.
    ///
    /// # Arguments
    ///
    /// * `luminance_measurements` - Luminance at each grid point (cd/m²)
    /// * `color_measurements` - Color at each grid point (RGB)
    #[must_use]
    pub fn analyze(
        &self,
        luminance_measurements: Vec<f64>,
        color_measurements: Vec<Rgb>,
    ) -> UniformityReport {
        let center_index = self.grid_size * self.grid_size / 2;
        let center_luminance = luminance_measurements
            .get(center_index)
            .copied()
            .unwrap_or(100.0);

        let (max_lum_dev, avg_lum_dev) =
            self.calculate_luminance_deviation(&luminance_measurements, center_luminance);

        let center_color = color_measurements
            .get(center_index)
            .copied()
            .unwrap_or([0.5, 0.5, 0.5]);

        let (max_color_dev, avg_color_dev) =
            self.calculate_color_deviation(&color_measurements, &center_color);

        UniformityReport::new(
            self.grid_size,
            luminance_measurements,
            color_measurements,
            max_lum_dev,
            avg_lum_dev,
            max_color_dev,
            avg_color_dev,
            center_luminance,
        )
    }

    /// Calculate luminance deviation statistics.
    fn calculate_luminance_deviation(&self, measurements: &[f64], reference: f64) -> (f64, f64) {
        if measurements.is_empty() || reference <= 0.0 {
            return (0.0, 0.0);
        }

        let mut max_deviation: f64 = 0.0;
        let mut sum_deviation = 0.0;

        for &lum in measurements {
            let deviation = ((lum - reference).abs() / reference) * 100.0;
            max_deviation = max_deviation.max(deviation);
            sum_deviation += deviation;
        }

        let avg_deviation = sum_deviation / measurements.len() as f64;

        (max_deviation, avg_deviation)
    }

    /// Calculate color deviation statistics (Delta E).
    fn calculate_color_deviation(&self, measurements: &[Rgb], reference: &Rgb) -> (f64, f64) {
        if measurements.is_empty() {
            return (0.0, 0.0);
        }

        let mut max_deviation: f64 = 0.0;
        let mut sum_deviation = 0.0;

        for color in measurements {
            let delta_e = Self::calculate_delta_e(color, reference);
            max_deviation = max_deviation.max(delta_e);
            sum_deviation += delta_e;
        }

        let avg_deviation = sum_deviation / measurements.len() as f64;

        (max_deviation, avg_deviation)
    }

    /// Calculate simplified Delta E between two RGB colors.
    fn calculate_delta_e(color1: &Rgb, color2: &Rgb) -> f64 {
        let dr = color1[0] - color2[0];
        let dg = color1[1] - color2[1];
        let db = color1[2] - color2[2];

        (dr * dr + dg * dg + db * db).sqrt() * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniformity_report_new() {
        let report = UniformityReport::new(
            3,
            vec![100.0; 9],
            vec![[0.5, 0.5, 0.5]; 9],
            5.0,
            2.5,
            3.0,
            1.5,
            100.0,
        );

        assert_eq!(report.grid_size, 3);
        assert_eq!(report.luminance_measurements.len(), 9);
        assert_eq!(report.color_measurements.len(), 9);
    }

    #[test]
    fn test_uniformity_report_is_acceptable() {
        let report = UniformityReport::new(
            3,
            vec![100.0; 9],
            vec![[0.5, 0.5, 0.5]; 9],
            5.0,
            2.5,
            3.0,
            1.5,
            100.0,
        );

        assert!(report.is_acceptable(10.0, 5.0));
        assert!(!report.is_acceptable(3.0, 5.0));
        assert!(!report.is_acceptable(10.0, 2.0));
    }

    #[test]
    fn test_uniformity_report_grade() {
        let report_a = UniformityReport::new(
            3,
            vec![100.0; 9],
            vec![[0.5, 0.5, 0.5]; 9],
            4.0,
            2.0,
            3.0,
            1.5,
            100.0,
        );
        assert_eq!(report_a.grade(), 'A');

        let report_b = UniformityReport::new(
            3,
            vec![100.0; 9],
            vec![[0.5, 0.5, 0.5]; 9],
            8.0,
            4.0,
            3.0,
            1.5,
            100.0,
        );
        assert_eq!(report_b.grade(), 'B');

        let report_f = UniformityReport::new(
            3,
            vec![100.0; 9],
            vec![[0.5, 0.5, 0.5]; 9],
            30.0,
            15.0,
            3.0,
            1.5,
            100.0,
        );
        assert_eq!(report_f.grade(), 'F');
    }

    #[test]
    fn test_uniformity_test_new() {
        let test = UniformityTest::new(3);
        assert_eq!(test.grid_size, 3);
    }

    #[test]
    fn test_uniformity_test_analyze_perfect() {
        let test = UniformityTest::new(3);
        let luminance = vec![100.0; 9];
        let colors = vec![[0.5, 0.5, 0.5]; 9];

        let report = test.analyze(luminance, colors);

        assert!((report.max_luminance_deviation - 0.0).abs() < 1e-10);
        assert!((report.avg_luminance_deviation - 0.0).abs() < 1e-10);
        assert!((report.max_color_deviation - 0.0).abs() < 1e-10);
        assert!((report.avg_color_deviation - 0.0).abs() < 1e-10);
        assert_eq!(report.grade(), 'A');
    }

    #[test]
    fn test_uniformity_test_analyze_variation() {
        let test = UniformityTest::new(3);
        let luminance = vec![100.0, 105.0, 95.0, 100.0, 100.0, 100.0, 98.0, 102.0, 100.0];
        let colors = vec![[0.5, 0.5, 0.5]; 9];

        let report = test.analyze(luminance, colors);

        // Maximum deviation should be 5% (105 vs 100)
        assert!((report.max_luminance_deviation - 5.0).abs() < 0.1);
        // Exactly 5.0% deviation: grade boundary is < 5.0 for 'A', so 5.0 gives 'B'
        assert_eq!(report.grade(), 'B');
    }

    #[test]
    fn test_calculate_delta_e() {
        let color1 = [0.5, 0.5, 0.5];
        let color2 = [0.5, 0.5, 0.5];
        let delta_e = UniformityTest::calculate_delta_e(&color1, &color2);
        assert!((delta_e - 0.0).abs() < 1e-10);

        let color3 = [0.6, 0.5, 0.5];
        let delta_e2 = UniformityTest::calculate_delta_e(&color1, &color3);
        assert!(delta_e2 > 0.0);
    }
}
