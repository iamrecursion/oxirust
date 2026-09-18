//! Measurement-based LUT creation.
//!
//! This module provides tools for creating LUTs from measurement data.

use crate::error::{CalibrationError, CalibrationResult};
use crate::Rgb;
use serde::{Deserialize, Serialize};

/// LUT measurement point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementPoint {
    /// Input RGB value (0.0-1.0).
    pub input: Rgb,
    /// Measured output RGB value (0.0-1.0).
    pub output: Rgb,
    /// Measurement confidence (0.0-1.0).
    pub confidence: f64,
}

impl MeasurementPoint {
    /// Create a new measurement point.
    #[must_use]
    pub fn new(input: Rgb, output: Rgb, confidence: f64) -> Self {
        Self {
            input,
            output,
            confidence,
        }
    }
}

/// LUT measurement data collection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutMeasurement {
    /// Measurement points.
    pub points: Vec<MeasurementPoint>,
    /// Description of the measurement.
    pub description: String,
}

impl LutMeasurement {
    /// Create a new LUT measurement.
    #[must_use]
    pub fn new(description: String) -> Self {
        Self {
            points: Vec::new(),
            description,
        }
    }

    /// Add a measurement point.
    pub fn add_point(&mut self, point: MeasurementPoint) {
        self.points.push(point);
    }

    /// Add multiple measurement points.
    pub fn add_points(&mut self, points: Vec<MeasurementPoint>) {
        self.points.extend(points);
    }

    /// Get the number of measurement points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Validate the measurement data.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement data is invalid.
    pub fn validate(&self) -> CalibrationResult<()> {
        if self.points.is_empty() {
            return Err(CalibrationError::InvalidMeasurementData(
                "No measurement points".to_string(),
            ));
        }

        // Check that all points have valid RGB values
        for point in &self.points {
            for &value in &point.input {
                if !(0.0..=1.0).contains(&value) {
                    return Err(CalibrationError::InvalidMeasurementData(format!(
                        "Input RGB value {value} out of range [0.0, 1.0]"
                    )));
                }
            }

            for &value in &point.output {
                if !(0.0..=1.0).contains(&value) {
                    return Err(CalibrationError::InvalidMeasurementData(format!(
                        "Output RGB value {value} out of range [0.0, 1.0]"
                    )));
                }
            }

            if !(0.0..=1.0).contains(&point.confidence) {
                return Err(CalibrationError::InvalidMeasurementData(format!(
                    "Confidence {} out of range [0.0, 1.0]",
                    point.confidence
                )));
            }
        }

        Ok(())
    }

    /// Filter measurement points by confidence threshold.
    #[must_use]
    pub fn filter_by_confidence(&self, min_confidence: f64) -> Vec<&MeasurementPoint> {
        self.points
            .iter()
            .filter(|p| p.confidence >= min_confidence)
            .collect()
    }

    /// Get the average measurement confidence.
    #[must_use]
    pub fn average_confidence(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        let total: f64 = self.points.iter().map(|p| p.confidence).sum();
        total / self.points.len() as f64
    }

    /// Create a measurement grid for a 3D LUT.
    ///
    /// # Arguments
    ///
    /// * `grid_size` - Number of points per dimension
    ///
    /// # Returns
    ///
    /// A vector of input RGB values to measure.
    #[must_use]
    pub fn create_measurement_grid(grid_size: usize) -> Vec<Rgb> {
        let mut grid = Vec::with_capacity(grid_size * grid_size * grid_size);

        for b in 0..grid_size {
            for g in 0..grid_size {
                for r in 0..grid_size {
                    let rgb = [
                        r as f64 / (grid_size - 1) as f64,
                        g as f64 / (grid_size - 1) as f64,
                        b as f64 / (grid_size - 1) as f64,
                    ];
                    grid.push(rgb);
                }
            }
        }

        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_point_new() {
        let point = MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95);

        assert!((point.input[0] - 0.5).abs() < 1e-10);
        assert!((point.output[0] - 0.6).abs() < 1e-10);
        assert!((point.confidence - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_lut_measurement_new() {
        let measurement = LutMeasurement::new("Test Measurement".to_string());

        assert_eq!(measurement.description, "Test Measurement");
        assert_eq!(measurement.point_count(), 0);
    }

    #[test]
    fn test_lut_measurement_add_point() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        let point = MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95);

        measurement.add_point(point);
        assert_eq!(measurement.point_count(), 1);
    }

    #[test]
    fn test_lut_measurement_add_points() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        let points = vec![
            MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95),
            MeasurementPoint::new([0.7, 0.7, 0.7], [0.8, 0.8, 0.8], 0.90),
        ];

        measurement.add_points(points);
        assert_eq!(measurement.point_count(), 2);
    }

    #[test]
    fn test_lut_measurement_validate_empty() {
        let measurement = LutMeasurement::new("Test".to_string());
        assert!(measurement.validate().is_err());
    }

    #[test]
    fn test_lut_measurement_validate_valid() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        let point = MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95);

        measurement.add_point(point);
        assert!(measurement.validate().is_ok());
    }

    #[test]
    fn test_lut_measurement_validate_invalid_input() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        let point = MeasurementPoint::new([1.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95);

        measurement.add_point(point);
        assert!(measurement.validate().is_err());
    }

    #[test]
    fn test_lut_measurement_filter_by_confidence() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        measurement.add_points(vec![
            MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.95),
            MeasurementPoint::new([0.7, 0.7, 0.7], [0.8, 0.8, 0.8], 0.90),
            MeasurementPoint::new([0.3, 0.3, 0.3], [0.4, 0.4, 0.4], 0.80),
        ]);

        let filtered = measurement.filter_by_confidence(0.85);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_lut_measurement_average_confidence() {
        let mut measurement = LutMeasurement::new("Test".to_string());
        measurement.add_points(vec![
            MeasurementPoint::new([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.90),
            MeasurementPoint::new([0.7, 0.7, 0.7], [0.8, 0.8, 0.8], 1.00),
        ]);

        let avg = measurement.average_confidence();
        assert!((avg - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_create_measurement_grid() {
        let grid = LutMeasurement::create_measurement_grid(3);

        // 3x3x3 = 27 points
        assert_eq!(grid.len(), 27);

        // First point should be [0, 0, 0]
        assert!((grid[0][0] - 0.0).abs() < 1e-10);
        assert!((grid[0][1] - 0.0).abs() < 1e-10);
        assert!((grid[0][2] - 0.0).abs() < 1e-10);

        // Last point should be [1, 1, 1]
        assert!((grid[26][0] - 1.0).abs() < 1e-10);
        assert!((grid[26][1] - 1.0).abs() < 1e-10);
        assert!((grid[26][2] - 1.0).abs() < 1e-10);
    }
}
