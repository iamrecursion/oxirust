//! Lens calibration and correction
//!
//! Provides lens calibration, distortion compensation, and FOV calculation.

pub mod calibrate;
pub mod distortion;
pub mod fov;

use serde::{Deserialize, Serialize};

/// Lens parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensParameters {
    /// Focal length in millimeters
    pub focal_length: f64,
    /// Sensor width in millimeters
    pub sensor_width: f64,
    /// Sensor height in millimeters
    pub sensor_height: f64,
    /// Radial distortion coefficients
    pub radial_distortion: Vec<f64>,
    /// Tangential distortion coefficients
    pub tangential_distortion: Vec<f64>,
}

impl LensParameters {
    /// Create new lens parameters
    #[must_use]
    pub fn new(focal_length: f64, sensor_width: f64, sensor_height: f64) -> Self {
        Self {
            focal_length,
            sensor_width,
            sensor_height,
            radial_distortion: vec![0.0, 0.0, 0.0],
            tangential_distortion: vec![0.0, 0.0],
        }
    }

    /// Get horizontal field of view in degrees
    #[must_use]
    pub fn horizontal_fov(&self) -> f64 {
        2.0 * (self.sensor_width / (2.0 * self.focal_length))
            .atan()
            .to_degrees()
    }

    /// Get vertical field of view in degrees
    #[must_use]
    pub fn vertical_fov(&self) -> f64 {
        2.0 * (self.sensor_height / (2.0 * self.focal_length))
            .atan()
            .to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lens_parameters() {
        let lens = LensParameters::new(50.0, 36.0, 24.0);
        assert_eq!(lens.focal_length, 50.0);

        let hfov = lens.horizontal_fov();
        assert!(hfov > 0.0 && hfov < 180.0);
    }
}
