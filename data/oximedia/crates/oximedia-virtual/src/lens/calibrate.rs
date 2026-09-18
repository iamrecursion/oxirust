//! Lens calibration workflow

use super::LensParameters;
use crate::Result;

/// Lens calibrator
pub struct LensCalibrator {
    parameters: Option<LensParameters>,
}

impl LensCalibrator {
    /// Create new lens calibrator
    #[must_use]
    pub fn new() -> Self {
        Self { parameters: None }
    }

    /// Start calibration
    pub fn calibrate(&mut self, focal_length: f64) -> Result<LensParameters> {
        let params = LensParameters::new(focal_length, 36.0, 24.0);
        self.parameters = Some(params.clone());
        Ok(params)
    }
}

impl Default for LensCalibrator {
    fn default() -> Self {
        Self::new()
    }
}
