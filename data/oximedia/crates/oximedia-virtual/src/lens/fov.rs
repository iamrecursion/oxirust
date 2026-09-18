//! Field of view calculation

use super::LensParameters;

/// FOV calculator
pub struct FovCalculator;

impl FovCalculator {
    /// Calculate FOV from lens parameters
    #[must_use]
    pub fn calculate(params: &LensParameters) -> (f64, f64) {
        (params.horizontal_fov(), params.vertical_fov())
    }
}
