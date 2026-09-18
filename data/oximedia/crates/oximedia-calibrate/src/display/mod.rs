//! Display calibration and profiling.
//!
//! This module provides tools for calibrating displays (monitors) and
//! generating display profiles for accurate color reproduction.

pub mod calibrate;
pub mod gamma;
pub mod uniformity;

pub use calibrate::{DisplayCalibrator, DisplayConfig};
pub use gamma::{GammaCorrection, GammaCurve};
pub use uniformity::{UniformityReport, UniformityTest};
