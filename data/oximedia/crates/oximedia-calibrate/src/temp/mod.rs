//! Color temperature estimation and adjustment.
//!
//! This module provides tools for estimating and adjusting color temperature.

pub mod estimate;
pub mod shift;

pub use estimate::estimate_color_temperature;
pub use shift::{apply_temperature_shift, temperature_to_rgb};
