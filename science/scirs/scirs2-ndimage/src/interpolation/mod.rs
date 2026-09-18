//! Interpolation functions for n-dimensional arrays
//!
//! This module provides functions for interpolating values in n-dimensional arrays.

use std::fmt::Debug;

mod coordinates;
mod geometric;
mod optimized;
mod specialized_transforms;
mod spline;
mod transform;
mod utils;

pub use coordinates::*;
pub use geometric::*;
pub use optimized::{
    map_coordinates_optimized, zoom_optimized, CoefficientCache, Interpolator1D, Interpolator2D,
};
pub use specialized_transforms::*;
pub use spline::*;
pub use transform::*;
pub use utils::*;

/// Order of interpolation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationOrder {
    /// Nearest-neighbor interpolation (order 0)
    Nearest,
    /// Linear interpolation (order 1)
    Linear,
    /// Cubic interpolation (order 3)
    Cubic,
    /// Spline interpolation (order 5)
    Spline,
}

/// Boundary mode for handling points outside the input array
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Fill with constant value
    Constant,
    /// Reflect across boundary
    Reflect,
    /// Mirror across boundary
    Mirror,
    /// Wrap around to opposite edge
    Wrap,
    /// Use nearest edge value
    Nearest,
}
