//! Transform calculation and optimization.
//!
//! Computes stabilization transforms from original and smoothed trajectories,
//! with optimization to minimize cropping and maintain output quality.

pub mod calculate;
pub mod interpolate;
pub mod optimize;

pub use calculate::TransformCalculator;
pub use interpolate::TransformInterpolator;
pub use optimize::TransformOptimizer;
