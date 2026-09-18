//! Trajectory smoothing algorithms.
//!
//! This module provides various algorithms for smoothing camera motion trajectories
//! to achieve stable video output.

pub mod adaptive;
pub mod filter;
pub mod temporal;

pub use adaptive::AdaptiveSmoother;
pub use filter::{GaussianFilter, KalmanFilter, LowPassFilter, TrajectorySmoother};
pub use temporal::TemporalCoherence;
