//! Motion estimation and tracking.
//!
//! This module provides algorithms for tracking camera motion across video frames,
//! including feature detection, tracking, and motion model estimation.

pub mod estimate;
pub mod model;
pub mod tracker;
pub mod trajectory;

pub use estimate::MotionEstimator;
pub use model::{AffineModel, MotionModel, PerspectiveModel, TranslationModel};
pub use tracker::{Feature, FeatureTrack, MotionTracker};
pub use trajectory::Trajectory;
