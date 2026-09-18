//! 3D stabilization with camera pose estimation.
//!
//! Estimates full 3D camera motion and applies 3D-aware stabilization.

pub mod estimate;
pub mod stabilize;

pub use estimate::CameraPoseEstimator;
pub use stabilize::ThreeDStabilizer;
