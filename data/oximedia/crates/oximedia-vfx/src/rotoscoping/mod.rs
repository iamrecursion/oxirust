//! Rotoscoping tools for manual and assisted masking.

pub mod assisted;
pub mod bezier;
pub mod keyframe;

pub use assisted::{AutoTrace, EdgeDetector, Propagator};
pub use bezier::{BezierCurve, BezierMask, BezierPoint};
pub use keyframe::{KeyframedMask, MaskKeyframe};
