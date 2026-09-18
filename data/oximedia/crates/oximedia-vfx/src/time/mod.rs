//! Time manipulation effects.

pub mod freeze;
pub mod remap;
pub mod reverse;
pub mod speed;

pub use freeze::FreezeFrame;
pub use remap::{RemapCurve, TimeRemap};
pub use reverse::Reverse;
pub use speed::{SpeedCurve, SpeedRamp};
