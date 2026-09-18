//! Reference frame management module.
//!
//! Optimal reference frame selection and decoded picture buffer optimization.

pub mod dpb;
pub mod selection;

pub use dpb::{DpbOptimizer, DpbStats};
pub use selection::{RefFrameScore, ReferenceSelection};
