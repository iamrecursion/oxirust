//! White balance algorithms and presets.
//!
//! This module provides tools for white balance adjustment and correction.

pub mod auto;
pub mod custom;
pub mod preset;

pub use auto::{AutoWhiteBalance, WhiteBalanceMethod};
pub use custom::CustomWhiteBalance;
pub use preset::WhiteBalancePreset;
