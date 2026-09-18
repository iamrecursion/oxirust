//! Video generator effects.
//!
//! This module provides generators for creating test patterns, color bars,
//! noise, gradients, and solid colors.

pub mod bars;
pub mod gradient;
pub mod noise;
pub mod pattern;
pub mod proc_noise;
pub mod solid;

pub use bars::{BarsType, ColorBars};
pub use gradient::{Gradient, GradientType};
pub use noise::{Noise, NoiseType};
pub use pattern::{Pattern, PatternType};
pub use proc_noise::{fbm_1d, fbm_2d, hash_u64, value_noise_1d, value_noise_2d, PerlinNoise};
pub use solid::Solid;
