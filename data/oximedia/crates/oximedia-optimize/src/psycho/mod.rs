//! Psychovisual optimization module.
//!
//! Implements perceptual quality optimization using visual masking models.

pub mod contrast;
pub mod masking;
pub mod visual;

pub use contrast::{ContrastSensitivity, ContrastSensitivityFunction};
pub use masking::{MaskingStrength, VisualMasking};
pub use visual::{EdgeAnalysis, PsychoAnalyzer, TextureAnalysis};
