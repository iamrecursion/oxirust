//! Compute shaders (GLSL for Vulkan, WGSL for WebGPU) for image processing.

pub mod bilateral;
pub mod color;
pub mod motion;
pub mod scale;
pub mod temporal_nr;

pub use bilateral::BILATERAL_WGSL;
pub use temporal_nr::TEMPORAL_NR_WGSL;
