//! AV1-specific SIMD operations.
//!
//! This module provides optimized SIMD implementations for AV1 codec operations:
//! - Transform operations (DCT, ADST, identity)
//! - Loop filtering
//! - CDEF (Constrained Directional Enhancement Filter)
//! - Intra prediction
//! - Motion compensation

pub mod cdef;
pub mod intra;
pub mod loop_filter;
pub mod motion_comp;
pub mod transform;

pub use cdef::CdefSimd;
pub use intra::IntraPredSimd;
pub use loop_filter::LoopFilterSimd;
pub use motion_comp::MotionCompSimd;
pub use transform::TransformSimd;
