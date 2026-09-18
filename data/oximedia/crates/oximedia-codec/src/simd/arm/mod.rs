//! ARM NEON SIMD implementations.
//!
//! This module provides optimized SIMD implementations for ARM processors
//! using NEON instruction sets.

pub mod neon;

pub use neon::NeonSimd;
