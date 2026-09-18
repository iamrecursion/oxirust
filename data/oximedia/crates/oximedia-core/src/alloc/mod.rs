//! Memory allocation utilities for `OxiMedia`.
//!
//! This module provides memory management utilities optimized for
//! multimedia processing:
//!
//! - [`BufferPool`] - Pool of reusable buffers for zero-copy operations
//! - [`aligned_vec::AlignedVec`] - Cache-line-aligned generic vector (64-byte)

pub mod aligned_vec;
pub mod buffer_pool;
pub mod cache_aligned;

pub use aligned_vec::AlignedVec;
pub use buffer_pool::{BufferPool, PressureConfig};
