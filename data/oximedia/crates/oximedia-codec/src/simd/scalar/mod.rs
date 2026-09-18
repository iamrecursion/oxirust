//! Portable scalar fallback implementation.
//!
//! This module provides pure Rust fallback implementations that work on
//! any platform without requiring SIMD support.

mod fallback;

pub use fallback::ScalarFallback;
