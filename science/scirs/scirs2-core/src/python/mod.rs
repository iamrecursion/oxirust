//! Python integration module for scirs2-core
//!
//! This module provides PyO3 bindings and NumPy compatibility layers for scirs2-core,
//! enabling seamless integration with Python scientific computing ecosystem.
//!
//! # Features
//!
//! - NumPy array compatibility
//! - Zero-copy conversions where possible
//! - Type-safe Python bindings
//! - Automatic trait implementations for numpy::Dimension
//!
//! # Usage
//!
//! Enable the `python` feature in your Cargo.toml:
//!
//! ```toml
//! scirs2-core = { version = "0.1", features = ["python"] }
//! ```
//!
//! # Architecture
//!
//! This module bridges the gap between scirs2-core's ndarray types and Python's NumPy arrays.
//! It provides:
//!
//! 1. **NumPy Compatibility** - Trait implementations that make scirs2_core::ndarray types
//!    compatible with the PyO3 `numpy` crate
//! 2. **Conversion Utilities** - Helper functions for converting between Rust and Python types
//! 3. **Array Protocol** - Support for Python's array protocol for third-party integrations

pub mod conversions;
pub mod numpy_compat;

#[cfg(feature = "python")]
pub use conversions::*;
#[cfg(feature = "python")]
pub use numpy_compat::*;
