//! SIMD-accelerated element-wise mathematical operations
//!
//! This module provides high-performance implementations of common element-wise
//! mathematical functions that are fundamental for scientific computing, numerical
//! analysis, and data processing.

pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod functions_4;
pub mod functions_5;
pub mod functions_6;

// Re-export all types
pub use functions::*;
pub use functions_2::*;
pub use functions_3::*;
pub use functions_4::*;
pub use functions_5::*;
pub use functions_6::*;

#[cfg(test)]
#[path = "../elementwise_tests.rs"]
mod tests;
