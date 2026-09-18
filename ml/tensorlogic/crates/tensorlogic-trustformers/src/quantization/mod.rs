//! Post-Training Quantization (PTQ) for transformer weight matrices.
//!
//! This module provides a CPU-first implementation of INT8 quantization for
//! linear layers, following the same "Paradigm B — numerical layers" design
//! used by the `moe` module.
//!
//! ## Architecture
//!
//! - [`QuantizedLinear`]: A weight matrix stored as `Array2<i8>` with per-channel
//!   or per-tensor scale/zero_point. Forward pass dequantizes on the fly then
//!   performs f64 matmul (integer-matmul kernel is a future follow-up).
//! - [`calibrate_linear`]: Wraps `tensorlogic-scirs-backend`'s
//!   `calibrate_quantization` to produce `QuantizationParams` from a weight
//!   matrix, including per-channel calibration.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ndarray::Array2;
//! use tensorlogic_trustformers::quantization::{calibrate_linear, QuantizedLinear};
//! use tensorlogic_scirs_backend::quantization::{QuantizationGranularity, QuantizationType};
//!
//! let weight = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64);
//! let params = calibrate_linear(&weight, QuantizationType::Int8,
//!                               QuantizationGranularity::PerChannel);
//! let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("quantize");
//!
//! let x = Array2::ones((2, 8));
//! let out = qlinear.forward(&x);
//! assert_eq!(out.shape(), &[2, 4]);
//! ```

pub mod calibration;
pub mod linear;

#[cfg(test)]
mod tests;

pub use calibration::calibrate_linear;
pub use linear::{QuantizationError, QuantizedLinear};
