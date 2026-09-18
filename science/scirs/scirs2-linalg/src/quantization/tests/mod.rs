//! Tests for the quantization module
//!
//! This module contains comprehensive tests for quantization functionality including:
//! - Basic quantization methods (Uniform, Symmetric, Affine, PowerOfTwo, Int4, UInt4)
//! - Quantized operations (matmul, matvec, dot)
//! - Fake quantization for QAT (Quantization-Aware Training)
//! - Float16/BFloat16 quantization and operations
//! - Per-channel quantization for improved accuracy

#[cfg(test)]
mod basic_tests;

#[cfg(test)]
mod operations_tests;

#[cfg(test)]
mod fake_quant_tests;

#[cfg(test)]
mod float_tests;

#[cfg(test)]
mod per_channel_tests;
