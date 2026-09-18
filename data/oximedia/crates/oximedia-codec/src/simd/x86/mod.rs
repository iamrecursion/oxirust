//! x86_64 SIMD implementations (AVX2 and AVX-512).
//!
//! This module provides optimized SIMD implementations for x86_64 processors
//! using AVX2 and AVX-512 instruction sets.

pub mod avx2;
pub mod avx512;

pub use avx2::Avx2Simd;
pub use avx512::Avx512Simd;
