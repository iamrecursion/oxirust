//! Spatial denoising filters.
//!
//! This module provides spatial domain denoising filters that operate on
//! individual frames without considering temporal information.

pub mod bilateral;
pub mod bilateral_simd;
pub mod bm3d;
pub mod nlm;
pub mod nlm_approx;
pub mod nlmeans;
pub mod wavelet;
pub mod wiener;
