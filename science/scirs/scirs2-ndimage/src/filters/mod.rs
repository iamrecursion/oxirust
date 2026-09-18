//! Filtering functions for n-dimensional arrays
//!
//! This module provides functions for filtering n-dimensional arrays,
//! including Gaussian filters, median filters, uniform filters, and various convolution operations.

use std::fmt::Debug;

// Bilateral module exports
pub use bilateral::{
    adaptive_bilateral_filter, bilateral_filter, multi_scale_bilateral_filter,
    MultiScaleBilateralConfig,
};
#[cfg(feature = "simd")]
pub use bilateral::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

// SIMD specialized exports
#[cfg(feature = "simd")]
pub use simd_specialized::{
    simd_anisotropic_diffusion, simd_bilateral_filter, simd_non_local_means,
};

// Advanced SIMD optimized exports
#[cfg(feature = "simd")]
pub use advanced_simd_optimized::{
    advanced_simd_gaussian_pyramid, advanced_simd_morphological_erosion_2d,
    advanced_simd_template_matching,
};

// Advanced SIMD enhanced exports
#[cfg(feature = "simd")]
pub use advanced_simd_enhanced::{
    advanced_simd_convolution_2d, advanced_simd_median_filter,
    advanced_simd_separable_convolution_2d,
};

// Advanced SIMD extensions exports
#[cfg(feature = "simd")]
pub use advanced_simd_extensions::{
    advanced_simd_advanced_edge_detection, advanced_simd_multi_scale_lbp,
    advanced_simd_wavelet_pyramid, WaveletPyramid, WaveletType,
};

// Enhanced SIMD optimizations exports
#[cfg(feature = "simd")]
pub use simd_enhanced_optimizations::{
    simd_gradient_magnitude, simd_histogram, simd_local_binary_pattern,
    simd_morphological_operation, simdimage_moments, GradientOperator, MorphologicalOperation,
};

pub mod advanced;
#[cfg(feature = "simd")]
mod advanced_simd_enhanced;
#[cfg(feature = "simd")]
mod advanced_simd_extensions;
#[cfg(feature = "simd")]
mod advanced_simd_optimized;
mod bilateral;
mod boundary_handler;
mod boundary_optimized;
mod convolve;
mod edge;
mod edge_optimized;
mod extrema;
mod fourier;
mod gaussian;
mod generic;
mod median;
mod memory_efficient;
mod memory_efficient_v2;
mod rank;
#[cfg(feature = "simd")]
mod simd_enhanced_optimizations;
#[cfg(feature = "simd")]
mod simd_specialized;
mod tests;
mod uniform;
mod utils;
mod vectorized;
pub mod wavelets;

// Convolve module exports
pub use convolve::{
    convolve,
    convolve_fast,
    // Re-export the uniform_filter from convolve with a different name
    uniform_filter as convolve_uniform_filter,
};

// Edge module exports
pub use edge::{gradient_magnitude, laplace, prewitt, roberts, scharr, sobel};

// Optimized edge detection exports
pub use edge_optimized::{gradient_magnitude_optimized, laplace_2d_optimized, sobel_2d_optimized};

// Extrema module exports (new implementation)
pub use extrema::{maximum_filter, minimum_filter};

// Fourier module exports
pub use fourier::{fourier_ellipsoid, fourier_gaussian, fourier_shift, fourier_uniform};

// Gaussian module exports
pub use gaussian::{gaussian_filter, gaussian_filter_f32, gaussian_filter_f64};

// Generic module exports
pub use generic::{filter_functions, generic_filter};

// Median module exports
pub use median::*;

// Rank module exports
// Add minimum_filter and maximum_filter from rank module with different names
// to avoid conflicts with extrema module
pub use rank::{
    maximum_filter as rank_maximum_filter, minimum_filter as rank_minimum_filter,
    percentile_filter, percentile_filter_footprint, rank_filter, rank_filter_footprint,
};

// Uniform module exports
pub use uniform::{uniform_filter, uniform_filter_separable};

// Utils module exports
pub use utils::*;

// Boundary handler exports
pub use boundary_handler::{
    apply_filter_with_boundary, convolve_optimized, BoundaryHandler, VirtualBoundaryHandler,
};

// Memory-efficient filter exports
pub use memory_efficient::{
    gaussian_filter_chunked, median_filter_chunked, uniform_filter_chunked,
};

// Optimized boundary handling exports
pub use boundary_optimized::{
    apply_filter2d_optimized, convolve1d_optimized, convolve2d_optimized, Boundary1D, Boundary2D,
    OptimizedBoundaryOps,
};

// Vectorized batch processing exports
pub use vectorized::{
    apply_filter_batch, convolve_batch, gaussian_filter_batch, median_filter_batch, sobel_batch,
    BatchConfig,
};

// Advanced filters exports
pub use advanced::{
    adaptive_wiener_filter, anisotropic_diffusion, bilateral_gradient_filter,
    coherence_enhancing_diffusion, gabor_filter, gabor_filter_bank, log_gabor_filter,
    non_local_means, shock_filter, steerable_filter, GaborParams,
};

// Wavelets module exports
pub use wavelets::{
    dwt_1d, dwt_2d, idwt_1d, idwt_2d, wavelet_decompose, wavelet_denoise, wavelet_reconstruct,
    WaveletFamily, WaveletFilter,
};

/// Border handling modes for filters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Pad with zeros
    Constant,

    /// Reflect values across the edges
    Reflect,

    /// Mirror and reflect values across the edges
    Mirror,

    /// Wrap around to the opposite edge (periodic)
    Wrap,

    /// Repeat edge values
    Nearest,
}
