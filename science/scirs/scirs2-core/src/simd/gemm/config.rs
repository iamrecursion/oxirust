//! Matrix multiplication configuration with cache-aware blocking parameters
//!
//! This module provides optimized cache blocking parameters for GEMM (General Matrix Multiply)
//! operations. The parameters are tuned for typical modern CPU cache hierarchies.

/// Configuration for cache-blocked matrix multiplication
///
/// The blocking strategy uses a 3-level hierarchy to optimize for L1, L2, and L3 caches:
/// - MC (M dimension blocking for L3 cache): Number of rows of A to process
/// - KC (K dimension blocking for L2 cache): Inner dimension blocking
/// - NC (N dimension blocking for L3 cache): Number of columns of B to process
/// - MR (Micro-panel height): Rows in micro-kernel (typically 4-8)
/// - NR (Micro-panel width): Columns in micro-kernel (typically 4-16)
///
/// # Cache Blocking Strategy
///
/// ```text
/// C[MC x NC] += A[MC x KC] * B[KC x NC]
///
/// L3 cache: Store MC x KC of A and KC x NC of B
/// L2 cache: Store MR x KC of A (packed) and KC x NR of B (packed)
/// L1 cache: Store MR x NR result accumulation
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MatMulConfig {
    /// L3 cache blocking for M dimension (rows of A)
    /// Typical: 256-512 for f32, 128-256 for f64
    pub mc: usize,

    /// L2 cache blocking for K dimension (inner dimension)
    /// Typical: 128-256 for f32, 64-128 for f64
    pub kc: usize,

    /// L3 cache blocking for N dimension (columns of B)
    /// Typical: 2048-4096 for f32, 1024-2048 for f64
    pub nc: usize,

    /// Micro-kernel height (rows processed in micro-kernel)
    /// Typical: 4-8 (depends on architecture)
    pub mr: usize,

    /// Micro-kernel width (columns processed in micro-kernel)
    /// Typical: 4-16 (depends on SIMD width)
    pub nr: usize,
}

impl MatMulConfig {
    /// Create optimal configuration for f32 GEMM
    ///
    /// These parameters are tuned for modern x86_64 CPUs with:
    /// - L1: 32KB per core
    /// - L2: 256KB-512KB per core
    /// - L3: 8MB-32MB shared
    ///
    /// # Returns
    ///
    /// Configuration optimized for single-precision floating point
    pub const fn for_f32() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx512f") {
                // AVX-512: 16 f32 per vector
                Self {
                    mc: 384,  // L3: 384 * 256 * 4 = 384KB
                    kc: 256,  // L2: 8 * 256 * 4 + 256 * 16 * 4 = 24KB
                    nc: 4096, // L3: 256 * 4096 * 4 = 4MB
                    mr: 8,    // 8 rows in micro-kernel
                    nr: 16,   // 16 cols (1 AVX-512 vector)
                }
            } else if cfg!(target_feature = "avx2") {
                // AVX2: 8 f32 per vector
                Self {
                    mc: 384,
                    kc: 256,
                    nc: 4096,
                    mr: 8, // 8 rows in micro-kernel
                    nr: 8, // 8 cols (1 AVX2 vector)
                }
            } else {
                // SSE: 4 f32 per vector
                Self {
                    mc: 256,
                    kc: 128,
                    nc: 2048,
                    mr: 4,
                    nr: 4,
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON: 4 f32 per vector
            Self {
                mc: 384,
                kc: 256,
                nc: 4096,
                mr: 8, // 8 rows in micro-kernel
                nr: 4, // 4 cols (1 NEON vector)
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative scalar fallback
            Self {
                mc: 128,
                kc: 128,
                nc: 1024,
                mr: 4,
                nr: 4,
            }
        }
    }

    /// Create optimal configuration for f64 GEMM
    ///
    /// These parameters are tuned for double-precision operations,
    /// which have half the SIMD width and double the memory footprint
    /// compared to single-precision.
    ///
    /// # Returns
    ///
    /// Configuration optimized for double-precision floating point
    pub const fn for_f64() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx512f") {
                // AVX-512: 8 f64 per vector
                Self {
                    mc: 192,  // L3: 192 * 128 * 8 = 192KB
                    kc: 128,  // L2: 8 * 128 * 8 + 128 * 8 * 8 = 16KB
                    nc: 2048, // L3: 128 * 2048 * 8 = 2MB
                    mr: 8,
                    nr: 8,
                }
            } else if cfg!(target_feature = "avx2") {
                // AVX2: 4 f64 per vector
                Self {
                    mc: 192,
                    kc: 128,
                    nc: 2048,
                    mr: 8,
                    nr: 4,
                }
            } else {
                // SSE2: 2 f64 per vector
                Self {
                    mc: 128,
                    kc: 64,
                    nc: 1024,
                    mr: 4,
                    nr: 2,
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON: 2 f64 per vector
            Self {
                mc: 192,
                kc: 128,
                nc: 2048,
                mr: 8,
                nr: 2,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative scalar fallback
            Self {
                mc: 64,
                kc: 64,
                nc: 512,
                mr: 4,
                nr: 2,
            }
        }
    }

    /// Get configuration automatically based on type
    ///
    /// # Type Parameters
    ///
    /// * `T` - Float type (f32 or f64)
    #[inline]
    pub fn auto<T>() -> Self
    where
        T: 'static,
    {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            Self::for_f32()
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            Self::for_f64()
        } else {
            // Fallback to f64 config for unknown types
            Self::for_f64()
        }
    }
}

impl Default for MatMulConfig {
    fn default() -> Self {
        Self::for_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_config() {
        let config = MatMulConfig::for_f32();

        // Verify reasonable values
        assert!(config.mc >= 64 && config.mc <= 512);
        assert!(config.kc >= 64 && config.kc <= 512);
        assert!(config.nc >= 512 && config.nc <= 8192);
        assert!(config.mr >= 4 && config.mr <= 16);
        assert!(config.nr >= 4 && config.nr <= 32);

        // Verify mr and nr are divisible by reasonable values
        assert_eq!(config.mr % 4, 0);
        assert_eq!(config.nr % 4, 0);
    }

    #[test]
    fn test_f64_config() {
        let config = MatMulConfig::for_f64();

        // Verify reasonable values
        assert!(config.mc >= 64 && config.mc <= 512);
        assert!(config.kc >= 64 && config.kc <= 256);
        assert!(config.nc >= 512 && config.nc <= 4096);
        assert!(config.mr >= 4 && config.mr <= 16);
        assert!(config.nr >= 2 && config.nr <= 16);
    }

    #[test]
    fn test_auto_config() {
        let config_f32 = MatMulConfig::auto::<f32>();
        let expected_f32 = MatMulConfig::for_f32();

        assert_eq!(config_f32.mc, expected_f32.mc);
        assert_eq!(config_f32.kc, expected_f32.kc);
        assert_eq!(config_f32.nc, expected_f32.nc);
        assert_eq!(config_f32.mr, expected_f32.mr);
        assert_eq!(config_f32.nr, expected_f32.nr);

        let config_f64 = MatMulConfig::auto::<f64>();
        let expected_f64 = MatMulConfig::for_f64();

        assert_eq!(config_f64.mc, expected_f64.mc);
        assert_eq!(config_f64.kc, expected_f64.kc);
        assert_eq!(config_f64.nc, expected_f64.nc);
        assert_eq!(config_f64.mr, expected_f64.mr);
        assert_eq!(config_f64.nr, expected_f64.nr);
    }
}
