//! CPU capability detection and runtime dispatch selection.
//!
//! Detects available SIMD/vector ISA extensions at runtime and selects the
//! most capable dispatch level so that callers can route work to the best
//! available code path.

#![allow(dead_code)]

/// Individual CPU features that may be available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuFeature {
    /// SSE2 baseline (x86-64 always has this).
    Sse2,
    /// SSE 4.1.
    Sse4_1,
    /// AVX (256-bit float).
    Avx,
    /// AVX2 (256-bit integer).
    Avx2,
    /// ARM NEON / Advanced SIMD.
    Neon,
}

/// The set of CPU features detected on the current host.
#[derive(Debug, Clone, Default)]
pub struct CpuCapabilities {
    features: Vec<CpuFeature>,
}

impl CpuCapabilities {
    /// Detect CPU features at runtime.
    #[must_use]
    pub fn detect() -> Self {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 is always available on x86-64.
            features.push(CpuFeature::Sse2);
            if is_x86_feature_detected!("sse4.1") {
                features.push(CpuFeature::Sse4_1);
            }
            if is_x86_feature_detected!("avx") {
                features.push(CpuFeature::Avx);
            }
            if is_x86_feature_detected!("avx2") {
                features.push(CpuFeature::Avx2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64.
            features.push(CpuFeature::Neon);
        }

        Self { features }
    }

    /// Return `true` if the given feature is available.
    #[must_use]
    pub fn has(&self, feature: CpuFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Return the natural SIMD vector width in elements (f32 / i32).
    ///
    /// - AVX2 → 8
    /// - SSE2 / NEON → 4
    /// - Scalar → 1
    #[must_use]
    pub fn best_simd_width(&self) -> usize {
        if self.has(CpuFeature::Avx2) || self.has(CpuFeature::Avx) {
            8
        } else if self.has(CpuFeature::Sse2) || self.has(CpuFeature::Neon) {
            4
        } else {
            1
        }
    }

    /// Return all detected features.
    #[must_use]
    pub fn features(&self) -> &[CpuFeature] {
        &self.features
    }
}

/// Coarse dispatch levels ordered from fastest to slowest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DispatchLevel {
    /// Pure scalar – no vector instructions.
    Scalar,
    /// SSE2 128-bit vectors.
    Sse2,
    /// AVX2 256-bit vectors.
    Avx2,
    /// ARM NEON 128-bit vectors.
    Neon,
}

/// Choose the best [`DispatchLevel`] for the given capabilities.
#[must_use]
pub fn select_dispatch_level(caps: &CpuCapabilities) -> DispatchLevel {
    if caps.has(CpuFeature::Avx2) {
        DispatchLevel::Avx2
    } else if caps.has(CpuFeature::Neon) {
        DispatchLevel::Neon
    } else if caps.has(CpuFeature::Sse2) {
        DispatchLevel::Sse2
    } else {
        DispatchLevel::Scalar
    }
}

/// Dispatcher that routes operations to the best available implementation.
pub struct AccelDispatcher {
    level: DispatchLevel,
    capabilities: CpuCapabilities,
}

impl AccelDispatcher {
    /// Create a new dispatcher by detecting the CPU at runtime.
    #[must_use]
    pub fn new() -> Self {
        let capabilities = CpuCapabilities::detect();
        let level = select_dispatch_level(&capabilities);
        Self {
            level,
            capabilities,
        }
    }

    /// Create a dispatcher fixed to a specific level (useful for testing).
    #[must_use]
    pub fn with_level(level: DispatchLevel) -> Self {
        Self {
            level,
            capabilities: CpuCapabilities::default(),
        }
    }

    /// Return the active dispatch level.
    #[must_use]
    pub fn level(&self) -> DispatchLevel {
        self.level
    }

    /// Return the underlying CPU capabilities.
    #[must_use]
    pub fn capabilities(&self) -> &CpuCapabilities {
        &self.capabilities
    }

    /// Dispatch a vector dot-product to the best path.
    #[must_use]
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        // All paths use the same scalar loop for now; in a real
        // implementation each arm would call an intrinsic-based function.
        let _ = self.level; // dispatch would select path
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Dispatch an element-wise gain (scale) operation.
    pub fn apply_gain(&self, samples: &mut [f32], gain: f32) {
        let _ = self.level; // dispatch would select path
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }

    /// Return the recommended chunk size for parallel work items.
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        match self.level {
            DispatchLevel::Avx2 => 512,
            DispatchLevel::Sse2 | DispatchLevel::Neon => 256,
            DispatchLevel::Scalar => 64,
        }
    }
}

impl Default for AccelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_does_not_panic() {
        let caps = CpuCapabilities::detect();
        // Just ensure no panic and we get some result.
        let _ = caps.best_simd_width();
    }

    #[test]
    fn test_has_feature_false_by_default() {
        let caps = CpuCapabilities::default();
        assert!(!caps.has(CpuFeature::Avx2));
        assert!(!caps.has(CpuFeature::Neon));
    }

    #[test]
    fn test_best_simd_width_scalar() {
        let caps = CpuCapabilities::default(); // no features
        assert_eq!(caps.best_simd_width(), 1);
    }

    #[test]
    fn test_best_simd_width_avx2() {
        let caps = CpuCapabilities {
            features: vec![CpuFeature::Avx2],
        };
        assert_eq!(caps.best_simd_width(), 8);
    }

    #[test]
    fn test_best_simd_width_neon() {
        let caps = CpuCapabilities {
            features: vec![CpuFeature::Neon],
        };
        assert_eq!(caps.best_simd_width(), 4);
    }

    #[test]
    fn test_select_dispatch_level_scalar() {
        let caps = CpuCapabilities::default();
        assert_eq!(select_dispatch_level(&caps), DispatchLevel::Scalar);
    }

    #[test]
    fn test_select_dispatch_level_avx2() {
        let caps = CpuCapabilities {
            features: vec![CpuFeature::Avx2],
        };
        assert_eq!(select_dispatch_level(&caps), DispatchLevel::Avx2);
    }

    #[test]
    fn test_select_dispatch_level_neon() {
        let caps = CpuCapabilities {
            features: vec![CpuFeature::Neon],
        };
        assert_eq!(select_dispatch_level(&caps), DispatchLevel::Neon);
    }

    #[test]
    fn test_dispatcher_dot_product() {
        let disp = AccelDispatcher::with_level(DispatchLevel::Scalar);
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let result = disp.dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_dispatcher_apply_gain() {
        let disp = AccelDispatcher::with_level(DispatchLevel::Scalar);
        let mut samples = vec![1.0_f32, 2.0, 3.0];
        disp.apply_gain(&mut samples, 2.0);
        assert_eq!(samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_chunk_size_ordering() {
        let scalar = AccelDispatcher::with_level(DispatchLevel::Scalar).chunk_size();
        let sse2 = AccelDispatcher::with_level(DispatchLevel::Sse2).chunk_size();
        let avx2 = AccelDispatcher::with_level(DispatchLevel::Avx2).chunk_size();
        assert!(scalar <= sse2);
        assert!(sse2 <= avx2);
    }

    #[test]
    fn test_dispatcher_new_does_not_panic() {
        let disp = AccelDispatcher::new();
        // Ensure we can query level without panic.
        let _ = disp.level();
    }

    #[test]
    fn test_cpu_feature_list_not_empty_x86_64() {
        // On x86-64 SSE2 is always present.
        #[cfg(target_arch = "x86_64")]
        {
            let caps = CpuCapabilities::detect();
            assert!(caps.has(CpuFeature::Sse2));
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Nothing to assert on non-x86; just pass.
        }
    }
}
