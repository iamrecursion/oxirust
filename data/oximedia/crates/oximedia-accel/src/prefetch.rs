//! Memory prefetching helpers.
//!
//! Provides utilities for cache-line prefetching, access-stride detection,
//! and prefetch-distance tuning to reduce memory-latency bottlenecks in
//! bulk media-processing loops.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ──────────────────────────────────────────────────────────────────────────────
// Cache-line constants
// ──────────────────────────────────────────────────────────────────────────────

/// Typical x86-64 / `AArch64` cache-line size in bytes.
pub const CACHE_LINE_BYTES: usize = 64;

/// Number of bytes in a 4 KiB memory page.
pub const PAGE_BYTES: usize = 4096;

// ──────────────────────────────────────────────────────────────────────────────
// Prefetch distance
// ──────────────────────────────────────────────────────────────────────────────

/// Prefetch distance policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchDistance {
    /// Minimal — prefetch 1 cache line ahead.
    Near,
    /// Moderate — prefetch 4 cache lines ahead (recommended default).
    Medium,
    /// Aggressive — prefetch 8 cache lines ahead (useful for streaming reads).
    Far,
    /// Custom number of cache lines.
    Custom(usize),
}

impl PrefetchDistance {
    /// Return the prefetch distance in cache lines.
    #[must_use]
    pub fn lines(self) -> usize {
        match self {
            Self::Near => 1,
            Self::Medium => 4,
            Self::Far => 8,
            Self::Custom(n) => n,
        }
    }

    /// Return the prefetch distance in bytes.
    #[must_use]
    pub fn bytes(self) -> usize {
        self.lines() * CACHE_LINE_BYTES
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Stride detection
// ──────────────────────────────────────────────────────────────────────────────

/// Result from stride-detection analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridePattern {
    /// Sequential / unit-stride access (stride == element size).
    Sequential,
    /// Constant stride of `n` bytes between accesses.
    Constant(usize),
    /// Irregular / unpredictable stride.
    Irregular,
}

/// Detect the dominant stride from a sequence of byte offsets.
///
/// Returns `Sequential` if all consecutive differences equal `element_bytes`,
/// `Constant(n)` if all differences are equal (but ≠ `element_bytes`), or
/// `Irregular` otherwise.
#[must_use]
pub fn detect_stride(offsets: &[usize], element_bytes: usize) -> StridePattern {
    if offsets.len() < 2 {
        return StridePattern::Sequential;
    }

    let first_delta = offsets[1].wrapping_sub(offsets[0]);
    let all_equal = offsets
        .windows(2)
        .all(|w| w[1].wrapping_sub(w[0]) == first_delta);

    if !all_equal {
        return StridePattern::Irregular;
    }

    if first_delta == element_bytes {
        StridePattern::Sequential
    } else {
        StridePattern::Constant(first_delta)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Prefetch hint emitter
// ──────────────────────────────────────────────────────────────────────────────

/// Emits software prefetch hints for a slice.
///
/// On platforms that support it this calls the architecture-native
/// prefetch intrinsic; on other platforms it is a no-op.
#[allow(unused_variables)]
pub fn prefetch_read<T>(ptr: *const T) {
    // Platform-specific prefetch.  The compiler/linker will eliminate
    // this on targets that don't support it.
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: We are issuing a hint only – no memory is dereferenced.
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0);
        }
    }
    // On aarch64 the compiler often auto-prefetches; explicit PRFM is
    // available via inline asm but we omit it here for simplicity.
}

/// Issue prefetch hints for a contiguous slice, stepping by `distance`.
pub fn prefetch_slice<T>(data: &[T], distance: PrefetchDistance) {
    let step = distance.bytes() / core::mem::size_of::<T>().max(1);
    let step = step.max(1);
    let mut i = 0;
    while i < data.len() {
        prefetch_read(data[i..].as_ptr());
        i += step;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Prefetch planner
// ──────────────────────────────────────────────────────────────────────────────

/// Plans optimal prefetch parameters for a given workload.
#[derive(Debug, Clone)]
pub struct PrefetchPlanner {
    /// Element size in bytes.
    element_bytes: usize,
    /// Target cache level (L1 / L2).
    cache_level: CacheLevel,
}

/// Target cache level for prefetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// L1 data cache.
    L1,
    /// L2 unified cache.
    L2,
}

impl PrefetchPlanner {
    /// Create a new planner.
    #[must_use]
    pub fn new(element_bytes: usize, cache_level: CacheLevel) -> Self {
        Self {
            element_bytes,
            cache_level,
        }
    }

    /// Recommend a prefetch distance for the workload.
    #[must_use]
    pub fn recommended_distance(&self) -> PrefetchDistance {
        match self.cache_level {
            CacheLevel::L1 => PrefetchDistance::Near,
            CacheLevel::L2 => PrefetchDistance::Medium,
        }
    }

    /// Return the recommended loop step size (number of elements per iteration)
    /// when prefetching at the recommended distance.
    #[must_use]
    pub fn loop_step(&self) -> usize {
        let dist = self.recommended_distance().bytes();
        (dist / self.element_bytes.max(1)).max(1)
    }

    /// Return how many elements fit in a single cache line.
    #[must_use]
    pub fn elements_per_cache_line(&self) -> usize {
        CACHE_LINE_BYTES / self.element_bytes.max(1)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Streaming read helper
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulate a sum of `f32` values using a software-prefetch pattern.
///
/// This is a demonstration of the prefetch pattern; real code would use
/// SIMD intrinsics or auto-vectorisation.
#[must_use]
pub fn prefetch_sum(data: &[f32], distance: PrefetchDistance) -> f64 {
    let step = (distance.bytes() / core::mem::size_of::<f32>()).max(1);
    let mut sum = 0.0f64;
    for (i, &v) in data.iter().enumerate() {
        // Issue a prefetch `step` elements ahead.
        if i + step < data.len() {
            prefetch_read(data[i + step..].as_ptr());
        }
        sum += f64::from(v);
    }
    sum
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_distance_lines_near() {
        assert_eq!(PrefetchDistance::Near.lines(), 1);
    }

    #[test]
    fn test_prefetch_distance_lines_medium() {
        assert_eq!(PrefetchDistance::Medium.lines(), 4);
    }

    #[test]
    fn test_prefetch_distance_lines_far() {
        assert_eq!(PrefetchDistance::Far.lines(), 8);
    }

    #[test]
    fn test_prefetch_distance_custom() {
        assert_eq!(PrefetchDistance::Custom(16).lines(), 16);
    }

    #[test]
    fn test_prefetch_distance_bytes() {
        // Medium = 4 lines × 64 bytes = 256 bytes.
        assert_eq!(PrefetchDistance::Medium.bytes(), 256);
    }

    #[test]
    fn test_detect_stride_sequential() {
        let offsets = vec![0, 4, 8, 12];
        assert_eq!(detect_stride(&offsets, 4), StridePattern::Sequential);
    }

    #[test]
    fn test_detect_stride_constant() {
        let offsets = vec![0, 16, 32, 48];
        assert_eq!(detect_stride(&offsets, 4), StridePattern::Constant(16));
    }

    #[test]
    fn test_detect_stride_irregular() {
        let offsets = vec![0, 4, 20, 24];
        assert_eq!(detect_stride(&offsets, 4), StridePattern::Irregular);
    }

    #[test]
    fn test_detect_stride_single_element() {
        let offsets = vec![42];
        assert_eq!(detect_stride(&offsets, 4), StridePattern::Sequential);
    }

    #[test]
    fn test_detect_stride_empty() {
        let offsets: Vec<usize> = vec![];
        assert_eq!(detect_stride(&offsets, 4), StridePattern::Sequential);
    }

    #[test]
    fn test_prefetch_slice_does_not_panic() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        prefetch_slice(&data, PrefetchDistance::Medium); // must not panic
    }

    #[test]
    fn test_prefetch_planner_loop_step_l2() {
        let planner = PrefetchPlanner::new(4, CacheLevel::L2);
        // Medium = 256 bytes / 4 bytes per f32 = 64
        assert_eq!(planner.loop_step(), 64);
    }

    #[test]
    fn test_prefetch_planner_loop_step_l1() {
        let planner = PrefetchPlanner::new(4, CacheLevel::L1);
        // Near = 64 bytes / 4 bytes = 16
        assert_eq!(planner.loop_step(), 16);
    }

    #[test]
    fn test_prefetch_planner_elements_per_cache_line() {
        let planner = PrefetchPlanner::new(4, CacheLevel::L1);
        assert_eq!(planner.elements_per_cache_line(), 16);
    }

    #[test]
    fn test_prefetch_planner_recommended_distance_l1() {
        let p = PrefetchPlanner::new(4, CacheLevel::L1);
        assert_eq!(p.recommended_distance(), PrefetchDistance::Near);
    }

    #[test]
    fn test_prefetch_sum_correctness() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let sum = prefetch_sum(&data, PrefetchDistance::Near);
        assert!((sum - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_prefetch_sum_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(prefetch_sum(&data, PrefetchDistance::Medium), 0.0);
    }
}
