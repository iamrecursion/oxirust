#![allow(dead_code)]
//! SIMD/vectorization status tracking and reporting for acceleration pipelines.

/// Whether a loop or operation has been vectorized.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorizationStatus {
    /// Fully vectorized using the widest available SIMD registers.
    FullyVectorized {
        /// SIMD width used (e.g. 4 for SSE, 8 for AVX, 16 for AVX-512).
        width: u32,
    },
    /// Partially vectorized — some iterations remain scalar.
    PartiallyVectorized {
        /// Fraction of work vectorized, in the range [0.0, 1.0].
        fraction: f32,
        /// SIMD width used.
        width: u32,
    },
    /// Not vectorized at all (scalar path only).
    Scalar,
    /// Vectorization was prevented by a detected dependency.
    BlockedByDependency,
}

impl VectorizationStatus {
    /// Returns `true` if any vectorization occurred.
    #[must_use]
    pub fn is_vectorized(&self) -> bool {
        !matches!(self, Self::Scalar | Self::BlockedByDependency)
    }

    /// Returns the SIMD width, or 1 for scalar/blocked paths.
    #[must_use]
    pub fn simd_width(&self) -> u32 {
        match self {
            Self::FullyVectorized { width } | Self::PartiallyVectorized { width, .. } => *width,
            Self::Scalar | Self::BlockedByDependency => 1,
        }
    }
}

/// Represents a loop or kernel that was analysed for vectorization potential.
#[derive(Debug, Clone)]
pub struct VectorizableLoop {
    /// Human-readable name identifying this loop / kernel.
    pub name: String,
    /// Number of iterations the loop executes per frame.
    pub iteration_count: u64,
    /// Vectorization outcome.
    pub status: VectorizationStatus,
}

impl VectorizableLoop {
    /// Creates a new `VectorizableLoop`.
    #[must_use]
    pub fn new(name: impl Into<String>, iteration_count: u64, status: VectorizationStatus) -> Self {
        Self {
            name: name.into(),
            iteration_count,
            status,
        }
    }

    /// Estimates the speedup factor over a fully scalar baseline.
    ///
    /// For fully-vectorized code this approaches the SIMD width; for partial
    /// vectorization the gain is proportionally reduced.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_speedup(&self) -> f32 {
        match &self.status {
            VectorizationStatus::FullyVectorized { width } => *width as f32,
            VectorizationStatus::PartiallyVectorized { fraction, width } => {
                // Harmonic-style blend of vectorized and scalar throughputs.
                1.0 / ((*fraction / *width as f32) + (1.0 - fraction) / 1.0)
            }
            VectorizationStatus::Scalar | VectorizationStatus::BlockedByDependency => 1.0,
        }
    }
}

/// Aggregated report for all loops analysed in a compilation unit or pass.
#[derive(Debug, Default)]
pub struct VectorizationReport {
    loops: Vec<VectorizableLoop>,
}

impl VectorizationReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a loop entry to the report.
    pub fn add_loop(&mut self, lp: VectorizableLoop) {
        self.loops.push(lp);
    }

    /// Returns a slice of all recorded loops.
    #[must_use]
    pub fn vectorized_loops(&self) -> &[VectorizableLoop] {
        &self.loops
    }

    /// Returns the number of fully-vectorized loops.
    #[must_use]
    pub fn fully_vectorized_count(&self) -> usize {
        self.loops
            .iter()
            .filter(|l| matches!(l.status, VectorizationStatus::FullyVectorized { .. }))
            .count()
    }

    /// Returns the geometric-mean speedup across all loops, weighted by
    /// iteration count.  Returns `1.0` if the report is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn speedup_ratio(&self) -> f32 {
        if self.loops.is_empty() {
            return 1.0;
        }
        let total_iters: u64 = self.loops.iter().map(|l| l.iteration_count).sum();
        if total_iters == 0 {
            return 1.0;
        }
        let weighted_sum: f64 = self
            .loops
            .iter()
            .map(|l| l.estimated_speedup() as f64 * l.iteration_count as f64)
            .sum();
        (weighted_sum / total_iters as f64) as f32
    }

    /// Returns the total number of loops recorded.
    #[must_use]
    pub fn loop_count(&self) -> usize {
        self.loops.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_fully_vectorized_is_vectorized() {
        let s = VectorizationStatus::FullyVectorized { width: 8 };
        assert!(s.is_vectorized());
    }

    #[test]
    fn test_status_scalar_not_vectorized() {
        assert!(!VectorizationStatus::Scalar.is_vectorized());
    }

    #[test]
    fn test_status_blocked_not_vectorized() {
        assert!(!VectorizationStatus::BlockedByDependency.is_vectorized());
    }

    #[test]
    fn test_status_partial_is_vectorized() {
        let s = VectorizationStatus::PartiallyVectorized {
            fraction: 0.5,
            width: 4,
        };
        assert!(s.is_vectorized());
    }

    #[test]
    fn test_simd_width_full() {
        let s = VectorizationStatus::FullyVectorized { width: 16 };
        assert_eq!(s.simd_width(), 16);
    }

    #[test]
    fn test_simd_width_scalar() {
        assert_eq!(VectorizationStatus::Scalar.simd_width(), 1);
    }

    #[test]
    fn test_simd_width_blocked() {
        assert_eq!(VectorizationStatus::BlockedByDependency.simd_width(), 1);
    }

    #[test]
    fn test_loop_speedup_scalar() {
        let lp = VectorizableLoop::new("scalar_loop", 1000, VectorizationStatus::Scalar);
        assert!((lp.estimated_speedup() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_speedup_fully_vectorized() {
        let lp = VectorizableLoop::new(
            "vec_loop",
            1000,
            VectorizationStatus::FullyVectorized { width: 8 },
        );
        assert!((lp.estimated_speedup() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_speedup_partial_greater_than_1() {
        let lp = VectorizableLoop::new(
            "partial",
            500,
            VectorizationStatus::PartiallyVectorized {
                fraction: 0.8,
                width: 4,
            },
        );
        assert!(lp.estimated_speedup() > 1.0);
    }

    #[test]
    fn test_report_empty_speedup_ratio() {
        let report = VectorizationReport::new();
        assert!((report.speedup_ratio() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_report_add_and_count() {
        let mut report = VectorizationReport::new();
        report.add_loop(VectorizableLoop::new(
            "lp1",
            100,
            VectorizationStatus::FullyVectorized { width: 4 },
        ));
        report.add_loop(VectorizableLoop::new(
            "lp2",
            50,
            VectorizationStatus::Scalar,
        ));
        assert_eq!(report.loop_count(), 2);
    }

    #[test]
    fn test_report_fully_vectorized_count() {
        let mut report = VectorizationReport::new();
        report.add_loop(VectorizableLoop::new(
            "a",
            100,
            VectorizationStatus::FullyVectorized { width: 8 },
        ));
        report.add_loop(VectorizableLoop::new("b", 100, VectorizationStatus::Scalar));
        report.add_loop(VectorizableLoop::new(
            "c",
            100,
            VectorizationStatus::FullyVectorized { width: 4 },
        ));
        assert_eq!(report.fully_vectorized_count(), 2);
    }

    #[test]
    fn test_report_speedup_ratio_all_full() {
        let mut report = VectorizationReport::new();
        report.add_loop(VectorizableLoop::new(
            "lp",
            1000,
            VectorizationStatus::FullyVectorized { width: 8 },
        ));
        // Weighted speedup should be 8.0.
        assert!((report.speedup_ratio() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_vectorized_loops_slice() {
        let mut report = VectorizationReport::new();
        report.add_loop(VectorizableLoop::new("x", 10, VectorizationStatus::Scalar));
        assert_eq!(report.vectorized_loops().len(), 1);
        assert_eq!(report.vectorized_loops()[0].name, "x");
    }
}
