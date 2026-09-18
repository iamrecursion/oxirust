//! Parallel sync verification across all angle pairs using Rayon.
//!
//! Provides `ParallelSyncVerifier` which distributes the `O(n²)` pairwise
//! offset-validation work across multiple threads via `rayon::par_iter`.
//! This is especially beneficial for multi-camera sessions with many angles
//! (≥ 8) where the sequential verifier in [`crate::sync_verify`] becomes a
//! throughput bottleneck.
//!
//! # Usage
//!
//! ```
//! use oximedia_multicam::sync::{SyncMethod, SyncOffset, SyncResult};
//! use oximedia_multicam::sync_verify_parallel::{
//!     ParallelSyncVerifier, ParallelVerifyConfig,
//! };
//!
//! let result = SyncResult {
//!     reference_angle: 0,
//!     offsets: vec![
//!         SyncOffset::new(0, 0,  0.0, 0.95),
//!         SyncOffset::new(1, 2,  0.3, 0.90),
//!         SyncOffset::new(2, -1, 0.1, 0.88),
//!     ],
//!     confidence: 0.91,
//!     method: SyncMethod::Audio,
//! };
//!
//! let verifier = ParallelSyncVerifier::default();
//! let report = verifier.verify(&result).expect("should succeed");
//! assert!(report.passed);
//! assert_eq!(report.pair_reports.len(), 3); // C(3,2) = 3 pairs
//! ```

use rayon::prelude::*;

use crate::sync::{SyncMethod, SyncOffset, SyncResult};
use crate::{AngleId, MultiCamError, Result};

// ── PairVerificationResult ────────────────────────────────────────────────────

/// Result of verifying a single pair of camera angles.
#[derive(Debug, Clone)]
pub struct PairVerificationResult {
    /// First angle in the pair.
    pub angle_a: AngleId,
    /// Second angle in the pair.
    pub angle_b: AngleId,
    /// Absolute difference in total frame offset between the two angles.
    pub drift_frames: f64,
    /// `true` when the drift is within the configured tolerance.
    pub within_tolerance: bool,
    /// Confidence of angle_a.
    pub confidence_a: f64,
    /// Confidence of angle_b.
    pub confidence_b: f64,
}

impl PairVerificationResult {
    /// Minimum confidence of the two angles.
    #[must_use]
    pub fn min_confidence(&self) -> f64 {
        self.confidence_a.min(self.confidence_b)
    }
}

// ── ParallelVerifyReport ──────────────────────────────────────────────────────

/// Aggregate report from a parallel verification pass.
#[derive(Debug, Clone)]
pub struct ParallelVerifyReport {
    /// Whether all pairs passed their checks.
    pub passed: bool,
    /// Per-pair verification results.
    pub pair_reports: Vec<PairVerificationResult>,
    /// Mean confidence across all angles.
    pub mean_confidence: f64,
    /// Maximum drift observed across any pair (frames).
    pub max_drift: f64,
    /// Number of pairs that exceeded the drift tolerance.
    pub drift_violation_count: usize,
    /// Number of angles whose confidence is below `min_confidence`.
    pub low_confidence_count: usize,
}

impl ParallelVerifyReport {
    /// Collect all pairs that exceeded the drift tolerance.
    #[must_use]
    pub fn violations(&self) -> Vec<&PairVerificationResult> {
        self.pair_reports
            .iter()
            .filter(|p| !p.within_tolerance)
            .collect()
    }
}

// ── ParallelVerifyConfig ──────────────────────────────────────────────────────

/// Configuration for the parallel verifier.
#[derive(Debug, Clone)]
pub struct ParallelVerifyConfig {
    /// Maximum tolerable drift between any pair of angles (frames).
    pub max_pairwise_drift: f64,
    /// Minimum acceptable confidence per angle.
    pub min_confidence: f64,
}

impl Default for ParallelVerifyConfig {
    fn default() -> Self {
        Self {
            max_pairwise_drift: 48.0,
            min_confidence: 0.6,
        }
    }
}

// ── ParallelSyncVerifier ──────────────────────────────────────────────────────

/// Verifies a [`SyncResult`] by checking all `C(n,2)` angle pairs in parallel
/// using Rayon's parallel iterators.
pub struct ParallelSyncVerifier {
    config: ParallelVerifyConfig,
}

impl Default for ParallelSyncVerifier {
    fn default() -> Self {
        Self::new(ParallelVerifyConfig::default())
    }
}

impl ParallelSyncVerifier {
    /// Create a new verifier with the given configuration.
    #[must_use]
    pub fn new(config: ParallelVerifyConfig) -> Self {
        Self { config }
    }

    /// Verify `result` by evaluating all angle pairs in parallel.
    ///
    /// # Errors
    ///
    /// Returns [`MultiCamError::SyncFailed`] when `result.offsets` is empty.
    pub fn verify(&self, result: &SyncResult) -> Result<ParallelVerifyReport> {
        if result.offsets.is_empty() {
            return Err(MultiCamError::SyncFailed(
                "sync result contains no offsets".into(),
            ));
        }

        let offsets = &result.offsets;

        // Build the list of all (i, j) index pairs — C(n, 2).
        let pairs: Vec<(usize, usize)> = (0..offsets.len())
            .flat_map(|i| (i + 1..offsets.len()).map(move |j| (i, j)))
            .collect();

        // Evaluate pairs in parallel.
        let pair_reports: Vec<PairVerificationResult> = pairs
            .par_iter()
            .map(|&(i, j)| self.check_pair(&offsets[i], &offsets[j]))
            .collect();

        // Aggregate statistics.
        let max_drift = pair_reports
            .iter()
            .map(|p| p.drift_frames)
            .fold(0.0_f64, f64::max);

        let drift_violation_count = pair_reports.iter().filter(|p| !p.within_tolerance).count();

        let mean_confidence = if offsets.is_empty() {
            0.0
        } else {
            offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64
        };

        let low_confidence_count = offsets
            .iter()
            .filter(|o| o.confidence < self.config.min_confidence)
            .count();

        // Also apply genlock-specific checks in parallel.
        let genlock_violations = if result.method == SyncMethod::Genlock {
            offsets
                .par_iter()
                .filter(|o| o.total_frames().abs() > 1.0)
                .count()
        } else {
            0
        };

        let passed =
            drift_violation_count == 0 && low_confidence_count == 0 && genlock_violations == 0;

        Ok(ParallelVerifyReport {
            passed,
            pair_reports,
            mean_confidence,
            max_drift,
            drift_violation_count,
            low_confidence_count,
        })
    }

    /// Check a single angle pair.
    fn check_pair(&self, a: &SyncOffset, b: &SyncOffset) -> PairVerificationResult {
        let drift = (a.total_frames() - b.total_frames()).abs();
        PairVerificationResult {
            angle_a: a.angle,
            angle_b: b.angle,
            drift_frames: drift,
            within_tolerance: drift <= self.config.max_pairwise_drift,
            confidence_a: a.confidence,
            confidence_b: b.confidence,
        }
    }

    /// Return the number of angle pairs that would be checked for an `n`-angle
    /// session (i.e. `n * (n - 1) / 2`).
    #[must_use]
    pub fn pair_count(angle_count: usize) -> usize {
        angle_count * angle_count.saturating_sub(1) / 2
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{SyncMethod, SyncOffset, SyncResult};

    fn make_result(offsets: Vec<SyncOffset>) -> SyncResult {
        let confidence = if offsets.is_empty() {
            0.0
        } else {
            offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64
        };
        SyncResult {
            reference_angle: 0,
            offsets,
            confidence,
            method: SyncMethod::Audio,
        }
    }

    // ── basic API ──────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_result_returns_error() {
        let v = ParallelSyncVerifier::default();
        assert!(v.verify(&make_result(vec![])).is_err());
    }

    #[test]
    fn test_pair_count_formula() {
        assert_eq!(ParallelSyncVerifier::pair_count(3), 3);
        assert_eq!(ParallelSyncVerifier::pair_count(4), 6);
        assert_eq!(ParallelSyncVerifier::pair_count(1), 0);
    }

    // ── parallel correctness ──────────────────────────────────────────────────

    /// Three well-synced angles → all pairs pass, report is `passed`.
    #[test]
    fn test_three_angles_all_pass() {
        let v = ParallelSyncVerifier::default();
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.95),
            SyncOffset::new(1, 2, 0.3, 0.90),
            SyncOffset::new(2, -1, 0.1, 0.88),
        ]);
        let report = v.verify(&result).expect("should succeed");
        assert!(report.passed, "Expected all pairs to pass");
        assert_eq!(report.pair_reports.len(), 3);
        assert_eq!(report.drift_violation_count, 0);
    }

    /// One angle has a large drift → `passed = false`.
    #[test]
    fn test_large_drift_fails() {
        let config = ParallelVerifyConfig {
            max_pairwise_drift: 10.0,
            min_confidence: 0.6,
        };
        let v = ParallelSyncVerifier::new(config);
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.95),
            SyncOffset::new(1, 100, 0.0, 0.90), // 100 frame drift
        ]);
        let report = v.verify(&result).expect("should succeed");
        assert!(!report.passed);
        assert!(report.drift_violation_count > 0);
    }

    /// Low confidence angle → `passed = false`.
    #[test]
    fn test_low_confidence_fails() {
        let v = ParallelSyncVerifier::default();
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.95),
            SyncOffset::new(1, 1, 0.0, 0.3), // below 0.6 threshold
        ]);
        let report = v.verify(&result).expect("should succeed");
        assert!(!report.passed);
        assert!(report.low_confidence_count > 0);
    }

    /// C(4,2) = 6 pairs for 4 angles.
    #[test]
    fn test_four_angles_six_pairs() {
        let v = ParallelSyncVerifier::default();
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.95),
            SyncOffset::new(1, 1, 0.0, 0.90),
            SyncOffset::new(2, 2, 0.0, 0.88),
            SyncOffset::new(3, 3, 0.0, 0.85),
        ]);
        let report = v.verify(&result).expect("should succeed");
        assert_eq!(report.pair_reports.len(), 6);
    }

    /// Violations accessor returns only failing pairs.
    #[test]
    fn test_violations_accessor() {
        let config = ParallelVerifyConfig {
            max_pairwise_drift: 5.0,
            ..Default::default()
        };
        let v = ParallelSyncVerifier::new(config);
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.95),
            SyncOffset::new(1, 2, 0.0, 0.90),  // drift 2 — ok
            SyncOffset::new(2, 20, 0.0, 0.88), // drift to angle 0 = 20 — fail
        ]);
        let report = v.verify(&result).expect("should succeed");
        let viols = report.violations();
        assert!(!viols.is_empty(), "Expected at least one violation");
    }

    /// Mean confidence is computed correctly.
    #[test]
    fn test_mean_confidence_value() {
        let v = ParallelSyncVerifier::default();
        let result = make_result(vec![
            SyncOffset::new(0, 0, 0.0, 0.8),
            SyncOffset::new(1, 0, 0.0, 0.6),
        ]);
        let report = v.verify(&result).expect("should succeed");
        assert!((report.mean_confidence - 0.7).abs() < 1e-9);
    }
}
