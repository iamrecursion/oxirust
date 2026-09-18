//! Tensor comparison utilities for testing and validation.
//!
//! Provides configurable tolerance-based tensor comparison, element-wise diff
//! analysis, and assertion helpers for backend validation and gradient checking.

use scirs2_core::ndarray::ArrayD;
use thiserror::Error;

/// Errors that can occur during tensor comparison.
#[derive(Debug, Error)]
pub enum ComparisonError {
    /// The two tensors have different shapes.
    #[error("Shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),
    /// Both tensors are empty (zero elements).
    #[error("Empty tensors")]
    EmptyTensors,
}

/// Tolerance configuration for tensor comparison.
///
/// Uses the NumPy-style closeness criterion:
/// `|a - b| <= atol + rtol * |b|`
#[derive(Debug, Clone)]
pub struct Tolerance {
    /// Relative tolerance (default 1e-5)
    pub rtol: f64,
    /// Absolute tolerance (default 1e-8)
    pub atol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance {
            rtol: 1e-5,
            atol: 1e-8,
        }
    }
}

impl Tolerance {
    /// Create a new tolerance with the given relative and absolute tolerances.
    pub fn new(rtol: f64, atol: f64) -> Self {
        Tolerance { rtol, atol }
    }

    /// Strict tolerance suitable for exact-ish comparisons.
    pub fn strict() -> Self {
        Tolerance {
            rtol: 1e-12,
            atol: 1e-15,
        }
    }

    /// Loose tolerance suitable for approximate comparisons (e.g., gradient checking).
    pub fn loose() -> Self {
        Tolerance {
            rtol: 1e-3,
            atol: 1e-6,
        }
    }

    /// Check if two values are close: `|a - b| <= atol + rtol * |b|`
    pub fn is_close(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.atol + self.rtol * b.abs()
    }
}

/// Result of comparing two tensors element-wise.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Whether all elements are within tolerance.
    pub all_close: bool,
    /// Maximum absolute difference across all elements.
    pub max_abs_diff: f64,
    /// Mean absolute difference across all elements.
    pub mean_abs_diff: f64,
    /// Maximum relative difference (relative to `|b|`).
    pub max_rel_diff: f64,
    /// Number of elements that differ beyond tolerance.
    pub mismatch_count: usize,
    /// Total number of elements compared.
    pub total_elements: usize,
    /// Flattened index of the maximum absolute difference.
    pub max_diff_index: usize,
    /// Number of NaN mismatches (one is NaN, other is not).
    pub nan_mismatches: usize,
    /// Number of Inf mismatches (one is infinite, other is not, or signs differ).
    pub inf_mismatches: usize,
}

impl ComparisonResult {
    /// Fraction of elements that match within tolerance.
    pub fn match_ratio(&self) -> f64 {
        if self.total_elements == 0 {
            1.0
        } else {
            (self.total_elements - self.mismatch_count) as f64 / self.total_elements as f64
        }
    }

    /// Human-readable summary of the comparison.
    pub fn summary(&self) -> String {
        if self.all_close {
            format!(
                "MATCH: {} elements, max_diff={:.2e}",
                self.total_elements, self.max_abs_diff
            )
        } else {
            format!(
                "MISMATCH: {}/{} elements differ, max_diff={:.2e}, mean_diff={:.2e}",
                self.mismatch_count, self.total_elements, self.max_abs_diff, self.mean_abs_diff
            )
        }
    }
}

/// Compare two tensors element-wise with configurable tolerance.
///
/// Handles NaN and Inf specially:
/// - Both NaN → considered matching
/// - One NaN, one not → nan_mismatch
/// - Both ±Inf with same sign → matching
/// - One Inf, one not (or different signs) → inf_mismatch
pub fn compare_tensors(
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
    tol: &Tolerance,
) -> Result<ComparisonResult, ComparisonError> {
    if a.shape() != b.shape() {
        return Err(ComparisonError::ShapeMismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }
    if a.is_empty() {
        return Err(ComparisonError::EmptyTensors);
    }

    let mut max_abs_diff = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    let mut mismatch_count = 0_usize;
    let mut max_diff_index = 0_usize;
    let mut nan_mismatches = 0_usize;
    let mut inf_mismatches = 0_usize;

    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        // Handle NaN cases
        if va.is_nan() != vb.is_nan() {
            nan_mismatches += 1;
            mismatch_count += 1;
            continue;
        }
        if va.is_nan() && vb.is_nan() {
            // Both NaN → considered matching
            continue;
        }

        // Handle Inf cases
        if va.is_infinite() != vb.is_infinite() {
            inf_mismatches += 1;
            mismatch_count += 1;
            continue;
        }
        if va.is_infinite() && vb.is_infinite() {
            if va.signum() == vb.signum() {
                // Both same-sign Inf → matching
                continue;
            }
            // Different sign Inf → mismatch
            inf_mismatches += 1;
            mismatch_count += 1;
            continue;
        }

        // Normal finite comparison
        let abs_diff = (va - vb).abs();
        sum_abs_diff += abs_diff;

        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_diff_index = i;
        }

        let rel_diff = if vb.abs() > 1e-15 {
            abs_diff / vb.abs()
        } else {
            abs_diff
        };
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }

        if !tol.is_close(va, vb) {
            mismatch_count += 1;
        }
    }

    let total = a.len();
    Ok(ComparisonResult {
        all_close: mismatch_count == 0,
        max_abs_diff,
        mean_abs_diff: sum_abs_diff / total as f64,
        max_rel_diff,
        mismatch_count,
        total_elements: total,
        max_diff_index,
        nan_mismatches,
        inf_mismatches,
    })
}

/// Assert two tensors are close (panics with detailed message if not).
///
/// Intended for use in tests where a panic on mismatch is appropriate.
pub fn assert_tensors_close(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: &Tolerance) {
    match compare_tensors(a, b, tol) {
        Ok(result) if result.all_close => {}
        Ok(result) => panic!(
            "Tensors not close: {}\nMax diff at index {}: {:.2e}",
            result.summary(),
            result.max_diff_index,
            result.max_abs_diff
        ),
        Err(e) => panic!("Tensor comparison failed: {e}"),
    }
}

/// Compute element-wise absolute difference tensor.
///
/// Returns a tensor of the same shape where each element is `|a_i - b_i|`.
pub fn abs_diff(a: &ArrayD<f64>, b: &ArrayD<f64>) -> Result<ArrayD<f64>, ComparisonError> {
    if a.shape() != b.shape() {
        return Err(ComparisonError::ShapeMismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }
    let diff = a - b;
    Ok(diff.mapv(f64::abs))
}

/// Check if a tensor contains only finite values (no NaN or Inf).
pub fn is_finite(tensor: &ArrayD<f64>) -> bool {
    tensor.iter().all(|v| v.is_finite())
}

/// Count non-finite values in a tensor.
///
/// Returns `(nan_count, inf_count)`.
pub fn count_non_finite(tensor: &ArrayD<f64>) -> (usize, usize) {
    let nan_count = tensor.iter().filter(|v| v.is_nan()).count();
    let inf_count = tensor.iter().filter(|v| v.is_infinite()).count();
    (nan_count, inf_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, ArrayD};

    fn arr_1d(values: &[f64]) -> ArrayD<f64> {
        arr1(values).into_dyn()
    }

    #[test]
    fn test_tolerance_default() {
        let tol = Tolerance::default();
        assert!((tol.rtol - 1e-5).abs() < 1e-20);
        assert!((tol.atol - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn test_tolerance_is_close_true() {
        let tol = Tolerance::default();
        assert!(tol.is_close(1.0, 1.0 + 1e-9));
    }

    #[test]
    fn test_tolerance_is_close_false() {
        let tol = Tolerance::default();
        assert!(!tol.is_close(1.0, 2.0));
    }

    #[test]
    fn test_tolerance_strict() {
        let tol = Tolerance::strict();
        assert!((tol.rtol - 1e-12).abs() < 1e-20);
        assert!((tol.atol - 1e-15).abs() < 1e-20);
    }

    #[test]
    fn test_tolerance_loose() {
        let tol = Tolerance::loose();
        assert!((tol.rtol - 1e-3).abs() < 1e-20);
        assert!((tol.atol - 1e-6).abs() < 1e-20);
    }

    #[test]
    fn test_compare_identical() {
        let a = arr_1d(&[1.0, 2.0, 3.0]);
        let b = arr_1d(&[1.0, 2.0, 3.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(result.all_close);
        assert!((result.max_abs_diff - 0.0).abs() < 1e-20);
        assert_eq!(result.mismatch_count, 0);
    }

    #[test]
    fn test_compare_close() {
        let a = arr_1d(&[1.0, 2.0, 3.0]);
        let b = arr_1d(&[1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(result.all_close);
    }

    #[test]
    fn test_compare_different() {
        let a = arr_1d(&[1.0, 2.0, 3.0]);
        let b = arr_1d(&[1.0, 2.0, 100.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(!result.all_close);
        assert!(result.mismatch_count > 0);
    }

    #[test]
    fn test_compare_shape_mismatch() {
        let a = arr_1d(&[1.0, 2.0]);
        let b = arr_1d(&[1.0, 2.0, 3.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_empty() {
        let a: ArrayD<f64> = ArrayD::zeros(vec![0]);
        let b: ArrayD<f64> = ArrayD::zeros(vec![0]);
        let result = compare_tensors(&a, &b, &Tolerance::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_nan_both() {
        let a = arr_1d(&[f64::NAN, 1.0]);
        let b = arr_1d(&[f64::NAN, 1.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(result.all_close);
        assert_eq!(result.nan_mismatches, 0);
    }

    #[test]
    fn test_compare_nan_one() {
        let a = arr_1d(&[f64::NAN, 1.0]);
        let b = arr_1d(&[1.0, 1.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(!result.all_close);
        assert_eq!(result.nan_mismatches, 1);
    }

    #[test]
    fn test_compare_inf_matching() {
        let a = arr_1d(&[f64::INFINITY, 1.0]);
        let b = arr_1d(&[f64::INFINITY, 1.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(result.all_close);
        assert_eq!(result.inf_mismatches, 0);
    }

    #[test]
    fn test_compare_match_ratio() {
        let a = arr_1d(&[1.0, 2.0, 3.0, 4.0]);
        let b = arr_1d(&[1.0, 2.0, 3.0, 100.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!((result.match_ratio() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_compare_summary() {
        let a = arr_1d(&[1.0, 2.0]);
        let b = arr_1d(&[1.0, 2.0]);
        let result = compare_tensors(&a, &b, &Tolerance::default()).expect("comparison failed");
        assert!(result.summary().contains("MATCH"));

        let c = arr_1d(&[1.0, 100.0]);
        let result2 = compare_tensors(&a, &c, &Tolerance::default()).expect("comparison failed");
        assert!(result2.summary().contains("MISMATCH"));
    }

    #[test]
    fn test_assert_tensors_close_passes() {
        let a = arr_1d(&[1.0, 2.0, 3.0]);
        let b = arr_1d(&[1.0, 2.0, 3.0]);
        assert_tensors_close(&a, &b, &Tolerance::default());
    }

    #[test]
    fn test_is_finite_true() {
        let a = arr_1d(&[1.0, 2.0, 3.0]);
        assert!(is_finite(&a));
    }

    #[test]
    fn test_count_non_finite() {
        let a = arr_1d(&[1.0, f64::NAN, f64::INFINITY]);
        let (nan_count, inf_count) = count_non_finite(&a);
        assert_eq!(nan_count, 1);
        assert_eq!(inf_count, 1);
    }
}
