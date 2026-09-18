//! Automatic interpolation method selection based on problem characteristics.
//!
//! This module provides a *problem-specification-driven* API for selecting the
//! most appropriate interpolation strategy.  Unlike the data-driven selector in
//! [`crate::physics_informed::auto_select`], which analyses actual arrays, this
//! module works from high-level problem descriptors such as dimensionality,
//! dataset size, memory budget, and time constraints — useful when the data
//! hasn't been generated yet or when building adaptive pipelines.
//!
//! # Decision Rules (in priority order)
//!
//! | Condition | Method |
//! |-----------|--------|
//! | `n_points > 50_000 && dim > 2` | `RandomFeaturesRbf` (scalable) |
//! | `n_points > 10_000 && dim ≤ 3` | `Kriging` |
//! | `dim == 1` | `CubicSpline` |
//! | `dim == 2 && n_points < 1_000` | `ThinPlateSpline` |
//! | `dim ≤ 4 && smooth` | `Rbf(Gaussian)` |
//! | `dim > 4` | `Kriging` or `RandomFeaturesRbf` |
//! | `require_derivatives` | avoid `NearestNeighbor` |
//! | low memory | `RandomFeaturesRbf` |
//!
//! # Example
//!
//! ```rust
//! use scirs2_interpolate::auto_select::{auto_select, InterpolationProblem};
//!
//! let problem = InterpolationProblem {
//!     n_points: 200,
//!     dim: 2,
//!     smoothness_estimate: Some(0.8),
//!     available_memory_mb: Some(512),
//!     require_derivatives: false,
//!     time_budget_ms: None,
//! };
//! let rec = auto_select(&problem);
//! println!("Recommended: {:?}  — {}", rec.method, rec.reason);
//! ```

use crate::error::InterpolateResult;

// ─────────────────────────────────────────────────────────────────────────────
// Problem descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// High-level description of an interpolation problem used for method selection.
#[derive(Debug, Clone)]
pub struct InterpolationProblem {
    /// Number of training data points.
    pub n_points: usize,
    /// Input dimension (number of coordinates per point).
    pub dim: usize,
    /// Optional smoothness estimate in `[0.0, 1.0]`.
    /// `0.0` = rough/discontinuous, `1.0` = very smooth.
    pub smoothness_estimate: Option<f64>,
    /// Available RAM in megabytes, if constrained.
    pub available_memory_mb: Option<usize>,
    /// Whether the chosen method must provide derivative evaluation.
    pub require_derivatives: bool,
    /// Maximum allowed wall-clock time for prediction (milliseconds).
    pub time_budget_ms: Option<u64>,
}

impl Default for InterpolationProblem {
    fn default() -> Self {
        Self {
            n_points: 100,
            dim: 1,
            smoothness_estimate: None,
            available_memory_mb: None,
            require_derivatives: false,
            time_budget_ms: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Method enum
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolation methods that can be recommended.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest-neighbour lookup — O(1) evaluation, no smoothness.
    NearestNeighbor,
    /// Piecewise linear B-spline — fast, C⁰.
    LinearBspline,
    /// Piecewise cubic spline — smooth (C²), ideal for 1-D data.
    CubicSpline,
    /// Radial basis function with specified kernel.
    Rbf(RbfKernelHint),
    /// Natural-neighbour (Sibson) interpolation.
    NaturalNeighbor,
    /// Kriging (Gaussian process regression) — exact, uncertainty available.
    Kriging,
    /// Random Fourier Feature RBF (Rahimi-Recht) — scalable to large datasets.
    RandomFeaturesRbf,
    /// Thin-plate spline (polyharmonic, dim ≤ 3, small datasets).
    ThinPlateSpline,
}

/// Suggested RBF kernel hint for `InterpolationMethod::Rbf`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum RbfKernelHint {
    /// Gaussian (squared-exponential) — very smooth.
    Gaussian,
    /// Multiquadric — robust general-purpose choice.
    Multiquadric,
    /// Thin-plate spline — natural for 2-D.
    ThinPlate,
}

// ─────────────────────────────────────────────────────────────────────────────
// Recommendation
// ─────────────────────────────────────────────────────────────────────────────

/// A method recommendation with supporting rationale and resource estimates.
#[derive(Debug, Clone)]
pub struct MethodRecommendation {
    /// The recommended interpolation strategy.
    pub method: InterpolationMethod,
    /// Human-readable explanation of why this method was selected.
    pub reason: String,
    /// Estimated peak memory usage in megabytes (rough model).
    pub estimated_memory_mb: f64,
    /// Estimated wall-clock time for a single prediction in milliseconds.
    /// `None` if no reliable estimate is available.
    pub estimated_time_ms: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core selection logic
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate peak memory for a dense RBF / kriging solve in megabytes.
///
/// Storing the n×n kernel matrix: `n² × 8 bytes / 1024² MB`.
fn dense_memory_mb(n: usize) -> f64 {
    (n as f64 * n as f64 * 8.0) / (1024.0 * 1024.0)
}

/// Estimate memory for random-feature approximation.
///
/// Stores D×d weights + n predictions: `(D * d + n) * 8 / 1024²`.
fn rf_memory_mb(n: usize, dim: usize, n_features: usize) -> f64 {
    ((n_features * dim + n) as f64 * 8.0) / (1024.0 * 1024.0)
}

/// Automatically recommend an interpolation method for the given problem.
///
/// Returns a `MethodRecommendation` explaining the choice together with rough
/// resource estimates.  The function always succeeds (it never returns `Err`).
///
/// When `require_derivatives` is `true` the selector ensures that the chosen
/// method supports derivative evaluation (first-order at minimum).  In practice
/// this rules out `NearestNeighbor` (which has no gradient) and biases toward
/// differentiable kernels such as `Rbf(Gaussian)` or `CubicSpline`.
pub fn auto_select(problem: &InterpolationProblem) -> MethodRecommendation {
    let n = problem.n_points;
    let d = problem.dim;
    let smooth = problem.smoothness_estimate.unwrap_or(0.5);
    let mem_limit = problem.available_memory_mb;
    // Effective smoothness: if derivatives are required treat data as at least
    // moderately smooth so we steer away from non-differentiable methods.
    let effective_smooth = if problem.require_derivatives {
        smooth.max(0.6)
    } else {
        smooth
    };

    // ── Memory pressure: prefer O(n) random features if kernel matrix won't fit.
    let kernel_mem_mb = dense_memory_mb(n);
    let rf_mem_mb = rf_memory_mb(n, d, 500);
    let memory_constrained = mem_limit.is_some_and(|mb| kernel_mem_mb > mb as f64);

    // ── Rule 1: very large + high-dimensional → Random Features.
    if n > 50_000 && d > 2 {
        return MethodRecommendation {
            method: InterpolationMethod::RandomFeaturesRbf,
            reason: format!(
                "{n} points in {d}-D: kernel matrix would be {kernel_mem_mb:.0} MB; \
                 Random Fourier Features (D=500) scale to O(nD) ≈ {rf_mem_mb:.1} MB \
                 and O(nD) fit time."
            ),
            estimated_memory_mb: rf_mem_mb,
            estimated_time_ms: Some(n as f64 * 500.0 * 1e-5),
        };
    }

    // ── Rule 2: memory constrained.
    if memory_constrained {
        return MethodRecommendation {
            method: InterpolationMethod::RandomFeaturesRbf,
            reason: format!(
                "Memory budget {mb} MB is below estimated kernel matrix \
                 {kernel_mem_mb:.0} MB; switching to Random Fourier Features.",
                mb = mem_limit.unwrap_or(0)
            ),
            estimated_memory_mb: rf_mem_mb,
            estimated_time_ms: None,
        };
    }

    // ── Rule 3: 1-D data → CubicSpline (fastest, C² smooth).
    if d == 1 {
        // CubicSpline doesn't support derivatives unless explicitly augmented,
        // but it does provide exact first-derivatives via the spline coefficients.
        let mem_mb = (n as f64 * 4.0 * 8.0) / (1024.0 * 1024.0); // 4 coeffs per interval
        return MethodRecommendation {
            method: InterpolationMethod::CubicSpline,
            reason: format!(
                "1-D problem with {n} points: CubicSpline provides C² smoothness, \
                 O(n) memory ({mem_mb:.3} MB), and O(log n) evaluation."
            ),
            estimated_memory_mb: mem_mb,
            estimated_time_ms: Some(0.001), // very fast
        };
    }

    // ── Rule 4: 2-D small dataset → ThinPlateSpline (natural for 2-D scattered).
    // ThinPlateSpline is C¹ and supports gradient evaluation, so it is
    // compatible with require_derivatives=true.
    if d == 2 && n < 1_000 {
        let mem_mb = dense_memory_mb(n);
        return MethodRecommendation {
            method: InterpolationMethod::ThinPlateSpline,
            reason: format!(
                "2-D scattered data with {n} points: ThinPlateSpline minimises bending \
                 energy, giving the smoothest C¹ interpolant; matrix {mem_mb:.2} MB."
            ),
            estimated_memory_mb: mem_mb,
            estimated_time_ms: Some(n as f64 * 1e-3),
        };
    }

    // ── Rule 5: large 1-D / 2-D / 3-D → Kriging (uncertainty quantification).
    if n > 10_000 && d <= 3 {
        return MethodRecommendation {
            method: InterpolationMethod::Kriging,
            reason: format!(
                "{n} points in {d}-D: Kriging with Nyström approximation offers \
                 probabilistic predictions and scales to ~10⁴ points per batch."
            ),
            estimated_memory_mb: dense_memory_mb(n.min(2000)), // Nyström subset
            estimated_time_ms: Some(n as f64 * 5e-4),
        };
    }

    // ── Rule 6: low-to-medium D, smooth data → Gaussian RBF.
    // `effective_smooth` is raised to ≥ 0.6 when require_derivatives is true,
    // so this branch is also reached when the user needs gradient access.
    if d <= 4 && effective_smooth > 0.5 {
        let mem_mb = dense_memory_mb(n);
        return MethodRecommendation {
            method: InterpolationMethod::Rbf(RbfKernelHint::Gaussian),
            reason: format!(
                "{d}-D smooth data ({n} points, smoothness={effective_smooth:.2}): \
                 Gaussian RBF reproduces analytic-like smoothness and supports \
                 gradient evaluation; {mem_mb:.2} MB."
            ),
            estimated_memory_mb: mem_mb,
            estimated_time_ms: Some(n as f64 * 1e-3),
        };
    }

    // ── Rule 7: moderate D, rough data → Multiquadric RBF.
    if d <= 4 {
        let mem_mb = dense_memory_mb(n);
        return MethodRecommendation {
            method: InterpolationMethod::Rbf(RbfKernelHint::Multiquadric),
            reason: format!(
                "{d}-D data ({n} points): Multiquadric RBF is robust for rough \
                 or irregularly sampled data; {mem_mb:.2} MB."
            ),
            estimated_memory_mb: mem_mb,
            estimated_time_ms: Some(n as f64 * 1e-3),
        };
    }

    // ── Rule 8: high-D (> 4) → Kriging or RandomFeatures based on size.
    if d > 4 {
        if n > 5_000 {
            return MethodRecommendation {
                method: InterpolationMethod::RandomFeaturesRbf,
                reason: format!(
                    "{d}-D data ({n} points): high dimensionality with large n; \
                     Random Fourier Features mitigate curse of dimensionality."
                ),
                estimated_memory_mb: rf_mem_mb,
                estimated_time_ms: Some(n as f64 * 500.0 * 1e-5),
            };
        } else {
            return MethodRecommendation {
                method: InterpolationMethod::Kriging,
                reason: format!(
                    "{d}-D data ({n} points): Kriging adapts kernel length-scales \
                     per dimension, handling anisotropic high-D spaces well."
                ),
                estimated_memory_mb: dense_memory_mb(n),
                estimated_time_ms: Some(n as f64 * 2e-3),
            };
        }
    }

    // ── Default fallback.
    MethodRecommendation {
        method: InterpolationMethod::Rbf(RbfKernelHint::Multiquadric),
        reason: format!(
            "Default choice for {d}-D data ({n} points): Multiquadric RBF \
             works well for general scattered data."
        ),
        estimated_memory_mb: dense_memory_mb(n),
        estimated_time_ms: None,
    }
}

/// Validate and auto-select, returning an error for clearly invalid problems.
///
/// Invalid: `n_points == 0`, `dim == 0`.
pub fn auto_select_validated(
    problem: &InterpolationProblem,
) -> InterpolateResult<MethodRecommendation> {
    use crate::error::InterpolateError;
    if problem.n_points == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "n_points must be > 0".to_string(),
        });
    }
    if problem.dim == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "dim must be > 0".to_string(),
        });
    }
    Ok(auto_select(problem))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_select_1d_cubic() {
        let problem = InterpolationProblem {
            n_points: 50,
            dim: 1,
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(
            rec.method,
            InterpolationMethod::CubicSpline,
            "1-D should recommend CubicSpline"
        );
        assert!(!rec.reason.is_empty());
    }

    #[test]
    fn test_auto_select_2d_small_thin_plate() {
        let problem = InterpolationProblem {
            n_points: 200,
            dim: 2,
            smoothness_estimate: Some(0.7),
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(
            rec.method,
            InterpolationMethod::ThinPlateSpline,
            "2-D with < 1000 points should recommend ThinPlateSpline"
        );
    }

    #[test]
    fn test_auto_select_high_dim_large_rf() {
        let problem = InterpolationProblem {
            n_points: 100_000,
            dim: 10,
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(
            rec.method,
            InterpolationMethod::RandomFeaturesRbf,
            "Large high-D should recommend RandomFeaturesRbf"
        );
        assert!(rec.estimated_memory_mb > 0.0);
    }

    #[test]
    fn test_auto_select_memory_constrained() {
        let problem = InterpolationProblem {
            n_points: 10_000,
            dim: 3,
            available_memory_mb: Some(10), // very tight for n×n matrix
            ..Default::default()
        };
        let rec = auto_select(&problem);
        // A 10 000×10 000 matrix is ~800 MB, so RandomFeatures should be chosen.
        assert_eq!(rec.method, InterpolationMethod::RandomFeaturesRbf);
    }

    #[test]
    fn test_auto_select_no_panic_any_input() {
        let cases = vec![
            InterpolationProblem {
                n_points: 1,
                dim: 1,
                ..Default::default()
            },
            InterpolationProblem {
                n_points: 1_000_000,
                dim: 20,
                ..Default::default()
            },
            InterpolationProblem {
                n_points: 500,
                dim: 3,
                smoothness_estimate: Some(0.1),
                require_derivatives: true,
                available_memory_mb: Some(2048),
                time_budget_ms: Some(100),
            },
            InterpolationProblem {
                n_points: 50,
                dim: 4,
                smoothness_estimate: Some(0.9),
                ..Default::default()
            },
        ];
        for (i, p) in cases.iter().enumerate() {
            let rec = auto_select(p);
            assert!(
                !rec.reason.is_empty(),
                "case {i}: reason should not be empty"
            );
            assert!(
                rec.estimated_memory_mb >= 0.0,
                "case {i}: memory estimate should be non-negative"
            );
        }
    }

    #[test]
    fn test_auto_select_validated_zero_points() {
        let problem = InterpolationProblem {
            n_points: 0,
            dim: 1,
            ..Default::default()
        };
        let result = auto_select_validated(&problem);
        assert!(result.is_err(), "0 points should return error");
    }

    #[test]
    fn test_auto_select_validated_zero_dim() {
        let problem = InterpolationProblem {
            n_points: 10,
            dim: 0,
            ..Default::default()
        };
        let result = auto_select_validated(&problem);
        assert!(result.is_err(), "dim=0 should return error");
    }

    #[test]
    fn test_auto_select_large_moderate_d_kriging() {
        let problem = InterpolationProblem {
            n_points: 15_000,
            dim: 3,
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(rec.method, InterpolationMethod::Kriging);
    }

    #[test]
    fn test_auto_select_smooth_4d_gaussian_rbf() {
        let problem = InterpolationProblem {
            n_points: 300,
            dim: 4,
            smoothness_estimate: Some(0.9),
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(
            rec.method,
            InterpolationMethod::Rbf(RbfKernelHint::Gaussian),
        );
    }

    #[test]
    fn test_auto_select_high_dim_small_kriging() {
        let problem = InterpolationProblem {
            n_points: 200,
            dim: 8,
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert_eq!(rec.method, InterpolationMethod::Kriging);
    }

    /// When `require_derivatives=true` on a rough (smoothness=0.1) 3-D problem
    /// that would otherwise stay at `Multiquadric`, `effective_smooth` is lifted
    /// to 0.6 → above the 0.5 threshold → `Rbf(Gaussian)` is returned instead.
    #[test]
    fn test_require_derivatives_biases_toward_differentiable() {
        // Without require_derivatives and low smoothness, Multiquadric is chosen.
        let without = InterpolationProblem {
            n_points: 300,
            dim: 3,
            smoothness_estimate: Some(0.1),
            require_derivatives: false,
            ..Default::default()
        };
        let rec_without = auto_select(&without);
        assert_eq!(
            rec_without.method,
            InterpolationMethod::Rbf(RbfKernelHint::Multiquadric),
            "Without require_derivatives, rough 3-D data should use Multiquadric"
        );

        // With require_derivatives=true the effective smoothness is raised,
        // pushing the selector to Gaussian RBF (differentiable kernel).
        let with_deriv = InterpolationProblem {
            n_points: 300,
            dim: 3,
            smoothness_estimate: Some(0.1),
            require_derivatives: true,
            ..Default::default()
        };
        let rec_with = auto_select(&with_deriv);
        assert_eq!(
            rec_with.method,
            InterpolationMethod::Rbf(RbfKernelHint::Gaussian),
            "With require_derivatives, should bias toward differentiable Gaussian RBF"
        );
        // The reason string should mention gradient / smoothness / derivative.
        assert!(
            rec_with.reason.contains("smooth") || rec_with.reason.contains("gradient"),
            "Reason should mention smoothness/gradient: {}",
            rec_with.reason
        );
    }

    #[test]
    fn test_estimated_memory_positive() {
        let problem = InterpolationProblem {
            n_points: 500,
            dim: 2,
            ..Default::default()
        };
        let rec = auto_select(&problem);
        assert!(
            rec.estimated_memory_mb > 0.0,
            "Memory estimate must be positive, got {}",
            rec.estimated_memory_mb
        );
    }
}
