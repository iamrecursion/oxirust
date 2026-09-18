//! Robust regression losses for constant fitting under outliers.
//!
//! Plain mean-squared error gives every residual quadratic weight, so a handful of gross
//! outliers dominate the fit and the recovered constants drift away from the clean law. A
//! **robust loss** caps (Huber) or discards (trimmed) the influence of large residuals.
//!
//! `oxieml`'s symbolic-regression engine exposes the same two robust variants through its
//! [`oxieml::SymRegLoss`] enum, but the loss math itself (`huber_loss`, `trimmed_mse`, and their
//! gradient factors) is private to that crate. phop re-implements the (short, standard) formulas
//! here so they can drive phop's *own* Levenberg–Marquardt polish via **IRLS** (iteratively
//! reweighted least squares): at each LM iteration the per-row residual and Jacobian are scaled by
//! a weight `w_i = √(grad_factor(r_i)/r_i)`, turning a robust objective into a sequence of weighted
//! least-squares solves. With [`RobustLoss::Mse`] every weight is `1`, so the polish reduces
//! exactly to ordinary least squares.

use serde::{Deserialize, Serialize};

/// A loss function for fitting an expression's constants to data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum RobustLoss {
    /// Mean-squared error (no robustness). Every residual gets full quadratic weight.
    #[default]
    Mse,
    /// Huber loss: quadratic for `|r| ≤ delta`, linear beyond. `delta` is the quadratic-to-linear
    /// transition; smaller `delta` is more aggressive at down-weighting outliers.
    Huber {
        /// Transition point between the quadratic and linear regimes.
        delta: f64,
    },
    /// Trimmed MSE: discard the largest `alpha` fraction of squared residuals before averaging.
    /// `alpha ∈ [0, 1)`; e.g. `alpha = 0.1` ignores the worst 10% of points.
    Trimmed {
        /// Fraction of largest-residual points to trim.
        alpha: f64,
    },
}

/// Clamp ceiling for IRLS weights, mirroring `oxieml`'s relaxation: keeps a near-zero residual
/// from producing an unbounded `grad_factor(r)/r` ratio.
const WEIGHT_CLAMP: f64 = 1e6;
/// Sharpness of the trimmed-loss smooth (sigmoid) soft-trim used for the gradient weight.
const TRIM_SHARPNESS: f64 = 3.0;

impl RobustLoss {
    /// The scalar loss (mean over points) for a residual vector `r = prediction − target`.
    #[must_use]
    pub fn cost(self, residuals: &[f64]) -> f64 {
        let n = residuals.len().max(1) as f64;
        match self {
            RobustLoss::Mse => residuals.iter().map(|r| r * r).sum::<f64>() / n,
            RobustLoss::Huber { delta } => {
                residuals
                    .iter()
                    .map(|&r| huber_point(r, delta))
                    .sum::<f64>()
                    / n
            }
            RobustLoss::Trimmed { alpha } => trimmed_mse(residuals, alpha),
        }
    }

    /// The IRLS multiplicative weight `w_i` applied to residual row `i` (and the matching Jacobian
    /// row) so that a weighted least-squares step descends the robust objective. `residuals` is the
    /// full current residual vector (needed by the trimmed loss to find its quantile threshold).
    #[must_use]
    pub fn irls_weight(self, r: f64, residuals: &[f64]) -> f64 {
        let ratio = match self {
            RobustLoss::Mse => 1.0,
            RobustLoss::Huber { delta } => {
                if r.abs() <= delta || r == 0.0 {
                    1.0
                } else {
                    delta / r.abs()
                }
            }
            RobustLoss::Trimmed { alpha } => soft_trim_weight(r, residuals, alpha),
        };
        ratio.clamp(0.0, WEIGHT_CLAMP).sqrt()
    }
}

/// Per-point Huber loss `L(r) = ½r²` for `|r| ≤ δ`, else `δ(|r| − ½δ)`.
fn huber_point(r: f64, delta: f64) -> f64 {
    let a = r.abs();
    if a <= delta {
        0.5 * r * r
    } else {
        delta * (a - 0.5 * delta)
    }
}

/// Trimmed MSE: square residuals, drop the largest `alpha` fraction, average the rest.
fn trimmed_mse(residuals: &[f64], alpha: f64) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut sq: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Keep the ceil((1-alpha)·n) smallest squared residuals (at least one).
    let keep = (((1.0 - alpha) * residuals.len() as f64).ceil() as usize).max(1);
    let keep = keep.min(sq.len());
    sq.iter().take(keep).sum::<f64>() / keep as f64
}

/// Smooth (sigmoid) soft-trim weight in `[0, 1]`: ≈1 for residuals below the `(1-alpha)` quantile
/// of `|r|`, decaying to 0 beyond it. Smoothness keeps the IRLS iteration well-behaved (a hard cut
/// would make the weight discontinuous in the parameters).
fn soft_trim_weight(r: f64, residuals: &[f64], alpha: f64) -> f64 {
    if residuals.is_empty() {
        return 1.0;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut abs: Vec<f64> = residuals.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // (1-alpha) quantile of |r| as the soft threshold.
    let q_idx = (((1.0 - alpha) * (abs.len() as f64 - 1.0)).round() as usize).min(abs.len() - 1);
    let q = abs[q_idx];
    1.0 / (1.0 + (TRIM_SHARPNESS * (r.abs() - q)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_weights_are_unit() {
        let r = [1.0, -5.0, 0.0, 100.0];
        for &ri in &r {
            assert!((RobustLoss::Mse.irls_weight(ri, &r) - 1.0).abs() < 1e-15);
        }
        let expected = (1.0 + 25.0 + 0.0 + 10_000.0) / 4.0;
        assert!((RobustLoss::Mse.cost(&r) - expected).abs() < 1e-9);
    }

    #[test]
    fn huber_caps_large_residuals() {
        let delta = 1.0;
        let loss = RobustLoss::Huber { delta };
        // Inside the quadratic regime: weight 1, cost ½r².
        assert!((loss.irls_weight(0.5, &[0.5]) - 1.0).abs() < 1e-12);
        assert!((huber_point(0.5, delta) - 0.125).abs() < 1e-12);
        // Outside: weight √(δ/|r|) < 1, cost linear.
        let w = loss.irls_weight(4.0, &[4.0]);
        assert!(w < 1.0 && (w - (1.0_f64 / 4.0).sqrt()).abs() < 1e-12);
        assert!((huber_point(4.0, delta) - (4.0 - 0.5)).abs() < 1e-12);
    }

    #[test]
    fn trimmed_drops_worst_points() {
        // Nine small residuals and one huge outlier; trimming 10% removes the outlier.
        let mut r = vec![0.1_f64; 9];
        r.push(1000.0);
        let full = RobustLoss::Mse.cost(&r);
        let trimmed = RobustLoss::Trimmed { alpha: 0.1 }.cost(&r);
        assert!(
            trimmed < full * 1e-3,
            "trim did not drop the outlier: {trimmed} vs {full}"
        );
        // The outlier gets a near-zero IRLS weight; a clean point keeps weight ≈ 1.
        let w_out = RobustLoss::Trimmed { alpha: 0.1 }.irls_weight(1000.0, &r);
        let w_in = RobustLoss::Trimmed { alpha: 0.1 }.irls_weight(0.1, &r);
        assert!(w_out < 0.1, "outlier weight too high: {w_out}");
        assert!(w_in > 0.5, "inlier weight too low: {w_in}");
    }
}
