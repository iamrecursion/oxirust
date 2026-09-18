//! Layer D⁺ — post-discovery analysis of a recovered law via the `oxieml` computer-algebra system.
//!
//! A discovered expression is a live [`oxieml::EmlTree`], not a string, so the whole CAS applies
//! to it directly: symbolic **differentiation**, **antidifferentiation**, **series** expansion,
//! **limits**, and **certified** root finding (interval Newton/Krawczyk) — all in-ecosystem, with
//! no round-trip through an external tool. This realizes phop's "discover → analyze → verify"
//! pipeline, which string-emitting symbolic-regression tools cannot offer without external glue.
//!
//! Operations lower the tree to `oxieml`'s `LoweredOp` IR (`EmlTree::lower`) and delegate; the
//! results are simplified and rendered to LaTeX. Certified roots come back as an
//! [`oxieml::RootCertificate`] that proves unique existence / no root / indeterminacy in an
//! interval enclosure.

use crate::error::{PhopError, Result};
use oxieml::numeric_verified::{find_root_verified, RootOpts};
use oxieml::{
    EmlTree, EvalCtx, IntegrateResult, IntervalLO, LimitPoint, LimitResult, RootCertificate,
};

/// Rendered symbolic analysis of a discovered law (each form simplified, as LaTeX).
#[derive(Debug, Clone)]
pub struct Analysis {
    /// The canonical law itself.
    pub latex: String,
    /// `d/d x_wrt` of the law.
    pub derivative: String,
    /// An antiderivative `∫ · d x_wrt`, or `None` if the CAS finds no closed form.
    pub antiderivative: Option<String>,
    /// Maclaurin series in `x_wrt` to the requested order, or `None` if it could not be formed.
    pub maclaurin: Option<String>,
    /// `lim x_wrt → +∞` of the law, if it is finite.
    pub limit_pos_inf: Option<f64>,
}

/// Differentiate, integrate, expand, and take the `+∞` limit of `tree` with respect to variable
/// `wrt`, rendering each canonical form to LaTeX. `series_order` is the Maclaurin truncation order.
#[must_use]
pub fn analyze(tree: &EmlTree, wrt: usize, series_order: usize) -> Analysis {
    let f = tree.lower().simplify();
    let derivative = f.grad(wrt).simplify().to_latex();
    let antiderivative = match f.integrate(wrt) {
        IntegrateResult::Closed(g) => Some(g.simplify().to_latex()),
        IntegrateResult::Unsupported => None,
    };
    let maclaurin = f
        .maclaurin(wrt, series_order)
        .ok()
        .map(|s| s.simplify().to_latex());
    let limit_pos_inf = match f.limit(wrt, LimitPoint::PosInf) {
        LimitResult::Finite(v) => Some(v),
        _ => None,
    };
    Analysis {
        latex: f.to_latex(),
        derivative,
        antiderivative,
        maclaurin,
        limit_pos_inf,
    }
}

/// Find a **certified** root of `tree` in `[lo, hi]` along variable `wrt`, with the other variables
/// fixed to `others` (the `wrt` slot is overwritten by the search; pass `&[]` for a single-variable
/// law). Uses `oxieml`'s interval Newton/Krawczyk verifier; the returned [`RootCertificate`] proves
/// unique existence, absence, or indeterminacy within the interval enclosure.
///
/// # Errors
/// Returns [`PhopError::Symbolic`] if the verifier errors.
pub fn certified_root(
    tree: &EmlTree,
    wrt: usize,
    others: &[f64],
    lo: f64,
    hi: f64,
) -> Result<RootCertificate> {
    let expr = tree.lower();
    let ctx = EvalCtx::new(others);
    find_root_verified(&expr, wrt, &ctx, lo, hi, &RootOpts::default())
        .map_err(|e| PhopError::Symbolic(format!("certified root finding failed: {e:?}")))
}

/// A **guaranteed** enclosure of the law's range over an axis-aligned box, via interval arithmetic.
///
/// `domain` gives `[lo, hi]` for each variable; the returned `(lo, hi)` is a sound enclosure: every
/// value the law takes on the box lies within it (`f(x) ∈ [lo, hi]` for all `x` in the box). This is
/// the second half of "certified discovery" — alongside [`certified_root`] — and lets a recovered
/// law come with proven bounds (sign, monotone bracketing, safety envelopes), which no
/// string-emitting symbolic-regression tool offers.
#[must_use]
pub fn certified_range(tree: &EmlTree, domain: &[(f64, f64)]) -> (f64, f64) {
    let ivs: Vec<IntervalLO> = domain
        .iter()
        .map(|&(lo, hi)| IntervalLO::new(lo, hi))
        .collect();
    let r = tree.lower().eval_interval(&ivs);
    (r.lo, r.hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxieml::RootStatus;

    #[test]
    fn analyze_exp_derivative_and_integral() {
        // eml(x0, 1) = exp(x0).  d/dx exp = exp ;  ∫ exp dx = exp.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let f = tree.lower();

        // Symbolic derivative evaluated numerically: (exp)'(0.7) = exp(0.7).
        let d = f.grad(0);
        assert!(
            (d.eval(&[0.7]) - 0.7_f64.exp()).abs() < 1e-6,
            "d/dx mismatch: {}",
            d.eval(&[0.7])
        );

        // Antiderivative F with F'(x) ≈ exp(x).
        let integ_ok = match f.integrate(0) {
            IntegrateResult::Closed(g) => (g.grad(0).eval(&[0.4]) - 0.4_f64.exp()).abs() < 1e-5,
            IntegrateResult::Unsupported => false,
        };
        assert!(integ_ok, "expected a closed antiderivative for exp");

        // Rendered analysis is populated.
        let a = analyze(&tree, 0, 4);
        assert!(!a.latex.is_empty() && !a.derivative.is_empty());
    }

    #[test]
    fn certified_range_encloses_exp() {
        // exp(x0) over x0 ∈ [0, 1] has true range [1, e]; the interval enclosure must contain it.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let (lo, hi) = certified_range(&tree, &[(0.0, 1.0)]);
        assert!(lo <= 1.0 + 1e-9, "lower bound {lo} should be ≤ 1");
        assert!(
            hi >= std::f64::consts::E - 1e-9,
            "upper bound {hi} should be ≥ e"
        );
    }

    #[test]
    fn certified_root_brackets_known_zero() {
        // eml(x0, e) = exp(x0) - ln(e) = exp(x0) - 1, whose unique root is x0 = 0.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(std::f64::consts::E));
        let cert = certified_root(&tree, 0, &[], -1.0, 1.0).expect("verifier ran");
        assert!(
            matches!(cert.status, RootStatus::UniqueExists),
            "expected a certified unique root, got {:?}",
            cert.status
        );
        assert!(
            cert.enclosure.contains(0.0),
            "enclosure must bracket the root x0 = 0"
        );
    }
}
