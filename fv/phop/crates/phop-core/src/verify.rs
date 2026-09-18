//! Logical verification of discovered laws — the proof tier above plain numeric fitting.
//!
//! A discovered law is a live `oxieml` EML tree, so we can *prove* properties of it over an
//! axis-aligned box rather than merely sampling. Two engines back this module, and we lean on each
//! only where it is **sound**:
//!
//! * **Interval arithmetic** ([`crate::analyze::certified_range`]) gives a guaranteed enclosure
//!   `f(x) ∈ [lo, hi]` for every `x` in the box. This is the source of every [`Verdict::Proven`]:
//!   if the enclosure already implies the property (e.g. `lo ≥ c` ⇒ `f ≥ c`), it holds rigorously.
//! * **OxiZ** (the cool-japan pure-Rust Z3-replacement SMT solver, reached through
//!   `oxieml::smt::EmlSmtSolver`) is used to *search for counterexamples*: a `Sat` witness is
//!   re-evaluated with phop's exact forward pass and only reported as a [`Verdict::Counterexample`]
//!   if it genuinely violates the property.
//!
//! ## Why `Proven` does **not** come from the SMT solver
//!
//! The bridge that encodes EML into OxiZ's linear real arithmetic was **unsound for `eml` nodes
//! with a constant `ln`-operand** in `oxieml` 0.1.2: it relaxed `ln` over a degenerate point interval
//! and returned a spurious `Unsat`. Empirically every comparison of `exp(x0) − 1 = eml(x0, e)`
//! against `0` over `[-1, 1]` came back `Unsat`, despite the obvious root at `x0 = 0`. This was
//! **fixed in `oxieml` 0.1.3** (cool-japan/oxieml#1): the interval layer now reports *indeterminate*
//! instead of a conflict, so `Unsat` is returned only for genuinely-infeasible constraints. We still
//! take proofs from interval arithmetic (which also handles `Const` correctly) and use OxiZ for the
//! always-rechecked counterexample direction; adding SMT `Unsat` as a second `Proven` source — it can
//! prove things interval arithmetic cannot, such as dependency cancellation — is now-unblocked follow-up.
//!
//! Enable with the `smt` feature (pulls `oxiz` through `oxieml`). This is the natural feed into a
//! future OxiLean proof-checking step (OxiZ can emit proofs via `oxiz-proof`).

use oxieml::{Canonical, EmlConstraint, EmlSmtSolver, EmlTree, SmtResult};
use scirs2_core::ndarray::Array2;

/// Tolerance for interval-based equivalence (`|a − b| ≤ EQ_TOL` ⇒ equal).
const EQ_TOL: f64 = 1e-9;

/// Outcome of a property check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The property is **proven** to hold over the whole box (via a rigorous interval enclosure).
    Proven,
    /// The property is **false**: a witness was found and verified to violate it.
    Counterexample,
    /// Neither proof nor counterexample was established.
    Unknown,
}

/// Evaluate `tree` at a single point `xs` (one value per variable) using phop's guarded forward.
fn eval_point(tree: &EmlTree, xs: &[f64]) -> Option<f64> {
    let row = Array2::from_shape_vec((1, xs.len()), xs.to_vec()).ok()?;
    crate::forest::eval_tree(tree, &row).ok().map(|y| y[0])
}

/// Ask OxiZ for a model of `constraint`; if it returns one, re-evaluate `tree` at that point and
/// return its value only when `violates(value)` actually holds (so a buggy/over-eager solver can
/// never produce a false counterexample). Witness coordinates are clamped into the box.
fn verified_witness(
    constraint: &EmlConstraint,
    tree: &EmlTree,
    bounds: &[(f64, f64)],
    violates: impl Fn(f64) -> bool,
) -> bool {
    let solver = EmlSmtSolver::new(bounds.to_vec());
    let Ok(SmtResult::Sat(sol)) = solver.check_sat(constraint) else {
        return false;
    };
    let xs: Vec<f64> = bounds
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| {
            sol.assignments
                .get(i)
                .copied()
                .unwrap_or((lo + hi) * 0.5)
                .clamp(lo, hi)
        })
        .collect();
    eval_point(tree, &xs).is_some_and(violates)
}

/// Prove the lower bound `f(x) ≥ c` over the box (one `(lo, hi)` per variable).
#[must_use]
pub fn prove_lower_bound(tree: &EmlTree, c: f64, bounds: &[(f64, f64)]) -> Verdict {
    let (lo, _hi) = crate::analyze::certified_range(tree, bounds);
    if lo >= c {
        return Verdict::Proven;
    }
    if verified_witness(
        &EmlConstraint::lt(tree.clone(), EmlTree::const_val(c)),
        tree,
        bounds,
        |v| v < c,
    ) {
        return Verdict::Counterexample;
    }
    Verdict::Unknown
}

/// Prove the upper bound `f(x) ≤ c` over the box.
#[must_use]
pub fn prove_upper_bound(tree: &EmlTree, c: f64, bounds: &[(f64, f64)]) -> Verdict {
    let (_lo, hi) = crate::analyze::certified_range(tree, bounds);
    if hi <= c {
        return Verdict::Proven;
    }
    if verified_witness(
        &EmlConstraint::gt(tree.clone(), EmlTree::const_val(c)),
        tree,
        bounds,
        |v| v > c,
    ) {
        return Verdict::Counterexample;
    }
    Verdict::Unknown
}

/// Prove the law is **strictly positive** (`f(x) > 0`) over the box.
#[must_use]
pub fn prove_positive(tree: &EmlTree, bounds: &[(f64, f64)]) -> Verdict {
    let (lo, _hi) = crate::analyze::certified_range(tree, bounds);
    if lo > 0.0 {
        return Verdict::Proven;
    }
    if verified_witness(
        &EmlConstraint::le(tree.clone(), EmlTree::const_val(0.0)),
        tree,
        bounds,
        |v| v <= 0.0,
    ) {
        return Verdict::Counterexample;
    }
    Verdict::Unknown
}

/// Prove the law is **strictly negative** (`f(x) < 0`) over the box.
#[must_use]
pub fn prove_negative(tree: &EmlTree, bounds: &[(f64, f64)]) -> Verdict {
    let (_lo, hi) = crate::analyze::certified_range(tree, bounds);
    if hi < 0.0 {
        return Verdict::Proven;
    }
    if verified_witness(
        &EmlConstraint::ge(tree.clone(), EmlTree::const_val(0.0)),
        tree,
        bounds,
        |v| v >= 0.0,
    ) {
        return Verdict::Counterexample;
    }
    Verdict::Unknown
}

/// Prove the law has **no root** over the box (`f(x) ≠ 0` everywhere).
///
/// Proven when the interval enclosure excludes zero (`lo > 0` or `hi < 0`). To *certify* an actual
/// root instead, use [`crate::analyze::certified_root`].
#[must_use]
pub fn prove_no_root(tree: &EmlTree, bounds: &[(f64, f64)]) -> Verdict {
    let (lo, hi) = crate::analyze::certified_range(tree, bounds);
    if lo > 0.0 || hi < 0.0 {
        Verdict::Proven
    } else {
        Verdict::Unknown
    }
}

/// Prove two laws are **equal** over the box (`a(x) = b(x)` everywhere, within [`EQ_TOL`]).
///
/// Proven if the trees are structurally identical, or if the interval enclosure of `a − b` is
/// within tolerance of zero. Note the enclosure of `a − b` is subject to the interval *dependency
/// problem* (shared variables are decorrelated), so non-trivial equivalences often return
/// [`Verdict::Unknown`] rather than a false negative.
#[must_use]
pub fn prove_equivalent(a: &EmlTree, b: &EmlTree, bounds: &[(f64, f64)]) -> Verdict {
    if a.root == b.root {
        return Verdict::Proven;
    }
    let diff = Canonical::sub(a, b);
    let (lo, hi) = crate::analyze::certified_range(&diff, bounds);
    if lo >= -EQ_TOL && hi <= EQ_TOL {
        return Verdict::Proven;
    }
    if verified_witness(
        &EmlConstraint::gt(a.clone(), b.clone()),
        &diff,
        bounds,
        |d| d.abs() > EQ_TOL,
    ) || verified_witness(
        &EmlConstraint::lt(a.clone(), b.clone()),
        &diff,
        bounds,
        |d| d.abs() > EQ_TOL,
    ) {
        return Verdict::Counterexample;
    }
    Verdict::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_x0() -> EmlTree {
        EmlTree::eml(&EmlTree::var(0), &EmlTree::one()) // exp(x0)
    }
    fn exp_x0_minus_1() -> EmlTree {
        EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(std::f64::consts::E))
        // exp(x0) - 1
    }

    #[test]
    fn proves_bounds_via_interval() {
        // exp(x0) ∈ [1, e] on [0, 1] — interval arithmetic proves both bounds rigorously.
        assert_eq!(
            prove_lower_bound(&exp_x0(), 1.0, &[(0.0, 1.0)]),
            Verdict::Proven
        );
        assert_eq!(
            prove_upper_bound(&exp_x0(), 3.0, &[(0.0, 1.0)]),
            Verdict::Proven
        );
        assert_eq!(prove_positive(&exp_x0(), &[(-2.0, 2.0)]), Verdict::Proven);
    }

    #[test]
    fn finds_verified_counterexamples() {
        // exp(x0) ≥ 2 on [0, 1] is false at x0 = 0 (value 1) — OxiZ finds it, we re-verify it.
        assert_eq!(
            prove_lower_bound(&exp_x0(), 2.0, &[(0.0, 1.0)]),
            Verdict::Counterexample
        );
        // exp(x0) ≤ 2 on [0, 1] is false near x0 = 1 (value e ≈ 2.72).
        assert_eq!(
            prove_upper_bound(&exp_x0(), 2.0, &[(0.0, 1.0)]),
            Verdict::Counterexample
        );
    }

    #[test]
    fn no_root_is_sound_despite_const_operands() {
        // exp(x0) > 0 everywhere ⇒ provably root-free.
        assert_eq!(prove_no_root(&exp_x0(), &[(-3.0, 3.0)]), Verdict::Proven);
        // exp(x0) - 1 ≥ e^0.5 - 1 > 0 on [0.5, 1] ⇒ provably root-free — note the Const(e) operand
        // that breaks the raw SMT path; interval arithmetic handles it correctly.
        assert_eq!(
            prove_no_root(&exp_x0_minus_1(), &[(0.5, 1.0)]),
            Verdict::Proven
        );
        // exp(x0) - 1 changes sign on [-1, 1] (root at 0) — must NOT be falsely proven root-free.
        assert_ne!(
            prove_no_root(&exp_x0_minus_1(), &[(-1.0, 1.0)]),
            Verdict::Proven
        );
    }

    #[test]
    fn proves_structural_equivalence() {
        let f = exp_x0();
        assert_eq!(
            prove_equivalent(&f, &f, &[(0.0, 1.0)]),
            Verdict::Proven,
            "a law is equal to itself"
        );
    }
}
