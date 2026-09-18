//! Library-backed constant refinement via `scirs2-optimize`.
//!
//! phop's default polish ([`crate::polish::polish_constants`]) is a small, panic-free, well-tested
//! Levenberg–Marquardt with named-constant snapping. This module offers two **alternative**
//! refinement backends drawn from the cool-japan `scirs2-optimize` crate so the constant fit can be
//! swapped without touching the discovery pipeline:
//!
//! - [`ScirsPolish::Lm`] — `scirs2_optimize::least_squares` (Levenberg–Marquardt), and
//! - [`ScirsPolish::Lbfgs`] — `scirs2_optimize::unconstrained::minimize_lbfgs` minimizing the sum
//!   of squared residuals.
//!
//! They are offered as opt-in backends rather than as a drop-in replacement: phop's hand-rolled LM
//! makes no contiguity assumptions and never panics, whereas `scirs2`'s least-squares uses internal
//! `expect`s on `as_slice()`. We feed it only contiguous [`Array1`]s built with `from_vec`, and treat
//! any solver error or non-finite step as a no-op (returning the starting constants), so this module
//! upholds phop's no-panic / no-`unwrap` policy.

use crate::dataset::DataSet;
use crate::fit::{collect_consts, mse, substitute_consts};
use crate::forest::eval_tree;
use oxieml::EmlTree;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_optimize::least_squares::{least_squares, Method as LsqMethod, Options as LsqOptions};
use scirs2_optimize::unconstrained::{minimize_lbfgs, Options as UncOptions};

/// A large but finite penalty substituted for non-finite residuals/objectives so the solver keeps
/// making progress instead of seeing `NaN`/`inf`.
const PENALTY: f64 = 1e12;

/// Which `scirs2-optimize` algorithm to use for the polish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScirsPolish {
    /// Levenberg–Marquardt least-squares (`scirs2_optimize::least_squares`).
    Lm,
    /// L-BFGS quasi-Newton minimization of the sum of squared residuals.
    Lbfgs,
}

/// Rebuild a tree from a flat constant vector (pre-order), mirroring [`crate::polish`].
fn tree_with(template: &EmlTree, consts: &[f64]) -> EmlTree {
    let mut idx = 0;
    EmlTree::from_node(substitute_consts(&template.root, consts, &mut idx))
}

/// Refine the constant leaves of `tree` against `ds` using a `scirs2-optimize` backend.
///
/// `budget` bounds solver work (max function evaluations for LM, max iterations for L-BFGS). Returns
/// the refined tree and its MSE; if the tree has no constants, or the solver fails / does not
/// improve on the starting fit, the original tree (and its MSE) is returned unchanged.
#[must_use]
pub fn polish_constants_scirs(
    tree: &EmlTree,
    ds: &DataSet,
    budget: usize,
    method: ScirsPolish,
) -> (EmlTree, f64) {
    let mut theta0 = Vec::new();
    collect_consts(&tree.root, &mut theta0);

    let base_mse = eval_tree(tree, &ds.x).map_or(f64::INFINITY, |p| mse(&p, &ds.y));
    if theta0.is_empty() {
        return (tree.clone(), base_mse);
    }

    let refined = match method {
        ScirsPolish::Lm => refine_lm(tree, ds, &theta0, budget),
        ScirsPolish::Lbfgs => refine_lbfgs(tree, ds, &theta0, budget),
    };
    let Some(consts) = refined else {
        return (tree.clone(), base_mse);
    };
    if consts.iter().any(|c| !c.is_finite()) {
        return (tree.clone(), base_mse);
    }

    let candidate = tree_with(tree, &consts);
    match eval_tree(&candidate, &ds.x) {
        Ok(pred) => {
            let m = mse(&pred, &ds.y);
            // Accept only a finite, non-worse fit; otherwise keep the input.
            if m.is_finite() && m <= base_mse {
                (candidate, m)
            } else {
                (tree.clone(), base_mse)
            }
        }
        Err(_) => (tree.clone(), base_mse),
    }
}

/// Levenberg–Marquardt least-squares refinement. The residual vector is `pred(θ) − y`.
fn refine_lm(tree: &EmlTree, ds: &DataSet, theta0: &[f64], max_nfev: usize) -> Option<Vec<f64>> {
    let x = &ds.x;
    let residual = |params: &[f64], data: &[f64]| -> Array1<f64> {
        let t = tree_with(tree, params);
        match eval_tree(&t, x) {
            Ok(pred) => Array1::from_iter(pred.iter().zip(data.iter()).map(|(p, d)| {
                let v = p - d;
                if v.is_finite() {
                    v
                } else {
                    PENALTY
                }
            })),
            Err(_) => Array1::from_elem(data.len(), PENALTY),
        }
    };

    let x0 = Array1::from_vec(theta0.to_vec());
    let data = Array1::from_vec(ds.y.to_vec());
    let opts = LsqOptions {
        max_nfev: Some(max_nfev),
        xtol: Some(1e-12),
        ftol: Some(1e-12),
        gtol: Some(1e-12),
        ..Default::default()
    };
    // No analytic Jacobian: pass `None` (turbofish fixes the unused generic), LM finite-differences.
    let res = least_squares(
        residual,
        &x0,
        LsqMethod::LevenbergMarquardt,
        None::<fn(&[f64], &[f64]) -> Array2<f64>>,
        &data,
        Some(opts),
    )
    .ok()?;
    Some(res.x.to_vec())
}

/// Sum of squared residuals of `tree` (with constants `theta`) against `ds`, or [`PENALTY`] if the
/// forward pass fails or is non-finite.
fn ssr(tree: &EmlTree, ds: &DataSet, theta: &[f64]) -> f64 {
    let t = tree_with(tree, theta);
    match eval_tree(&t, &ds.x) {
        Ok(pred) => {
            let s: f64 = pred
                .iter()
                .zip(ds.y.iter())
                .map(|(p, d)| (p - d) * (p - d))
                .sum();
            if s.is_finite() {
                s
            } else {
                PENALTY
            }
        }
        Err(_) => PENALTY,
    }
}

/// L-BFGS refinement minimizing the sum of squared residuals over the constant vector.
///
/// We supply an explicit forward-difference gradient with a parameter-scaled step (`h = 1e-6·(|θ|+1)`,
/// as in the hand-rolled LM) rather than relying on `scirs2`'s tiny default ε, which converges
/// poorly on the ill-scaled constants that sit inside `exp`/`ln`.
fn refine_lbfgs(tree: &EmlTree, ds: &DataSet, theta0: &[f64], max_iter: usize) -> Option<Vec<f64>> {
    let objective = |params: &ArrayView1<f64>| -> f64 {
        let theta: Vec<f64> = params.iter().copied().collect();
        ssr(tree, ds, &theta)
    };
    let gradient = |params: &ArrayView1<f64>| -> Array1<f64> {
        let theta: Vec<f64> = params.iter().copied().collect();
        let f0 = ssr(tree, ds, &theta);
        let mut g = Array1::zeros(theta.len());
        for (k, gk) in g.iter_mut().enumerate() {
            let h = 1e-6 * (theta[k].abs() + 1.0);
            let mut tp = theta.clone();
            tp[k] += h;
            *gk = (ssr(tree, ds, &tp) - f0) / h;
        }
        g
    };

    let opts = UncOptions {
        max_iter,
        gtol: 1e-12,
        ftol: 1e-14,
        use_gpu: false,
        ..Default::default()
    };
    let res = minimize_lbfgs(
        objective,
        Some(gradient),
        Array1::from_vec(theta0.to_vec()),
        &opts,
    )
    .ok()?;
    Some(res.x.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::collect_consts;
    use scirs2_core::ndarray::Array2;

    fn ds_from(xs: &[f64], ys: &[f64]) -> DataSet {
        let x = Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).expect("shape");
        DataSet::from_arrays(x, Array1::from(ys.to_vec())).expect("dataset")
    }

    fn recovered_c(method: ScirsPolish, start_c: f64) -> (f64, f64) {
        // y = exp(x) - ln(c), true c = 3.0.
        let true_c: f64 = 3.0;
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - true_c.ln()).collect();
        let ds = ds_from(&xs, &ys);
        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(start_c));
        let (refined, m) = polish_constants_scirs(&start, &ds, 200, method);
        let mut consts = Vec::new();
        collect_consts(&refined.root, &mut consts);
        (consts[0], m)
    }

    #[test]
    fn scirs_lm_recovers_constant() {
        // LM is robust enough to cold-start far from the truth.
        let (c, m) = recovered_c(ScirsPolish::Lm, 1.0);
        assert!((c - 3.0).abs() < 1e-3, "LM c = {c}, mse = {m}");
        assert!(m < 1e-6, "LM mse not tight: {m}");
    }

    #[test]
    fn scirs_lbfgs_polishes_a_coarse_fit() {
        // L-BFGS is offered as a *polish* — it refines an already-good (e.g. post-Adam) fit. From a
        // coarse constant it sharpens to the truth; it is not a cold-start solver.
        let (c, m) = recovered_c(ScirsPolish::Lbfgs, 2.7);
        assert!((c - 3.0).abs() < 1e-2, "L-BFGS c = {c}, mse = {m}");
        assert!(m < 1e-4, "L-BFGS mse not tight: {m}");
    }

    #[test]
    fn never_worsens_the_fit() {
        // A constant-free tree is returned unchanged with its base MSE.
        let xs: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let ds = ds_from(&xs, &ys);
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let (_, m) = polish_constants_scirs(&tree, &ds, 50, ScirsPolish::Lm);
        assert!(m < 1e-9, "constant-free exp recovered exactly: mse = {m}");
    }
}
