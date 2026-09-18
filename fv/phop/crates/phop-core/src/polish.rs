//! Post-fit refinement of an expression's constants.
//!
//! Gradient descent (Adam) gives only coarse constants — especially where a constant sits
//! inside `exp` and the problem is badly conditioned. This module refines them with
//! **Levenberg–Marquardt** (a small dense least-squares solve with a finite-difference
//! Jacobian over the few constant leaves), which converges quadratically near the optimum, and
//! then **snaps** constants to recognizable named values (π, e, √2, small integers/rationals)
//! when doing so does not meaningfully worsen the fit.

use crate::dataset::DataSet;
use crate::fit::{collect_consts, mse, substitute_consts};
use crate::forest::eval_tree;
use crate::loss::RobustLoss;
use oxieml::symreg::snap_to_named_const;
use oxieml::EmlTree;
use scirs2_core::ndarray::Array1;

/// Rebuild a tree from a flat constant vector (pre-order).
fn tree_with_consts(template: &EmlTree, consts: &[f64]) -> EmlTree {
    let mut idx = 0;
    EmlTree::from_node(substitute_consts(&template.root, consts, &mut idx))
}

/// Residual vector `pred - y`, or `None` if any prediction is not finite.
fn residuals(pred: &Array1<f64>, y: &Array1<f64>) -> Option<Vec<f64>> {
    let mut r = Vec::with_capacity(pred.len());
    for (p, t) in pred.iter().zip(y.iter()) {
        if !p.is_finite() {
            return None;
        }
        r.push(p - t);
    }
    Some(r)
}

/// Solve the small dense system `a x = b` by Gaussian elimination with partial pivoting.
/// Returns `None` if the matrix is singular. Shared with [`crate::affine`]'s LM fitter.
pub(crate) fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        // Eliminate below. Clone the pivot row so the inner update borrows `a[r]` mutably
        // without aliasing the pivot row `a[col]`.
        let pivot_row = a[col].clone();
        let pivot_b = b[col];
        for r in (col + 1)..n {
            let f = a[r][col] / pivot_row[col];
            for (rc, pc) in a[r].iter_mut().zip(pivot_row.iter()).skip(col) {
                *rc -= f * pc;
            }
            b[r] -= f * pivot_b;
        }
    }
    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Refine the constant leaves of `tree` against `ds` with Levenberg–Marquardt (plain MSE).
///
/// Returns the refined tree and its MSE. If the tree has no constants, it is returned as-is.
#[must_use]
pub fn polish_constants(tree: &EmlTree, ds: &DataSet, iters: usize) -> (EmlTree, f64) {
    lm_refine(tree, ds, iters, RobustLoss::Mse)
}

/// Refine the constant leaves of `tree` against `ds` with a **robust** Levenberg–Marquardt polish.
///
/// Uses IRLS: at each iteration the residual and Jacobian rows are reweighted by `loss` so gross
/// outliers stop dominating the fit (see [`crate::loss`]). With [`RobustLoss::Mse`] this is exactly
/// [`polish_constants`]. Returns the refined tree and its plain MSE on the (full) data.
#[must_use]
pub fn polish_constants_robust(
    tree: &EmlTree,
    ds: &DataSet,
    iters: usize,
    loss: RobustLoss,
) -> (EmlTree, f64) {
    lm_refine(tree, ds, iters, loss)
}

/// Shared Levenberg–Marquardt core. Minimizes the IRLS surrogate of `loss`: each outer iteration
/// freezes per-row weights `w_i` from the current residuals and takes a damped Gauss–Newton step on
/// the weighted least-squares problem `min ‖W(pred − y)‖²`. For `RobustLoss::Mse` (`w_i ≡ 1`) this
/// is ordinary LM.
fn lm_refine(tree: &EmlTree, ds: &DataSet, iters: usize, loss: RobustLoss) -> (EmlTree, f64) {
    let mut theta = Vec::new();
    collect_consts(&tree.root, &mut theta);
    let p = theta.len();
    let y = &ds.y;

    let eval = |th: &[f64]| -> Option<Array1<f64>> {
        let t = tree_with_consts(tree, th);
        eval_tree(&t, &ds.x).ok()
    };

    if p == 0 {
        let m = eval(&theta).map_or(f64::INFINITY, |pred| mse(&pred, y));
        return (tree.clone(), m);
    }

    let n = y.len();
    let mut pred = match eval(&theta) {
        Some(p) => p,
        None => return (tree.clone(), f64::INFINITY),
    };
    let mut r = match residuals(&pred, y) {
        Some(r) => r,
        None => return (tree.clone(), f64::INFINITY),
    };
    let mut lambda = 1e-3_f64;

    for _ in 0..iters {
        // Freeze IRLS weights from the current residuals; `wcost` is the surrogate to decrease.
        let w: Vec<f64> = r.iter().map(|&ri| loss.irls_weight(ri, &r)).collect();
        let wcost: f64 = r.iter().zip(&w).map(|(ri, wi)| (wi * ri) * (wi * ri)).sum();

        // Finite-difference Jacobian J [n x p].
        let mut jac: Vec<Vec<f64>> = vec![vec![0.0; p]; n];
        let mut ok = true;
        for j in 0..p {
            let h = 1e-6 * (theta[j].abs() + 1.0);
            let mut th = theta.clone();
            th[j] += h;
            let Some(pj) = eval(&th) else {
                ok = false;
                break;
            };
            for i in 0..n {
                jac[i][j] = (pj[i] - pred[i]) / h;
            }
        }
        if !ok {
            break;
        }

        // Weighted normal equations: A = (WJ)ᵀ(WJ) = Jᵀ W² J, g = (WJ)ᵀ(Wr) = Jᵀ W² r.
        let mut a = vec![vec![0.0; p]; p];
        let mut grad = vec![0.0; p];
        for col in 0..p {
            for row in 0..n {
                let w2 = w[row] * w[row];
                grad[col] += w2 * jac[row][col] * r[row];
            }
            for col2 in col..p {
                let mut s = 0.0;
                for (row, jrow) in jac.iter().enumerate() {
                    s += w[row] * w[row] * jrow[col] * jrow[col2];
                }
                a[col][col2] = s;
                a[col2][col] = s;
            }
        }

        // Try damped steps, increasing lambda until one decreases the (frozen-weight) surrogate.
        let mut accepted = false;
        for _ in 0..12 {
            let mut a_damped = a.clone();
            for d in 0..p {
                a_damped[d][d] += lambda * a[d][d].max(1e-12);
            }
            let rhs: Vec<f64> = grad.iter().map(|g| -g).collect();
            let Some(delta) = solve_dense(a_damped, rhs) else {
                lambda *= 4.0;
                continue;
            };
            let cand: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
            if let Some(p_new) = eval(&cand) {
                if let Some(r_new) = residuals(&p_new, y) {
                    let wcost_new: f64 = r_new
                        .iter()
                        .zip(&w)
                        .map(|(ri, wi)| (wi * ri) * (wi * ri))
                        .sum();
                    if wcost_new < wcost {
                        theta = cand;
                        pred = p_new;
                        r = r_new;
                        lambda = (lambda * 0.5).max(1e-12);
                        accepted = true;
                        break;
                    }
                }
            }
            lambda *= 4.0;
        }
        if !accepted {
            break; // converged or stuck
        }
    }

    let refined = tree_with_consts(tree, &theta);
    let m = mse(&pred, y);
    (refined, m)
}

/// Maximum distance from a clean target (integer / small rational) for it to be *proposed* as a snap
/// candidate. Generous on purpose — the MSE-tolerance check below is the real gate, so this only has
/// to exceed the post-polish fit jitter while staying well under the spacing of meaningful constants.
const SNAP_CAND_EPS: f64 = 1e-3;

/// "Clean" snap candidates for a fitted constant, **simplest first**: a nearby small integer, a named
/// mathematical constant (π, e, √2, …), then a small rational `k/d` (`d ∈ {2,3,4,6}`, `|k/d| ≤ 12`).
/// The caller accepts the first candidate that keeps the fit within tolerance, so over-proposing here
/// is safe. Crucially this includes **plain integers** — `snap_to_named_const` alone never rounds e.g.
/// a fitted `1.0000000438` to `1`, which leaves an `ln(1)≠0` residue that breaks exact-equivalence.
fn snap_candidates(c: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let r = c.round();
    if (c - r).abs() < SNAP_CAND_EPS && r.abs() <= 1e6 {
        out.push(r);
    }
    if let Some(nc) = snap_to_named_const(c) {
        out.push(nc.value());
    }
    for d in [2.0, 3.0, 4.0, 6.0] {
        let q = (c * d).round() / d;
        if (c - q).abs() < SNAP_CAND_EPS && q.abs() <= 12.0 {
            out.push(q);
        }
    }
    out
}

/// Snap each constant to a recognizable clean value — a small integer, a named constant (π, e, √2, …),
/// or a small rational — when the snap keeps the MSE within `rel_tol` of the unsnapped fit. Constants
/// are snapped one at a time (simplest candidate first), each accepted only if it does not meaningfully
/// worsen the fit, so a constant that genuinely is `1.000…` collapses to exactly `1` (removing the
/// `ln(1)` residue that otherwise defeats exact-equivalence proofs). Returns the snapped tree + MSE.
#[must_use]
pub fn snap_constants(tree: &EmlTree, ds: &DataSet, rel_tol: f64) -> (EmlTree, f64) {
    let mut theta = Vec::new();
    collect_consts(&tree.root, &mut theta);
    let y = &ds.y;
    let base = match eval_tree(tree, &ds.x) {
        Ok(pred) => mse(&pred, y),
        Err(_) => return (tree.clone(), f64::INFINITY),
    };
    if theta.is_empty() {
        return (tree.clone(), base);
    }
    let tol = base * (1.0 + rel_tol) + 1e-12;
    let mut changed = false;
    for j in 0..theta.len() {
        for cv in snap_candidates(theta[j]) {
            if cv == theta[j] {
                continue; // already *exactly* this value (a near-but-inexact value must still snap,
                          // or a machine-epsilon residue like ln(1+4e-16)≠0 defeats equivalence proofs)
            }
            let mut cand = theta.clone();
            cand[j] = cv;
            let t = tree_with_consts(tree, &cand);
            if let Ok(pred) = eval_tree(&t, &ds.x) {
                if mse(&pred, y) <= tol {
                    theta = cand; // keep the simplest candidate that holds
                    changed = true;
                    break;
                }
            }
        }
    }
    if !changed {
        return (tree.clone(), base);
    }
    let snapped = tree_with_consts(tree, &theta);
    let m = eval_tree(&snapped, &ds.x).map_or(base, |pred| mse(&pred, y));
    (snapped, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn ds_from(xs: &[f64], ys: &[f64]) -> DataSet {
        let x = Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).unwrap();
        DataSet::from_arrays(x, Array1::from(ys.to_vec())).unwrap()
    }

    #[test]
    fn lm_recovers_constant_precisely() {
        // y = exp(x) - ln(c), true c = 3.0; start far away and polish to high precision.
        let true_c: f64 = 3.0;
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - true_c.ln()).collect();
        let ds = ds_from(&xs, &ys);

        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let (refined, m) = polish_constants(&start, &ds, 50);
        let mut consts = Vec::new();
        collect_consts(&refined.root, &mut consts);
        assert!(
            (consts[0] - true_c).abs() < 1e-4,
            "c = {} (want {true_c}), mse = {m}",
            consts[0]
        );
        assert!(m < 1e-8, "mse not tight: {m}");
    }

    #[test]
    fn robust_polish_resists_outliers() {
        // y = exp(x) - ln(c) is a pure vertical-offset model (b = -ln c). True c = 3.0. Corrupt
        // four points with gross +50 outliers. Plain MSE chases the squared penalty and the offset
        // collapses; the robust losses ignore the outliers and recover c. Both robust variants must
        // beat MSE; the trimmed loss (which hard-drops the worst residuals) recovers c near-exactly.
        let true_c: f64 = 3.0;
        let xs: Vec<f64> = (1..=30).map(|i| f64::from(i) * 0.1).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| x.exp() - true_c.ln()).collect();
        for &k in &[3usize, 11, 19, 27] {
            ys[k] += 50.0;
        }
        let ds = ds_from(&xs, &ys);

        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let consts_of = |t: &EmlTree| {
            let mut v = Vec::new();
            collect_consts(&t.root, &mut v);
            v[0]
        };

        let mse_c = consts_of(&polish_constants(&start, &ds, 80).0);
        let hub_c = consts_of(
            &polish_constants_robust(&start, &ds, 80, RobustLoss::Huber { delta: 0.2 }).0,
        );
        let trim_c = consts_of(
            &polish_constants_robust(&start, &ds, 80, RobustLoss::Trimmed { alpha: 0.2 }).0,
        );

        let (mse_err, hub_err, trim_err) = (
            (mse_c - true_c).abs(),
            (hub_c - true_c).abs(),
            (trim_c - true_c).abs(),
        );
        assert!(
            hub_err < mse_err,
            "Huber ({hub_c}) did not beat MSE ({mse_c})"
        );
        assert!(
            trim_err < mse_err,
            "Trimmed ({trim_c}) did not beat MSE ({mse_c})"
        );
        assert!(
            trim_err < 0.1,
            "Trimmed did not recover c: {trim_c} (want {true_c})"
        );
    }

    #[test]
    fn snaps_near_one_removes_ln_residue() {
        // The screen-1 bug: a fitted `eml(x0, 1.0000000438)` is exp(x0) − ln(1.0000000438), i.e.
        // exp(x0) − 4.4e-8, which defeats exact-equivalence. Snapping must round the constant to
        // exactly 1 so the law equals exp(x0) and the residue vanishes.
        let xs: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.15).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let ds = ds_from(&xs, &ys);
        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.000_000_043_8));
        let (snapped, m) = snap_constants(&start, &ds, 0.02);
        let mut consts = Vec::new();
        collect_consts(&snapped.root, &mut consts);
        assert!(
            (consts[0] - 1.0).abs() < 1e-12,
            "did not snap to exactly 1: {}",
            consts[0]
        );
        assert!(m < 1e-20, "snapped mse should be machine-zero, got {m}");
    }

    #[test]
    fn snaps_near_integer_constant() {
        // y = exp(x) − ln(2); a constant fitted to 2.0000003 must snap to exactly 2.
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - 2.0_f64.ln()).collect();
        let ds = ds_from(&xs, &ys);
        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(2.000_000_3));
        let (snapped, _) = snap_constants(&start, &ds, 0.02);
        let mut consts = Vec::new();
        collect_consts(&snapped.root, &mut consts);
        assert!(
            (consts[0] - 2.0).abs() < 1e-12,
            "did not snap to 2: {}",
            consts[0]
        );
    }

    #[test]
    fn does_not_snap_a_genuinely_fractional_constant() {
        // y = exp(x) − ln(1.37): 1.37 is not near any clean target, so it must be left intact.
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - 1.37_f64.ln()).collect();
        let ds = ds_from(&xs, &ys);
        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.37));
        let (snapped, _) = snap_constants(&start, &ds, 0.02);
        let mut consts = Vec::new();
        collect_consts(&snapped.root, &mut consts);
        assert!(
            (consts[0] - 1.37).abs() < 1e-9,
            "wrongly snapped 1.37 to {}",
            consts[0]
        );
    }

    #[test]
    fn snaps_to_pi() {
        // y = exp(x) - ln(pi); polish then snap should land exactly on pi.
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| x.exp() - std::f64::consts::PI.ln())
            .collect();
        let ds = ds_from(&xs, &ys);

        let start = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(3.0));
        let (refined, _) = polish_constants(&start, &ds, 50);
        let (snapped, _) = snap_constants(&refined, &ds, 0.05);
        let mut consts = Vec::new();
        collect_consts(&snapped.root, &mut consts);
        assert!(
            (consts[0] - std::f64::consts::PI).abs() < 1e-12,
            "did not snap to pi: {}",
            consts[0]
        );
    }
}
