//! Layer A — tensorized EML forest forward evaluation.
//!
//! This module builds an autograd forward graph for an EML tree (or, later, a whole
//! population) over a batch of data, with **guarded** evaluation so that `exp` cannot
//! overflow and `ln` is never applied to non-positive values (Risk T1).
//!
//! Data enters through fed **placeholders** rather than `constant` nodes: this is the
//! idiomatic scirs2-autograd path, avoids per-call allocation, and sidesteps verbose
//! debug output emitted by the constant op.

use crate::error::{PhopError, Result};
use oxieml::{EmlNode, EmlTree};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_autograd::Context;
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::{Mutex, OnceLock};

/// Lower clamp on the argument of `ln` (keeps it strictly positive).
pub const LN_EPS: f64 = 1e-12;
/// Symmetric clamp on the argument of `exp` (prevents overflow to `inf`).
pub const EXP_CLAMP: f64 = 50.0;

/// Stable `'static` placeholder name for feature column `j` (leaked once, then cached).
pub(crate) fn col_placeholder_name(j: usize) -> &'static str {
    static CACHE: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let m = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = m.lock().expect("placeholder name cache poisoned");
    while v.len() <= j {
        let name: &'static str = Box::leak(format!("phop_x{}", v.len()).into_boxed_str());
        v.push(name);
    }
    v[j]
}

/// Guarded EML primitive on graph tensors: `eml(a, b) = exp(clip(a)) - ln(clip(b))`.
pub fn eml_guarded<'g>(a: ag::Tensor<'g, f64>, b: ag::Tensor<'g, f64>) -> ag::Tensor<'g, f64> {
    let ea = T::exp(T::clip(a, -EXP_CLAMP, EXP_CLAMP));
    let lb = T::ln(T::clip(b, LN_EPS, f64::MAX));
    T::sub(ea, lb)
}

/// Build the forward graph for an EML node over per-variable column tensors.
///
/// `ones` is a fed `[batch]` tensor of ones; deriving every leaf from it (rather than from
/// `ones`/`constant` ops) keeps the graph free of const-generating ops. The result is a
/// `[batch]` prediction tensor: `1` leaves become `ones`, `Const(c)` becomes `ones * c`.
pub(crate) fn build_forward<'g>(
    node: &EmlNode,
    cols: &[ag::Tensor<'g, f64>],
    ones: ag::Tensor<'g, f64>,
    _g: &'g Context<f64>,
) -> ag::Tensor<'g, f64> {
    match node {
        EmlNode::One => ones,
        EmlNode::Const(c) => ones.scalar_mul(*c),
        EmlNode::Var(i) => cols[*i],
        EmlNode::Eml { left, right } => {
            let a = build_forward(left, cols, ones, _g);
            let b = build_forward(right, cols, ones, _g);
            eml_guarded(a, b)
        }
    }
}

/// Evaluate an [`EmlTree`] forward through autograd over the given data.
///
/// Returns predictions of length `data.nrows()`.
///
/// # Errors
/// Returns [`PhopError::Eval`] if the autograd evaluation fails, or
/// [`PhopError::NumericalInstability`] if the output contains non-finite values.
pub fn eval_tree(tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
    let batch = data.nrows();
    let n_vars = data.ncols();
    let col_vals: Vec<Array1<f64>> = (0..n_vars).map(|j| data.column(j).to_owned()).collect();
    let ones_val: Array1<f64> = Array1::from_elem(batch, 1.0);

    let values: std::result::Result<Vec<f64>, String> = ag::run(|g: &mut Context<f64>| {
        let cols: Vec<ag::Tensor<f64>> = (0..n_vars)
            .map(|j| g.placeholder(col_placeholder_name(j), &[-1]))
            .collect();
        let ones = g.placeholder("phop_ones", &[-1]);
        let out = build_forward(&tree.root, &cols, ones, g);
        let mut feeder = ag::Feeder::new();
        for (j, col) in cols.iter().enumerate() {
            feeder = feeder.push(*col, col_vals[j].view().into_dyn());
        }
        feeder = feeder.push(ones, ones_val.view().into_dyn());
        let results = g.evaluator().push(&out).set_feeder(feeder).run();
        match &results[0] {
            Ok(arr) => Ok(arr.iter().copied().collect()),
            Err(e) => Err(format!("{e:?}")),
        }
    });

    let values = values.map_err(PhopError::Eval)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PhopError::NumericalInstability(
            "forward pass produced non-finite values".to_string(),
        ));
    }
    Ok(Array1::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxieml::{Canonical, EvalCtx};

    #[test]
    fn forward_matches_oxieml_exp() {
        let x = EmlTree::var(0);
        let tree = Canonical::exp(&x);
        let data = Array2::from_shape_vec((4, 1), vec![0.0, 0.5, 1.0, 2.0]).unwrap();

        let ours = eval_tree(&tree, &data).unwrap();
        for i in 0..4 {
            let ctx = EvalCtx::new(&[data[[i, 0]]]);
            let theirs = tree.eval_real(&ctx).unwrap();
            assert!(
                (ours[i] - theirs).abs() < 1e-9,
                "row {i}: ours={} theirs={theirs}",
                ours[i]
            );
        }
    }

    #[test]
    fn forward_matches_oxieml_two_vars() {
        let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::var(1));
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 2.0, 0.5, 3.0]).unwrap();
        let ours = eval_tree(&t, &data).unwrap();
        for i in 0..3 {
            let ctx = EvalCtx::new(&[data[[i, 0]], data[[i, 1]]]);
            let theirs = t.eval_real(&ctx).unwrap();
            assert!((ours[i] - theirs).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn const_leaf_evaluates() {
        // eml(x0, c) = exp(x0) - ln(c)
        let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(2.0));
        let data = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let ours = eval_tree(&t, &data).unwrap();
        for i in 0..2 {
            let ctx = EvalCtx::new(&[data[[i, 0]]]);
            let theirs = t.eval_real(&ctx).unwrap();
            assert!((ours[i] - theirs).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn guarded_eval_stays_finite_on_extremes(/* Risk T1 */) {
        // eml(x0, x1) on inputs that would overflow exp or take ln of a non-positive number:
        // the guard (clip exp arg to ±50, clip ln arg to >= 1e-12) must keep the output finite.
        let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::var(1));
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1000.0,
                -5.0, // exp(1000) would be inf; ln(-5) is undefined
                -1000.0,
                0.0, // exp(-1000) underflows; ln(0) is -inf
                1e308,
                1e-300, // extreme magnitudes
                f64::from(0),
                1.0,
            ],
        )
        .unwrap();
        let out = eval_tree(&t, &data).expect("guarded eval must not error on extremes");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "guarded eval produced a non-finite value: {out:?}"
        );
    }
}
