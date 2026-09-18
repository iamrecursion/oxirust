//! M1 — gradient-based fitting of the real-valued constants of an EML tree.
//!
//! Given a *template* tree whose [`EmlNode::Const`] leaves are free parameters, this
//! module fits those constants to data by minimizing mean-squared error with Adam,
//! flowing gradients through phop's guarded autograd forward (Layer A). This is the
//! numeric half of discovery; topology is supplied by the caller (the [`crate::Discoverer`]).

use crate::error::{PhopError, Result};
use crate::forest::eml_guarded;
use oxieml::{EmlNode, EmlTree};
use scirs2_autograd as ag;
use scirs2_autograd::optimizers::adam::Adam;
use scirs2_autograd::prelude::*;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::config::Config;
use crate::dataset::DataSet;

/// Mean-squared error between predictions and targets.
#[must_use]
pub fn mse(pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = y.len().max(1) as f64;
    pred.iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / n
}

/// Number of free constant leaves in a tree.
#[must_use]
pub fn n_constants(tree: &EmlTree) -> usize {
    let mut v = Vec::new();
    collect_consts(&tree.root, &mut v);
    v.len()
}

/// Collect the initial values of every constant leaf, in pre-order.
pub(crate) fn collect_consts(node: &EmlNode, out: &mut Vec<f64>) {
    match node {
        EmlNode::Const(c) => out.push(*c),
        EmlNode::Eml { left, right } => {
            collect_consts(left, out);
            collect_consts(right, out);
        }
        EmlNode::One | EmlNode::Var(_) => {}
    }
}

/// Rebuild a node tree, substituting fitted constant values in pre-order.
pub(crate) fn substitute_consts(node: &EmlNode, fitted: &[f64], idx: &mut usize) -> Arc<EmlNode> {
    match node {
        EmlNode::One => Arc::new(EmlNode::One),
        EmlNode::Var(i) => Arc::new(EmlNode::Var(*i)),
        EmlNode::Const(_) => {
            let v = fitted[*idx];
            *idx += 1;
            Arc::new(EmlNode::Const(v))
        }
        EmlNode::Eml { left, right } => Arc::new(EmlNode::Eml {
            left: substitute_consts(left, fitted, idx),
            right: substitute_consts(right, fitted, idx),
        }),
    }
}

/// Build a parameterized forward graph: constant leaves map to autograd variables.
fn build_param<'g>(
    node: &EmlNode,
    cols: &[ag::Tensor<'g, f64>],
    ones: ag::Tensor<'g, f64>,
    var_tensors: &[ag::Tensor<'g, f64>],
    idx: &mut usize,
) -> ag::Tensor<'g, f64> {
    match node {
        EmlNode::One => ones,
        EmlNode::Var(i) => cols[*i],
        EmlNode::Const(_) => {
            let t = var_tensors[*idx];
            *idx += 1;
            // Broadcast the scalar variable to [batch] via `mul` (whose backward pass
            // correctly reduces the broadcast gradient back to the variable's [1] shape;
            // `sub`'s broadcast backward does not, which would corrupt the Adam update).
            T::mul(t, ones)
        }
        EmlNode::Eml { left, right } => {
            let a = build_param(left, cols, ones, var_tensors, idx);
            let b = build_param(right, cols, ones, var_tensors, idx);
            eml_guarded(a, b)
        }
    }
}

/// Fit the constant leaves of `template` to `ds`, returning the fitted tree and its MSE.
///
/// If the template has no constant leaves, it is evaluated as-is.
///
/// # Errors
/// Returns [`PhopError`] on evaluation failure or non-finite output.
pub fn fit_constants(template: &EmlTree, ds: &DataSet, cfg: &Config) -> Result<(EmlTree, f64)> {
    let mut inits = Vec::new();
    collect_consts(&template.root, &mut inits);
    let batch = ds.len();
    let x: &Array2<f64> = &ds.x;
    let y: &Array1<f64> = &ds.y;

    if inits.is_empty() {
        let pred = crate::forest::eval_tree(template, x)?;
        return Ok((template.clone(), mse(&pred, y)));
    }

    let mut env = ag::VariableEnvironment::<f64>::new();
    for (i, init) in inits.iter().enumerate() {
        env.name(format!("c{i}"))
            .set(scirs2_core::ndarray::arr1(&[*init]));
    }
    let ids: Vec<_> = env.default_namespace().current_var_ids();
    let adam = Adam::new(
        cfg.learning_rate,
        1e-8,
        0.9,
        0.999,
        ids.clone(),
        &mut env,
        "phop_adam",
    );

    // Pre-materialize data columns (fed through placeholders each epoch).
    let col_vals: Vec<Array1<f64>> = (0..ds.n_vars()).map(|j| x.column(j).to_owned()).collect();
    let ones_val: Array1<f64> = Array1::from_elem(batch, 1.0);
    let n_vars = ds.n_vars();

    // Suppress dependency debug output emitted during the gradient/Adam loop.
    let _silencer = crate::silence::SilenceStdout::new();

    for _ in 0..cfg.max_epochs {
        env.run(|g| {
            let cols: Vec<ag::Tensor<f64>> = (0..n_vars)
                .map(|j| g.placeholder(crate::forest::col_placeholder_name(j), &[-1]))
                .collect();
            let ones = g.placeholder("phop_ones", &[-1]);
            let yt = g.placeholder("phop_y", &[-1]);
            let var_tensors: Vec<ag::Tensor<f64>> =
                ids.iter().map(|&vid| g.variable_by_id(vid)).collect();
            let mut idx = 0;
            let pred = build_param(&template.root, &cols, ones, &var_tensors, &mut idx);
            let loss = T::reduce_mean(T::square(T::sub(pred, yt)), &[0], false);
            let grads = T::grad(&[loss], &var_tensors);
            let mut feeder = ag::Feeder::new();
            for (j, col) in cols.iter().enumerate() {
                feeder = feeder.push(*col, col_vals[j].view().into_dyn());
            }
            feeder = feeder.push(ones, ones_val.view().into_dyn());
            feeder = feeder.push(yt, y.view().into_dyn());
            adam.update(&var_tensors, &grads, g, feeder);
        });
    }

    let fitted: Vec<f64> = env.run(|g| {
        ids.iter()
            .map(|&vid| {
                g.variable_by_id(vid)
                    .eval(g)
                    .ok()
                    .and_then(|a| a.iter().copied().next())
                    .unwrap_or(f64::NAN)
            })
            .collect()
    });
    if fitted.iter().any(|v| !v.is_finite()) {
        return Err(PhopError::NumericalInstability(
            "fitted constants are non-finite".to_string(),
        ));
    }

    let mut idx = 0;
    let new_root = substitute_consts(&template.root, &fitted, &mut idx);
    let fitted_tree = EmlTree::from_node(new_root);
    let pred = crate::forest::eval_tree(&fitted_tree, x)?;
    Ok((fitted_tree, mse(&pred, y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_additive_constant() {
        // Model: y = eml(x, c) = exp(x) - ln(c), true c = 3.0. The constant sits in the
        // additive (ln) position, which is well-conditioned for gradient fitting.
        let true_c = 3.0_f64;
        let xs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - true_c.ln()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let y = Array1::from(ys);
        let ds = DataSet::from_arrays(x, y).unwrap();

        // Start the constant away from the truth to prove it is actually learned.
        let template = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let cfg = Config::default().learning_rate(0.05).max_epochs(8000);
        let (fitted, err) = fit_constants(&template, &ds, &cfg).unwrap();

        let mut consts = Vec::new();
        collect_consts(&fitted.root, &mut consts);
        assert_eq!(consts.len(), 1);
        assert!(
            (consts[0] - true_c).abs() < 0.05,
            "recovered c = {} (want {true_c}), mse = {err}",
            consts[0]
        );
        assert!(err < 1e-2, "mse too high: {err}");
    }

    #[test]
    fn fit_reduces_error() {
        // A poorly-initialized constant fit must strictly reduce MSE versus the template.
        let xs: Vec<f64> = (1..=15).map(|i| f64::from(i) * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - 5.0_f64.ln()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let y = Array1::from(ys.clone());
        let ds = DataSet::from_arrays(x, y).unwrap();

        let template = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let before = {
            let pred = crate::forest::eval_tree(&template, &ds.x).unwrap();
            mse(&pred, &ds.y)
        };
        let cfg = Config::default().learning_rate(0.05).max_epochs(4000);
        let (_, after) = fit_constants(&template, &ds, &cfg).unwrap();
        assert!(after < before * 0.5, "before={before} after={after}");
    }
}
