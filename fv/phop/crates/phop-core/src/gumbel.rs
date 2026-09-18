//! Layer B (M2) — differentiable topology via Gumbel-Softmax leaf selection.
//!
//! Over a fixed complete-binary-tree skeleton (every internal node is `eml`), each leaf
//! *softly* selects its source — one of the input variables or a learnable constant — through
//! a Gumbel-Softmax relaxation. The selection logits and the leaf constants are learned jointly
//! by Adam, with the temperature `tau` annealed from high (exploratory, near-uniform) to low
//! (near-discrete). At the end, each leaf is hardened by `argmax` into a concrete [`EmlTree`].
//!
//! This is the phop contribution that ordinary symbolic regression cannot do: because every
//! internal node is the *same* operator, the only categorical choice left — "which source feeds
//! this leaf" — is linear in tree size and can be relaxed and descended end-to-end.
//!
//! The forward pass is built from fed placeholders and a manual normalized softmax over scalar
//! ([]) logits (`exp`/`div`/`add` plus the proven `mul([],[batch])` broadcast), so it is free of
//! const-generating ops, fully differentiable, and avoids the library's fragile multi-axis
//! broadcast paths. A small population of independent restarts (distinct Gumbel seeds) is run
//! and all results contribute to the Pareto front.

use crate::config::Config;
use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use crate::pareto::ParetoFront;
use crate::solution::Solution;
use oxieml::EmlTree;
use scirs2_autograd as ag;
use scirs2_autograd::optimizers::adam::Adam;
use scirs2_autograd::prelude::*;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array1;
#[cfg(test)]
use scirs2_core::ndarray::Array2;

/// Maximum number of independent restarts (population members) we will run.
const MAX_RESTARTS: usize = 16;

use crate::rng::SplitMix64;

/// Stable `'static` placeholder name for the Gumbel-noise tensor of leaf `l`.
fn gumbel_name(l: usize) -> &'static str {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let m = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = m.lock().expect("gumbel name cache poisoned");
    while v.len() <= l {
        let name: &'static str = Box::leak(format!("phop_g{}", v.len()).into_boxed_str());
        v.push(name);
    }
    v[l]
}

/// A hardened leaf choice produced by `argmax` over a leaf's selection logits.
enum LeafChoice {
    Var(usize),
    Const(f64),
}

/// Build a concrete EML tree from a complete-tree skeleton and per-leaf choices.
fn build_tree(node: usize, internal_count: usize, choices: &[LeafChoice]) -> EmlTree {
    if node >= internal_count {
        match &choices[node - internal_count] {
            LeafChoice::Var(j) => EmlTree::var(*j),
            LeafChoice::Const(c) => EmlTree::const_val(*c),
        }
    } else {
        let l = build_tree(2 * node + 1, internal_count, choices);
        let r = build_tree(2 * node + 1 + 1, internal_count, choices);
        EmlTree::eml(&l, &r)
    }
}

/// Run one Gumbel-Softmax restart; returns the hardened solution it converges to.
fn run_restart(ds: &DataSet, cfg: &Config, depth: usize, seed: u64) -> Result<Solution> {
    let n_vars = ds.n_vars();
    let k = n_vars + 1; // sources: each variable, plus one learnable constant
    let internal_count = (1usize << depth) - 1;
    let n_leaves = 1usize << depth;
    let total = (1usize << (depth + 1)) - 1;
    let batch = ds.len();
    let x = &ds.x;
    let y = &ds.y;

    // Variables: per leaf, `k` scalar selection logits and one scalar constant. Using scalar
    // ([1]) parameters and only proven broadcasts (`mul([1],[batch])`, same-shape `div`) keeps
    // us clear of the library's fragile multi-axis broadcast/softmax/matmul backward paths.
    let mut env = ag::VariableEnvironment::<f64>::new();
    for l in 0..n_leaves {
        for i in 0..k {
            // Scalars (shape []) so div/mul take the well-behaved scalar-broadcast path.
            env.name(format!("z{l}_{i}"))
                .set(scirs2_core::ndarray::arr0(0.0));
        }
        env.name(format!("c{l}"))
            .set(scirs2_core::ndarray::arr0(1.0));
    }
    let ids: Vec<_> = env.default_namespace().current_var_ids();
    // Per-leaf blocks of (k logits, then 1 constant).
    let block = k + 1;
    let logit_id = |l: usize, i: usize| ids[l * block + i];
    let const_id = |l: usize| ids[l * block + k];
    let adam = Adam::new(
        cfg.learning_rate,
        1e-8,
        0.9,
        0.999,
        ids.clone(),
        &mut env,
        "phop_g_adam",
    );

    let col_vals: Vec<Array1<f64>> = (0..n_vars).map(|j| x.column(j).to_owned()).collect();
    let ones_val: Array1<f64> = Array1::from_elem(batch, 1.0);
    let mut rng = SplitMix64::new(seed);

    let _silencer = crate::silence::SilenceStdout::new();
    for epoch in 0..cfg.max_epochs {
        let tau = cfg
            .temperature(epoch as f64 / cfg.max_epochs.max(1) as f64)
            .max(1e-2);
        let inv_tau = 1.0 / tau;
        // Sample one Gumbel noise scalar per (leaf, source) this epoch (shape []).
        let gval_arrays: Vec<_> = (0..n_leaves * k)
            .map(|_| scirs2_core::ndarray::arr0(rng.gumbel()))
            .collect();

        env.run(|g| {
            let cols: Vec<ag::Tensor<f64>> = (0..n_vars)
                .map(|j| g.placeholder(crate::forest::col_placeholder_name(j), &[-1]))
                .collect();
            let ones = g.placeholder("phop_ones", &[-1]);
            let yt = g.placeholder("phop_y", &[-1]);
            // One scalar gumbel placeholder per (leaf, source).
            let gphs: Vec<ag::Tensor<f64>> = (0..n_leaves * k)
                .map(|idx| g.placeholder(gumbel_name(idx), &[]))
                .collect();

            let mut vals: Vec<Option<ag::Tensor<f64>>> = vec![None; total];
            // Differentiable structural penalty (Layer C / M3): the expected number of
            // *non-trivial* (variable-selecting) leaves — i.e. the probability mass each leaf's
            // soft selection places on a variable rather than on the prune-to-constant source.
            // Minimizing it pushes leaves toward the constant `1`, collapsing their subtrees, so
            // it simultaneously expresses the complexity (active-node count) and sparsity
            // (pressure toward `1`) objectives. Weighted below by the Config lambdas.
            let mut struct_pen: Option<ag::Tensor<f64>> = None;
            for l in 0..n_leaves {
                // Sources for this leaf: each variable column, then a learnable constant column.
                let cst = g.variable_by_id(const_id(l));
                let const_col = T::mul(cst, ones); // [batch] via proven [1]x[batch] broadcast
                let source = |i: usize| if i < n_vars { cols[i] } else { const_col };

                // Unnormalized weights a_i = exp((z_i + gumbel_i)/tau); normalize by their sum.
                let a: Vec<ag::Tensor<f64>> = (0..k)
                    .map(|i| {
                        let z = g.variable_by_id(logit_id(l, i));
                        let perturbed = T::add(z, gphs[l * k + i]).scalar_mul(inv_tau);
                        T::exp(T::clip(perturbed, -30.0, 30.0)) // [1]
                    })
                    .collect();
                let mut denom = a[0];
                for ai in a.iter().skip(1) {
                    denom = T::add(denom, *ai); // [1]
                }
                // Normalized soft selection weights w_i = a_i / denom (shape []).
                let weights: Vec<ag::Tensor<f64>> = a.iter().map(|&ai| T::div(ai, denom)).collect();
                // leaf_val = sum_i w_i * source_i  ([batch])
                let mut leaf_val: Option<ag::Tensor<f64>> = None;
                for (i, &w) in weights.iter().enumerate() {
                    let term = T::mul(w, source(i)); // [] x [batch] -> [batch]
                    leaf_val = Some(match leaf_val {
                        None => term,
                        Some(acc) => T::add(acc, term),
                    });
                }
                // Accumulate the probability mass this leaf places on variable sources.
                for &w in weights.iter().take(n_vars) {
                    struct_pen = Some(match struct_pen {
                        None => w,
                        Some(acc) => T::add(acc, w),
                    });
                }
                vals[internal_count + l] = leaf_val;
            }
            for i in (0..internal_count).rev() {
                let a = vals[2 * i + 1].expect("child computed");
                let b = vals[2 * i + 2].expect("child computed");
                vals[i] = Some(crate::forest::eml_guarded(a, b));
            }
            let pred = vals[0].expect("root computed");
            let mut loss = T::reduce_mean(T::square(T::sub(pred, yt)), &[0], false);
            // Add the weighted differentiable structural penalty. The three Config lambdas map to
            // the complexity (active-node count), sparsity (prune-to-`1`), and parsimony (depth)
            // objectives; parsimony scales with the skeleton depth so deeper trees cost more.
            let struct_lambda =
                cfg.lambda_complexity + cfg.lambda_sparsity + cfg.lambda_parsimony * depth as f64;
            if let Some(pen) = struct_pen {
                if struct_lambda != 0.0 {
                    loss = T::add(loss, pen.scalar_mul(struct_lambda));
                }
            }

            let var_tensors: Vec<ag::Tensor<f64>> =
                ids.iter().map(|&vid| g.variable_by_id(vid)).collect();
            let grads = T::grad(&[loss], &var_tensors);
            let mut feeder = ag::Feeder::new();
            for (j, cv) in col_vals.iter().enumerate() {
                feeder = feeder.push(cols[j], cv.view().into_dyn());
            }
            feeder = feeder.push(ones, ones_val.view().into_dyn());
            feeder = feeder.push(yt, y.view().into_dyn());
            for (idx, gph) in gphs.iter().enumerate() {
                feeder = feeder.push(*gph, gval_arrays[idx].view().into_dyn());
            }
            adam.update(&var_tensors, &grads, g, feeder);
        });
    }

    // Harden: argmax each leaf's logits; read its constant.
    let choices: Vec<LeafChoice> = env.run(|g| {
        (0..n_leaves)
            .map(|l| {
                let read = |vid| {
                    g.variable_by_id(vid)
                        .eval(g)
                        .ok()
                        .and_then(|a| a.iter().copied().next())
                        .unwrap_or(0.0)
                };
                let logits: Vec<f64> = (0..k).map(|i| read(logit_id(l, i))).collect();
                let cst = read(const_id(l));
                let best = (0..k)
                    .max_by(|&i, &j| {
                        logits[i]
                            .partial_cmp(&logits[j])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                if best < n_vars {
                    LeafChoice::Var(best)
                } else {
                    LeafChoice::Const(cst)
                }
            })
            .collect()
    });
    drop(_silencer);

    let tree = build_tree(0, internal_count, &choices);
    // Close the soft/hard gap: after hardening, sharpen the leaf constants with Levenberg–Marquardt
    // and snap them to named constants — the coarse Gumbel constants become f64-precise.
    let (polished, _) = crate::polish::polish_constants(&tree, ds, 40);
    let (snapped, m) = crate::polish::snap_constants(&polished, ds, 0.02);
    Ok(Solution::new(snapped, m))
}

/// Discover expressions by differentiable Gumbel-Softmax topology search.
///
/// Runs up to `min(cfg.population, 16)` independent restarts over a complete tree of depth
/// `min(cfg.max_depth, 4)` and returns their Pareto front.
///
/// # Errors
/// Returns [`PhopError`] if the dataset is empty or no restart yields a finite solution.
pub fn discover_gumbel(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let depth = cfg.max_depth.clamp(1, 4);
    let restarts = cfg.population.clamp(1, MAX_RESTARTS);
    let mut sols: Vec<Solution> = Vec::new();
    for r in 0..restarts {
        if let Ok(sol) = run_restart(ds, cfg, depth, cfg.seed.wrapping_add(r as u64 + 1)) {
            if sol.mse.is_finite() {
                sols.push(sol);
            }
        }
    }
    if sols.is_empty() {
        return Err(PhopError::NotConverged(
            "no Gumbel-Softmax restart converged to a finite solution".to_string(),
        ));
    }
    Ok(ParetoFront::from_candidates(sols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gumbel_recovers_exp_structure() {
        // Target y = exp(x0) = eml(x0, 1). A depth-1 skeleton eml(leaf0, leaf1) can express it
        // when the Gumbel search drives leaf0 -> x0 and leaf1 -> a constant near 1.
        let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let y = Array1::from(ys.clone());
        let ds = DataSet::from_arrays(x, y).unwrap();

        let cfg = Config::default()
            .max_depth(1)
            .population(6)
            .max_epochs(1200)
            .learning_rate(0.1);
        let front = discover_gumbel(&ds, &cfg).unwrap();
        assert!(!front.is_empty());

        // Baseline: best constant predictor MSE = variance of y.
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ys.len() as f64;
        let best = front.best().unwrap();
        assert!(
            best.mse < var * 0.5,
            "gumbel best mse {} not below half-variance {} ({})",
            best.mse,
            var * 0.5,
            best.pretty()
        );
    }

    #[test]
    fn structural_penalty_preserves_fit_and_is_deterministic() {
        // With the differentiable structural penalty active (non-zero lambdas), the search must
        // still fit y = exp(x), and a fixed seed must reproduce the same best MSE.
        let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys.clone())).unwrap();

        let mut cfg = Config::default()
            .max_depth(1)
            .population(4)
            .max_epochs(1000)
            .learning_rate(0.1)
            .seed(11);
        cfg.lambda_complexity = 1e-2;
        cfg.lambda_sparsity = 1e-2;
        cfg.lambda_parsimony = 1e-2;

        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ys.len() as f64;

        let a = discover_gumbel(&ds, &cfg).unwrap();
        let b = discover_gumbel(&ds, &cfg).unwrap();
        let best_a = a.best().unwrap();
        assert!(
            best_a.mse < var * 0.5,
            "penalized gumbel did not fit: mse={} ({})",
            best_a.mse,
            best_a.pretty()
        );
        assert!(
            (best_a.mse - b.best().unwrap().mse).abs() < 1e-9,
            "gumbel not deterministic for a fixed seed"
        );
    }
}
