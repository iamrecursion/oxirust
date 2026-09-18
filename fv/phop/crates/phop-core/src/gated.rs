//! Differentiable *tree shape* via per-node expand/terminate gates (the depth-learning step).
//!
//! [`crate::gumbel`] relaxes only *which source feeds each leaf* over a **fixed** complete-tree
//! skeleton. This module additionally makes the tree's **depth and shape** differentiable: over a
//! maximal complete tree, every node carries its own soft source selection (so it can act as a
//! leaf) and every *internal* node carries a learnable **gate** `σ(z_n)`:
//!
//! ```text
//! val(n) = leaf(n) + σ(z_n) · ( eml(val(2n+1), val(2n+2)) − leaf(n) )
//! ```
//!
//! With `σ → 0` the node *terminates* (it is its own leaf and its subtree is pruned); with
//! `σ → 1` it *expands* into `eml(L, R)`. A complexity penalty `λ·Σ σ(z_n)` (the expected number
//! of expanded nodes) pressures the search toward shallow trees; at the end each gate is hardened
//! by a `0.5` threshold and the discrete tree is read off top-down.
//!
//! Why not DARTS-style "skip/zero" pruning? `eml` has **no identity element** — there is no
//! constant `b` with `exp(a) − ln(b) = a` for all `a` — so a subtree can only be collapsed by
//! *terminating the node into a leaf*, which is exactly what the gate does.
//!
//! This is the CPU (autograd) reference implementation; like [`crate::gumbel`] it builds one
//! `scirs2-autograd` graph per epoch and descends logits, leaf constants, and gates jointly.

use crate::config::Config;
use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use crate::pareto::ParetoFront;
use crate::rng::SplitMix64;
use crate::solution::Solution;
use oxieml::{EmlNode, EmlTree};
use scirs2_autograd as ag;
use scirs2_autograd::optimizers::adam::Adam;
use scirs2_autograd::prelude::*;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array1;
#[cfg(test)]
use scirs2_core::ndarray::Array2;

/// Maximum number of independent restarts (population members).
const MAX_RESTARTS: usize = 16;

/// Stable `'static` placeholder name for the Gumbel-noise scalar of (node, source) `idx`.
fn gate_gumbel_name(idx: usize) -> &'static str {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let m = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = m.lock().expect("gate gumbel name cache poisoned");
    while v.len() <= idx {
        let name: &'static str = Box::leak(format!("phop_gate_g{}", v.len()).into_boxed_str());
        v.push(name);
    }
    v[idx]
}

/// A hardened leaf choice: a variable column or a learned constant.
enum LeafChoice {
    Var(usize),
    Const(f64),
}

/// Read the discrete tree top-down: a node is a leaf if it is a bottom leaf or its gate is off.
fn harden(
    node: usize,
    internal_count: usize,
    expanded: &[bool],
    choices: &[LeafChoice],
) -> EmlTree {
    let is_leaf = node >= internal_count || !expanded[node];
    if is_leaf {
        match &choices[node] {
            LeafChoice::Var(j) => EmlTree::var(*j),
            LeafChoice::Const(c) => EmlTree::const_val(*c),
        }
    } else {
        let l = harden(2 * node + 1, internal_count, expanded, choices);
        let r = harden(2 * node + 2, internal_count, expanded, choices);
        EmlTree::eml(&l, &r)
    }
}

/// Warm-start initialization for one gated restart: per-node leaf-selection logits and constants,
/// and per-internal-node gate logits. Built from a discrete seed tree by [`seed_to_init`].
struct GatedInit {
    leaf_logits: Vec<Vec<f64>>, // [total][k]
    consts: Vec<f64>,           // [total]
    gates: Vec<f64>,            // [internal_count] (index 0 = root, unused)
}

/// Run one gated restart; returns the hardened solution it converges to.
///
/// `init` warm-starts the logits/constants/gates (e.g. from an `enumerate` seed); `None` uses the
/// cold depth curriculum (root expanded, deeper nodes terminated, uniform leaves).
fn run_restart(
    ds: &DataSet,
    cfg: &Config,
    depth: usize,
    seed: u64,
    init: Option<&GatedInit>,
) -> Result<Solution> {
    let n_vars = ds.n_vars();
    let k = n_vars + 1; // sources per node: each variable, plus a learnable constant
    let internal_count = (1usize << depth) - 1;
    let total = (1usize << (depth + 1)) - 1;
    let batch = ds.len();
    let x = &ds.x;
    let y = &ds.y;

    // Variables: per node, k selection logits + 1 constant (block of k+1); per internal node, a
    // gate logit. All are scalars (shape []) so div/mul take the well-behaved broadcast paths.
    let mut env = ag::VariableEnvironment::<f64>::new();
    for n in 0..total {
        for i in 0..k {
            let v = init.map_or(0.0, |gi| gi.leaf_logits[n][i]);
            env.name(format!("gz{n}_{i}"))
                .set(scirs2_core::ndarray::arr0(v));
        }
        let cv = init.map_or(1.0, |gi| gi.consts[n]);
        env.name(format!("gc{n}"))
            .set(scirs2_core::ndarray::arr0(cv)); // constant leaf
    }
    // Gates for the *non-root* internal nodes only. The root always expands (a single-node tree is
    // never the goal — `enumerate`/`gumbel` cover that), which removes the dominant failure mode:
    // a soft root over garbage children is locally worse than terminating, so a gated root collapses
    // to a leaf. Cold start: every gated node starts terminated (σ(−3) ≈ 0.05) — a depth curriculum.
    // Warm start: gates come from the seed tree's shape.
    for n in 1..internal_count {
        let gv = init.map_or(-3.0, |gi| gi.gates[n]);
        env.name(format!("gg{n}"))
            .set(scirs2_core::ndarray::arr0(gv));
    }
    let ids: Vec<_> = env.default_namespace().current_var_ids();
    let block = k + 1;
    let logit_id = |n: usize, i: usize| ids[n * block + i];
    let const_id = |n: usize| ids[n * block + k];
    // Gate vars are stored for nodes 1..internal_count, in order, after all node blocks.
    let gate_id = |n: usize| ids[total * block + (n - 1)];

    let adam = Adam::new(
        cfg.learning_rate,
        1e-8,
        0.9,
        0.999,
        ids.clone(),
        &mut env,
        "phop_gate_adam",
    );

    let col_vals: Vec<Array1<f64>> = (0..n_vars).map(|j| x.column(j).to_owned()).collect();
    let ones_val: Array1<f64> = Array1::from_elem(batch, 1.0);
    let lambda = cfg.lambda_complexity + cfg.lambda_parsimony * depth as f64;
    let mut rng = SplitMix64::new(seed);

    let _silencer = crate::silence::SilenceStdout::new();
    for epoch in 0..cfg.max_epochs {
        let tau = cfg
            .temperature(epoch as f64 / cfg.max_epochs.max(1) as f64)
            .max(1e-2);
        let inv_tau = 1.0 / tau;
        let gval_arrays: Vec<_> = (0..total * k)
            .map(|_| scirs2_core::ndarray::arr0(rng.gumbel()))
            .collect();

        env.run(|g| {
            let cols: Vec<ag::Tensor<f64>> = (0..n_vars)
                .map(|j| g.placeholder(crate::forest::col_placeholder_name(j), &[-1]))
                .collect();
            let ones = g.placeholder("phop_ones", &[-1]);
            let yt = g.placeholder("phop_y", &[-1]);
            let gphs: Vec<ag::Tensor<f64>> = (0..total * k)
                .map(|idx| g.placeholder(gate_gumbel_name(idx), &[]))
                .collect();

            // Per-node soft "leaf" value (the source mixture, as in the Gumbel leaf relaxation).
            let mut leaf_vals: Vec<ag::Tensor<f64>> = Vec::with_capacity(total);
            for n in 0..total {
                let cst = g.variable_by_id(const_id(n));
                let const_col = T::mul(cst, ones);
                let source = |i: usize| if i < n_vars { cols[i] } else { const_col };
                let a: Vec<ag::Tensor<f64>> = (0..k)
                    .map(|i| {
                        let z = g.variable_by_id(logit_id(n, i));
                        let perturbed = T::add(z, gphs[n * k + i]).scalar_mul(inv_tau);
                        T::exp(T::clip(perturbed, -30.0, 30.0))
                    })
                    .collect();
                let mut denom = a[0];
                for ai in a.iter().skip(1) {
                    denom = T::add(denom, *ai);
                }
                let mut lv: Option<ag::Tensor<f64>> = None;
                for (i, &ai) in a.iter().enumerate() {
                    let w = T::div(ai, denom);
                    let term = T::mul(w, source(i));
                    lv = Some(match lv {
                        None => term,
                        Some(acc) => T::add(acc, term),
                    });
                }
                leaf_vals.push(lv.expect("k >= 1"));
            }

            // Bottom-up gated composition; accumulate the expected-expansion penalty.
            let mut vals: Vec<Option<ag::Tensor<f64>>> = vec![None; total];
            for (n, lv) in leaf_vals.iter().enumerate().skip(internal_count) {
                vals[n] = Some(*lv);
            }
            let mut pen: Option<ag::Tensor<f64>> = None;
            for n in (0..internal_count).rev() {
                let left = vals[2 * n + 1].expect("child computed");
                let right = vals[2 * n + 2].expect("child computed");
                let expand = crate::forest::eml_guarded(left, right);
                if n == 0 {
                    // Root always expands.
                    vals[n] = Some(expand);
                } else {
                    let gate = T::sigmoid(g.variable_by_id(gate_id(n)));
                    let diff = T::sub(expand, leaf_vals[n]);
                    let gated = T::add(leaf_vals[n], T::mul(gate, diff));
                    vals[n] = Some(gated);
                    pen = Some(match pen {
                        None => gate,
                        Some(acc) => T::add(acc, gate),
                    });
                }
            }

            let pred = vals[0].expect("root computed");
            let mut loss = T::reduce_mean(T::square(T::sub(pred, yt)), &[0], false);
            if lambda != 0.0 {
                if let Some(p) = pen {
                    loss = T::add(loss, p.scalar_mul(lambda));
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

    // Harden: argmax each node's source; a node expands iff its gate logit >= 0 (σ >= 0.5).
    let (choices, gate_logits) = env.run(|g| {
        let read = |vid| {
            g.variable_by_id(vid)
                .eval(g)
                .ok()
                .and_then(|a| a.iter().copied().next())
                .unwrap_or(0.0)
        };
        let choices: Vec<LeafChoice> = (0..total)
            .map(|n| {
                let logits: Vec<f64> = (0..k).map(|i| read(logit_id(n, i))).collect();
                let cst = read(const_id(n));
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
            .collect();
        // Gate logit per non-root internal node (root has none).
        let gate_logits: Vec<f64> = (1..internal_count).map(|n| read(gate_id(n))).collect();
        (choices, gate_logits)
    });
    drop(_silencer);
    // Root (node 0) always expands; nodes 1.. expand iff their gate logit >= 0 (σ >= 0.5).
    let mut expanded = vec![true; internal_count];
    for (n, &z) in gate_logits.iter().enumerate() {
        expanded[n + 1] = z >= 0.0;
    }

    let tree = harden(0, internal_count, &expanded, &choices);
    // Close the soft/hard gap: sharpen the hardened tree's constants (LM polish + named-const snap).
    let (polished, _) = crate::polish::polish_constants(&tree, ds, 40);
    let (snapped, m) = crate::polish::snap_constants(&polished, ds, 0.02);
    Ok(Solution::new(snapped, m))
}

/// Discover expressions by differentiable **tree-shape** search (per-node expand/terminate gates).
///
/// Runs up to `min(cfg.population, 16)` restarts over a maximal complete tree of depth
/// `min(cfg.max_depth, 4)`, learning each node's source selection **and** whether it expands, and
/// returns the Pareto front of the hardened trees.
///
/// # Errors
/// Returns [`PhopError`] if the dataset is empty or no restart yields a finite solution.
pub fn discover_gated(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let depth = cfg.max_depth.clamp(1, 4);
    let restarts = cfg.population.clamp(1, MAX_RESTARTS);
    let mut sols: Vec<Solution> = Vec::new();
    for r in 0..restarts {
        if let Ok(sol) = run_restart(ds, cfg, depth, cfg.seed.wrapping_add(r as u64 + 1), None) {
            if sol.mse.is_finite() {
                sols.push(sol);
            }
        }
    }
    if sols.is_empty() {
        return Err(PhopError::NotConverged(
            "no gated-topology restart converged to a finite solution".to_string(),
        ));
    }
    Ok(ParetoFront::from_candidates(sols))
}

/// Map a discrete seed tree onto the complete-tree (heap) skeleton, producing warm-start logits:
/// a node that is `eml(l, r)` in the seed is marked *expand* and recursed into; a leaf is marked
/// *terminate* with its source's selection logit set high. Skeleton positions the seed does not
/// reach default to *terminate* with uniform leaves (so they stay pruned unless data expands them).
fn seed_to_init(seed: &EmlNode, depth: usize, n_vars: usize) -> GatedInit {
    let k = n_vars + 1;
    let internal_count = (1usize << depth) - 1;
    let total = (1usize << (depth + 1)) - 1;
    const HI: f64 = 6.0;
    let mut gi = GatedInit {
        leaf_logits: vec![vec![0.0; k]; total],
        consts: vec![1.0; total],
        gates: vec![-HI; internal_count], // default terminate
    };

    // Recurse over the seed, writing into `gi`. All sizes are derived from `gi` (gates.len() =
    // internal_count, leaf_logits.len() = total, leaf_logits[*].len() = k = n_vars + 1).
    fn go(node: &EmlNode, idx: usize, gi: &mut GatedInit) {
        let internal_count = gi.gates.len();
        let total = gi.leaf_logits.len();
        let k = gi.leaf_logits[0].len();
        let n_vars = k - 1;
        if idx >= total {
            return; // seed deeper than the skeleton: drop (the skeleton caps depth)
        }
        let mut leaf = |idx: usize, src: usize, c: f64| {
            gi.leaf_logits[idx][src] = HI;
            gi.consts[idx] = c;
        };
        match node {
            EmlNode::Eml { left, right } if idx < internal_count => {
                gi.gates[idx] = HI; // expand (root entry ignored — root always expands)
                go(left.as_ref(), 2 * idx + 1, gi);
                go(right.as_ref(), 2 * idx + 2, gi);
            }
            EmlNode::Var(i) => {
                if idx < internal_count {
                    gi.gates[idx] = -HI;
                }
                leaf(idx, (*i).min(n_vars - 1), 1.0);
            }
            EmlNode::Const(c) => {
                if idx < internal_count {
                    gi.gates[idx] = -HI;
                }
                leaf(idx, k - 1, *c);
            }
            EmlNode::One => {
                if idx < internal_count {
                    gi.gates[idx] = -HI;
                }
                leaf(idx, k - 1, 1.0);
            }
            // `Eml` landing on a leaf position (seed deeper than skeleton): terminate to a constant.
            EmlNode::Eml { .. } => leaf(idx, k - 1, 1.0),
        }
    }
    go(seed, 0, &mut gi);
    gi
}

/// Discover by **warm-started** differentiable tree-shape search: run the cheap `enumerate`
/// discoverer for a discrete seed, map it onto the gated skeleton, and refine with the gated
/// (depth-learning) optimizer. This is addition.md's key remedy — seeding the differentiable
/// search from a discrete solution instead of a uniform forest that explores garbage. The seed and
/// the refined trees are merged into one Pareto front, so warm-start never does worse than `enumerate`.
///
/// # Errors
/// Returns [`PhopError`] if the dataset is empty or discovery fails entirely.
pub fn discover_gated_warm(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    // 1. Discrete seed from structural enumeration (always include it in the front).
    let seed_front = crate::discoverer::Discoverer::new(cfg.clone()).fit(ds)?;
    let mut sols: Vec<Solution> = seed_front.solutions.clone();
    let seed_tree = seed_front
        .best()
        .map(|s| s.tree.clone())
        .ok_or_else(|| PhopError::NotConverged("enumerate produced no seed".to_string()))?;

    // 2. Skeleton deep enough for the seed (capped for memory), then warm-start the gated search.
    let depth = seed_tree.depth().max(cfg.max_depth).clamp(1, 4);
    let init = seed_to_init(seed_tree.root.as_ref(), depth, ds.n_vars());
    let restarts = cfg.population.clamp(1, MAX_RESTARTS);
    for r in 0..restarts {
        if let Ok(sol) = run_restart(
            ds,
            cfg,
            depth,
            cfg.seed.wrapping_add(r as u64 + 1),
            Some(&init),
        ) {
            if sol.mse.is_finite() {
                sols.push(sol);
            }
        }
    }
    if sols.is_empty() {
        return Err(PhopError::NotConverged(
            "warm-started gated search produced no finite solution".to_string(),
        ));
    }
    Ok(ParetoFront::from_candidates(sols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_recovers_exp_and_prunes_depth() {
        // y = exp(x0) = eml(x0, 1) needs only a shallow tree. Over a depth-3 maximal skeleton the
        // gates should terminate the deeper nodes (so the recovered tree stays small) while still
        // fitting well.
        let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys.clone())).unwrap();

        let mut cfg = Config::default()
            .max_depth(3)
            .population(3)
            .max_epochs(600)
            .learning_rate(0.1)
            .seed(7);
        cfg.lambda_complexity = 2e-3; // gentle parsimony; the curriculum init does the pruning
        let front = discover_gated(&ds, &cfg).unwrap();
        assert!(!front.is_empty());

        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ys.len() as f64;
        let best = front.best().unwrap();
        assert!(
            best.mse < var * 0.5,
            "gated best mse {} not below half-variance {} ({})",
            best.mse,
            var * 0.5,
            best.pretty()
        );
        // The maximal depth-3 tree has 15 nodes; a pruned recovery must be much smaller.
        assert!(
            best.complexity < 15,
            "expected pruning to a small tree, got complexity {} ({})",
            best.complexity,
            best.pretty()
        );
    }

    #[test]
    fn warm_start_recovers_nested_exp() {
        // y = exp(exp(x)) = eml(eml(x, 1), 1) is depth 2 — its left child must *expand*, which the
        // cold depth-curriculum (children start terminated) struggles to discover. Warm-starting
        // the gated search from the `enumerate` seed lands directly in the right basin.
        let xs: Vec<f64> = (0..30).map(|i| f64::from(i) / 29.0).collect(); // x in [0, 1]
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp().exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys.clone())).unwrap();

        let cfg = Config::default()
            .max_depth(2)
            .population(2)
            .max_epochs(400)
            .learning_rate(0.05)
            .seed(1);
        let front = discover_gated_warm(&ds, &cfg).unwrap();
        let best = front.best().unwrap();
        assert!(
            best.mse < 1e-4,
            "warm-started gated did not recover exp(exp(x)): mse {} ({})",
            best.mse,
            best.pretty()
        );
    }
}
