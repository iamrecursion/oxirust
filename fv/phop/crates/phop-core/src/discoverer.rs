//! The top-level discovery engine.
//!
//! [`Discoverer::fit`] searches over EML tree topologies and, for each, fits real-valued
//! constants by gradient descent (Layer A forward + Adam), scores candidates on
//! (accuracy, complexity), and returns the Pareto front.
//!
//! Topology proposal currently uses `oxieml`'s structural enumeration (a strong, exact
//! baseline at shallow depth). The differentiable Gumbel-Softmax topology relaxation
//! (Layer B / M2) plugs in here as an alternative proposer for deeper search.

use crate::affine::discover_affine_pareto;
use crate::any_solution::{merge_pareto, AnySolution};
use crate::config::Config;
use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use crate::fit::{fit_constants, n_constants};
use crate::forest::eval_tree;
use crate::pareto::ParetoFront;
use crate::solution::Solution;
use oxieml::symreg::{dedupe_by_semantics, enumerate_topologies};
use oxieml::{EmlNode, EmlTree};
use std::sync::Arc;

/// Crate-local fitting interface implemented by every GPU engine, so the discoverer can dispatch
/// over `Box<dyn GpuFitEngine>` regardless of which backend feature(s) are compiled in. Each impl
/// is a one-line forward to the engine's inherent `fit_constants`, so enabling exactly one backend
/// is behaviorally identical to calling that engine directly.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
pub(crate) trait GpuFitEngine {
    /// Coarse single-precision constant fit on the GPU (the CPU LM polish sharpens it to f64).
    fn gpu_fit_constants(
        &self,
        template: &EmlTree,
        ds: &DataSet,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)>;
}

#[cfg(feature = "gpu-cuda")]
impl GpuFitEngine for crate::gpu::CudaEmlEngine {
    fn gpu_fit_constants(
        &self,
        template: &EmlTree,
        ds: &DataSet,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)> {
        self.fit_constants(template, ds, learning_rate, max_epochs)
    }
}

#[cfg(feature = "gpu-metal")]
impl GpuFitEngine for crate::metal::MetalEmlEngine {
    fn gpu_fit_constants(
        &self,
        template: &EmlTree,
        ds: &DataSet,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)> {
        self.fit_constants(template, ds, learning_rate, max_epochs)
    }
}

/// Maximum depth used for exact structural enumeration (deeper search is the M2 path).
const MAX_ENUM_DEPTH: usize = 3;
/// Hard cap on enumerated topologies considered, to bound wall-clock.
const MAX_CANDIDATES: usize = 512;
/// Maximum number of free constants a candidate may have to be worth fitting.
const MAX_FIT_CONSTS: usize = 6;
/// Number of most-promising topologies whose constants are actually fitted by Adam.
///
/// Constant fitting is the expensive step (one autograd loop per candidate). We rank
/// topologies by their *raw* (unfitted) MSE and fit only the best few, which is both far
/// cheaper and well-targeted. Joint fitting of the whole population in one graph is the
/// M2/M4 (tensorized) upgrade.
const FIT_BUDGET: usize = 24;

/// Discovery engine configured by a [`Config`].
#[derive(Debug, Clone)]
pub struct Discoverer {
    cfg: Config,
}

/// Replace every `One` leaf with a free `Const(1.0)`, turning literal ones into
/// fittable parameters while preserving structure and variable placement.
fn constantize(node: &EmlNode) -> Arc<EmlNode> {
    match node {
        EmlNode::One => Arc::new(EmlNode::Const(1.0)),
        EmlNode::Const(c) => Arc::new(EmlNode::Const(*c)),
        EmlNode::Var(i) => Arc::new(EmlNode::Var(*i)),
        EmlNode::Eml { left, right } => Arc::new(EmlNode::Eml {
            left: constantize(left),
            right: constantize(right),
        }),
    }
}

/// Whether a solution's rendered LaTeX is a finite, valid closed form.
fn latex_is_finite(sol: &Solution) -> bool {
    let tex = sol.latex();
    let lower = tex.to_ascii_lowercase();
    !lower.contains("nan") && !lower.contains("inf")
}

/// Cheaply score one raw topology: forward-evaluate it and pair its MSE with a [`Solution`], or
/// `None` if the forward pass fails or is non-finite.
fn eval_one(topo: &EmlTree, ds: &DataSet) -> Option<(f64, Solution)> {
    let pred = eval_tree(topo, &ds.x).ok()?;
    let m = crate::fit::mse(&pred, &ds.y);
    m.is_finite().then(|| (m, Solution::new(topo.clone(), m)))
}

/// Score every topology. With the `parallel` feature this fans the independent evaluations across
/// cores via `scirs2-core`'s order-preserving parallel map; otherwise it is a plain sequential map.
/// Either way the output order matches `topos`, keeping discovery deterministic.
#[cfg(feature = "parallel")]
fn raw_evals(topos: &[EmlTree], ds: &DataSet) -> Vec<Option<(f64, Solution)>> {
    scirs2_core::parallel_ops::parallel_map(topos, |topo| eval_one(topo, ds))
}

/// Sequential fallback used when the `parallel` feature is disabled.
#[cfg(not(feature = "parallel"))]
fn raw_evals(topos: &[EmlTree], ds: &DataSet) -> Vec<Option<(f64, Solution)>> {
    topos.iter().map(|topo| eval_one(topo, ds)).collect()
}

impl Discoverer {
    /// Create a discoverer with the given configuration.
    #[must_use]
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    /// Run discovery on a dataset, returning the Pareto front of candidate expressions.
    ///
    /// # Errors
    /// Returns [`PhopError::NotConverged`] if no finite-scoring candidate is found.
    pub fn fit(&self, ds: &DataSet) -> Result<ParetoFront> {
        if ds.is_empty() {
            return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
        }
        let n_vars = ds.n_vars();
        let depth = self.cfg.max_depth.clamp(1, MAX_ENUM_DEPTH);

        let mut topologies = dedupe_by_semantics(enumerate_topologies(depth, n_vars));
        topologies.truncate(MAX_CANDIDATES);

        let mut candidates: Vec<Solution> = Vec::new();

        // 1. Cheaply evaluate every raw topology and record its MSE for ranking. This sweep is
        //    embarrassingly parallel (each topology is scored independently); with the `parallel`
        //    feature it fans out across cores while preserving input order, so the ranking — and
        //    therefore the whole discovery — stays deterministic.
        let mut ranked: Vec<(f64, usize)> = Vec::new();
        for (i, eval) in raw_evals(&topologies, ds).into_iter().enumerate() {
            if let Some((m, sol)) = eval {
                candidates.push(sol);
                ranked.push((m, i));
            }
        }

        // Optional GPU backend for the (expensive) constant-fitting step. The GPU produces a coarse
        // single-precision fit which the CPU Levenberg–Marquardt polish below sharpens to f64, so
        // the final front stays exact regardless of backend. Falls back to CPU silently.
        #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
        let gpu_engine: Option<Box<dyn GpuFitEngine>> = self.select_gpu_engine();

        // 2. Fit constants only on the most promising topologies (bounded budget).
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut fits_done = 0;
        for (_, i) in ranked {
            if fits_done >= FIT_BUDGET {
                break;
            }
            let ct = EmlTree::from_node(constantize(&topologies[i].root));
            let nc = n_constants(&ct);
            if (1..=MAX_FIT_CONSTS).contains(&nc) {
                let fitted_res = {
                    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
                    {
                        self.fit_one(&ct, ds, gpu_engine.as_deref())
                    }
                    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
                    {
                        fit_constants(&ct, ds, &self.cfg)
                    }
                };
                if let Ok((fitted, _)) = fitted_res {
                    // Sharpen the coarse Adam constants with Levenberg–Marquardt, then snap to
                    // named constants (π, e, …) when that does not worsen the fit.
                    let (polished, _) = crate::polish::polish_constants(&fitted, ds, 40);
                    let (snapped, m) = crate::polish::snap_constants(&polished, ds, 0.02);
                    let sol = Solution::new(snapped, m);
                    // Reject fits whose *symbolic* form is degenerate (e.g. a negative
                    // constant in a `ln` position): the guarded numeric eval stays finite,
                    // but the closed form is not a valid elementary expression.
                    if m.is_finite() && latex_is_finite(&sol) {
                        candidates.push(sol);
                    }
                }
                fits_done += 1;
            }
        }

        if candidates.is_empty() {
            return Err(PhopError::NotConverged(
                "no finite-scoring candidate expression found".to_string(),
            ));
        }
        Ok(ParetoFront::from_candidates(candidates))
    }

    /// Choose the GPU fitting engine for the configured backend, or `None` to fit on the CPU.
    /// Only the backend whose feature is compiled in can be selected; a missing device falls back
    /// to `None` (CPU). With `--all-features` both arms are present and `cfg.backend` disambiguates.
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    fn select_gpu_engine(&self) -> Option<Box<dyn GpuFitEngine>> {
        use crate::config::Backend;
        match self.cfg.backend {
            #[cfg(feature = "gpu-cuda")]
            Backend::Cuda if crate::gpu::cuda_available() => crate::gpu::CudaEmlEngine::new()
                .ok()
                .map(|e| Box::new(e) as Box<dyn GpuFitEngine>),
            #[cfg(feature = "gpu-metal")]
            Backend::Metal if crate::metal::metal_available() => {
                crate::metal::MetalEmlEngine::new()
                    .ok()
                    .map(|e| Box::new(e) as Box<dyn GpuFitEngine>)
            }
            _ => None,
        }
    }

    /// Fit the constant leaves of `ct` on the GPU when an engine is supplied, else on the CPU.
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
    fn fit_one(
        &self,
        ct: &EmlTree,
        ds: &DataSet,
        gpu: Option<&dyn GpuFitEngine>,
    ) -> Result<(EmlTree, f64)> {
        match gpu {
            Some(engine) => {
                engine.gpu_fit_constants(ct, ds, self.cfg.learning_rate, self.cfg.max_epochs)
            }
            None => fit_constants(ct, ds, &self.cfg),
        }
    }
}

/// Robust "just works" discovery: a **meta-ensemble** that runs algorithmically diverse searches
/// and returns the **merged** Pareto front — the best across methods and depths without the caller
/// picking one. It combines phop's own searches (structural enumeration, Gumbel-Softmax topology,
/// gated depth learning) with the cool-japan `oxieml` symbolic-regression engine (its GA/beam/MCTS
/// strategies). All `oxieml` candidates are re-scored with phop's evaluator so the front is
/// consistent. This is the shape-mixture idea in spirit, and the diversity is something no single
/// monolithic SR algorithm provides.
///
/// # Errors
/// Returns [`PhopError::NotConverged`] only if *every* method fails to produce a finite solution.
pub fn discover_auto(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let mut cands: Vec<Solution> = Vec::new();
    if let Ok(f) = Discoverer::new(cfg.clone()).fit(ds) {
        cands.extend(f.solutions);
    }
    if let Ok(f) = crate::gumbel::discover_gumbel(ds, cfg) {
        cands.extend(f.solutions);
    }
    if let Ok(f) = crate::gated::discover_gated(ds, cfg) {
        cands.extend(f.solutions);
    }
    cands.extend(oxieml_symreg_candidates(ds));
    if cands.is_empty() {
        return Err(PhopError::NotConverged(
            "no discovery method produced a finite solution".to_string(),
        ));
    }
    Ok(ParetoFront::from_candidates(cands))
}

/// The **full** meta-ensemble: [`discover_auto`]'s EML-tree front merged with the **rich-leaf affine
/// engine** ([`crate::discover_affine_pareto`]) into one Pareto front over both representations.
///
/// This is the discovery the CLI `auto` method runs, so the default search *reaches the affine
/// engine* — the one that recovers product / power / ratio laws the shallow EML core cannot assemble.
/// Without it, `auto` silently excludes phop's strongest recovery path. `max_internal` bounds the
/// affine eml-skeleton size and `cand_cap` its candidate pool. Returns solutions sorted by MSE
/// ascending (most accurate first); see [`AnySolution`].
///
/// # Errors
/// Returns [`PhopError::ShapeMismatch`] only for an empty dataset. Individual searches that fail are
/// omitted; if *every* contributor is empty the returned vector is empty (not an error).
pub fn discover_auto_all(
    ds: &DataSet,
    cfg: &Config,
    max_internal: usize,
    cand_cap: usize,
) -> Result<Vec<AnySolution>> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let mut cands: Vec<AnySolution> = Vec::new();
    // EML-tree core (enumerate + Gumbel + gated + oxieml-symreg). It only errors when every method
    // fails — in that case proceed with the affine engine alone.
    if let Ok(front) = discover_auto(ds, cfg) {
        cands.extend(front.solutions.into_iter().map(AnySolution::Eml));
    }
    // Rich-leaf affine engine (the product/power/ratio recovery path).
    cands.extend(
        discover_affine_pareto(&ds.x, &ds.y, max_internal, cand_cap)
            .into_iter()
            .map(AnySolution::Affine),
    );
    Ok(merge_pareto(cands))
}

/// Run the `oxieml` symbolic-regression engine and return its Pareto formulas as phop solutions,
/// re-scored with phop's evaluator for a consistent front. Failures yield an empty vector (the
/// meta-ensemble simply proceeds without this contributor).
fn oxieml_symreg_candidates(ds: &DataSet) -> Vec<Solution> {
    use oxieml::symreg::{SymRegConfig, SymRegEngine};
    let inputs: Vec<Vec<f64>> = (0..ds.len())
        .map(|i| (0..ds.n_vars()).map(|j| ds.x[[i, j]]).collect())
        .collect();
    let targets: Vec<f64> = ds.y.to_vec();
    // Dependency searches can emit debug output; suppress it during the call.
    let _silencer = crate::silence::SilenceStdout::new();
    let engine = SymRegEngine::new(SymRegConfig::quick());
    let result = engine.discover_pareto(&inputs, &targets, ds.n_vars());
    drop(_silencer);
    match result {
        Ok(formulas) => formulas
            .into_iter()
            .filter_map(|f| {
                let pred = eval_tree(&f.eml_tree, &ds.x).ok()?;
                let m = crate::fit::mse(&pred, &ds.y);
                m.is_finite().then(|| Solution::new(f.eml_tree, m))
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn discovers_exp() {
        // Target: y = exp(x0), recoverable exactly as eml(x0, 1) at depth 1.
        let xs: Vec<f64> = (0..30).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let y = Array1::from(ys);
        let ds = DataSet::from_arrays(x, y).unwrap();

        let cfg = Config::default().max_depth(1).max_epochs(300);
        let front = Discoverer::new(cfg).fit(&ds).unwrap();
        assert!(!front.is_empty());
        let best = front.best().unwrap();
        assert!(
            best.mse < 1e-6,
            "best mse = {} ({})",
            best.mse,
            best.pretty()
        );
    }

    #[test]
    fn enumerate_discovery_is_deterministic() {
        // A fixed seed must produce an identical Pareto front (same size, same MSEs/complexities).
        let xs: Vec<f64> = (1..=30).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp() - 2.0_f64.ln()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys)).unwrap();

        let cfg = Config::default().max_depth(2).max_epochs(200).seed(5);
        let a = Discoverer::new(cfg.clone()).fit(&ds).unwrap();
        let b = Discoverer::new(cfg).fit(&ds).unwrap();
        assert_eq!(a.len(), b.len(), "front sizes differ across identical runs");
        for (sa, sb) in a.solutions.iter().zip(&b.solutions) {
            assert_eq!(sa.complexity, sb.complexity);
            assert!(
                (sa.mse - sb.mse).abs() < 1e-9,
                "non-deterministic MSE: {} vs {}",
                sa.mse,
                sb.mse
            );
        }
    }

    #[test]
    fn discover_auto_merges_methods_and_recovers_exp() {
        // y = exp(x): the merged front (enumerate + gumbel + gated) must contain the exact law.
        let xs: Vec<f64> = (0..24).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys)).unwrap();

        // Tiny config so the three methods together stay fast.
        let cfg = Config::default()
            .max_depth(2)
            .population(2)
            .max_epochs(120)
            .seed(0);
        let front = discover_auto(&ds, &cfg).unwrap();
        assert!(!front.is_empty());
        assert!(
            front.best().unwrap().mse < 1e-6,
            "discover_auto did not recover exp: best mse = {}",
            front.best().unwrap().mse
        );
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn discovers_exp_on_cuda_backend() {
        use crate::config::Backend;
        if !crate::gpu::cuda_available() {
            eprintln!("skipping CUDA-backend test: no device");
            return;
        }
        // y = exp(x0): GPU does the coarse constant fit, CPU LM polish sharpens to f64.
        let xs: Vec<f64> = (0..30).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys)).unwrap();

        let cfg = Config::default()
            .max_depth(1)
            .max_epochs(300)
            .backend(Backend::Cuda);
        let front = Discoverer::new(cfg).fit(&ds).unwrap();
        let best = front.best().unwrap();
        assert!(
            best.mse < 1e-6,
            "CUDA-backend best mse = {} ({})",
            best.mse,
            best.pretty()
        );
    }

    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    #[test]
    fn discovers_exp_on_metal_backend() {
        use crate::config::Backend;
        if !crate::metal::metal_available() {
            eprintln!("skipping Metal-backend test: no device");
            return;
        }
        // y = exp(x0): Metal does the coarse constant fit, CPU LM polish sharpens to f64.
        let xs: Vec<f64> = (0..30).map(|i| f64::from(i) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let y = Array1::from(ys);
        let ds = DataSet::from_arrays(x, y).unwrap();

        let cfg = Config::default()
            .max_depth(1)
            .max_epochs(300)
            .backend(Backend::Metal);
        let front = Discoverer::new(cfg).fit(&ds).unwrap();
        let best = front.best().unwrap();
        assert!(
            best.mse < 1e-6,
            "Metal-backend best mse = {} ({})",
            best.mse,
            best.pretty()
        );
    }
}
