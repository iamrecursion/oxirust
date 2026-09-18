//! # phop-wasm — WebAssembly bindings for phop
//!
//! Exposes the [`phop-core`](phop_core) discovery engine to JavaScript via
//! `wasm-bindgen`. The single entry point [`discover_json`] takes JSON-encoded data
//! and configuration, runs discovery, and returns a JSON-encoded Pareto front, making
//! it trivial to call from a browser without any custom marshalling.
//!
//! ```ignore
//! import init, { discover_json, set_panic_hook } from "./pkg/phop_wasm.js";
//! await init();
//! set_panic_hook();
//! const data = JSON.stringify({ x: [[1],[2],[3]], y: [2,4,6] });
//! const cfg  = JSON.stringify({ population: 64, max_epochs: 50, seed: 0 });
//! const out  = JSON.parse(discover_json(data, cfg));
//! console.log(out.solutions);
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use phop_core::{AnySolution, Config, DataSet, Discoverer, Solution};
use scirs2_core::ndarray::{Array1, Array2};

/// Input dataset: a row-major matrix `x` and a target vector `y`.
#[derive(Debug, Deserialize)]
struct DataJson {
    /// Feature matrix, `rows × n_vars`.
    x: Vec<Vec<f64>>,
    /// Target vector, length `rows`.
    y: Vec<f64>,
}

/// Optional discovery hyperparameters. Missing fields fall back to defaults.
#[derive(Debug, Default, Deserialize)]
struct ConfigJson {
    /// Population size (number of candidate trees).
    population: Option<usize>,
    /// Maximum tree depth.
    max_depth: Option<usize>,
    /// Maximum optimization epochs.
    max_epochs: Option<usize>,
    /// Gradient-descent learning rate.
    learning_rate: Option<f64>,
    /// RNG seed.
    seed: Option<u64>,
    /// Number of top solutions to report.
    top_k: Option<usize>,
    /// When `true`, include a symbolic CAS analysis (derivative + antiderivative w.r.t. x0) of
    /// each returned solution — the in-browser "discover → analyze" path.
    analyze: Option<bool>,
}

/// One solution in the returned Pareto front.
#[derive(Debug, Serialize)]
struct SolutionJson {
    /// LaTeX rendering.
    latex: String,
    /// Human-readable infix rendering.
    pretty: String,
    /// Mean squared error.
    mse: f64,
    /// Structural complexity (node count).
    complexity: usize,
    /// `d/dx0` of the law (only when `analyze` was requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    derivative: Option<String>,
    /// An antiderivative `∫ · dx0` (only when `analyze` was requested and a closed form exists).
    #[serde(skip_serializing_if = "Option::is_none")]
    antiderivative: Option<String>,
}

/// The full result payload returned by [`discover_json`].
#[derive(Debug, Serialize)]
struct ResultJson {
    /// Top Pareto-optimal solutions, best first.
    solutions: Vec<SolutionJson>,
}

/// JSON error envelope returned when discovery fails.
#[derive(Debug, Serialize)]
struct ErrorJson {
    /// Human-readable error message.
    error: String,
}

/// Install [`console_error_panic_hook`] so Rust panics surface as readable
/// `console.error` messages in the browser.
#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Run symbolic discovery from JSON.
///
/// `data_json` must decode to `{ "x": [[..],..], "y": [..] }` and `config_json` to an
/// object with optional `population`, `max_depth`, `max_epochs`, `learning_rate`,
/// `seed`, and `top_k`. Returns a JSON string `{ "solutions": [ ... ] }` on success,
/// or `{ "error": "..." }` on failure.
#[wasm_bindgen]
pub fn discover_json(data_json: &str, config_json: &str) -> String {
    match run(data_json, config_json) {
        Ok(result) => serde_json::to_string(&result)
            .unwrap_or_else(|e| format!("{{\"error\":\"serialization failed: {e}\"}}")),
        Err(msg) => serde_json::to_string(&ErrorJson { error: msg })
            .unwrap_or_else(|_| "{\"error\":\"unknown error\"}".to_string()),
    }
}

/// Internal worker that performs the discovery and maps every failure to a `String`.
fn run(data_json: &str, config_json: &str) -> Result<ResultJson, String> {
    let data: DataJson = serde_json::from_str(data_json).map_err(|e| e.to_string())?;
    let cfg_json: ConfigJson = if config_json.trim().is_empty() {
        ConfigJson::default()
    } else {
        serde_json::from_str(config_json).map_err(|e| e.to_string())?
    };

    let rows = data.x.len();
    if rows == 0 {
        return Err("`x` must contain at least one row".to_string());
    }
    let cols = data.x[0].len();
    if data.x.iter().any(|r| r.len() != cols) {
        return Err("all rows of `x` must have the same length".to_string());
    }
    let flat: Vec<f64> = data.x.into_iter().flatten().collect();
    let x_arr = Array2::from_shape_vec((rows, cols), flat).map_err(|e| e.to_string())?;
    let y_arr = Array1::from(data.y);

    let ds = DataSet::from_arrays(x_arr, y_arr).map_err(|e| e.to_string())?;

    let mut config = Config::default();
    if let Some(v) = cfg_json.population {
        config = config.population(v);
    }
    if let Some(v) = cfg_json.max_depth {
        config = config.max_depth(v);
    }
    if let Some(v) = cfg_json.max_epochs {
        config = config.max_epochs(v);
    }
    if let Some(v) = cfg_json.learning_rate {
        config = config.learning_rate(v);
    }
    if let Some(v) = cfg_json.seed {
        config = config.seed(v);
    }
    let top_k = cfg_json.top_k.unwrap_or(config.top_k);
    if let Some(v) = cfg_json.top_k {
        config = config.top_k(v);
    }

    let front = Discoverer::new(config)
        .fit(&ds)
        .map_err(|e| e.to_string())?;

    let analyze = cfg_json.analyze.unwrap_or(false);
    let solutions = front
        .pareto_top(top_k)
        .into_iter()
        .map(|s| {
            let (derivative, antiderivative) = if analyze {
                let a = s.analyze(0, 4);
                (Some(a.derivative), a.antiderivative)
            } else {
                (None, None)
            };
            SolutionJson {
                latex: s.latex(),
                pretty: s.pretty(),
                mse: s.mse,
                complexity: s.complexity,
                derivative,
                antiderivative,
            }
        })
        .collect();

    Ok(ResultJson { solutions })
}

// ===================== Browser-complete verified pipeline =====================
//
// `discover_and_verify` runs the WHOLE pipeline client-side: discover a law, then (optionally)
// canonicalize it (e-graph), enclose its range and search for a certified root (interval
// Newton/Krawczyk), reduce dimensioned inputs to Buckingham-π groups, and **prove** properties —
// that the law has no root over the data box, or that it is exactly equivalent to a supplied target
// law — via the OxiZ SMT solver. Every tier is pure Rust compiled to wasm; no server is contacted.

/// Per-feature SI dimension: 7 integer exponents `[T, L, M, Θ, I, N, J]`.
type DimVec = Vec<i32>;

/// Extended configuration for [`discover_and_verify`].
#[derive(Debug, Default, Deserialize)]
struct VerifyConfigJson {
    /// Population / candidate budget.
    population: Option<usize>,
    /// Maximum tree depth (also bounds the rich-leaf eml-skeleton size).
    max_depth: Option<usize>,
    /// Maximum optimization epochs.
    max_epochs: Option<usize>,
    /// Gradient-descent learning rate.
    learning_rate: Option<f64>,
    /// RNG seed.
    seed: Option<u64>,
    /// Number of top solutions to report.
    top_k: Option<usize>,
    /// Search method: `"enumerate"` (default, fast/exact-shallow), `"auto"` (EML + rich-leaf
    /// ensemble — recovers products/powers), or `"rich"` (affine/log-linear leaves only).
    method: Option<String>,
    /// Add a CAS derivative + antiderivative w.r.t. x0 (always available).
    analyze: Option<bool>,
    /// Add a stronger e-graph canonical LaTeX (requires the `egraph` feature).
    canonical: Option<bool>,
    /// Add a certified interval range enclosure + certified root search (always available).
    certify: Option<bool>,
    /// Prove (SMT) that the best law has no root over the data box (requires the `smt` feature).
    prove_no_root: Option<bool>,
    /// Optional per-feature SI dimensions; when present, the inputs are reduced to dimensionless
    /// Buckingham-π groups before discovery and the groups are returned in [`VerifiedResultJson`].
    units: Option<Vec<DimVec>>,
    /// Optional serialized EML target law (from `Solution::to_model_json`); when present, the best
    /// EML law is **proved equivalent** to it over the data box via SMT (requires `smt`).
    target_model: Option<String>,
}

/// A discovered law plus every verification artifact that was requested and is supported.
#[derive(Debug, Serialize)]
struct VerifiedSolutionJson {
    /// Producing engine: `"eml"` (differentiable core) or `"affine"` (rich-leaf).
    source: String,
    /// LaTeX rendering.
    latex: String,
    /// Human-readable infix rendering.
    pretty: String,
    /// Mean squared error on the (possibly reduced) data.
    mse: f64,
    /// Coefficient of determination on the data.
    r2: f64,
    /// Structural complexity.
    complexity: usize,
    /// Whether this is a clean symbolic recovery (affine engine) / exact EML form.
    symbolic: bool,
    /// `d/dx0` of the law (when `analyze`).
    #[serde(skip_serializing_if = "Option::is_none")]
    derivative: Option<String>,
    /// An antiderivative `∫ · dx0` (when `analyze` and a closed form exists).
    #[serde(skip_serializing_if = "Option::is_none")]
    antiderivative: Option<String>,
    /// Stronger e-graph canonical LaTeX (when `canonical` and the `egraph` feature is built).
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_latex: Option<String>,
    /// Guaranteed range enclosure `[lo, hi]` of the law over the data box (when `certify`).
    #[serde(skip_serializing_if = "Option::is_none")]
    certified_range: Option<[f64; 2]>,
    /// Certified root search along x0 over the data box (when `certify`).
    #[serde(skip_serializing_if = "Option::is_none")]
    certified_root: Option<String>,
    /// SMT verdict: does the law have no root over the data box? (when `prove_no_root` + `smt`).
    #[serde(skip_serializing_if = "Option::is_none")]
    proven_no_root: Option<String>,
    /// SMT verdict: is the law exactly equivalent to the supplied target over the box? (when
    /// `target_model` + `smt`).
    #[serde(skip_serializing_if = "Option::is_none")]
    equivalent_to_target: Option<String>,
}

/// Which verification tiers this particular wasm build supports (which features were compiled in).
#[derive(Debug, Serialize)]
struct Capabilities {
    /// CAS analysis (always).
    analyze: bool,
    /// Interval certified range/root (always).
    certify: bool,
    /// E-graph canonicalization (`egraph` feature).
    canonical: bool,
    /// SMT proofs via OxiZ (`smt` feature).
    smt: bool,
}

/// The full payload from [`discover_and_verify`].
#[derive(Debug, Serialize)]
struct VerifiedResultJson {
    /// Top solutions, best first, each with its verification artifacts.
    solutions: Vec<VerifiedSolutionJson>,
    /// Buckingham-π group exponent vectors, when `units` were supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pi_groups: Option<Vec<DimVec>>,
    /// The verification tiers this build supports.
    capabilities: Capabilities,
}

/// Report which verification tiers this wasm build supports — lets the UI light up only what is
/// actually available (e.g. hide the SMT panel when built without `smt`).
#[wasm_bindgen]
pub fn capabilities() -> String {
    let caps = Capabilities {
        analyze: true,
        certify: true,
        canonical: cfg!(feature = "egraph"),
        smt: cfg!(feature = "smt"),
    };
    serde_json::to_string(&caps).unwrap_or_else(|_| "{}".to_string())
}

/// Run the full discover→verify pipeline from JSON, entirely in-browser.
///
/// `data_json` decodes to `{ "x": [[..],..], "y": [..] }`; `config_json` to a `VerifyConfigJson`.
/// Returns a JSON `VerifiedResultJson` on success, or `{ "error": "..." }` on failure.
#[wasm_bindgen]
pub fn discover_and_verify(data_json: &str, config_json: &str) -> String {
    match verify_run(data_json, config_json) {
        Ok(result) => serde_json::to_string(&result)
            .unwrap_or_else(|e| format!("{{\"error\":\"serialization failed: {e}\"}}")),
        Err(msg) => serde_json::to_string(&ErrorJson { error: msg })
            .unwrap_or_else(|_| "{\"error\":\"unknown error\"}".to_string()),
    }
}

/// Coefficient of determination of `pred` vs `y`.
fn r2_of(pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = y.len().max(1) as f64;
    let mean = y.sum() / n;
    let (mut sr, mut st) = (0.0, 0.0);
    for (p, t) in pred.iter().zip(y.iter()) {
        sr += (t - p) * (t - p);
        st += (t - mean) * (t - mean);
    }
    if st == 0.0 {
        f64::NAN
    } else {
        1.0 - sr / st
    }
}

/// Per-feature `(min, max)` bounding box of the dataset — the domain over which range/root/proof
/// verification is performed.
fn data_box(ds: &DataSet) -> Vec<(f64, f64)> {
    (0..ds.n_vars())
        .map(|j| {
            let col = ds.x.column(j);
            (
                col.iter().copied().fold(f64::INFINITY, f64::min),
                col.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect()
}

/// E-graph canonical LaTeX — `Some` only when the `egraph` feature is built in.
fn canonical_of(_s: &Solution) -> Option<String> {
    #[cfg(feature = "egraph")]
    {
        Some(_s.latex_egraph())
    }
    #[cfg(not(feature = "egraph"))]
    {
        None
    }
}

/// SMT verdict that `_s` has no root over `_bounds` — `Some` only when the `smt` feature is built in.
fn prove_no_root_of(_s: &Solution, _bounds: &[(f64, f64)]) -> Option<String> {
    #[cfg(feature = "smt")]
    {
        Some(format!("{:?}", _s.prove_no_root(_bounds)))
    }
    #[cfg(not(feature = "smt"))]
    {
        None
    }
}

/// SMT verdict that `_s` is equivalent to `_target` over `_bounds` — `Some` only with `smt`.
fn prove_equiv_of(_s: &Solution, _target: &Solution, _bounds: &[(f64, f64)]) -> Option<String> {
    #[cfg(feature = "smt")]
    {
        Some(format!("{:?}", _s.prove_equivalent(_target, _bounds)))
    }
    #[cfg(not(feature = "smt"))]
    {
        None
    }
}

/// Internal worker for [`discover_and_verify`].
fn verify_run(data_json: &str, config_json: &str) -> Result<VerifiedResultJson, String> {
    let data: DataJson = serde_json::from_str(data_json).map_err(|e| e.to_string())?;
    let cfg_json: VerifyConfigJson = if config_json.trim().is_empty() {
        VerifyConfigJson::default()
    } else {
        serde_json::from_str(config_json).map_err(|e| e.to_string())?
    };

    let rows = data.x.len();
    if rows == 0 {
        return Err("`x` must contain at least one row".to_string());
    }
    let cols = data.x[0].len();
    if data.x.iter().any(|r| r.len() != cols) {
        return Err("all rows of `x` must have the same length".to_string());
    }
    let flat: Vec<f64> = data.x.into_iter().flatten().collect();
    let x_arr = Array2::from_shape_vec((rows, cols), flat).map_err(|e| e.to_string())?;
    let y_arr = Array1::from(data.y);
    let mut ds = DataSet::from_arrays(x_arr, y_arr).map_err(|e| e.to_string())?;

    // Optional dimensional reduction to Buckingham-π groups.
    let mut pi_groups: Option<Vec<DimVec>> = None;
    if let Some(units) = &cfg_json.units {
        let dims: Vec<phop_core::Dimension> = units
            .iter()
            .map(|v| {
                <[i32; 7]>::try_from(v.clone())
                    .map_err(|_| "each `units` entry needs exactly 7 integer exponents".to_string())
            })
            .collect::<Result<_, _>>()?;
        let (reduced, groups) = ds.to_dimensionless(&dims).map_err(|e| e.to_string())?;
        ds = reduced;
        pi_groups = Some(groups);
    }

    let mut config = Config::default();
    if let Some(v) = cfg_json.population {
        config = config.population(v);
    }
    if let Some(v) = cfg_json.max_depth {
        config = config.max_depth(v);
    }
    if let Some(v) = cfg_json.max_epochs {
        config = config.max_epochs(v);
    }
    if let Some(v) = cfg_json.learning_rate {
        config = config.learning_rate(v);
    }
    if let Some(v) = cfg_json.seed {
        config = config.seed(v);
    }
    let top_k = cfg_json.top_k.unwrap_or(config.top_k);
    config = config.top_k(top_k);

    let max_internal = cfg_json.max_depth.unwrap_or(3).clamp(1, 5);
    const WASM_CAND_CAP: usize = 800;
    let front: Vec<AnySolution> = match cfg_json.method.as_deref() {
        Some("rich") => {
            phop_core::discover_affine_pareto(&ds.x, &ds.y, max_internal, WASM_CAND_CAP)
                .into_iter()
                .map(AnySolution::Affine)
                .collect()
        }
        Some("auto") => phop_core::discover_auto_all(&ds, &config, max_internal, WASM_CAND_CAP)
            .map_err(|e| e.to_string())?,
        _ => Discoverer::new(config)
            .fit(&ds)
            .map_err(|e| e.to_string())?
            .solutions
            .into_iter()
            .map(AnySolution::Eml)
            .collect(),
    };

    // Score by R² on the data, then DISPLAY-order for the demo: drop near-trivial baselines (the
    // R²≈0 constant that otherwise gets the full proof ceremony), and surface the most accurate, then
    // SIMPLEST, then lowest-MSE solution first (Occam's razor among equally-accurate fits — so a clean
    // depth-1 law ranks above a higher-complexity "bush" that happens to shave 1e-31 off the MSE).
    let mut scored: Vec<(AnySolution, f64)> = front
        .into_iter()
        .map(|s| {
            let r2 = s
                .predict(&ds.x)
                .ok()
                .map(|p| r2_of(&p, &ds.y))
                .unwrap_or(f64::NAN);
            (s, r2)
        })
        .collect();
    const MIN_DISPLAY_R2: f64 = 0.5;
    if scored
        .iter()
        .any(|(_, r2)| r2.is_finite() && *r2 >= MIN_DISPLAY_R2)
    {
        scored.retain(|(_, r2)| r2.is_finite() && *r2 >= MIN_DISPLAY_R2);
    }
    scored.sort_by(|a, b| {
        // Bucket R² so machine-precision ties (≈1.0) fall through to complexity, not raw MSE.
        let (ra, rb) = ((a.1 * 1e6).round(), (b.1 * 1e6).round());
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.complexity().cmp(&b.0.complexity()))
            .then(
                a.0.mse()
                    .partial_cmp(&b.0.mse())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    scored.truncate(top_k);

    let analyze = cfg_json.analyze.unwrap_or(false);
    let canonical = cfg_json.canonical.unwrap_or(false);
    let certify = cfg_json.certify.unwrap_or(false);
    let prove_no_root = cfg_json.prove_no_root.unwrap_or(false);
    let bounds = data_box(&ds);
    let target: Option<Solution> = match &cfg_json.target_model {
        Some(m) => Some(Solution::from_model_json(m).map_err(|e| e.to_string())?),
        None => None,
    };

    let solutions = scored
        .iter()
        .map(|(s, r2)| {
            let r2 = *r2;
            let mut out = VerifiedSolutionJson {
                source: s.source().to_string(),
                latex: s.latex(),
                pretty: s.expr(),
                mse: s.mse(),
                r2,
                complexity: s.complexity(),
                symbolic: s.is_symbolic(),
                derivative: None,
                antiderivative: None,
                canonical_latex: None,
                certified_range: None,
                certified_root: None,
                proven_no_root: None,
                equivalent_to_target: None,
            };
            // The CAS / certified / SMT tiers operate on EML-tree laws only.
            if let Some(e) = s.as_eml() {
                if analyze {
                    let a = e.analyze(0, 4);
                    out.derivative = Some(a.derivative);
                    out.antiderivative = a.antiderivative;
                }
                if canonical {
                    out.canonical_latex = canonical_of(e);
                }
                if certify {
                    let (lo, hi) = e.certified_range(&bounds);
                    out.certified_range = Some([lo, hi]);
                    if let Some(&(x0lo, x0hi)) = bounds.first() {
                        let others: Vec<f64> =
                            bounds.iter().skip(1).map(|(a, b)| 0.5 * (a + b)).collect();
                        out.certified_root = Some(match e.certified_root(0, &others, x0lo, x0hi) {
                            Ok(cert) => format!("{cert:?}"),
                            Err(err) => format!("error: {err}"),
                        });
                    }
                }
                if prove_no_root {
                    out.proven_no_root = prove_no_root_of(e, &bounds);
                }
                if let Some(t) = &target {
                    out.equivalent_to_target = prove_equiv_of(e, t, &bounds);
                }
            }
            out
        })
        .collect();

    Ok(VerifiedResultJson {
        solutions,
        pi_groups,
        capabilities: Capabilities {
            analyze: true,
            certify: true,
            canonical: cfg!(feature = "egraph"),
            smt: cfg!(feature = "smt"),
        },
    })
}

/// Native (host-runnable) unit tests.
///
/// `discover_json` is plain Rust with no JS interop on its hot path, so it can be
/// exercised directly on the host. These tests verify both the success and error
/// envelopes end to end.
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a JSON dataset for `y = exp(x)` sampled on a small grid.
    fn exp_dataset_json() -> String {
        let xs: Vec<f64> = (0..21).map(|i| i as f64 * 0.1).collect();
        let x: Vec<Vec<f64>> = xs.iter().map(|&v| vec![v]).collect();
        let y: Vec<f64> = xs.iter().map(|&v| v.exp()).collect();
        serde_json::to_string(&serde_json::json!({ "x": x, "y": y })).expect("dataset serializes")
    }

    #[test]
    fn discover_json_recovers_exp() {
        let data = exp_dataset_json();
        let cfg = serde_json::to_string(&serde_json::json!({
            "max_epochs": 100,
            "max_depth": 1,
            "seed": 0,
            "top_k": 3,
        }))
        .expect("config serializes");

        let out = discover_json(&data, &cfg);
        assert!(
            !out.contains("\"error\""),
            "discovery should not return an error envelope, got: {out}"
        );

        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("result must be valid JSON");
        let obj = parsed.as_object().expect("result must be a JSON object");
        let solutions = obj
            .get("solutions")
            .and_then(|v| v.as_array())
            .expect("result must contain a `solutions` array");
        assert!(
            !solutions.is_empty(),
            "the Pareto front must contain at least one solution"
        );

        let mse = solutions[0]
            .get("mse")
            .and_then(|v| v.as_f64())
            .expect("first solution must report a numeric `mse`");
        assert!(
            mse < 0.5,
            "best solution MSE should be small for y = exp(x), got {mse}"
        );
    }

    #[test]
    fn discover_json_includes_analysis_when_requested() {
        // The in-browser "discover → analyze" path: with `analyze: true`, each solution carries a
        // symbolic derivative computed via the oxieml CAS, entirely client-side.
        let data = exp_dataset_json();
        let cfg = serde_json::to_string(&serde_json::json!({
            "max_epochs": 100, "max_depth": 1, "seed": 0, "top_k": 1, "analyze": true,
        }))
        .expect("config serializes");

        let out = discover_json(&data, &cfg);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let first = parsed["solutions"][0]
            .as_object()
            .expect("a first solution object");
        assert!(
            first.contains_key("derivative"),
            "analyze=true must add a `derivative` field, got: {out}"
        );
    }

    #[test]
    fn verify_pipeline_runs_and_certifies() {
        // The full client-side pipeline: discover exp(x), attach a CAS derivative, and a CERTIFIED
        // interval range enclosure — all without the smt/egraph features.
        let data = exp_dataset_json();
        let cfg = serde_json::json!({
            "method": "enumerate", "max_epochs": 120, "max_depth": 1, "seed": 0, "top_k": 1,
            "analyze": true, "certify": true,
        })
        .to_string();
        let out = discover_and_verify(&data, &cfg);
        assert!(!out.contains("\"error\""), "pipeline errored: {out}");
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let s0 = &v["solutions"][0];
        assert!(
            s0["certified_range"].is_array(),
            "certify must add a certified_range, got: {out}"
        );
        assert!(
            s0["derivative"].is_string(),
            "analyze must add a derivative"
        );
        let r2 = s0["r2"].as_f64().expect("r2 present");
        assert!(r2 > 0.9, "exp fit should be accurate, r2 = {r2}");
        assert!(v["capabilities"]["certify"].as_bool().unwrap_or(false));
    }

    #[cfg(feature = "smt")]
    #[test]
    fn verify_pipeline_proves_no_root_with_smt() {
        // exp(x) > 0 everywhere → no root over the box. The OxiZ-backed SMT verdict must be present
        // and the build must advertise the smt capability.
        let data = exp_dataset_json();
        let cfg = serde_json::json!({
            "method": "enumerate", "max_epochs": 120, "max_depth": 1, "seed": 0, "top_k": 1,
            "prove_no_root": true,
        })
        .to_string();
        let out = discover_and_verify(&data, &cfg);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(
            v["solutions"][0]["proven_no_root"].is_string(),
            "smt build must add a proven_no_root verdict, got: {out}"
        );
        assert!(v["capabilities"]["smt"].as_bool().unwrap_or(false));
    }

    #[cfg(feature = "smt")]
    #[test]
    fn verify_proves_equivalence_after_integer_snap() {
        // Regression for the demo bug: at max_depth 3 the constant in eml(x0, c) fit to ~1.0000000438,
        // leaving an ln(c)≠0 residue that made ≡-true-law return a Counterexample. Integer snapping
        // must round it to exactly 1 so the law equals exp(x0) and SMT proves equivalence to the
        // true model. (Also exercises the R²-filter dropping the trivial constant baseline.)
        let data = exp_dataset_json();
        let cfg = serde_json::json!({
            "method": "enumerate", "max_depth": 3, "max_epochs": 200, "seed": 0, "top_k": 1,
            "target_model": "{\"root\":{\"Eml\":{\"left\":{\"Var\":0},\"right\":\"One\"}},\"num_vars\":1}",
        })
        .to_string();
        let out = discover_and_verify(&data, &cfg);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let s0 = &v["solutions"][0];
        assert_eq!(
            s0["equivalent_to_target"].as_str(),
            Some("Proven"),
            "≡ true law must be Proven after integer snapping, got: {out}"
        );
        assert_eq!(
            s0["symbolic"].as_bool(),
            Some(true),
            "best law should be symbolic"
        );
    }

    #[test]
    fn discover_json_reports_malformed_input() {
        let out = discover_json("{ this is not valid json", "{}");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("error envelope must itself be valid JSON");
        let obj = parsed
            .as_object()
            .expect("error envelope must be an object");
        assert!(
            obj.contains_key("error"),
            "malformed input must yield an `error` field, got: {out}"
        );
    }
}

/// WebAssembly smoke test.
///
/// This module is compiled only for `wasm32` targets and is exercised with
/// `wasm-pack test` / `wasm-bindgen-test`. It does not run on the host but
/// documents and enables the in-browser test path required by the TODO.
#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn discover_json_runs_in_wasm() {
        let data =
            r#"{ "x": [[0.0],[0.1],[0.2],[0.3],[0.4]], "y": [1.0, 1.105, 1.221, 1.350, 1.492] }"#;
        let cfg = r#"{ "max_epochs": 50, "max_depth": 1, "seed": 0, "top_k": 1 }"#;
        let out = discover_json(data, cfg);
        assert!(
            !out.is_empty(),
            "discover_json must return a non-empty string"
        );
    }
}
