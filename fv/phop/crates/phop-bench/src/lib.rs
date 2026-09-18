//! Benchmark utilities for phop: synthetic data generators and a recovery-metric helper.
//!
//! The Feynman Symbolic Regression Benchmark loader (design §4 / M6) lands here once the
//! dataset is fetched and converted; for now this crate provides synthetic generators, a
//! symbolic-recovery check, and a small recovery benchmark harness (a Feynman-like suite of
//! elementary one-variable laws plus a runner) used by the criterion benches, the `recovery`
//! binary, and integration tests.

use phop_core::{eval_tree, Config, DataSet, Discoverer, Solution};
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt::Write as _;
use std::time::Instant;

/// Synthesize a single-feature dataset `y = f(x)` over `n` points in `[lo, hi]`.
pub fn synth_1d(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> DataSet {
    let step = if n > 1 {
        (hi - lo) / (n as f64 - 1.0)
    } else {
        0.0
    };
    let xs: Vec<f64> = (0..n).map(|i| lo + step * i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();
    let x = Array2::from_shape_vec((n, 1), xs).expect("shape");
    let y = Array1::from(ys);
    DataSet::from_arrays(x, y).expect("dataset")
}

/// Whether a solution counts as a "recovery": MSE below `tol`.
#[must_use]
pub fn is_recovery(sol: &Solution, tol: f64) -> bool {
    sol.mse < tol
}

/// A single benchmark case: a named elementary law and its sampled dataset.
pub struct BenchCase {
    /// Human-readable name of the case.
    pub name: &'static str,
    /// LaTeX rendering of the ground-truth law (for display only).
    pub target_latex: &'static str,
    /// The sampled dataset `y = f(x)`.
    pub dataset: DataSet,
}

/// The outcome of running a single [`BenchCase`] through the discoverer.
pub struct BenchResult {
    /// Name of the case (mirrors [`BenchCase::name`]).
    pub name: &'static str,
    /// Whether the best solution recovered the law (best MSE below the tolerance).
    pub recovered: bool,
    /// MSE of the best solution (`f64::INFINITY` if discovery failed/empty).
    pub best_mse: f64,
    /// Coefficient of determination (R^2) of the best solution on `case.dataset`.
    ///
    /// Computed as `1 - SS_res / SS_tot` where `SS_tot` uses the mean of `y`. It is
    /// `f64::NAN` when no solution was found, evaluation failed, or `SS_tot == 0`.
    pub r2: f64,
    /// Structural complexity of the best solution (`0` if none).
    pub best_complexity: usize,
    /// LaTeX rendering of the best discovered expression (empty if none).
    pub best_latex: String,
    /// Wall-clock time spent in `fit`, in milliseconds.
    pub millis: u128,
}

/// A Feynman-like suite of elementary one-variable laws phop can plausibly attempt at
/// shallow depth.
///
/// The suite mixes laws that should be easy to recover (e.g. `y = x`, `y = exp(x)`) with
/// harder ones (e.g. `y = x^{1.5}`, `y = 1/x`). Input ranges are chosen to avoid
/// singularities (e.g. `x in [0.5, 5]` for `ln`, `1/x`, `sqrt`).
#[must_use]
pub fn feynman_like_suite() -> Vec<BenchCase> {
    // Number of sample points per case (time-boxed: small but well-conditioned).
    const N: usize = 48;
    vec![
        BenchCase {
            name: "identity",
            target_latex: r"y = x",
            dataset: synth_1d(|x| x, 0.5, 5.0, N),
        },
        BenchCase {
            name: "exp",
            target_latex: r"y = e^{x}",
            dataset: synth_1d(f64::exp, 0.0, 2.0, N),
        },
        BenchCase {
            name: "exp_2x",
            target_latex: r"y = e^{2x}",
            dataset: synth_1d(|x| (2.0 * x).exp(), 0.0, 2.0, N),
        },
        BenchCase {
            name: "exp_minus_x",
            target_latex: r"y = e^{-x}",
            dataset: synth_1d(|x| (-x).exp(), 0.0, 3.0, N),
        },
        BenchCase {
            name: "exp_minus_ln",
            target_latex: r"y = e^{x} - \ln(x)",
            dataset: synth_1d(|x| x.exp() - x.ln(), 0.5, 3.0, N),
        },
        BenchCase {
            name: "ln",
            target_latex: r"y = \ln(x)",
            dataset: synth_1d(f64::ln, 0.5, 5.0, N),
        },
        BenchCase {
            name: "square",
            target_latex: r"y = x^{2}",
            dataset: synth_1d(|x| x * x, 0.5, 5.0, N),
        },
        BenchCase {
            name: "cube",
            target_latex: r"y = x^{3}",
            dataset: synth_1d(|x| x * x * x, 0.5, 3.0, N),
        },
        BenchCase {
            name: "reciprocal",
            target_latex: r"y = 1/x",
            dataset: synth_1d(|x| 1.0 / x, 0.5, 5.0, N),
        },
        BenchCase {
            name: "sqrt",
            target_latex: r"y = \sqrt{x}",
            dataset: synth_1d(f64::sqrt, 0.5, 5.0, N),
        },
        BenchCase {
            name: "pow_1_5",
            target_latex: r"y = x^{1.5}",
            dataset: synth_1d(|x| x.powf(1.5), 0.5, 5.0, N),
        },
        BenchCase {
            name: "linear",
            target_latex: r"y = 2x + 1",
            dataset: synth_1d(|x| 2.0 * x + 1.0, 0.5, 5.0, N),
        },
        BenchCase {
            name: "sin",
            target_latex: r"y = \sin(x)",
            dataset: synth_1d(f64::sin, 0.0, std::f64::consts::PI, N),
        },
        BenchCase {
            name: "exp_neg_square",
            target_latex: r"y = e^{-x^{2}}",
            dataset: synth_1d(|x| (-(x * x)).exp(), 0.0, 2.5, N),
        },
    ]
}

/// Coefficient of determination (R^2) of `pred` against the observed targets `y`.
///
/// Returns `1 - SS_res / SS_tot` using the mean of `y` for `SS_tot`. Returns `f64::NAN`
/// when the shapes disagree or `SS_tot == 0` (a constant target), which keeps callers safe
/// from divide-by-zero.
#[must_use]
pub fn r2_score(y: &Array1<f64>, pred: &Array1<f64>) -> f64 {
    if y.len() != pred.len() || y.is_empty() {
        return f64::NAN;
    }
    let mean = y.sum() / y.len() as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&yi, &pi) in y.iter().zip(pred.iter()) {
        let d = yi - pi;
        ss_res += d * d;
        let t = yi - mean;
        ss_tot += t * t;
    }
    if ss_tot == 0.0 {
        return f64::NAN;
    }
    1.0 - ss_res / ss_tot
}

/// Run a single benchmark case through the discoverer, timing the `fit` call.
///
/// On any `fit` error (or an empty Pareto front) the result is reported as not recovered
/// with an infinite MSE rather than panicking.
#[must_use]
pub fn run_case(case: &BenchCase, cfg: &Config, tol: f64) -> BenchResult {
    let discoverer = Discoverer::new(cfg.clone());
    let start = Instant::now();
    let outcome = discoverer.fit(&case.dataset);
    let millis = start.elapsed().as_millis();

    match outcome.ok().and_then(|front| front.best().cloned()) {
        Some(best) => {
            // R^2 on the (training) dataset. Evaluation may fail for pathological trees;
            // fall back to NAN rather than panicking.
            let r2 = match eval_tree(&best.tree, &case.dataset.x) {
                Ok(pred) => r2_score(&case.dataset.y, &pred),
                Err(_) => f64::NAN,
            };
            BenchResult {
                name: case.name,
                recovered: best.mse < tol,
                best_mse: best.mse,
                r2,
                best_complexity: best.complexity,
                best_latex: best.latex(),
                millis,
            }
        }
        None => BenchResult {
            name: case.name,
            recovered: false,
            best_mse: f64::INFINITY,
            r2: f64::NAN,
            best_complexity: 0,
            best_latex: String::new(),
            millis,
        },
    }
}

/// Run the full [`feynman_like_suite`] with the given config and recovery tolerance.
#[must_use]
pub fn run_suite(cfg: &Config, tol: f64) -> Vec<BenchResult> {
    feynman_like_suite()
        .iter()
        .map(|case| run_case(case, cfg, tol))
        .collect()
}

/// Quote a field for CSV output, escaping per RFC 4180 when it contains a delimiter,
/// quote, or newline.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Render results as CSV (RFC 4180-ish): a header line followed by one row per result.
///
/// Columns: `name,recovered,mse,r2,complexity,ms,latex`.
#[must_use]
pub fn results_to_csv(results: &[BenchResult]) -> String {
    let mut out = String::from("name,recovered,mse,r2,complexity,ms,latex\n");
    for r in results {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{}",
            csv_field(r.name),
            r.recovered,
            r.best_mse,
            r.r2,
            r.best_complexity,
            r.millis,
            csv_field(&r.best_latex),
        );
    }
    out
}

/// Escape a string for embedding inside a JSON string literal (hand-rolled, no deps).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Render an `f64` as a JSON number, mapping non-finite values to `null` (JSON has no
/// representation for `NaN`/`Infinity`).
fn json_number(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".to_string()
    }
}

/// Render results as a JSON array of objects (hand-rolled; no `serde` dependency).
///
/// Each object has keys `name`, `recovered`, `mse`, `r2`, `complexity`, `ms`, `latex`.
/// Non-finite `mse`/`r2` values are emitted as JSON `null`.
#[must_use]
pub fn results_to_json(results: &[BenchResult]) -> String {
    let mut out = String::from("[");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{{\"name\":\"{}\",\"recovered\":{},\"mse\":{},\"r2\":{},\"complexity\":{},\"ms\":{},\"latex\":\"{}\"}}",
            json_escape(r.name),
            r.recovered,
            json_number(r.best_mse),
            json_number(r.r2),
            r.best_complexity,
            r.millis,
            json_escape(&r.best_latex),
        );
    }
    out.push(']');
    out
}

/// Escape characters that are special in LaTeX text (underscores in particular, which are
/// common in case names).
fn latex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '_' | '%' | '&' | '#' | '$' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out
}

/// Format an `f64` for a LaTeX table cell, rendering non-finite values as a dash.
fn latex_number(v: f64, sci: bool) -> String {
    if !v.is_finite() {
        return "--".to_string();
    }
    if sci {
        format!("{v:.3e}")
    } else {
        format!("{v:.4}")
    }
}

/// Render results as a LaTeX `tabular` environment for the paper's experiments section.
///
/// Columns: name, recovered, mse, r2, complexity, ms. Underscores (and other LaTeX
/// specials) in the case name are escaped.
#[must_use]
pub fn results_to_latex_table(results: &[BenchResult]) -> String {
    let mut out = String::new();
    out.push_str("\\begin{tabular}{lrrrrr}\n");
    out.push_str("\\hline\n");
    out.push_str("name & recovered & mse & r2 & complexity & ms \\\\\n");
    out.push_str("\\hline\n");
    for r in results {
        let _ = writeln!(
            out,
            "{} & {} & {} & {} & {} & {} \\\\",
            latex_escape(r.name),
            if r.recovered { "yes" } else { "no" },
            latex_number(r.best_mse, true),
            latex_number(r.r2, false),
            r.best_complexity,
            r.millis,
        );
    }
    out.push_str("\\hline\n");
    out.push_str("\\end{tabular}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_has_expected_shape() {
        let ds = synth_1d(|x| x * x, 0.0, 1.0, 11);
        assert_eq!(ds.len(), 11);
        assert_eq!(ds.n_vars(), 1);
    }

    #[test]
    fn recovers_trivial_cases() {
        // Small/fast config: keep the test well under a minute.
        let cfg = Config::default()
            .max_depth(2)
            .max_epochs(150)
            .top_k(3)
            .seed(7);
        let tol = 1e-3;
        let results = run_suite(&cfg, tol);

        // The trivially-recoverable cases must come out with a tiny best MSE.
        for name in ["identity", "exp"] {
            let r = results
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("missing result for {name}"));
            assert!(
                r.best_mse < tol,
                "expected {name} to be recovered, got best_mse={} latex={}",
                r.best_mse,
                r.best_latex
            );
        }
    }

    /// Build a small synthetic `BenchResult` set without running discovery, for emitter tests.
    fn sample_results() -> Vec<BenchResult> {
        vec![
            BenchResult {
                name: "exp",
                recovered: true,
                best_mse: 1.2e-9,
                r2: 0.999_99,
                best_complexity: 3,
                best_latex: r"e^{x}".to_string(),
                millis: 12,
            },
            BenchResult {
                name: "exp_minus_ln",
                recovered: false,
                best_mse: f64::INFINITY,
                r2: f64::NAN,
                best_complexity: 0,
                best_latex: String::new(),
                millis: 7,
            },
        ]
    }

    #[test]
    fn r2_score_perfect_and_constant() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        // Perfect prediction -> R^2 == 1.
        assert!((r2_score(&y, &y) - 1.0).abs() < 1e-12);
        // Constant target -> SS_tot == 0 -> NAN.
        let c = Array1::from(vec![5.0, 5.0, 5.0]);
        assert!(r2_score(&c, &c).is_nan());
        // Mismatched lengths -> NAN.
        let p = Array1::from(vec![1.0, 2.0]);
        assert!(r2_score(&y, &p).is_nan());
    }

    #[test]
    fn r2_is_near_one_for_recovered_exp() {
        let cfg = Config::default()
            .max_depth(2)
            .max_epochs(150)
            .top_k(3)
            .seed(7);
        let case = feynman_like_suite()
            .into_iter()
            .find(|c| c.name == "exp")
            .expect("exp case present");
        let r = run_case(&case, &cfg, 1e-3);
        assert!(r.recovered, "exp should recover; best_mse={}", r.best_mse);
        assert!(
            r.r2 > 0.999,
            "expected R^2 ~ 1.0 for recovered exp, got {}",
            r.r2
        );
    }

    #[test]
    fn csv_has_header_and_one_row_per_result() {
        let results = sample_results();
        let csv = results_to_csv(&results);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "name,recovered,mse,r2,complexity,ms,latex");
        // Header + one row per result.
        assert_eq!(lines.len(), 1 + results.len());
        assert!(lines[1].starts_with("exp,true,"));
        assert!(lines[2].starts_with("exp_minus_ln,false,"));
    }

    #[test]
    fn csv_quotes_fields_with_commas() {
        let results = vec![BenchResult {
            name: "weird",
            recovered: true,
            best_mse: 0.0,
            r2: 1.0,
            best_complexity: 1,
            best_latex: r"a, b".to_string(),
            millis: 1,
        }];
        let csv = results_to_csv(&results);
        assert!(
            csv.contains("\"a, b\""),
            "comma field must be quoted: {csv}"
        );
    }

    #[test]
    fn json_has_expected_shape() {
        let results = sample_results();
        let json = results_to_json(&results);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"name\":\"exp\""));
        assert!(json.contains("\"recovered\":true"));
        assert!(json.contains("\"complexity\":3"));
        // Non-finite numbers map to JSON null.
        assert!(json.contains("\"mse\":null"));
        assert!(json.contains("\"r2\":null"));
    }

    #[test]
    fn latex_table_well_formed_and_escapes_underscores() {
        let results = sample_results();
        let tex = results_to_latex_table(&results);
        assert!(tex.contains("\\begin{tabular}"));
        assert!(tex.contains("\\end{tabular}"));
        // Underscore in the case name must be escaped.
        assert!(tex.contains(r"exp\_minus\_ln"));
        // Non-finite mse/r2 render as a dash.
        assert!(tex.contains("--"));
    }
}
