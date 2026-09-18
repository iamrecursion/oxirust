//! Shared helpers for the phop demonstration binaries.
//!
//! Each example under `examples/` synthesizes data for a known physical law, runs the
//! [`phop_core::Discoverer`], and prints the resulting Pareto front. Exact recovery of the
//! deeper laws (Planck, Black-Scholes) awaits the M2 Gumbel-Softmax topology search and M4
//! GPU scaling; the demos run end-to-end today and show the engine discovering structure.
//!
//! The data-generating functions ([`exp_growth_dataset`], [`kepler_dataset`],
//! [`michaelis_menten_dataset`], [`planck_dataset`], [`black_scholes_dataset`]) are shared
//! between the example binaries and the budget integration tests, so the demos and tests
//! exercise byte-identical data.

use phop_core::{Config, DataSet, Discoverer};
use scirs2_core::ndarray::{Array1, Array2};

/// Build a single-feature dataset from `(x, y)` samples.
///
/// # Panics
/// Panics if `xs` and `ys` differ in length (callers construct them in lock-step).
#[must_use]
pub fn dataset_1d(xs: &[f64], ys: &[f64]) -> DataSet {
    let x = Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).expect("shape");
    let y = Array1::from(ys.to_vec());
    DataSet::from_arrays(x, y).expect("dataset")
}

/// Build a multi-feature dataset from row-major samples.
///
/// `xs` is a slice of rows; every row must have the same length (the number of features).
/// `ys[i]` is the target for `xs[i]`.
///
/// # Panics
/// Panics if `xs` is empty, if `xs` and `ys` differ in length, or if the rows are ragged.
#[must_use]
pub fn dataset_nd(xs: &[Vec<f64>], ys: &[f64]) -> DataSet {
    assert!(!xs.is_empty(), "dataset_nd requires at least one row");
    assert_eq!(xs.len(), ys.len(), "row count must match target count");
    let n_rows = xs.len();
    let n_cols = xs[0].len();
    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for row in xs {
        assert_eq!(row.len(), n_cols, "all rows must have equal width");
        flat.extend_from_slice(row);
    }
    let x = Array2::from_shape_vec((n_rows, n_cols), flat).expect("shape");
    let y = Array1::from(ys.to_vec());
    DataSet::from_arrays(x, y).expect("dataset")
}

// --------------------------------------------------------------------------------------------
// Data-generating functions (shared by examples and budget tests).
// --------------------------------------------------------------------------------------------

/// Exponential growth `y = exp(x)` over `x in [0, 4)` (50 samples).
///
/// phop recovers this exactly at depth 1, since `eml(x, 1) = exp(x) - ln(1) = exp(x)`.
#[must_use]
pub fn exp_growth_dataset() -> DataSet {
    let xs: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.08).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
    dataset_1d(&xs, &ys)
}

/// Kepler's third law `T = a^(3/2)` (i.e. `T^2 = a^3`), semi-major axis `a in [0.5, 20]`.
#[must_use]
pub fn kepler_dataset() -> DataSet {
    let xs: Vec<f64> = (1..=40).map(|i| f64::from(i) * 0.5).collect();
    let ys: Vec<f64> = xs.iter().map(|&a| a.powf(1.5)).collect();
    dataset_1d(&xs, &ys)
}

/// Michaelis-Menten kinetics `v = V_max * [S] / (K_m + [S])` with `V_max = 2`, `K_m = 0.5`.
#[must_use]
pub fn michaelis_menten_dataset() -> DataSet {
    let v_max = 2.0;
    let k_m = 0.5;
    let xs: Vec<f64> = (1..=40).map(|i| f64::from(i) * 0.25).collect();
    let ys: Vec<f64> = xs.iter().map(|&s| v_max * s / (k_m + s)).collect();
    dataset_1d(&xs, &ys)
}

/// Planck spectral radiance in normalized units at a fixed temperature.
///
/// We use the dimensionless Planck curve
/// `B(u) = u^5 / (exp(u) - 1)`
/// where `u = h c / (lambda k T)` is the inverse wavelength in units of the thermal scale.
/// Sampling `u in [0.5, 8]` keeps us clear of the `u -> 0` singularity, brackets the Wien
/// peak near `u ~ 5`, and keeps `B(u)` well-scaled (order 1-30). Treating `u` as the single
/// feature avoids the `lambda^-5` blow-up while preserving the law's shape.
#[must_use]
pub fn planck_dataset() -> DataSet {
    let us: Vec<f64> = (0..48).map(|i| 0.5 + f64::from(i) * (7.5 / 47.0)).collect();
    let ys: Vec<f64> = us.iter().map(|&u| u.powi(5) / (u.exp() - 1.0)).collect();
    dataset_1d(&us, &ys)
}

/// Synthetic Black-Scholes European call price as a function of two normalized features.
///
/// To keep the demo self-contained we fix the strike `K = 1`, risk-free rate `r = 0`, and
/// then vary the underlying spot `S in [0.6, 1.4]` and the total volatility
/// `w = sigma * sqrt(t) in [0.1, 0.5]`. With `r = 0` the call price is
/// `C = S * N(d1) - K * N(d2)`, where
/// `d1 = (ln(S/K) + w^2/2) / w`, `d2 = d1 - w`, and `N` is the standard normal CDF.
/// Feature column 0 is `S`, column 1 is `w`; the target is `C`. Prices stay in `[0, 0.5]`.
#[must_use]
pub fn black_scholes_dataset() -> DataSet {
    let strike = 1.0_f64;
    let spots: Vec<f64> = (0..9).map(|i| 0.6 + f64::from(i) * (0.8 / 8.0)).collect();
    let vols: Vec<f64> = (0..5).map(|i| 0.1 + f64::from(i) * (0.4 / 4.0)).collect();

    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(spots.len() * vols.len());
    let mut prices: Vec<f64> = Vec::with_capacity(spots.len() * vols.len());
    for &s in &spots {
        for &w in &vols {
            let d1 = ((s / strike).ln() + 0.5 * w * w) / w;
            let d2 = d1 - w;
            let call = s * norm_cdf(d1) - strike * norm_cdf(d2);
            rows.push(vec![s, w]);
            prices.push(call);
        }
    }
    dataset_nd(&rows, &prices)
}

/// Standard normal cumulative distribution function via the error function identity
/// `N(x) = 0.5 * (1 + erf(x / sqrt(2)))`.
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function, Abramowitz & Stegun 7.1.26 rational approximation (max abs error ~1.5e-7).
#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // Coefficients for the A&S 7.1.26 approximation.
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

// --------------------------------------------------------------------------------------------
// Reporting.
// --------------------------------------------------------------------------------------------

/// Run discovery on `ds` and print the top-`k` Pareto solutions.
pub fn report(title: &str, target: &str, ds: &DataSet, cfg: Config) {
    let k = cfg.top_k;
    println!("=== {title} ===");
    println!("target law: {target}");
    println!("data: {} rows, {} feature(s)\n", ds.len(), ds.n_vars());
    match Discoverer::new(cfg).fit(ds) {
        Ok(front) => {
            println!("Pareto front (top {k} by accuracy):");
            for (i, s) in front.pareto_top(k).iter().enumerate() {
                println!(
                    "  #{}  complexity={:<3}  mse={:.4e}  {}",
                    i + 1,
                    s.complexity,
                    s.mse,
                    s.pretty()
                );
                println!("       latex: {}", s.latex());
            }
        }
        Err(e) => println!("  discovery failed: {e}"),
    }
    println!();
}
