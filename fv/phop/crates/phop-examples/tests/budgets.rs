//! Wall-clock budget integration tests for the example laws.
//!
//! Each law's dataset is built via the shared `phop_examples` helpers (the same data the
//! example binaries use), then discovery runs under a SMALL config so the tests are fast and
//! robust on CI. We assert that:
//!   * `Discoverer::fit` returns `Ok`,
//!   * the resulting Pareto front is non-empty,
//!   * elapsed wall-clock time stays under a generous ceiling.
//!
//! The ceiling is intentionally loose (well above the design budgets) so transient CI slowness
//! never flakes the suite; the demos themselves carry the real (tighter) budgets in their docs.

use std::time::Instant;

use phop_core::{Config, DataSet, Discoverer};
use phop_examples::{
    black_scholes_dataset, exp_growth_dataset, kepler_dataset, michaelis_menten_dataset,
    planck_dataset,
};

/// Generous wall-clock ceiling shared by every law (seconds).
const CEILING_SECS: f64 = 60.0;

/// Build a small, fast config: shallow trees and few epochs keep each law well under budget.
fn small_config() -> Config {
    Config::default()
        .max_depth(3)
        .max_epochs(150)
        .population(64)
        .top_k(5)
        .seed(7)
}

/// Run discovery on `ds` and assert Ok + non-empty front + under the wall-clock ceiling.
fn assert_within_budget(name: &str, ds: &DataSet) {
    let cfg = small_config();
    let start = Instant::now();
    let result = Discoverer::new(cfg).fit(ds);
    let elapsed = start.elapsed();

    let front = result.unwrap_or_else(|e| panic!("[{name}] discovery failed: {e}"));
    assert!(
        !front.is_empty(),
        "[{name}] Pareto front was empty after fit"
    );
    assert!(
        elapsed.as_secs_f64() < CEILING_SECS,
        "[{name}] fit took {:.2}s, exceeding the {CEILING_SECS:.0}s ceiling",
        elapsed.as_secs_f64()
    );
}

#[test]
fn exp_growth_within_budget() {
    assert_within_budget("exp_growth", &exp_growth_dataset());
}

#[test]
fn exp_growth_recovers_exactly() {
    // y = exp(x) is exactly representable as eml(x, 1); the discoverer must recover it (mse ~ 0)
    // and render it as an exponential — the README's headline guarantee.
    let ds = exp_growth_dataset();
    let front = Discoverer::new(small_config()).fit(&ds).expect("fit");
    let best = front.best().expect("non-empty front");
    assert!(
        best.mse < 1e-6,
        "exp not recovered exactly: mse = {}",
        best.mse
    );
    let latex = best.latex();
    assert!(
        latex.contains("e^") || latex.contains("exp"),
        "expected an exponential form, got: {latex}"
    );
}

#[test]
fn kepler_within_budget() {
    assert_within_budget("kepler", &kepler_dataset());
}

#[test]
fn michaelis_menten_within_budget() {
    assert_within_budget("michaelis_menten", &michaelis_menten_dataset());
}

#[test]
fn planck_within_budget() {
    assert_within_budget("planck", &planck_dataset());
}

#[test]
fn black_scholes_within_budget() {
    assert_within_budget("black_scholes", &black_scholes_dataset());
}
