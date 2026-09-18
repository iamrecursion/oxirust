//! Integration tests for beam-search topology exploration.

use oxieml::symreg::{SymRegConfig, SymRegEngine, SymRegStrategy};

/// The default strategy is `Exhaustive`.
#[test]
fn beam_default_strategy_is_exhaustive() {
    let config = SymRegConfig::quick();
    assert_eq!(config.strategy, SymRegStrategy::Exhaustive);
}

/// Beam with width=5 on y=x data returns at least one formula with mse < 0.1.
#[test]
fn beam_returns_formula() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 300,
        num_restarts: 1,
        strategy: SymRegStrategy::Beam { width: 5 },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("beam discover should succeed");

    assert!(
        !formulas.is_empty(),
        "beam should return at least one formula"
    );
    assert!(
        formulas[0].mse < 0.1,
        "best formula mse should be < 0.1 for y=x, got {}",
        formulas[0].mse
    );
}

/// Beam with width=1000 on simple data returns at least one formula.
#[test]
fn beam_at_width_max_returns_results() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 200,
        num_restarts: 1,
        strategy: SymRegStrategy::Beam { width: 1000 },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("wide beam should succeed");

    assert!(
        !formulas.is_empty(),
        "wide beam should return at least one formula"
    );
}

/// Exhaustive strategy still produces results on simple data.
#[test]
fn exhaustive_strategy_still_works() {
    // Use y=x (identity) — trivially learned by depth-1 topology Var(0).
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 1000,
        num_restarts: 2,
        strategy: SymRegStrategy::Exhaustive,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("exhaustive discover should succeed");

    assert!(
        !formulas.is_empty(),
        "exhaustive should return at least one formula"
    );
    assert!(
        formulas[0].mse < 0.01,
        "best formula mse should be very low for y=x, got {}",
        formulas[0].mse
    );
}

fn exp_dataset(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();
    (inputs, targets)
}

/// `discover_beam` returns results without panicking for a trivial dataset.
#[test]
fn beam_search_returns_results() {
    let (inputs, targets) = exp_dataset(15);

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 200,
        num_restarts: 1,
        strategy: SymRegStrategy::Beam { width: 5 },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("beam discover should succeed");

    assert!(
        !formulas.is_empty(),
        "beam should return at least one formula"
    );
}

/// `discover_beam` with width ≥ total topologies gives same results as exhaustive.
#[test]
fn beam_wide_equals_exhaustive() {
    let (inputs, targets) = exp_dataset(15);

    let config_exhaustive = SymRegConfig {
        max_depth: 1,
        max_iter: 300,
        num_restarts: 2,
        seed: Some(42),
        strategy: SymRegStrategy::Exhaustive,
        ..SymRegConfig::default()
    };

    let config_beam = SymRegConfig {
        max_depth: 1,
        max_iter: 300,
        num_restarts: 2,
        seed: Some(42),
        // Width much larger than the depth-1 topology count: effectively exhaustive.
        strategy: SymRegStrategy::Beam { width: 10_000 },
        ..SymRegConfig::default()
    };

    let engine_ex = SymRegEngine::new(config_exhaustive);
    let engine_bm = SymRegEngine::new(config_beam);

    let formulas_ex = engine_ex
        .discover(&inputs, &targets, 1)
        .expect("exhaustive should succeed");
    let formulas_bm = engine_bm
        .discover(&inputs, &targets, 1)
        .expect("wide beam should succeed");

    // Both should find at least one formula.
    assert!(!formulas_ex.is_empty());
    assert!(!formulas_bm.is_empty());

    // Wide beam best MSE should be at most the exhaustive best MSE
    // (beam ⊆ exhaustive candidates, so beam can't find something exhaustive missed).
    let mse_ex = formulas_ex[0].mse;
    let mse_bm = formulas_bm[0].mse;
    assert!(
        mse_bm <= mse_ex * 1.05 + 1e-6,
        "wide beam should match exhaustive quality: exhaustive_mse={mse_ex}, beam_mse={mse_bm}"
    );
}

/// `discover_beam` with a very narrow beam still returns something useful.
#[test]
fn beam_narrow_completes_without_panic() {
    let (inputs, targets) = exp_dataset(10);

    let config = SymRegConfig {
        max_depth: 2,
        max_iter: 100,
        num_restarts: 1,
        strategy: SymRegStrategy::Beam { width: 3 },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine.discover(&inputs, &targets, 1);
    assert!(result.is_ok(), "narrow beam should not fail: {result:?}");
}

/// `discover_exhaustive` directly called returns same quality as `discover` (Exhaustive default).
#[test]
fn discover_exhaustive_matches_default_discover() {
    let (inputs, targets) = exp_dataset(15);

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 300,
        num_restarts: 2,
        seed: Some(7),
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config.clone());
    let formulas_via_dispatch = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");
    let formulas_direct = engine
        .discover_exhaustive(&inputs, &targets, 1)
        .expect("discover_exhaustive should succeed");

    // Both paths should return at least one formula.
    assert!(!formulas_via_dispatch.is_empty());
    assert!(!formulas_direct.is_empty());

    // MSE should be identical (same config + seed + same path).
    let mse_dispatch = formulas_via_dispatch[0].mse;
    let mse_direct = formulas_direct[0].mse;
    assert!(
        (mse_dispatch - mse_direct).abs() < 1e-10,
        "dispatch vs direct exhaustive MSE mismatch: {mse_dispatch} vs {mse_direct}"
    );
}
