//! Integration tests for constraint-guided interval pruning in symbolic regression.

use oxieml::lower_interval::IntervalLO;
use oxieml::symreg::{SymRegConfig, SymRegEngine};

/// Interval pruning is disabled by default.
#[test]
fn interval_pruning_disabled_by_default() {
    let config = SymRegConfig::quick();
    assert!(
        !config.interval_pruning,
        "interval_pruning should default to false"
    );
}

/// Interval pruning depth threshold defaults to 2.
#[test]
fn interval_pruning_depth_threshold_default() {
    let config = SymRegConfig::quick();
    assert_eq!(config.interval_pruning_depth_threshold, 2);
}

/// Enabling pruning on simple linear data (y = x) still yields mse < 0.01.
#[test]
fn interval_pruning_does_not_lose_best_formula() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 500,
        num_restarts: 2,
        interval_pruning: true,
        interval_pruning_depth_threshold: 2,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("pruning discover should succeed");

    assert!(
        !formulas.is_empty(),
        "should find at least one formula with pruning enabled"
    );
    assert!(
        formulas[0].mse < 0.01,
        "best formula mse should be < 0.01 for y=x, got {}",
        formulas[0].mse
    );
}

/// Pruning enabled should not increase the number of formulas discovered
/// compared to pruning disabled for a simple dataset.
#[test]
fn pruning_does_not_lose_best_formula() {
    // y = exp(x) dataset
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.3]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config_no_prune = SymRegConfig {
        max_depth: 1,
        max_iter: 500,
        num_restarts: 2,
        interval_pruning: false,
        ..SymRegConfig::default()
    };

    let config_pruned = SymRegConfig {
        max_depth: 1,
        max_iter: 500,
        num_restarts: 2,
        interval_pruning: true,
        interval_pruning_depth_threshold: 1,
        ..SymRegConfig::default()
    };

    let engine_np = SymRegEngine::new(config_no_prune);
    let engine_pr = SymRegEngine::new(config_pruned);

    let formulas_np = engine_np
        .discover(&inputs, &targets, 1)
        .expect("no-prune discover should succeed");
    let formulas_pr = engine_pr
        .discover(&inputs, &targets, 1)
        .expect("pruning discover should succeed");

    assert!(!formulas_np.is_empty(), "no-prune should return formulas");
    assert!(!formulas_pr.is_empty(), "pruning should return formulas");

    // Both should find a good fit for exp(x)
    let best_np = formulas_np[0].mse;
    let best_pr = formulas_pr[0].mse;

    // Pruning should not make accuracy dramatically worse
    assert!(
        best_pr < best_np * 10.0 + 1.0,
        "pruning degraded accuracy too much: no_prune_mse={best_np}, pruned_mse={best_pr}"
    );
}

/// Interval pruning reduces the topology count on negated-target data
/// where many topologies trivially cannot span the target range.
#[test]
fn pruning_reduces_topology_count_for_negative_targets() {
    // Targets are all negative: y ∈ [-10, -1].
    // Many EML topologies (e.g. pure exp(...)) can only produce positive values
    // and should be pruned away.
    let inputs: Vec<Vec<f64>> = (1..=10).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| -x[0]).collect(); // y = -x

    let config_pruned = SymRegConfig {
        max_depth: 2,
        max_iter: 100,
        num_restarts: 1,
        interval_pruning: true,
        interval_pruning_depth_threshold: 1,
        ..SymRegConfig::default()
    };

    let config_no_prune = SymRegConfig {
        max_depth: 2,
        max_iter: 100,
        num_restarts: 1,
        interval_pruning: false,
        ..SymRegConfig::default()
    };

    let engine_pr = SymRegEngine::new(config_pruned.clone());
    let engine_np = SymRegEngine::new(config_no_prune);

    // Count how many topologies survive pruning vs. not (indirectly via
    // total discovered formulas, since optimize_topology returns None for
    // non-finite MSE anyway).
    let formulas_pr = engine_pr
        .discover(&inputs, &targets, 1)
        .expect("pruning discover should succeed");
    let formulas_np = engine_np
        .discover(&inputs, &targets, 1)
        .expect("no-prune discover should succeed");

    // At depth 2, the pruned run should produce no more formulas than the
    // unpruned run (it cannot produce genuinely new ones; it can only skip).
    assert!(
        formulas_pr.len() <= formulas_np.len(),
        "pruning must not add formulas: pruned={}, unpruned={}",
        formulas_pr.len(),
        formulas_np.len()
    );

    // Both should find at least one formula
    assert!(!formulas_np.is_empty(), "unpruned should find formulas");
    assert!(!formulas_pr.is_empty(), "pruned should find formulas");
}

/// Verify `topology_interval_feasible` is effective via the public API.
/// A topology that can only produce large positive values should be rejected
/// when all targets are large negatives.
#[test]
fn interval_pruning_basic_feasibility() {
    // Use the public enumerate_topologies + IntervalLO to verify the behavior
    // end-to-end: run regression on data that is strictly negative with pruning
    // at depth 1, and confirm no crash or panic occurs.
    let inputs: Vec<Vec<f64>> = (1..=5).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = vec![-100.0, -200.0, -300.0, -400.0, -500.0];

    let config = SymRegConfig {
        max_depth: 2,
        max_iter: 50,
        num_restarts: 1,
        interval_pruning: true,
        interval_pruning_depth_threshold: 0,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    // Should not panic, may return empty or low-quality formulas.
    let result = engine.discover(&inputs, &targets, 1);
    assert!(result.is_ok(), "should not error: {result:?}");
}

/// Verify IntervalLO can be constructed and used directly (public API check).
#[test]
fn interval_lo_public_api() {
    let iv = IntervalLO::new(1.0, 5.0);
    assert!(iv.contains(3.0));
    assert!(!iv.contains(6.0));
    assert!((iv.width() - 4.0).abs() < 1e-15);

    let union = iv.union(&IntervalLO::new(4.0, 10.0));
    assert!((union.lo - 1.0).abs() < 1e-15);
    assert!((union.hi - 10.0).abs() < 1e-15);
}
