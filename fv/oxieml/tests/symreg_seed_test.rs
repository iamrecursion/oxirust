//! Tests for deterministic RNG seeding in SymRegEngine.

use oxieml::{SymRegConfig, SymRegEngine};

fn make_exp_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();
    (inputs, targets)
}

/// Two discover() calls with the same seed must produce identical results.
#[test]
fn seeded_discover_is_deterministic() {
    let (inputs, targets) = make_exp_data();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 100,
        complexity_penalty: 1e-4,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(42),
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config.clone());
    let r1 = engine
        .discover(&inputs, &targets, 1)
        .expect("first discover should succeed");
    let r2 = engine
        .discover(&inputs, &targets, 1)
        .expect("second discover should succeed");

    assert_eq!(
        r1.len(),
        r2.len(),
        "seeded runs must produce same number of formulas"
    );

    for (f1, f2) in r1.iter().zip(r2.iter()) {
        assert_eq!(
            f1.params, f2.params,
            "seeded runs must produce identical params"
        );
        assert!(
            (f1.mse - f2.mse).abs() < 1e-15,
            "seeded runs must produce identical MSE"
        );
    }
}

/// Two calls with different seeds may produce different results.
/// (Probabilistic test: verify the engine runs without error; actual
/// divergence is expected but not guaranteed for very shallow trees.)
#[test]
fn different_seeds_may_differ() {
    let (inputs, targets) = make_exp_data();

    let make_config = |seed: u64| SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-8,
        max_iter: 200,
        complexity_penalty: 1e-4,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(seed),
        ..SymRegConfig::default()
    };

    let e1 = SymRegEngine::new(make_config(1));
    let e2 = SymRegEngine::new(make_config(99999));

    let r1 = e1.discover(&inputs, &targets, 1).expect("discover seed 1");
    let r2 = e2
        .discover(&inputs, &targets, 1)
        .expect("discover seed 99999");

    // Both should succeed and return formulas
    assert!(!r1.is_empty(), "seed 1 should find formulas");
    assert!(!r2.is_empty(), "seed 99999 should find formulas");
}

/// Unseeded runs complete without error.
#[test]
fn unseeded_runs_complete() {
    let (inputs, targets) = make_exp_data();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 100,
        complexity_penalty: 1e-4,
        num_restarts: 1,
        integer_rounding: false,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine.discover(&inputs, &targets, 1);
    assert!(result.is_ok(), "unseeded discover should succeed");
    assert!(
        !result.unwrap().is_empty(),
        "unseeded run should find formulas"
    );
}
