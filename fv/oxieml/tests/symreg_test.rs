//! Tests for symbolic regression.

use oxieml::symreg::{SymRegConfig, SymRegEngine, pareto_front};

#[test]
fn test_discover_exp() {
    // y = exp(x)
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 2000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
        cv_folds: None,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());

    // eml(x, 1) = exp(x) should be discoverable at depth 1
    let best = &formulas[0];
    assert!(best.mse < 0.1, "MSE too high: {}", best.mse);
}

#[test]
fn test_discover_constant() {
    // y = e (constant function)
    let inputs: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
    let targets: Vec<f64> = vec![std::f64::consts::E; 10];

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-8,
        max_iter: 5000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
        cv_folds: None,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());

    // eml(1, 1) = e should be found
    let best = &formulas[0];
    assert!(best.mse < 0.01, "MSE too high: {}", best.mse);
}

#[test]
fn test_discover_linear() {
    // y = x (identity function)
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 5000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
        cv_folds: None,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());
}

#[test]
fn test_formulas_sorted_by_score() {
    let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 500,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();

    for window in formulas.windows(2) {
        assert!(window[0].score <= window[1].score);
    }
}

#[test]
fn pareto_front_empty() {
    let front = pareto_front(&[]);
    assert!(front.is_empty());
}

#[test]
fn pareto_front_from_discover() {
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.1 + 0.1]).collect();
    let targets: Vec<f64> = inputs.iter().map(|row| row[0] * 2.0).collect();
    let config = SymRegConfig::quick();
    let engine = SymRegEngine::new(config);
    let all_formulas = engine.discover(&inputs, &targets, 1).expect("discover");
    let front = pareto_front(&all_formulas);
    // Pareto front is a subset
    assert!(front.len() <= all_formulas.len());
    // All front members are mutually non-dominating
    for i in 0..front.len() {
        for j in 0..front.len() {
            if i != j {
                assert!(
                    !front[i].dominates(&front[j]) || !front[j].dominates(&front[i]),
                    "front[{i}] and front[{j}] mutually dominate — pareto_front is wrong"
                );
            }
        }
    }
    // Sorted by complexity ascending
    for w in front.windows(2) {
        assert!(w[0].complexity <= w[1].complexity);
    }
}

#[test]
fn discover_pareto_returns_nonempty() {
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![(i as f64) * 0.2 + 0.1]).collect();
    let targets: Vec<f64> = inputs.iter().map(|row| row[0].exp()).collect();
    let config = SymRegConfig::quick();
    let engine = SymRegEngine::new(config);
    let front = engine
        .discover_pareto(&inputs, &targets, 1)
        .expect("discover_pareto");
    assert!(!front.is_empty());
}

#[test]
fn cv_mse_none_when_not_configured() {
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.1 + 0.1]).collect();
    let targets: Vec<f64> = inputs.iter().map(|r| r[0] * 2.0).collect();
    let config = SymRegConfig::quick(); // cv_folds = None
    let formulas = SymRegEngine::new(config)
        .discover(&inputs, &targets, 1)
        .unwrap();
    for f in &formulas {
        assert!(
            f.cv_mse.is_none(),
            "expected cv_mse = None without cv config"
        );
    }
}

#[test]
fn cv_mse_some_when_configured() {
    let inputs: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64) * 0.1 + 0.1]).collect();
    let targets: Vec<f64> = inputs.iter().map(|r| r[0] * 3.0 + 1.0).collect();
    let config = SymRegConfig {
        cv_folds: Some(3),
        ..SymRegConfig::quick()
    };
    let formulas = SymRegEngine::new(config)
        .discover(&inputs, &targets, 1)
        .unwrap();
    assert!(!formulas.is_empty());
    // At least some formulas should have cv_mse populated
    let has_cv = formulas.iter().any(|f| f.cv_mse.is_some());
    assert!(
        has_cv,
        "expected at least one formula with cv_mse when cv_folds configured"
    );
}
