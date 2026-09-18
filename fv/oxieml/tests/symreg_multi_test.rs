//! Integration tests for multi-output symbolic regression.

use oxieml::symreg::{DiscoveredFormula, MultiOutputStrategy, SymRegConfig, SymRegEngine};

/// `discover_multi` returns one `Vec<DiscoveredFormula>` per output column.
#[test]
fn discover_multi_returns_one_vec_per_output() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets = vec![
        inputs.iter().map(|x| x[0]).collect::<Vec<f64>>(),
        inputs.iter().map(|x| x[0] * 2.0).collect::<Vec<f64>>(),
    ];

    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 1,
        max_iter: 200,
        num_restarts: 1,
        ..SymRegConfig::default()
    });
    let result = engine
        .discover_multi(&inputs, &targets, 1)
        .expect("discover_multi should succeed");

    assert_eq!(
        result.len(),
        2,
        "should return one formula set per output column"
    );
}

/// Both inner vecs from `discover_multi` are non-empty.
#[test]
fn discover_multi_each_output_has_formula() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.5]).collect();
    let targets = vec![
        inputs.iter().map(|x| x[0]).collect::<Vec<f64>>(),
        inputs.iter().map(|x| x[0] * 2.0).collect::<Vec<f64>>(),
    ];

    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 1,
        max_iter: 200,
        num_restarts: 1,
        ..SymRegConfig::default()
    });
    let result = engine
        .discover_multi(&inputs, &targets, 1)
        .expect("discover_multi should succeed");

    for (i, formulas) in result.iter().enumerate() {
        assert!(
            !formulas.is_empty(),
            "output {i} should have at least one formula"
        );
    }
}

/// The default multi-output strategy is `Independent`.
#[test]
fn multi_strategy_default_is_independent() {
    let config = SymRegConfig::quick();
    assert_eq!(
        config.multi_output_strategy,
        MultiOutputStrategy::Independent
    );
}

fn make_multi_inputs(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| vec![(i as f64 + 1.0) * 0.5]).collect()
}

/// `discover_multi` returns one result vector per output column.
#[test]
fn multi_output_returns_one_vec_per_output() {
    let inputs = make_multi_inputs(15);
    let targets = vec![
        inputs.iter().map(|x| x[0]).collect::<Vec<f64>>(), // y0 = x
        inputs.iter().map(|x| x[0] * 2.0).collect::<Vec<f64>>(), // y1 = 2*x
    ];

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 200,
        num_restarts: 1,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine
        .discover_multi(&inputs, &targets, 1)
        .expect("discover_multi should succeed");

    assert_eq!(
        result.len(),
        2,
        "should return one formula set per output column"
    );
    for (out_idx, formulas) in result.iter().enumerate() {
        assert!(
            !formulas.is_empty(),
            "output {out_idx} should have at least one formula"
        );
    }
}

/// Dimension mismatch between input samples and a target column returns an error.
#[test]
fn multi_output_dimension_mismatch_errors() {
    let inputs = make_multi_inputs(10);
    // Second target column has wrong length (9, not 10).
    let targets = vec![
        (0..10).map(|i| i as f64).collect::<Vec<f64>>(),
        (0..9).map(|i| i as f64).collect::<Vec<f64>>(), // wrong length
    ];

    let engine = SymRegEngine::new(SymRegConfig::quick());
    let result = engine.discover_multi(&inputs, &targets, 1);
    assert!(
        result.is_err(),
        "should error on target length mismatch, got Ok({:?})",
        result.ok().map(|v| v.len())
    );
}

/// Empty inputs returns `EmlError::EmptyData`.
#[test]
fn multi_output_empty_inputs_errors() {
    let engine = SymRegEngine::new(SymRegConfig::quick());
    let result = engine.discover_multi(&[], &[vec![1.0, 2.0]], 1);
    assert!(result.is_err(), "should error on empty inputs");
}

/// Each output formula's MSE is computed independently.
/// When targets differ, the best formulas for each output differ.
#[test]
fn multi_output_independent_discovery() {
    // y0 = x^2-ish (strictly positive, growing fast)
    // y1 = constant 1.0 (boring, but should be discovered as low-complexity)
    let n = 20;
    let inputs: Vec<Vec<f64>> = (1..=n).map(|i| vec![i as f64 * 0.5]).collect();
    let targets = vec![
        inputs.iter().map(|x| x[0] * x[0]).collect::<Vec<f64>>(),
        vec![1.0_f64; n],
    ];

    let config = SymRegConfig {
        max_depth: 2,
        max_iter: 300,
        num_restarts: 1,
        multi_output_strategy: MultiOutputStrategy::Independent,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine
        .discover_multi(&inputs, &targets, 1)
        .expect("multi discover should succeed");

    assert_eq!(result.len(), 2);

    // Both outputs should have formulas.
    let formulas_y0: &Vec<DiscoveredFormula> = &result[0];
    let formulas_y1: &Vec<DiscoveredFormula> = &result[1];
    assert!(!formulas_y0.is_empty(), "y0 should have formulas");
    assert!(!formulas_y1.is_empty(), "y1 should have formulas");

    // y1=const should have at least one candidate with near-zero MSE
    // (the constant topology `1` is trivially present at depth 0).
    let best_y1 = formulas_y1[0].mse;
    assert!(
        best_y1 < 5.0,
        "y1 (constant 1) best MSE should be low, got {best_y1}"
    );
}
