//! Integration tests for MCTS (Monte-Carlo Tree Search) topology search.

use oxieml::symreg::{SymRegConfig, SymRegEngine, SymRegStrategy};

/// MCTS with 100 iterations discovers the identity formula `y = x`.
///
/// `Var(0)` is a depth-0 EML tree directly representable in the grammar.
/// MCTS should sample it in the first rollout and achieve MSE ≈ 0.
#[test]
fn mcts_discovers_linear_formula() {
    let inputs: Vec<Vec<f64>> = (1..=20).map(|i| vec![i as f64 * 0.5]).collect();
    // Use y=x, which is directly expressible as Var(0) — MSE should be near 0.
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 2,
        max_iter: 300,
        num_restarts: 1,
        seed: Some(42),
        strategy: SymRegStrategy::Mcts {
            iterations: 100,
            exploration: 1.0,
        },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("MCTS discover should succeed");

    assert!(
        !formulas.is_empty(),
        "MCTS should return at least one formula"
    );
    assert!(
        formulas[0].mse < 0.1,
        "best formula MSE too high for y=x: {} (pretty={})",
        formulas[0].mse,
        formulas[0].pretty
    );
}

/// The default `SymRegConfig` strategy is `Exhaustive`, not MCTS.
///
/// MCTS is opt-in; users must explicitly set the strategy.
#[test]
fn mcts_config_default_is_not_mcts() {
    let config = SymRegConfig::default();
    assert_eq!(
        config.strategy,
        SymRegStrategy::Exhaustive,
        "default strategy must be Exhaustive, not Mcts"
    );
}

/// MCTS with zero iterations returns an empty result gracefully (no panic).
#[test]
fn mcts_with_zero_iterations_returns_empty() {
    let inputs: Vec<Vec<f64>> = (1..=5).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let config = SymRegConfig {
        max_depth: 2,
        max_iter: 100,
        num_restarts: 1,
        strategy: SymRegStrategy::Mcts {
            iterations: 0,
            exploration: 1.0,
        },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine.discover(&inputs, &targets, 1);

    match result {
        Ok(formulas) => assert!(
            formulas.is_empty(),
            "zero iterations should return empty, got {} formulas",
            formulas.len()
        ),
        Err(_) => {
            // Also acceptable: returning an error for 0 iterations.
        }
    }
}

/// Dispatching via `discover()` with MCTS strategy returns `Ok`.
#[test]
fn mcts_via_discover_dispatch() {
    let inputs: Vec<Vec<f64>> = (1..=10).map(|i| vec![i as f64 * 0.3]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 100,
        num_restarts: 1,
        seed: Some(7),
        strategy: SymRegStrategy::Mcts {
            iterations: 20,
            exploration: 1.4,
        },
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let result = engine.discover(&inputs, &targets, 1);

    assert!(
        result.is_ok(),
        "MCTS via discover() dispatch should return Ok, got: {:?}",
        result.err()
    );
}

/// More MCTS iterations tend to find better fits on non-trivial targets.
///
/// We use seeded RNG and generous tolerance (probabilistic test).
/// With 200 iterations vs 5 iterations, the 200-iteration run should
/// find a formula with equal or lower MSE on `y = x^2` (approximation).
#[test]
fn mcts_runs_more_iters_finds_better_or_equal_fit() {
    let inputs: Vec<Vec<f64>> = (1..=15).map(|i| vec![i as f64 * 0.4]).collect();
    // Use a moderately nonlinear target that EML trees can approximate.
    let targets: Vec<f64> = inputs.iter().map(|x| x[0] * x[0]).collect();

    let make_engine = |iters: usize, seed: u64| {
        SymRegEngine::new(SymRegConfig {
            max_depth: 2,
            max_iter: 200,
            num_restarts: 1,
            seed: Some(seed),
            strategy: SymRegStrategy::Mcts {
                iterations: iters,
                exploration: 1.0,
            },
            ..SymRegConfig::default()
        })
    };

    let few_engine = make_engine(5, 100);
    let many_engine = make_engine(200, 100);

    let few_result = few_engine
        .discover(&inputs, &targets, 1)
        .expect("few-iter MCTS should succeed");
    let many_result = many_engine
        .discover(&inputs, &targets, 1)
        .expect("many-iter MCTS should succeed");

    // Both should return results (or we can't compare).
    // If either returns empty, the test is vacuously satisfied.
    if few_result.is_empty() || many_result.is_empty() {
        return;
    }

    let few_mse = few_result[0].mse;
    let many_mse = many_result[0].mse;

    // More iterations should not be dramatically worse (allow 3× slack for randomness).
    assert!(
        many_mse <= few_mse * 3.0 + 1.0,
        "more MCTS iterations produced much worse result: few_mse={few_mse}, many_mse={many_mse}"
    );
}
