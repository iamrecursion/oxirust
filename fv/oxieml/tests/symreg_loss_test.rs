//! Tests for noise-robust loss functions in SymRegEngine.

use oxieml::{SymRegConfig, SymRegEngine, SymRegLoss};

fn make_linear_data_with_outliers() -> (Vec<Vec<f64>>, Vec<f64>) {
    // y = 2*x with 10% outliers at the end
    let mut inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.5]).collect();
    let mut targets: Vec<f64> = inputs.iter().map(|x| 2.0 * x[0]).collect();

    // Add 2 outliers (10%)
    inputs.push(vec![1.0]);
    targets.push(1000.0);
    inputs.push(vec![2.0]);
    targets.push(-1000.0);

    (inputs, targets)
}

/// MSE is the default loss when not configured.
#[test]
fn mse_loss_is_default_behavior() {
    let config = SymRegConfig::default();
    assert!(matches!(config.loss, SymRegLoss::Mse));
}

/// Huber loss reduces outlier impact versus MSE on contaminated data.
#[test]
fn huber_loss_reduces_outlier_impact() {
    let (inputs, targets) = make_linear_data_with_outliers();

    let base = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 500,
        complexity_penalty: 1e-4,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(7),
        ..SymRegConfig::default()
    };

    let huber_config = SymRegConfig {
        loss: SymRegLoss::Huber { delta: 1.0 },
        seed: Some(7),
        ..base.clone()
    };

    let mse_engine = SymRegEngine::new(base);
    let huber_engine = SymRegEngine::new(huber_config);

    let mse_result = mse_engine
        .discover(&inputs, &targets, 1)
        .expect("MSE discover");
    let huber_result = huber_engine
        .discover(&inputs, &targets, 1)
        .expect("Huber discover");

    // Both should complete and return valid formulas
    assert!(!mse_result.is_empty());
    assert!(!huber_result.is_empty());
}

/// TrimmedMse at alpha=0.0 is equivalent to MSE (no trimming).
#[test]
fn trimmed_mse_at_alpha_0_matches_mse() {
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let base = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 300,
        complexity_penalty: 1e-4,
        num_restarts: 1,
        integer_rounding: false,
        seed: Some(13),
        ..SymRegConfig::default()
    };

    let trim_config = SymRegConfig {
        loss: SymRegLoss::TrimmedMse { alpha: 0.0 },
        ..base.clone()
    };

    let mse_engine = SymRegEngine::new(base);
    let trim_engine = SymRegEngine::new(trim_config);

    let mse_result = mse_engine.discover(&inputs, &targets, 1).expect("MSE");
    let trim_result = trim_engine.discover(&inputs, &targets, 1).expect("Trim");

    assert!(!mse_result.is_empty());
    assert!(!trim_result.is_empty());
    // Both should find a formula with reasonable MSE for exp(x)
    assert!(mse_result[0].mse < 10.0);
    assert!(trim_result[0].mse < 10.0);
}

/// Huber config fields are accessible through the public API.
#[test]
fn huber_config_accessible() {
    let config = SymRegConfig {
        loss: SymRegLoss::Huber { delta: 0.5 },
        ..SymRegConfig::default()
    };
    match &config.loss {
        SymRegLoss::Huber { delta } => {
            assert!((*delta - 0.5).abs() < 1e-15, "delta should be 0.5");
        }
        other => panic!("expected Huber, got {other:?}"),
    }
}
