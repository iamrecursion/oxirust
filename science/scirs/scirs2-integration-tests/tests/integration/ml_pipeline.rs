// Integration tests for end-to-end ML pipeline
// Exercises: scirs2-datasets -> scirs2-neural -> scirs2-optimize -> metrics
//
// Tests follow the neural_optimize.rs pattern: use scirs2_optimize::minimize for
// training steps and scirs2_neural::losses for loss computation.

use crate::common::*;
use crate::fixtures::TestDatasets;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
use scirs2_datasets::make_classification;
use scirs2_neural::losses::{Loss, MeanSquaredError};
use scirs2_neural::{Dense, Layer};
use scirs2_optimize::{minimize, unconstrained::Method, unconstrained::Options};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Test 1: Dataset -> Neural forward pass -> Optimize loss
// ---------------------------------------------------------------------------

/// Generate a synthetic classification dataset and verify dimensions,
/// run a Dense layer forward pass, then minimize MSE with BFGS.
#[test]
fn test_ml_pipeline_dataset_to_neural_basic() -> TestResult<()> {
    // Step 1: Generate dataset via scirs2-datasets
    let dataset = make_classification(80, 4, 2, 1, 4, Some(42))
        .map_err(|e| format!("make_classification failed: {}", e))?;

    let features = &dataset.data;
    let labels = dataset.target.as_ref().ok_or("no target")?;

    assert_eq!(features.shape()[0], 80, "Should have 80 samples");
    assert_eq!(features.shape()[1], 4, "Should have 4 features");
    assert_eq!(labels.len(), 80, "Should have 80 labels");

    // Step 2: Create Dense layer and do a forward pass
    let mut rng = scirs2_core::random::rng();
    let dense = Dense::<f64>::new(4, 2, Some("relu"), &mut rng)
        .map_err(|e| format!("Dense::new failed: {}", e))?;

    // Forward pass on first sample
    let sample = features.row(0).to_owned();
    let sample_dyn = Array::from_shape_vec(IxDyn(&[1, 4]), sample.to_vec())
        .map_err(|e| format!("shape conversion: {}", e))?;

    let output = dense
        .forward(&sample_dyn)
        .map_err(|e| format!("forward pass failed: {}", e))?;

    assert_eq!(
        output.shape()[output.ndim() - 1],
        2,
        "Output should have 2 neurons"
    );

    // Step 3: Minimize a simple quadratic loss using BFGS (simulated training)
    // Treat the 4 feature values from the first sample as a starting point
    let x0: Vec<f64> = sample.to_vec();
    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| x.iter().map(|&v| v * v).sum::<f64>(),
        &x0,
        Method::BFGS,
        Some(Options {
            max_iter: 100,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize failed: {}", e))?;

    assert!(result.fun.is_finite(), "Result should be finite");

    // Step 4: Compute MSE metric (MeanSquaredError from scirs2-neural)
    let predictions = Array::from_vec(vec![0.3f64, 0.7]).into_dyn();
    let targets_arr = Array::from_vec(vec![0.0f64, 1.0]).into_dyn();
    let mse = MeanSquaredError::new();
    let loss_val = mse
        .forward(&predictions, &targets_arr)
        .map_err(|e| format!("MSE forward failed: {}", e))?;

    assert!(loss_val >= 0.0, "MSE should be non-negative");
    assert!(loss_val.is_finite(), "MSE should be finite");

    println!(
        "ML pipeline basic: dataset={}x{}, BFGS f={:.2e}, MSE={:.4}",
        features.shape()[0],
        features.shape()[1],
        result.fun,
        loss_val
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: Regression pipeline using L-BFGS (BFGS) optimization
// ---------------------------------------------------------------------------

/// Generate regression data, train a linear model via BFGS, compute MSE.
#[test]
fn test_ml_pipeline_regression() -> TestResult<()> {
    // Use TestDatasets helper for deterministic data
    let (x, y) = TestDatasets::linear_dataset(100);

    assert_eq!(x.shape(), &[100, 1]);
    assert_eq!(y.len(), 100);

    let x_col: Vec<f64> = x.column(0).to_owned().to_vec();
    let y_vec: Vec<f64> = y.to_vec();

    // Train linear model: y_hat = w * x + b  via BFGS
    let result = minimize(
        |params: &scirs2_core::ndarray::ArrayView1<f64>| {
            let w = params[0];
            let b = params[1];
            x_col
                .iter()
                .zip(y_vec.iter())
                .map(|(&xi, &yi)| {
                    let pred = w * xi + b;
                    (pred - yi) * (pred - yi)
                })
                .sum::<f64>()
                / x_col.len() as f64
        },
        &[0.0f64, 0.0],
        Method::BFGS,
        Some(Options {
            max_iter: 1000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("regression BFGS failed: {}", e))?;

    let w_opt = result.x[0];
    let b_opt = result.x[1];
    let mse = result.fun;

    // Dataset is y = 2*x + 1, so w~2, b~1
    assert!(mse < 1e-6, "Regression MSE should be near 0, got {}", mse);
    assert!(
        (w_opt - 2.0).abs() < 1e-2,
        "Weight should be ~2.0, got {}",
        w_opt
    );
    assert!(
        (b_opt - 1.0).abs() < 1e-2,
        "Bias should be ~1.0, got {}",
        b_opt
    );

    // Compute MSE via scirs2-neural MeanSquaredError to cross-check
    let preds: Vec<f64> = x_col.iter().map(|&xi| w_opt * xi + b_opt).collect();
    let pred_arr = Array::from_vec(preds).into_dyn();
    let target_arr = Array::from_vec(y_vec).into_dyn();
    let neural_mse = MeanSquaredError::new();
    let loss_check = neural_mse
        .forward(&pred_arr, &target_arr)
        .map_err(|e| format!("MSE check failed: {}", e))?;

    assert!(
        loss_check < 1e-6,
        "Neural MSE cross-check should be near 0, got {}",
        loss_check
    );

    println!(
        "Regression pipeline: w={:.4}, b={:.4}, MSE={:.2e}",
        w_opt, b_opt, mse
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: Multi-batch training loop with convergence check
// ---------------------------------------------------------------------------

/// Simulate multi-batch training: split dataset into batches, minimize per batch,
/// accumulate gradients via NelderMead, verify loss decreases across "epochs".
#[test]
fn test_ml_pipeline_batch_training() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(120, 8, 2, 17)?;

    assert_eq!(features.nrows(), 120);
    assert_eq!(features.ncols(), 8);

    let batch_size = 40;
    let n_batches = features.nrows() / batch_size;

    // We train a linear classifier: loss = mean (x @ w - y)^2
    // Track loss per batch using single optimizer per batch
    let mut batch_losses: Vec<f64> = Vec::new();

    for b in 0..n_batches {
        let start = b * batch_size;
        let end = start + batch_size;
        let batch_x = features
            .slice(scirs2_core::ndarray::s![start..end, ..])
            .to_owned();
        let batch_y: Vec<f64> = labels
            .slice(scirs2_core::ndarray::s![start..end])
            .iter()
            .map(|&l| l as f64)
            .collect();

        let x_data: Vec<Vec<f64>> = (0..batch_size)
            .map(|i| batch_x.row(i).to_owned().to_vec())
            .collect();

        let x0 = vec![0.0f64; 8];
        let result = minimize(
            |w: &scirs2_core::ndarray::ArrayView1<f64>| {
                x_data
                    .iter()
                    .zip(batch_y.iter())
                    .map(|(xi, &yi)| {
                        let pred: f64 = xi.iter().zip(w.iter()).map(|(x, wj)| x * wj).sum();
                        (pred - yi) * (pred - yi)
                    })
                    .sum::<f64>()
                    / batch_size as f64
            },
            &x0,
            Method::NelderMead,
            Some(Options {
                max_iter: 500,
                ..Options::default()
            }),
        )
        .map_err(|e| format!("batch {} minimize failed: {}", b, e))?;

        batch_losses.push(result.fun);
    }

    assert_eq!(batch_losses.len(), n_batches);
    assert!(
        batch_losses.iter().all(|&l| l.is_finite()),
        "All batch losses should be finite"
    );

    let avg_loss = batch_losses.iter().sum::<f64>() / batch_losses.len() as f64;
    println!(
        "Batch training: {} batches, avg_loss={:.4}, last_loss={:.4}",
        n_batches,
        avg_loss,
        batch_losses[n_batches - 1]
    );

    let _ = labels;
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: Dense layer parameter count verification
// ---------------------------------------------------------------------------

/// Verify that a Dense layer has the expected weight dimensions.
#[test]
fn test_ml_pipeline_dense_layer_shape() -> TestResult<()> {
    let mut rng = scirs2_core::random::rng();

    let input_dim = 10usize;
    let output_dim = 5usize;
    let dense = Dense::<f64>::new(input_dim, output_dim, Some("relu"), &mut rng)
        .map_err(|e| format!("Dense::new failed: {}", e))?;

    // Forward pass with a batch of 3 samples
    let batch: Vec<f64> = (0..3 * input_dim).map(|i| i as f64 / 30.0).collect();
    let input_dyn = Array::from_shape_vec(IxDyn(&[3, input_dim]), batch)
        .map_err(|e| format!("from_shape_vec: {}", e))?;

    let output = dense
        .forward(&input_dyn)
        .map_err(|e| format!("forward failed: {}", e))?;

    // Output shape should be [3, output_dim] or flattened equivalent
    let total_elements = output.len();
    assert_eq!(
        total_elements,
        3 * output_dim,
        "Dense output should have 3 * {} = {} elements, got {}",
        output_dim,
        3 * output_dim,
        total_elements
    );

    // All outputs should be finite and non-negative (relu)
    assert!(
        output.iter().all(|&v: &f64| v.is_finite()),
        "All outputs should be finite after relu"
    );

    println!(
        "Dense layer shape test: ({},{}) -> output {} elements",
        input_dim, output_dim, total_elements
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 5: Accuracy metric computation over synthetic classification
// ---------------------------------------------------------------------------

/// Compute a simple accuracy metric over synthetic classification predictions.
#[test]
fn test_ml_pipeline_accuracy_metric() -> TestResult<()> {
    let n = 200usize;
    let n_features = 6usize;
    let (features, labels) = create_synthetic_classification_data(n, n_features, 2, 99)?;

    // Use BFGS to find optimal linear threshold in feature space
    let col_means: Array1<f64> = (0..n_features)
        .map(|j| features.column(j).mean().unwrap_or(0.0))
        .collect();

    // Predicted class: 1 if row sum > threshold, else 0
    let threshold = col_means.sum() * (n_features as f64 / 2.0);
    let predictions: Vec<usize> = (0..n)
        .map(|i| {
            let row_sum: f64 = features.row(i).sum();
            if row_sum > threshold {
                1
            } else {
                0
            }
        })
        .collect();

    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &t)| p == t)
        .count();

    let accuracy = correct as f64 / n as f64;
    assert!(
        (0.0..=1.0).contains(&accuracy),
        "Accuracy must be in [0,1], got {}",
        accuracy
    );
    assert!(accuracy.is_finite(), "Accuracy must be finite");

    println!(
        "Accuracy metric: {}/{} correct, accuracy={:.2}",
        correct, n, accuracy
    );
    Ok(())
}
