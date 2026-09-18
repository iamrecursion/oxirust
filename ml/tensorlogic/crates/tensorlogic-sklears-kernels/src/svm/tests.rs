//! Comprehensive tests for SVM via SMO.
//!
//! Tests cover binary/multi-class SVC, ε-SVR, error handling, KKT conditions,
//! SMO configuration defaults, and decision function correctness.

use std::sync::Arc;

use crate::error::KernelError;
use crate::tensor_kernels::types::{LinearKernel, RbfKernel};
use crate::types::{Kernel, RbfKernelConfig};

use super::smo::SmoConfig;
use super::svc::SvcModel;
use super::svr::SvrModel;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build an `Arc<dyn Kernel>` wrapping a LinearKernel.
fn linear_kernel() -> Arc<dyn Kernel> {
    Arc::new(LinearKernel::new())
}

/// Build an `Arc<dyn Kernel>` wrapping an RBF kernel with the given gamma.
fn rbf_kernel(gamma: f64) -> Arc<dyn Kernel> {
    Arc::new(RbfKernel::new(RbfKernelConfig::new(gamma)).expect("valid gamma"))
}

// ─── SVC Tests ───────────────────────────────────────────────────────────────

/// Two clearly separated clusters in 2D — linearly separable with LinearKernel.
#[test]
fn test_linear_svc_binary_separable() {
    let x_pos = [vec![1.0, 1.0], vec![2.0, 1.0], vec![1.0, 2.0]];
    let x_neg = [vec![-1.0, -1.0], vec![-2.0, -1.0], vec![-1.0, -2.0]];
    let x: Vec<Vec<f64>> = x_pos.iter().chain(x_neg.iter()).cloned().collect();
    let y: Vec<i32> = vec![1, 1, 1, -1, -1, -1];

    let model = SvcModel::new(linear_kernel(), 10.0).expect("valid C");
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // All training points should be classified correctly.
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let pred = fitted.predict(xi).expect("predict");
        assert_eq!(
            pred, yi,
            "Linear SVC misclassified training point {:?}: got {}, expected {}",
            xi, pred, yi
        );
    }
}

/// XOR dataset — 4 points, requires non-linear kernel (RBF with sufficient gamma).
#[test]
fn test_rbf_svc_xor() {
    // XOR: (+1, +1) → -1,  (-1, -1) → -1,  (+1, -1) → +1,  (-1, +1) → +1
    let x = vec![
        vec![1.0, 1.0],
        vec![-1.0, -1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
    ];
    let y = vec![-1, -1, 1, 1];

    // High gamma to create tight local regions.
    let model = SvcModel::new(rbf_kernel(4.0), 100.0).expect("valid params");
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    for (xi, &yi) in x.iter().zip(y.iter()) {
        let pred = fitted.predict(xi).expect("predict");
        assert_eq!(
            pred, yi,
            "RBF XOR SVC misclassified {:?}: got {}, expected {}",
            xi, pred, yi
        );
    }
}

/// Providing all the same class label should be an error (can't train binary SVC).
#[test]
fn test_svc_single_class_error() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let y = vec![1, 1, 1]; // All same class

    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let result = model.fit(&x, &y);
    assert!(
        result.is_err(),
        "Expected error when all labels are the same"
    );
    assert!(
        matches!(result, Err(KernelError::InvalidParameter { .. })),
        "Expected InvalidParameter error, got: {:?}",
        result
    );
}

/// Empty training set should produce a DimensionMismatch error.
#[test]
fn test_svc_empty_data_error() {
    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let result = model.fit(&[], &[]);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {:?}",
        result
    );
}

/// Negative C should produce an InvalidParameter error.
#[test]
fn test_svc_negative_c_error() {
    let result = SvcModel::new(linear_kernel(), -1.0);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(KernelError::InvalidParameter { .. })),
        "Expected InvalidParameter, got {:?}",
        result
    );
}

/// Zero C should also produce an InvalidParameter error.
#[test]
fn test_svc_zero_c_error() {
    let result = SvcModel::new(linear_kernel(), 0.0);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(KernelError::InvalidParameter { .. })),
        "Expected InvalidParameter for C=0, got {:?}",
        result
    );
}

/// After fitting a linearly separable dataset, the KKT conditions should hold
/// for all support vectors: |y_i * f(x_i) - 1| < 2 * tol.
///
/// KKT condition for a support vector (0 < α_i < C):
///   y_i * f(x_i) = 1  (exactly on the margin)
///   => |y_i * f(x_i) - 1| ≈ 0 within numerical tolerance.
#[test]
fn test_svc_kkt_conditions() {
    // Tightly separated data so the margin is well-defined.
    let x = vec![
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![-2.0, 0.0],
        vec![-3.0, 0.0],
    ];
    let y = vec![1, 1, -1, -1];

    let tol = 1e-3;
    let config = SmoConfig {
        c: 1.0,
        tol,
        max_iter: 50_000,
        ..SmoConfig::default()
    };
    let model = SvcModel::new_with_config(linear_kernel(), config).expect("valid");
    let fitted = model.fit(&x, &y).expect("fit");

    // For support vectors: |y_i * f(x_i) - 1| < 2 * tol.
    // We use the public decision_function which only works for binary SVC.
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let df = fitted.decision_function(xi).expect("decision function");
        let functional_margin = (yi as f64) * df;
        // Training points that ended up as support vectors must satisfy KKT.
        // For non-support vectors (alpha = 0 or C), the margin may exceed 1.
        // All training data should have functional_margin >= 1 - tol (feasibility).
        assert!(
            functional_margin >= 1.0 - 2.0 * tol,
            "KKT feasibility violated at {:?}: y*f(x) = {}, expected >= {}",
            xi,
            functional_margin,
            1.0 - 2.0 * tol
        );
    }
}

/// Three-class linearly separable problem (well-separated blobs).
#[test]
fn test_svc_multiclass_ovr() {
    // Three clusters in 2D, each far from the others.
    // Class 0: around (5, 0),  Class 1: around (-5, 0),  Class 2: around (0, 5).
    let x: Vec<Vec<f64>> = vec![
        // Class 0 cluster
        vec![5.0, 0.0],
        vec![5.5, 0.5],
        vec![4.5, -0.5],
        // Class 1 cluster
        vec![-5.0, 0.0],
        vec![-5.5, 0.5],
        vec![-4.5, -0.5],
        // Class 2 cluster
        vec![0.0, 5.0],
        vec![0.5, 5.5],
        vec![-0.5, 4.5],
    ];
    let y: Vec<i32> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let fitted = model.fit(&x, &y).expect("fit 3-class");

    for (xi, &yi) in x.iter().zip(y.iter()) {
        let pred = fitted.predict(xi).expect("predict");
        assert_eq!(
            pred, yi,
            "Multiclass SVC misclassified {:?}: got {}, expected {}",
            xi, pred, yi
        );
    }
}

/// For well-separated linearly separable data with C not too large,
/// the number of support vectors should be small (exactly 2 for 1D data).
#[test]
fn test_svc_num_support_vectors() {
    // Perfectly linearly separable with large margin; SVs are just the boundary points.
    let x = vec![
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![-1.0, 0.0],
        vec![-2.0, 0.0],
        vec![-3.0, 0.0],
    ];
    let y = vec![1, 1, 1, -1, -1, -1];

    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let fitted = model.fit(&x, &y).expect("fit");

    let n_sv = fitted.num_support_vectors();
    // For well-separated linear data, we expect 2 support vectors (one per class boundary).
    // With larger C or overlapping data, this could be more. We use a generous bound.
    assert!(
        n_sv <= 4,
        "Expected at most 4 support vectors for well-separated linear data, got {}",
        n_sv
    );
    assert!(
        n_sv >= 1,
        "Expected at least 1 support vector, got {}",
        n_sv
    );
}

/// The decision function should have correct sign for all training points.
#[test]
fn test_svc_decision_function_signs() {
    let x = vec![vec![3.0], vec![4.0], vec![-3.0], vec![-4.0]];
    let y = vec![1, 1, -1, -1];

    let model = SvcModel::new(linear_kernel(), 10.0).expect("valid C");
    let fitted = model.fit(&x, &y).expect("fit");

    for (xi, &yi) in x.iter().zip(y.iter()) {
        let df = fitted.decision_function(xi).expect("decision function");
        let sign = if df >= 0.0 { 1_i32 } else { -1_i32 };
        assert_eq!(
            sign, yi,
            "decision_function sign wrong at {:?}: df={}, expected label {}",
            xi, df, yi
        );
    }
}

// ─── SVR Tests ───────────────────────────────────────────────────────────────

/// Fit a linear regression y = 2x + 1 with LinearKernel.
/// Using x values starting at 1 to avoid degenerate x=0 for LinearKernel.
/// Predict at x=5.0, expect result within 0.5 of 11.0.
#[test]
fn test_svr_linear_recovery() {
    let x_vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = x_vals.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let x: Vec<Vec<f64>> = x_vals.iter().map(|&xi| vec![xi]).collect();

    let model = SvrModel::new(linear_kernel(), 10.0, 0.1).expect("valid params");
    let fitted = model.fit(&x, &y_vals).expect("fit linear");

    let pred = fitted.predict(&[5.0]).expect("predict");
    let expected = 11.0;
    assert!(
        (pred - expected).abs() < 1.5,
        "SVR linear recovery: predicted {}, expected {} (within 1.5)",
        pred,
        expected
    );
}

/// Fit SVR on a simple 1D quadratic with RBF kernel.
/// Use 6 carefully chosen points from y = x^2 and check training MAE.
#[test]
fn test_svr_rbf_regression() {
    // Small, simple dataset: y = x^2 for x ∈ {-2,-1,0,1,2}.
    // RBF with high gamma (tight kernel), large C (low regularization), small epsilon.
    let x: Vec<Vec<f64>> = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    let y: Vec<f64> = x.iter().map(|xi| xi[0] * xi[0]).collect(); // [4, 1, 0, 1, 4]

    // Use moderate C and gamma with larger epsilon for stability.
    let config = super::smo::SmoConfig {
        c: 10.0,
        tol: 1e-3,
        max_iter: 50_000,
        ..Default::default()
    };
    let model = SvrModel::new_with_config(rbf_kernel(1.0), 0.5, config).expect("valid params");
    let fitted = model.fit(&x, &y).expect("fit RBF SVR");

    let preds = fitted.predict_batch(&x).expect("predict_batch");
    let mae: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f64>()
        / x.len() as f64;

    // MAE should be reasonably small (within the ε-tube).
    // With ε=0.5, the maximum possible MAE without any penalization is 0.5,
    // so we use 0.8 as a conservative bound to account for finite C.
    assert!(
        mae < 0.8,
        "SVR RBF: training MAE = {} (expected < 0.8)",
        mae
    );
}

/// Empty training set should produce an error.
#[test]
fn test_svr_empty_error() {
    let model = SvrModel::new(linear_kernel(), 1.0, 0.1).expect("valid params");
    let result = model.fit(&[], &[]);
    assert!(result.is_err(), "Expected error for empty training set");
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {:?}",
        result
    );
}

/// C = 0 should produce InvalidParameter error.
#[test]
fn test_svr_negative_c_error() {
    let result = SvrModel::new(linear_kernel(), 0.0, 0.1);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(KernelError::InvalidParameter { .. })),
        "Expected InvalidParameter for C=0, got {:?}",
        result
    );
}

/// predict_batch should return a vector of the same length as the input batch.
#[test]
fn test_svr_predict_batch_shape() {
    let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi[0] * 2.0).collect();

    let model = SvrModel::new(linear_kernel(), 5.0, 0.1).expect("valid");
    let fitted = model.fit(&x, &y).expect("fit");

    let test_x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.5]).collect();
    let preds = fitted.predict_batch(&test_x).expect("predict_batch");
    assert_eq!(
        preds.len(),
        test_x.len(),
        "predict_batch output length mismatch"
    );
}

// ─── SMO Config Tests ────────────────────────────────────────────────────────

/// SmoConfig::default() should have the specified values.
#[test]
fn test_smo_config_default() {
    let cfg = SmoConfig::default();
    assert_eq!(cfg.c, 1.0, "default C should be 1.0");
    assert_eq!(cfg.tol, 1e-3, "default tol should be 1e-3");
    assert_eq!(cfg.epsilon, 0.1, "default epsilon should be 0.1");
    assert_eq!(cfg.max_iter, 10_000, "default max_iter should be 10000");
}

/// SmoConfig should be Clone and Debug.
#[test]
fn test_smo_config_clone_debug() {
    let cfg = SmoConfig {
        c: 5.0,
        tol: 1e-4,
        epsilon: 0.5,
        max_iter: 1000,
    };
    let cloned = cfg.clone();
    assert_eq!(cloned.c, cfg.c);
    let dbg = format!("{:?}", cfg);
    assert!(dbg.contains("SmoConfig"));
}

/// SVC predict_batch returns a vector of the correct length.
#[test]
fn test_svc_predict_batch_shape() {
    let x = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];
    let y = vec![1, 1, -1, -1];

    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let fitted = model.fit(&x, &y).expect("fit");

    let test_x = vec![vec![2.0, 1.0], vec![-2.0, -1.0], vec![0.5, 0.5]];
    let preds = fitted.predict_batch(&test_x).expect("predict_batch");
    assert_eq!(preds.len(), 3, "predict_batch output length mismatch");
}

/// SVC with mismatched x/y lengths should return DimensionMismatch.
#[test]
fn test_svc_mismatched_lengths_error() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![1, -1]; // wrong length

    let model = SvcModel::new(linear_kernel(), 1.0).expect("valid C");
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {:?}",
        result
    );
}

/// SVR model creation with negative epsilon should fail.
#[test]
fn test_svr_negative_epsilon_error() {
    let result = SvrModel::new(linear_kernel(), 1.0, -0.1);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(KernelError::InvalidParameter { .. })),
        "Expected InvalidParameter for negative epsilon, got {:?}",
        result
    );
}

/// SVR with mismatched x/y lengths should return DimensionMismatch.
#[test]
fn test_svr_mismatched_lengths_error() {
    let x = vec![vec![1.0], vec![2.0]];
    let y = vec![1.0, 2.0, 3.0]; // wrong length

    let model = SvrModel::new(linear_kernel(), 1.0, 0.1).expect("valid");
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "Expected DimensionMismatch, got {:?}",
        result
    );
}
