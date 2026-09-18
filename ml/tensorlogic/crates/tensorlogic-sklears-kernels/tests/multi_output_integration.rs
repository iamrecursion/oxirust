//! Integration tests for multi-output / vector-valued kernels.
//!
//! These tests exercise the full pipeline:
//! 1. Construct a `KroneckerICMKernel` from a base kernel and task covariance.
//! 2. Fit a `VvgpModel` to synthetic training data.
//! 3. Predict at training and held-out points.
//! 4. Verify shapes, posterior-mean accuracy, and covariance PSD-ness.

use std::sync::Arc;

use tensorlogic_sklears_kernels::{
    multi_output::{KroneckerICMKernel, KroneckerLMCKernel, MultiOutputKernel, VvgpModel},
    multitask::LMCKernel,
    ICMKernel, RbfKernel, RbfKernelConfig,
};

// ─── helper ─────────────────────────────────────────────────────────────────

fn rbf_icm_2output() -> KroneckerICMKernel {
    let base = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid gamma"));
    let covariance = vec![vec![2.0, 0.7], vec![0.7, 1.5]];
    KroneckerICMKernel::from_base(base, covariance).expect("valid ICM kernel")
}

// ─── end-to-end VvgpModel ────────────────────────────────────────────────────

/// Full end-to-end test: fit a 2-output VVGP on (sin, cos) targets and verify
/// posterior mean accuracy at training points.
#[test]
fn vvgp_icm_end_to_end() {
    // 5 1-D training inputs with 2 outputs each.
    let inputs: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| vec![x[0].sin(), x[0].cos()])
        .collect();

    let model = VvgpModel::new(Arc::new(rbf_icm_2output()), 1e-4).expect("valid noise");
    let fitted = model.fit(&inputs, &targets).expect("fit succeeded");

    // At training points the posterior mean must approximate the targets.
    for (input, target) in inputs.iter().zip(&targets) {
        let (mean, cov) = fitted.predict(input).expect("predict at training point");

        assert_eq!(mean.len(), 2, "mean length must equal n_outputs=2");
        assert_eq!(cov.shape(), &[2, 2], "covariance shape must be (p, p)");

        for p_idx in 0..2 {
            assert!(
                (mean[p_idx] - target[p_idx]).abs() < 0.2,
                "posterior mean[{p_idx}]={:.6} should be near target={:.6} (tol=0.2)",
                mean[p_idx],
                target[p_idx]
            );
            // Diagonal variance must be non-negative.
            assert!(
                cov[[p_idx, p_idx]] >= -1e-10,
                "posterior variance cov[{p},{p}]={v:.6e} must be non-negative",
                p = p_idx,
                v = cov[[p_idx, p_idx]]
            );
        }
    }
}

/// Verify shape consistency for held-out test points (never seen during fit).
#[test]
fn vvgp_icm_predict_at_held_out_point() {
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
    let targets: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.84, 0.54], vec![0.91, -0.42]];

    let model = VvgpModel::new(Arc::new(rbf_icm_2output()), 1e-4).expect("valid noise");
    let fitted = model.fit(&inputs, &targets).expect("fit");

    let x_test = vec![1.5_f64];
    let (mean, cov) = fitted.predict(&x_test).expect("predict held-out");

    assert_eq!(mean.len(), 2);
    assert_eq!(cov.shape(), &[2, 2]);
}

// ─── block Gram matrix properties ───────────────────────────────────────────

/// Block Gram matrix returned by `KroneckerICMKernel` must be symmetric and
/// have the correct shape `(N·p × N·p)`.
#[test]
fn icm_block_gram_symmetric_and_correct_shape() {
    let kernel = rbf_icm_2output();
    let inputs: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
    let gram = kernel.block_gram_matrix(&inputs).expect("block gram");

    let n = inputs.len();
    let p = kernel.n_outputs();
    assert_eq!(gram.shape(), &[n * p, n * p]);

    let size = n * p;
    for i in 0..size {
        for j in 0..size {
            assert!(
                (gram[[i, j]] - gram[[j, i]]).abs() < 1e-12,
                "gram[{i},{j}]={a} != gram[{j},{i}]={b} (not symmetric)",
                a = gram[[i, j]],
                b = gram[[j, i]]
            );
        }
    }
}

// ─── LMC single-component agrees with ICM ───────────────────────────────────

/// A single-component LMC kernel with the same covariance and base kernel as
/// an ICM kernel must produce identical blocks.
#[test]
fn lmc_single_component_matches_icm() {
    let base_icm = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid"));
    let base_lmc = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid"));
    let cov = vec![vec![2.0, 0.7], vec![0.7, 1.5]];

    let icm = KroneckerICMKernel::from_base(base_icm, cov.clone()).expect("valid ICM");
    let mut lmc_inner = LMCKernel::new(2);
    lmc_inner
        .add_component(base_lmc, cov)
        .expect("add component");
    let lmc = KroneckerLMCKernel::new(lmc_inner);

    let test_pairs: &[(&[f64], &[f64])] = &[(&[0.0], &[0.0]), (&[0.0], &[1.0]), (&[1.5], &[0.5])];

    for (x, y) in test_pairs {
        let b_icm = icm.compute_block(x, y).expect("ICM block");
        let b_lmc = lmc.compute_block(x, y).expect("LMC block");
        for ri in 0..2 {
            for ci in 0..2 {
                assert!(
                    (b_icm[[ri, ci]] - b_lmc[[ri, ci]]).abs() < 1e-10,
                    "ICM[{ri},{ci}]={a} != LMC[{ri},{ci}]={b} for x={x:?}, y={y:?}",
                    a = b_icm[[ri, ci]],
                    b = b_lmc[[ri, ci]]
                );
            }
        }
    }
}

// ─── VvgpModel metadata accessors ───────────────────────────────────────────

#[test]
fn fitted_metadata_accessors() {
    let inputs = vec![vec![0.0_f64], vec![1.0]];
    let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let model = VvgpModel::new(Arc::new(rbf_icm_2output()), 1e-3).expect("valid");
    let fitted = model.fit(&inputs, &targets).expect("fit");

    assert_eq!(fitted.n_train(), 2);
    assert_eq!(fitted.n_outputs(), 2);
    assert!((fitted.noise() - 1e-3).abs() < 1e-15);
}

// ─── Error handling ──────────────────────────────────────────────────────────

#[test]
fn vvgp_negative_noise_is_rejected() {
    let kernel = Arc::new(rbf_icm_2output());
    assert!(
        VvgpModel::new(kernel, -0.01).is_err(),
        "negative noise must return Err"
    );
}

#[test]
fn vvgp_fit_rejects_mismatched_target_count() {
    let inputs = vec![vec![0.0_f64], vec![1.0]];
    let targets = vec![vec![1.0, 0.0]]; // Only 1 target for 2 inputs.
    let model = VvgpModel::new(Arc::new(rbf_icm_2output()), 1e-4).expect("valid");
    assert!(model.fit(&inputs, &targets).is_err());
}

#[test]
fn vvgp_fit_rejects_wrong_target_dimension() {
    let inputs = vec![vec![0.0_f64], vec![1.0]];
    // Each target must have length p=2 but we provide length 3.
    let targets = vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0, 0.5]];
    let model = VvgpModel::new(Arc::new(rbf_icm_2output()), 1e-4).expect("valid");
    assert!(model.fit(&inputs, &targets).is_err());
}

// ─── ICMKernel::new constructor path ────────────────────────────────────────

#[test]
fn kronecker_icm_from_icm_kernel() {
    let base = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid"));
    let cov = vec![vec![1.0, 0.3], vec![0.3, 1.0]];
    let icm_inner = ICMKernel::new(base, cov).expect("valid ICMKernel");
    let kernel = KroneckerICMKernel::new(icm_inner);
    assert_eq!(kernel.n_outputs(), 2);

    let block = kernel
        .compute_block(&[0.0_f64], &[0.0_f64])
        .expect("block at same point");
    // K(x, x) = k(x, x) * B = 1.0 * B for RBF at distance 0.
    assert!((block[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((block[[1, 1]] - 1.0).abs() < 1e-10);
    assert!((block[[0, 1]] - 0.3).abs() < 1e-10);
}
