//! Unit tests for the `multi_output` module.

use std::sync::Arc;

use crate::multitask::{ICMKernel, LMCKernel};
use crate::tensor_kernels::RbfKernel;
use crate::types::RbfKernelConfig;

use super::icm::KroneckerICMKernel;
use super::lmc::KroneckerLMCKernel;
use super::trait_def::MultiOutputKernel;
use super::vvgp::VvgpModel;

/// Build a 2-task RBF-ICM kernel with covariance [[2.0, 0.5], [0.5, 1.5]].
fn make_icm_2task() -> KroneckerICMKernel {
    let base = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid RBF gamma"));
    let cov = vec![vec![2.0, 0.5], vec![0.5, 1.5]];
    KroneckerICMKernel::from_base(base, cov).expect("valid ICM kernel")
}

/// Build a single-component 2-task RBF-LMC kernel with the same covariance.
fn make_lmc_2task() -> KroneckerLMCKernel {
    let base = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid RBF gamma"));
    let cov = vec![vec![2.0, 0.5], vec![0.5, 1.5]];
    let mut lmc = LMCKernel::new(2);
    lmc.add_component(base, cov).expect("valid component");
    KroneckerLMCKernel::new(lmc)
}

// ─── ICM block properties ────────────────────────────────────────────────────

#[test]
fn icm_block_symmetric_psd() {
    let kernel = make_icm_2task();
    let x = &[0.0_f64];
    let block = kernel.compute_block(x, x).expect("compute_block");

    // 2×2 block must be symmetric.
    assert!(
        (block[[0, 1]] - block[[1, 0]]).abs() < 1e-12,
        "block must be symmetric: [[{}, {}], [{}, {}]]",
        block[[0, 0]],
        block[[0, 1]],
        block[[1, 0]],
        block[[1, 1]]
    );

    // PSD check for 2×2: positive diagonal and positive determinant.
    let det = block[[0, 0]] * block[[1, 1]] - block[[0, 1]] * block[[1, 0]];
    assert!(
        block[[0, 0]] > 0.0,
        "diagonal block[0,0]={} must be positive",
        block[[0, 0]]
    );
    assert!(
        block[[1, 1]] > 0.0,
        "diagonal block[1,1]={} must be positive",
        block[[1, 1]]
    );
    assert!(det > 0.0, "determinant={} must be positive (PSD)", det);
}

#[test]
fn icm_block_gram_shape() {
    let kernel = make_icm_2task();
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
    let gram = kernel
        .block_gram_matrix(&inputs)
        .expect("block_gram_matrix");

    // N=3 inputs, p=2 outputs => shape (6, 6).
    assert_eq!(gram.shape(), &[6, 6]);
}

#[test]
fn icm_block_gram_symmetric() {
    let kernel = make_icm_2task();
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
    let gram = kernel
        .block_gram_matrix(&inputs)
        .expect("block_gram_matrix");

    let n = gram.shape()[0];
    for i in 0..n {
        for j in 0..n {
            assert!(
                (gram[[i, j]] - gram[[j, i]]).abs() < 1e-12,
                "gram[{},{}]={} != gram[{},{}]={} (not symmetric)",
                i,
                j,
                gram[[i, j]],
                j,
                i,
                gram[[j, i]]
            );
        }
    }
}

#[test]
fn icm_n_outputs() {
    let kernel = make_icm_2task();
    assert_eq!(kernel.n_outputs(), 2);
}

// ─── LMC matches ICM for single-component ───────────────────────────────────

#[test]
fn lmc_block_matches_icm_single_component() {
    let icm = make_icm_2task();
    let lmc = make_lmc_2task();

    let inputs_pairs: &[(&[f64], &[f64])] = &[
        (&[0.0], &[0.0]),
        (&[0.0], &[1.0]),
        (&[1.0], &[2.5]),
        (&[-1.0], &[1.0]),
    ];

    for (x, y) in inputs_pairs {
        let block_icm = icm.compute_block(x, y).expect("ICM compute_block");
        let block_lmc = lmc.compute_block(x, y).expect("LMC compute_block");
        for ri in 0..2 {
            for ci in 0..2 {
                assert!(
                    (block_icm[[ri, ci]] - block_lmc[[ri, ci]]).abs() < 1e-10,
                    "ICM[{},{}]={} != LMC[{},{}]={} for inputs {:?}, {:?}",
                    ri,
                    ci,
                    block_icm[[ri, ci]],
                    ri,
                    ci,
                    block_lmc[[ri, ci]],
                    x,
                    y
                );
            }
        }
    }
}

// ─── VvgpModel fitting and prediction ─────────────────────────────────────

#[test]
fn vvgp_posterior_mean_recovers_training_targets() {
    // Three 1-D training points with 2 outputs each.
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
    let targets: Vec<Vec<f64>> = vec![vec![1.0, -1.0], vec![0.5, 0.5], vec![-1.0, 2.0]];

    let kernel = Arc::new(make_icm_2task());
    // Use very small noise so the posterior mean closely interpolates targets.
    let model = VvgpModel::new(kernel, 1e-6).expect("valid VvgpModel");
    let fitted = model.fit(&inputs, &targets).expect("fit");

    for (inp, target) in inputs.iter().zip(&targets) {
        let (mean, _cov) = fitted.predict(inp).expect("predict");
        assert_eq!(mean.len(), 2, "mean length must equal n_outputs");
        for p_idx in 0..2 {
            assert!(
                (mean[p_idx] - target[p_idx]).abs() < 0.1,
                "mean[{}]={} should be close to target[{}]={} (tol=0.1)",
                p_idx,
                mean[p_idx],
                p_idx,
                target[p_idx]
            );
        }
    }
}

#[test]
fn vvgp_predict_returns_correct_shapes() {
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0]];
    let targets: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let kernel = Arc::new(make_icm_2task());
    let model = VvgpModel::new(kernel, 1e-4).expect("valid VvgpModel");
    let fitted = model.fit(&inputs, &targets).expect("fit");

    let x_test = vec![0.5_f64];
    let (mean, cov) = fitted.predict(&x_test).expect("predict");

    assert_eq!(mean.len(), 2, "mean must have length p=2");
    assert_eq!(cov.shape(), &[2, 2], "covariance must be p×p");
}

#[test]
fn vvgp_covariance_diagonal_non_negative() {
    let inputs: Vec<Vec<f64>> = vec![vec![0.0], vec![2.0]];
    let targets: Vec<Vec<f64>> = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];

    let kernel = Arc::new(make_icm_2task());
    let model = VvgpModel::new(kernel, 1e-4).expect("valid VvgpModel");
    let fitted = model.fit(&inputs, &targets).expect("fit");

    // Predict at a test point not in training set.
    let x_test = vec![1.0_f64];
    let (_mean, cov) = fitted.predict(&x_test).expect("predict");
    for p_idx in 0..2 {
        assert!(
            cov[[p_idx, p_idx]] >= -1e-10,
            "posterior variance cov[{0},{0}]={1} must be non-negative",
            p_idx,
            cov[[p_idx, p_idx]]
        );
    }
}

#[test]
fn vvgp_invalid_noise_rejected() {
    let kernel = Arc::new(make_icm_2task());
    assert!(
        VvgpModel::new(kernel, -1.0).is_err(),
        "negative noise must be rejected"
    );
}

#[test]
fn vvgp_mismatched_targets_rejected() {
    let kernel = Arc::new(make_icm_2task());
    let model = VvgpModel::new(kernel, 1e-4).expect("valid VvgpModel");
    // 2 inputs but 3 target vectors.
    let inputs = vec![vec![0.0], vec![1.0]];
    let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    assert!(
        model.fit(&inputs, &targets).is_err(),
        "mismatched target count must be rejected"
    );
}

// ─── Kernel name ─────────────────────────────────────────────────────────────

#[test]
fn kernel_names() {
    let icm = make_icm_2task();
    let lmc = make_lmc_2task();
    assert_eq!(icm.name(), "KroneckerICM");
    assert_eq!(lmc.name(), "KroneckerLMC");
}

// ─── ICMKernel constructor ───────────────────────────────────────────────────

#[test]
fn icm_from_existing_icm_kernel() {
    let base = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid"));
    let cov = vec![vec![1.0, 0.3], vec![0.3, 1.0]];
    let icm_inner = ICMKernel::new(base, cov).expect("valid ICMKernel");
    let kernel = KroneckerICMKernel::new(icm_inner);
    assert_eq!(kernel.n_outputs(), 2);
}
