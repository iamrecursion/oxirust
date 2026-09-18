//! Integration test: build a Deep Kernel with a 2-layer MLP feature
//! extractor on top of an RBF base, evaluate on a small synthetic
//! dataset, and assert:
//!
//! 1. The Gram matrix is exactly symmetric (DKL inherits symmetry from
//!    `K_RBF(g(x), g(y)) == K_RBF(g(y), g(x))`).
//! 2. Every diagonal entry equals 1 (`K_RBF(g(x), g(x)) = exp(0) = 1`).
//! 3. The Gram matrix is positive semi-definite — verified via Cholesky
//!    decomposition of `G + εI` for a small ε used to absorb numerical
//!    jitter.

use tensorlogic_sklears_kernels::{
    deep_kernel::{Activation, DeepKernelBuilder},
    Kernel, RbfKernel, RbfKernelConfig,
};

#[allow(clippy::needless_range_loop)]
fn is_symmetric(g: &[Vec<f64>], tol: f64) -> bool {
    let n = g.len();
    for i in 0..n {
        if g[i].len() != n {
            return false;
        }
        for j in 0..n {
            if (g[i][j] - g[j][i]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Cholesky decomposition of a symmetric matrix. Returns `Err` if the
/// matrix is not PSD (i.e. a leading principal minor is non-positive).
#[allow(clippy::needless_range_loop)]
fn cholesky(g: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = g.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = g[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "not PSD: leading minor ({}) ≤ 0 at index {}",
                        sum, i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

#[test]
fn deep_kernel_gram_is_symmetric_and_psd() {
    // Small 2-D synthetic dataset with two weakly separated clusters.
    let xs: Vec<Vec<f64>> = vec![
        vec![-1.0, -1.0],
        vec![-0.8, -0.9],
        vec![-1.1, -0.7],
        vec![0.9, 1.0],
        vec![1.1, 0.8],
        vec![1.0, 0.9],
    ];

    let rbf = RbfKernel::new(RbfKernelConfig::new(0.75)).expect("valid gamma");
    let dkl = DeepKernelBuilder::new()
        .input_dim(2)
        .hidden_layer(4, Activation::Tanh)
        .output_dim(3, Activation::Identity)
        .seed(0x1337)
        .build(rbf)
        .expect("valid topology");

    let gram = dkl
        .compute_symmetric_gram(&xs)
        .expect("symmetric gram succeeds");

    // 1. Exact symmetry — the helper writes G[i][j] = G[j][i] from the
    // same base-kernel evaluation, so tolerance is numerical roundoff.
    assert!(
        is_symmetric(&gram, 1e-12),
        "DKL Gram must be symmetric: {:?}",
        gram
    );

    // 2. Every diagonal entry must be exactly 1 (RBF at distance 0).
    for (i, row) in gram.iter().enumerate() {
        let diag = row[i];
        assert!(
            (diag - 1.0).abs() < 1e-12,
            "diagonal {} should be 1, got {}",
            i,
            diag
        );
    }

    // 3. PSD-ness via Cholesky. RBF kernels are strictly positive on
    // distinct inputs, but we still add a tiny ridge to keep the test
    // robust against near-zero eigenvalues from the nonlinear feature
    // map collapsing close inputs.
    let ridge = 1e-9;
    let mut ridged = gram.clone();
    for (i, row) in ridged.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let _l = cholesky(&ridged).expect("Gram + εI must be PSD");
}

#[test]
fn deep_kernel_matches_direct_composition() {
    // Sanity check: compute_symmetric_gram must agree with pairwise
    // evaluate calls on every pair.
    let xs: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![-1.0, -1.0]];
    let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid gamma");
    let dkl = DeepKernelBuilder::new()
        .input_dim(2)
        .hidden_layer(5, Activation::ReLU)
        .output_dim(2, Activation::Identity)
        .seed(99)
        .build(rbf)
        .expect("valid");

    let gram = dkl.compute_symmetric_gram(&xs).expect("gram");
    for (i, xi) in xs.iter().enumerate() {
        for (j, xj) in xs.iter().enumerate() {
            let direct = dkl.compute(xi, xj).expect("direct");
            assert!(
                (gram[i][j] - direct).abs() < 1e-12,
                "pair ({},{}) mismatch: gram={}, direct={}",
                i,
                j,
                gram[i][j],
                direct
            );
        }
    }
}
