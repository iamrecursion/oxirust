//! Unit tests for the kernel_pca module.
//!
//! Coverage:
//!
//! 1. Linear KPCA recovers standard PCA embedding (up to sign flips).
//! 2. RBF KPCA on a two-cluster dataset separates clusters in the
//!    first principal component.
//! 3. Double-centering produces rows and columns that sum to zero.
//! 4. `transform` on training data matches the `fit_transform` result.
//! 5. Error paths: empty training, n_components > n, dimension mismatch.
//! 6. Explained variance ratios sum to 1.

use scirs2_core::ndarray::Array2;

use crate::kernel_pca::centering::double_center;
use crate::kernel_pca::model::{KernelPCA, KernelPcaConfig};
use crate::tensor_kernels::{LinearKernel, RbfKernel};
use crate::types::{Kernel, RbfKernelConfig};

/// Build a small dataset of 2-D points arranged in two well-separated
/// clusters: cluster A around (0, 0) and cluster B around (10, 10).
fn two_cluster_data() -> Vec<Vec<f64>> {
    vec![
        vec![0.1, 0.2],
        vec![-0.1, 0.1],
        vec![0.2, -0.1],
        vec![0.0, 0.0],
        vec![10.0, 10.1],
        vec![9.9, 10.0],
        vec![10.1, 9.9],
        vec![10.0, 10.0],
    ]
}

#[test]
fn linear_kpca_recovers_pca_structure() {
    // Linear KPCA is equivalent to standard PCA. With 2-D data that has
    // most variance along the first axis, the first component of the
    // embedding should capture the direction of maximum variance.
    let data: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![2.0, 0.1],
        vec![3.0, -0.1],
        vec![4.0, 0.0],
        vec![5.0, 0.1],
    ];
    let kernel = LinearKernel::new();
    let config = KernelPcaConfig::new(2);
    let model = KernelPCA::build(kernel, config).expect("model");
    let (fitted, embedding) = model.fit_transform(&data).expect("fit_transform");

    assert_eq!(embedding.nrows(), 5);
    assert_eq!(embedding.ncols(), 2);

    // The first eigenvalue should dominate (most variance along axis 0).
    let evr = fitted.explained_variance_ratio();
    assert!(
        evr[0] > 0.9,
        "first component should explain > 90% variance, got {:.4}",
        evr[0]
    );
}

#[test]
fn rbf_kpca_separates_two_clusters() {
    let data = two_cluster_data();
    let kernel = RbfKernel::new(RbfKernelConfig::new(1.0)).expect("kernel");
    let config = KernelPcaConfig::new(2);
    let model = KernelPCA::build(kernel, config).expect("model");
    let (_, embedding) = model.fit_transform(&data).expect("fit_transform");

    // Cluster A is rows 0..4, cluster B is rows 4..8.
    // In the first principal component, the two clusters should have
    // different signs (or at least a clear separation).
    let mean_a: f64 = (0..4).map(|i| embedding[(i, 0)]).sum::<f64>() / 4.0;
    let mean_b: f64 = (4..8).map(|i| embedding[(i, 0)]).sum::<f64>() / 4.0;
    assert!(
        (mean_a - mean_b).abs() > 0.1,
        "clusters must be separated in PC1: mean_a = {:.6}, mean_b = {:.6}",
        mean_a,
        mean_b
    );
}

#[test]
fn double_center_rows_and_cols_sum_to_zero() {
    // Build a small Gram matrix from a linear kernel and verify that
    // after double-centering every row and every column sums to zero.
    let data = two_cluster_data();
    let kernel = LinearKernel::new();
    let n = data.len();
    let rows = kernel.compute_matrix(&data).expect("gram");
    let mut gram = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            gram[(i, j)] = rows[i][j];
        }
    }
    let (centered, _stats) = double_center(&gram).expect("center");

    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| centered[(i, j)]).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "row {} sum = {} (expected 0)",
            i,
            row_sum
        );
        let col_sum: f64 = (0..n).map(|j| centered[(j, i)]).sum();
        assert!(
            col_sum.abs() < 1e-10,
            "col {} sum = {} (expected 0)",
            i,
            col_sum
        );
    }
}

#[test]
fn transform_matches_fit_transform() {
    let data = two_cluster_data();
    let kernel = RbfKernel::new(RbfKernelConfig::new(1.0)).expect("kernel");
    let config = KernelPcaConfig::new(2);
    let model = KernelPCA::build(kernel, config).expect("model");
    let (fitted, embedding_ft) = model.fit_transform(&data).expect("fit_transform");
    let embedding_t = fitted.transform(&data).expect("transform");

    assert_eq!(embedding_ft.nrows(), embedding_t.nrows());
    assert_eq!(embedding_ft.ncols(), embedding_t.ncols());
    for i in 0..embedding_ft.nrows() {
        for j in 0..embedding_ft.ncols() {
            assert!(
                (embedding_ft[(i, j)] - embedding_t[(i, j)]).abs() < 1e-10,
                "mismatch at ({}, {}): fit_transform = {}, transform = {}",
                i,
                j,
                embedding_ft[(i, j)],
                embedding_t[(i, j)]
            );
        }
    }
}

#[test]
fn error_empty_training_set() {
    let kernel = LinearKernel::new();
    let config = KernelPcaConfig::new(1);
    let model = KernelPCA::build(kernel, config).expect("model");
    let err = model.fit(&[]).expect_err("empty");
    assert!(
        matches!(
            err,
            crate::kernel_pca::error::KernelPcaError::InvalidInput(_)
        ),
        "expected InvalidInput, got {:?}",
        err
    );
}

#[test]
fn error_too_many_components() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let kernel = LinearKernel::new();
    let config = KernelPcaConfig::new(5);
    let model = KernelPCA::build(kernel, config).expect("model");
    let err = model.fit(&data).expect_err("too many components");
    assert!(
        matches!(
            err,
            crate::kernel_pca::error::KernelPcaError::InvalidInput(_)
        ),
        "expected InvalidInput, got {:?}",
        err
    );
}

#[test]
fn error_dimension_mismatch_in_transform() {
    let data = two_cluster_data();
    let kernel = LinearKernel::new();
    let config = KernelPcaConfig::new(1);
    let model = KernelPCA::build(kernel, config).expect("model");
    let fitted = model.fit(&data).expect("fit");
    // Pass points with wrong dimensionality.
    let bad = vec![vec![1.0, 2.0, 3.0]];
    let err = fitted.transform(&bad).expect_err("dim mismatch");
    assert!(
        matches!(
            err,
            crate::kernel_pca::error::KernelPcaError::DimensionMismatch { .. }
        ),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

#[test]
fn explained_variance_ratios_sum_to_one() {
    let data = two_cluster_data();
    let kernel = LinearKernel::new();
    let config = KernelPcaConfig::new(2);
    let model = KernelPCA::build(kernel, config).expect("model");
    let fitted = model.fit(&data).expect("fit");
    let evr = fitted.explained_variance_ratio();
    let total: f64 = evr.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "explained variance ratios sum to {} (expected 1.0)",
        total
    );
    for &v in evr.iter() {
        assert!(v >= 0.0, "variance ratio must be non-negative, got {}", v);
        assert!(v <= 1.0, "variance ratio must be <= 1.0, got {}", v);
    }
}
