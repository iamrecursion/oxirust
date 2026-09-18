//! Integration test: Kernel PCA on a Swiss-roll-like dataset.
//!
//! We generate a 3-D "Swiss roll" using a deterministic parameter sweep
//! (no randomness needed), embed it into 2 components with an RBF
//! kernel, and verify:
//!
//! 1. The embedding has the expected shape `(n, 2)`.
//! 2. The two retained eigenvalues are strictly positive.
//! 3. Cumulative explained-variance ratio is >= 0.70 (RBF on a
//!    well-spread Swiss roll captures the dominant nonlinear structure).
//! 4. Out-of-sample projection is dimensionally consistent with the
//!    fitted model.

use tensorlogic_sklears_kernels::kernel_pca::{KernelPCA, KernelPcaConfig};
use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};

/// Generate a deterministic Swiss-roll dataset: `n` points in R^3
/// parameterised by an angle `t` that sweeps from `t_min` to `t_max`.
///
/// x = t cos(t), y = height, z = t sin(t), where `height` is linearly
/// spaced in `[0, h_max]`.
fn swiss_roll(n: usize, t_min: f64, t_max: f64, h_max: f64) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f64 / (n - 1).max(1) as f64;
        let t = t_min + frac * (t_max - t_min);
        let h = frac * h_max;
        data.push(vec![t * t.cos(), h, t * t.sin()]);
    }
    data
}

#[test]
fn swiss_roll_rbf_kpca_2_components() {
    let n = 40;
    let data = swiss_roll(
        n,
        1.5 * std::f64::consts::PI,
        4.5 * std::f64::consts::PI,
        10.0,
    );
    assert_eq!(data.len(), n);
    assert_eq!(data[0].len(), 3);

    // RBF gamma chosen to match the typical inter-point distances on
    // the roll (~5-30 in Euclidean space).
    let kernel = RbfKernel::new(RbfKernelConfig::new(0.01)).expect("kernel");
    let config = KernelPcaConfig::new(2);
    let model = KernelPCA::build(kernel, config).expect("model");
    let (fitted, embedding) = model.fit_transform(&data).expect("fit_transform");

    // Shape check.
    assert_eq!(embedding.nrows(), n, "embedding should have {} rows", n);
    assert_eq!(embedding.ncols(), 2, "embedding should have 2 columns");

    // Eigenvalues must be positive.
    let evs = fitted.eigenvalues();
    for (i, &v) in evs.iter().enumerate() {
        assert!(v > 0.0, "eigenvalue[{}] = {} must be positive", i, v);
    }

    // Cumulative explained variance >= 0.70.
    let evr = fitted.explained_variance_ratio();
    let cumulative: f64 = evr.iter().sum();
    assert!(
        cumulative >= 0.70,
        "cumulative explained variance = {:.4} (expected >= 0.70)",
        cumulative
    );

    // Out-of-sample projection: take a few new points and project them.
    let new_points = swiss_roll(
        5,
        2.0 * std::f64::consts::PI,
        3.0 * std::f64::consts::PI,
        5.0,
    );
    let projected = fitted.transform(&new_points).expect("transform new");
    assert_eq!(projected.nrows(), 5);
    assert_eq!(projected.ncols(), 2);
    // Values should be finite.
    for i in 0..projected.nrows() {
        for j in 0..projected.ncols() {
            assert!(
                projected[(i, j)].is_finite(),
                "projection[{}, {}] = {} is not finite",
                i,
                j,
                projected[(i, j)]
            );
        }
    }
}

#[test]
fn kpca_linear_on_collinear_data_first_component_dominates() {
    // 1-D manifold embedded in 3-D: all points lie on a line.
    // Linear KPCA should put ~100% variance into the first component.
    let data: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let t = i as f64;
            vec![t, 2.0 * t, -t]
        })
        .collect();
    let kernel = tensorlogic_sklears_kernels::LinearKernel::new();
    // Collinear data is rank 1 in feature space (after centering), so
    // we request only 1 component. The single retained eigenvalue must
    // account for essentially all variance.
    let config = KernelPcaConfig::new(1);
    let model = KernelPCA::build(kernel, config).expect("model");
    let fitted = model.fit(&data).expect("fit");
    let evr = fitted.explained_variance_ratio();
    assert!(
        evr[0] > 0.99,
        "first component should explain > 99% for collinear data, got {:.6}",
        evr[0]
    );
}
