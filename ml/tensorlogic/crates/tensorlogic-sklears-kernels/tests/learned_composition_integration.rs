//! Integration test: train a learned mixture over two RBF kernels and
//! assert that the loss decreases across ≥ 50 gradient-descent steps.
//!
//! Synthetic setup:
//!
//! * A small dataset `xs` of 1-D points drawn from two well-separated
//!   clusters.
//! * A target Gram matrix `G*` computed from an RBF kernel with γ = 2.0
//!   (tight bandwidth — acts as the "correct" kernel for this problem).
//! * A learned mixture over `RBF(γ=0.5)` and `RBF(γ=2.0)` with logits
//!   initialised to zero (uniform 50/50 start).
//! * Mean-squared-error loss
//!   `L(w) = mean_{i,j} (G_mix[i,j] - G*[i,j])^2`.
//! * Vanilla gradient descent on the logits for 80 steps.
//!
//! The test asserts:
//!
//! 1. Loss strictly decreased (L_final < L_initial).
//! 2. The weight on `RBF(γ=2.0)` grew (the mixture learned to prefer the
//!    target bandwidth).

use std::sync::Arc;

use tensorlogic_sklears_kernels::{
    learned_composition::LearnedMixtureBuilder, Kernel, RbfKernel, RbfKernelConfig,
    TrainableKernelMixture,
};

fn synthetic_points() -> Vec<Vec<f64>> {
    // Two clusters on the real line, separated by ~3 units.
    vec![
        vec![-1.5],
        vec![-1.2],
        vec![-0.9],
        vec![-0.7],
        vec![0.9],
        vec![1.3],
        vec![1.6],
        vec![1.9],
    ]
}

fn gram_matrix(kernel: &dyn Kernel, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = xs.len();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            g[i][j] = kernel
                .compute(&xs[i], &xs[j])
                .expect("compute must succeed on valid inputs");
        }
    }
    g
}

fn mse_and_residual(
    mixture: &tensorlogic_sklears_kernels::LearnedMixtureKernel,
    xs: &[Vec<f64>],
    target: &[Vec<f64>],
) -> (f64, Vec<Vec<f64>>) {
    let n = xs.len();
    let mut residuals = vec![vec![0.0; n]; n];
    let mut loss = 0.0;
    for i in 0..n {
        for j in 0..n {
            let k_mix = mixture
                .evaluate(&xs[i], &xs[j])
                .expect("compute must succeed on valid inputs");
            let r = k_mix - target[i][j];
            residuals[i][j] = r;
            loss += r * r;
        }
    }
    let denom = (n * n) as f64;
    (loss / denom, residuals)
}

#[test]
fn learned_mixture_converges_toward_target_rbf_bandwidth() {
    let xs = synthetic_points();
    let n = xs.len();

    // Ground-truth kernel (the one we want the mixture to learn).
    let target_kernel = RbfKernel::new(RbfKernelConfig::new(2.0)).expect("valid gamma");
    let target = gram_matrix(&target_kernel, &xs);

    let rbf_wide: Arc<dyn Kernel> =
        Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid gamma"));
    let rbf_tight: Arc<dyn Kernel> =
        Arc::new(RbfKernel::new(RbfKernelConfig::new(2.0)).expect("valid gamma"));

    let mixture = LearnedMixtureBuilder::new()
        .push_kernel(rbf_wide)
        .push_kernel(rbf_tight)
        .build()
        .expect("non-empty library");
    let mut trainable = TrainableKernelMixture::new(mixture);

    // Initial loss (50/50 mixture).
    let (initial_loss, _) = mse_and_residual(trainable.inner(), &xs, &target);
    let initial_weights = trainable.weights();
    assert!(
        (initial_weights[0] - 0.5).abs() < 1e-12 && (initial_weights[1] - 0.5).abs() < 1e-12,
        "uniform logits must yield uniform weights"
    );

    // Gradient descent loop on the logits. We use a moderate step count
    // and a step size tuned to the loss curvature (second derivative is
    // bounded by O(1) at this scale).
    let steps = 400;
    let learning_rate = 4.0;
    let denom = (n * n) as f64;
    let mut prev_loss = initial_loss;
    for step in 0..steps {
        // Accumulate dL/dw = 2/(n^2) * Σ_{i,j} (K_mix[i,j] - G*[i,j]) *
        // gradient_wrt_logits(x_i, x_j).
        let (loss, residuals) = mse_and_residual(trainable.inner(), &xs, &target);
        let mut grad = vec![0.0; trainable.num_parameters()];
        for i in 0..n {
            for j in 0..n {
                let local = trainable
                    .gradient(&xs[i], &xs[j])
                    .expect("gradient must succeed on valid inputs");
                let scale = 2.0 * residuals[i][j] / denom;
                for (g_acc, g_local) in grad.iter_mut().zip(local.iter()) {
                    *g_acc += scale * g_local;
                }
            }
        }
        trainable
            .step(&grad, learning_rate)
            .expect("sgd step must succeed");

        // Loss must be non-increasing in the limit (with this step size
        // and mixture size, it is strictly decreasing every iteration).
        if step > 0 {
            assert!(
                loss <= prev_loss + 1e-9,
                "loss increased at step {}: {} > {}",
                step,
                loss,
                prev_loss
            );
        }
        prev_loss = loss;
    }

    let (final_loss, _) = mse_and_residual(trainable.inner(), &xs, &target);
    assert!(
        final_loss < initial_loss,
        "learned mixture failed to reduce loss: initial {} vs final {}",
        initial_loss,
        final_loss
    );

    // The mixture should now lean toward the "tight" RBF (the target
    // bandwidth). Require a noticeable shift beyond the starting 0.5.
    let final_weights = trainable.weights();
    assert!(
        final_weights[1] > 0.7,
        "expected weight on target RBF > 0.7, got {:?}",
        final_weights
    );
    assert!(
        final_weights[0] + final_weights[1] > 0.999 && final_weights[0] + final_weights[1] < 1.001,
        "weights must sum to 1"
    );
}
