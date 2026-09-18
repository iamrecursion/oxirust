//! Unit tests for the learned mixture kernel.

use std::sync::Arc;

use crate::error::KernelError;
use crate::learned_composition::{
    builder::LearnedMixtureBuilder, mixture::LearnedMixtureKernel,
    trainable::TrainableKernelMixture,
};
use crate::types::{Kernel, RbfKernelConfig};
use crate::{LinearKernel, RbfKernel};

/// Helper: build a two-kernel mixture with explicit logits.
fn two_kernel_mixture(w0: f64, w1: f64) -> LearnedMixtureKernel {
    let linear: Arc<dyn Kernel> = Arc::new(LinearKernel::new());
    let rbf: Arc<dyn Kernel> =
        Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid gamma"));
    LearnedMixtureBuilder::new()
        .push_kernel_with_logit(linear, w0)
        .push_kernel_with_logit(rbf, w1)
        .build()
        .expect("non-empty library, matching logits")
}

#[test]
fn softmax_weights_sum_to_one() {
    let mixture = two_kernel_mixture(0.0, 0.0);
    let weights = mixture.weights();
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "weights sum {} != 1.0", sum);
    for w in &weights {
        assert!(*w > 0.0, "softmax weights must be strictly positive");
    }

    // Non-trivial logits.
    let mixture = two_kernel_mixture(1.5, -0.7);
    let sum: f64 = mixture.weights().iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn single_kernel_identity_case() {
    // A one-kernel library always yields weight 1.0 and reproduces the
    // base kernel exactly.
    let linear: Arc<dyn Kernel> = Arc::new(LinearKernel::new());
    let mixture = LearnedMixtureBuilder::new()
        .push_kernel(Arc::clone(&linear))
        .build()
        .expect("non-empty library");
    assert_eq!(mixture.num_kernels(), 1);
    let weights = mixture.weights();
    assert_eq!(weights.len(), 1);
    assert!((weights[0] - 1.0).abs() < 1e-12);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    let mix_val = mixture.evaluate(&x, &y).expect("evaluate");
    let base_val = linear.compute(&x, &y).expect("linear");
    assert!((mix_val - base_val).abs() < 1e-12);
}

#[test]
fn two_kernel_mixture_matches_hand_computation() {
    // logits [ln 2, 0] → weights [2/3, 1/3].
    let mixture = two_kernel_mixture((2.0f64).ln(), 0.0);
    let weights = mixture.weights();
    assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0]; // Same point ⇒ RBF = 1.0
    let linear_val = LinearKernel::new().compute(&x, &y).expect("linear");
    let rbf_val = RbfKernel::new(RbfKernelConfig::new(0.5))
        .expect("valid")
        .compute(&x, &y)
        .expect("rbf");
    let expected = (2.0 / 3.0) * linear_val + (1.0 / 3.0) * rbf_val;

    let mix_val = mixture.evaluate(&x, &y).expect("evaluate");
    assert!((mix_val - expected).abs() < 1e-10);
}

#[test]
fn gradient_matches_finite_difference() {
    // Three-kernel library, non-trivial logits, distinct inputs.
    let linear: Arc<dyn Kernel> = Arc::new(LinearKernel::new());
    let rbf1: Arc<dyn Kernel> =
        Arc::new(RbfKernel::new(RbfKernelConfig::new(0.25)).expect("valid"));
    let rbf2: Arc<dyn Kernel> = Arc::new(RbfKernel::new(RbfKernelConfig::new(1.5)).expect("valid"));
    let mixture = LearnedMixtureBuilder::new()
        .push_kernel_with_logit(linear, 0.3)
        .push_kernel_with_logit(rbf1, -0.4)
        .push_kernel_with_logit(rbf2, 0.9)
        .build()
        .expect("non-empty library");

    let x = vec![0.1, -0.4, 0.7];
    let y = vec![-0.2, 0.3, 0.5];

    let analytical = mixture
        .gradient_wrt_logits(&x, &y)
        .expect("analytical gradient");

    // Central finite differences: f(w + h e_i) vs f(w - h e_i).
    let h = 1e-5;
    let base_logits = mixture.logits().to_vec();
    for i in 0..base_logits.len() {
        let mut plus = base_logits.clone();
        plus[i] += h;
        let mut minus = base_logits.clone();
        minus[i] -= h;

        let m_plus = LearnedMixtureKernel::new(
            vec![
                Arc::new(LinearKernel::new()),
                Arc::new(RbfKernel::new(RbfKernelConfig::new(0.25)).expect("valid")),
                Arc::new(RbfKernel::new(RbfKernelConfig::new(1.5)).expect("valid")),
            ],
            plus,
        )
        .expect("valid");
        let m_minus = LearnedMixtureKernel::new(
            vec![
                Arc::new(LinearKernel::new()),
                Arc::new(RbfKernel::new(RbfKernelConfig::new(0.25)).expect("valid")),
                Arc::new(RbfKernel::new(RbfKernelConfig::new(1.5)).expect("valid")),
            ],
            minus,
        )
        .expect("valid");

        let f_plus = m_plus.evaluate(&x, &y).expect("eval +");
        let f_minus = m_minus.evaluate(&x, &y).expect("eval -");
        let numerical = (f_plus - f_minus) / (2.0 * h);
        let err = (analytical[i] - numerical).abs();
        assert!(
            err < 1e-4,
            "gradient mismatch at i={}: analytical={}, numerical={}, err={}",
            i,
            analytical[i],
            numerical,
            err
        );
    }
}

#[test]
fn empty_library_errors() {
    let err = LearnedMixtureBuilder::new().build().expect_err("must fail");
    match err {
        KernelError::InvalidParameter { parameter, .. } => {
            assert_eq!(parameter, "base_kernels");
        }
        other => panic!("expected InvalidParameter, got {:?}", other),
    }

    // Direct constructor also rejects empty libraries.
    let err = LearnedMixtureKernel::new(Vec::new(), Vec::new()).expect_err("must fail");
    matches!(err, KernelError::InvalidParameter { .. });
}

#[test]
fn softmax_weights_stay_positive_for_extreme_negative_logits() {
    // Mixture weights come from softmax — they are strictly positive even
    // for large negative logits. This is the contractual guarantee that
    // replaces the "no negative weights" check a weighted-sum API would
    // need.
    let mixture = two_kernel_mixture(-1_000.0, -1_000.0);
    let weights = mixture.weights();
    for w in &weights {
        assert!(
            *w > 0.0 && w.is_finite(),
            "extreme negative logits must still yield positive finite weights, got {}",
            w
        );
    }
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn set_logits_enforces_length_and_finiteness() {
    let mut mixture = two_kernel_mixture(0.0, 0.0);
    let err = mixture.set_logits(vec![1.0]).expect_err("length mismatch");
    matches!(err, KernelError::DimensionMismatch { .. });

    let err = mixture
        .set_logits(vec![f64::NAN, 0.0])
        .expect_err("non-finite rejected");
    matches!(err, KernelError::InvalidParameter { .. });

    mixture
        .set_logits(vec![0.5, -0.5])
        .expect("valid update succeeds");
    assert_eq!(mixture.logits(), &[0.5, -0.5]);
}

#[test]
fn trainable_adapter_round_trip() {
    let mixture = two_kernel_mixture(0.0, 0.0);
    let mut trainable = TrainableKernelMixture::new(mixture);
    assert_eq!(trainable.num_parameters(), 2);
    assert_eq!(trainable.parameters(), &[0.0, 0.0]);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    let (value_before, gradient) = trainable
        .evaluate_with_gradient(&x, &y)
        .expect("forward + grad");

    // Sanity: gradient sum is always zero (softmax invariance).
    let grad_sum: f64 = gradient.iter().sum();
    assert!(grad_sum.abs() < 1e-12);

    trainable.step(&gradient, 0.05).expect("sgd step");
    let params_after = trainable.parameters().to_vec();
    assert_ne!(params_after, vec![0.0, 0.0]);

    // Stepping backwards should return to approximately the original
    // logits (check that step is deterministic gradient descent).
    let neg: Vec<f64> = gradient.iter().map(|g| -*g).collect();
    trainable.step(&neg, 0.05).expect("reverse step");
    for (p, orig) in trainable.parameters().iter().zip([0.0, 0.0].iter()) {
        assert!((p - orig).abs() < 1e-12);
    }

    let value_restored = trainable.evaluate(&x, &y).expect("eval restored");
    assert!((value_restored - value_before).abs() < 1e-12);
}

#[test]
fn compute_gram_cross_set() {
    let mixture = two_kernel_mixture(0.0, 0.0);
    let xs_owned = [vec![1.0, 0.0], vec![0.0, 1.0]];
    let ys_owned = [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let xs: Vec<&[f64]> = xs_owned.iter().map(|v| v.as_slice()).collect();
    let ys: Vec<&[f64]> = ys_owned.iter().map(|v| v.as_slice()).collect();

    let gram = mixture.compute_gram(&xs, &ys).expect("gram");
    assert_eq!(gram.len(), 2);
    assert_eq!(gram[0].len(), 3);

    for (i, xi) in xs.iter().enumerate() {
        for (j, yj) in ys.iter().enumerate() {
            let expected = mixture.evaluate(xi, yj).expect("pair");
            assert!((gram[i][j] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn gradient_entries_sum_to_zero() {
    // The softmax Jacobian identity forces sum_i dK/dw_i = 0 for every
    // input pair — acts as a fast sanity check across the entire input
    // space.
    let mixture = two_kernel_mixture(0.7, -0.3);
    let pairs = [
        (vec![0.0, 0.0], vec![0.0, 0.0]),
        (vec![1.0, 2.0], vec![-1.0, 0.5]),
        (vec![3.0, -2.0], vec![1.0, 1.0]),
    ];
    for (x, y) in &pairs {
        let grad = mixture.gradient_wrt_logits(x, y).expect("grad");
        let sum: f64 = grad.iter().sum();
        assert!(
            sum.abs() < 1e-12,
            "gradient entries must sum to zero, got {}",
            sum
        );
    }
}

#[test]
fn psd_propagates_from_base_kernels() {
    // Linear + RBF are both PSD; mixture should be flagged PSD.
    let mixture = two_kernel_mixture(0.0, 0.0);
    assert!(mixture.is_psd());
}

#[test]
fn integrates_with_symbolic_kernel() {
    use crate::symbolic::KernelBuilder;

    // KernelBuilder produces Box<dyn Kernel>; convert to Arc for the
    // mixture library. This exercises the contract that the builder
    // accepts anything implementing the `Kernel` trait.
    let symbolic: Arc<dyn Kernel> = Arc::from(
        KernelBuilder::new()
            .add_scaled(Arc::new(LinearKernel::new()), 0.5)
            .build(),
    );
    let rbf: Arc<dyn Kernel> = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid"));

    let mixture = LearnedMixtureBuilder::new()
        .push_kernel(symbolic)
        .push_kernel(rbf)
        .build()
        .expect("non-empty library");
    let x = vec![1.0, 2.0];
    let y = vec![2.0, 1.0];
    let value = mixture.evaluate(&x, &y).expect("eval");
    assert!(value.is_finite());
}
