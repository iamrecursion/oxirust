//! Unit tests for the Deep Kernel Learning module.
//!
//! Covers: MLP forward, ReLU correctness, Xavier init scale bounds,
//! DeepKernel-equals-base under identity feature map, gradient-check
//! via finite differences, PSD propagation, and empty-extractor error
//! path.

use crate::deep_kernel::feature_extractor::{MLPFeatureExtractor, NeuralFeatureMap};
use crate::deep_kernel::gradient::{finite_difference_gradient, rbf_dkl_gradient};
use crate::deep_kernel::kernel::DeepKernel;
use crate::deep_kernel::layer::{Activation, DenseLayer};
use crate::error::KernelError;
use crate::types::{Kernel, RbfKernelConfig};
use crate::{LinearKernel, RbfKernel};

/// Three-layer MLP with hand-computable forward pass:
///
/// * Layer 0: `2 → 3`, weights `[[1,0],[0,1],[1,1]]`, biases `[0,0,0]`,
///   activation ReLU.
/// * Layer 1: `3 → 2`, weights `[[1,1,0],[0,1,1]]`, biases `[0,0]`,
///   activation ReLU.
/// * Layer 2: `2 → 1`, weights `[[1,1]]`, biases `[0]`, activation
///   Identity.
///
/// Given `x = [1, 2]`, the per-layer outputs are:
///
/// * Layer 0 pre: `[1, 2, 3]`, post (ReLU): `[1, 2, 3]`.
/// * Layer 1 pre: `[3, 5]`, post (ReLU): `[3, 5]`.
/// * Layer 2 pre: `[8]`, post (Identity): `[8]`.
#[test]
fn mlp_forward_three_layer_hand_computed() {
    let l0 = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        vec![0.0, 0.0, 0.0],
        Activation::ReLU,
    )
    .expect("valid layer 0");
    let l1 = DenseLayer::new(
        vec![vec![1.0, 1.0, 0.0], vec![0.0, 1.0, 1.0]],
        vec![0.0, 0.0],
        Activation::ReLU,
    )
    .expect("valid layer 1");
    let l2 = DenseLayer::new(vec![vec![1.0, 1.0]], vec![0.0], Activation::Identity)
        .expect("valid layer 2");
    let mlp = MLPFeatureExtractor::from_layers(vec![l0, l1, l2]).expect("valid");

    let out = mlp.forward(&[1.0, 2.0]).expect("forward");
    assert_eq!(out, vec![8.0]);
}

#[test]
fn relu_activation_is_zero_for_negative_inputs() {
    // ReLU must clamp all negatives to 0 and leave positives unchanged.
    let layer = DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::ReLU).expect("valid");
    let negatives = layer.forward(&[-3.0]).expect("forward");
    let positives = layer.forward(&[3.0]).expect("forward");
    assert_eq!(negatives, vec![0.0]);
    assert_eq!(positives, vec![3.0]);
}

#[test]
fn xavier_init_produces_bounded_weights() {
    // Xavier/Glorot normal std = sqrt(2 / (fan_in + fan_out)). For a
    // `[8, 16, 4]` topology the per-layer std is:
    //   layer 0: sqrt(2/24) ≈ 0.2887
    //   layer 1: sqrt(2/20) ≈ 0.3162
    // So individual weights should land within a few std deviations of
    // zero at seed time. We check the coarser property that every
    // sampled weight is within `5σ_max` of the origin — a
    // fails-with-probability < 10⁻⁶ check that is robust across any
    // seed value.
    let mlp = MLPFeatureExtractor::xavier_init(
        &[8, 16, 4],
        &[Activation::ReLU, Activation::Identity],
        0xABCDEF,
    )
    .expect("xavier init");

    let stds = [(2.0f64 / 24.0).sqrt(), (2.0f64 / 20.0).sqrt()];
    let max_std = stds.iter().cloned().fold(0.0f64, f64::max);

    for (layer_idx, layer) in mlp.layers().iter().enumerate() {
        let std_here = stds[layer_idx];
        assert!(
            std_here > 0.0 && std_here.is_finite(),
            "xavier std must be positive finite, got {}",
            std_here
        );
        for row in &layer.weights {
            for &w in row {
                assert!(w.is_finite(), "Xavier sample must be finite, got {}", w);
                assert!(
                    w.abs() < 5.0 * max_std,
                    "weight {} outside 5σ bound (σ_max={})",
                    w,
                    max_std
                );
            }
        }
        // Biases start at zero.
        for &b in &layer.biases {
            assert_eq!(b, 0.0);
        }
    }
}

#[test]
fn deep_kernel_equals_base_under_identity_mlp_1x1() {
    // A 1→1 identity-MLP of weight 1.0 bias 0.0 should make DKL
    // indistinguishable from the base kernel. Tested on both Linear
    // and RBF to exercise both sides.
    let layer = DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).expect("valid");
    let mlp = MLPFeatureExtractor::from_layers(vec![layer]).expect("valid");
    let linear = LinearKernel::new();
    let dkl = DeepKernel::new(mlp.clone(), linear);
    let base = LinearKernel::new();

    let pairs = [
        (vec![1.0], vec![1.0]),
        (vec![0.5], vec![-2.0]),
        (vec![5.25], vec![0.0]),
    ];
    for (x, y) in &pairs {
        let got = dkl.compute(x, y).expect("dkl");
        let want = base.compute(x, y).expect("base");
        assert!((got - want).abs() < 1e-12, "dkl {} != base {}", got, want);
    }

    let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
    let dkl_rbf = DeepKernel::new(mlp, rbf);
    let rbf_base = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
    for (x, y) in &pairs {
        let got = dkl_rbf.compute(x, y).expect("dkl");
        let want = rbf_base.compute(x, y).expect("base");
        assert!((got - want).abs() < 1e-12);
    }
}

#[test]
fn gradient_check_finite_difference_matches_analytical() {
    // Full pipeline: a 2-2-2 MLP with Tanh hidden activation + RBF base.
    // The analytical RBF gradient must agree with central finite
    // differences to within 1e-3 at every parameter slot.
    let mlp =
        MLPFeatureExtractor::xavier_init(&[2, 3, 2], &[Activation::Tanh, Activation::Identity], 7)
            .expect("xavier");
    let rbf = RbfKernel::new(RbfKernelConfig::new(0.7)).expect("valid");
    let mut dkl = DeepKernel::new(mlp, rbf);

    let x = vec![0.1, -0.2];
    let y = vec![0.3, 0.4];
    let analytical = rbf_dkl_gradient(&dkl, &x, &y).expect("analytical");
    let numerical = finite_difference_gradient(&mut dkl, &x, &y, 1e-5).expect("numerical");

    assert_eq!(analytical.len(), numerical.len());
    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < 1e-3,
            "gradient mismatch at param {}: analytical {} vs numerical {}",
            i,
            a,
            n
        );
    }
}

#[test]
fn deep_kernel_psd_follows_base() {
    // DKL is PSD iff the base kernel is PSD. RBF is PSD → DKL PSD.
    let mlp =
        MLPFeatureExtractor::xavier_init(&[2, 3, 2], &[Activation::ReLU, Activation::Identity], 0)
            .expect("xavier");
    let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
    let dkl = DeepKernel::new(mlp, rbf);
    assert!(dkl.is_psd(), "DKL(RBF) should inherit PSD from RBF");
}

#[test]
fn empty_mlp_construction_errors() {
    let err = MLPFeatureExtractor::from_layers(vec![]).expect_err("must fail");
    assert!(matches!(err, KernelError::InvalidParameter { .. }));

    // Xavier init also rejects widths of length < 2.
    let err = MLPFeatureExtractor::xavier_init(&[3], &[], 0).expect_err("must fail");
    assert!(matches!(err, KernelError::InvalidParameter { .. }));

    // Widths containing a zero dimension are also rejected.
    let err =
        MLPFeatureExtractor::xavier_init(&[3, 0, 2], &[Activation::ReLU, Activation::Identity], 0)
            .expect_err("must fail");
    assert!(matches!(err, KernelError::InvalidParameter { .. }));
}

#[test]
fn sync_from_flat_updates_forward_output() {
    // Mutating the flat parameter buffer and calling sync_from_flat
    // must change the subsequent forward output. This is the
    // invariant optimisers rely on.
    let mlp =
        MLPFeatureExtractor::xavier_init(&[2, 2, 1], &[Activation::Tanh, Activation::Identity], 3)
            .expect("xavier");
    let mut mlp = mlp;
    let before = mlp.forward(&[0.5, -0.5]).expect("forward");

    // Perturb every parameter by +1 and sync.
    let perturbed: Vec<f64> = mlp.parameters().iter().map(|p| p + 1.0).collect();
    mlp.parameters_mut().copy_from_slice(&perturbed);
    mlp.sync_from_flat().expect("sync");
    let after = mlp.forward(&[0.5, -0.5]).expect("forward");

    let any_change = before
        .iter()
        .zip(after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(any_change, "sync_from_flat must affect the forward pass");
}

#[test]
fn sync_from_flat_rejects_non_finite_parameters() {
    let mut mlp =
        MLPFeatureExtractor::xavier_init(&[2, 2], &[Activation::Identity], 0).expect("xavier");
    mlp.parameters_mut()[0] = f64::NAN;
    let err = mlp.sync_from_flat().expect_err("NaN must be rejected");
    assert!(matches!(err, KernelError::InvalidParameter { .. }));
}
