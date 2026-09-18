//! Pure-CPU numeric known-answer tests for `oximedia-neural`.
//!
//! Wave 29 / Slice 5 — test-hardening only (no production change).
//!
//! These tests pin down three classes of behaviour with hand-computed oracles:
//!
//! 1. **Per-op NaN semantics** — the exact way each operation treats `NaN`
//!    depends on whether its guard is written `> 0.0` (collapses NaN to the
//!    "else" branch) or `< 0.0` (lets NaN fall through unchanged). We assert
//!    the *actual* behaviour of each op as written in the source, not a
//!    hypothetical one.
//! 2. **Shape errors** — `matmul` inner-dim mismatch and zero-dim
//!    `Tensor::new` produce the documented error variants.
//! 3. **Conv2d known-answer** — cross-correlation (no kernel flip) against
//!    hand-traced expected outputs, including multi-channel summation.
//!
//! All numeric expectations are computed by hand in the comments above each
//! assertion so the oracle is auditable.

// Test-file pragmatics: exact float equality is intentional for known-answer
// checks, and small numeric casts are unavoidable. These mirror the
// workspace-wide clippy allowances but are restated here for clarity/robustness.
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::unreadable_literal)]

use oximedia_neural::activations::{relu, softmax};
use oximedia_neural::error::NeuralError;
use oximedia_neural::layers::Conv2dLayer;
use oximedia_neural::tensor::{matmul, relu_inplace, Tensor};

// ──────────────────────────────────────────────────────────────────────────────
// 1. NaN is per-op
// ──────────────────────────────────────────────────────────────────────────────

/// Scalar `relu` uses the guard `if x > 0.0 { x } else { 0.0 }` (see
/// `activations.rs`). Because `NaN > 0.0` is `false`, NaN takes the *else*
/// branch and **collapses to `0.0`** — it does NOT propagate.
#[test]
fn test_relu_scalar_nan_collapses_to_zero() {
    let out = relu(f32::NAN);
    assert!(
        !out.is_nan(),
        "relu(NaN) must NOT be NaN: the `x > 0.0` guard sends NaN to the else branch"
    );
    assert_eq!(out, 0.0, "relu(NaN) collapses to 0.0");
}

/// Sanity anchors for the scalar relu guard around the collapse boundary.
#[test]
fn test_relu_scalar_boundary_values() {
    assert_eq!(relu(2.5), 2.5);
    assert_eq!(relu(-3.0), 0.0);
    // Exactly 0.0 is NOT > 0.0, so it takes the else branch and returns 0.0.
    assert_eq!(relu(0.0), 0.0);
}

/// Tensor element-wise `relu_inplace` uses the guard `if *v < 0.0 { *v = 0.0 }`
/// (see `tensor.rs`). Because `NaN < 0.0` is `false`, NaN is **left untouched**
/// and therefore PROPAGATES. This is the opposite-direction guard from the
/// scalar `relu`, so the two ops disagree on NaN — that asymmetry is the point
/// of this test.
#[test]
fn test_relu_inplace_propagates_nan() {
    let mut t = Tensor::from_data(vec![f32::NAN, -1.0, 3.0], vec![3]).expect("tensor from_data");
    relu_inplace(&mut t);
    let data = t.data();
    // NaN < 0.0 is false → NaN survives.
    assert!(
        data[0].is_nan(),
        "relu_inplace must PROPAGATE NaN (the `*v < 0.0` guard leaves NaN unchanged)"
    );
    // -1.0 < 0.0 is true → clamped to 0.0.
    assert_eq!(data[1], 0.0, "negative value clamps to 0.0");
    // 3.0 < 0.0 is false → unchanged.
    assert_eq!(data[2], 3.0, "positive value is unchanged");
}

/// `softmax` has an explicit NaN/non-finite fallback: it subtracts the max,
/// exponentiates, sums, then checks `if sum == 0.0 || !sum.is_finite()` and
/// returns a uniform distribution `1/len` in that case (see `activations.rs`).
///
/// With `[NaN, 1.0, 2.0]`: `max = max(NaN, 1.0, 2.0)`. `f32::max` returns the
/// non-NaN argument when one is NaN, so `max = 2.0`. Then `exp(NaN - 2.0) =
/// NaN`, so `sum = NaN`, which is not finite → the uniform fallback fires,
/// yielding `[1/3, 1/3, 1/3]`.
#[test]
fn test_softmax_nan_falls_back_to_uniform() {
    let out = softmax(&[f32::NAN, 1.0, 2.0]);
    assert_eq!(out.len(), 3);
    let third = 1.0_f32 / 3.0;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - third).abs() < 1e-6,
            "softmax([NaN,1,2])[{i}] = {v}, expected uniform 1/3 = {third}"
        );
    }
    // The fallback emits finite uniform values, not NaN.
    assert!(out.iter().all(|&v| v.is_finite()));
}

/// Free `matmul` with a NaN seeded into A[0][0]. A is a 2×2 with a NaN in the
/// top-left and the rest of an identity-like matrix; B is the 2×2 identity.
///
///   A = [[NaN, 0],      B = [[1, 0],
///        [0,   1]]           [0, 1]]
///
/// C = A·B. Row 0 of C is `[NaN·1 + 0·0, NaN·0 + 0·1] = [NaN, NaN-or-0]`.
/// The default `matmul` path uses a blocked SIMD GEMM (`scirs2-core`), which
/// may spread NaN across a row (e.g. `NaN·0.0 = NaN` rather than `0.0`).
/// We therefore assert NaN is PRESENT somewhere in row 0, and assert EXACT
/// finite values only for the fully-clean row 1 (which never touches A[0][0]):
///   Row 1 = [0·1 + 1·0, 0·0 + 1·1] = [0.0, 1.0].
#[test]
fn test_matmul_nan_contained_to_contaminated_row() {
    let a = Tensor::from_data(vec![f32::NAN, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor a");
    let b = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor b identity");
    let c = matmul(&a, &b).expect("matmul");
    assert_eq!(c.shape(), &[2, 2]);
    let d = c.data();

    // Row 0 was contaminated by A[0][0] = NaN → at least one NaN must appear.
    assert!(
        d[0].is_nan() || d[1].is_nan(),
        "matmul: NaN seeded in A[0][0] must contaminate row 0, got [{}, {}]",
        d[0],
        d[1]
    );

    // Row 1 never references A[0][0]; with B = identity it must be exactly the
    // second row of A: [0.0, 1.0], fully finite and exact.
    assert!(d[2].is_finite() && d[3].is_finite(), "row 1 must be finite");
    assert_eq!(d[2], 0.0, "C[1][0] = 0·1 + 1·0 = 0.0");
    assert_eq!(d[3], 1.0, "C[1][1] = 0·0 + 1·1 = 1.0");
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Shape errors
// ──────────────────────────────────────────────────────────────────────────────

/// `matmul` of `[2,3]` and `[4,5]`: the inner dimensions (3 vs 4) disagree, so
/// the op returns `NeuralError::ShapeMismatch` (the rank check passes — both
/// are 2-D — so we reach the inner-dim comparison).
#[test]
fn test_matmul_inner_dim_mismatch_is_shape_mismatch() {
    let a = Tensor::zeros(vec![2, 3]).expect("tensor a");
    let b = Tensor::zeros(vec![4, 5]).expect("tensor b");
    let err = matmul(&a, &b).expect_err("inner dims 3 != 4 must error");
    assert!(
        matches!(err, NeuralError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

/// `Tensor::new(vec![1,0,1])` contains a zero dimension, which is rejected with
/// `NeuralError::InvalidShape`.
#[test]
fn test_tensor_new_zero_dim_is_invalid_shape() {
    let err = Tensor::new(vec![1, 0, 1]).expect_err("zero dim must error");
    assert!(
        matches!(err, NeuralError::InvalidShape(_)),
        "expected InvalidShape, got {err:?}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Conv2d known-answer (cross-correlation, no kernel flip)
// ──────────────────────────────────────────────────────────────────────────────

/// Helper: build a `Conv2dLayer` with explicit weight + scalar bias and run a
/// single `[C,H,W]` forward pass, returning the flattened output data.
fn conv_forward(
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    weight_data: Vec<f32>,
    bias_value: f32,
    input: Tensor,
) -> Tensor {
    let mut layer = Conv2dLayer::new(
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        (1, 1), // stride
        (0, 0), // padding
    )
    .expect("conv2d new");
    assert_eq!(
        weight_data.len(),
        out_channels * in_channels * kernel_h * kernel_w,
        "weight_data length must match [out,in,kH,kW]"
    );
    *layer.weight.data_mut() = weight_data;
    // Single out-channel bias.
    for oc in 0..out_channels {
        layer.bias.data_mut()[oc] = bias_value;
    }
    layer.forward(&input).expect("conv forward")
}

/// 1×1 output, single channel, full-kernel dot product plus bias.
///
/// input `[1,2,2]` = [1,2,3,4]
/// weight `[1,1,2,2]` = [1,2,3,4]
/// bias = 0.5
///
/// out = 1·1 + 2·2 + 3·3 + 4·4 + 0.5 = 1+4+9+16 + 0.5 = 30.5
#[test]
fn test_conv2d_single_window_with_bias() {
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).expect("input");
    let out = conv_forward(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0], 0.5, input);
    assert_eq!(out.shape(), &[1, 1, 1], "single 2x2 window over 2x2 input");
    assert_eq!(out.data(), &[30.5]);
}

/// Identity-diagonal kernel extracts the main-diagonal sum of each 2×2 window.
///
/// input `[1,3,3]` = [1,2,3,4,5,6,7,8,9] (row-major)
/// weight `[1,1,2,2]` = [1,0,0,1]  (top-left + bottom-right)
/// bias = 0
///
/// Windows (cross-correlation, stride 1):
///   (0,0): in[0,0]=1 · 1 + in[1,1]=5 · 1 = 6
///   (0,1): in[0,1]=2 · 1 + in[1,2]=6 · 1 = 8
///   (1,0): in[1,0]=4 · 1 + in[2,1]=8 · 1 = 12
///   (1,1): in[1,1]=5 · 1 + in[2,2]=9 · 1 = 14
/// → [6, 8, 12, 14]
#[test]
fn test_conv2d_identity_diagonal_kernel() {
    let input =
        Tensor::from_data((1..=9).map(|x| x as f32).collect(), vec![1, 3, 3]).expect("input");
    let out = conv_forward(1, 1, 2, 2, vec![1.0, 0.0, 0.0, 1.0], 0.0, input);
    assert_eq!(out.shape(), &[1, 2, 2], "2x2 output over 3x3 input");
    assert_eq!(out.data(), &[6.0, 8.0, 12.0, 14.0]);
}

/// Multi-channel summation: an all-ones 2×2 kernel over a 2-channel 2×2 input
/// sums every element of both channels into a single output value.
///
/// input `[2,2,2]`: ch0 = [1,2,3,4], ch1 = [10,20,30,40]
/// weight `[1,2,2,2]` = all ones (one out-channel, two in-channels)
/// bias = 0
///
/// out = (1+2+3+4) + (10+20+30+40) = 10 + 100 = 110
#[test]
fn test_conv2d_multichannel_sum() {
    let input = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        vec![2, 2, 2],
    )
    .expect("input");
    let weight = vec![1.0_f32; 8]; // [1,2,2,2] all ones
    let out = conv_forward(2, 1, 2, 2, weight, 0.0, input);
    assert_eq!(out.shape(), &[1, 1, 1], "single window, single out-channel");
    assert_eq!(out.data(), &[110.0]);
}
