//! Tests for the quantization module.

use ndarray::Array2;
use tensorlogic_scirs_backend::quantization::{
    QuantizationGranularity, QuantizationParams, QuantizationScheme, QuantizationType,
};

use crate::quantization::{calibrate_linear, QuantizedLinear};

// -----------------------------------------------------------------------
// Helper: 4×8 weight matrix with simple values.
// -----------------------------------------------------------------------

fn sample_weight_4x8() -> Array2<f64> {
    Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64 * 0.5 - 14.0)
}

// -----------------------------------------------------------------------
// Test 1: round-trip per-tensor INT8 accuracy
// -----------------------------------------------------------------------

#[test]
fn test_roundtrip_per_tensor() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerTensor,
    );
    let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("from_fp per-tensor");

    let deq = qlinear.dequantize();
    let max_abs_err = weight
        .iter()
        .zip(deq.iter())
        .map(|(o, d)| (o - d).abs())
        .fold(0.0_f64, f64::max);

    // INT8 quantization noise: step = scale/2 ≤ (max_range/127)/2 ≈ 0.11
    // Allow up to 2.0 for safety across different calibrated ranges.
    assert!(
        max_abs_err < 2.0,
        "per-tensor round-trip max error={max_abs_err} >= 2.0"
    );
}

// -----------------------------------------------------------------------
// Test 2: round-trip per-channel INT8 accuracy
// -----------------------------------------------------------------------

#[test]
fn test_roundtrip_per_channel() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerChannel,
    );
    let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("from_fp per-channel");

    let deq = qlinear.dequantize();
    let max_abs_err = weight
        .iter()
        .zip(deq.iter())
        .map(|(o, d)| (o - d).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_err < 2.0,
        "per-channel round-trip max error={max_abs_err} >= 2.0"
    );
}

// -----------------------------------------------------------------------
// Test 3: forward pass matches f64 matmul within tolerance
// -----------------------------------------------------------------------

#[test]
fn test_forward_matches_fp() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerChannel,
    );
    let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("from_fp forward test");

    // x: [batch=2, in_features=8]
    let x = Array2::from_shape_fn((2, 8), |(i, j)| (i + j) as f64 * 0.1);

    let out_q = qlinear.forward(&x);

    // Reference: dequantize weights and matmul
    let weight_fp = qlinear.dequantize();
    let out_ref = x.dot(&weight_fp.t());

    assert_eq!(out_q.shape(), &[2, 4]);
    // forward() calls dequantize() internally, so they must match exactly.
    for (a, b) in out_q.iter().zip(out_ref.iter()) {
        assert!((a - b).abs() < 1e-12, "forward mismatch: {a} vs {b}");
    }
}

// -----------------------------------------------------------------------
// Test 4: calibration produces sensible scale and zero_point
// -----------------------------------------------------------------------

#[test]
fn test_calibration_sanity() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerTensor,
    );

    // scale must be positive and finite
    assert!(params.scale[0] > 0.0, "scale[0]={}", params.scale[0]);
    assert!(params.scale[0].is_finite(), "scale not finite");
    // Symmetric calibration → zero_point == 0
    assert_eq!(
        params.zero_point[0], 0,
        "symmetric should have zero_point==0"
    );
    // sanity: scale < 1.0 for values in [-14, 14]  (14/127 ≈ 0.11)
    assert!(
        params.scale[0] < 1.0,
        "scale[0]={} unreasonably large",
        params.scale[0]
    );
}

// -----------------------------------------------------------------------
// Test 5: per-channel granularity uses different scales per output row
// -----------------------------------------------------------------------

#[test]
fn test_per_channel_uses_different_scales() {
    // Design a weight where rows have very different magnitudes:
    //   row 0 ~ [-100, 100]  → large dynamic range
    //   row 1 ~ [-1,    1]   → tiny dynamic range
    let data = vec![100.0_f64, -100.0, 50.0, -50.0, 1.0_f64, -1.0, 0.5, -0.5];
    let weight = Array2::from_shape_vec((2, 4), data).expect("build test weight");
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerChannel,
    );

    assert_eq!(params.scale.len(), 2, "PerChannel needs 2 scales");
    // Row 0 scale ≈ 100/127 ≈ 0.787; Row 1 scale ≈ 1/127 ≈ 0.0079
    // They must differ significantly (at least 10x ratio).
    let ratio = params.scale[0] / params.scale[1];
    assert!(
        ratio > 10.0,
        "scale[0]={} scale[1]={} ratio={} (expected >10)",
        params.scale[0],
        params.scale[1],
        ratio
    );
}

// -----------------------------------------------------------------------
// Test 6: bias support
// -----------------------------------------------------------------------

#[test]
fn test_bias_shapes_checked() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerTensor,
    );
    let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("from_fp bias test");

    // Wrong bias length must fail
    let bad_bias = ndarray::Array1::zeros(3_usize); // out_features == 4
    assert!(qlinear.with_bias(bad_bias).is_err());
}

#[test]
fn test_bias_forward_correct() {
    let weight = sample_weight_4x8();
    let params = calibrate_linear(
        &weight,
        QuantizationType::Int8,
        QuantizationGranularity::PerTensor,
    );
    let qlinear = QuantizedLinear::from_fp(&weight, &params).expect("from_fp bias forward");

    let bias = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let qlinear_b = qlinear.with_bias(bias.clone()).expect("with_bias");

    let x = Array2::from_shape_fn((2, 8), |(i, j)| (i + j) as f64 * 0.1);
    let out_no_bias = QuantizedLinear::from_fp(&weight, &params)
        .expect("from_fp no-bias")
        .forward(&x);
    let out_with_bias = qlinear_b.forward(&x);

    // Each row of output_with_bias == output_no_bias + bias
    for batch in 0..2 {
        for ch in 0..4 {
            let expected = out_no_bias[[batch, ch]] + bias[ch];
            let got = out_with_bias[[batch, ch]];
            assert!(
                (expected - got).abs() < 1e-12,
                "bias mismatch at [{batch},{ch}]: expected={expected} got={got}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Test 7: invalid qtype returns error
// -----------------------------------------------------------------------

#[test]
fn test_invalid_qtype_returns_error() {
    let weight = sample_weight_4x8();
    let params = QuantizationParams {
        qtype: QuantizationType::Int4,
        scheme: QuantizationScheme::Symmetric,
        granularity: QuantizationGranularity::PerTensor,
        scale: vec![1.0],
        zero_point: vec![0],
        min_val: vec![-1.0],
        max_val: vec![1.0],
    };
    assert!(
        QuantizedLinear::from_fp(&weight, &params).is_err(),
        "Int4 should be rejected"
    );
}
