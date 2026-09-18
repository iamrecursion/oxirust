//! Calibration helpers for [`crate::quantization::QuantizedLinear`] weight matrices.
//!
//! The core function [`calibrate_linear`] computes per-tensor or per-channel
//! [`QuantizationParams`] from a 2-D weight matrix (shape `(out_features,
//! in_features)`).
//!
//! For `PerTensor`, `tensorlogic_scirs_backend::quantization::calibrate_quantization`
//! is invoked with the flattened weight as the single sample.
//!
//! For `PerChannel`, row-wise calibration is performed: each row (output
//! channel) is calibrated independently, producing a `Vec<f64>` of length
//! `out_features` for both `scale` and `zero_point`.

use ndarray::ArrayD;
use ndarray::Axis;
use tensorlogic_scirs_backend::quantization::{
    calibrate_quantization, QuantizationGranularity, QuantizationParams, QuantizationScheme,
    QuantizationType,
};

/// Calibrate quantization parameters for a 2-D weight matrix.
///
/// # Arguments
///
/// * `weight`      — Weight matrix of shape `(out_features, in_features)`.
/// * `qtype`       — Target quantization type (currently `Int8` is best supported).
/// * `granularity` — `PerTensor` or `PerChannel` (one scale per output channel).
///
/// # Panics
///
/// Never panics; any edge cases (e.g. zero-magnitude rows) produce a scale of
/// `0.0 / 127.0` which is 0 — the caller should validate if needed.
pub fn calibrate_linear(
    weight: &ndarray::Array2<f64>,
    qtype: QuantizationType,
    granularity: QuantizationGranularity,
) -> QuantizationParams {
    match granularity {
        QuantizationGranularity::PerTensor => calibrate_per_tensor(weight, qtype),
        QuantizationGranularity::PerChannel => calibrate_per_channel(weight, qtype),
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn calibrate_per_tensor(
    weight: &ndarray::Array2<f64>,
    qtype: QuantizationType,
) -> QuantizationParams {
    // Flatten the weight into a 1-D sample for calibrate_quantization.
    let flat: Vec<f64> = weight.iter().copied().collect();
    let sample = ArrayD::from_shape_vec(vec![flat.len()], flat)
        .expect("calibrate_linear: flat sample shape");
    // calibrate_quantization returns PerTensor params by design.
    calibrate_quantization(&[sample], qtype, QuantizationScheme::Symmetric).unwrap_or_else(|_| {
        // Fallback: unit scale, zero zero_point.
        QuantizationParams {
            qtype,
            scheme: QuantizationScheme::Symmetric,
            granularity: QuantizationGranularity::PerTensor,
            scale: vec![1.0],
            zero_point: vec![0],
            min_val: vec![-1.0],
            max_val: vec![1.0],
        }
    })
}

fn calibrate_per_channel(
    weight: &ndarray::Array2<f64>,
    qtype: QuantizationType,
) -> QuantizationParams {
    let out_features = weight.nrows();
    let mut scales = Vec::with_capacity(out_features);
    let mut zero_points = Vec::with_capacity(out_features);
    let mut min_vals = Vec::with_capacity(out_features);
    let mut max_vals = Vec::with_capacity(out_features);

    for row in weight.axis_iter(Axis(0)) {
        let row_vec: Vec<f64> = row.iter().copied().collect();
        let sample = ArrayD::from_shape_vec(vec![row_vec.len()], row_vec)
            .expect("calibrate_linear: row sample shape");

        // Use calibrate_quantization per-row.
        match calibrate_quantization(&[sample], qtype, QuantizationScheme::Symmetric) {
            Ok(p) => {
                scales.push(p.scale[0]);
                zero_points.push(p.zero_point[0]);
                min_vals.push(p.min_val[0]);
                max_vals.push(p.max_val[0]);
            }
            Err(_) => {
                scales.push(1.0);
                zero_points.push(0);
                min_vals.push(-1.0);
                max_vals.push(1.0);
            }
        }
    }

    QuantizationParams {
        qtype,
        scheme: QuantizationScheme::Symmetric,
        granularity: QuantizationGranularity::PerChannel,
        scale: scales,
        zero_point: zero_points,
        min_val: min_vals,
        max_val: max_vals,
    }
}
