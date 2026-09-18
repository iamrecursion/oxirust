//! [`QuantizedLinear`]: an INT8 weight matrix with per-channel or per-tensor
//! scale/zero_point.
//!
//! ## Design
//!
//! Weights are stored as `Array2<i8>` with shape `(out_features, in_features)`.
//! The forward pass dequantizes the weight matrix on every call and then
//! performs a standard f64 `matmul`.  This is the *CPU-first honest cut*;
//! integer-matmul (packed int8 GEMM) is a future follow-up.
//!
//! For `PerChannel` granularity each row (`output channel`) has its own
//! `scale[c]` and `zero_point[c]`.  For `PerTensor` a single pair applies to
//! all elements.

use ndarray::{Array1, Array2, Axis};
use tensorlogic_scirs_backend::quantization::{
    QuantizationGranularity, QuantizationParams, QuantizationType, QuantizedTensor,
};

/// Error type for quantization operations on linear layers.
#[derive(Debug)]
pub enum QuantizationError {
    /// Weight matrix shape does not match expectations.
    ShapeMismatch(String),
    /// Quantization parameters are inconsistent.
    InvalidParams(String),
}

impl std::fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            QuantizationError::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
        }
    }
}

impl std::error::Error for QuantizationError {}

/// A weight matrix quantized to i8, with per-channel or per-tensor
/// `scale`/`zero_point`.
///
/// Forward pass: dequantize weights on the fly, then f64 matmul.
///
/// ## Layout
///
/// `weight_q` has shape `(out_features, in_features)` — the same convention
/// as `LinearExpert::weights` (see `moe/expert.rs`).
pub struct QuantizedLinear {
    /// Quantized weights of shape `(out_features, in_features)`.
    weight_q: Array2<i8>,
    /// Scale factor per channel (length 1 for PerTensor, `out_features` for PerChannel).
    scale: Vec<f64>,
    /// Zero-point per channel.
    zero_point: Vec<i32>,
    /// Granularity used during quantization.
    granularity: QuantizationGranularity,
    /// Optional bias of length `out_features`.
    bias: Option<Array1<f64>>,
}

impl QuantizedLinear {
    /// Quantize an existing f64 weight matrix using the provided params.
    ///
    /// Only `Int8` quantization type is supported.  Use
    /// [`crate::quantization::calibrate_linear`] to produce `params`.
    ///
    /// # Errors
    ///
    /// - [`QuantizationError::InvalidParams`] if `qtype != Int8`.
    /// - [`QuantizationError::ShapeMismatch`] if the weight is not 2-D or the
    ///   scale/zero_point vectors are the wrong length for `PerChannel`.
    pub fn from_fp(
        weight: &Array2<f64>,
        params: &QuantizationParams,
    ) -> Result<Self, QuantizationError> {
        if params.qtype != QuantizationType::Int8 {
            return Err(QuantizationError::InvalidParams(format!(
                "only Int8 is supported, got {:?}",
                params.qtype
            )));
        }

        let (out_features, _in_features) = weight.dim();

        // Validate per-channel scale length.
        if params.granularity == QuantizationGranularity::PerChannel
            && params.scale.len() != out_features
        {
            return Err(QuantizationError::ShapeMismatch(format!(
                "PerChannel: scale.len()={} but out_features={}",
                params.scale.len(),
                out_features,
            )));
        }

        // Call scirs-backend to get the quantized f64 array.
        let weight_dyn = weight.clone().into_dyn();
        let qt = QuantizedTensor::quantize(&weight_dyn, params.clone());

        // Cast f64 quantized values to i8.
        let weight_i8 = qt
            .data
            .mapv(|x| x as i8)
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| {
                QuantizationError::ShapeMismatch(format!("dimensionality cast failed: {e}"))
            })?;

        Ok(Self {
            weight_q: weight_i8,
            scale: params.scale.clone(),
            zero_point: params.zero_point.clone(),
            granularity: params.granularity,
            bias: None,
        })
    }

    /// Attach a bias vector of length `out_features`.
    ///
    /// # Errors
    ///
    /// [`QuantizationError::ShapeMismatch`] if `bias.len() != out_features`.
    pub fn with_bias(mut self, bias: Array1<f64>) -> Result<Self, QuantizationError> {
        let out_features = self.weight_q.nrows();
        if bias.len() != out_features {
            return Err(QuantizationError::ShapeMismatch(format!(
                "bias.len()={} but out_features={}",
                bias.len(),
                out_features
            )));
        }
        self.bias = Some(bias);
        Ok(self)
    }

    /// Dequantize and run matmul.
    ///
    /// Input `x` must have shape `[batch, in_features]`.
    /// Output has shape `[batch, out_features]`.
    ///
    /// `fp = (q - zero_point[c]) * scale[c]` per element, where `c` is the
    /// output channel (row) index when `granularity == PerChannel`, or `0`
    /// for `PerTensor`.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let weight_fp = self.dequantize();
        // matmul: x @ weight_fp.t() gives [batch, out_features]
        let out = x.dot(&weight_fp.t());
        match &self.bias {
            Some(b) => out + b,
            None => out,
        }
    }

    /// Dequantize the stored i8 weights back to f64.
    ///
    /// For `PerTensor`: all elements use `scale[0]` / `zero_point[0]`.
    /// For `PerChannel`: each row `c` uses `scale[c]` / `zero_point[c]`.
    pub fn dequantize(&self) -> Array2<f64> {
        let (out_features, in_features) = self.weight_q.dim();
        let mut fp = Array2::<f64>::zeros((out_features, in_features));

        match self.granularity {
            QuantizationGranularity::PerTensor => {
                let s = self.scale[0];
                let zp = self.zero_point[0] as f64;
                for (q_row, mut fp_row) in self
                    .weight_q
                    .axis_iter(Axis(0))
                    .zip(fp.axis_iter_mut(Axis(0)))
                {
                    for (q_val, fp_val) in q_row.iter().zip(fp_row.iter_mut()) {
                        *fp_val = (*q_val as f64 - zp) * s;
                    }
                }
            }
            QuantizationGranularity::PerChannel => {
                for (c, (q_row, mut fp_row)) in self
                    .weight_q
                    .axis_iter(Axis(0))
                    .zip(fp.axis_iter_mut(Axis(0)))
                    .enumerate()
                {
                    let s = self.scale.get(c).copied().unwrap_or(self.scale[0]);
                    let zp = self.zero_point.get(c).copied().unwrap_or(0) as f64;
                    for (q_val, fp_val) in q_row.iter().zip(fp_row.iter_mut()) {
                        *fp_val = (*q_val as f64 - zp) * s;
                    }
                }
            }
        }

        fp
    }

    /// Return the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.weight_q.nrows()
    }

    /// Return the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.weight_q.ncols()
    }

    /// Return the quantization granularity.
    pub fn granularity(&self) -> QuantizationGranularity {
        self.granularity
    }

    /// Return the scale factors (length 1 for PerTensor, `out_features` for PerChannel).
    pub fn scales(&self) -> &[f64] {
        &self.scale
    }
}
