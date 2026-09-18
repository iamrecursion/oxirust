//! Dense layer and element-wise activations for deep kernel feature
//! extractors.
//!
//! A [`DenseLayer`] is an affine transformation `y = Wx + b` followed by
//! an element-wise [`Activation`]. The layer owns row-major weights
//! (`weights[i][j]` is the contribution of input `j` to output `i`) and a
//! parallel bias vector. The forward pass is plain `f64` arithmetic — we
//! deliberately stay out of ndarray here to keep the module lightweight
//! and easy to reason about for gradient derivations.

use crate::error::{KernelError, Result};

/// Element-wise activation applied at the output of a [`DenseLayer`].
///
/// Kept intentionally small for the v0.2.0 preview: `Identity` (no-op, for
/// the terminal layer or for identity-MLP test fixtures), `ReLU`, and
/// `Tanh`. Any more exotic activation (GELU, Swish, etc.) is a deliberate
/// non-goal for this release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activation {
    /// `f(x) = x`.
    Identity,
    /// `f(x) = max(0, x)`.
    ReLU,
    /// `f(x) = tanh(x)`.
    Tanh,
}

impl Activation {
    /// Apply the activation element-wise in place.
    pub fn apply_inplace(&self, values: &mut [f64]) {
        match self {
            Self::Identity => {}
            Self::ReLU => {
                for v in values.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Self::Tanh => {
                for v in values.iter_mut() {
                    *v = v.tanh();
                }
            }
        }
    }

    /// Apply the activation element-wise to a single scalar.
    pub fn apply_scalar(&self, v: f64) -> f64 {
        match self {
            Self::Identity => v,
            Self::ReLU => v.max(0.0),
            Self::Tanh => v.tanh(),
        }
    }

    /// Derivative `f'(z)` evaluated at the *pre-activation* `z`. Used by
    /// the analytical gradient paths in
    /// [`crate::deep_kernel::gradient`].
    ///
    /// For ReLU we follow the common convention `f'(0) = 0` — this is a
    /// sub-gradient (any value in `[0, 1]` is admissible), but `0` is the
    /// widely adopted choice that matches finite-difference tests at most
    /// random initialisations.
    pub fn derivative(&self, pre_activation: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::ReLU => {
                if pre_activation > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Tanh => {
                let t = pre_activation.tanh();
                1.0 - t * t
            }
        }
    }

    /// Human-readable name — handy for `Debug` impls and error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::ReLU => "ReLU",
            Self::Tanh => "Tanh",
        }
    }
}

/// An affine-plus-activation layer `y = activation(Wx + b)`.
///
/// Weights are stored row-major as `weights[out][in]`. The layer itself
/// has no learnable state beyond `weights` and `biases`; gradient and
/// optimiser logic lives with the caller
/// (see [`crate::deep_kernel::gradient`]).
#[derive(Clone, Debug)]
pub struct DenseLayer {
    /// Row-major weight matrix with shape `[output_dim][input_dim]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector with length `output_dim`.
    pub biases: Vec<f64>,
    /// Element-wise activation applied to the affine pre-activation.
    pub activation: Activation,
}

impl DenseLayer {
    /// Build a layer from raw weights and biases.
    ///
    /// Fails when weight rows disagree in width, when the bias vector
    /// does not match the weight row count, or when any entry is
    /// non-finite.
    pub fn new(weights: Vec<Vec<f64>>, biases: Vec<f64>, activation: Activation) -> Result<Self> {
        if weights.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "weights".to_string(),
                value: "[]".to_string(),
                reason: "dense layer must have at least one output".to_string(),
            });
        }
        let input_dim = weights[0].len();
        if input_dim == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "weights[0]".to_string(),
                value: "[]".to_string(),
                reason: "dense layer must have at least one input".to_string(),
            });
        }
        for (i, row) in weights.iter().enumerate() {
            if row.len() != input_dim {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![input_dim],
                    got: vec![row.len()],
                    context: format!("DenseLayer::new weights[{}]", i),
                });
            }
            for (j, &w) in row.iter().enumerate() {
                if !w.is_finite() {
                    return Err(KernelError::InvalidParameter {
                        parameter: format!("weights[{}][{}]", i, j),
                        value: w.to_string(),
                        reason: "weights must be finite".to_string(),
                    });
                }
            }
        }
        if biases.len() != weights.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![weights.len()],
                got: vec![biases.len()],
                context: "DenseLayer::new biases length".to_string(),
            });
        }
        for (i, &b) in biases.iter().enumerate() {
            if !b.is_finite() {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("biases[{}]", i),
                    value: b.to_string(),
                    reason: "biases must be finite".to_string(),
                });
            }
        }
        Ok(Self {
            weights,
            biases,
            activation,
        })
    }

    /// Input dimension (width of each weight row).
    pub fn input_dim(&self) -> usize {
        self.weights[0].len()
    }

    /// Output dimension (number of weight rows).
    pub fn output_dim(&self) -> usize {
        self.weights.len()
    }

    /// Activation attached to this layer.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Forward pass returning only the post-activation output.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let (_, post) = self.forward_with_preactivation(input)?;
        Ok(post)
    }

    /// Forward pass returning `(pre_activation, post_activation)`.
    ///
    /// The pre-activation is retained so callers implementing analytical
    /// gradients can feed it back into [`Activation::derivative`].
    pub fn forward_with_preactivation(&self, input: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if input.len() != self.input_dim() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.input_dim()],
                got: vec![input.len()],
                context: "DenseLayer::forward input length".to_string(),
            });
        }
        let mut pre = Vec::with_capacity(self.output_dim());
        for (row, &bias) in self.weights.iter().zip(self.biases.iter()) {
            let mut acc = bias;
            for (w, x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            pre.push(acc);
        }
        let mut post = pre.clone();
        self.activation.apply_inplace(&mut post);
        Ok((pre, post))
    }

    /// Count of trainable scalar parameters (`weights` + `biases`).
    pub fn parameter_count(&self) -> usize {
        self.output_dim() * self.input_dim() + self.output_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_relu_clamps_negative_to_zero() {
        let mut v = vec![-2.0, -0.5, 0.0, 0.5, 3.5];
        Activation::ReLU.apply_inplace(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.5, 3.5]);
    }

    #[test]
    fn activation_tanh_matches_std() {
        let v = Activation::Tanh.apply_scalar(0.7);
        assert!((v - 0.7_f64.tanh()).abs() < 1e-12);
    }

    #[test]
    fn activation_derivative_identity_is_one() {
        assert_eq!(Activation::Identity.derivative(-5.0), 1.0);
        assert_eq!(Activation::Identity.derivative(7.0), 1.0);
    }

    #[test]
    fn dense_layer_forward_identity() {
        let layer = DenseLayer::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
            Activation::Identity,
        )
        .expect("valid layer");
        let out = layer.forward(&[3.0, 4.0]).expect("forward");
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn dense_layer_rejects_dim_mismatch_input() {
        let layer =
            DenseLayer::new(vec![vec![1.0, 2.0]], vec![0.5], Activation::Identity).expect("valid");
        let err = layer
            .forward(&[1.0, 2.0, 3.0])
            .expect_err("must fail on 3-dim input");
        assert!(matches!(err, KernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn dense_layer_rejects_jagged_weights() {
        let err = DenseLayer::new(
            vec![vec![1.0, 2.0], vec![3.0]],
            vec![0.0, 0.0],
            Activation::Identity,
        )
        .expect_err("must fail");
        assert!(matches!(err, KernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn dense_layer_rejects_bias_length_mismatch() {
        let err = DenseLayer::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0.0],
            Activation::Identity,
        )
        .expect_err("must fail");
        assert!(matches!(err, KernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn dense_layer_parameter_count() {
        let layer = DenseLayer::new(
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![0.1, 0.2],
            Activation::ReLU,
        )
        .expect("valid");
        // 2 outputs * 3 inputs = 6 weights, + 2 biases = 8 params.
        assert_eq!(layer.parameter_count(), 8);
    }
}
