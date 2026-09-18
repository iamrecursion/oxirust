//! Huber Loss implementation
//!
//! Huber loss combines the best properties of MSE and MAE:
//! - Quadratic for small errors (like MSE, smooth gradients)
//! - Linear for large errors (like MAE, robust to outliers)

use crate::error::Result;
use crate::losses::Loss;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// Huber Loss (Smooth L1 Loss)
///
/// Huber loss is a robust loss function that is less sensitive to outliers than MSE.
/// It behaves like MSE for small errors and like MAE for large errors:
///
/// L = { 0.5 * (y - ŷ)²           if |y - ŷ| <= δ
///     { δ * (|y - ŷ| - 0.5 * δ)  if |y - ŷ| > δ
///
/// where δ (delta) is the threshold that controls the transition point.
///
/// # Properties
/// - Smooth gradients near zero (unlike MAE)
/// - Linear growth for large errors (unlike MSE)
/// - Configurable sensitivity via delta parameter
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::losses::{Loss, HuberLoss};
/// use scirs2_core::ndarray::Array;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// // Create Huber loss with delta=1.0
/// let huber = HuberLoss::new(1.0);
///
/// let predictions = Array::from_vec(vec![1.0, 2.0, 10.0]).into_dyn();
/// let targets = Array::from_vec(vec![1.5, 2.0, 3.0]).into_dyn();
///
/// let loss = huber.forward(&predictions, &targets)?;
/// // Small errors (0.5, 0) use quadratic, large error (7) uses linear
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss<F: Float + NumAssign> {
    /// Delta threshold for switching between quadratic and linear
    delta: F,
}

impl<F: Float + NumAssign> HuberLoss<F> {
    /// Create a new HuberLoss with the specified delta threshold
    ///
    /// # Arguments
    /// * `delta` - The threshold at which to switch from quadratic to linear loss.
    ///   Common values are 1.0 or 1.35 (for 95% efficiency at Gaussian data).
    pub fn new(delta: F) -> Self {
        Self { delta }
    }

    /// Create a HuberLoss with delta=1.0 (common default)
    pub fn default_delta() -> Self {
        Self {
            delta: F::from(1.0).unwrap_or(F::one()),
        }
    }
}

impl<F: Float + NumAssign> Default for HuberLoss<F> {
    fn default() -> Self {
        Self::default_delta()
    }
}

impl<F: Float + Debug + NumAssign> Loss<F> for HuberLoss<F> {
    /// Calculate the Huber loss
    ///
    /// # Arguments
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth target values
    ///
    /// # Returns
    /// The mean Huber loss as a scalar
    fn forward(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        let n = F::from(predictions.len()).unwrap_or(F::one());
        let half = F::from(0.5).unwrap_or(F::one() / (F::one() + F::one()));

        let loss_sum = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                let abs_diff = diff.abs();

                if abs_diff <= self.delta {
                    // Quadratic region: 0.5 * diff^2
                    half * diff * diff
                } else {
                    // Linear region: delta * (|diff| - 0.5 * delta)
                    self.delta * (abs_diff - half * self.delta)
                }
            })
            .fold(F::zero(), |acc, x| acc + x);

        Ok(loss_sum / n)
    }

    /// Calculate the gradient of Huber loss with respect to predictions
    ///
    /// The gradient is:
    /// - (prediction - target) / n    if |diff| <= delta
    /// - delta * sign(diff) / n       if |diff| > delta
    fn backward(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let n = F::from(predictions.len()).unwrap_or(F::one());

        let gradients: Vec<F> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                let abs_diff = diff.abs();

                if abs_diff <= self.delta {
                    // Quadratic region gradient: diff / n
                    diff / n
                } else {
                    // Linear region gradient: delta * sign(diff) / n
                    if diff > F::zero() {
                        self.delta / n
                    } else {
                        -self.delta / n
                    }
                }
            })
            .collect();

        Array::from_shape_vec(predictions.dim(), gradients).map_err(|e| {
            crate::error::NeuralError::InferenceError(format!(
                "Failed to create gradient array: {}",
                e
            ))
        })
    }
}

/// Smooth L1 Loss (variant of Huber with delta=1.0)
///
/// This is identical to Huber loss with delta=1.0, commonly used in
/// object detection (e.g., Faster R-CNN for bounding box regression).
pub type SmoothL1Loss<F> = HuberLoss<F>;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_huber_quadratic_region() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let targets = Array::from_vec(vec![1.2, 1.8, 3.1]).into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");

        // All diffs are <= 1.0, so use quadratic: 0.5 * diff^2
        // diffs: -0.2, 0.2, -0.1
        // losses: 0.02, 0.02, 0.005
        // mean: 0.045 / 3 = 0.015
        let expected = (0.5 * 0.04 + 0.5 * 0.04 + 0.5 * 0.01) / 3.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_huber_linear_region() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![0.0, 5.0]).into_dyn();
        let targets = Array::from_vec(vec![3.0, 0.0]).into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");

        // diffs: -3.0 (|diff|=3 > 1), 5.0 (|diff|=5 > 1)
        // Linear: delta * (|diff| - 0.5*delta) = 1.0 * (3.0 - 0.5) = 2.5, 1.0 * (5.0 - 0.5) = 4.5
        // mean: (2.5 + 4.5) / 2 = 3.5
        let expected = (2.5 + 4.5) / 2.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_huber_mixed_regions() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![0.0, 0.5]).into_dyn();
        let targets = Array::from_vec(vec![3.0, 0.3]).into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");

        // diff1 = -3.0, |diff| = 3 > 1, linear: 1.0 * (3.0 - 0.5) = 2.5
        // diff2 = 0.2, |diff| = 0.2 <= 1, quadratic: 0.5 * 0.04 = 0.02
        // mean: (2.5 + 0.02) / 2 = 1.26
        let expected = (2.5 + 0.02) / 2.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_huber_perfect_predictions() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let targets = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_huber_backward_quadratic() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![1.0, 2.0]).into_dyn();
        let targets = Array::from_vec(vec![1.5, 1.5]).into_dyn();

        let gradients = huber
            .backward(&predictions, &targets)
            .expect("Operation failed");

        // diffs: -0.5, 0.5 (both in quadratic region)
        // gradients: diff / n = -0.5/2, 0.5/2 = -0.25, 0.25
        assert!((gradients[[0]] - (-0.25)).abs() < 1e-6);
        assert!((gradients[[1]] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_huber_backward_linear() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_vec(vec![0.0, 5.0]).into_dyn();
        let targets = Array::from_vec(vec![3.0, 0.0]).into_dyn();

        let gradients = huber
            .backward(&predictions, &targets)
            .expect("Operation failed");

        // diffs: -3.0 (linear, negative), 5.0 (linear, positive)
        // gradients: -delta/n, delta/n = -0.5, 0.5
        assert!((gradients[[0]] - (-0.5)).abs() < 1e-6);
        assert!((gradients[[1]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_custom_delta() {
        let huber = HuberLoss::new(0.5);
        let predictions = Array::from_vec(vec![0.0]).into_dyn();
        let targets = Array::from_vec(vec![1.0]).into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");

        // diff = -1.0, |diff| = 1.0 > 0.5 (delta), so linear
        // loss = 0.5 * (1.0 - 0.25) = 0.5 * 0.75 = 0.375
        assert!((loss - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_huber_2d() {
        let huber = HuberLoss::new(1.0);
        let predictions = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Operation failed")
            .into_dyn();
        let targets = Array::from_shape_vec((2, 2), vec![1.0, 2.5, 6.0, 4.0])
            .expect("Operation failed")
            .into_dyn();

        let loss = huber
            .forward(&predictions, &targets)
            .expect("Operation failed");

        // diffs: 0, -0.5, -3, 0
        // losses: 0, 0.125 (quadratic), 2.5 (linear), 0
        // mean: 2.625 / 4 = 0.65625
        let expected = (0.0 + 0.125 + 2.5 + 0.0) / 4.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_l1_alias() {
        let smooth_l1: SmoothL1Loss<f64> = SmoothL1Loss::new(1.0);
        let predictions = Array::from_vec(vec![1.0, 2.0]).into_dyn();
        let targets = Array::from_vec(vec![1.5, 5.0]).into_dyn();

        let loss = smooth_l1
            .forward(&predictions, &targets)
            .expect("Operation failed");
        assert!(loss > 0.0);
    }
}
