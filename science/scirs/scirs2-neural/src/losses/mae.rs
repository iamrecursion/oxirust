//! Mean Absolute Error (MAE) loss implementation
//!
//! MAE is a robust loss function for regression tasks that is less sensitive
//! to outliers compared to MSE.

use crate::error::Result;
use crate::losses::Loss;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// Mean Absolute Error (MAE / L1 Loss)
///
/// Computes the mean of absolute differences between predictions and targets:
/// L = (1/n) * Σ|prediction - target|
///
/// MAE is more robust to outliers than MSE because it doesn't square the errors,
/// making it a good choice when your data contains outliers or when you want
/// to penalize all errors equally regardless of magnitude.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::losses::{Loss, MeanAbsoluteError};
/// use scirs2_core::ndarray::Array;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let mae = MeanAbsoluteError::new();
///
/// let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
/// let targets = Array::from_vec(vec![1.5, 1.5, 3.5]).into_dyn();
///
/// let loss = mae.forward(&predictions, &targets)?;
/// // MAE = mean(|1.0-1.5|, |2.0-1.5|, |3.0-3.5|) = mean(0.5, 0.5, 0.5) = 0.5
/// assert!((loss - 0.5_f64).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Create a new MeanAbsoluteError loss function
    pub fn new() -> Self {
        Self
    }
}

impl<F: Float + Debug + NumAssign> Loss<F> for MeanAbsoluteError {
    /// Calculate the mean absolute error loss
    ///
    /// # Arguments
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth target values
    ///
    /// # Returns
    /// The mean absolute error as a scalar
    fn forward(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        let n = F::from(predictions.len()).unwrap_or(F::one());

        // Compute |predictions - targets| and sum
        let abs_diff_sum = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).abs())
            .fold(F::zero(), |acc, x| acc + x);

        Ok(abs_diff_sum / n)
    }

    /// Calculate the gradient of MAE with respect to predictions
    ///
    /// The gradient is sign(predictions - targets) / n
    /// This is subgradient at zero (we use 0 when prediction == target)
    fn backward(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let n = F::from(predictions.len()).unwrap_or(F::one());

        // Gradient: sign(predictions - targets) / n
        let gradients: Vec<F> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                if diff > F::zero() {
                    F::one() / n
                } else if diff < F::zero() {
                    -F::one() / n
                } else {
                    F::zero() // Subgradient at 0
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_mae_forward_basic() {
        let mae = MeanAbsoluteError::new();
        let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let targets = Array::from_vec(vec![1.5, 1.5, 3.5]).into_dyn();

        let loss = mae
            .forward(&predictions, &targets)
            .expect("Operation failed");
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mae_forward_perfect() {
        let mae = MeanAbsoluteError::new();
        let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let targets = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();

        let loss = mae
            .forward(&predictions, &targets)
            .expect("Operation failed");
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_mae_backward() {
        let mae = MeanAbsoluteError::new();
        let predictions = Array::from_vec(vec![1.0, 3.0, 2.0]).into_dyn();
        let targets = Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn();

        let gradients = mae
            .backward(&predictions, &targets)
            .expect("Operation failed");

        // predictions - targets = [-1, 1, 0]
        // signs = [-1, 1, 0], divided by n=3
        let expected = [-1.0 / 3.0, 1.0 / 3.0, 0.0];
        for (g, e) in gradients.iter().zip(expected.iter()) {
            assert!((*g - *e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mae_with_negative_values() {
        let mae = MeanAbsoluteError::new();
        let predictions = Array::from_vec(vec![-1.0, -2.0, 0.0]).into_dyn();
        let targets = Array::from_vec(vec![1.0, -1.0, -1.0]).into_dyn();

        let loss = mae
            .forward(&predictions, &targets)
            .expect("Operation failed");
        // |−1 − 1| + |−2 − (−1)| + |0 − (−1)| = 2 + 1 + 1 = 4, mean = 4/3
        assert!((loss - 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae_2d() {
        let mae = MeanAbsoluteError::new();
        let predictions = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Operation failed")
            .into_dyn();
        let targets = Array::from_shape_vec((2, 2), vec![1.0, 3.0, 2.0, 5.0])
            .expect("Operation failed")
            .into_dyn();

        let loss = mae
            .forward(&predictions, &targets)
            .expect("Operation failed");
        // Differences: |0|, |1|, |1|, |1| = sum 3, mean = 0.75
        assert!((loss - 0.75).abs() < 1e-6);
    }
}
