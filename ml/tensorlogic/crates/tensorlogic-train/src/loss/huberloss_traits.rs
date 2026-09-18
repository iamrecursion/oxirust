//! # HuberLoss - Trait Implementations
//!
//! This module contains trait implementations for `HuberLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::HuberLoss;

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for HuberLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }
        let n = predictions.len() as f64;
        let mut total_loss = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let diff = (predictions[[i, j]] - targets[[i, j]]).abs();
                if diff <= self.delta {
                    total_loss += 0.5 * diff * diff;
                } else {
                    total_loss += self.delta * (diff - 0.5 * self.delta);
                }
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }
        let n = predictions.len() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let diff = predictions[[i, j]] - targets[[i, j]];
                let abs_diff = diff.abs();
                if abs_diff <= self.delta {
                    grad[[i, j]] = diff / n;
                } else {
                    grad[[i, j]] = self.delta * diff.signum() / n;
                }
            }
        }
        Ok(grad)
    }
}
