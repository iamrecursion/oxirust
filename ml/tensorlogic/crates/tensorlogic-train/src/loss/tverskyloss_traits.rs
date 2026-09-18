//! # TverskyLoss - Trait Implementations
//!
//! This module contains trait implementations for `TverskyLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::TverskyLoss;

impl Default for TverskyLoss {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            smooth: 1.0,
        }
    }
}

impl Loss for TverskyLoss {
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
        let mut true_pos = 0.0;
        let mut false_pos = 0.0;
        let mut false_neg = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];
                true_pos += pred * target;
                false_pos += pred * (1.0 - target);
                false_neg += (1.0 - pred) * target;
            }
        }
        let tversky_index = (true_pos + self.smooth)
            / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth);
        Ok(1.0 - tversky_index)
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
        let mut true_pos = 0.0;
        let mut false_pos = 0.0;
        let mut false_neg = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];
                true_pos += pred * target;
                false_pos += pred * (1.0 - target);
                false_neg += (1.0 - pred) * target;
            }
        }
        let denominator = true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth;
        let numerator = true_pos + self.smooth;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let target = targets[[i, j]];
                let d_tp = target;
                let d_fp = self.alpha * (1.0 - target);
                let d_fn = -self.beta * target;
                grad[[i, j]] = -(d_tp * denominator - numerator * (d_tp + d_fp + d_fn))
                    / (denominator * denominator);
            }
        }
        Ok(grad)
    }
}
