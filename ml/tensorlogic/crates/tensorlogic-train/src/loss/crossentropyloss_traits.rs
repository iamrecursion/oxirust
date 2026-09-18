//! # CrossEntropyLoss - Trait Implementations
//!
//! This module contains trait implementations for `CrossEntropyLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::CrossEntropyLoss;

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl Loss for CrossEntropyLoss {
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
        let n = predictions.nrows() as f64;
        let mut total_loss = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                total_loss -= target * pred.ln();
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
        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                grad[[i, j]] = -(target / pred) / n;
            }
        }
        Ok(grad)
    }
}
