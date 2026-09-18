//! # HingeLoss - Trait Implementations
//!
//! This module contains trait implementations for `HingeLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::HingeLoss;

impl Default for HingeLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for HingeLoss {
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
        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];
                let loss = (self.margin - target * pred).max(0.0);
                total_loss += loss;
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        let n = predictions.nrows() as f64;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];
                if self.margin - target * pred > 0.0 {
                    grad[[i, j]] = -target / n;
                }
            }
        }
        Ok(grad)
    }
}
