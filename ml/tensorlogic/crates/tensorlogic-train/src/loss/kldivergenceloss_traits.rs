//! # KLDivergenceLoss - Trait Implementations
//!
//! This module contains trait implementations for `KLDivergenceLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::KLDivergenceLoss;

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl Loss for KLDivergenceLoss {
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
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]].max(self.epsilon);
                let target = targets[[i, j]].max(self.epsilon);
                total_loss += target * (target / pred).ln();
            }
        }
        Ok(total_loss)
    }
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]].max(self.epsilon);
                let target = targets[[i, j]].max(self.epsilon);
                grad[[i, j]] = -target / pred;
            }
        }
        Ok(grad)
    }
}
