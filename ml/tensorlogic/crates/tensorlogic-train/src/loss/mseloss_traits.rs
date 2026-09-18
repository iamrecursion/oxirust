//! # MseLoss - Trait Implementations
//!
//! This module contains trait implementations for `MseLoss`.
//!
//! ## Implemented Traits
//!
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::MseLoss;

impl Loss for MseLoss {
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
                let diff = predictions[[i, j]] - targets[[i, j]];
                total_loss += diff * diff;
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
                grad[[i, j]] = 2.0 * (predictions[[i, j]] - targets[[i, j]]) / n;
            }
        }
        Ok(grad)
    }
}
