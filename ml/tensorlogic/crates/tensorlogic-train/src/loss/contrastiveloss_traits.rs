//! # ContrastiveLoss - Trait Implementations
//!
//! This module contains trait implementations for `ContrastiveLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::ContrastiveLoss;

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for ContrastiveLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.ncols() != 2 || targets.ncols() != 1 {
            return Err(
                TrainError::LossError(
                    format!(
                        "ContrastiveLoss expects predictions shape [N, 2] (distances) and targets shape [N, 1] (labels), got {:?} and {:?}",
                        predictions.shape(), targets.shape()
                    ),
                ),
            );
        }
        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;
        for i in 0..predictions.nrows() {
            let distance = predictions[[i, 0]];
            let label = targets[[i, 0]];
            if label > 0.5 {
                total_loss += distance * distance;
            } else {
                total_loss += (self.margin - distance).max(0.0).powi(2);
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
            let distance = predictions[[i, 0]];
            let label = targets[[i, 0]];
            if label > 0.5 {
                grad[[i, 0]] = 2.0 * distance / n;
            } else {
                if distance < self.margin {
                    grad[[i, 0]] = -2.0 * (self.margin - distance) / n;
                }
            }
        }
        Ok(grad)
    }
}
