//! # FocalLoss - Trait Implementations
//!
//! This module contains trait implementations for `FocalLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::FocalLoss;

impl Default for FocalLoss {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            gamma: 2.0,
            epsilon: 1e-10,
        }
    }
}

impl Loss for FocalLoss {
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
                if target > 0.5 {
                    let focal_weight = (1.0 - pred).powf(self.gamma);
                    total_loss -= self.alpha * focal_weight * pred.ln();
                } else {
                    let focal_weight = pred.powf(self.gamma);
                    total_loss -= (1.0 - self.alpha) * focal_weight * (1.0 - pred).ln();
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
        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                if target > 0.5 {
                    let focal_weight = (1.0 - pred).powf(self.gamma);
                    let d_focal = self.gamma * (1.0 - pred).powf(self.gamma - 1.0);
                    grad[[i, j]] = -self.alpha * (focal_weight / pred - d_focal * pred.ln()) / n;
                } else {
                    let focal_weight = pred.powf(self.gamma);
                    let d_focal = self.gamma * pred.powf(self.gamma - 1.0);
                    grad[[i, j]] = -(1.0 - self.alpha)
                        * (d_focal * (1.0 - pred).ln() - focal_weight / (1.0 - pred))
                        / n;
                }
            }
        }
        Ok(grad)
    }
}
