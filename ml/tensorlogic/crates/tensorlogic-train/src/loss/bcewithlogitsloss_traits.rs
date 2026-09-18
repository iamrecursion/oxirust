//! # BCEWithLogitsLoss - Trait Implementations
//!
//! This module contains trait implementations for `BCEWithLogitsLoss`.
//!
//! ## Implemented Traits
//!
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::BCEWithLogitsLoss;

impl Loss for BCEWithLogitsLoss {
    fn compute(
        &self,
        logits: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if logits.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: logits {:?} vs targets {:?}",
                logits.shape(),
                targets.shape()
            )));
        }
        let n = logits.len() as f64;
        let mut total_loss = 0.0;
        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                let logit = logits[[i, j]];
                let target = targets[[i, j]];
                let max_val = logit.max(0.0);
                total_loss += max_val - logit * target + (1.0 + (-logit.abs()).exp()).ln();
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        logits: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if logits.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: logits {:?} vs targets {:?}",
                logits.shape(),
                targets.shape()
            )));
        }
        let n = logits.len() as f64;
        let mut grad = Array::zeros(logits.raw_dim());
        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                let logit = logits[[i, j]];
                let target = targets[[i, j]];
                let sigmoid = 1.0 / (1.0 + (-logit).exp());
                grad[[i, j]] = (sigmoid - target) / n;
            }
        }
        Ok(grad)
    }
}
