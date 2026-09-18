//! # DiceLoss - Trait Implementations
//!
//! This module contains trait implementations for `DiceLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::DiceLoss;

impl Default for DiceLoss {
    fn default() -> Self {
        Self { smooth: 1.0 }
    }
}

impl Loss for DiceLoss {
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
        let mut intersection = 0.0;
        let mut pred_sum = 0.0;
        let mut target_sum = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];
                intersection += pred * target;
                pred_sum += pred;
                target_sum += target;
            }
        }
        let dice_coef = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth);
        Ok(1.0 - dice_coef)
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
        let mut intersection = 0.0;
        let mut pred_sum = 0.0;
        let mut target_sum = 0.0;
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                intersection += predictions[[i, j]] * targets[[i, j]];
                pred_sum += predictions[[i, j]];
                target_sum += targets[[i, j]];
            }
        }
        let denominator = pred_sum + target_sum + self.smooth;
        let numerator = 2.0 * intersection + self.smooth;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let target = targets[[i, j]];
                grad[[i, j]] =
                    -2.0 * (target * denominator - numerator) / (denominator * denominator);
            }
        }
        Ok(grad)
    }
}
