//! # PolyLoss - Trait Implementations
//!
//! This module contains trait implementations for `PolyLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::PolyLoss;

impl Default for PolyLoss {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            poly_coeff: 1.0,
        }
    }
}

impl Loss for PolyLoss {
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
                let ce = -target * pred.ln();
                let poly_term = if target > 0.5 {
                    self.poly_coeff * (1.0 - pred)
                } else {
                    self.poly_coeff * pred
                };
                total_loss += ce + poly_term;
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                let ce_grad = -target / pred;
                let poly_grad = if target > 0.5 {
                    -self.poly_coeff
                } else {
                    self.poly_coeff
                };
                grad[[i, j]] = (ce_grad + poly_grad) / n;
            }
        }
        Ok(grad)
    }
    fn name(&self) -> &str {
        "poly_loss"
    }
}
