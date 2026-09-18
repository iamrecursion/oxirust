//! # ConstraintViolationLoss - Trait Implementations
//!
//! This module contains trait implementations for `ConstraintViolationLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::ConstraintViolationLoss;

impl Default for ConstraintViolationLoss {
    fn default() -> Self {
        Self {
            penalty_weight: 10.0,
        }
    }
}

impl Loss for ConstraintViolationLoss {
    fn compute(
        &self,
        constraint_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if constraint_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: constraint_values {:?} vs targets {:?}",
                constraint_values.shape(),
                targets.shape()
            )));
        }
        let n = constraint_values.len() as f64;
        let mut total_loss = 0.0;
        for i in 0..constraint_values.nrows() {
            for j in 0..constraint_values.ncols() {
                let violation = (constraint_values[[i, j]] - targets[[i, j]]).max(0.0);
                total_loss += self.penalty_weight * violation * violation;
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        constraint_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if constraint_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: constraint_values {:?} vs targets {:?}",
                constraint_values.shape(),
                targets.shape()
            )));
        }
        let n = constraint_values.len() as f64;
        let mut grad = Array::zeros(constraint_values.raw_dim());
        for i in 0..constraint_values.nrows() {
            for j in 0..constraint_values.ncols() {
                let violation = constraint_values[[i, j]] - targets[[i, j]];
                if violation > 0.0 {
                    grad[[i, j]] = 2.0 * self.penalty_weight * violation / n;
                }
            }
        }
        Ok(grad)
    }
}
