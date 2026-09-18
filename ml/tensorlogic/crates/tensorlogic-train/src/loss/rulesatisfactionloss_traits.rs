//! # RuleSatisfactionLoss - Trait Implementations
//!
//! This module contains trait implementations for `RuleSatisfactionLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::RuleSatisfactionLoss;

impl Default for RuleSatisfactionLoss {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl Loss for RuleSatisfactionLoss {
    fn compute(
        &self,
        rule_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if rule_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: rule_values {:?} vs targets {:?}",
                rule_values.shape(),
                targets.shape()
            )));
        }
        let n = rule_values.len() as f64;
        let mut total_loss = 0.0;
        for i in 0..rule_values.nrows() {
            for j in 0..rule_values.ncols() {
                let diff = targets[[i, j]] - rule_values[[i, j]];
                total_loss += (diff / self.temperature).powi(2);
            }
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        rule_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if rule_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: rule_values {:?} vs targets {:?}",
                rule_values.shape(),
                targets.shape()
            )));
        }
        let n = rule_values.len() as f64;
        let mut grad = Array::zeros(rule_values.raw_dim());
        for i in 0..rule_values.nrows() {
            for j in 0..rule_values.ncols() {
                let diff = targets[[i, j]] - rule_values[[i, j]];
                grad[[i, j]] = -2.0 * diff / (self.temperature * self.temperature * n);
            }
        }
        Ok(grad)
    }
}
