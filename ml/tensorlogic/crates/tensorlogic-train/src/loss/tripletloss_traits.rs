//! # TripletLoss - Trait Implementations
//!
//! This module contains trait implementations for `TripletLoss`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Loss`

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

use super::functions::Loss;
use super::types::TripletLoss;

impl Default for TripletLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for TripletLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        _targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.ncols() != 2 {
            return Err(TrainError::LossError(format!(
                "TripletLoss expects predictions shape [N, 2] (pos_dist, neg_dist), got {:?}",
                predictions.shape()
            )));
        }
        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;
        for i in 0..predictions.nrows() {
            let pos_distance = predictions[[i, 0]];
            let neg_distance = predictions[[i, 1]];
            let loss = (pos_distance - neg_distance + self.margin).max(0.0);
            total_loss += loss;
        }
        Ok(total_loss / n)
    }
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        _targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        let n = predictions.nrows() as f64;
        for i in 0..predictions.nrows() {
            let pos_distance = predictions[[i, 0]];
            let neg_distance = predictions[[i, 1]];
            if pos_distance - neg_distance + self.margin > 0.0 {
                grad[[i, 0]] = 1.0 / n;
                grad[[i, 1]] = -1.0 / n;
            }
        }
        Ok(grad)
    }
}
