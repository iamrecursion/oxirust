//! Loss trait and loss function test suite.

use crate::TrainResult;
use scirs2_core::ndarray::{Array, ArrayView, Ix2};
use std::fmt::Debug;

/// Trait for loss functions.
pub trait Loss: Debug {
    /// Compute loss value.
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64>;
    /// Compute loss gradient with respect to predictions.
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>>;
    /// Get the name of the loss function.
    fn name(&self) -> &str {
        "unknown"
    }
}
#[cfg(test)]
mod tests {
    use super::super::types::{
        BCEWithLogitsLoss, ConstraintViolationLoss, ContrastiveLoss, CrossEntropyLoss, DiceLoss,
        FocalLoss, HingeLoss, HuberLoss, KLDivergenceLoss, MseLoss, PolyLoss, RuleSatisfactionLoss,
        TripletLoss, TverskyLoss,
    };
    use super::*;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_cross_entropy_loss() {
        let loss = CrossEntropyLoss::default();
        let predictions = array![[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_mse_loss() {
        let loss = MseLoss;
        let predictions = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![[1.5, 2.5], [3.5, 4.5]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!((loss_val - 0.25).abs() < 1e-6);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_rule_satisfaction_loss() {
        let loss = RuleSatisfactionLoss::default();
        let rule_values = array![[0.9, 0.8], [0.95, 0.85]];
        let targets = array![[1.0, 1.0], [1.0, 1.0]];
        let loss_val = loss
            .compute(&rule_values.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let grad = loss
            .gradient(&rule_values.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), rule_values.shape());
    }
    #[test]
    fn test_constraint_violation_loss() {
        let loss = ConstraintViolationLoss::default();
        let constraint_values = array![[0.1, -0.1], [0.2, -0.2]];
        let targets = array![[0.0, 0.0], [0.0, 0.0]];
        let loss_val = loss
            .compute(&constraint_values.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let grad = loss
            .gradient(&constraint_values.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), constraint_values.shape());
    }
    #[test]
    fn test_focal_loss() {
        let loss = FocalLoss::default();
        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::default();
        let predictions = array![[1.0, 3.0], [2.0, 5.0]];
        let targets = array![[1.5, 2.0], [2.5, 4.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_bce_with_logits_loss() {
        let loss = BCEWithLogitsLoss;
        let logits = array![[0.5, -0.5], [1.0, -1.0]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];
        let loss_val = loss
            .compute(&logits.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&logits.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), logits.shape());
    }
    #[test]
    fn test_dice_loss() {
        let loss = DiceLoss::default();
        let predictions = array![[0.9, 0.1], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        assert!(loss_val <= 1.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_tversky_loss() {
        let loss = TverskyLoss::default();
        let predictions = array![[0.9, 0.1], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        assert!(loss_val <= 1.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
    }
    #[test]
    fn test_contrastive_loss() {
        let loss = ContrastiveLoss::default();
        let predictions = array![[0.5, 0.0], [1.5, 0.0], [0.2, 0.0]];
        let targets = array![[1.0], [0.0], [1.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
        assert!(grad[[0, 0]] > 0.0);
        assert_eq!(grad[[1, 0]], 0.0);
    }
    #[test]
    fn test_triplet_loss() {
        let loss = TripletLoss::default();
        let predictions = array![[0.5, 2.0], [1.0, 0.5], [0.3, 1.5]];
        let targets = array![[0.0], [0.0], [0.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
        assert_eq!(grad[[0, 0]], 0.0);
        assert_eq!(grad[[0, 1]], 0.0);
        assert!(grad[[1, 0]] > 0.0);
        assert!(grad[[1, 1]] < 0.0);
    }
    #[test]
    fn test_hinge_loss() {
        let loss = HingeLoss::default();
        let predictions = array![[0.5, -0.5], [2.0, -2.0]];
        let targets = array![[1.0, -1.0], [1.0, -1.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
        assert_eq!(grad[[1, 0]], 0.0);
        assert_eq!(grad[[1, 1]], 0.0);
    }
    #[test]
    fn test_kl_divergence_loss() {
        let loss = KLDivergenceLoss::default();
        let predictions = array![[0.6, 0.4], [0.7, 0.3]];
        let targets = array![[0.5, 0.5], [0.8, 0.2]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
        let identical_preds = array![[0.5, 0.5]];
        let identical_targets = array![[0.5, 0.5]];
        let identical_loss = loss
            .compute(&identical_preds.view(), &identical_targets.view())
            .expect("unwrap");
        assert!(identical_loss.abs() < 1e-6);
    }
    #[test]
    fn test_poly_loss() {
        let loss = PolyLoss::default();
        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let grad = loss
            .gradient(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert_eq!(grad.shape(), predictions.shape());
        let ce_loss = CrossEntropyLoss::default();
        let ce_val = ce_loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val >= ce_val);
    }
    #[test]
    fn test_poly_loss_custom_coefficient() {
        let loss = PolyLoss::new(2.0);
        let predictions = array![[0.8, 0.2]];
        let targets = array![[1.0, 0.0]];
        let loss_val = loss
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > 0.0);
        let loss_low_coeff = PolyLoss::new(0.5);
        let loss_val_low = loss_low_coeff
            .compute(&predictions.view(), &targets.view())
            .expect("unwrap");
        assert!(loss_val > loss_val_low);
    }
}
