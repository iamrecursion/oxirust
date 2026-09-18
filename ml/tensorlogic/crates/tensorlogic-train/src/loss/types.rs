//! Loss function type definitions.

use crate::TrainResult;
use scirs2_core::ndarray::{Array, ArrayView, Ix2};
use std::fmt::Debug;

use super::functions::Loss;

/// Dice loss for segmentation tasks.
#[derive(Debug, Clone)]
pub struct DiceLoss {
    /// Smoothing factor to avoid division by zero.
    pub smooth: f64,
}
/// Contrastive loss for metric learning.
/// Used to learn embeddings where similar pairs are close and dissimilar pairs are far apart.
#[derive(Debug, Clone)]
pub struct ContrastiveLoss {
    /// Margin for dissimilar pairs.
    pub margin: f64,
}
/// Rule satisfaction loss - measures how well rules are satisfied.
#[derive(Debug, Clone)]
pub struct RuleSatisfactionLoss {
    /// Temperature for soft satisfaction.
    pub temperature: f64,
}
/// Binary cross-entropy with logits loss (numerically stable).
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss;
/// Configuration for loss functions.
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Weight for supervised loss component.
    pub supervised_weight: f64,
    /// Weight for constraint violation loss component.
    pub constraint_weight: f64,
    /// Weight for rule satisfaction loss component.
    pub rule_weight: f64,
    /// Temperature for soft constraint penalties.
    pub temperature: f64,
}
/// Tversky loss (generalization of Dice loss).
/// Useful for handling class imbalance in segmentation.
#[derive(Debug, Clone)]
pub struct TverskyLoss {
    /// Alpha parameter (weight for false positives).
    pub alpha: f64,
    /// Beta parameter (weight for false negatives).
    pub beta: f64,
    /// Smoothing factor.
    pub smooth: f64,
}
/// Mean squared error loss for regression.
#[derive(Debug, Clone, Default)]
pub struct MseLoss;
/// Triplet loss for metric learning.
/// Learns embeddings where anchor-positive distance < anchor-negative distance + margin.
#[derive(Debug, Clone)]
pub struct TripletLoss {
    /// Margin between positive and negative distances.
    pub margin: f64,
}
/// Focal loss for addressing class imbalance.
/// Reference: Lin et al., "Focal Loss for Dense Object Detection"
#[derive(Debug, Clone)]
pub struct FocalLoss {
    /// Alpha weighting factor for positive class (range: [0, 1]).
    pub alpha: f64,
    /// Gamma focusing parameter (typically 2.0).
    pub gamma: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}
/// Hinge loss for maximum-margin classification (SVM-style).
#[derive(Debug, Clone)]
pub struct HingeLoss {
    /// Margin for classification.
    pub margin: f64,
}
/// Poly Loss - Polynomial Expansion of Cross-Entropy Loss.
///
/// Paper: "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions" (Leng et al., 2022)
/// <https://arxiv.org/abs/2204.12511>
///
/// PolyLoss adds polynomial terms to cross-entropy to provide better gradient flow
/// for well-classified examples. It helps with:
/// - Label noise robustness
/// - Improved generalization
/// - Better handling of class imbalance
///
/// The loss is defined as:
/// L_poly = CE + ε₁(1 - p_t) + ε₂(1 - p_t)² + ... + εⱼ(1 - p_t)^j
///
/// where p_t is the predicted probability of the target class, and εⱼ are polynomial coefficients.
/// In practice, Poly-1 (j=1) is most commonly used.
#[derive(Debug, Clone)]
pub struct PolyLoss {
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Polynomial coefficient (typically between 0.5 and 2.0)
    pub poly_coeff: f64,
}
impl PolyLoss {
    /// Create a new Poly Loss with custom coefficient.
    pub fn new(poly_coeff: f64) -> Self {
        Self {
            epsilon: 1e-10,
            poly_coeff,
        }
    }
}
/// Logical loss combining multiple objectives.
#[derive(Debug)]
pub struct LogicalLoss {
    /// Configuration.
    pub config: LossConfig,
    /// Supervised loss component.
    pub supervised_loss: Box<dyn Loss>,
    /// Rule satisfaction components.
    pub rule_losses: Vec<Box<dyn Loss>>,
    /// Constraint violation components.
    pub constraint_losses: Vec<Box<dyn Loss>>,
}
impl LogicalLoss {
    /// Create a new logical loss.
    pub fn new(
        config: LossConfig,
        supervised_loss: Box<dyn Loss>,
        rule_losses: Vec<Box<dyn Loss>>,
        constraint_losses: Vec<Box<dyn Loss>>,
    ) -> Self {
        Self {
            config,
            supervised_loss,
            rule_losses,
            constraint_losses,
        }
    }
    /// Compute total loss with all components.
    pub fn compute_total(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
        rule_values: &[ArrayView<f64, Ix2>],
        constraint_values: &[ArrayView<f64, Ix2>],
    ) -> TrainResult<f64> {
        let mut total = 0.0;
        let supervised = self.supervised_loss.compute(predictions, targets)?;
        total += self.config.supervised_weight * supervised;
        if !rule_values.is_empty() && !self.rule_losses.is_empty() {
            let expected_true = Array::ones((rule_values[0].nrows(), rule_values[0].ncols()));
            let expected_true_view = expected_true.view();
            for (rule_val, rule_loss) in rule_values.iter().zip(self.rule_losses.iter()) {
                let rule_loss_val = rule_loss.compute(rule_val, &expected_true_view)?;
                total += self.config.rule_weight * rule_loss_val;
            }
        }
        if !constraint_values.is_empty() && !self.constraint_losses.is_empty() {
            let expected_zero =
                Array::zeros((constraint_values[0].nrows(), constraint_values[0].ncols()));
            let expected_zero_view = expected_zero.view();
            for (constraint_val, constraint_loss) in
                constraint_values.iter().zip(self.constraint_losses.iter())
            {
                let constraint_loss_val =
                    constraint_loss.compute(constraint_val, &expected_zero_view)?;
                total += self.config.constraint_weight * constraint_loss_val;
            }
        }
        Ok(total)
    }
}
/// Constraint violation loss - penalizes constraint violations.
#[derive(Debug, Clone)]
pub struct ConstraintViolationLoss {
    /// Penalty weight for violations.
    pub penalty_weight: f64,
}
/// Cross-entropy loss for classification.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}
/// Kullback-Leibler Divergence loss.
/// Measures how one probability distribution diverges from a reference distribution.
#[derive(Debug, Clone)]
pub struct KLDivergenceLoss {
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}
/// Huber loss for robust regression.
#[derive(Debug, Clone)]
pub struct HuberLoss {
    /// Delta threshold for switching between L1 and L2.
    pub delta: f64,
}
