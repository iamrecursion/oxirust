//! C-SVM Classification via SMO.
//!
//! Implements binary and multi-class C-Support Vector Classification using the
//! Sequential Minimal Optimization solver from [`super::smo`].
//!
//! ## Binary Classification
//!
//! Given training data (x_i, y_i) with y_i ∈ {-1, +1}, binary SVC solves the
//! dual problem and produces a decision function:
//!
//! ```text
//! f(x) = Σ_i α_i y_i K(x_i, x) - b
//! ```
//!
//! The predicted class is sign(f(x)).
//!
//! ## Multi-class Classification (One-vs-Rest)
//!
//! For `k` classes, `k` binary classifiers are trained. Each binary classifier
//! treats one class as +1 and all others as -1. At prediction time, the class
//! with the highest decision function score wins.
//!
//! ## References
//!
//! - Boser, B.E., Guyon, I.M., Vapnik, V.N. (1992). A training algorithm for
//!   optimal margin classifiers. COLT.
//! - Cortes, C., Vapnik, V. (1995). Support-vector networks. Machine Learning 20(3).

use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::smo::{smo_svc, SmoConfig};

// ─── Binary Classifier ───────────────────────────────────────────────────────

/// A fitted binary SVC classifier for labels ±1.
///
/// Stores only the support vectors (training points with α > threshold) and
/// their signed dual coefficients (α_i * y_i).
pub(super) struct SvcFittedBinary {
    /// Support vectors (subset of training inputs where α_i > 1e-8).
    pub(super) support_vectors: Vec<Vec<f64>>,
    /// Signed dual coefficients α_i * y_i for each support vector.
    pub(super) support_alphas: Vec<f64>,
    /// Bias / threshold term b.
    pub(super) bias: f64,
    /// The kernel used during training.
    kernel: Arc<dyn Kernel>,
    /// Original positive class label (for OvR multi-class bookkeeping).
    pub(super) positive_label: i32,
}

impl SvcFittedBinary {
    /// Compute the unclamped SVM decision function value at `x`:
    ///   f(x) = Σ_i (α_i y_i) K(sv_i, x) - b
    ///
    /// A positive value means the sample is predicted to belong to the positive
    /// class; negative means the negative class.
    pub(super) fn decision_function(&self, x: &[f64]) -> Result<f64> {
        let mut score = 0.0_f64;
        for (sv, &coef) in self.support_vectors.iter().zip(self.support_alphas.iter()) {
            score += coef * self.kernel.compute(sv, x)?;
        }
        Ok(score - self.bias)
    }

    /// Predict the binary label (+1 or -1) for `x`.
    #[allow(dead_code)]
    pub(super) fn predict(&self, x: &[f64]) -> Result<i32> {
        let df = self.decision_function(x)?;
        Ok(if df >= 0.0 { 1 } else { -1 })
    }

    /// Number of support vectors (active training examples).
    pub(super) fn num_support_vectors(&self) -> usize {
        self.support_vectors.len()
    }
}

// ─── Unfitted Model ──────────────────────────────────────────────────────────

/// Unfitted C-SVM classification model.
///
/// Call [`SvcModel::fit`] to obtain an [`SvcFitted`] that can make predictions.
pub struct SvcModel {
    /// Kernel function shared between all binary sub-classifiers.
    kernel: Arc<dyn Kernel>,
    /// SMO solver configuration (C, tolerance, max iterations).
    config: SmoConfig,
}

impl std::fmt::Debug for SvcModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvcModel")
            .field("kernel", &self.kernel.name())
            .field("config", &self.config)
            .finish()
    }
}

impl SvcModel {
    /// Create a new SVC model with the given kernel and regularization parameter C.
    ///
    /// # Arguments
    ///
    /// * `kernel` – Any `Arc<dyn Kernel>`. Using a PSD kernel guarantees SMO convergence.
    /// * `c`      – Regularization parameter (C > 0). Larger C means less regularization
    ///   (harder margin).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidParameter`] if `c ≤ 0`.
    pub fn new(kernel: Arc<dyn Kernel>, c: f64) -> Result<Self> {
        if c <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "C".to_string(),
                value: c.to_string(),
                reason: "regularization parameter C must be strictly positive".to_string(),
            });
        }
        Ok(Self {
            kernel,
            config: SmoConfig {
                c,
                ..SmoConfig::default()
            },
        })
    }

    /// Create a new SVC model with full solver configuration.
    pub fn new_with_config(kernel: Arc<dyn Kernel>, config: SmoConfig) -> Result<Self> {
        if config.c <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "C".to_string(),
                value: config.c.to_string(),
                reason: "regularization parameter C must be strictly positive".to_string(),
            });
        }
        Ok(Self { kernel, config })
    }

    /// Fit the SVC model to training data.
    ///
    /// Automatically detects the number of unique classes and dispatches to
    /// binary or one-vs-rest multi-class training.
    ///
    /// # Arguments
    ///
    /// * `x` – Training inputs (N × d), all vectors must have the same dimension.
    /// * `y` – Integer class labels, length N.
    ///
    /// # Errors
    ///
    /// - [`KernelError::DimensionMismatch`] – empty data or inconsistent dimensions.
    /// - [`KernelError::InvalidParameter`]  – fewer than 2 unique classes.
    /// - [`KernelError::ComputationError`]  – SMO did not converge.
    pub fn fit(&self, x: &[Vec<f64>], y: &[i32]) -> Result<SvcFitted> {
        let n = x.len();
        if n == 0 {
            return Err(KernelError::DimensionMismatch {
                expected: vec![1],
                got: vec![0],
                context: "SvcModel::fit: training set cannot be empty".to_string(),
            });
        }
        if y.len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n],
                got: vec![y.len()],
                context: "SvcModel::fit: y must have same length as x".to_string(),
            });
        }

        // Collect and sort unique class labels.
        let mut classes: Vec<i32> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(KernelError::InvalidParameter {
                parameter: "y".to_string(),
                value: format!("{:?}", classes),
                reason: "at least 2 distinct class labels are required for SVC".to_string(),
            });
        }

        if classes.len() == 2 {
            // Binary case: map the two classes to ±1 and train one binary SVC.
            let neg_class = classes[0];
            let pos_class = classes[1];
            let y_binary: Vec<f64> = y
                .iter()
                .map(|&yi| if yi == pos_class { 1.0 } else { -1.0 })
                .collect();

            let binary = self.fit_binary(x, &y_binary, pos_class)?;
            Ok(SvcFitted {
                // Expose the top-level fields for binary convenience.
                support_vectors: binary.support_vectors.clone(),
                support_alphas: binary.support_alphas.clone(),
                bias: binary.bias,
                kernel: Arc::clone(&self.kernel),
                mode: SvcMode::Binary {
                    classifier: binary,
                    neg_class,
                    pos_class,
                },
            })
        } else {
            // Multi-class: one-vs-rest (OvR) strategy.
            let mut classifiers = Vec::with_capacity(classes.len());
            for &pos_class in &classes {
                let y_ovr: Vec<f64> = y
                    .iter()
                    .map(|&yi| if yi == pos_class { 1.0 } else { -1.0 })
                    .collect();
                let binary = self.fit_binary(x, &y_ovr, pos_class)?;
                classifiers.push(binary);
            }

            // Convenience fields: aggregate all support vectors (may contain duplicates).
            let all_sv: Vec<Vec<f64>> = classifiers
                .iter()
                .flat_map(|c| c.support_vectors.iter().cloned())
                .collect();
            let all_alphas: Vec<f64> = classifiers
                .iter()
                .flat_map(|c| c.support_alphas.iter().copied())
                .collect();

            Ok(SvcFitted {
                support_vectors: all_sv,
                support_alphas: all_alphas,
                bias: 0.0, // not meaningful for multi-class
                kernel: Arc::clone(&self.kernel),
                mode: SvcMode::MultiClass { classifiers },
            })
        }
    }

    /// Train a single binary SVC on data with labels in {-1, +1}.
    fn fit_binary(
        &self,
        x: &[Vec<f64>],
        y_binary: &[f64],
        positive_label: i32,
    ) -> Result<SvcFittedBinary> {
        let (alpha, b) = smo_svc(x, y_binary, &self.kernel, &self.config)?;

        // Keep only support vectors (α_i > threshold).
        let sv_threshold = 1e-8 * self.config.c;
        let mut support_vectors = Vec::new();
        let mut support_alphas = Vec::new();

        for (i, &a) in alpha.iter().enumerate() {
            if a > sv_threshold {
                support_vectors.push(x[i].clone());
                // Store signed coefficient α_i * y_i.
                support_alphas.push(a * y_binary[i]);
            }
        }

        Ok(SvcFittedBinary {
            support_vectors,
            support_alphas,
            bias: b,
            kernel: Arc::clone(&self.kernel),
            positive_label,
        })
    }
}

// ─── Fitted Model ────────────────────────────────────────────────────────────

/// Internal representation of the fitted mode.
enum SvcMode {
    Binary {
        classifier: SvcFittedBinary,
        neg_class: i32,
        pos_class: i32,
    },
    MultiClass {
        classifiers: Vec<SvcFittedBinary>,
    },
}

/// Fitted C-SVM classification model supporting prediction.
pub struct SvcFitted {
    /// Support vectors (flat list; for binary this is the binary SV set;
    /// for multiclass it aggregates all OvR classifiers).
    pub support_vectors: Vec<Vec<f64>>,
    /// Signed dual coefficients (α_i * y_i) for each entry in `support_vectors`.
    pub support_alphas: Vec<f64>,
    /// Bias term (meaningful only for binary SVC; set to 0.0 for multi-class).
    pub bias: f64,
    /// Shared kernel (kept for potential future use: e.g. online learning, refit).
    #[allow(dead_code)]
    kernel: Arc<dyn Kernel>,
    /// Internal mode-specific data.
    mode: SvcMode,
}

impl std::fmt::Debug for SvcFitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvcFitted")
            .field("num_support_vectors", &self.support_vectors.len())
            .field("bias", &self.bias)
            .finish()
    }
}

impl SvcFitted {
    /// Compute the unclamped decision function value at `x` (binary SVC only).
    ///
    /// Returns `f(x) = Σ_i α_i y_i K(sv_i, x) - b`.
    /// A positive value indicates the positive class.
    ///
    /// For multi-class models, use [`SvcFitted::decision_scores`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::ComputationError`] if this is a multi-class model.
    pub fn decision_function(&self, x: &[f64]) -> Result<f64> {
        match &self.mode {
            SvcMode::Binary { classifier, .. } => classifier.decision_function(x),
            SvcMode::MultiClass { .. } => Err(KernelError::ComputationError(
                "decision_function is not defined for multi-class SVC; \
                 use decision_scores() to get per-class OvR scores"
                    .to_string(),
            )),
        }
    }

    /// Compute OvR decision scores for all classes (multi-class models).
    ///
    /// Returns a `Vec<(i32, f64)>` pairing each class label with its OvR decision
    /// function value. Higher score = more likely.
    pub fn decision_scores(&self, x: &[f64]) -> Result<Vec<(i32, f64)>> {
        match &self.mode {
            SvcMode::Binary {
                classifier,
                pos_class,
                neg_class,
            } => {
                let df = classifier.decision_function(x)?;
                Ok(vec![(*neg_class, -df), (*pos_class, df)])
            }
            SvcMode::MultiClass { classifiers } => {
                let mut scores = Vec::with_capacity(classifiers.len());
                for clf in classifiers {
                    let df = clf.decision_function(x)?;
                    scores.push((clf.positive_label, df));
                }
                Ok(scores)
            }
        }
    }

    /// Predict the class label for a single test input `x`.
    ///
    /// - Binary: returns the positive or negative class label based on sign of f(x).
    /// - Multi-class (OvR): returns the class label with the highest decision score.
    pub fn predict(&self, x: &[f64]) -> Result<i32> {
        match &self.mode {
            SvcMode::Binary {
                classifier,
                neg_class,
                pos_class,
            } => {
                let df = classifier.decision_function(x)?;
                Ok(if df >= 0.0 { *pos_class } else { *neg_class })
            }
            SvcMode::MultiClass { classifiers } => {
                let mut best_label = classifiers[0].positive_label;
                let mut best_score = f64::NEG_INFINITY;
                for clf in classifiers {
                    let df = clf.decision_function(x)?;
                    if df > best_score {
                        best_score = df;
                        best_label = clf.positive_label;
                    }
                }
                Ok(best_label)
            }
        }
    }

    /// Predict class labels for a batch of test inputs.
    ///
    /// Returns a `Vec<i32>` of length `x.len()`.
    pub fn predict_batch(&self, x: &[Vec<f64>]) -> Result<Vec<i32>> {
        x.iter().map(|xi| self.predict(xi)).collect()
    }

    /// Number of support vectors.
    ///
    /// For binary SVC: count of training points with α > threshold.
    /// For multi-class OvR: sum over all binary classifiers (may count a point
    /// multiple times if it is a support vector in more than one sub-classifier).
    pub fn num_support_vectors(&self) -> usize {
        match &self.mode {
            SvcMode::Binary { classifier, .. } => classifier.num_support_vectors(),
            SvcMode::MultiClass { classifiers } => {
                classifiers.iter().map(|c| c.num_support_vectors()).sum()
            }
        }
    }

    /// Check whether this is a binary classifier.
    pub fn is_binary(&self) -> bool {
        matches!(&self.mode, SvcMode::Binary { .. })
    }

    /// Expose a reference to the underlying binary classifier's support vectors
    /// and coefficients for KKT verification (test helpers).
    pub fn binary_support_vectors(&self) -> Option<&[Vec<f64>]> {
        match &self.mode {
            SvcMode::Binary { classifier, .. } => Some(&classifier.support_vectors),
            _ => None,
        }
    }
}
