//! The [`KernelPCA`] estimator and its fitted counterpart
//! [`FittedKernelPCA`].
//!
//! This module wires together the centering utilities in
//! [`crate::kernel_pca::centering`] and the eigendecomposition in
//! [`crate::kernel_pca::eigendecomp`] into the standard scikit-learn
//! style estimator API:
//!
//! ```text
//! let model = KernelPCA::new(kernel, KernelPcaConfig::new(2));
//! let fitted = model.fit(&training_data)?;
//! let embedding = fitted.transform(&new_points)?;
//! ```
//!
//! `fit_transform` is also provided as a convenience when the training
//! data itself is to be embedded (this is how a pipeline like
//! *Gaussian KPCA for visualisation* is usually driven).

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::KernelError;
use crate::kernel_pca::centering::{center_test_kernel, double_center, KernelCenteringStats};
use crate::kernel_pca::eigendecomp::{symmetric_eigendecomp, TopKEigen};
use crate::kernel_pca::error::{KernelPcaError, KernelPcaResult};
use crate::types::Kernel;

/// Configuration for [`KernelPCA`].
///
/// Cheap to clone and `Debug`/`PartialEq`-comparable so it composes
/// cleanly inside pipelines and hyperparameter sweeps.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPcaConfig {
    /// Number of principal components to retain.
    pub n_components: usize,
    /// Whether to double-center the Gram matrix before eigendecomp.
    ///
    /// Leaving this `true` (the default) is the standard Kernel PCA
    /// behaviour; setting it to `false` lets callers who have already
    /// centered their kernel (e.g. when chaining two `kernel_pca`
    /// instances) skip the redundant step.
    pub center: bool,
}

impl KernelPcaConfig {
    /// Build a configuration requesting `n_components` components with
    /// centering enabled.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            center: true,
        }
    }

    /// Override the centering flag.
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }
}

/// Kernel-PCA estimator generic over any kernel that implements the
/// crate's [`Kernel`] trait. Typical ones are
/// [`crate::RbfKernel`], [`crate::LinearKernel`], and
/// [`crate::PolynomialKernel`], but [`crate::SymbolicKernel`] (built
/// with [`crate::KernelBuilder`]) also slots in.
///
/// `KernelPCA` is stateless — the "fitted model" is
/// [`FittedKernelPCA`], returned by [`KernelPCA::fit`] or
/// [`KernelPCA::fit_transform`].
#[derive(Clone, Debug)]
pub struct KernelPCA<K: Kernel> {
    kernel: K,
    config: KernelPcaConfig,
}

impl<K: Kernel> KernelPCA<K> {
    /// Build a new Kernel-PCA estimator from a kernel and a config.
    ///
    /// # Errors
    ///
    /// * [`KernelPcaError::InvalidInput`] when the config requests
    ///   `n_components == 0`.
    pub fn new(kernel: K, config: KernelPcaConfig) -> KernelPcaResult<Self> {
        if config.n_components == 0 {
            return Err(KernelPcaError::InvalidInput(
                "KernelPCA::new: n_components must be >= 1".to_string(),
            ));
        }
        Ok(Self { kernel, config })
    }

    /// Access the underlying kernel (useful for diagnostics).
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Access the configuration.
    pub fn config(&self) -> &KernelPcaConfig {
        &self.config
    }
}

/// A fitted Kernel-PCA model. Stores everything required to project
/// new data into the learned principal subspace: the kernel, the
/// scaled eigenvectors `alpha = v / sqrt(lambda)`, the raw
/// eigenvalues, the training points, and the centering statistics.
pub struct FittedKernelPCA<K: Kernel> {
    kernel: Box<dyn Kernel>,
    alphas: Array2<f64>,
    eigenvalues: Array1<f64>,
    training_data: Vec<Vec<f64>>,
    centering_stats: KernelCenteringStats,
    n_components: usize,
    feature_dim: usize,
    // Phantom to remember the original static kernel type for the
    // convenience accessor below.
    _marker: std::marker::PhantomData<K>,
}

impl<K: Kernel> std::fmt::Debug for FittedKernelPCA<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedKernelPCA")
            .field("kernel_name", &self.kernel.name())
            .field("n_components", &self.n_components)
            .field("feature_dim", &self.feature_dim)
            .field("n_training_points", &self.training_data.len())
            .field("eigenvalues", &self.eigenvalues)
            .finish()
    }
}

impl<K: Kernel> FittedKernelPCA<K> {
    /// Number of components kept at `fit` time.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Feature dimension expected by [`Self::transform`].
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Eigenvalues of the centered Gram matrix corresponding to the
    /// retained components, sorted in descending order.
    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.eigenvalues
    }

    /// Scaled eigenvectors `alpha_k = v_k / sqrt(lambda_k)` as an
    /// `(n_training, n_components)` matrix.
    pub fn alphas(&self) -> &Array2<f64> {
        &self.alphas
    }

    /// Borrowed view of the training data retained for projection.
    pub fn training_data(&self) -> &[Vec<f64>] {
        &self.training_data
    }

    /// Centering statistics captured at `fit` time.
    pub fn centering_stats(&self) -> &KernelCenteringStats {
        &self.centering_stats
    }

    /// Fraction of centered-Gram variance explained by each retained
    /// component. The `i`-th entry is
    /// `eigenvalues[i] / sum_j eigenvalues[j]`. Returns a zero-length
    /// array if the kept eigenvalues are (numerically) zero.
    pub fn explained_variance_ratio(&self) -> Array1<f64> {
        let total: f64 = self.eigenvalues.iter().copied().sum();
        if total <= 0.0 {
            return Array1::<f64>::zeros(self.n_components);
        }
        let mut out = Array1::<f64>::zeros(self.n_components);
        for (i, v) in self.eigenvalues.iter().enumerate() {
            out[i] = v / total;
        }
        out
    }

    /// Project new points into the learned principal subspace.
    ///
    /// The returned matrix has shape `(points.len(), n_components)`
    /// and row `i` is the embedding of `points[i]`.
    ///
    /// # Errors
    ///
    /// * [`KernelPcaError::DimensionMismatch`] if any row of `points`
    ///   has a different feature dimension than the training data.
    /// * [`KernelPcaError::InvalidInput`] if `points` is empty.
    pub fn transform(&self, points: &[Vec<f64>]) -> KernelPcaResult<Array2<f64>> {
        if points.is_empty() {
            return Err(KernelPcaError::InvalidInput(
                "FittedKernelPCA::transform: points must not be empty".to_string(),
            ));
        }
        let n_train = self.training_data.len();
        let k = self.n_components;
        let mut out = Array2::<f64>::zeros((points.len(), k));

        for (pi, point) in points.iter().enumerate() {
            if point.len() != self.feature_dim {
                return Err(KernelPcaError::DimensionMismatch {
                    expected: self.feature_dim,
                    got: point.len(),
                    context: format!("FittedKernelPCA::transform: points[{}]", pi),
                });
            }

            // Row of test-time kernel evaluations against the training set.
            let mut k_test = vec![0.0f64; n_train];
            for (ti, train_row) in self.training_data.iter().enumerate() {
                k_test[ti] = self
                    .kernel
                    .compute(point, train_row)
                    .map_err(KernelPcaError::from_kernel)?;
            }

            let centered: Array1<f64> = center_test_kernel(&k_test, &self.centering_stats)?;

            // Embedding: z_c = sum_i centered[i] * alpha[i, c].
            for c in 0..k {
                let mut acc = 0.0f64;
                for i in 0..n_train {
                    acc += centered[i] * self.alphas[(i, c)];
                }
                out[(pi, c)] = acc;
            }
        }

        Ok(out)
    }
}

// KernelPCA is not a kernel itself — this impl simply provides a
// descriptive error when someone tries to use it as one.
impl<K> Kernel for KernelPCA<K>
where
    K: Kernel,
{
    fn compute(&self, _x: &[f64], _y: &[f64]) -> crate::error::Result<f64> {
        // KernelPCA is not itself a kernel; reject any attempt to
        // treat it as one. We only implement this trait so that the
        // inherent `fit` can call `self.kernel.compute_matrix` without
        // paying for another constraint on top.
        Err(KernelError::InvalidParameter {
            parameter: "KernelPCA".to_string(),
            value: "not a kernel".to_string(),
            reason: "KernelPCA is an estimator, not a Kernel; use fit/transform instead"
                .to_string(),
        })
    }

    fn name(&self) -> &str {
        "KernelPCA"
    }

    fn is_psd(&self) -> bool {
        false
    }
}

/// The `clone_box` helper on `Kernel` used inside `fit`. Requires
/// `Clone + 'static` which every crate kernel satisfies. We wire it in
/// via a blanket extension trait on top of the public `Kernel` trait
/// to avoid modifying the trait itself.
pub(crate) trait KernelCloneExt {
    fn clone_box(&self) -> Box<dyn Kernel>;
}

impl<K: Kernel + Clone + 'static> KernelCloneExt for K {
    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// Re-route `self.kernel.clone_box()` inside `fit` to the extension
// trait: a blanket impl would shadow the helper trait defined earlier
// in this file for trait objects, so we scope the helper here by
// name.

impl<K: Kernel + Clone + 'static> KernelPCA<K> {
    /// Preferred constructor when the kernel is `Clone + 'static` —
    /// identical signature to [`KernelPCA::new`] but re-exposed here so
    /// that auto-derefs pick this up for the common case.
    pub fn build(kernel: K, config: KernelPcaConfig) -> KernelPcaResult<Self> {
        Self::new(kernel, config)
    }

    /// Fit the model on `training` — compute the Gram matrix, double
    /// center it, eigendecompose, and cache the top-`n_components`
    /// components for later projection.
    ///
    /// # Errors
    ///
    /// * [`KernelPcaError::InvalidInput`] for empty or ragged training
    ///   sets.
    /// * [`KernelPcaError::EigendecompositionFailed`] if the underlying
    ///   solver fails.
    /// * [`KernelPcaError::InsufficientComponents`] if the kernel matrix
    ///   does not have enough positive eigenvalues.
    pub fn fit(&self, training: &[Vec<f64>]) -> KernelPcaResult<FittedKernelPCA<K>> {
        let n = training.len();
        if n == 0 {
            return Err(KernelPcaError::InvalidInput(
                "KernelPCA::fit: training set must not be empty".to_string(),
            ));
        }
        let d = training[0].len();
        if d == 0 {
            return Err(KernelPcaError::InvalidInput(
                "KernelPCA::fit: feature dimension must be >= 1".to_string(),
            ));
        }
        for (i, row) in training.iter().enumerate() {
            if row.len() != d {
                return Err(KernelPcaError::InvalidInput(format!(
                    "KernelPCA::fit: training[{}] has {} features (expected {})",
                    i,
                    row.len(),
                    d
                )));
            }
        }
        if self.config.n_components > n {
            return Err(KernelPcaError::InvalidInput(format!(
                "KernelPCA::fit: n_components ({}) cannot exceed training size ({})",
                self.config.n_components, n
            )));
        }

        // Compute the raw Gram matrix via the kernel's matrix routine;
        // symmetrise to absorb any per-entry rounding drift.
        let gram_rows = self
            .kernel
            .compute_matrix(training)
            .map_err(KernelPcaError::from_kernel)?;
        let mut gram = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            if gram_rows[i].len() != n {
                return Err(KernelPcaError::EigendecompositionFailed(format!(
                    "kernel.compute_matrix returned ragged row {} (len {}, expected {})",
                    i,
                    gram_rows[i].len(),
                    n
                )));
            }
            for j in 0..n {
                gram[(i, j)] = gram_rows[i][j];
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (gram[(i, j)] + gram[(j, i)]);
                gram[(i, j)] = avg;
                gram[(j, i)] = avg;
            }
        }

        // Optional double-centering.
        let (centered, centering_stats) = if self.config.center {
            double_center(&gram)?
        } else {
            // No centering requested — synthesise null stats so that
            // `transform` can still apply the same zero-valued offsets.
            (
                gram.clone(),
                KernelCenteringStats {
                    row_means: Array1::<f64>::zeros(n),
                    grand_mean: 0.0,
                },
            )
        };

        // Eigendecompose.
        let TopKEigen {
            eigenvalues,
            eigenvectors,
        } = symmetric_eigendecomp(&centered, self.config.n_components)?;

        // Normalise each eigenvector `v_k` to `alpha_k = v_k / sqrt(lambda_k)`
        // so that projections read off as `K_c(x, ·) dot alpha_k`. This is
        // the standard KPCA scaling (Scholkopf et al. 1998, eq. (4.3)).
        let k = self.config.n_components;
        let mut alphas = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            let lam = eigenvalues[c];
            if lam <= 0.0 {
                // symmetric_eigendecomp already filters on POSITIVITY_FLOOR,
                // so this branch is defensive.
                return Err(KernelPcaError::InsufficientComponents {
                    requested: k,
                    available: c,
                });
            }
            let scale = 1.0 / lam.sqrt();
            for r in 0..n {
                alphas[(r, c)] = eigenvectors[(r, c)] * scale;
            }
        }

        Ok(FittedKernelPCA {
            kernel: KernelCloneExt::clone_box(&self.kernel),
            alphas,
            eigenvalues,
            training_data: training.to_vec(),
            centering_stats,
            n_components: k,
            feature_dim: d,
            _marker: std::marker::PhantomData,
        })
    }

    /// Convenience: fit on `training` and immediately project those
    /// same points.
    pub fn fit_transform(
        &self,
        training: &[Vec<f64>],
    ) -> KernelPcaResult<(FittedKernelPCA<K>, Array2<f64>)> {
        let fitted = self.fit(training)?;
        let projected = fitted.transform(training)?;
        Ok((fitted, projected))
    }
}

// Users with a bespoke kernel that is *not* `Clone + 'static` should
// wrap it in an `Arc` (which the crate's existing `SymbolicKernel`
// already does internally).
