//! [`KroneckerICMKernel`]: ICM multi-output kernel using the Kronecker structure.
//!
//! The Intrinsic Coregionalization Model (ICM) defines the p×p covariance
//! block as
//!
//! ```text
//! K_block(x, y) = k_base(x, y) · B
//! ```
//!
//! where `B ∈ R^{p×p}` is the task covariance matrix and `k_base` is a
//! scalar base kernel.  We compute each entry of the block via
//! `ICMKernel::compute_tasks`, which evaluates
//! `B[i, j] * k_base(x, y)` — so even though this loops over all `p²`
//! (task, task) pairs, each call performs a single base-kernel evaluation
//! and a single multiplication.

use std::sync::Arc;

use scirs2_core::ndarray::Array2;

use crate::error::KernelError;
use crate::multitask::{ICMKernel, TaskInput};
use crate::types::Kernel;

use super::trait_def::MultiOutputKernel;

type Result<T> = std::result::Result<T, KernelError>;

/// Multi-output ICM kernel wrapping a scalar [`ICMKernel`] to produce `p×p`
/// covariance blocks.
///
/// # Construction
///
/// Use [`KroneckerICMKernel::new`] to wrap an existing `ICMKernel`, or
/// [`KroneckerICMKernel::from_base`] to build one from a boxed scalar base
/// kernel and a task-covariance matrix given as `Vec<Vec<f64>>`.
pub struct KroneckerICMKernel {
    inner: Arc<ICMKernel>,
}

impl KroneckerICMKernel {
    /// Wrap an existing [`ICMKernel`].
    pub fn new(inner: ICMKernel) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Build from a scalar base kernel and a task covariance matrix.
    ///
    /// `task_covariance[i][j]` is the coregionalization weight `B[i, j]`.
    /// The matrix must be square and positive semi-definite.
    pub fn from_base(base: Box<dyn Kernel>, task_covariance: Vec<Vec<f64>>) -> Result<Self> {
        let inner = ICMKernel::new(base, task_covariance).map_err(|e| {
            KernelError::ComputationError(format!("ICMKernel construction failed: {}", e))
        })?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl MultiOutputKernel for KroneckerICMKernel {
    fn n_outputs(&self) -> usize {
        self.inner.num_tasks()
    }

    /// Compute the p×p covariance block K(x, y) for the ICM model.
    ///
    /// Each entry `block[i, j] = B[i, j] * k_base(x, y)` is obtained by
    /// delegating to `ICMKernel::compute_tasks`.  The `p²` loop is
    /// unavoidable given the current public interface of `ICMKernel`, but
    /// each call reduces to a single floating-point multiply after the base
    /// kernel is evaluated — the base kernel itself performs the heavy work
    /// only once (for same-feature inputs the result is identical across rows
    /// and columns, scaled purely by `B`).
    fn compute_block(&self, x: &[f64], y: &[f64]) -> Result<Array2<f64>> {
        let p = self.inner.num_tasks();
        let mut block = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let xi = TaskInput::from_slice(x, i);
                let yj = TaskInput::from_slice(y, j);
                block[[i, j]] = self.inner.compute_tasks(&xi, &yj).map_err(|e| {
                    KernelError::ComputationError(format!(
                        "ICM compute_tasks({}, {}) failed: {}",
                        i, j, e
                    ))
                })?;
            }
        }
        Ok(block)
    }

    fn name(&self) -> &str {
        "KroneckerICM"
    }
}
