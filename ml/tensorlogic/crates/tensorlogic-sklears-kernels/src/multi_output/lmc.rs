//! [`KroneckerLMCKernel`]: LMC multi-output kernel using the Kronecker structure.
//!
//! The Linear Model of Coregionalization (LMC) generalises the ICM model by
//! summing over Q latent processes:
//!
//! ```text
//! K_block(x, y) = Σ_q B_q[i, j] * k_q(x, y)
//! ```
//!
//! where each `(B_q, k_q)` pair is an independent ICM-like component.  We
//! compute each entry of the block by delegating to
//! `LMCKernel::compute_tasks`, which internally performs this summation.

use std::sync::Arc;

use scirs2_core::ndarray::Array2;

use crate::error::KernelError;
use crate::multitask::{LMCKernel, TaskInput};

use super::trait_def::MultiOutputKernel;

type Result<T> = std::result::Result<T, KernelError>;

/// Multi-output LMC kernel wrapping a scalar [`LMCKernel`] to produce `p×p`
/// covariance blocks.
///
/// # Construction
///
/// Use [`KroneckerLMCKernel::new`] to wrap an existing `LMCKernel`.
///
/// Build the inner `LMCKernel` via `LMCKernel::new(num_tasks)` followed by
/// repeated `add_component(kernel, covariance)` calls, then pass it here.
pub struct KroneckerLMCKernel {
    inner: Arc<LMCKernel>,
}

impl KroneckerLMCKernel {
    /// Wrap an existing [`LMCKernel`].
    pub fn new(inner: LMCKernel) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl MultiOutputKernel for KroneckerLMCKernel {
    fn n_outputs(&self) -> usize {
        self.inner.num_tasks()
    }

    /// Compute the p×p covariance block K(x, y) for the LMC model.
    ///
    /// Entry `block[i, j] = Σ_q B_q[i, j] * k_q(x, y)` is obtained by
    /// delegating to `LMCKernel::compute_tasks`.
    fn compute_block(&self, x: &[f64], y: &[f64]) -> Result<Array2<f64>> {
        let p = self.inner.num_tasks();
        let mut block = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let xi = TaskInput::from_slice(x, i);
                let yj = TaskInput::from_slice(y, j);
                block[[i, j]] = self.inner.compute_tasks(&xi, &yj).map_err(|e| {
                    KernelError::ComputationError(format!(
                        "LMC compute_tasks({}, {}) failed: {}",
                        i, j, e
                    ))
                })?;
            }
        }
        Ok(block)
    }

    fn name(&self) -> &str {
        "KroneckerLMC"
    }
}
