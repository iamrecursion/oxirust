//! Fluent builder for assembling [`LearnedMixtureKernel`] instances.
//!
//! Accepts anything that implements the crate's [`Kernel`] trait (which
//! includes `SymbolicKernel` produced by
//! [`crate::symbolic::KernelBuilder`]). Logits default to zero, giving a
//! uniform starting mixture; callers may override individual logits or
//! provide an explicit parallel logit vector at build time.

use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::learned_composition::mixture::LearnedMixtureKernel;
use crate::types::Kernel;

/// Fluent builder that collects a kernel library and trainable logits
/// before producing a [`LearnedMixtureKernel`].
#[derive(Default)]
pub struct LearnedMixtureBuilder {
    base_kernels: Vec<Arc<dyn Kernel>>,
    logits: Vec<f64>,
}

impl LearnedMixtureBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            base_kernels: Vec::new(),
            logits: Vec::new(),
        }
    }

    /// Push a kernel with a default logit of `0.0` (uniform weight).
    pub fn push_kernel(mut self, kernel: Arc<dyn Kernel>) -> Self {
        self.base_kernels.push(kernel);
        self.logits.push(0.0);
        self
    }

    /// Push a kernel with an explicit starting logit.
    pub fn push_kernel_with_logit(mut self, kernel: Arc<dyn Kernel>, logit: f64) -> Self {
        self.base_kernels.push(kernel);
        self.logits.push(logit);
        self
    }

    /// Push every kernel from an iterable; each gets a default logit
    /// of `0.0`.
    pub fn extend_kernels<I>(mut self, kernels: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn Kernel>>,
    {
        for kernel in kernels {
            self.base_kernels.push(kernel);
            self.logits.push(0.0);
        }
        self
    }

    /// Override the full logit vector. Must match the current kernel
    /// library length at build time.
    pub fn with_logits(mut self, logits: Vec<f64>) -> Self {
        self.logits = logits;
        self
    }

    /// Number of kernels queued so far.
    pub fn len(&self) -> usize {
        self.base_kernels.len()
    }

    /// Whether no kernels have been queued yet.
    pub fn is_empty(&self) -> bool {
        self.base_kernels.is_empty()
    }

    /// Finalise the builder.
    ///
    /// Fails when no kernels were pushed, or when a custom logit vector
    /// disagrees in length with the kernel library.
    pub fn build(self) -> Result<LearnedMixtureKernel> {
        if self.base_kernels.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "base_kernels".to_string(),
                value: "[]".to_string(),
                reason: "LearnedMixtureBuilder requires at least one kernel".to_string(),
            });
        }
        if self.logits.len() != self.base_kernels.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.base_kernels.len()],
                got: vec![self.logits.len()],
                context: "LearnedMixtureBuilder::with_logits length".to_string(),
            });
        }
        LearnedMixtureKernel::new(self.base_kernels, self.logits)
    }
}
