//! The [`DeepKernel`] type — a Deep Kernel Learning wrapper that
//! composes a base kernel with a neural feature extractor.
//!
//! Given a base kernel `K_base` and a feature map `g_θ`, the Deep
//! Kernel is
//!
//! ```text
//! K_DKL(x, y) = K_base(g_θ(x), g_θ(y)).
//! ```
//!
//! This generic wrapper implements the crate-level [`Kernel`] trait so a
//! `DeepKernel` can slot into any downstream machinery that consumes
//! `dyn Kernel` (SVM adapters, Gram-matrix utilities, kernel-alignment
//! search, etc.).
//!
//! The base kernel and feature extractor are both owned by the
//! wrapper. Cloning clones both; mutating parameters requires holding a
//! `&mut DeepKernel` and going through [`DeepKernel::feature_extractor_mut`].

use std::fmt;

use crate::deep_kernel::feature_extractor::NeuralFeatureMap;
use crate::error::Result;
use crate::types::Kernel;

/// Composition of a neural feature extractor with a classical kernel.
///
/// `F` — the neural feature extractor (e.g.
/// [`crate::deep_kernel::MLPFeatureExtractor`]).
///
/// `K` — the base kernel (e.g. [`crate::RbfKernel`]).
#[derive(Clone, Debug)]
pub struct DeepKernel<F: NeuralFeatureMap, K: Kernel> {
    extractor: F,
    base: K,
}

impl<F: NeuralFeatureMap, K: Kernel> DeepKernel<F, K> {
    /// Compose a feature extractor with a base kernel.
    pub fn new(extractor: F, base: K) -> Self {
        Self { extractor, base }
    }

    /// Immutable view of the feature extractor.
    pub fn feature_extractor(&self) -> &F {
        &self.extractor
    }

    /// Mutable view of the feature extractor — needed by optimisers
    /// that write into `parameters_mut()` and then call `sync_from_flat`
    /// on a concrete MLP.
    pub fn feature_extractor_mut(&mut self) -> &mut F {
        &mut self.extractor
    }

    /// Immutable view of the base kernel.
    pub fn base_kernel(&self) -> &K {
        &self.base
    }

    /// Apply the feature map to a single input.
    pub fn features(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.extractor.forward(x)
    }

    /// Evaluate the composed kernel on a single input pair.
    pub fn evaluate(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let fx = self.extractor.forward(x)?;
        let fy = self.extractor.forward(y)?;
        self.base.compute(&fx, &fy)
    }

    /// Compute a Gram matrix `G[i,j] = K_DKL(xs[i], ys[j])`.
    ///
    /// Feature maps are cached — each `xs[i]` is passed through the
    /// extractor at most once, same for `ys[j]`. For square `xs == ys`
    /// callers should prefer [`Self::compute_symmetric_gram`].
    pub fn compute_gram(&self, xs: &[&[f64]], ys: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
        let fx: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| self.extractor.forward(x))
            .collect::<Result<Vec<_>>>()?;
        let fy: Vec<Vec<f64>> = ys
            .iter()
            .map(|y| self.extractor.forward(y))
            .collect::<Result<Vec<_>>>()?;
        let mut matrix = vec![vec![0.0; fy.len()]; fx.len()];
        for i in 0..fx.len() {
            for j in 0..fy.len() {
                matrix[i][j] = self.base.compute(&fx[i], &fy[j])?;
            }
        }
        Ok(matrix)
    }

    /// Symmetric Gram matrix for a single input set.
    ///
    /// Takes advantage of `K(a, b) == K(b, a)` to halve the base-kernel
    /// evaluations; still calls the feature extractor `n` times.
    pub fn compute_symmetric_gram(&self, xs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let fx: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| self.extractor.forward(x))
            .collect::<Result<Vec<_>>>()?;
        let n = fx.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let v = self.base.compute(&fx[i], &fx[j])?;
                matrix[i][j] = v;
                matrix[j][i] = v;
            }
        }
        Ok(matrix)
    }
}

impl<F: NeuralFeatureMap, K: Kernel> Kernel for DeepKernel<F, K> {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.evaluate(x, y)
    }

    fn name(&self) -> &str {
        "DeepKernel"
    }

    fn is_psd(&self) -> bool {
        // K_DKL(x, y) = K_base(g(x), g(y)) is PSD iff K_base is PSD, by
        // the classical "kernels are closed under composition with
        // arbitrary maps" result.
        self.base.is_psd()
    }
}

/// Helper trait for kinds of feature extractor whose output dimension
/// matches the base kernel's expected input dimension. Implemented
/// automatically for every `NeuralFeatureMap`; exists purely as a
/// documentation anchor.
pub trait FeatureMapShape {
    /// Output dimension of the feature map — i.e. the dimension the
    /// base kernel will see.
    fn feature_dim(&self) -> usize;
}

impl<M: NeuralFeatureMap> FeatureMapShape for M {
    fn feature_dim(&self) -> usize {
        self.output_dim()
    }
}

/// Debug helper — prints extractor shape and base kernel name.
pub struct DeepKernelSummary<'a, F, K>
where
    F: NeuralFeatureMap,
    K: Kernel,
{
    pub kernel: &'a DeepKernel<F, K>,
}

impl<F, K> fmt::Display for DeepKernelSummary<'_, F, K>
where
    F: NeuralFeatureMap,
    K: Kernel,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeepKernel(in={}, features={}, base={})",
            self.kernel.extractor.input_dim(),
            self.kernel.extractor.output_dim(),
            self.kernel.base.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deep_kernel::feature_extractor::MLPFeatureExtractor;
    use crate::deep_kernel::layer::{Activation, DenseLayer};
    use crate::types::RbfKernelConfig;
    use crate::{LinearKernel, RbfKernel};

    fn identity_mlp_1x1() -> MLPFeatureExtractor {
        let layer =
            DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).expect("valid");
        MLPFeatureExtractor::from_layers(vec![layer]).expect("valid")
    }

    #[test]
    fn deep_kernel_with_identity_equals_base() {
        let linear = LinearKernel::new();
        let dkl = DeepKernel::new(identity_mlp_1x1(), linear);
        let expected = LinearKernel::new().compute(&[3.0], &[4.0]).expect("linear");
        let got = dkl.compute(&[3.0], &[4.0]).expect("deep");
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn deep_kernel_propagates_psd_from_base() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let dkl = DeepKernel::new(identity_mlp_1x1(), rbf);
        assert!(dkl.is_psd());
    }
}
