//! The [`MultiOutputKernel`] trait for matrix-valued covariance functions.

use scirs2_core::ndarray::Array2;

use crate::error::KernelError;

type Result<T> = std::result::Result<T, KernelError>;

/// A kernel that produces a `p×p` matrix-valued covariance block for each pair
/// of inputs, enabling vector-valued Gaussian Process regression.
///
/// For two feature vectors `x, y ∈ R^d`, the kernel returns a block
/// `K(x, y) ∈ R^{p×p}` where `p` is the number of outputs.  The standard
/// scalar Gram matrix is recovered from the block Gram matrix of shape
/// `(N·p × N·p)` which tiles the p×p blocks for every pair of N training
/// inputs.
pub trait MultiOutputKernel: Send + Sync {
    /// Number of outputs `p`.
    fn n_outputs(&self) -> usize;

    /// Compute the `p×p` covariance block K(x, x') between two feature vectors.
    fn compute_block(&self, x: &[f64], y: &[f64]) -> Result<Array2<f64>>;

    /// Build the block Gram matrix of shape `(N·p × N·p)` for N input points.
    ///
    /// The default implementation fills the `(N·p × N·p)` matrix by calling
    /// `compute_block` for each pair `(i, j)` with `i ≤ j` and placing the
    /// `p×p` result at rows `[i*p..(i+1)*p]`, cols `[j*p..(j+1)*p]`.
    /// Symmetry is enforced by copying the transpose of block `(i,j)` to
    /// position `(j,i)`.
    fn block_gram_matrix(&self, inputs: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n = inputs.len();
        let p = self.n_outputs();
        let np = n * p;
        let mut gram = Array2::<f64>::zeros((np, np));
        for i in 0..n {
            for j in i..n {
                let block = self.compute_block(&inputs[i], &inputs[j])?;
                for ri in 0..p {
                    for ci in 0..p {
                        gram[[i * p + ri, j * p + ci]] = block[[ri, ci]];
                        // Symmetric counterpart: K(x_j, x_i)[ci, ri] = K(x_i, x_j)[ri, ci]
                        gram[[j * p + ci, i * p + ri]] = block[[ci, ri]];
                    }
                }
            }
        }
        Ok(gram)
    }

    /// Human-readable kernel name.
    fn name(&self) -> &str;
}
