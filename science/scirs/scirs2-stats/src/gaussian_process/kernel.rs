//! Kernel functions for Gaussian Processes
//!
//! Kernels (covariance functions) determine the properties of functions drawn from
//! a Gaussian Process. They encode assumptions about the function being modeled.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;

/// Trait for kernel functions (covariance functions)
pub trait Kernel: Clone + Send + Sync {
    /// Compute covariance between two points
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64;

    /// Compute covariance matrix for a set of points
    fn compute_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let kij = self.compute(&x.row(i), &x.row(j));
                k[[i, j]] = kij;
                if i != j {
                    k[[j, i]] = kij;
                }
            }
        }

        k
    }

    /// Compute covariance matrix between two sets of points
    fn compute_cross_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                k[[i, j]] = self.compute(&x1.row(i), &x2.row(j));
            }
        }

        k
    }

    /// Get kernel parameters (for optimization)
    fn get_params(&self) -> Vec<f64>;

    /// Set kernel parameters
    fn set_params(&mut self, params: &[f64]);

    /// Get number of parameters
    fn n_params(&self) -> usize {
        self.get_params().len()
    }
}

/// Squared Exponential (RBF) kernel: k(x, x') = σ² exp(-||x - x'||² / (2 l²))
///
/// This is the most commonly used kernel. It assumes smoothness and is infinitely
/// differentiable.
#[derive(Debug, Clone)]
pub struct SquaredExponential {
    /// Length scale parameter (controls how quickly correlation decays)
    pub length_scale: f64,
    /// Signal variance (controls output scale)
    pub signal_variance: f64,
}

impl SquaredExponential {
    /// Create a new Squared Exponential kernel
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Default for SquaredExponential {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }
}

impl Kernel for SquaredExponential {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sq_dist += diff * diff;
        }

        self.signal_variance * (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance]
    }

    fn set_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
        }
    }
}

/// Matérn kernel with ν = 1/2
///
/// Equivalent to Exponential kernel: k(x, x') = σ² exp(-||x - x'|| / l)
/// This kernel produces rougher functions than RBF.
#[derive(Debug, Clone)]
pub struct Matern12 {
    pub length_scale: f64,
    pub signal_variance: f64,
}

impl Matern12 {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Default for Matern12 {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }
}

impl Kernel for Matern12 {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sq_dist += diff * diff;
        }
        let dist = sq_dist.sqrt();

        self.signal_variance * (-dist / self.length_scale).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance]
    }

    fn set_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
        }
    }
}

/// Matérn kernel with ν = 3/2
///
/// k(x, x') = σ² (1 + √3 r / l) exp(-√3 r / l), where r = ||x - x'||
/// Once differentiable, smoother than Matérn 1/2.
#[derive(Debug, Clone)]
pub struct Matern32 {
    pub length_scale: f64,
    pub signal_variance: f64,
}

impl Matern32 {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Default for Matern32 {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }
}

impl Kernel for Matern32 {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sq_dist += diff * diff;
        }
        let dist = sq_dist.sqrt();
        let sqrt3 = 3.0_f64.sqrt();
        let arg = sqrt3 * dist / self.length_scale;

        self.signal_variance * (1.0 + arg) * (-arg).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance]
    }

    fn set_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
        }
    }
}

/// Matérn kernel with ν = 5/2
///
/// k(x, x') = σ² (1 + √5 r / l + 5 r² / (3 l²)) exp(-√5 r / l)
/// Twice differentiable, very smooth.
#[derive(Debug, Clone)]
pub struct Matern52 {
    pub length_scale: f64,
    pub signal_variance: f64,
}

impl Matern52 {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Default for Matern52 {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }
}

impl Kernel for Matern52 {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sq_dist += diff * diff;
        }
        let dist = sq_dist.sqrt();
        let sqrt5 = 5.0_f64.sqrt();
        let arg = sqrt5 * dist / self.length_scale;
        let arg2 = 5.0 * sq_dist / (3.0 * self.length_scale * self.length_scale);

        self.signal_variance * (1.0 + arg + arg2) * (-arg).exp()
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance]
    }

    fn set_params(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
        }
    }
}

/// White kernel (noise): k(x, x') = σ² δ(x, x')
///
/// This kernel only produces variance on the diagonal, representing
/// independent noise on observations.
#[derive(Debug, Clone)]
pub struct WhiteKernel {
    pub noise_level: f64,
}

impl WhiteKernel {
    pub fn new(noise_level: f64) -> Self {
        Self { noise_level }
    }
}

impl Default for WhiteKernel {
    fn default() -> Self {
        Self { noise_level: 0.01 }
    }
}

impl Kernel for WhiteKernel {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        // Check if points are identical
        let identical = x1
            .iter()
            .zip(x2.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-10);

        if identical {
            self.noise_level
        } else {
            0.0
        }
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.noise_level]
    }

    fn set_params(&mut self, params: &[f64]) {
        if !params.is_empty() {
            self.noise_level = params[0];
        }
    }
}

/// Sum of two kernels: k(x, x') = k1(x, x') + k2(x, x')
#[derive(Debug, Clone)]
pub struct SumKernel<K1: Kernel, K2: Kernel> {
    pub kernel1: K1,
    pub kernel2: K2,
}

impl<K1: Kernel, K2: Kernel> SumKernel<K1, K2> {
    pub fn new(kernel1: K1, kernel2: K2) -> Self {
        Self { kernel1, kernel2 }
    }
}

impl<K1: Kernel, K2: Kernel> Kernel for SumKernel<K1, K2> {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.kernel1.compute(x1, x2) + self.kernel2.compute(x1, x2)
    }

    fn get_params(&self) -> Vec<f64> {
        let mut params = self.kernel1.get_params();
        params.extend(self.kernel2.get_params());
        params
    }

    fn set_params(&mut self, params: &[f64]) {
        let n1 = self.kernel1.n_params();
        if params.len() >= n1 {
            self.kernel1.set_params(&params[..n1]);
            if params.len() > n1 {
                self.kernel2.set_params(&params[n1..]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_squared_exponential() {
        let kernel = SquaredExponential::default();
        let x1 = array![0.0, 0.0];
        let x2 = array![1.0, 1.0];

        // Self-covariance should be signal_variance
        assert!((kernel.compute(&x1.view(), &x1.view()) - 1.0).abs() < 1e-10);

        // Cross-covariance should decay with distance
        let k12 = kernel.compute(&x1.view(), &x2.view());
        assert!(k12 < 1.0);
        assert!(k12 > 0.0);
    }

    #[test]
    fn test_matern_kernels() {
        let m12 = Matern12::default();
        let m32 = Matern32::default();
        let m52 = Matern52::default();

        let x1 = array![0.0];
        let x2 = array![1.0];

        // All should decay but at different rates
        let k12 = m12.compute(&x1.view(), &x2.view());
        let k32 = m32.compute(&x1.view(), &x2.view());
        let k52 = m52.compute(&x1.view(), &x2.view());

        assert!(k12 > 0.0 && k12 < 1.0);
        assert!(k32 > 0.0 && k32 < 1.0);
        assert!(k52 > 0.0 && k52 < 1.0);
    }

    #[test]
    fn test_white_kernel() {
        let kernel = WhiteKernel::new(0.1);
        let x1 = array![0.0, 0.0];
        let x2 = array![1.0, 1.0];

        // Same points should give noise_level
        assert!((kernel.compute(&x1.view(), &x1.view()) - 0.1).abs() < 1e-10);

        // Different points should give 0
        assert!((kernel.compute(&x1.view(), &x2.view())).abs() < 1e-10);
    }

    #[test]
    fn test_sum_kernel() {
        let rbf = SquaredExponential::default();
        let noise = WhiteKernel::new(0.1);
        let kernel = SumKernel::new(rbf, noise);

        let x1 = array![0.0];

        // Self-covariance should be signal_variance + noise
        let k = kernel.compute(&x1.view(), &x1.view());
        assert!((k - 1.1).abs() < 1e-10);
    }
}
