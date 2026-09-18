//! ε-Support Vector Regression (ε-SVR) via a direct SMO solver.
//!
//! Implements ε-insensitive Support Vector Regression using a purpose-built
//! Sequential Minimal Optimization solver for the 2n-variable dual.
//!
//! ## Mathematical Formulation
//!
//! For training data `{(x_i, y_i)}` with `y_i ∈ ℝ`, ε-SVR minimizes:
//!
//! ```text
//! minimize  (1/2)||w||² + C Σ_i (ξ_i + ξ_i*)
//! subject to  y_i - f(x_i) ≤ ε + ξ_i
//!             f(x_i) - y_i ≤ ε + ξ_i*
//!             ξ_i, ξ_i* ≥ 0
//! ```
//!
//! where `f(x) = Σ_j β_j K(x_j, x) - b` and `β_j = α^+_j - α^-_j`.
//!
//! ## Dual Formulation
//!
//! Introducing Lagrange multipliers `α^+_i, α^-_i ∈ [0, C]` with the equality
//! constraint `Σ_i (α^+_i - α^-_i) = 0`, the dual objective is:
//!
//! ```text
//! max W = -ε Σ_i (α^+_i + α^-_i) + Σ_i y_i (α^+_i - α^-_i)
//!         - (1/2) Σ_{i,j} β_i β_j K(x_i, x_j)
//! ```
//!
//! ## KKT Conditions
//!
//! At optimum, for each training point `i`:
//! - `α^+_i ∈ (0,C)` ⟹ `f(x_i) - y_i = ε`   (upper tube boundary)
//! - `α^-_i ∈ (0,C)` ⟹ `f(x_i) - y_i = -ε`  (lower tube boundary)
//! - `α^+_i = 0`     ⟹ `f(x_i) - y_i ≤ ε`   (inside or below upper)
//! - `α^-_i = 0`     ⟹ `f(x_i) - y_i ≥ -ε`  (inside or above lower)
//!
//! ## SMO for ε-SVR
//!
//! Following the approach in Chang & Lin (2001) (the libsvm SVR solver), we
//! expand the 2n-variable dual into a 2n-sample problem.
//!
//! Define an extended dataset with 2n indices:
//! - Index i (0 ≤ i < n): variable α^+_i, "sign" s_i = +1
//! - Index i+n (0 ≤ i < n): variable α^-_i, "sign" s_{i+n} = -1
//!
//! The regression function in this notation:
//! `f(x) = Σ_{k=0}^{2n-1} α_k s_k K(x_{k%n}, x) - b`
//!
//! KKT violation checks:
//! - For k < n (α^+ variable): KKT requires `s_k (f(x_k) - y_k) ≤ ε`
//!   i.e., `f(x_k) - y_k ≤ ε`. Error: `E^+_k = f(x_k) - y_k - ε`; non-bound KKT: E^+_k = 0.
//! - For k ≥ n (α^- variable): KKT requires `s_k (f(x_{k%n}) - y_{k%n}) ≤ ε`
//!   i.e., `-(f(x_{k%n}) - y_{k%n}) ≤ ε`. Error: `E^-_k = -(f(x_{k%n}) - y_{k%n}) - ε`; non-bound KKT: E^-_k = 0.
//!
//! Define a unified error for the 2n system:
//! `G_k = s_k * f(x_{k%n}) - y_{k%n} * s_k - ε`
//!       = `f(x_{k%n}) * s_k - y_{k%n} * s_k - ε`
//!
//! For k < n: `G_k = f(x_k) - y_k - ε`. Non-bound KKT: G_k = 0.
//! For k ≥ n: `G_k = -f(x_{k%n}) + y_{k%n} - ε`. Non-bound KKT: G_k = 0.
//!
//! ## Bias Update
//!
//! After updating variables at positions i1 and i2:
//! - If α^+_j is non-bound: `b_new = b_old + G_j + delta_j * K_aug[j,j] + ...`
//! - If α^-_j is non-bound: same formula with appropriate signs.
//!
//! ## References
//!
//! - Chang, C-C., Lin, C-J. (2001). LIBSVM: Implementation of Support Vector
//!   Machines. Software at <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>.
//! - Smola, A.J., Schölkopf, B. (1998). A tutorial on support vector regression.
//!   NeuroCOLT2 Technical Report NC2-TR-1998-030.
//! - Schölkopf, B., Smola, A.J. (2002). Learning with Kernels. MIT Press. Ch. 9.

use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::smo::SmoConfig;

/// State for the SVR SMO solver.
///
/// We maintain 2n dual variables:
///   alpha_pos[i] = α^+_i ∈ [0,C]  for i = 0..n-1
///   alpha_neg[i] = α^-_i ∈ [0,C]  for i = 0..n-1
///
/// The regression function is:
///   f(x) = Σ_i (α^+_i - α^-_i) K(x_i, x) - b
///
/// For KKT bookkeeping we use the 2n-index scheme with:
///   - index k ∈ [0,n): corresponds to α^+_{k}, sign = +1
///   - index k ∈ [n,2n): corresponds to α^-_{k-n}, sign = -1
///
/// The "gradient" (negative marginal gain) for index k:
///   G_k = sign_k * f(x_{k%n}) - y_{k%n} - ε * sign_k
///   (= f(x_k) - y_k - ε for k < n,  = -f(x_k) + y_k - ε for k ≥ n)
struct SvrState {
    /// Dual variables α^+ (length n).
    alpha_pos: Vec<f64>,
    /// Dual variables α^- (length n).
    alpha_neg: Vec<f64>,
    /// Bias term.
    b: f64,
    /// Error / gradient cache G_k for k in [0, 2n).
    /// G_k = sign_k * f(x_{k%n}) - y_{k%n} * sign_k - ε   (= 0 at non-bound SVs).
    grad_cache: Vec<f64>,
    /// Number of original training examples.
    n: usize,
    /// Regularization box constraint.
    c: f64,
    /// ε-insensitive tube half-width.
    eps: f64,
    /// KKT tolerance.
    tol: f64,
    /// Regression targets.
    y: Vec<f64>,
    /// Pre-computed n×n original kernel matrix.
    kernel_matrix: Vec<Vec<f64>>,
}

impl SvrState {
    /// Compute f(x_i) for original index i using current α^+, α^-.
    fn regression_fn(&self, i: usize) -> f64 {
        let mut val = 0.0_f64;
        for j in 0..self.n {
            let beta_j = self.alpha_pos[j] - self.alpha_neg[j];
            if beta_j.abs() > 1e-12 {
                val += beta_j * self.kernel_matrix[j][i];
            }
        }
        val - self.b
    }

    /// Refresh gradient cache entry for augmented index k using exact computation.
    ///
    /// H_k = -∂W/∂α_k:
    ///   k < n:  H_k = f(x_k) - y_k + ε
    ///   k ≥ n:  H_k = y_{k-n} + ε - f(x_{k-n})
    fn refresh_grad(&mut self, k: usize) {
        let i = k % self.n;
        let fx = self.regression_fn(i);
        if k < self.n {
            self.grad_cache[k] = fx - self.y[i] + self.eps;
        } else {
            self.grad_cache[k] = self.y[i] + self.eps - fx;
        }
    }

    /// Return the α value for augmented index k.
    fn alpha(&self, k: usize) -> f64 {
        if k < self.n {
            self.alpha_pos[k]
        } else {
            self.alpha_neg[k - self.n]
        }
    }

    /// Set the α value for augmented index k.
    fn set_alpha(&mut self, k: usize, val: f64) {
        if k < self.n {
            self.alpha_pos[k] = val;
        } else {
            self.alpha_neg[k - self.n] = val;
        }
    }

    /// Sign for augmented index k: +1 for k < n, -1 for k ≥ n.
    fn sign(&self, k: usize) -> f64 {
        if k < self.n {
            1.0
        } else {
            -1.0
        }
    }

    /// Original index for augmented index k.
    fn orig(&self, k: usize) -> usize {
        k % self.n
    }

    /// Augmented kernel value K_aug[k1][k2] = sign(k1) * sign(k2) * K[orig(k1)][orig(k2)].
    fn k_aug(&self, k1: usize, k2: usize) -> f64 {
        self.sign(k1) * self.sign(k2) * self.kernel_matrix[self.orig(k1)][self.orig(k2)]
    }

    /// Is augmented variable k non-bound (0 < α_k < C)?
    fn is_non_bound(&self, k: usize) -> bool {
        let a = self.alpha(k);
        a > 0.0 && a < self.c
    }

    /// Does augmented variable k have a KKT violation?
    ///
    /// KKT requires: if G_k < -tol → should increase α_k (violation if α_k = C).
    ///               if G_k > +tol → should decrease α_k (violation if α_k = 0).
    fn kkt_violated(&self, k: usize) -> bool {
        let g = self.grad_cache[k];
        let a = self.alpha(k);
        (g < -self.tol && a < self.c) || (g > self.tol && a > 0.0)
    }

    /// Execute the two-variable SMO update for augmented indices (k1, k2).
    ///
    /// Returns true if a meaningful update was made.
    fn take_step(&mut self, k1: usize, k2: usize) -> bool {
        if k1 == k2 {
            return false;
        }
        // Do not pair the α^+ and α^- for the SAME training point (k and k+n).
        // Pairing them would only cancel in f, and keeping both non-zero simultaneously
        // violates the complementary slackness condition (α^+_i * α^-_i = 0).
        if k1 % self.n == k2 % self.n {
            return false;
        }

        let a1 = self.alpha(k1);
        let a2 = self.alpha(k2);
        let g1 = self.grad_cache[k1];
        let g2 = self.grad_cache[k2];

        // The augmented kernel values.
        let k_aug11 = self.k_aug(k1, k1);
        let k_aug12 = self.k_aug(k1, k2);
        let k_aug22 = self.k_aug(k2, k2);

        // Sign product for the box constraint determination.
        let s_aug = self.sign(k1) * self.sign(k2);

        // Box constraints for the updated a2 (following the SVC derivation in the
        // augmented 2n variable space, identical to smo.rs::take_step).
        let (lo, hi) = if (s_aug - 1.0).abs() > 1e-10 {
            (f64::max(0.0, a2 - a1), f64::min(self.c, self.c + a2 - a1))
        } else {
            (f64::max(0.0, a1 + a2 - self.c), f64::min(self.c, a1 + a2))
        };

        if lo >= hi {
            return false;
        }

        let eta = k_aug11 + k_aug22 - 2.0 * k_aug12;

        let a2_new = if eta > 1e-12 {
            // Normal update: gradient-based step.
            // The gradient of W w.r.t. α_aug[k2] is:
            //   ∂W/∂α[k2] = -G_k2  (negative of the error)
            // So: a2_unc = a2 + sign(k2) * (G_k1 - G_k2) / η  ... this needs care.
            //
            // Using the standard SVC step with the augmented error:
            //   a2_unc = a2 + sign(k2) * (g1 - g2) / η
            let a2_unc = a2 + self.sign(k2) * (g1 - g2) / eta;
            a2_unc.max(lo).min(hi)
        } else {
            // Degenerate: evaluate objective at endpoints.
            let gamma = a1 + s_aug * a2;
            let a1_at_lo = gamma - s_aug * lo;
            let a1_at_hi = gamma - s_aug * hi;
            // Objective contributions (using gradient values as proxies for f values).
            let f1_proxy = g1 + self.sign(k1);
            let f2_proxy = g2 + self.sign(k2);
            let lobj = -0.5 * k_aug11 * a1_at_lo * a1_at_lo
                - 0.5 * k_aug22 * lo * lo
                - s_aug * k_aug12 * a1_at_lo * lo
                - self.sign(k1) * a1_at_lo * f1_proxy
                - self.sign(k2) * lo * f2_proxy
                + a1_at_lo
                + lo;
            let hobj = -0.5 * k_aug11 * a1_at_hi * a1_at_hi
                - 0.5 * k_aug22 * hi * hi
                - s_aug * k_aug12 * a1_at_hi * hi
                - self.sign(k1) * a1_at_hi * f1_proxy
                - self.sign(k2) * hi * f2_proxy
                + a1_at_hi
                + hi;
            if lobj > hobj + 1e-12 {
                lo
            } else if hobj > lobj + 1e-12 {
                hi
            } else {
                a2
            }
        };

        if (a2_new - a2).abs() < 1e-5 * (a2_new + a2 + 1e-10) {
            return false;
        }

        let a1_new = a1 + s_aug * (a2 - a2_new);
        let b_old = self.b;

        // Bias update using KKT at non-bound variables.
        // The KKT condition for non-bound augmented variable k:
        //   G_k = 0  ⟺  sign_k * f(x_{orig_k}) - y_{orig_k} * sign_k - ε = 0
        //         ⟺  f(x_{orig_k}) = y_{orig_k} + ε * sign_k   (if sign_k = +1)
        //         ⟺  f(x_{orig_k}) = y_{orig_k} - ε            (if sign_k = -1)
        //
        // f_new(x_{orig_k}) = f_old(x_{orig_k}) + sign_{k1}*(a1_new-a1)*K[orig1][orig] +
        //                      sign_{k2}*(a2_new-a2)*K[orig2][orig]
        //                    = f_old + delta1 * K[orig1][orig] + delta2 * K[orig2][orig]
        //  (where delta_k = sign_k * (alpha_k_new - alpha_k_old))
        //
        // For non-bound k1: G_k1_new = 0:
        //   sign_k1 * f_new(x_{orig1}) - y_{orig1} * sign_k1 - ε = 0
        //   sign_k1 * (f_old(x_{orig1}) + Δ) - y_{orig1} * sign_k1 - ε = 0
        //   G_k1_old + sign_k1 * Δ = 0
        // where Δ = sign_k1*(a1_new-a1)*K[orig1][orig1] + sign_k2*(a2_new-a2)*K[orig2][orig1]
        //         (change in f(x_{orig1})) - (b_new - b_old)
        //
        // G_k1_old + sign_k1 * [sign_k1*(a1_new-a1)*K[orig1][orig1] + sign_k2*(a2_new-a2)*K[orig2][orig1]]
        //           - sign_k1*(b_new - b_old) = 0
        //
        // sign_k1*(b_new - b_old) = G_k1_old + K_aug[k1][k1]*(a1_new-a1) + K_aug[k2][k1]*(a2_new-a2)
        // b_new = b_old + sign_k1 * [G_k1_old + K_aug[k1][k1]*(a1_new-a1) + K_aug[k2][k1]*(a2_new-a2)]
        //
        // For k < n: sign_k = +1, so b_new = b_old + G_k1_old + ...
        // For k ≥ n: sign_k = -1, so b_new = b_old - G_k1_old - ...
        let b1_cand =
            b_old + self.sign(k1) * (g1 + k_aug11 * (a1_new - a1) + k_aug12 * (a2_new - a2));
        let b2_cand =
            b_old + self.sign(k2) * (g2 + k_aug12 * (a1_new - a1) + k_aug22 * (a2_new - a2));

        let b_new = if a1_new > 1e-8 * self.c && a1_new < self.c * (1.0 - 1e-8) {
            b1_cand
        } else if a2_new > 1e-8 * self.c && a2_new < self.c * (1.0 - 1e-8) {
            b2_cand
        } else {
            (b1_cand + b2_cand) * 0.5
        };

        // Commit the updates.
        self.set_alpha(k1, a1_new);
        self.set_alpha(k2, a2_new);
        self.b = b_new;

        // Update gradient cache for all non-bound variables.
        let delta_b = b_new - b_old;
        let delta_a1 = a1_new - a1;
        let delta_a2 = a2_new - a2;
        let n2 = 2 * self.n;

        for j in 0..n2 {
            if self.is_non_bound(j) {
                // Change in f(x_{orig_j}):
                //   Δf = sign(k1)*(a1_new-a1)*K[orig1][orig_j] + sign(k2)*(a2_new-a2)*K[orig2][orig_j]
                let delta_f = self.sign(k1)
                    * delta_a1
                    * self.kernel_matrix[self.orig(k1)][self.orig(j)]
                    + self.sign(k2) * delta_a2 * self.kernel_matrix[self.orig(k2)][self.orig(j)];
                // G_j_new = sign_j * (f_new(x_{orig_j}) - b_new) - y_{orig_j} * sign_j - ε
                //         = G_j_old + sign_j * (Δf - delta_b)
                // But wait: G_j = sign_j * f(x_{orig_j}) - y_{orig_j} - ε * sign_j
                // where f(x) = Σ_k beta_k K(x_k, x) - b
                //
                // Δ(G_j) = sign_j * (Δf - delta_b)
                self.grad_cache[j] += self.sign(j) * (delta_f - delta_b);
            }
        }

        // Recompute exact gradient for k1 and k2.
        self.refresh_grad(k1);
        self.refresh_grad(k2);

        true
    }

    /// Examine augmented variable k2 for KKT violation and try to optimize it.
    ///
    /// Uses Platt's heuristics:
    /// 1. Best non-bound partner (max |G_k1 - G_k2|)
    /// 2. Random sweep of non-bound examples
    /// 3. Random sweep of all examples
    fn examine(&mut self, k2: usize, random_offset: usize) -> bool {
        // For bound variables (α=0 or C) the gradient cache may be stale because
        // incremental updates only apply to non-bound variables.
        // Always refresh the gradient for bound variables before the KKT check.
        if !self.is_non_bound(k2) {
            self.refresh_grad(k2);
        }
        if !self.kkt_violated(k2) {
            return false;
        }

        let n2 = 2 * self.n;
        let g2 = self.grad_cache[k2];

        // Heuristic step 1: best non-bound partner.
        let non_bound: Vec<usize> = (0..n2).filter(|&j| self.is_non_bound(j)).collect();
        if non_bound.len() > 1 {
            let mut best_k1 = None;
            let mut best_diff = 0.0_f64;
            for &j in &non_bound {
                if j == k2 {
                    continue;
                }
                let diff = (self.grad_cache[j] - g2).abs();
                if diff > best_diff {
                    best_diff = diff;
                    best_k1 = Some(j);
                }
            }
            if let Some(k1) = best_k1 {
                if self.take_step(k1, k2) {
                    return true;
                }
            }
        }

        // Heuristic step 2: sweep non-bound starting from random offset.
        if !non_bound.is_empty() {
            let start = random_offset % non_bound.len();
            for jj in 0..non_bound.len() {
                let k1 = non_bound[(start + jj) % non_bound.len()];
                if k1 == k2 {
                    continue;
                }
                if self.take_step(k1, k2) {
                    return true;
                }
            }
        }

        // Step 3: sweep all variables.
        let start = random_offset % n2;
        for jj in 0..n2 {
            let k1 = (start + jj) % n2;
            if k1 == k2 {
                continue;
            }
            if self.take_step(k1, k2) {
                return true;
            }
        }

        false
    }
}

/// Run the ε-SVR SMO solver.
///
/// Returns `(alpha_pos, alpha_neg, b)` where:
/// - `alpha_pos[i]` = α^+_i ∈ [0, C]
/// - `alpha_neg[i]` = α^-_i ∈ [0, C]
/// - `b` = bias term
///
/// The regression function is `f(x) = Σ_i (α^+_i - α^-_i) K(x_i, x) - b`.
fn smo_svr_direct(
    x: &[Vec<f64>],
    y: &[f64],
    kernel: &Arc<dyn Kernel>,
    config: &SmoConfig,
    eps: f64,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let n = x.len();
    if n == 0 {
        return Err(KernelError::DimensionMismatch {
            expected: vec![1],
            got: vec![0],
            context: "smo_svr_direct: empty training set".to_string(),
        });
    }

    // Pre-compute n×n kernel matrix.
    let mut kernel_matrix = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let k_val = kernel.compute(&x[i], &x[j])?;
            kernel_matrix[i][j] = k_val;
            kernel_matrix[j][i] = k_val;
        }
    }

    let alpha_pos = vec![0.0_f64; n];
    let alpha_neg = vec![0.0_f64; n];
    let b = 0.0_f64;

    // Initialize gradient cache.
    // G_k = sign_k * f(x_{orig_k}) - y_{orig_k} * sign_k - ε
    // With α=0, b=0: f(x_i) = 0 for all i.
    // G_k (k < n):  = 0 - y[k] - ε = -(y[k] + ε)... wait:
    //   G_k = sign_k * f(x_k) - y_k * sign_k - ε
    //   For k < n: sign_k = +1: G_k = f(x_k) - y_k - ε = 0 - y_k - ε = -(y_k + ε)
    //   Wait that seems odd. Let me re-derive.
    //
    // G_k should be 0 at optimum for non-bound variables.
    // For k < n (α^+ variable): KKT: f(x_k) = y_k + ε. So G_k = f(x_k) - y_k - ε.
    //   Initially: G_k = 0 - y_k - ε = -(y_k + ε). This is generally non-zero.
    //
    // For k ≥ n (α^- variable): KKT: f(x_{k-n}) = y_{k-n} - ε.
    //   But the sign is -1, so G_k = -f(x_{k-n}) + y_{k-n} - ε... let me re-think.
    //
    // G_k for k < n: G_k = f(x_k) - y_k - ε (= 0 at non-bound α^+)
    // G_k for k ≥ n: G_k = -(f(x_{k-n}) - y_{k-n}) - ε = y_{k-n} - f(x_{k-n}) - ε
    //                     (= 0 at non-bound α^-)
    //
    // Initially f = 0, b = 0:
    // G_k (k<n):  -(y[k] + ε)
    // G_k (k≥n):  y[k-n] - ε
    // CORRECTED gradient initialization.
    // H_k = -∂W/∂α_k (negative gradient for the MAXIMIZATION dual).
    // From ∂W/∂α^+_k = y_k - ε - f(x_k) → H_k (k<n) = f(x_k) - y_k + ε.
    // From ∂W/∂α^-_k = f(x_k) - y_k - ε → H_{k+n} = y_k + ε - f(x_k).
    // Initially (f=0, b=0): H_k (k<n) = ε - y_k; H_{k+n} = y_k + ε.
    let n2 = 2 * n;
    let mut grad_cache = vec![0.0_f64; n2];
    for k in 0..n {
        grad_cache[k] = eps - y[k]; // H_k for α^+ (k<n): ε - y_k
        grad_cache[k + n] = y[k] + eps; // H_{k+n} for α^- (k≥n): y_k + ε
    }

    let mut state = SvrState {
        alpha_pos,
        alpha_neg,
        b,
        grad_cache,
        n,
        c: config.c,
        eps,
        tol: config.tol,
        y: y.to_vec(),
        kernel_matrix,
    };

    // Main SMO loop.
    let mut examine_all = true;
    let mut total_passes = 0usize;
    let mut random_offset: usize = 37;

    loop {
        let mut num_changed = 0usize;
        let was_examine_all = examine_all;

        if examine_all {
            for k2 in 0..n2 {
                if state.examine(k2, random_offset) {
                    num_changed += 1;
                }
                random_offset = random_offset
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    & 0xFFFF;
            }
        } else {
            let non_bound: Vec<usize> = (0..n2).filter(|&j| state.is_non_bound(j)).collect();
            for k2 in non_bound {
                if state.examine(k2, random_offset) {
                    num_changed += 1;
                }
                random_offset = random_offset
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    & 0xFFFF;
            }
        }

        total_passes += 1;
        if total_passes > config.max_iter {
            return Err(KernelError::ComputationError(format!(
                "SVR SMO did not converge after {} passes (tol={}, C={}, ε={}). \
                 Consider increasing max_iter or adjusting hyperparameters.",
                config.max_iter, config.tol, config.c, eps
            )));
        }

        // Convergence: full-all pass with 0 changes → KKT satisfied.
        if was_examine_all && num_changed == 0 {
            break;
        }

        if was_examine_all {
            examine_all = false;
        } else if num_changed == 0 {
            examine_all = true;
        }
    }

    Ok((state.alpha_pos, state.alpha_neg, state.b))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Unfitted ε-Support Vector Regression model.
///
/// Call [`SvrModel::fit`] to produce an [`SvrFitted`] that supports prediction.
pub struct SvrModel {
    /// Kernel function.
    kernel: Arc<dyn Kernel>,
    /// SMO solver configuration (C, tolerance, max iterations).
    config: SmoConfig,
    /// ε-insensitive tube half-width. Residuals smaller than ε in absolute
    /// value are treated as zero loss.
    pub epsilon: f64,
}

impl std::fmt::Debug for SvrModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvrModel")
            .field("kernel", &self.kernel.name())
            .field("C", &self.config.c)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

impl SvrModel {
    /// Create a new SVR model.
    ///
    /// # Arguments
    ///
    /// * `kernel`  – Any `Arc<dyn Kernel>`. PSD kernels guarantee convergence.
    /// * `c`       – Box constraint (C > 0). Controls the penalty for margin
    ///   violations. Larger C = less regularization.
    /// * `epsilon` – ε-tube half-width (ε ≥ 0). Points within ε of the
    ///   prediction surface incur zero loss.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidParameter`] if `c ≤ 0` or `epsilon < 0`.
    pub fn new(kernel: Arc<dyn Kernel>, c: f64, epsilon: f64) -> Result<Self> {
        if c <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "C".to_string(),
                value: c.to_string(),
                reason: "C must be strictly positive".to_string(),
            });
        }
        if epsilon < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "epsilon".to_string(),
                value: epsilon.to_string(),
                reason: "epsilon must be non-negative".to_string(),
            });
        }
        Ok(Self {
            kernel,
            config: SmoConfig {
                c,
                epsilon,
                ..SmoConfig::default()
            },
            epsilon,
        })
    }

    /// Create a new SVR model with full solver configuration.
    pub fn new_with_config(
        kernel: Arc<dyn Kernel>,
        epsilon: f64,
        config: SmoConfig,
    ) -> Result<Self> {
        if config.c <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "C".to_string(),
                value: config.c.to_string(),
                reason: "C must be strictly positive".to_string(),
            });
        }
        if epsilon < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "epsilon".to_string(),
                value: epsilon.to_string(),
                reason: "epsilon must be non-negative".to_string(),
            });
        }
        Ok(Self {
            kernel,
            config,
            epsilon,
        })
    }

    /// Fit the SVR model to training data.
    ///
    /// # Arguments
    ///
    /// * `x` – Training inputs (N × d feature vectors, all same dimension).
    /// * `y` – Regression targets (ℝ^N), length N.
    ///
    /// # Errors
    ///
    /// - [`KernelError::DimensionMismatch`] – empty or inconsistent data.
    /// - [`KernelError::ComputationError`]  – SMO did not converge.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<SvrFitted> {
        let n = x.len();
        if n == 0 {
            return Err(KernelError::DimensionMismatch {
                expected: vec![1],
                got: vec![0],
                context: "SvrModel::fit: training set cannot be empty".to_string(),
            });
        }
        if y.len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n],
                got: vec![y.len()],
                context: "SvrModel::fit: y must have the same length as x".to_string(),
            });
        }
        if n > 1 {
            let d0 = x[0].len();
            for (i, xi) in x.iter().enumerate().skip(1) {
                if xi.len() != d0 {
                    return Err(KernelError::DimensionMismatch {
                        expected: vec![d0],
                        got: vec![xi.len()],
                        context: format!("SvrModel::fit: x[{}] has wrong dimension", i),
                    });
                }
            }
        }

        let (alpha_pos, alpha_neg, b) =
            smo_svr_direct(x, y, &self.kernel, &self.config, self.epsilon)?;

        // Extract support vectors: only those with |β_i| = |α^+_i - α^-_i| > threshold.
        let sv_threshold = 1e-8 * self.config.c;
        let mut support_vectors = Vec::new();
        let mut support_coefficients = Vec::new();

        for i in 0..n {
            let beta_i = alpha_pos[i] - alpha_neg[i];
            if beta_i.abs() > sv_threshold {
                support_vectors.push(x[i].clone());
                support_coefficients.push(beta_i);
            }
        }

        Ok(SvrFitted {
            support_vectors,
            support_coefficients,
            bias: b,
            kernel: Arc::clone(&self.kernel),
        })
    }
}

/// Fitted ε-SVR model.
pub struct SvrFitted {
    /// Support vectors (training points with |β_i| = |α^+_i - α^-_i| > threshold).
    pub support_vectors: Vec<Vec<f64>>,
    /// Regression coefficients β_i = α^+_i - α^-_i for each support vector.
    pub support_coefficients: Vec<f64>,
    /// Bias / threshold b.
    pub bias: f64,
    /// Kernel function.
    kernel: Arc<dyn Kernel>,
}

impl std::fmt::Debug for SvrFitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvrFitted")
            .field("num_support_vectors", &self.support_vectors.len())
            .field("bias", &self.bias)
            .finish()
    }
}

impl SvrFitted {
    /// Predict the regression output at a single test point `x`.
    ///
    /// `f(x) = Σ_i β_i K(sv_i, x) - b`
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        let mut val = 0.0_f64;
        for (sv, &coef) in self
            .support_vectors
            .iter()
            .zip(self.support_coefficients.iter())
        {
            val += coef * self.kernel.compute(sv, x)?;
        }
        Ok(val - self.bias)
    }

    /// Predict regression outputs for a batch of test inputs.
    ///
    /// Returns a `Vec<f64>` of the same length as `x`.
    pub fn predict_batch(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        x.iter().map(|xi| self.predict(xi)).collect()
    }

    /// Number of support vectors.
    pub fn num_support_vectors(&self) -> usize {
        self.support_vectors.len()
    }
}
