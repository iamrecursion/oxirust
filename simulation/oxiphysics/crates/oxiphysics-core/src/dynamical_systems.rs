// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Dynamical systems analysis: ODE fixed points, Lyapunov exponents,
//! bifurcations, phase plane, Poincaré sections, strange attractors,
//! chaos indicators (RQA), Hamiltonian systems, KAM tori, and symplectic
//! integration.
//!
//! All arithmetic uses plain `f64` and `[f64; 3]` arrays — no nalgebra.

// ─────────────────────────────────────────────────────────────────────────────
// Helper vector arithmetic on [f64; 3]
// ─────────────────────────────────────────────────────────────────────────────

/// Add two 3-vectors.
#[inline]
pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtract two 3-vectors.
#[inline]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scale a 3-vector by a scalar.
#[inline]
pub fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two 3-vectors.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3-vector.
#[inline]
pub fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Cross product of two 3-vectors.
#[inline]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// ODE system trait
// ─────────────────────────────────────────────────────────────────────────────

/// A continuous-time ODE system  `ẋ = f(t, x)`.
pub trait OdeSystem {
    /// Dimension of the state vector.
    fn dim(&self) -> usize;

    /// Right-hand side evaluated at time `t` and state `x`.
    fn rhs(&self, t: f64, x: &[f64]) -> Vec<f64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Runge–Kutta 4 integrator
// ─────────────────────────────────────────────────────────────────────────────

/// Classic 4th-order Runge–Kutta step.
///
/// Advances `x` by one step of size `h` under `sys` at time `t`,
/// returning the new state.
pub fn rk4_step(sys: &dyn OdeSystem, t: f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let k1 = sys.rhs(t, x);
    let x2: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * h * k1[i]).collect();
    let k2 = sys.rhs(t + 0.5 * h, &x2);
    let x3: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * h * k2[i]).collect();
    let k3 = sys.rhs(t + 0.5 * h, &x3);
    let x4: Vec<f64> = (0..n).map(|i| x[i] + h * k3[i]).collect();
    let k4 = sys.rhs(t + h, &x4);
    (0..n)
        .map(|i| x[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Integrate an ODE from `t0` to `t_end` with fixed step `h`.
///
/// Returns a vector of `(t, state)` samples including the initial condition.
pub fn integrate_rk4(
    sys: &dyn OdeSystem,
    x0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> Vec<(f64, Vec<f64>)> {
    let mut result = Vec::new();
    let mut t = t0;
    let mut x = x0.to_vec();
    result.push((t, x.clone()));
    while t + h <= t_end + 1e-12 {
        x = rk4_step(sys, t, &x, h);
        t += h;
        result.push((t, x.clone()));
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobian (numerical)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Jacobian matrix of `sys.rhs` at `(t, x)` numerically
/// using central differences with step `eps`.
///
/// Returns a row-major `n×n` matrix stored as `Vec`f64` of length `n*n`.
pub fn numerical_jacobian(sys: &dyn OdeSystem, t: f64, x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let mut jac = vec![0.0f64; n * n];
    for j in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += eps;
        xm[j] -= eps;
        let fp = sys.rhs(t, &xp);
        let fm = sys.rhs(t, &xm);
        for i in 0..n {
            jac[i * n + j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-point finder (Newton iteration)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a fixed-point search.
#[derive(Debug, Clone)]
pub struct FixedPointResult {
    /// The approximate fixed point (where `f(x*)≈0`).
    pub point: Vec<f64>,
    /// Residual norm at convergence.
    pub residual: f64,
    /// Whether the iteration converged within tolerance.
    pub converged: bool,
    /// Number of Newton iterations performed.
    pub iterations: usize,
}

/// Find a fixed point of `sys.rhs(0, x) = 0` using Newton–Raphson iteration
/// starting from `x0`.
///
/// `max_iter` limits the number of Newton steps; `tol` is the residual
/// stopping criterion.
pub fn find_fixed_point(
    sys: &dyn OdeSystem,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    eps_jac: f64,
) -> FixedPointResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    for iter in 0..max_iter {
        let f = sys.rhs(0.0, &x);
        let res: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        if res < tol {
            return FixedPointResult {
                point: x,
                residual: res,
                converged: true,
                iterations: iter,
            };
        }
        let jac = numerical_jacobian(sys, 0.0, &x, eps_jac);
        // Solve J * delta = -f  using Gaussian elimination (small n)
        let delta = gauss_solve(&jac, n, &f.iter().map(|v| -v).collect::<Vec<_>>());
        match delta {
            Some(d) => {
                for i in 0..n {
                    x[i] += d[i];
                }
            }
            None => break,
        }
    }
    let f = sys.rhs(0.0, &x);
    let res: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
    FixedPointResult {
        point: x,
        residual: res,
        converged: false,
        iterations: max_iter,
    }
}

/// Solve `A x = b` for `x` using Gaussian elimination with partial pivoting.
///
/// `a_flat` is a row-major `n×n` matrix; returns `None` if singular.
fn gauss_solve(a_flat: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut mat: Vec<f64> = a_flat.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();
    for col in 0..n {
        // partial pivot
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            rhs.swap(col, max_row);
        }
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let val = mat[col * n + k];
                mat[row * n + k] -= factor * val;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    // back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i * n + j] * x[j];
        }
        x[i] = sum / mat[i * n + i];
    }
    Some(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigenvalue analysis (2×2 and 3×3)
// ─────────────────────────────────────────────────────────────────────────────

/// Eigenvalues of a real 2×2 matrix `\[\[a,b\\],\[c,d\]]`.
///
/// Returns `(λ1, λ2)` which may be complex encoded as `(re±im)`.
/// The imaginary parts are returned as a separate pair.
pub fn eigen2(a: f64, b: f64, c: f64, d: f64) -> ((f64, f64), (f64, f64)) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        (((tr + sq) / 2.0, 0.0), ((tr - sq) / 2.0, 0.0))
    } else {
        let sq = (-disc).sqrt();
        ((tr / 2.0, sq / 2.0), (tr / 2.0, -sq / 2.0))
    }
}

/// Stability classification of a 2-D fixed point given the Jacobian entries.
///
/// Returns a human-readable label: `"stable_node"`, `"unstable_node"`,
/// `"saddle"`, `"stable_spiral"`, `"unstable_spiral"`, `"center"`.
pub fn stability_2d(a: f64, b: f64, c: f64, d: f64) -> &'static str {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if det < 0.0 {
        "saddle"
    } else if disc >= 0.0 {
        if tr < -1e-10 {
            "stable_node"
        } else if tr > 1e-10 {
            "unstable_node"
        } else {
            "center"
        }
    } else if tr < -1e-10 {
        "stable_spiral"
    } else if tr > 1e-10 {
        "unstable_spiral"
    } else {
        "center"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variational (tangent) flow
// ─────────────────────────────────────────────────────────────────────────────

/// Step of the central differences that build the Jacobians of the tangent
/// flow in the Lyapunov-exponent routines.
const TANGENT_JACOBIAN_EPS: f64 = 1e-6;

/// Product of the row-major square matrix `jac` with the vector `v`.
fn jacobian_times(jac: &[f64], v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    jac.chunks(v.len())
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// `J · q` for every tangent column `q`.
fn jacobian_times_all(jac: &[f64], tangents: &[Vec<f64>]) -> Vec<Vec<f64>> {
    tangents.iter().map(|q| jacobian_times(jac, q)).collect()
}

/// `q + weight · k` for every tangent column `q` with matching slope `k`.
fn offset_tangents(tangents: &[Vec<f64>], slopes: &[Vec<f64>], weight: f64) -> Vec<Vec<f64>> {
    tangents
        .iter()
        .zip(slopes.iter())
        .map(|(q, k)| {
            q.iter()
                .zip(k.iter())
                .map(|(qi, ki)| qi + weight * ki)
                .collect()
        })
        .collect()
}

/// `x + weight · k` componentwise.
fn offset_state(x: &[f64], slope: &[f64], weight: f64) -> Vec<f64> {
    x.iter()
        .zip(slope.iter())
        .map(|(xi, ki)| xi + weight * ki)
        .collect()
}

/// One classic Runge–Kutta 4 step of the augmented system
/// `(ẋ, q̇) = (f(t, x), J(t, x) q)` for the base state `x` and a set of
/// tangent vectors `tangents`.
///
/// Every tangent vector `q` obeys the variational equation
/// `q̇ = J(t, x(t)) q` with `J = ∂f/∂x`.  The base state and the tangent
/// vectors are advanced by the *same* RK4 scheme: the Jacobian of each of the
/// four stages is evaluated at that stage's own time and state
/// (`x`, `x + h/2·k₁`, `x + h/2·k₂`, `x + h·k₃`), so the tangent map is a
/// fourth-order accurate approximation of the derivative of the flow map.
/// Evaluating `J` once at the end of the step and advancing the tangents with
/// Euler would only be first-order accurate and biases Lyapunov exponents by
/// `O(h)`.
///
/// The Jacobians are formed by central differences with step `jac_eps`.
/// Returns the advanced state and the advanced tangent vectors (in the order
/// given).
pub fn rk4_step_with_tangents(
    sys: &dyn OdeSystem,
    t: f64,
    x: &[f64],
    tangents: &[Vec<f64>],
    h: f64,
    jac_eps: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let half = 0.5 * h;
    let k1 = sys.rhs(t, x);
    let kt1 = jacobian_times_all(&numerical_jacobian(sys, t, x, jac_eps), tangents);

    let x2 = offset_state(x, &k1, half);
    let k2 = sys.rhs(t + half, &x2);
    let jac2 = numerical_jacobian(sys, t + half, &x2, jac_eps);
    let kt2 = jacobian_times_all(&jac2, &offset_tangents(tangents, &kt1, half));

    let x3 = offset_state(x, &k2, half);
    let k3 = sys.rhs(t + half, &x3);
    let jac3 = numerical_jacobian(sys, t + half, &x3, jac_eps);
    let kt3 = jacobian_times_all(&jac3, &offset_tangents(tangents, &kt2, half));

    let x4 = offset_state(x, &k3, h);
    let k4 = sys.rhs(t + h, &x4);
    let jac4 = numerical_jacobian(sys, t + h, &x4, jac_eps);
    let kt4 = jacobian_times_all(&jac4, &offset_tangents(tangents, &kt3, h));

    let sixth = h / 6.0;
    let next_state = x
        .iter()
        .zip(k1.iter().zip(k2.iter()).zip(k3.iter().zip(k4.iter())))
        .map(|(xi, ((a, b), (c, d)))| xi + sixth * (a + 2.0 * b + 2.0 * c + d))
        .collect();
    let next_tangents = tangents
        .iter()
        .zip(kt1.iter().zip(kt2.iter()).zip(kt3.iter().zip(kt4.iter())))
        .map(|(q, ((a, b), (c, d)))| {
            q.iter()
                .zip(a.iter().zip(b.iter()).zip(c.iter().zip(d.iter())))
                .map(|(qi, ((ai, bi), (ci, di)))| qi + sixth * (ai + 2.0 * bi + 2.0 * ci + di))
                .collect()
        })
        .collect();
    (next_state, next_tangents)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lyapunov exponents (QR method)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the `n` Lyapunov exponents of an `n`-dimensional ODE system using
/// the QR-decomposition (Gram–Schmidt) method.
///
/// The variational equations are integrated simultaneously with the base
/// trajectory by [`rk4_step_with_tangents`] (Runge–Kutta 4 with the Jacobian
/// evaluated at every stage), and the tangent vectors are re-orthonormalised
/// after every step.  `n_steps` steps of size `h` are taken; the exponents
/// are accumulated at each step using log|R_ii|.  The base trajectory starts
/// at `x0` at time `0`, so a transient has to be discarded by the caller
/// (integrate onto the attractor first).
///
/// Returns a `Vec<f64>` of length `n` sorted in descending order; all zeros
/// if `n_steps == 0` or `h` is zero or not finite.
pub fn lyapunov_exponents_qr(sys: &dyn OdeSystem, x0: &[f64], h: f64, n_steps: usize) -> Vec<f64> {
    let n = sys.dim();
    if n_steps == 0 || h == 0.0 || !h.is_finite() {
        return vec![0.0; n];
    }
    let mut x = x0.to_vec();
    // Initialise Q as identity
    let mut q: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0f64; n];
            v[i] = 1.0;
            v
        })
        .collect();
    let mut sums = vec![0.0f64; n];
    let total_time = h * n_steps as f64;

    for step in 0..n_steps {
        let t = step as f64 * h;
        // Advance the base trajectory and every tangent vector with the same
        // RK4 scheme.
        let (x_next, new_q) = rk4_step_with_tangents(sys, t, &x, &q, h, TANGENT_JACOBIAN_EPS);
        x = x_next;
        // Gram–Schmidt orthonormalisation
        for col in 0..n {
            let mut v = new_q[col].clone();
            for qrow in q[..col].iter() {
                let proj: f64 = qrow.iter().zip(v.iter()).map(|(qi, vi)| qi * vi).sum();
                for (vi, qi) in v.iter_mut().zip(qrow.iter()) {
                    *vi -= proj * qi;
                }
            }
            let norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
            if norm > 1e-14 {
                q[col] = v.iter().map(|vi| vi / norm).collect();
            }
            sums[col] += if norm > 1e-14 { norm.ln() } else { -100.0 };
        }
    }
    let mut exponents: Vec<f64> = sums.iter().map(|s| s / total_time).collect();
    exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    exponents
}

// ─────────────────────────────────────────────────────────────────────────────
// Bifurcation detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detected bifurcation type.
#[derive(Debug, Clone, PartialEq)]
pub enum BifurcationType {
    /// Saddle-node: two fixed points collide and annihilate.
    SaddleNode,
    /// Transcritical: two fixed points exchange stability.
    Transcritical,
    /// Pitchfork (supercritical or subcritical): one fixed point splits into three.
    Pitchfork,
    /// Hopf: a fixed point loses stability and a limit cycle is born.
    Hopf,
    /// No bifurcation detected near the tested parameter value.
    None,
}

/// Result of a bifurcation scan along one parameter.
#[derive(Debug, Clone)]
pub struct BifurcationEvent {
    /// Parameter value at which the bifurcation was detected.
    pub param_value: f64,
    /// Type of the detected bifurcation.
    pub bif_type: BifurcationType,
    /// Fixed-point location at the bifurcation (if found).
    pub fixed_point: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenz system
// ─────────────────────────────────────────────────────────────────────────────

/// The Lorenz strange attractor.
///
/// ```text
/// ẋ = σ(y − x)
/// ẏ = x(ρ − z) − y
/// ż = xy − βz
/// ```
///
/// Classic chaos parameters: `σ=10`, `ρ=28`, `β=8/3`.
#[derive(Debug, Clone)]
pub struct LorenzSystem {
    /// Prandtl number σ.
    pub sigma: f64,
    /// Rayleigh number ρ.
    pub rho: f64,
    /// Geometric factor β.
    pub beta: f64,
}

impl LorenzSystem {
    /// Create a Lorenz system.
    pub fn new(sigma: f64, rho: f64, beta: f64) -> Self {
        Self { sigma, rho, beta }
    }

    /// Create the classic chaotic Lorenz system (σ=10, ρ=28, β=8/3).
    pub fn classic() -> Self {
        Self::new(10.0, 28.0, 8.0 / 3.0)
    }

    /// Fixed points of the Lorenz system.
    ///
    /// Returns origin, C+, and C- when `ρ > 1`, otherwise only the origin.
    pub fn fixed_points(&self) -> Vec<[f64; 3]> {
        let mut fps = vec![[0.0, 0.0, 0.0]];
        if self.rho > 1.0 {
            let c = (self.beta * (self.rho - 1.0)).sqrt();
            fps.push([c, c, self.rho - 1.0]);
            fps.push([-c, -c, self.rho - 1.0]);
        }
        fps
    }
}

impl OdeSystem for LorenzSystem {
    fn dim(&self) -> usize {
        3
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![
            self.sigma * (x[1] - x[0]),
            x[0] * (self.rho - x[2]) - x[1],
            x[0] * x[1] - self.beta * x[2],
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rössler system
// ─────────────────────────────────────────────────────────────────────────────

/// The Rössler system, a simpler strange attractor.
///
/// ```text
/// ẋ = −y − z
/// ẏ = x + ay
/// ż = b + z(x − c)
/// ```
///
/// Classic chaos: `a=0.2`, `b=0.2`, `c=5.7`.
#[derive(Debug, Clone)]
pub struct RosslerSystem {
    /// Parameter a.
    pub a: f64,
    /// Parameter b.
    pub b: f64,
    /// Parameter c.
    pub c: f64,
}

impl RosslerSystem {
    /// Create a Rössler system.
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    /// Classic chaotic parameters (a=0.2, b=0.2, c=5.7).
    pub fn classic() -> Self {
        Self::new(0.2, 0.2, 5.7)
    }
}

impl OdeSystem for RosslerSystem {
    fn dim(&self) -> usize {
        3
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![
            -x[1] - x[2],
            x[0] + self.a * x[1],
            self.b + x[2] * (x[0] - self.c),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Poincaré section
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a Poincaré section.
///
/// The section is defined by `x[axis] = level` with crossings in the
/// direction specified by `positive_direction`.
#[derive(Debug, Clone)]
pub struct PoincareSectionConfig {
    /// Index of the state variable defining the hyperplane.
    pub axis: usize,
    /// Value of `x[axis]` on the hyperplane.
    pub level: f64,
    /// If `true`, only record crossings where `x[axis]` is increasing.
    pub positive_direction: bool,
}

/// Record of a single Poincaré crossing.
#[derive(Debug, Clone)]
pub struct PoincareCrossing {
    /// Time of the crossing.
    pub time: f64,
    /// Full state vector at the crossing.
    pub state: Vec<f64>,
}

/// Compute a Poincaré section by integrating `sys` and recording crossings.
///
/// Integration uses RK4 with step `h` for `n_steps` steps.
pub fn poincare_section(
    sys: &dyn OdeSystem,
    x0: &[f64],
    h: f64,
    n_steps: usize,
    cfg: &PoincareSectionConfig,
) -> Vec<PoincareCrossing> {
    let mut crossings = Vec::new();
    let mut t = 0.0f64;
    let mut x_prev = x0.to_vec();
    let mut x = rk4_step(sys, t, &x_prev, h);
    for _i in 1..n_steps {
        let v_prev = x_prev[cfg.axis] - cfg.level;
        let v_cur = x[cfg.axis] - cfg.level;
        let crossing = if cfg.positive_direction {
            v_prev < 0.0 && v_cur >= 0.0
        } else {
            v_prev > 0.0 && v_cur <= 0.0
        };
        if crossing {
            // Linear interpolation to find crossing time
            let frac = v_prev.abs() / (v_prev.abs() + v_cur.abs() + 1e-30);
            let t_cross = t + frac * h;
            let state: Vec<f64> = x_prev
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a + frac * (b - a))
                .collect();
            crossings.push(PoincareCrossing {
                time: t_cross,
                state,
            });
        }
        x_prev = x.clone();
        x = rk4_step(sys, t, &x, h);
        t += h;
    }
    crossings
}

// ─────────────────────────────────────────────────────────────────────────────
// Limit cycle detection
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a limit-cycle period estimate.
#[derive(Debug, Clone)]
pub struct LimitCycleEstimate {
    /// Estimated period T.
    pub period: f64,
    /// Number of full cycles used in the estimate.
    pub n_cycles: usize,
    /// Whether a convincing cycle was found.
    pub found: bool,
}

/// Estimate the period of a limit cycle via successive Poincaré crossings.
///
/// Returns `None` if fewer than 2 crossings were found.
pub fn estimate_limit_cycle_period(
    sys: &dyn OdeSystem,
    x0: &[f64],
    h: f64,
    n_steps: usize,
    poincare_axis: usize,
    level: f64,
) -> Option<LimitCycleEstimate> {
    let cfg = PoincareSectionConfig {
        axis: poincare_axis,
        level,
        positive_direction: true,
    };
    let crossings = poincare_section(sys, x0, h, n_steps, &cfg);
    if crossings.len() < 2 {
        return None;
    }
    let n = crossings.len();
    let total_time = crossings[n - 1].time - crossings[0].time;
    let period = total_time / (n - 1) as f64;
    Some(LimitCycleEstimate {
        period,
        n_cycles: n - 1,
        found: true,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Recurrence Quantification Analysis (RQA)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for RQA computation.
#[derive(Debug, Clone)]
pub struct RqaParams {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Time delay τ (in samples).
    pub tau: usize,
    /// Recurrence threshold ε.
    pub epsilon: f64,
    /// Minimum diagonal line length.
    pub min_line: usize,
}

/// Results of a recurrence quantification analysis.
#[derive(Debug, Clone)]
pub struct RqaResult {
    /// Recurrence Rate (RR): fraction of recurrence points.
    pub rr: f64,
    /// Determinism (DET): fraction of recurrence points on diagonal lines.
    pub det: f64,
    /// Average diagonal line length (L).
    pub avg_line: f64,
    /// Longest diagonal line length (L_max).
    pub l_max: usize,
    /// Entropy of diagonal line-length distribution.
    pub entropy: f64,
    /// Laminarity (LAM): fraction of recurrence points in vertical lines.
    pub lam: f64,
    /// Trapping time (TT): average length of vertical lines.
    pub tt: f64,
}

/// Build the delay-embedded trajectory from a scalar time series.
///
/// Returns a `Vec<Vec`f64`>` where each inner vector has length `embed_dim`.
pub fn delay_embed(series: &[f64], embed_dim: usize, tau: usize) -> Vec<Vec<f64>> {
    let n = series.len();
    let max_start = (embed_dim - 1) * tau;
    if n <= max_start {
        return Vec::new();
    }
    (0..=(n - 1 - max_start))
        .map(|i| (0..embed_dim).map(|d| series[i + d * tau]).collect())
        .collect()
}

/// Compute the recurrence matrix (as a flat `bool` vector, row-major).
///
/// `points[i]` are the embedded vectors.  Entry `(i,j)` is `true` when
/// `‖points[i] − points[j]‖ < epsilon`.
pub fn recurrence_matrix(points: &[Vec<f64>], epsilon: f64) -> Vec<bool> {
    let n = points.len();
    let mut mat = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            let dist: f64 = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            mat[i * n + j] = dist < epsilon;
        }
    }
    mat
}

/// Perform full RQA on a scalar time series.
///
/// Uses Chebyshev (max-norm) distance in embedding space.
pub fn rqa(series: &[f64], params: &RqaParams) -> RqaResult {
    let points = delay_embed(series, params.embed_dim, params.tau);
    let n = points.len();
    if n == 0 {
        return RqaResult {
            rr: 0.0,
            det: 0.0,
            avg_line: 0.0,
            l_max: 0,
            entropy: 0.0,
            lam: 0.0,
            tt: 0.0,
        };
    }
    // Build recurrence matrix (exclude identity)
    let mat = recurrence_matrix(&points, params.epsilon);

    // Recurrence rate (exclude diagonal)
    let total_off = (n * n - n) as f64;
    let recurrent_pts: usize = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j && mat[i * n + j])
        .count();
    let rr = if total_off > 0.0 {
        recurrent_pts as f64 / total_off
    } else {
        0.0
    };

    // Diagonal line statistics
    let mut diag_lines: Vec<usize> = Vec::new();
    for diag in (-(n as isize - 1))..(n as isize) {
        if diag == 0 {
            continue;
        }
        let mut run = 0usize;
        let i_start = if diag < 0 { (-diag) as usize } else { 0 };
        let j_start = if diag > 0 { diag as usize } else { 0 };
        let len = n - i_start.max(j_start);
        for k in 0..len {
            let i = i_start + k;
            let j = j_start + k;
            if mat[i * n + j] {
                run += 1;
            } else {
                if run >= params.min_line {
                    diag_lines.push(run);
                }
                run = 0;
            }
        }
        if run >= params.min_line {
            diag_lines.push(run);
        }
    }
    let diag_total: usize = diag_lines.iter().sum();
    let det = if recurrent_pts > 0 {
        diag_total as f64 / recurrent_pts as f64
    } else {
        0.0
    };
    let l_max = diag_lines.iter().copied().max().unwrap_or(0);
    let avg_line = if !diag_lines.is_empty() {
        diag_total as f64 / diag_lines.len() as f64
    } else {
        0.0
    };
    // Shannon entropy of line-length distribution
    let entropy = if !diag_lines.is_empty() {
        let total = diag_total as f64;
        let mut ent = 0.0f64;
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &l in &diag_lines {
            *counts.entry(l).or_insert(0) += 1;
        }
        for &cnt in counts.values() {
            let p = cnt as f64 / total;
            if p > 0.0 {
                ent -= p * p.ln();
            }
        }
        ent
    } else {
        0.0
    };

    // Vertical line statistics (laminarity / trapping time)
    let mut vert_lines: Vec<usize> = Vec::new();
    for col in 0..n {
        let mut run = 0usize;
        for row in 0..n {
            if row != col && mat[row * n + col] {
                run += 1;
            } else {
                if run >= params.min_line {
                    vert_lines.push(run);
                }
                run = 0;
            }
        }
        if run >= params.min_line {
            vert_lines.push(run);
        }
    }
    let vert_total: usize = vert_lines.iter().sum();
    let lam = if recurrent_pts > 0 {
        vert_total as f64 / recurrent_pts as f64
    } else {
        0.0
    };
    let tt = if !vert_lines.is_empty() {
        vert_total as f64 / vert_lines.len() as f64
    } else {
        0.0
    };

    RqaResult {
        rr,
        det,
        avg_line,
        l_max,
        entropy,
        lam,
        tt,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamiltonian systems
// ─────────────────────────────────────────────────────────────────────────────

/// A Hamiltonian system with `n` degrees of freedom.
///
/// State is `(q1,…,qn, p1,…,pn)`.  Implement `hamiltonian()`,
/// `dh_dq()`, and `dh_dp()`.
pub trait HamiltonianSystem {
    /// Number of degrees of freedom.
    fn ndof(&self) -> usize;

    /// Evaluate the Hamiltonian `H(q, p)`.
    fn hamiltonian(&self, q: &[f64], p: &[f64]) -> f64;

    /// Partial derivatives `∂H/∂q` (forces, length `ndof`).
    fn dh_dq(&self, q: &[f64], p: &[f64]) -> Vec<f64>;

    /// Partial derivatives `∂H/∂p` (velocities, length `ndof`).
    fn dh_dp(&self, q: &[f64], p: &[f64]) -> Vec<f64>;
}

/// Wrap a `HamiltonianSystem` as a plain `OdeSystem`.
///
/// Hamilton's equations: `q̇ = ∂H/∂p`, `ṗ = −∂H/∂q`.
pub struct HamiltonianOde<'a> {
    /// The underlying Hamiltonian.
    pub ham: &'a dyn HamiltonianSystem,
}

impl<'a> OdeSystem for HamiltonianOde<'a> {
    fn dim(&self) -> usize {
        2 * self.ham.ndof()
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        let n = self.ham.ndof();
        let q = &x[..n];
        let p = &x[n..];
        let qdot = self.ham.dh_dp(q, p);
        let dhdq = self.ham.dh_dq(q, p);
        let pdot: Vec<f64> = dhdq.iter().map(|v| -v).collect();
        [qdot, pdot].concat()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symplectic integration (Störmer–Verlet / leapfrog)
// ─────────────────────────────────────────────────────────────────────────────

/// One Störmer–Verlet (leapfrog) step for a separable Hamiltonian
/// `H = T(p) + V(q)`.
///
/// `grad_v` computes `∂V/∂q` (force = `−grad_v`).
/// Returns the updated `(q_new, p_new)`.
pub fn stormer_verlet_step(
    q: &[f64],
    p: &[f64],
    h: f64,
    grad_v: &dyn Fn(&[f64]) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = q.len();
    let f = grad_v(q);
    // half kick
    let p_half: Vec<f64> = (0..n).map(|i| p[i] - 0.5 * h * f[i]).collect();
    // drift
    let q_new: Vec<f64> = (0..n).map(|i| q[i] + h * p_half[i]).collect();
    // second gradient
    let f2 = grad_v(&q_new);
    // half kick
    let p_new: Vec<f64> = (0..n).map(|i| p_half[i] - 0.5 * h * f2[i]).collect();
    (q_new, p_new)
}

/// 4th-order symplectic Forest–Ruth integrator.
///
/// Coefficients from Forest & Ruth (1990).
pub fn forest_ruth_step(
    q: &[f64],
    p: &[f64],
    h: f64,
    grad_v: &dyn Fn(&[f64]) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let theta: f64 = 1.0 / (2.0 - 2.0f64.powf(1.0 / 3.0));
    let xi = 1.0 - 2.0 * theta;
    let n = q.len();
    let mut qq = q.to_vec();
    let mut pp = p.to_vec();
    for &c in &[theta, xi, theta] {
        // drift
        let f = grad_v(&qq);
        for i in 0..n {
            pp[i] -= 0.5 * c * h * f[i];
        }
        for i in 0..n {
            qq[i] += c * h * pp[i];
        }
        let f2 = grad_v(&qq);
        for i in 0..n {
            pp[i] -= 0.5 * c * h * f2[i];
        }
    }
    (qq, pp)
}

/// Integrate a Hamiltonian system using the Störmer–Verlet method.
///
/// Returns `Vec<(q, p)>` snapshots at every step.
pub fn integrate_symplectic(
    q0: &[f64],
    p0: &[f64],
    h: f64,
    n_steps: usize,
    grad_v: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();
    traj.push((q.clone(), p.clone()));
    for _ in 0..n_steps {
        let (qn, pn) = stormer_verlet_step(&q, &p, h, grad_v);
        q = qn;
        p = pn;
        traj.push((q.clone(), p.clone()));
    }
    traj
}

// ─────────────────────────────────────────────────────────────────────────────
// KAM torus winding number
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the winding number (frequency ratio) of a 2-DOF Hamiltonian orbit
/// on a KAM torus using the mean frequency ratio from angle variables.
///
/// Angles are estimated from successive Poincaré crossings of the `(q[0], p[0])`
/// sub-plane.  Returns the ratio `ω1/ω2`.
pub fn estimate_winding_number(q_traj: &[(Vec<f64>, Vec<f64>)], _period: f64) -> Option<f64> {
    // Count angle-variable increments by tracking q[0] sign changes
    if q_traj.len() < 4 {
        return None;
    }
    let mut crossings_0: usize = 0;
    let mut crossings_1: usize = 0;
    for w in q_traj.windows(2) {
        if w[0].0[0] * w[1].0[0] < 0.0 {
            crossings_0 += 1;
        }
        if w[0].0.len() > 1 && w[1].0.len() > 1 && w[0].0[1] * w[1].0[1] < 0.0 {
            crossings_1 += 1;
        }
    }
    if crossings_1 == 0 {
        return None;
    }
    Some(crossings_0 as f64 / crossings_1 as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase plane helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a vector field on a 2-D grid for phase-plane visualisation.
///
/// `x_range = (x_min, x_max, nx)`, `y_range = (y_min, y_max, ny)`.
/// Returns `Vec<(x, y, dx, dy)>` — one entry per grid point.
pub fn phase_plane_grid(
    sys: &dyn OdeSystem,
    x_range: (f64, f64, usize),
    y_range: (f64, f64, usize),
) -> Vec<(f64, f64, f64, f64)> {
    let (x_min, x_max, nx) = x_range;
    let (y_min, y_max, ny) = y_range;
    let mut out = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        let xv = x_min + (x_max - x_min) * i as f64 / (nx.saturating_sub(1).max(1)) as f64;
        for j in 0..ny {
            let yv = y_min + (y_max - y_min) * j as f64 / (ny.saturating_sub(1).max(1)) as f64;
            let f = sys.rhs(0.0, &[xv, yv]);
            out.push((xv, yv, f[0], f[1]));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Van der Pol oscillator (example 2-D system)
// ─────────────────────────────────────────────────────────────────────────────

/// Van der Pol oscillator:
///
/// ```text
/// ẋ = y
/// ẏ = μ(1 − x²)y − x
/// ```
///
/// Exhibits a stable limit cycle for any `μ > 0`.
#[derive(Debug, Clone)]
pub struct VanDerPol {
    /// Nonlinearity parameter μ.
    pub mu: f64,
}

impl VanDerPol {
    /// Create a Van der Pol oscillator.
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }
}

impl OdeSystem for VanDerPol {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![x[1], self.mu * (1.0 - x[0] * x[0]) * x[1] - x[0]]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Double pendulum (Hamiltonian)
// ─────────────────────────────────────────────────────────────────────────────

/// Double pendulum: two point masses `m = 1` on massless rods of length `l = 1`.
///
/// With `Δ = θ1 − θ2` the Lagrangian is
/// `L = θ̇1² + θ̇2²/2 + θ̇1 θ̇2 cos Δ + g (2 cos θ1 + cos θ2)`, hence the
/// canonical momenta `pθ1 = 2 θ̇1 + θ̇2 cos Δ`, `pθ2 = θ̇2 + θ̇1 cos Δ` and the
/// Hamiltonian
/// `H = (pθ1² + 2 pθ2² − 2 pθ1 pθ2 cos Δ) / (2 (1 + sin² Δ)) − g (2 cos θ1 + cos θ2)`.
///
/// Canonical coordinates: `q = (θ1, θ2)`, `p = (pθ1, pθ2)`; both angles are
/// measured from the downward vertical.
#[derive(Debug, Clone)]
pub struct DoublePendulum {
    /// Gravitational acceleration.
    pub g: f64,
}

impl DoublePendulum {
    /// Create a double pendulum.
    pub fn new(g: f64) -> Self {
        Self { g }
    }

    /// Integrate using 4th-order Runge–Kutta for `n` steps of size `h`.
    ///
    /// State: `[θ1, θ2, pθ1, pθ2]`.
    pub fn integrate(&self, state0: [f64; 4], h: f64, n: usize) -> Vec<[f64; 4]> {
        let mut traj = Vec::with_capacity(n + 1);
        let mut s = state0;
        traj.push(s);
        for _ in 0..n {
            let rhs = self.rhs_arr(s);
            let k1 = rhs;
            let s2 = [
                s[0] + 0.5 * h * k1[0],
                s[1] + 0.5 * h * k1[1],
                s[2] + 0.5 * h * k1[2],
                s[3] + 0.5 * h * k1[3],
            ];
            let k2 = self.rhs_arr(s2);
            let s3 = [
                s[0] + 0.5 * h * k2[0],
                s[1] + 0.5 * h * k2[1],
                s[2] + 0.5 * h * k2[2],
                s[3] + 0.5 * h * k2[3],
            ];
            let k3 = self.rhs_arr(s3);
            let s4 = [
                s[0] + h * k3[0],
                s[1] + h * k3[1],
                s[2] + h * k3[2],
                s[3] + h * k3[3],
            ];
            let k4 = self.rhs_arr(s4);
            for i in 0..4 {
                s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            traj.push(s);
        }
        traj
    }

    /// `(θ̇1, θ̇2, sin Δ)` for the state `[θ1, θ2, pθ1, pθ2]`: the momenta
    /// multiplied by the inverse of the mass matrix `[[2, cos Δ], [cos Δ, 1]]`,
    /// whose determinant is `1 + sin² Δ`.
    fn angular_velocities(s: [f64; 4]) -> (f64, f64, f64) {
        let (sin_delta, cos_delta) = (s[0] - s[1]).sin_cos();
        let det = 1.0 + sin_delta * sin_delta;
        (
            (s[2] - cos_delta * s[3]) / det,
            (2.0 * s[3] - cos_delta * s[2]) / det,
            sin_delta,
        )
    }

    /// Hamilton's equations for `[θ1, θ2, pθ1, pθ2]`:
    /// `θ̇ = ∂H/∂p`, `ṗθ1 = −θ̇1 θ̇2 sin Δ − 2 g sin θ1`,
    /// `ṗθ2 = +θ̇1 θ̇2 sin Δ − g sin θ2`.
    fn rhs_arr(&self, s: [f64; 4]) -> [f64; 4] {
        let (t1dot, t2dot, sin_delta) = Self::angular_velocities(s);
        let coupling = t1dot * t2dot * sin_delta;
        [
            t1dot,
            t2dot,
            -coupling - 2.0 * self.g * s[0].sin(),
            coupling - self.g * s[1].sin(),
        ]
    }

    /// Total energy `H = T + V` of the double pendulum, with the kinetic
    /// energy `T = (pθ1 θ̇1 + pθ2 θ̇2) / 2` and `V = −g (2 cos θ1 + cos θ2)`.
    pub fn energy(&self, s: [f64; 4]) -> f64 {
        let (t1dot, t2dot, _) = Self::angular_velocities(s);
        let kinetic = 0.5 * (s[2] * t1dot + s[3] * t2dot);
        let potential = -self.g * (2.0 * s[0].cos() + s[1].cos());
        kinetic + potential
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Duffing oscillator
// ─────────────────────────────────────────────────────────────────────────────

/// Duffing oscillator:
///
/// ```text
/// ẋ = y
/// ẏ = −δ y − α x − β x³ + γ cos(ω t)
/// ```
#[derive(Debug, Clone)]
pub struct DuffingOscillator {
    /// Damping coefficient δ.
    pub delta: f64,
    /// Linear stiffness α.
    pub alpha: f64,
    /// Cubic stiffness β.
    pub beta: f64,
    /// Forcing amplitude γ.
    pub gamma: f64,
    /// Forcing frequency ω.
    pub omega: f64,
}

impl DuffingOscillator {
    /// Create a Duffing oscillator.
    pub fn new(delta: f64, alpha: f64, beta: f64, gamma: f64, omega: f64) -> Self {
        Self {
            delta,
            alpha,
            beta,
            gamma,
            omega,
        }
    }
}

impl OdeSystem for DuffingOscillator {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, t: f64, x: &[f64]) -> Vec<f64> {
        vec![
            x[1],
            -self.delta * x[1] - self.alpha * x[0] - self.beta * x[0].powi(3)
                + self.gamma * (self.omega * t).cos(),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bifurcation normal form helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse a 1-D saddle-node bifurcation `ẋ = μ + x²`.
///
/// Returns the fixed points for a given parameter `mu`.
pub fn saddle_node_fixed_points(mu: f64) -> Vec<f64> {
    if mu > 0.0 {
        Vec::new()
    } else if mu == 0.0 {
        vec![0.0]
    } else {
        let sq = (-mu).sqrt();
        vec![-sq, sq]
    }
}

/// Analyse a 1-D pitchfork bifurcation (supercritical) `ẋ = μ x − x³`.
///
/// Returns the stable fixed points.
pub fn pitchfork_fixed_points(mu: f64) -> Vec<f64> {
    if mu <= 0.0 {
        vec![0.0]
    } else {
        let sq = mu.sqrt();
        vec![-sq, 0.0, sq]
    }
}

/// Check whether a 2-D system near a fixed point undergoes a Hopf bifurcation
/// at the given Jacobian parameters.
///
/// A Hopf bifurcation occurs when the Jacobian has purely imaginary eigenvalues:
/// `trace(J) = 0`, `det(J) > 0`.
///
/// Returns `true` if the conditions are (approximately) met.
pub fn is_hopf_bifurcation(tr: f64, det: f64, tol: f64) -> bool {
    det > tol && tr.abs() < tol
}

// ─────────────────────────────────────────────────────────────────────────────
// Chaos indicator: maximal Lyapunov exponent (fast Benettin)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the maximal Lyapunov exponent (MLE) using the Benettin algorithm.
///
/// A single tangent vector is integrated alongside the base trajectory by
/// [`rk4_step_with_tangents`] (Runge–Kutta 4 with the Jacobian evaluated at
/// every stage of the base step) and periodically renormalised.  The log of
/// the growth rate is accumulated.  The base trajectory starts at `x0` at
/// time `0`, so a transient has to be discarded by the caller (integrate onto
/// the attractor first).
///
/// # Arguments
/// * `sys`     — the ODE system
/// * `x0`      — initial state
/// * `h`       — step size
/// * `n_steps` — total integration steps
/// * `renorm`  — steps between renormalisations; `0` is treated as `1`
///   (renormalise after every step), the smallest meaningful interval
///
/// Returns `0.0` if no renormalisation took place (for example `n_steps`
/// smaller than `renorm`, or a vanishing tangent vector) or if `h` is zero or
/// not finite.
pub fn maximal_lyapunov_exponent(
    sys: &dyn OdeSystem,
    x0: &[f64],
    h: f64,
    n_steps: usize,
    renorm: usize,
) -> f64 {
    if h == 0.0 || !h.is_finite() {
        return 0.0;
    }
    let renorm = renorm.max(1);
    let n = sys.dim();
    let mut x = x0.to_vec();
    // Initial tangent vector: unit vector along the first coordinate.
    let mut tangents: Vec<Vec<f64>> =
        vec![(0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect()];
    let mut sum_log = 0.0f64;
    let mut count = 0usize;
    for step in 0..n_steps {
        let t = step as f64 * h;
        let (x_next, next_tangents) =
            rk4_step_with_tangents(sys, t, &x, &tangents, h, TANGENT_JACOBIAN_EPS);
        x = x_next;
        tangents = next_tangents;
        // Renormalise
        if (step + 1) % renorm == 0 {
            for dv in tangents.iter_mut() {
                let norm: f64 = dv.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-14 {
                    sum_log += norm.ln();
                    for vi in dv.iter_mut() {
                        *vi /= norm;
                    }
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    sum_log / (count * renorm) as f64 / h
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy drift monitor
// ─────────────────────────────────────────────────────────────────────────────

/// Monitor energy conservation in a Hamiltonian trajectory.
///
/// Returns the maximum relative energy drift `|ΔH/H₀|` over the trajectory.
pub fn energy_drift(ham: &dyn HamiltonianSystem, traj: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    if traj.is_empty() {
        return 0.0;
    }
    let n = ham.ndof();
    let (q0, p0) = &traj[0];
    let h0 = ham.hamiltonian(q0, p0);
    if h0.abs() < 1e-30 {
        return 0.0;
    }
    traj.iter()
        .skip(1)
        .map(|(q, p)| {
            let _ = n;
            ((ham.hamiltonian(q, p) - h0) / h0).abs()
        })
        .fold(0.0f64, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple harmonic oscillator (Hamiltonian)
// ─────────────────────────────────────────────────────────────────────────────

/// 1-DOF simple harmonic oscillator: `H = p²/2 + ω²q²/2`.
#[derive(Debug, Clone)]
pub struct HarmonicOscillatorHam {
    /// Angular frequency ω.
    pub omega: f64,
}

impl HarmonicOscillatorHam {
    /// Create a harmonic oscillator Hamiltonian.
    pub fn new(omega: f64) -> Self {
        Self { omega }
    }
}

impl HamiltonianSystem for HarmonicOscillatorHam {
    fn ndof(&self) -> usize {
        1
    }
    fn hamiltonian(&self, q: &[f64], p: &[f64]) -> f64 {
        0.5 * p[0] * p[0] + 0.5 * self.omega * self.omega * q[0] * q[0]
    }
    fn dh_dq(&self, q: &[f64], _p: &[f64]) -> Vec<f64> {
        vec![self.omega * self.omega * q[0]]
    }
    fn dh_dp(&self, _q: &[f64], p: &[f64]) -> Vec<f64> {
        vec![p[0]]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Saddle connection / heteroclinic / homoclinic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for heteroclinic orbit detection.
#[derive(Debug, Clone)]
pub struct HeteroclinicConfig {
    /// Perturbation magnitude along the unstable direction from `fp_a`.
    pub eps_perturb: f64,
    /// Integration step size.
    pub h: f64,
    /// Maximum number of integration steps.
    pub n_steps: usize,
    /// Convergence tolerance (distance to `fp_b`).
    pub tol: f64,
}

/// Check whether two fixed points are connected by a heteroclinic orbit.
///
/// A rudimentary test: integrate from near the unstable manifold of `fp_a`
/// and check if the trajectory approaches `fp_b` within `cfg.tol`.
///
/// Returns the time at which the trajectory first enters the `cfg.tol`-ball
/// around `fp_b`, or `None` if it does not within `cfg.n_steps` steps.
pub fn detect_heteroclinic(
    sys: &dyn OdeSystem,
    fp_a: &[f64],
    fp_b: &[f64],
    unstable_dir: &[f64],
    cfg: &HeteroclinicConfig,
) -> Option<f64> {
    let _n = fp_a.len();
    let mut x: Vec<f64> = fp_a
        .iter()
        .zip(unstable_dir.iter())
        .map(|(&a, &d)| a + cfg.eps_perturb * d)
        .collect();
    let mut t = 0.0f64;
    for _ in 0..cfg.n_steps {
        let dist: f64 = x
            .iter()
            .zip(fp_b.iter())
            .map(|(&xi, &bi)| (xi - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        if dist < cfg.tol {
            return Some(t);
        }
        x = rk4_step(sys, t, &x, cfg.h);
        t += cfg.h;
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Attractor dimension (Kaplan–Yorke)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the Kaplan–Yorke dimension from sorted Lyapunov exponents.
///
/// `exponents` must be sorted in descending order (largest first).
/// Returns the KY dimension `D = j + (λ1 + … + λj)/|λ_{j+1}|`
/// where `j` is the largest index such that the sum is non-negative.
/// An empty spectrum has dimension `0`.
pub fn kaplan_yorke_dim(exponents: &[f64]) -> f64 {
    let n = exponents.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut j = 0usize;
    for (i, &le) in exponents.iter().enumerate() {
        sum += le;
        if sum < 0.0 {
            j = i;
            sum -= le;
            break;
        }
        if i == n - 1 {
            return n as f64;
        }
        j = i + 1;
    }
    if j == 0 && exponents[0] < 0.0 {
        return 0.0;
    }
    let denom = exponents[j].abs();
    if denom < 1e-14 {
        return j as f64;
    }
    j as f64 + sum / denom
}

// ─────────────────────────────────────────────────────────────────────────────
// Coupled oscillators (Kuramoto model)
// ─────────────────────────────────────────────────────────────────────────────

/// Kuramoto model of `n` coupled phase oscillators.
///
/// ```text
/// θ̇ᵢ = ωᵢ + (K/n) Σⱼ sin(θⱼ − θᵢ)
/// ```
#[derive(Debug, Clone)]
pub struct KuramotoModel {
    /// Natural frequencies ωᵢ.
    pub omegas: Vec<f64>,
    /// Coupling strength K.
    pub k: f64,
}

impl KuramotoModel {
    /// Create a Kuramoto model.
    pub fn new(omegas: Vec<f64>, k: f64) -> Self {
        Self { omegas, k }
    }

    /// Compute the order parameter `r exp(iψ)` where `r` measures synchronisation.
    ///
    /// Returns `(r, psi)`.
    pub fn order_parameter(theta: &[f64]) -> (f64, f64) {
        let n = theta.len() as f64;
        let re: f64 = theta.iter().map(|&t| t.cos()).sum::<f64>() / n;
        let im: f64 = theta.iter().map(|&t| t.sin()).sum::<f64>() / n;
        let r = (re * re + im * im).sqrt();
        let psi = im.atan2(re);
        (r, psi)
    }
}

impl OdeSystem for KuramotoModel {
    fn dim(&self) -> usize {
        self.omegas.len()
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let k_over_n = self.k / n as f64;
        (0..n)
            .map(|i| {
                let coupling: f64 = (0..n).map(|j| (x[j] - x[i]).sin()).sum();
                self.omegas[i] + k_over_n * coupling
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Brusselator (chemical oscillator, Hopf bifurcation example)
// ─────────────────────────────────────────────────────────────────────────────

/// Brusselator chemical oscillator.
///
/// ```text
/// ẋ = a − (b+1)x + x²y
/// ẏ = bx − x²y
/// ```
///
/// Undergoes a Hopf bifurcation at `b = 1 + a²`.
#[derive(Debug, Clone)]
pub struct Brusselator {
    /// Parameter a.
    pub a: f64,
    /// Parameter b.
    pub b: f64,
}

impl Brusselator {
    /// Create a Brusselator.
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Fixed point: `(a, b/a)`.
    pub fn fixed_point(&self) -> (f64, f64) {
        (self.a, self.b / self.a)
    }

    /// Hopf bifurcation value of b for the given a.
    pub fn hopf_b(&self) -> f64 {
        1.0 + self.a * self.a
    }
}

impl OdeSystem for Brusselator {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, _t: f64, x: &[f64]) -> Vec<f64> {
        vec![
            self.a - (self.b + 1.0) * x[0] + x[0] * x[0] * x[1],
            self.b * x[0] - x[0] * x[0] * x[1],
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// #[cfg(test)] unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lyapunov_tests;

#[cfg(test)]
#[path = "dynamical_systems_double_pendulum_tests.rs"]
mod double_pendulum_tests;
