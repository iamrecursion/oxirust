// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Nonlinear structural dynamics: Duffing oscillator, Lyapunov exponents,
//! Poincaré sections, bifurcation diagrams, nonlinear FEM with Newton–Raphson,
//! and classical discrete maps (logistic, Hénon).
//!
//! All computations are self-contained (no nalgebra); matrices are represented
//! as flat `Vec`f64` in row-major order.

use crate::error::{Error, Result};
use std::f64::consts::PI;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  HELPER – FEIGENBAUM CONSTANT AND STANDALONE FREE FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

/// Feigenbaum's first constant δ ≈ 4.669 201 609 … (period-doubling ratio).
///
/// Appears as the universal ratio of successive period-doubling bifurcation
/// intervals for one-dimensional maps.
pub fn period_doubling_ratio() -> f64 {
    4.669_201_609_102_99
}

/// Evaluate one step of the logistic map: xₙ₊₁ = r · xₙ · (1 − xₙ).
///
/// # Parameters
/// * `r` – growth (bifurcation) parameter (typically 0 ≤ r ≤ 4).
/// * `x` – current state (0 ≤ x ≤ 1).
pub fn logistic_map(r: f64, x: f64) -> f64 {
    r * x * (1.0 - x)
}

/// Evaluate one step of the Hénon map.
///
/// Returns `\[x_new, y_new\]` where
/// `x_new = 1 − a·x² + y`, `y_new = b·x`.
///
/// Classical parameters: `a = 1.4`, `b = 0.3`.
pub fn henon_map(x: f64, y: f64, a: f64, b: f64) -> [f64; 2] {
    [1.0 - a * x * x + y, b * x]
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  DUFFING OSCILLATOR
// ══════════════════════════════════════════════════════════════════════════════

/// Forced Duffing oscillator:
/// ẍ + δ·ẋ − α·x + β·x³ = γ·cos(ωd·t)
///
/// Fields
/// * `alpha` – linear coefficient of the restoring force `α·x` (positive → negative
///   linear stiffness, i.e. a double-well potential for `β > 0`; negative → a single
///   well with a hardening spring).
/// * `beta`  – nonlinear (cubic) stiffness coefficient.
/// * `delta` – linear damping coefficient.
/// * `gamma` – forcing amplitude.
/// * `omega_d` – driving angular frequency.
#[derive(Debug, Clone)]
pub struct DuffingOscillator {
    /// Linear stiffness coefficient α.
    pub alpha: f64,
    /// Cubic stiffness coefficient β.
    pub beta: f64,
    /// Damping coefficient δ.
    pub delta: f64,
    /// Forcing amplitude γ.
    pub gamma: f64,
    /// Driving angular frequency ωd.
    pub omega_d: f64,
}

impl DuffingOscillator {
    /// Create a Duffing oscillator with the given parameters.
    pub fn new(alpha: f64, beta: f64, delta: f64, gamma: f64, omega_d: f64) -> Self {
        Self {
            alpha,
            beta,
            delta,
            gamma,
            omega_d,
        }
    }

    /// Evaluate the right-hand side at time `t`, displacement `x`, velocity `v`.
    ///
    /// Returns `\[ẋ, ẍ\]`, i.e. `\[v, rhs_acceleration\]`.
    pub fn response(&self, t: f64, x: f64, v: f64) -> [f64; 2] {
        let accel = -self.delta * v + self.alpha * x - self.beta * x.powi(3)
            + self.gamma * (self.omega_d * t).cos();
        [v, accel]
    }

    /// Integrate the phase-space trajectory using RK4 for `n_steps` steps of
    /// size `dt` starting from `(x0, v0)` at `t0`.
    ///
    /// Returns a `Vec<\[f64; 2\]>` of `\[x, v\]` pairs at each step.
    pub fn phase_portrait(
        &self,
        x0: f64,
        v0: f64,
        t0: f64,
        n_steps: usize,
        dt: f64,
    ) -> Vec<[f64; 2]> {
        let mut traj = Vec::with_capacity(n_steps + 1);
        let mut x = x0;
        let mut v = v0;
        let mut t = t0;
        traj.push([x, v]);
        for _ in 0..n_steps {
            let [dx1, dv1] = self.response(t, x, v);
            let [dx2, dv2] = self.response(t + 0.5 * dt, x + 0.5 * dt * dx1, v + 0.5 * dt * dv1);
            let [dx3, dv3] = self.response(t + 0.5 * dt, x + 0.5 * dt * dx2, v + 0.5 * dt * dv2);
            let [dx4, dv4] = self.response(t + dt, x + dt * dx3, v + dt * dv3);
            x += dt / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
            v += dt / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
            t += dt;
            traj.push([x, v]);
        }
        traj
    }

    /// Vector field of the state `[x, v]` extended by tangent (variational) vectors.
    ///
    /// `state` holds `[x, v, δx₁, δv₁, δx₂, δv₂, …]`.  The base flow is
    /// [`response`](Self::response); every tangent vector follows the linearised flow
    /// `δẋ = δv`, `δv̇ = (α − 3β x²)·δx − δ·δv`, whose Jacobian does not depend on the
    /// (additive) forcing.
    fn extended_response<const N: usize>(&self, t: f64, state: &[f64; N]) -> [f64; N] {
        let [dx, dv] = self.response(t, state[0], state[1]);
        let mut out = [0.0; N];
        out[0] = dx;
        out[1] = dv;
        let stiffness = self.alpha - 3.0 * self.beta * state[0] * state[0];
        for pair in (2..N.saturating_sub(1)).step_by(2) {
            out[pair] = state[pair + 1];
            out[pair + 1] = stiffness * state[pair] - self.delta * state[pair + 1];
        }
        out
    }

    /// One classical RK4 step of the extended system, so that the tangent vectors see
    /// exactly the same intermediate base-flow stages as the trajectory itself.
    fn rk4_extended<const N: usize>(&self, t: f64, state: &[f64; N], dt: f64) -> [f64; N] {
        let advance =
            |k: &[f64; N], h: f64| -> [f64; N] { std::array::from_fn(|i| state[i] + h * k[i]) };
        let k1 = self.extended_response(t, state);
        let k2 = self.extended_response(t + 0.5 * dt, &advance(&k1, 0.5 * dt));
        let k3 = self.extended_response(t + 0.5 * dt, &advance(&k2, 0.5 * dt));
        let k4 = self.extended_response(t + dt, &advance(&k3, dt));
        std::array::from_fn(|i| state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    /// Validate the arguments shared by the Lyapunov routines.
    fn check_lyapunov_arguments(
        x0: f64,
        v0: f64,
        t0: f64,
        dt: f64,
        n_steps: usize,
        renorm_every: usize,
    ) -> Result<()> {
        if !(x0.is_finite() && v0.is_finite() && t0.is_finite()) {
            return Err(Error::InvalidInput(
                "initial state and time must be finite".into(),
            ));
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::InvalidInput(format!(
                "time step must be positive and finite, got {dt}"
            )));
        }
        if n_steps == 0 || renorm_every == 0 {
            return Err(Error::InvalidInput(
                "n_steps and renorm_every must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// Largest Lyapunov exponent `λ₁` of the (forced) Duffing flow by Benettin's method.
    ///
    /// The state `(x, v)` and one tangent vector `(δx, δv)` are integrated together with
    /// RK4 from `(x0, v0)` at time `t0` for `n_steps` steps of size `dt`; the tangent
    /// obeys the variational equations `δẋ = δv`, `δv̇ = (α − 3β x²)·δx − δ·δv`.  Every
    /// `renorm_every` steps (and at the end) the tangent vector is rescaled to unit
    /// length and `ln‖δ‖` is accumulated; the exponent is the accumulated logarithm
    /// divided by the total time `n_steps · dt`.  The result converges as
    /// `O(1/T)` (plus the statistical error of a chaotic orbit) and does not depend on
    /// the sampling step `dt`.
    ///
    /// Check values: the damped linear oscillator `ẍ + δẋ + x = 0` (`α = −1`, `β = γ = 0`)
    /// has `λ₁ = −δ/2`, and the chaotic double-well oscillator
    /// (`α = 1`, `β = 1`, `δ = 0.3`, `γ = 0.5`, `ω = 1.2`) has `λ₁ ≈ 0.1`.  Because the
    /// divergence of the flow is `−δ`, the two exponents of the planar flow satisfy
    /// `λ₁ + λ₂ = −δ` (see [`lyapunov_spectrum`](Self::lyapunov_spectrum)).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a non-finite initial state or time, `dt <= 0`, or
    /// `n_steps == 0` / `renorm_every == 0`; [`Error::General`] if the integration
    /// diverges (non-finite state).
    pub fn max_lyapunov(
        &self,
        x0: f64,
        v0: f64,
        t0: f64,
        dt: f64,
        n_steps: usize,
        renorm_every: usize,
    ) -> Result<f64> {
        Self::check_lyapunov_arguments(x0, v0, t0, dt, n_steps, renorm_every)?;
        let start = std::f64::consts::FRAC_1_SQRT_2;
        let mut state = [x0, v0, start, start];
        let mut log_sum = 0.0_f64;
        let mut since_renorm = 0usize;
        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            state = self.rk4_extended(t, &state, dt);
            since_renorm += 1;
            if since_renorm == renorm_every || step + 1 == n_steps {
                let norm = state[2].hypot(state[3]);
                if !(norm.is_finite() && norm > 0.0 && state[0].is_finite()) {
                    return Err(Error::General(format!(
                        "Lyapunov integration diverged at step {}",
                        step + 1
                    )));
                }
                log_sum += norm.ln();
                state[2] /= norm;
                state[3] /= norm;
                since_renorm = 0;
            }
        }
        Ok(log_sum / (n_steps as f64 * dt))
    }

    /// Both Lyapunov exponents `[λ₁, λ₂]` (`λ₁ ≥ λ₂`) of the planar Duffing flow.
    ///
    /// Two tangent vectors are integrated with the base trajectory and re-orthonormalised
    /// by Gram–Schmidt every `renorm_every` steps; the exponents are the time averages of
    /// the logarithms of the diagonal of the `R` factor.  `λ₁` equals
    /// [`max_lyapunov`](Self::max_lyapunov); since the divergence of the flow is the
    /// constant `−δ`, `λ₁ + λ₂ = −δ` holds up to the integration error, which provides
    /// an independent check of the result.
    ///
    /// # Errors
    ///
    /// As for [`max_lyapunov`](Self::max_lyapunov); additionally [`Error::General`] if the
    /// two tangent vectors become linearly dependent.
    pub fn lyapunov_spectrum(
        &self,
        x0: f64,
        v0: f64,
        t0: f64,
        dt: f64,
        n_steps: usize,
        renorm_every: usize,
    ) -> Result<[f64; 2]> {
        Self::check_lyapunov_arguments(x0, v0, t0, dt, n_steps, renorm_every)?;
        let mut state = [x0, v0, 1.0, 0.0, 0.0, 1.0];
        let mut log_sums = [0.0_f64; 2];
        let mut since_renorm = 0usize;
        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            state = self.rk4_extended(t, &state, dt);
            since_renorm += 1;
            if since_renorm == renorm_every || step + 1 == n_steps {
                // Gram–Schmidt: R₁₁ = ‖δ₁‖, then R₂₂ = ‖δ₂ − (δ₂·q₁) q₁‖.
                let r11 = state[2].hypot(state[3]);
                if !(r11.is_finite() && r11 > 0.0 && state[0].is_finite()) {
                    return Err(Error::General(format!(
                        "Lyapunov integration diverged at step {}",
                        step + 1
                    )));
                }
                let q1 = [state[2] / r11, state[3] / r11];
                let projection = state[4] * q1[0] + state[5] * q1[1];
                let orthogonal = [state[4] - projection * q1[0], state[5] - projection * q1[1]];
                let r22 = orthogonal[0].hypot(orthogonal[1]);
                if !(r22.is_finite() && r22 > 0.0) {
                    return Err(Error::General(format!(
                        "tangent vectors became linearly dependent at step {}",
                        step + 1
                    )));
                }
                log_sums[0] += r11.ln();
                log_sums[1] += r22.ln();
                state[2] = q1[0];
                state[3] = q1[1];
                state[4] = orthogonal[0] / r22;
                state[5] = orthogonal[1] / r22;
                since_renorm = 0;
            }
        }
        let total_time = n_steps as f64 * dt;
        Ok([log_sums[0] / total_time, log_sums[1] / total_time])
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  LYAPUNOV EXPONENT
// ══════════════════════════════════════════════════════════════════════════════

/// Legacy holder of the trajectory-only "MLE" routine, kept for API compatibility.
///
/// A Lyapunov exponent measures how fast *nearby* trajectories separate, so it cannot be
/// obtained from a single sampled orbit by differencing it.  Use
/// [`DuffingOscillator::max_lyapunov`] (Benettin's method with the variational equations)
/// or [`DuffingOscillator::lyapunov_spectrum`] instead.
#[derive(Debug, Clone)]
pub struct LyapunovExponent {
    /// Tolerance below which a finite-difference chord is ignored by
    /// [`compute_mle`](Self::compute_mle).
    pub tol: f64,
}

impl LyapunovExponent {
    /// Create with the given chord-length tolerance.
    pub fn new(tol: f64) -> Self {
        Self { tol }
    }

    /// Mean logarithmic chord length of a sampled 2-D phase trajectory per unit time.
    ///
    /// The trajectory is `trajectory\[i\] = \[x_i, v_i\]`, sampled at uniform time steps
    /// of size `dt`.  The result is `Σ ln‖p_{i+1} − p_{i−1}‖ / ((n − 2)·dt)` over the chords
    /// longer than `tol`.
    ///
    /// **This is not a Lyapunov exponent.**  It depends on the sampling step (about
    /// `ln(2·dt·|ṗ|)/dt` for a smooth orbit, e.g. ≈ −390 for a periodic orbit at
    /// `dt = 0.01`) and measures no growth of perturbations.
    #[deprecated(
        since = "0.1.3",
        note = "not a Lyapunov exponent (it depends on the sampling step); use \
                `DuffingOscillator::max_lyapunov` or `DuffingOscillator::lyapunov_spectrum`"
    )]
    pub fn compute_mle(&self, trajectory: &[[f64; 2]], dt: f64) -> f64 {
        if trajectory.len() < 3 {
            return 0.0;
        }
        let n = trajectory.len();
        let mut sum_log = 0.0;
        let mut count = 0usize;

        // Accumulate the logarithm of the central-difference chord length.
        for i in 1..n - 1 {
            let dx = trajectory[i + 1][0] - trajectory[i - 1][0];
            let dv = trajectory[i + 1][1] - trajectory[i - 1][1];
            let norm = (dx * dx + dv * dv).sqrt();
            if norm > self.tol {
                sum_log += norm.ln();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }
        // Normalise by the total time spanned by the chords.
        let total_time = (n - 2) as f64 * dt;
        sum_log / total_time
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  POINCARÉ SECTION
// ══════════════════════════════════════════════════════════════════════════════

/// Computes the Poincaré section of a continuous-time trajectory.
///
/// Records phase-space points whenever the time is an (approximate) multiple
/// of the forcing period `period`.
#[derive(Debug, Clone)]
pub struct PoincareSection {
    /// Forcing period T = 2π/ωd.
    pub period: f64,
    /// Tolerance for matching a period crossing.
    pub tolerance: f64,
}

impl PoincareSection {
    /// Create with the given forcing period and crossing tolerance.
    pub fn new(period: f64, tolerance: f64) -> Self {
        Self { period, tolerance }
    }

    /// Extract Poincaré section points from a trajectory.
    ///
    /// `trajectory` is a slice of `\[x, v\]` pairs, and `t_values` are the
    /// corresponding times. Returns the `\[x, v\]` points where
    /// `t mod period < tolerance`.
    pub fn compute(&self, trajectory: &[[f64; 2]], t_values: &[f64]) -> Vec<[f64; 2]> {
        let n = trajectory.len().min(t_values.len());
        let mut section = Vec::new();
        for i in 0..n {
            let phase = t_values[i] % self.period;
            if phase < self.tolerance || (self.period - phase) < self.tolerance {
                section.push(trajectory[i]);
            }
        }
        section
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  BIFURCATION DIAGRAM
// ══════════════════════════════════════════════════════════════════════════════

/// Bifurcation diagram computed by sampling the long-time attractor of a
/// parameterised system.
#[derive(Debug, Clone)]
pub struct BifurcationDiagram {
    /// Parameter values swept.
    pub parameter_range: Vec<f64>,
    /// Attractor samples for each parameter value.
    pub attractors: Vec<Vec<f64>>,
}

impl BifurcationDiagram {
    /// Create an empty diagram.
    pub fn new() -> Self {
        Self {
            parameter_range: Vec::new(),
            attractors: Vec::new(),
        }
    }

    /// Compute the bifurcation diagram for the **logistic map**
    /// xₙ₊₁ = r·xₙ·(1−xₙ) over the supplied `r_range`.
    ///
    /// For each r value:
    /// 1. Transient of `n_transient` steps from x₀ = 0.5 is discarded.
    /// 2. `n_attractor` subsequent values are recorded.
    pub fn compute_logistic(r_range: &[f64], n_transient: usize, n_attractor: usize) -> Self {
        let mut param_range = Vec::with_capacity(r_range.len());
        let mut attractors = Vec::with_capacity(r_range.len());
        for &r in r_range {
            let mut x = 0.5_f64;
            for _ in 0..n_transient {
                x = logistic_map(r, x);
            }
            let mut att = Vec::with_capacity(n_attractor);
            for _ in 0..n_attractor {
                x = logistic_map(r, x);
                att.push(x);
            }
            param_range.push(r);
            attractors.push(att);
        }
        Self {
            parameter_range: param_range,
            attractors,
        }
    }

    /// Compute the bifurcation diagram for the Duffing oscillator by varying
    /// `gamma` (forcing amplitude) while holding α, β fixed.
    ///
    /// For each γ value the system is integrated for `n_transient` periods
    /// (transient discarded), then `n_attractor` Poincaré crossings are
    /// recorded.
    pub fn compute_duffing(alpha: f64, beta: f64, gamma_range: &[f64]) -> Vec<Vec<f64>> {
        let delta = 0.3;
        let omega_d = 1.0;
        let period = 2.0 * PI / omega_d;
        let dt = period / 200.0;
        let n_transient = 500;
        let n_attractor = 50;

        gamma_range
            .iter()
            .map(|&gam| {
                let osc = DuffingOscillator::new(alpha, beta, delta, gam, omega_d);
                let total = n_transient + n_attractor * 200;
                let traj = osc.phase_portrait(0.1, 0.0, 0.0, total, dt);
                // Sample once per period after transient
                let ps = PoincareSection::new(period, dt * 0.5);
                let t_vals: Vec<f64> = (0..=total).map(|i| i as f64 * dt).collect();
                let section = ps.compute(&traj, &t_vals);
                let skip = section.len().saturating_sub(n_attractor);
                section[skip..].iter().map(|p| p[0]).collect()
            })
            .collect()
    }
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  NONLINEAR FEM WITH NEWTON–RAPHSON
// ══════════════════════════════════════════════════════════════════════════════

/// Nonlinear FEM assembly and Newton–Raphson solver for a 2-D triangular mesh.
///
/// Nodes are 2-D points; each element is a constant-strain triangle (CST).
/// The tangent stiffness is assembled using a simple Neo-Hookean material
/// linearisation.
#[derive(Debug, Clone)]
pub struct NonlinearFem {
    /// Node coordinates `\[x, y\]`.
    pub nodes: Vec<[f64; 2]>,
    /// Element connectivity (three node indices).
    pub connectivity: Vec<[usize; 3]>,
    /// Young's modulus.
    pub young: f64,
    /// Poisson's ratio.
    pub nu: f64,
}

impl NonlinearFem {
    /// Create a `NonlinearFem` problem.
    pub fn new(nodes: Vec<[f64; 2]>, connectivity: Vec<[usize; 3]>, young: f64, nu: f64) -> Self {
        Self {
            nodes,
            connectivity,
            young,
            nu,
        }
    }

    /// Assemble the (linear) tangent stiffness matrix into a dense `ndof×ndof`
    /// row-major `Vec`f64` where `ndof = 2 * nodes.len()`.
    pub fn k_tangent(&self) -> Vec<f64> {
        let ndof = 2 * self.nodes.len();
        let mut k = vec![0.0_f64; ndof * ndof];

        let e = self.young;
        let nu = self.nu;
        let fac = e / (1.0 - nu * nu);
        // Plane-stress constitutive matrix D (3×3 Voigt)
        let d = [
            [fac, fac * nu, 0.0],
            [fac * nu, fac, 0.0],
            [0.0, 0.0, fac * (1.0 - nu) / 2.0],
        ];

        for elem in &self.connectivity {
            let [i0, i1, i2] = *elem;
            let [x0, y0] = self.nodes[i0];
            let [x1, y1] = self.nodes[i1];
            let [x2, y2] = self.nodes[i2];

            // Area of the triangle
            let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
            if area < 1e-30 {
                continue;
            }

            // Strain-displacement matrix B (3×6 for CST in plane stress)
            let b1x = (y1 - y2) / (2.0 * area);
            let b2x = (y2 - y0) / (2.0 * area);
            let b3x = (y0 - y1) / (2.0 * area);
            let b1y = (x2 - x1) / (2.0 * area);
            let b2y = (x0 - x2) / (2.0 * area);
            let b3y = (x1 - x0) / (2.0 * area);

            // B is 3 rows × 6 cols
            let b = [
                [b1x, 0.0, b2x, 0.0, b3x, 0.0],
                [0.0, b1y, 0.0, b2y, 0.0, b3y],
                [b1y, b1x, b2y, b2x, b3y, b3x],
            ];

            // K_e = B^T D B * area (6×6)
            let global_dofs = [2 * i0, 2 * i0 + 1, 2 * i1, 2 * i1 + 1, 2 * i2, 2 * i2 + 1];

            // DB = D * B  (3×6)
            let mut db = [[0.0_f64; 6]; 3];
            for ri in 0..3 {
                for ci in 0..6 {
                    for m in 0..3 {
                        db[ri][ci] += d[ri][m] * b[m][ci];
                    }
                }
            }
            // K_e[a,b] = Σ_r B[r,a] * DB[r,b] * area
            for a in 0..6 {
                for bi in 0..6 {
                    let mut val = 0.0;
                    for r in 0..3 {
                        val += b[r][a] * db[r][bi];
                    }
                    val *= area;
                    let ga = global_dofs[a];
                    let gb = global_dofs[bi];
                    k[ga * ndof + gb] += val;
                }
            }
        }
        k
    }

    /// Solve the linear system K u = f using Gaussian elimination with partial
    /// pivoting. Returns the displacement vector (length `2 * nodes.len()`).
    ///
    /// This is used as the Newton–Raphson linear step; for a linear problem
    /// one iteration converges exactly.
    ///
    /// The first two DOFs are fixed (Dirichlet BC = 0) to prevent rigid-body
    /// motion.
    pub fn solve_newton(&self, force: &[f64], max_iter: usize) -> Vec<f64> {
        let ndof = 2 * self.nodes.len();
        assert_eq!(force.len(), ndof, "force vector length mismatch");

        let mut u = vec![0.0_f64; ndof];

        for _iter in 0..max_iter {
            let k = self.k_tangent();
            // Residual: r = f - K u
            let mut r = vec![0.0_f64; ndof];
            for i in 0..ndof {
                let mut ku_i = 0.0;
                for j in 0..ndof {
                    ku_i += k[i * ndof + j] * u[j];
                }
                r[i] = force[i] - ku_i;
            }
            // Apply homogeneous Dirichlet on first two DOFs
            r[0] = 0.0;
            r[1] = 0.0;

            let norm_r = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_r < 1e-12 {
                break;
            }

            // Solve K Δu = r with the same BCs applied to K
            let delta_u = solve_linear_system(&k, &r, ndof);
            for i in 0..ndof {
                u[i] += delta_u[i];
            }
            // Fix pinned DOFs
            u[0] = 0.0;
            u[1] = 0.0;
        }
        u
    }
}

/// Dense Gaussian elimination with partial pivoting: solves A x = b.
///
/// `n` is the system dimension. Returns the solution vector.
fn solve_linear_system(a_in: &[f64], b_in: &[f64], n: usize) -> Vec<f64> {
    let mut a: Vec<f64> = a_in.to_vec();
    let mut b: Vec<f64> = b_in.to_vec();

    // Forward elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            continue; // singular sub-block (pinned DOF)
        }
        // Swap rows
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        // Eliminate below
        let pivot = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let tmp = a[col * n + k];
                a[row * n + k] -= factor * tmp;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() > 1e-30 { s / diag } else { 0.0 };
    }
    x
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  TESTS
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Feigenbaum constant ───────────────────────────────────────────────────

    #[test]
    fn test_feigenbaum_value() {
        let delta = period_doubling_ratio();
        assert!((delta - 4.669_201_609).abs() < 1e-6);
    }

    #[test]
    fn test_feigenbaum_positive() {
        assert!(period_doubling_ratio() > 4.6);
    }

    // ── Logistic map ─────────────────────────────────────────────────────────

    #[test]
    fn test_logistic_fixed_point_r1() {
        // For r=1, x=0 is a fixed point
        let x = logistic_map(1.0, 0.0);
        assert_eq!(x, 0.0);
    }

    #[test]
    fn test_logistic_fixed_point_r3() {
        // Non-trivial fixed point: x* = (r-1)/r = 2/3 for r=3
        let r = 3.0;
        let x_star = (r - 1.0) / r;
        let x_next = logistic_map(r, x_star);
        assert!((x_next - x_star).abs() < 1e-14);
    }

    #[test]
    fn test_logistic_stays_in_unit_interval() {
        let r = 3.8;
        let mut x = 0.4;
        for _ in 0..1000 {
            x = logistic_map(r, x);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn test_logistic_period2_r33() {
        // r=3.3 exhibits period-2 orbit
        let r = 3.3;
        let mut x = 0.5;
        for _ in 0..2000 {
            x = logistic_map(r, x);
        }
        let x_a = x;
        let x_b = logistic_map(r, x_a);
        let x_c = logistic_map(r, x_b);
        // After transient x_a ≈ x_c (period 2)
        assert!((x_a - x_c).abs() < 1e-10);
    }

    #[test]
    fn test_logistic_zero_boundary() {
        assert_eq!(logistic_map(4.0, 0.0), 0.0);
        assert_eq!(logistic_map(4.0, 1.0), 0.0);
    }

    // ── Hénon map ────────────────────────────────────────────────────────────

    #[test]
    fn test_henon_classical_attractor() {
        let a = 1.4;
        let b = 0.3;
        let mut x = 0.1_f64;
        let mut y = 0.1_f64;
        for _ in 0..10_000 {
            let [nx, ny] = henon_map(x, y, a, b);
            x = nx;
            y = ny;
        }
        // Classical Hénon attractor resides inside x ∈ [-1.5, 1.5]
        assert!(x.abs() < 1.5);
        assert!(y.abs() < 0.5);
    }

    #[test]
    fn test_henon_b_zero_collapses_y() {
        // b=0 means y_new=0 always after one step
        let [_, y_new] = henon_map(0.5, 0.5, 1.4, 0.0);
        assert_eq!(y_new, 0.0);
    }

    #[test]
    fn test_henon_origin_shift() {
        // x=0,y=1,a=0,b=0 → [2.0, 0.0]
        let [xn, yn] = henon_map(0.0, 1.0, 0.0, 0.0);
        assert!((xn - 2.0).abs() < 1e-14);
        assert_eq!(yn, 0.0);
    }

    // ── Duffing oscillator ────────────────────────────────────────────────────

    fn make_duffing() -> DuffingOscillator {
        DuffingOscillator::new(-1.0, 1.0, 0.3, 0.4, 1.2)
    }

    #[test]
    fn test_duffing_response_length() {
        let osc = make_duffing();
        let r = osc.response(0.0, 0.5, 0.0);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_duffing_undamped_no_force_returns_dx_as_v() {
        let osc = DuffingOscillator::new(1.0, 0.0, 0.0, 0.0, 1.0);
        let v = 0.7;
        let [dx, _] = osc.response(0.0, 1.0, v);
        assert!((dx - v).abs() < 1e-14);
    }

    #[test]
    fn test_duffing_phase_portrait_count() {
        let osc = make_duffing();
        let traj = osc.phase_portrait(0.1, 0.0, 0.0, 500, 0.01);
        assert_eq!(traj.len(), 501);
    }

    #[test]
    fn test_duffing_rk4_reversibility_approx() {
        // Short forward-backward integration should approximately return to start
        let osc = DuffingOscillator::new(1.0, 0.0, 0.0, 0.0, 1.0);
        let traj_fwd = osc.phase_portrait(1.0, 0.0, 0.0, 100, 0.01);
        let last = traj_fwd.last().unwrap();
        // Just check it moved
        assert!(last[0].abs() < 2.0); // bounded
    }

    #[test]
    fn test_duffing_initial_state_preserved() {
        let osc = make_duffing();
        let traj = osc.phase_portrait(0.5, 0.3, 0.0, 10, 0.01);
        assert!((traj[0][0] - 0.5).abs() < 1e-14);
        assert!((traj[0][1] - 0.3).abs() < 1e-14);
    }

    #[test]
    fn test_duffing_zero_gamma_bounded() {
        // Without forcing and with damping the oscillator should decay
        let osc = DuffingOscillator::new(-1.0, 1.0, 1.0, 0.0, 1.0);
        let traj = osc.phase_portrait(0.5, 0.0, 0.0, 2000, 0.01);
        let last = traj.last().unwrap();
        // Amplitude should decay toward an equilibrium
        assert!(last[0].abs() < 1.5);
    }

    // ── Lyapunov exponent ─────────────────────────────────────────────────────

    /// Chaotic double-well oscillator of Moon & Holmes (`δ = 0.3`, `γ = 0.5`, `ω = 1.2`).
    fn make_chaotic_duffing() -> DuffingOscillator {
        DuffingOscillator::new(1.0, 1.0, 0.3, 0.5, 1.2)
    }

    #[test]
    fn test_max_lyapunov_of_damped_linear_oscillator_is_minus_half_the_damping() {
        // ẍ + δẋ + x = 0 has the characteristic roots −δ/2 ± i√(1 − δ²/4), so λ₁ = −δ/2.
        for &delta in &[0.2, 0.5] {
            let osc = DuffingOscillator::new(-1.0, 0.0, delta, 0.0, 1.0);
            let coarse = osc.max_lyapunov(1.0, 0.0, 0.0, 0.01, 100_000, 10).unwrap();
            let fine = osc.max_lyapunov(1.0, 0.0, 0.0, 0.005, 200_000, 10).unwrap();
            assert!(
                (coarse + 0.5 * delta).abs() < 2e-3,
                "δ = {delta}: λ = {coarse}"
            );
            assert!((fine + 0.5 * delta).abs() < 2e-3, "δ = {delta}: λ = {fine}");
            // The old trajectory-speed statistic depended on dt (≈ −390 at dt = 0.01);
            // the exponent must not.
            assert!((coarse - fine).abs() < 1e-3, "{coarse} vs {fine}");
        }
    }

    #[test]
    fn test_max_lyapunov_of_chaotic_double_well_oscillator_is_positive() {
        let osc = make_chaotic_duffing();
        // Discard a transient so that the orbit sits on the attractor.
        let transient = osc.phase_portrait(0.1, 0.0, 0.0, 10_000, 0.01);
        let [x, v] = *transient.last().unwrap();
        let lambda = osc.max_lyapunov(x, v, 100.0, 0.01, 200_000, 10).unwrap();
        assert!(lambda > 0.05 && lambda < 0.2, "λ₁ = {lambda}");
        // A different initial condition gives the same exponent within the statistical error.
        let other = osc.max_lyapunov(-0.5, 0.3, 0.0, 0.01, 200_000, 10).unwrap();
        assert!((lambda - other).abs() < 0.03, "{lambda} vs {other}");
    }

    #[test]
    fn test_lyapunov_spectrum_satisfies_the_volume_contraction_sum_rule() {
        // The divergence of the planar flow is −δ, so λ₁ + λ₂ = −δ for every orbit.
        for osc in [make_chaotic_duffing(), make_duffing()] {
            let [l1, l2] = osc
                .lyapunov_spectrum(0.1, 0.0, 0.0, 0.01, 100_000, 10)
                .unwrap();
            assert!((l1 + l2 + osc.delta).abs() < 1e-3, "λ₁ + λ₂ = {}", l1 + l2);
            let single = osc.max_lyapunov(0.1, 0.0, 0.0, 0.01, 100_000, 10).unwrap();
            assert!((l1 - single).abs() < 5e-3, "{l1} vs {single}");
        }
        // The chaotic orbit has a positive and a strongly negative exponent.
        let [l1, l2] = make_chaotic_duffing()
            .lyapunov_spectrum(0.1, 0.0, 0.0, 0.01, 200_000, 10)
            .unwrap();
        assert!(l1 > 0.05 && l2 < -0.3, "spectrum = [{l1}, {l2}]");
    }

    #[test]
    fn test_max_lyapunov_of_regular_forced_oscillator_is_negative() {
        // Single-well hardening oscillator with moderate forcing settles on a stable limit
        // cycle: both Floquet exponents are negative (λ₁ = λ₂ = −δ/2).
        let osc = make_duffing();
        let lambda = osc.max_lyapunov(0.1, 0.0, 0.0, 0.01, 300_000, 10).unwrap();
        assert!(lambda < -0.1 && lambda > -0.2, "λ₁ = {lambda}");
    }

    #[test]
    fn test_max_lyapunov_is_independent_of_the_renormalisation_schedule() {
        // The tangent flow is linear, so rescaling only moves factors between ln‖δ‖ and the
        // accumulated sum: the estimate must agree to rounding for any interval.
        let osc = make_chaotic_duffing();
        let every_step = osc.max_lyapunov(0.1, 0.0, 0.0, 0.01, 20_000, 1).unwrap();
        let sparse = osc.max_lyapunov(0.1, 0.0, 0.0, 0.01, 20_000, 97).unwrap();
        let once = osc
            .max_lyapunov(0.1, 0.0, 0.0, 0.01, 20_000, 1_000_000)
            .unwrap();
        assert!(
            (every_step - sparse).abs() < 1e-9,
            "{every_step} vs {sparse}"
        );
        assert!((every_step - once).abs() < 1e-9, "{every_step} vs {once}");
    }

    #[test]
    fn test_max_lyapunov_matches_the_separation_of_nearby_trajectories() {
        // For a short window the tangent vector reproduces the growth of a small, finite
        // perturbation of the real nonlinear flow along the same initial direction.
        let osc = make_chaotic_duffing();
        let (dt, n_steps, eps) = (0.01, 2_000usize, 1e-8);
        let lambda = osc.max_lyapunov(0.3, -0.2, 0.0, dt, n_steps, 25).unwrap();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let reference = osc.phase_portrait(0.3, -0.2, 0.0, n_steps, dt);
        let perturbed = osc.phase_portrait(0.3 + eps * s, -0.2 + eps * s, 0.0, n_steps, dt);
        let (end, end_p) = (reference[n_steps], perturbed[n_steps]);
        let separation = (end_p[0] - end[0]).hypot(end_p[1] - end[1]);
        let finite_time = (separation / eps).ln() / (n_steps as f64 * dt);
        assert!(
            (lambda - finite_time).abs() < 1e-5,
            "{lambda} vs {finite_time}"
        );
        // The window is long enough for the perturbation to have changed appreciably.
        assert!((separation / eps - 1.0).abs() > 0.1);
    }

    #[test]
    fn test_lyapunov_routines_validate_their_arguments() {
        let osc = make_duffing();
        for (dt, n, k) in [
            (0.0, 10, 1),
            (-0.01, 10, 1),
            (f64::NAN, 10, 1),
            (0.01, 0, 1),
            (0.01, 10, 0),
        ] {
            assert!(
                matches!(
                    osc.max_lyapunov(0.1, 0.0, 0.0, dt, n, k),
                    Err(Error::InvalidInput(_))
                ),
                "max_lyapunov accepted dt = {dt}, n = {n}, k = {k}"
            );
            assert!(
                matches!(
                    osc.lyapunov_spectrum(0.1, 0.0, 0.0, dt, n, k),
                    Err(Error::InvalidInput(_))
                ),
                "lyapunov_spectrum accepted dt = {dt}, n = {n}, k = {k}"
            );
        }
        assert!(matches!(
            osc.max_lyapunov(f64::NAN, 0.0, 0.0, 0.01, 10, 1),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            osc.max_lyapunov(0.1, f64::INFINITY, 0.0, 0.01, 10, 1),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_lyapunov_integration_reports_divergence() {
        // A step far beyond the stability limit of RK4 blows the state up to infinity.
        let osc = DuffingOscillator::new(1.0, 1.0, 0.0, 0.0, 1.0);
        assert!(matches!(
            osc.max_lyapunov(5.0, 5.0, 0.0, 10.0, 100, 1),
            Err(Error::General(_))
        ));
    }

    // ── Poincaré section ──────────────────────────────────────────────────────

    #[test]
    fn test_poincare_selects_crossings() {
        let period = 2.0 * PI;
        let ps = PoincareSection::new(period, 0.05);
        let traj: Vec<[f64; 2]> = (0..1000).map(|i| [i as f64 * 0.1, 0.0]).collect();
        let t_vals: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let section = ps.compute(&traj, &t_vals);
        // Should select some points
        assert!(!section.is_empty());
    }

    #[test]
    fn test_poincare_empty_when_no_match() {
        let ps = PoincareSection::new(1.0, 1e-15);
        let traj = vec![[0.5_f64, 0.0], [0.7, 0.1]];
        let t_vals = vec![0.3, 0.8];
        let section = ps.compute(&traj, &t_vals);
        assert!(section.is_empty());
    }

    #[test]
    fn test_poincare_length_bound() {
        let period = 2.0 * PI;
        let ps = PoincareSection::new(period, 0.1);
        let osc = make_duffing();
        let dt = 0.01;
        let n = 5000usize;
        let traj = osc.phase_portrait(0.1, 0.0, 0.0, n, dt);
        let t_vals: Vec<f64> = (0..=n).map(|i| i as f64 * dt).collect();
        let section = ps.compute(&traj, &t_vals);
        assert!(section.len() <= traj.len());
    }

    // ── Bifurcation diagram ───────────────────────────────────────────────────

    #[test]
    fn test_bifurcation_logistic_length() {
        let r_range: Vec<f64> = (0..50).map(|i| 2.5 + i as f64 * 0.03).collect();
        let diag = BifurcationDiagram::compute_logistic(&r_range, 500, 30);
        assert_eq!(diag.parameter_range.len(), 50);
        assert_eq!(diag.attractors.len(), 50);
    }

    #[test]
    fn test_bifurcation_logistic_in_unit_interval() {
        let r_range = vec![3.5, 3.7, 3.9];
        let diag = BifurcationDiagram::compute_logistic(&r_range, 300, 20);
        for att in &diag.attractors {
            for &x in att {
                assert!(
                    (0.0..=1.0).contains(&x),
                    "logistic map left unit interval: {x}"
                );
            }
        }
    }

    #[test]
    fn test_bifurcation_duffing_length() {
        let gamma_range: Vec<f64> = (0..5).map(|i| 0.1 + i as f64 * 0.1).collect();
        let diags = BifurcationDiagram::compute_duffing(-1.0, 1.0, &gamma_range);
        assert_eq!(diags.len(), 5);
    }

    #[test]
    fn test_bifurcation_default() {
        let d = BifurcationDiagram::default();
        assert!(d.parameter_range.is_empty());
    }

    // ── NonlinearFem ──────────────────────────────────────────────────────────

    fn single_triangle() -> NonlinearFem {
        let nodes = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let connectivity = vec![[0, 1, 2]];
        NonlinearFem::new(nodes, connectivity, 1e6, 0.3)
    }

    #[test]
    fn test_fem_k_tangent_size() {
        let fem = single_triangle();
        let k = fem.k_tangent();
        let ndof = 2 * fem.nodes.len();
        assert_eq!(k.len(), ndof * ndof);
    }

    #[test]
    fn test_fem_k_tangent_symmetry() {
        let fem = single_triangle();
        let k = fem.k_tangent();
        let ndof = 2 * fem.nodes.len();
        for i in 0..ndof {
            for j in 0..ndof {
                let diff = (k[i * ndof + j] - k[j * ndof + i]).abs();
                assert!(diff < 1e-6, "K not symmetric at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_fem_solve_zero_force() {
        let fem = single_triangle();
        let ndof = 2 * fem.nodes.len();
        let force = vec![0.0; ndof];
        let u = fem.solve_newton(&force, 5);
        for ui in u {
            assert!(ui.abs() < 1e-10);
        }
    }

    #[test]
    fn test_fem_solve_nonzero_force_length() {
        let fem = single_triangle();
        let ndof = 2 * fem.nodes.len();
        let mut force = vec![0.0; ndof];
        force[ndof - 1] = 1000.0; // apply load at last DOF
        let u = fem.solve_newton(&force, 5);
        assert_eq!(u.len(), ndof);
    }

    #[test]
    fn test_fem_k_positive_diagonal() {
        let fem = single_triangle();
        let k = fem.k_tangent();
        let ndof = 2 * fem.nodes.len();
        // Interior DOFs (not pinned) should have positive diagonal
        for i in 2..ndof {
            assert!(k[i * ndof + i] >= 0.0, "negative diagonal at DOF {i}");
        }
    }

    #[test]
    fn test_fem_two_elements() {
        let nodes = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let connectivity = vec![[0, 1, 3], [0, 3, 2]];
        let fem = NonlinearFem::new(nodes, connectivity, 2e5, 0.25);
        let k = fem.k_tangent();
        let ndof = 8;
        assert_eq!(k.len(), ndof * ndof);
    }

    // ── Linear solver ─────────────────────────────────────────────────────────

    #[test]
    fn test_linear_solver_identity() {
        let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solve_linear_system(&a, &b, 2);
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_solver_2x2() {
        // 2x + y = 5, x + 3y = 10 → x=1, y=3
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = solve_linear_system(&a, &b, 2);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }
}
