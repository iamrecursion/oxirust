//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{
    compute_rayleigh_coefficients, modal_truncation_correction, rayleigh_damping_matrix,
};
use crate::sparse::CsrMatrix;

/// State for the Bathe two-sub-step implicit time integrator.
///
/// The Bathe method splits each time step Δt into two sub-steps of Δt/2:
///
/// 1. the **trapezoidal rule** over `[t_n, t_n + Δt/2]` gives the intermediate state
///    `(u_m, v_m)`;
/// 2. the **three-point Euler backward** scheme over `[t_n + Δt/2, t_{n+1}]` uses the
///    states at `t_n`, `t_n + Δt/2` and `t_{n+1}`:
///
///    ```text
///    v_{n+1} = (u_n − 4 u_m + 3 u_{n+1}) / Δt
///    a_{n+1} = (v_n − 4 v_m + 3 v_{n+1}) / Δt
///    ```
///
///    Together with the equation of motion at `t_{n+1}` this yields
///    `(9 M/Δt² + 3 C/Δt + K) u_{n+1} = f_{n+1} − M [(v_n − 4 v_m)/Δt + 3 (u_n − 4 u_m)/Δt²]
///    − C (u_n − 4 u_m)/Δt`.
///
/// The composite scheme is second-order accurate, unconditionally stable for linear
/// systems and strongly dissipates spurious high-frequency content (its spectral radius
/// tends to zero as ω Δt → ∞) while preserving the low-frequency response.
///
/// References: Bathe & Baig (2005) "On a composite implicit time integration
/// procedure for nonlinear dynamics", *Computers & Structures* 83, 2513–2524;
/// Bathe & Noh (2012) "Insight into an implicit time integration scheme for
/// structural dynamics", *Computers & Structures* 98–99, 1–6.
pub struct BatheIntegrator {
    /// Number of DOF.
    pub n_dof: usize,
    /// Displacement at end of last full step.
    pub u: Vec<f64>,
    /// Velocity at end of last full step.
    pub v: Vec<f64>,
    /// Acceleration at end of last full step.
    pub a: Vec<f64>,
}
impl BatheIntegrator {
    /// Create a new Bathe integrator with zero initial conditions.
    pub fn new(n_dof: usize) -> Self {
        Self {
            n_dof,
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
        }
    }
    /// Advance one full step Δt with the load held constant over the step.
    ///
    /// Equivalent to [`step_diagonal_with_loads`](Self::step_diagonal_with_loads) with
    /// `f_mid = f_end = f`.  For a load that varies within the step pass the load at
    /// `t + Δt/2` explicitly through `step_diagonal_with_loads`.
    ///
    /// # Arguments
    /// * `m_diag`  – diagonal mass vector (length n_dof)
    /// * `k_diag`  – diagonal stiffness vector (length n_dof)
    /// * `c_diag`  – diagonal damping vector (length n_dof)
    /// * `f`       – force vector, constant over the step (length n_dof)
    /// * `dt`      – full step size Δt
    ///
    /// # Panics
    ///
    /// Panics if one of the slices is shorter than `n_dof`.
    pub fn step_diagonal(
        &mut self,
        m_diag: &[f64],
        k_diag: &[f64],
        c_diag: &[f64],
        f: &[f64],
        dt: f64,
    ) {
        self.step_diagonal_with_loads(m_diag, k_diag, c_diag, f, f, dt);
    }
    /// Advance one full step Δt using the two-sub-step Bathe scheme.
    ///
    /// Solves `M a + C v + K u = f` for diagonal (lumped) `M`, `C` and `K`, for which
    /// the system decouples into `n_dof` independent scalar problems.  `self.a` must be
    /// the acceleration consistent with `self.u`, `self.v` and the load at the start of
    /// the step, i.e. `a = (f_n − C v − K u) / M`.  A DOF without inertia, damping and
    /// stiffness has no equation to solve and keeps its previous state.
    ///
    /// # Arguments
    /// * `m_diag`  – diagonal mass vector (length n_dof)
    /// * `k_diag`  – diagonal stiffness vector (length n_dof)
    /// * `c_diag`  – diagonal damping vector (length n_dof)
    /// * `f_mid`   – force vector at `t + Δt/2` (first, trapezoidal sub-step)
    /// * `f_end`   – force vector at `t + Δt` (second, three-point backward sub-step)
    /// * `dt`      – full step size Δt
    ///
    /// # Panics
    ///
    /// Panics if one of the slices is shorter than `n_dof`.
    pub fn step_diagonal_with_loads(
        &mut self,
        m_diag: &[f64],
        k_diag: &[f64],
        c_diag: &[f64],
        f_mid: &[f64],
        f_end: &[f64],
        dt: f64,
    ) {
        let h = 0.5 * dt;
        // Sub-step 1, trapezoidal rule over h = Δt/2: k_eff = 4/h² M + 2/h C + K.
        let trap_u = 4.0 / (h * h);
        let trap_v = 2.0 / h;
        // Sub-step 2, three-point backward Euler over Δt: k_eff = 9/Δt² M + 3/Δt C + K.
        let back_u = 9.0 / (dt * dt);
        let back_v = 3.0 / dt;
        for i in 0..self.n_dof {
            let (m, c, k) = (m_diag[i], c_diag[i], k_diag[i]);
            let k_eff_mid = trap_u * m + trap_v * c + k;
            let k_eff_end = back_u * m + back_v * c + k;
            if k_eff_mid.abs() < 1e-60 || k_eff_end.abs() < 1e-60 {
                continue;
            }
            let (u_n, v_n, a_n) = (self.u[i], self.v[i], self.a[i]);
            let f_eff_mid =
                f_mid[i] + m * (trap_u * u_n + 2.0 * trap_v * v_n + a_n) + c * (trap_v * u_n + v_n);
            let u_mid = f_eff_mid / k_eff_mid;
            let v_mid = trap_v * (u_mid - u_n) - v_n;
            let history_u = u_n - 4.0 * u_mid;
            let history_v = v_n - 4.0 * v_mid;
            let f_eff_end =
                f_end[i] - m * (history_v / dt + 3.0 * history_u / (dt * dt)) - c * history_u / dt;
            let u_new = f_eff_end / k_eff_end;
            let v_new = (history_u + 3.0 * u_new) / dt;
            let a_new = (history_v + 3.0 * v_new) / dt;
            self.u[i] = u_new;
            self.v[i] = v_new;
            self.a[i] = a_new;
        }
    }
}
/// Newmark linear acceleration parameters (β = 1/6, γ = 1/2).
///
/// Conditionally stable: Δt ≤ √3 / ω_max.
pub struct NewmarkLinearAcceleration;
impl NewmarkLinearAcceleration {
    /// Single SDOF step using linear acceleration assumption.
    ///
    /// Returns (u_new, v_new, a_new).
    pub fn step(
        u: f64,
        v: f64,
        a: f64,
        f_new: f64,
        omega: f64,
        zeta: f64,
        dt: f64,
    ) -> (f64, f64, f64) {
        let beta = 1.0 / 6.0_f64;
        let gamma = 0.5_f64;
        let k_eff =
            omega * omega + 2.0 * zeta * omega * gamma / (beta * dt) + 1.0 / (beta * dt * dt);
        let f_eff = f_new
            + u / (beta * dt * dt)
            + v / (beta * dt)
            + (1.0 / (2.0 * beta) - 1.0) * a
            + 2.0
                * zeta
                * omega
                * (gamma / (beta * dt) * u
                    + (gamma / beta - 1.0) * v
                    + dt * (gamma / (2.0 * beta) - 1.0) * a);
        let u_new = if k_eff.abs() < 1e-60 {
            u
        } else {
            f_eff / k_eff
        };
        let v_new = gamma / (beta * dt) * (u_new - u)
            - (gamma / beta - 1.0) * v
            - dt * (gamma / (2.0 * beta) - 1.0) * a;
        let a_new =
            (u_new - u) / (beta * dt * dt) - v / (beta * dt) - (1.0 / (2.0 * beta) - 1.0) * a;
        (u_new, v_new, a_new)
    }
}
/// State for the central difference explicit time integrator.
///
/// This is a conditionally stable explicit method requiring:
/// `dt < 2/omega_max` where `omega_max` is the highest natural frequency.
///
/// The update equations are:
/// ```text
/// a_n = M^{-1} * (F_n - K * u_n)
/// u_{n+1} = 2*u_n - u_{n-1} + dt^2 * a_n
/// v_n = (u_{n+1} - u_{n-1}) / (2*dt)
/// ```
///
/// Requires a diagonal (lumped) mass matrix for efficiency.
pub struct CentralDifferenceState {
    /// Current displacements.
    pub u: Vec<f64>,
    /// Previous displacements (u_{n-1}).
    pub u_prev: Vec<f64>,
    /// Current velocities (computed from central difference).
    pub v: Vec<f64>,
    /// Current accelerations.
    pub a: Vec<f64>,
    /// Diagonal of the lumped mass matrix (inverse).
    pub inv_m_diag: Vec<f64>,
}
impl CentralDifferenceState {
    /// Create a new central difference state with zero initial conditions.
    ///
    /// `inv_m_diag` is the diagonal of the inverse lumped mass matrix.
    pub fn new(n_dof: usize, inv_m_diag: Vec<f64>) -> Self {
        Self {
            u: vec![0.0; n_dof],
            u_prev: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
            inv_m_diag,
        }
    }
    /// Advance the state by one explicit time step.
    pub fn step(&mut self, stiffness: &CsrMatrix, force: &[f64], dt: f64, fixed_dofs: &[usize]) {
        let n = self.u.len();
        let f_int = stiffness.mul_vec(&self.u);
        for i in 0..n {
            self.a[i] = self.inv_m_diag[i] * (force[i] - f_int[i]);
        }
        let dt2 = dt * dt;
        let mut u_new = vec![0.0; n];
        for (i, u_new_i) in u_new.iter_mut().enumerate().take(n) {
            *u_new_i = 2.0 * self.u[i] - self.u_prev[i] + dt2 * self.a[i];
        }
        let inv_2dt = 0.5 / dt;
        for (i, v_i) in self.v.iter_mut().enumerate().take(n) {
            *v_i = (u_new[i] - self.u_prev[i]) * inv_2dt;
        }
        for &dof in fixed_dofs {
            u_new[dof] = 0.0;
            self.v[dof] = 0.0;
            self.a[dof] = 0.0;
        }
        self.u_prev = self.u.clone();
        self.u = u_new;
    }
}
/// State for the generalized-alpha time integrator.
///
/// Provides user-controlled high-frequency dissipation through the
/// spectral radius parameter `rho_inf` in `[0, 1]`:
/// - `rho_inf = 1.0`: no dissipation (equivalent to Newmark trapezoidal rule)
/// - `rho_inf = 0.0`: asymptotic annihilation of highest mode
pub struct GeneralizedAlphaState {
    /// Current displacements.
    pub u: Vec<f64>,
    /// Current velocities.
    pub v: Vec<f64>,
    /// Current accelerations.
    pub a: Vec<f64>,
    /// Alpha_m parameter.
    pub alpha_m: f64,
    /// Alpha_f parameter.
    pub alpha_f: f64,
    /// Newmark beta.
    pub beta: f64,
    /// Newmark gamma.
    pub gamma: f64,
}
impl GeneralizedAlphaState {
    /// Create a new generalized-alpha state from the spectral radius
    /// at infinity `rho_inf` in `[0, 1]`.
    pub fn new(n_dof: usize, rho_inf: f64) -> Self {
        let rho = rho_inf.clamp(0.0, 1.0);
        let alpha_m = (2.0 * rho - 1.0) / (rho + 1.0);
        let alpha_f = rho / (rho + 1.0);
        let gamma = 0.5 - alpha_m + alpha_f;
        let beta = 0.25 * (1.0 - alpha_m + alpha_f).powi(2);
        Self {
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
            alpha_m,
            alpha_f,
            beta,
            gamma,
        }
    }
    /// Advance the state by one time step.
    ///
    /// The generalized-alpha method evaluates the equation of motion
    /// at interpolated time levels:
    /// ```text
    /// M*a_{n+1-alpha_m} + K*u_{n+1-alpha_f} = F_{n+1-alpha_f}
    /// ```
    pub fn step(
        &mut self,
        mass: &CsrMatrix,
        stiffness: &CsrMatrix,
        force: &[f64],
        dt: f64,
        fixed_dofs: &[usize],
    ) {
        let n = self.u.len();
        let beta = self.beta;
        let gamma = self.gamma;
        let alpha_m = self.alpha_m;
        let alpha_f = self.alpha_f;
        let c0 = 1.0 / (beta * dt * dt);
        let u_pred: Vec<f64> = (0..n)
            .map(|i| self.u[i] + dt * self.v[i] + dt * dt * (0.5 - beta) * self.a[i])
            .collect();
        let v_pred: Vec<f64> = (0..n)
            .map(|i| self.v[i] + dt * (1.0 - gamma) * self.a[i])
            .collect();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for row in 0..stiffness.nrows {
            let start = stiffness.row_ptr[row];
            let end = stiffness.row_ptr[row + 1];
            for idx in start..end {
                let val = (1.0 - alpha_f) * stiffness.values[idx];
                triplets.push((row, stiffness.col_indices[idx], val));
            }
        }
        for row in 0..mass.nrows {
            let start = mass.row_ptr[row];
            let end = mass.row_ptr[row + 1];
            for idx in start..end {
                let val = (1.0 - alpha_m) * c0 * mass.values[idx];
                triplets.push((row, mass.col_indices[idx], val));
            }
        }
        let mut k_eff = CsrMatrix::from_triplets(n, n, &triplets);
        let k_u_n = stiffness.mul_vec(&self.u);
        let a_tilde: Vec<f64> = (0..n)
            .map(|i| self.u[i] * c0 + self.v[i] / (beta * dt) + self.a[i] * (0.5 / beta - 1.0))
            .collect();
        let m_a_tilde = mass.mul_vec(&a_tilde);
        let m_a_n = mass.mul_vec(&self.a);
        let mut f_eff: Vec<f64> = (0..n)
            .map(|i| {
                (1.0 - alpha_f) * force[i] + alpha_f * (force[i]) + (1.0 - alpha_m) * m_a_tilde[i]
                    - alpha_m * m_a_n[i]
                    - alpha_f * k_u_n[i]
            })
            .collect();
        for &dof in fixed_dofs {
            let start = k_eff.row_ptr[dof];
            let end = k_eff.row_ptr[dof + 1];
            for idx in start..end {
                k_eff.values[idx] = if k_eff.col_indices[idx] == dof {
                    1.0
                } else {
                    0.0
                };
            }
            for row in 0..n {
                if row == dof {
                    continue;
                }
                let rs = k_eff.row_ptr[row];
                let re = k_eff.row_ptr[row + 1];
                for idx in rs..re {
                    if k_eff.col_indices[idx] == dof {
                        k_eff.values[idx] = 0.0;
                    }
                }
            }
            f_eff[dof] = 0.0;
        }
        let u_new = crate::solvers::conjugate_gradient(&k_eff, &f_eff, &u_pred, 10000, 1e-10);
        let a_new: Vec<f64> = (0..n).map(|i| (u_new[i] - u_pred[i]) * c0).collect();
        let v_new: Vec<f64> = (0..n).map(|i| v_pred[i] + dt * gamma * a_new[i]).collect();
        self.u = u_new;
        self.v = v_new;
        self.a = a_new;
        for &dof in fixed_dofs {
            self.u[dof] = 0.0;
            self.v[dof] = 0.0;
            self.a[dof] = 0.0;
        }
    }
}
/// State for the Wilson-θ time integration method.
///
/// The Wilson-θ method assumes the acceleration varies linearly over the
/// extended interval `[t, t + θ Δt]`, evaluates equilibrium at `t + θ Δt`
/// (θ ≥ 1.37 for unconditional stability) and interpolates the acceleration
/// back to `t + Δt`.  It is second-order accurate; for θ > 1 it adds numerical
/// damping of the high-frequency response and some period elongation.
///
/// Reference: K.-J. Bathe, *Finite Element Procedures*, Sec. 9.2 (direct integration
/// methods).
pub struct WilsonThetaState {
    /// Number of DOF.
    pub n_dof: usize,
    /// Current displacements.
    pub u: Vec<f64>,
    /// Current velocities.
    pub v: Vec<f64>,
    /// Current accelerations.
    pub a: Vec<f64>,
    /// θ parameter (default 1.4).
    pub theta: f64,
}
impl WilsonThetaState {
    /// Create a new Wilson-θ integrator (default θ = 1.4).
    pub fn new(n_dof: usize) -> Self {
        Self {
            n_dof,
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
            theta: 1.4,
        }
    }
    /// Single step for diagonal (lumped) systems.
    ///
    /// Solves the modified equation at t + θ Δt then interpolates.
    pub fn step_diagonal(
        &mut self,
        m_diag: &[f64],
        k_diag: &[f64],
        c_diag: &[f64],
        f_n: &[f64],
        f_np1: &[f64],
        dt: f64,
    ) {
        let th = self.theta;
        let n = self.n_dof;
        let f_th: Vec<f64> = (0..n).map(|i| f_n[i] + th * (f_np1[i] - f_n[i])).collect();
        // Integration constants of Bathe (Finite Element Procedures, Sec. 9.2):
        // a0 = 6/(θ·Δt)², a1 = 3/(θ·Δt), a2 = 2·a1 = 6/(θ·Δt), a3 = θ·Δt/2.
        let a0 = 6.0 / (th * th * dt * dt);
        let a1 = 3.0 / (th * dt);
        let a2 = 2.0 * a1;
        let a3 = 0.5 * th * dt;
        let mut a_th = vec![0.0_f64; n];
        for i in 0..n {
            let k_eff = k_diag[i] + a0 * m_diag[i] + a1 * c_diag[i];
            let f_eff = f_th[i]
                + m_diag[i] * (a0 * self.u[i] + a2 * self.v[i] + 2.0 * self.a[i])
                + c_diag[i] * (a1 * self.u[i] + 2.0 * self.v[i] + a3 * self.a[i]);
            let u_th = if k_eff.abs() < 1e-60 {
                self.u[i]
            } else {
                f_eff / k_eff
            };
            a_th[i] = (u_th - self.u[i]) * a0 - a2 * self.v[i] - 2.0 * self.a[i];
        }
        for (i, a_th_i) in a_th.iter().enumerate().take(n) {
            let a_new = self.a[i] + (a_th_i - self.a[i]) / th;
            let v_new = self.v[i] + dt / 2.0 * (a_new + self.a[i]);
            let u_new = self.u[i] + dt * self.v[i] + dt * dt / 6.0 * (2.0 * self.a[i] + a_new);
            self.a[i] = a_new;
            self.v[i] = v_new;
            self.u[i] = u_new;
        }
    }
}
/// State for seismic analysis including relative displacements.
///
/// In seismic analysis, the equation of motion is expressed in relative coordinates:
/// `u = u_abs - u_g`
/// where `u_g` is the ground displacement.
pub struct SeismicState {
    /// Relative displacements (m).
    pub u_rel: Vec<f64>,
    /// Relative velocities (m/s).
    pub v_rel: Vec<f64>,
    /// Relative accelerations (m/s^2).
    pub a_rel: Vec<f64>,
    /// Ground displacement history.
    pub u_ground: Vec<f64>,
}
impl SeismicState {
    /// Create a new seismic state with zero initial conditions.
    pub fn new(n_dof: usize) -> Self {
        Self {
            u_rel: vec![0.0; n_dof],
            v_rel: vec![0.0; n_dof],
            a_rel: vec![0.0; n_dof],
            u_ground: vec![0.0; n_dof],
        }
    }
    /// Compute the total (absolute) displacement.
    pub fn absolute_displacement(&self) -> Vec<f64> {
        self.u_rel
            .iter()
            .zip(self.u_ground.iter())
            .map(|(ur, ug)| ur + ug)
            .collect()
    }
}
/// Parameters for the linear acceleration Newmark variant (β = 1/6, γ = 1/2).
///
/// This method is conditionally stable with:
/// Ω_crit = sqrt(3) ≈ 1.732
/// (where Ω = ω * dt)
pub struct NewmarkLinearAccelerationParams;
impl NewmarkLinearAccelerationParams {
    /// Beta parameter for linear acceleration.
    pub const BETA: f64 = 1.0 / 6.0;
    /// Gamma parameter.
    pub const GAMMA: f64 = 0.5;
    /// Critical time step: dt_crit = sqrt(3) / omega_max.
    pub fn critical_dt(omega_max: f64) -> f64 {
        if omega_max < f64::EPSILON {
            return f64::MAX;
        }
        3.0_f64.sqrt() / omega_max
    }
}
/// State for the HHT-alpha (Hilber-Hughes-Taylor) time integrator.
///
/// The HHT-alpha method introduces numerical dissipation via a parameter
/// `alpha_hht` in `[-1/3, 0]`. Setting `alpha_hht = 0` recovers the
/// standard Newmark method.
///
/// The method modifies the equation of motion to:
/// ```text
/// M*a_{n+1} + (1+alpha)*C*v_{n+1} - alpha*C*v_n
///   + (1+alpha)*K*u_{n+1} - alpha*K*u_n = (1+alpha)*F_{n+1} - alpha*F_n
/// ```
pub struct HhtAlphaState {
    /// Current displacements.
    pub u: Vec<f64>,
    /// Current velocities.
    pub v: Vec<f64>,
    /// Current accelerations.
    pub a: Vec<f64>,
    /// HHT alpha parameter, typically in `[-1/3, 0]`.
    pub alpha_hht: f64,
    /// Newmark beta parameter (derived from alpha).
    pub beta: f64,
    /// Newmark gamma parameter (derived from alpha).
    pub gamma: f64,
}
impl HhtAlphaState {
    /// Create a new HHT-alpha state with optimal dissipation parameters.
    ///
    /// `alpha_hht` should be in `[-1/3, 0]`.
    /// The Newmark parameters are chosen for optimal accuracy:
    /// `gamma = 0.5 - alpha_hht`, `beta = (1 - alpha_hht)^2 / 4`.
    pub fn new(n_dof: usize, alpha_hht: f64) -> Self {
        let gamma = 0.5 - alpha_hht;
        let beta = (1.0 - alpha_hht).powi(2) / 4.0;
        Self {
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
            alpha_hht,
            beta,
            gamma,
        }
    }
    /// Advance the state by one time step.
    ///
    /// This is a simplified implementation using the effective stiffness
    /// approach with HHT modification. Without damping, the effective
    /// equation becomes:
    /// ```text
    /// M*a_{n+1} + (1+alpha)*K*u_{n+1} = (1+alpha)*F_{n+1} - alpha*F_n + alpha*K*u_n
    /// ```
    pub fn step(
        &mut self,
        mass: &CsrMatrix,
        stiffness: &CsrMatrix,
        force_new: &[f64],
        force_old: &[f64],
        dt: f64,
        fixed_dofs: &[usize],
    ) {
        let n = self.u.len();
        let alpha = self.alpha_hht;
        let beta = self.beta;
        let gamma = self.gamma;
        let inv_beta_dt2 = 1.0 / (beta * dt * dt);
        let u_pred: Vec<f64> = (0..n)
            .map(|i| self.u[i] + dt * self.v[i] + dt * dt * (0.5 - beta) * self.a[i])
            .collect();
        let v_pred: Vec<f64> = (0..n)
            .map(|i| self.v[i] + dt * (1.0 - gamma) * self.a[i])
            .collect();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for row in 0..stiffness.nrows {
            let start = stiffness.row_ptr[row];
            let end = stiffness.row_ptr[row + 1];
            for idx in start..end {
                let val = (1.0 + alpha) * stiffness.values[idx];
                triplets.push((row, stiffness.col_indices[idx], val));
            }
        }
        for row in 0..mass.nrows {
            let start = mass.row_ptr[row];
            let end = mass.row_ptr[row + 1];
            for idx in start..end {
                let val = mass.values[idx] * inv_beta_dt2;
                triplets.push((row, mass.col_indices[idx], val));
            }
        }
        let mut k_eff = CsrMatrix::from_triplets(n, n, &triplets);
        let k_u_n = stiffness.mul_vec(&self.u);
        let a_tilde: Vec<f64> = (0..n)
            .map(|i| {
                self.u[i] * inv_beta_dt2 + self.v[i] / (beta * dt) + self.a[i] * (0.5 / beta - 1.0)
            })
            .collect();
        let m_a_tilde = mass.mul_vec(&a_tilde);
        let mut f_eff: Vec<f64> = (0..n)
            .map(|i| {
                (1.0 + alpha) * force_new[i] - alpha * force_old[i]
                    + alpha * k_u_n[i]
                    + m_a_tilde[i]
            })
            .collect();
        for &dof in fixed_dofs {
            let start = k_eff.row_ptr[dof];
            let end = k_eff.row_ptr[dof + 1];
            for idx in start..end {
                k_eff.values[idx] = if k_eff.col_indices[idx] == dof {
                    1.0
                } else {
                    0.0
                };
            }
            for row in 0..n {
                if row == dof {
                    continue;
                }
                let rs = k_eff.row_ptr[row];
                let re = k_eff.row_ptr[row + 1];
                for idx in rs..re {
                    if k_eff.col_indices[idx] == dof {
                        k_eff.values[idx] = 0.0;
                    }
                }
            }
            f_eff[dof] = 0.0;
        }
        let u_new = crate::solvers::conjugate_gradient(&k_eff, &f_eff, &u_pred, 10000, 1e-10);
        let a_new: Vec<f64> = (0..n)
            .map(|i| (u_new[i] - u_pred[i]) * inv_beta_dt2)
            .collect();
        let v_new: Vec<f64> = (0..n).map(|i| v_pred[i] + dt * gamma * a_new[i]).collect();
        self.u = u_new;
        self.v = v_new;
        self.a = a_new;
        for &dof in fixed_dofs {
            self.u[dof] = 0.0;
            self.v[dof] = 0.0;
            self.a[dof] = 0.0;
        }
    }
}
/// State for the Newmark-beta implicit time integrator.
///
/// Implements the average acceleration method (beta=0.25, gamma=0.5),
/// which is unconditionally stable for linear systems without damping.
pub struct NewmarkState {
    /// Current displacements (length = n_dof).
    pub u: Vec<f64>,
    /// Current velocities (length = n_dof).
    pub v: Vec<f64>,
    /// Current accelerations (length = n_dof).
    pub a: Vec<f64>,
    /// Newmark beta parameter (0.25 for average acceleration).
    pub beta: f64,
    /// Newmark gamma parameter (0.5 for average acceleration).
    pub gamma: f64,
}
impl NewmarkState {
    /// Create a new `NewmarkState` with zero initial conditions.
    pub fn new(n_dof: usize) -> Self {
        Self {
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
            beta: 0.25,
            gamma: 0.5,
        }
    }
    /// Advance the state by one time step using the Newmark-beta method.
    pub fn step(
        &mut self,
        mass: &CsrMatrix,
        stiffness: &CsrMatrix,
        force: &[f64],
        dt: f64,
        fixed_dofs: &[usize],
    ) {
        let n = self.u.len();
        assert_eq!(mass.nrows, n);
        assert_eq!(stiffness.nrows, n);
        assert_eq!(force.len(), n);
        let beta = self.beta;
        let gamma = self.gamma;
        let inv_beta_dt2 = 1.0 / (beta * dt * dt);
        let inv_beta_dt = 1.0 / (beta * dt);
        let one_over_2beta_minus1 = 0.5 / beta - 1.0;
        let u_pred: Vec<f64> = (0..n)
            .map(|i| self.u[i] + dt * self.v[i] + dt * dt * (0.5 - beta) * self.a[i])
            .collect();
        let v_pred: Vec<f64> = (0..n)
            .map(|i| self.v[i] + dt * (1.0 - gamma) * self.a[i])
            .collect();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for row in 0..stiffness.nrows {
            let start = stiffness.row_ptr[row];
            let end = stiffness.row_ptr[row + 1];
            for idx in start..end {
                triplets.push((row, stiffness.col_indices[idx], stiffness.values[idx]));
            }
        }
        for row in 0..mass.nrows {
            let start = mass.row_ptr[row];
            let end = mass.row_ptr[row + 1];
            for idx in start..end {
                let val = mass.values[idx] * inv_beta_dt2;
                triplets.push((row, mass.col_indices[idx], val));
            }
        }
        let mut k_eff = CsrMatrix::from_triplets(n, n, &triplets);
        let a_tilde: Vec<f64> = (0..n)
            .map(|i| {
                self.u[i] * inv_beta_dt2
                    + self.v[i] * inv_beta_dt
                    + self.a[i] * one_over_2beta_minus1
            })
            .collect();
        let m_a_tilde = mass.mul_vec(&a_tilde);
        let mut f_eff: Vec<f64> = (0..n).map(|i| force[i] + m_a_tilde[i]).collect();
        for &dof in fixed_dofs {
            assert!(dof < n, "fixed DOF {dof} out of range");
            let start = k_eff.row_ptr[dof];
            let end = k_eff.row_ptr[dof + 1];
            for idx in start..end {
                if k_eff.col_indices[idx] == dof {
                    k_eff.values[idx] = 1.0;
                } else {
                    k_eff.values[idx] = 0.0;
                }
            }
            for row in 0..n {
                if row == dof {
                    continue;
                }
                let rs = k_eff.row_ptr[row];
                let re = k_eff.row_ptr[row + 1];
                for idx in rs..re {
                    if k_eff.col_indices[idx] == dof {
                        k_eff.values[idx] = 0.0;
                    }
                }
            }
            f_eff[dof] = 0.0;
        }
        let u_new = crate::solvers::conjugate_gradient(&k_eff, &f_eff, &u_pred, 10000, 1e-10);
        let a_new: Vec<f64> = (0..n)
            .map(|i| (u_new[i] - u_pred[i]) * inv_beta_dt2)
            .collect();
        let v_new: Vec<f64> = (0..n).map(|i| v_pred[i] + dt * gamma * a_new[i]).collect();
        self.u = u_new;
        self.v = v_new;
        self.a = a_new;
        for &dof in fixed_dofs {
            self.u[dof] = 0.0;
            self.v[dof] = 0.0;
            self.a[dof] = 0.0;
        }
    }
}
/// Input for a multi-mode response spectrum analysis.
pub struct ResponseSpectrumInput {
    /// Natural frequencies \[rad/s\].
    pub omega_n: Vec<f64>,
    /// Modal damping ratios.
    pub zeta: Vec<f64>,
    /// Mode shapes: `phi[mode][dof]`.
    pub phi: Vec<Vec<f64>>,
    /// Modal participation factors (one per mode).
    pub gamma: Vec<f64>,
    /// Spectral displacement values Sd(omega, zeta) \[m\].
    pub sd: Vec<f64>,
    /// Spectral acceleration values Sa(omega, zeta) \[m/s²\].
    pub sa: Vec<f64>,
}
/// Result of a modal superposition analysis.
pub struct ModalSuperpositionResult {
    /// Time history of displacement for each DOF: `disp[time_step][dof]`.
    pub disp: Vec<Vec<f64>>,
    /// Time history of modal coordinates: `q[time_step][mode]`.
    pub modal_coords: Vec<Vec<f64>>,
}
/// Lightweight dynamic solver that wraps state vectors for time integration.
pub struct DynamicSolver {
    /// Number of DOF.
    pub n_dof: usize,
    /// Displacement vector.
    pub u: Vec<f64>,
    /// Velocity vector.
    pub v: Vec<f64>,
    /// Acceleration vector.
    pub a: Vec<f64>,
}
impl DynamicSolver {
    /// Create a new solver with zero initial conditions.
    pub fn new(n_dof: usize) -> Self {
        Self {
            n_dof,
            u: vec![0.0; n_dof],
            v: vec![0.0; n_dof],
            a: vec![0.0; n_dof],
        }
    }
    /// Perform one generalized-alpha time step.
    ///
    /// The generalized-alpha method (Chung & Hulbert, 1993) achieves high-frequency
    /// numerical dissipation while maintaining second-order accuracy.  The step is
    /// delegated to [`GeneralizedAlphaState::step`], which assembles the full sparse
    /// effective matrix `(1 − α_m)/(β Δt²) M + (1 − α_f) K` (so consistent, non-diagonal
    /// `M` and `K` are supported) and solves it with conjugate gradients.
    ///
    /// # Parameters
    /// * `mass`, `stiffness` – symmetric positive (semi-)definite system matrices.
    /// * `force` – external load at the generalized-alpha time level `t_{n+1−α_f}`.
    /// * `dt` – time step.
    /// * `rho_inf` – spectral radius at infinity ∈ \[0, 1\].
    ///   - `rho_inf = 1.0`: no dissipation (Newmark average acceleration).
    ///   - `rho_inf = 0.0`: maximum dissipation.
    ///
    /// `self.a` must be the acceleration consistent with `self.u`, `self.v` and the
    /// load at the start of the step.
    pub fn generalized_alpha(
        &mut self,
        mass: &CsrMatrix,
        stiffness: &CsrMatrix,
        force: &[f64],
        dt: f64,
        rho_inf: f64,
    ) {
        let mut state = GeneralizedAlphaState::new(self.n_dof, rho_inf);
        state.u.clone_from(&self.u);
        state.v.clone_from(&self.v);
        state.a.clone_from(&self.a);
        state.step(mass, stiffness, force, dt, &[]);
        self.u = state.u;
        self.v = state.v;
        self.a = state.a;
    }
    /// Compute Rayleigh damping matrix C = alpha * M + beta * K using
    /// two-mode targeting.
    ///
    /// Returns the `CsrMatrix` damping matrix.
    pub fn compute_rayleigh_damping(
        &self,
        mass: &CsrMatrix,
        stiffness: &CsrMatrix,
        omega1: f64,
        zeta1: f64,
        omega2: f64,
        zeta2: f64,
    ) -> CsrMatrix {
        let (alpha, beta) = compute_rayleigh_coefficients(omega1, zeta1, omega2, zeta2);
        rayleigh_damping_matrix(mass, stiffness, alpha, beta)
    }
    /// Apply modal truncation correction (static correction for residual modes).
    ///
    /// See `modal_truncation_correction`.
    pub fn modal_truncation_correction(
        &self,
        omega_n: &[f64],
        phi: &[Vec<f64>],
        force: &[f64],
        n_modes_included: usize,
    ) -> Vec<f64> {
        modal_truncation_correction(omega_n, phi, force, n_modes_included)
    }
}
/// Monitor energy conservation over a sequence of time steps.
///
/// Returns a vector of `(time, kinetic_energy, strain_energy, total_energy)`.
pub struct EnergyMonitor {
    /// History: `(time, T, U, T+U)`.
    pub history: Vec<(f64, f64, f64, f64)>,
}
impl EnergyMonitor {
    /// Create a new energy monitor.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }
    /// Record an energy sample.
    pub fn record(&mut self, time: f64, kinetic: f64, strain: f64) {
        self.history.push((time, kinetic, strain, kinetic + strain));
    }
    /// Compute the maximum relative energy drift: `max(|E_i - E_0|) / |E_0|`.
    pub fn max_relative_drift(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let e0 = self.history[0].3;
        if e0.abs() < 1e-60 {
            return 0.0;
        }
        self.history
            .iter()
            .map(|&(_, _, _, e)| ((e - e0) / e0).abs())
            .fold(0.0_f64, f64::max)
    }
    /// Get the number of recorded samples.
    pub fn len(&self) -> usize {
        self.history.len()
    }
    /// Check if the monitor has no recorded samples.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

impl Default for EnergyMonitor {
    fn default() -> Self {
        Self::new()
    }
}
