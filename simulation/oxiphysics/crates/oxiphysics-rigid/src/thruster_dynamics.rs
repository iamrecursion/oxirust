// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Thruster and propulsion dynamics for underwater/aerial vehicles.
//!
//! Provides first-order thruster models, propeller coefficients, thrust allocation,
//! and a simplified brushless motor electrical model.
//!
//! # Overview
//!
//! - [`Thruster`] — first-order thruster with saturation and response time.
//! - [`ThrusterArray`] — collection of thrusters with force-torque summation.
//! - [`BrushlessMotor`] — electrical brushless-DC motor model.
//! - [`kt_kq_polynomial`] — propeller Kt/Kq as polynomial in advance ratio J.
//! - [`propeller_thrust`] / [`propeller_torque`] — thrust and torque from Kt/Kq.
//! - [`advance_ratio`] — dimensionless advance ratio J.
//! - [`propeller_efficiency`] — open-water propeller efficiency.
//! - [`thrust_allocation_pseudoinverse`] — minimum-norm wrench allocation.
//! - [`required_battery_capacity`] — required battery capacity in Ah.

use nalgebra::linalg::SVD;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Thruster
// ─────────────────────────────────────────────────────────────────────────────

/// First-order thruster model with saturation.
#[derive(Debug, Clone)]
pub struct Thruster {
    /// Maximum achievable thrust \[N\].
    pub max_thrust: f64,
    /// Maximum achievable torque \[N·m\].
    pub max_torque: f64,
    /// First-order time constant \[s\].
    pub response_time: f64,
    /// Steady-state efficiency (0..1].
    pub efficiency: f64,
    /// Current thrust output \[N\].
    pub current_thrust: f64,
    /// Current shaft speed \[rev/s\].
    pub current_rpm: f64,
}

impl Thruster {
    /// Create a new thruster.
    ///
    /// * `max_thrust` — saturation thrust \[N\]
    /// * `response_time` — first-order time constant τ \[s\]
    /// * `efficiency` — propulsive efficiency (clamped to (0, 1])
    pub fn new(max_thrust: f64, response_time: f64, efficiency: f64) -> Self {
        Self {
            max_thrust,
            max_torque: max_thrust * 0.1, // heuristic: torque ~ 10 % of thrust * 1 m
            response_time: response_time.max(1e-6),
            efficiency: efficiency.clamp(1e-6, 1.0),
            current_thrust: 0.0,
            current_rpm: 0.0,
        }
    }

    /// Set the desired thrust command \[N\] (clamped to ±max_thrust).
    pub fn set_command(&mut self, cmd: f64) {
        // Stored as desired; step() performs the first-order lag.
        let _clamped = cmd.clamp(-self.max_thrust, self.max_thrust);
        // We store command in current_rpm as a proxy (sign + magnitude).
        self.current_rpm = _clamped;
    }

    /// Advance the thruster state by one time step `dt` \[s\].
    ///
    /// Applies first-order lag: `T_dot = (T_cmd − T) / τ`.
    pub fn step(&mut self, dt: f64) {
        let cmd = self.current_rpm; // desired thrust stored here
        let error = cmd - self.current_thrust;
        self.current_thrust += error * dt / self.response_time;
        // Saturate
        self.current_thrust = self.current_thrust.clamp(-self.max_thrust, self.max_thrust);
    }

    /// Current thrust output \[N\].
    pub fn thrust(&self) -> f64 {
        self.current_thrust
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Propeller free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Propeller thrust and torque coefficients as polynomials in advance ratio J.
///
/// Returns `(Kt, Kq)` using a simple two-term Wageningen B-series approximation:
/// `Kt(J) = 0.3 − 0.25 · J`
/// `Kq(J) = 0.04 − 0.03 · J`
///
/// Negative values are clamped to zero (non-propulsive regime).
pub fn kt_kq_polynomial(j: f64) -> (f64, f64) {
    let kt = (0.3 - 0.25 * j).max(0.0);
    let kq = (0.04 - 0.03 * j).max(1e-9);
    (kt, kq)
}

/// Propeller thrust \[N\].
///
/// `T = Kt · ρ · n² · D⁴`
///
/// * `kt` — thrust coefficient \[-\]
/// * `rho` — fluid density \[kg/m³\]
/// * `n` — shaft speed \[rev/s\]
/// * `d` — propeller diameter \[m\]
pub fn propeller_thrust(kt: f64, rho: f64, n: f64, d: f64) -> f64 {
    kt * rho * n * n * d.powi(4)
}

/// Propeller torque \[N·m\].
///
/// `Q = Kq · ρ · n² · D⁵`
pub fn propeller_torque(kq: f64, rho: f64, n: f64, d: f64) -> f64 {
    kq * rho * n * n * d.powi(5)
}

/// Advance ratio J \[-\].
///
/// `J = Va / (n · D)`
///
/// * `va` — advance velocity (water inflow) \[m/s\]
/// * `n` — shaft speed \[rev/s\]
/// * `d` — propeller diameter \[m\]
pub fn advance_ratio(va: f64, n: f64, d: f64) -> f64 {
    if n.abs() < 1e-12 || d.abs() < 1e-12 {
        return 0.0;
    }
    va / (n * d)
}

/// Open-water propeller efficiency \[-\].
///
/// `η = J · Kt / (2π · Kq)`
///
/// Clamped to \[0, 1\].
pub fn propeller_efficiency(j: f64, kt: f64, kq: f64) -> f64 {
    if kq.abs() < 1e-15 {
        return 0.0;
    }
    (j * kt / (2.0 * PI * kq)).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// ThrusterArray
// ─────────────────────────────────────────────────────────────────────────────

/// Array of thrusters with positions and 6-DOF direction vectors.
#[derive(Debug, Clone)]
pub struct ThrusterArray {
    /// Thrusters with (thruster, position \[m\], 6-DOF wrench column).
    pub thrusters: Vec<(Thruster, [f64; 3], [f64; 6])>,
}

impl ThrusterArray {
    /// Create an empty thruster array.
    pub fn new() -> Self {
        Self {
            thrusters: Vec::new(),
        }
    }

    /// Add a thruster with body-frame position and 6-DOF wrench direction.
    pub fn add_thruster(&mut self, thruster: Thruster, pos: [f64; 3], direction: [f64; 6]) {
        self.thrusters.push((thruster, pos, direction));
    }

    /// Number of thrusters in the array.
    pub fn count(&self) -> usize {
        self.thrusters.len()
    }

    /// Total 6-DOF wrench `[Fx, Fy, Fz, Mx, My, Mz]` produced by all thrusters.
    pub fn total_force_torque(&self) -> [f64; 6] {
        let mut result = [0.0_f64; 6];
        for (thruster, _pos, dir) in &self.thrusters {
            let t = thruster.thrust();
            for i in 0..6 {
                result[i] += t * dir[i];
            }
        }
        result
    }
}

impl Default for ThrusterArray {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thrust allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Relative singular-value cutoff of the thrust-allocation pseudo-inverse.
///
/// Singular values of the configuration matrix below this fraction of the
/// largest one are treated as zero: the corresponding wrench directions cannot
/// be produced by the thrusters and are left unallocated instead of being
/// amplified into absurd thrust commands.
const ALLOCATION_RANK_TOLERANCE: f64 = 1e-10;

/// Iteration budget of the SVD behind the thrust-allocation pseudo-inverse.
///
/// A 6 × n Golub–Kahan SVD converges in a few dozen iterations; the bound only
/// exists so that pathological input can never hang the allocator.
const ALLOCATION_SVD_MAX_ITERATIONS: usize = 10_000;

/// Minimum-norm least-squares thrust command `u = B⁺ w` for the 6 × n thruster
/// configuration matrix `B`, given column by column.
///
/// * `columns` — one entry per thruster: the wrench `[Fx, Fy, Fz, Mx, My, Mz]`
///   produced by that thruster at unit thrust (so `B = [columns[0] … columns[n-1]]`).
/// * `wrench` — desired wrench `w`.
///
/// `B⁺` is the Moore–Penrose pseudo-inverse computed from the singular value
/// decomposition of `B` itself (never from the normal equations `B Bᵀ`, which
/// square the condition number and are singular whenever the rank of `B` is
/// below 6). The result is therefore well defined for every layout:
///
/// * full row rank (`rank B = 6`): `B u = w` exactly and `u` has minimum norm,
/// * rank-deficient or fewer than six thrusters: `u` minimises `‖B u − w‖` and,
///   among all minimisers, `‖u‖` (so `Bᵀ (B u − w) = 0`).
///
/// An empty configuration yields an empty vector. A non-finite entry in
/// `columns` or `wrench`, or an SVD that fails to converge, yields a vector of
/// `NaN` (one per thruster) so that the failure cannot be mistaken for a valid
/// command.
pub(crate) fn wrench_pseudo_inverse(columns: &[[f64; 6]], wrench: &[f64; 6]) -> Vec<f64> {
    let n = columns.len();
    if n == 0 {
        return Vec::new();
    }
    let finite =
        columns.iter().flatten().all(|x| x.is_finite()) && wrench.iter().all(|x| x.is_finite());
    if !finite {
        return vec![f64::NAN; n];
    }

    let b = DMatrix::<f64>::from_fn(6, n, |row, col| columns[col][row]);
    let w = DVector::<f64>::from_column_slice(wrench);
    let Some(svd) = SVD::try_new_unordered(
        b,
        true,
        true,
        5.0 * f64::EPSILON,
        ALLOCATION_SVD_MAX_ITERATIONS,
    ) else {
        return vec![f64::NAN; n];
    };
    let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    match svd.pseudo_inverse(ALLOCATION_RANK_TOLERANCE * sigma_max) {
        Ok(pinv) => (pinv * w).as_slice().to_vec(),
        Err(_) => vec![f64::NAN; n],
    }
}

/// Minimum-norm thrust allocation via the Moore–Penrose pseudo-inverse.
///
/// Given the thruster configuration matrix `B` (one entry of `b_matrix` per
/// thruster: its 6-DOF wrench at unit thrust) and a desired wrench
/// `[Fx, Fy, Fz, Mx, My, Mz]`, returns the commanded thrust vector `u = B⁺ w`
/// computed from the SVD of `B`.
///
/// For a full-rank layout `u = Bᵀ (B Bᵀ)⁻¹ w` reproduces the wrench exactly with
/// minimum thrust norm. For rank-deficient layouts (e.g. a single thruster or
/// coplanar horizontal thrusters) `u` is the minimum-norm least-squares
/// solution, so two parallel thrusters share the load instead of each one
/// delivering the full wrench. No thrust limits are applied here.
///
/// Returns an empty vector for an empty configuration. Non-finite input (or an
/// SVD that fails to converge) yields a vector of `NaN`, one per thruster.
pub fn thrust_allocation_pseudoinverse(b_matrix: &[[f64; 6]], wrench: [f64; 6]) -> Vec<f64> {
    wrench_pseudo_inverse(b_matrix, &wrench)
}

// ─────────────────────────────────────────────────────────────────────────────
// BrushlessMotor
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified brushless DC motor (electrical + mechanical model).
///
/// The motor is described by its DC-equivalent circuit
///
/// ```text
/// L di/dt = V − R i − Ke ω        (electrical)
/// J dω/dt = Kt i − τ_load         (mechanical)
/// ```
///
/// where, in SI units, the back-EMF constant `Ke` \[V·s/rad\] and the torque
/// constant `Kt` \[N·m/A\] are both the reciprocal of the velocity constant,
/// `Ke = Kt = 1 / Kv` (`Kv` in rad/(s·V)). Consequently the no-load speed at a
/// terminal voltage `V` is `ω = Kv · V` and the electrical power balance
/// `V i = R i² + Kt i ω` holds in steady state.
///
/// The model covers the motoring quadrant only, like an ESC without active
/// braking: the phase current never becomes negative (when the terminal voltage
/// falls below the back-EMF the rotor freewheels and only the load torque slows
/// it down — there is no regenerative braking) and the rotor does not reverse.
#[derive(Debug, Clone)]
pub struct BrushlessMotor {
    /// Motor velocity constant `Kv` \[rad/(s·V)\] — no-load speed per volt.
    /// [`BrushlessMotor::new`] converts it from the catalogue unit rpm/V.
    pub kv: f64,
    /// Phase resistance \[Ω\].
    pub resistance: f64,
    /// Phase inductance \[H\].
    pub inductance: f64,
    /// Rotor inertia \[kg·m²\].
    pub inertia: f64,
    /// Angular velocity \[rad/s\].
    pub omega: f64,
    /// Phase current \[A\].
    pub current: f64,
}

impl BrushlessMotor {
    /// Create a new brushless motor.
    ///
    /// * `kv` — velocity constant \[rpm/V\]; internally converted to \[rad/(s·V)\]
    /// * `resistance` — phase resistance \[Ω\]
    pub fn new(kv: f64, resistance: f64) -> Self {
        Self {
            kv: kv * (2.0 * PI / 60.0), // store as rad/(s·V)
            resistance: resistance.max(1e-9),
            inductance: 1e-4, // 0.1 mH default
            inertia: 1e-5,    // 10 μkg·m² default
            omega: 0.0,
            current: 0.0,
        }
    }

    /// Back-EMF constant `Ke = 1 / Kv` \[V·s/rad\].
    pub fn back_emf_constant(&self) -> f64 {
        1.0 / self.kv.max(1e-12)
    }

    /// Torque constant `Kt` \[N·m/A\]. In SI units it equals the back-EMF
    /// constant, `Kt = Ke = 1 / Kv`.
    pub fn torque_constant(&self) -> f64 {
        self.back_emf_constant()
    }

    /// Advance the motor model by one time step `dt` \[s\].
    ///
    /// * `voltage` — applied terminal voltage \[V\]
    /// * `load_torque` — external load torque opposing rotation \[N·m\]
    pub fn step(&mut self, voltage: f64, load_torque: f64, dt: f64) {
        let ke = self.back_emf_constant();
        // Back-EMF: e = Ke · ω  (equals ω / Kv, i.e. V at the no-load speed).
        let back_emf = ke * self.omega;
        // Electrical: L · di/dt = V − R·i − e
        let di_dt =
            (voltage - self.resistance * self.current - back_emf) / self.inductance.max(1e-12);
        self.current += di_dt * dt;
        // Motoring quadrant only: the drive cannot push current back into the
        // supply, so a rotor spinning above V / Ke freewheels (see type docs).
        self.current = self.current.max(0.0);

        // Motor torque: τ_motor = Kt · i
        let motor_torque = self.torque_constant() * self.current;
        // Mechanical: J · dω/dt = τ_motor − τ_load
        let d_omega = (motor_torque - load_torque.abs()) / self.inertia.max(1e-15);
        self.omega = (self.omega + d_omega * dt).max(0.0);
    }

    /// Electrical input power \[W\].
    pub fn power_in(&self, voltage: f64) -> f64 {
        voltage * self.current.abs()
    }

    /// Mechanical output power \[W\] (`τ_motor · ω = Kt · i · ω`).
    pub fn power_out(&self) -> f64 {
        self.torque_constant() * self.current * self.omega
    }

    /// Motor efficiency \[-\] (clamped to \[0, 1\]).
    pub fn efficiency(&self, voltage: f64) -> f64 {
        let p_in = self.power_in(voltage);
        if p_in < 1e-12 {
            return 0.0;
        }
        (self.power_out() / p_in).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Battery capacity
// ─────────────────────────────────────────────────────────────────────────────

/// Required battery capacity \[Ah\] for a given mission.
///
/// `C = P_shaft / (V · η) · t_hours`
///
/// where `P_shaft = thrust * v_vehicle` is approximated as `thrust / efficiency`
/// if `v_vehicle` is unknown.
///
/// * `thrust` — required thrust \[N\]
/// * `efficiency` — overall system efficiency \[-\]
/// * `duration_hours` — mission duration \[h\]
/// * `voltage` — battery nominal voltage \[V\]
pub fn required_battery_capacity(
    thrust: f64,
    efficiency: f64,
    duration_hours: f64,
    voltage: f64,
) -> f64 {
    let eta = efficiency.clamp(1e-6, 1.0);
    let power_watts = thrust / eta; // P = F / η (simplified)
    let current_amps = power_watts / voltage.max(1e-6);
    current_amps * duration_hours
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ── Thruster ─────────────────────────────────────────────────────────

    #[test]
    fn thruster_initial_thrust_zero() {
        let t = Thruster::new(100.0, 0.1, 0.8);
        assert_eq!(t.thrust(), 0.0);
    }

    #[test]
    fn thruster_saturates_at_max() {
        let mut t = Thruster::new(100.0, 0.01, 0.8);
        t.set_command(9999.0);
        // Run many steps to converge
        for _ in 0..10000 {
            t.step(0.01);
        }
        assert!(t.thrust() <= 100.0 + EPS);
    }

    #[test]
    fn thruster_saturates_negative() {
        let mut t = Thruster::new(100.0, 0.01, 0.8);
        t.set_command(-9999.0);
        for _ in 0..10000 {
            t.step(0.01);
        }
        assert!(t.thrust() >= -100.0 - EPS);
    }

    #[test]
    fn thruster_step_moves_toward_command() {
        let mut t = Thruster::new(100.0, 0.1, 0.8);
        t.set_command(50.0);
        t.step(0.05); // half time-constant
        assert!(t.thrust() > 0.0);
        assert!(t.thrust() < 50.0);
    }

    #[test]
    fn thruster_converges_to_command() {
        let mut t = Thruster::new(100.0, 0.1, 0.8);
        t.set_command(40.0);
        for _ in 0..10000 {
            t.step(0.001);
        }
        assert!((t.thrust() - 40.0).abs() < 0.01);
    }

    #[test]
    fn thruster_zero_command_stays_zero() {
        let mut t = Thruster::new(100.0, 0.1, 0.8);
        t.set_command(0.0);
        for _ in 0..100 {
            t.step(0.01);
        }
        assert!(t.thrust().abs() < EPS);
    }

    // ── advance_ratio ────────────────────────────────────────────────────

    #[test]
    fn advance_ratio_zero_at_zero_speed() {
        assert_eq!(advance_ratio(0.0, 10.0, 0.3), 0.0);
    }

    #[test]
    fn advance_ratio_positive() {
        let j = advance_ratio(2.0, 10.0, 0.3);
        assert!(j > 0.0);
    }

    #[test]
    fn advance_ratio_zero_n() {
        // n = 0 → should return 0 without panic
        assert_eq!(advance_ratio(5.0, 0.0, 0.3), 0.0);
    }

    #[test]
    fn advance_ratio_formula() {
        // J = 3 / (10 * 0.3) = 1.0
        let j = advance_ratio(3.0, 10.0, 0.3);
        assert!((j - 1.0).abs() < EPS);
    }

    // ── kt_kq_polynomial ─────────────────────────────────────────────────

    #[test]
    fn kt_kq_at_j_zero() {
        let (kt, kq) = kt_kq_polynomial(0.0);
        assert!((kt - 0.3).abs() < EPS);
        assert!((kq - 0.04).abs() < EPS);
    }

    #[test]
    fn kt_kq_positive() {
        let (kt, kq) = kt_kq_polynomial(0.5);
        assert!(kt >= 0.0);
        assert!(kq >= 0.0);
    }

    #[test]
    fn kt_kq_decreases_with_j() {
        let (kt0, _) = kt_kq_polynomial(0.0);
        let (kt1, _) = kt_kq_polynomial(0.5);
        assert!(kt1 < kt0);
    }

    // ── propeller_thrust ─────────────────────────────────────────────────

    #[test]
    fn propeller_thrust_positive() {
        let t = propeller_thrust(0.3, 1000.0, 10.0, 0.3);
        assert!(t > 0.0);
    }

    #[test]
    fn propeller_thrust_scales_with_n_squared() {
        let t1 = propeller_thrust(0.3, 1000.0, 10.0, 0.3);
        let t2 = propeller_thrust(0.3, 1000.0, 20.0, 0.3);
        assert!((t2 / t1 - 4.0).abs() < 1e-6);
    }

    // ── propeller_efficiency ─────────────────────────────────────────────

    #[test]
    fn propeller_efficiency_in_zero_one() {
        let (kt, kq) = kt_kq_polynomial(0.5);
        let eta = propeller_efficiency(0.5, kt, kq);
        assert!((0.0..=1.0).contains(&eta));
    }

    #[test]
    fn propeller_efficiency_zero_at_j_zero() {
        let (kt, kq) = kt_kq_polynomial(0.0);
        let eta = propeller_efficiency(0.0, kt, kq);
        assert_eq!(eta, 0.0);
    }

    // ── ThrusterArray ────────────────────────────────────────────────────

    #[test]
    fn thruster_array_count() {
        let mut arr = ThrusterArray::new();
        arr.add_thruster(
            Thruster::new(50.0, 0.1, 0.8),
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        arr.add_thruster(
            Thruster::new(50.0, 0.1, 0.8),
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(arr.count(), 2);
    }

    #[test]
    fn thruster_array_total_zero_at_start() {
        let arr = ThrusterArray::new();
        let w = arr.total_force_torque();
        assert!(w.iter().all(|x| x.abs() < EPS));
    }

    #[test]
    fn thruster_array_total_force_after_step() {
        let mut arr = ThrusterArray::new();
        let mut thr = Thruster::new(50.0, 0.01, 0.8);
        thr.set_command(50.0);
        for _ in 0..1000 {
            thr.step(0.01);
        }
        arr.add_thruster(thr, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let w = arr.total_force_torque();
        assert!(w[0] > 40.0);
    }

    // ── thrust_allocation_pseudoinverse ──────────────────────────────────

    #[test]
    fn thrust_allocation_empty_returns_empty() {
        let result = thrust_allocation_pseudoinverse(&[], [0.0; 6]);
        assert!(result.is_empty());
    }

    #[test]
    fn thrust_allocation_single_thruster() {
        let b: &[[f64; 6]] = &[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let wrench = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = thrust_allocation_pseudoinverse(b, wrench);
        assert_eq!(u.len(), 1);
        assert!((u[0] - 10.0).abs() < EPS);
    }

    /// Wrench `[f; p × f]` of a thruster at `pos` pushing along the unit vector `dir`.
    fn wrench_column(pos: [f64; 3], dir: [f64; 3]) -> [f64; 6] {
        [
            dir[0],
            dir[1],
            dir[2],
            pos[1] * dir[2] - pos[2] * dir[1],
            pos[2] * dir[0] - pos[0] * dir[2],
            pos[0] * dir[1] - pos[1] * dir[0],
        ]
    }

    /// `B u` for a configuration given column by column.
    fn apply_configuration(columns: &[[f64; 6]], u: &[f64]) -> [f64; 6] {
        let mut wrench = [0.0; 6];
        for (column, thrust) in columns.iter().zip(u) {
            for (acc, entry) in wrench.iter_mut().zip(column) {
                *acc += entry * thrust;
            }
        }
        wrench
    }

    /// `Bᵀ r` — the least-squares optimality residual for `r = B u − w`.
    fn transpose_apply(columns: &[[f64; 6]], residual: &[f64; 6]) -> Vec<f64> {
        columns
            .iter()
            .map(|column| column.iter().zip(residual).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn thrust_allocation_parallel_thrusters_share_the_wrench() {
        // The per-row projection of the old implementation returned [2, 2]
        // (4 N in total); the minimum-norm solution is [1, 1].
        let b = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let u = thrust_allocation_pseudoinverse(&b, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.len(), 2);
        assert!((u[0] - 1.0).abs() < 1e-12, "u = {u:?}");
        assert!((u[1] - 1.0).abs() < 1e-12, "u = {u:?}");
    }

    #[test]
    fn thrust_allocation_unreachable_wrench_is_least_squares() {
        // Thruster 1 m to the side pushing along x: column [1,0,0, 0,0,-1].
        // Least squares for w = [10,0,0,0,0,0]: u = bᵀw / bᵀb = 10 / 2 = 5.
        let column = wrench_column([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
        let u = thrust_allocation_pseudoinverse(&[column], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.len(), 1);
        assert!((u[0] - 5.0).abs() < 1e-12, "u = {u:?}");

        // Thruster at the origin along (1,1,0)/√2: u = 10·(1/√2) / 1 = 7.0711.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let column = wrench_column([0.0; 3], [s, s, 0.0]);
        let u = thrust_allocation_pseudoinverse(&[column], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((u[0] - 10.0 * s).abs() < 1e-12, "u = {u:?}");

        // Same direction 1 m off-axis adds a yaw arm: column [s,s,0,0,0,-s],
        // u = 10·s / (3·s²) = 4.714.
        let column = wrench_column([0.0, 1.0, 0.0], [s, s, 0.0]);
        let u = thrust_allocation_pseudoinverse(&[column], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((u[0] - 10.0 * s / (3.0 * s * s)).abs() < 1e-12, "u = {u:?}");
    }

    /// Four vectored horizontal thrusters plus four vertical thrusters (rank 6).
    fn eight_thruster_layout() -> Vec<[f64; 6]> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        vec![
            wrench_column([0.4, 0.3, 0.0], [s, s, 0.0]),
            wrench_column([0.4, -0.3, 0.0], [s, -s, 0.0]),
            wrench_column([-0.4, 0.3, 0.0], [s, -s, 0.0]),
            wrench_column([-0.4, -0.3, 0.0], [s, s, 0.0]),
            wrench_column([0.3, 0.2, 0.0], [0.0, 0.0, 1.0]),
            wrench_column([0.3, -0.2, 0.0], [0.0, 0.0, 1.0]),
            wrench_column([-0.3, 0.2, 0.0], [0.0, 0.0, 1.0]),
            wrench_column([-0.3, -0.2, 0.0], [0.0, 0.0, 1.0]),
        ]
    }

    #[test]
    fn thrust_allocation_full_rank_layout_reproduces_wrench_with_minimum_norm() {
        let columns = eight_thruster_layout();
        let wrench = [12.0, -7.0, 30.0, 1.5, -2.5, 0.8];
        let u = thrust_allocation_pseudoinverse(&columns, wrench);
        assert_eq!(u.len(), 8);

        let produced = apply_configuration(&columns, &u);
        for (p, w) in produced.iter().zip(&wrench) {
            assert!((p - w).abs() < 1e-9, "B·u = {produced:?}, w = {wrench:?}");
        }

        // Independent reference: u = Bᵀ (B Bᵀ)⁻¹ w through the normal equations
        // (valid because this layout has full row rank).
        let b = DMatrix::<f64>::from_fn(6, 8, |r, c| columns[c][r]);
        let bbt_inverse = (&b * b.transpose())
            .try_inverse()
            .expect("full-rank layout has an invertible B·Bᵀ");
        let reference = b.transpose() * bbt_inverse * DVector::from_column_slice(&wrench);
        for (got, want) in u.iter().zip(reference.iter()) {
            assert!(
                (got - want).abs() < 1e-9,
                "u = {u:?}, reference = {reference:?}"
            );
        }
    }

    #[test]
    fn thrust_allocation_rank_deficient_layout_is_finite_and_optimal() {
        // Four horizontal thrusters 0.1 m below the origin: the pitch row is
        // −0.1 × the surge row, so B has rank 3 and B·Bᵀ is singular.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let columns = [
            wrench_column([0.4, 0.3, -0.1], [s, s, 0.0]),
            wrench_column([0.4, -0.3, -0.1], [s, -s, 0.0]),
            wrench_column([-0.4, 0.3, -0.1], [s, -s, 0.0]),
            wrench_column([-0.4, -0.3, -0.1], [s, s, 0.0]),
        ];

        // Reachable wrench: surge 10 N necessarily carries a −1 N·m pitch moment.
        let reachable = [10.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let u = thrust_allocation_pseudoinverse(&columns, reachable);
        assert!(u.iter().all(|x| x.is_finite()), "u = {u:?}");
        let produced = apply_configuration(&columns, &u);
        for (p, w) in produced.iter().zip(&reachable) {
            assert!((p - w).abs() < 1e-6, "B·u = {produced:?}");
        }

        // Pure surge cannot be produced exactly: the answer must be the
        // least-squares optimum, i.e. satisfy the normal equations Bᵀ(B u − w) = 0.
        let pure_surge = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = thrust_allocation_pseudoinverse(&columns, pure_surge);
        assert!(u.iter().all(|x| x.is_finite()), "u = {u:?}");
        let produced = apply_configuration(&columns, &u);
        let mut residual = [0.0; 6];
        for ((r, p), w) in residual.iter_mut().zip(&produced).zip(&pure_surge) {
            *r = p - w;
        }
        for g in transpose_apply(&columns, &residual) {
            assert!(g.abs() < 1e-9, "normal-equation residual {g}");
        }
        let miss: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(miss < 10.0, "residual {miss} must beat the zero command");
    }

    #[test]
    fn thrust_allocation_degenerate_inputs() {
        // All-zero configuration: nothing can be produced, nothing is commanded.
        let u = thrust_allocation_pseudoinverse(&[[0.0; 6]], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(u, vec![0.0]);

        // Non-finite input is reported as NaN, one entry per thruster.
        let b = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let u = thrust_allocation_pseudoinverse(&b, [f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.len(), 2);
        assert!(u.iter().all(|x| x.is_nan()));
        let mut bad = b;
        bad[1][2] = f64::INFINITY;
        let u = thrust_allocation_pseudoinverse(&bad, [1.0; 6]);
        assert!(u.iter().all(|x| x.is_nan()));
    }

    // ── BrushlessMotor ───────────────────────────────────────────────────

    #[test]
    fn motor_initial_state_zero() {
        let m = BrushlessMotor::new(1000.0, 0.1);
        assert_eq!(m.omega, 0.0);
        assert_eq!(m.current, 0.0);
    }

    #[test]
    fn motor_spins_up_under_voltage() {
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..1000 {
            m.step(12.0, 0.0, 0.001);
        }
        assert!(m.omega > 0.0);
    }

    #[test]
    fn motor_power_in_non_negative() {
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        m.step(12.0, 0.0, 0.001);
        assert!(m.power_in(12.0) >= 0.0);
    }

    #[test]
    fn motor_efficiency_in_zero_one() {
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..500 {
            m.step(12.0, 0.01, 0.001);
        }
        let eta = m.efficiency(12.0);
        assert!((0.0..=1.0).contains(&eta));
    }

    #[test]
    fn motor_power_in_ge_power_out() {
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..500 {
            m.step(12.0, 0.01, 0.001);
        }
        assert!(m.power_in(12.0) >= m.power_out() - EPS);
    }

    /// 1000 rpm/V expressed in rad/(s·V).
    const KV_1000_RPM_PER_VOLT: f64 = 1000.0 * 2.0 * PI / 60.0;

    #[test]
    fn motor_new_converts_kv_and_derives_consistent_constants() {
        let m = BrushlessMotor::new(1000.0, 0.1);
        assert!((m.kv - KV_1000_RPM_PER_VOLT).abs() < 1e-9);
        // Ke = Kt = 1 / Kv  [V·s/rad] = [N·m/A].
        assert!((m.back_emf_constant() * m.kv - 1.0).abs() < 1e-12);
        assert!((m.torque_constant() - m.back_emf_constant()).abs() < 1e-18);
        assert!((m.torque_constant() - 1.0 / KV_1000_RPM_PER_VOLT).abs() < 1e-15);
    }

    #[test]
    fn motor_back_emf_is_omega_divided_by_kv() {
        // 1000 rad/s on a 1000 rpm/V motor generates 9.55 V — the old model
        // multiplied by Kv and reported 104 720 V.
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        m.omega = 1000.0;
        let back_emf = m.back_emf_constant() * m.omega;
        assert!((back_emf - 1000.0 / KV_1000_RPM_PER_VOLT).abs() < 1e-9);
        assert!((back_emf - 9.549).abs() < 1e-3, "back-EMF = {back_emf}");

        // One step from this state: V = 12 V supplies (12 − 9.55) V across R, so
        // the current rises towards 24.5 A instead of being driven negative.
        m.step(12.0, 0.0, 1e-5);
        assert!(m.current > 0.0, "current = {}", m.current);
    }

    #[test]
    fn motor_unloaded_speed_reaches_kv_times_voltage() {
        // Kv = 1000 rpm/V, R = 0.1 Ω, 12 V, no load, 2 s at dt = 1e-5 s:
        // the rotor must settle at the no-load speed 12 · 104.72 = 1256.6 rad/s.
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..200_000 {
            m.step(12.0, 0.0, 1e-5);
        }
        let expected = 12.0 * KV_1000_RPM_PER_VOLT;
        assert!(
            (m.omega - expected).abs() / expected < 0.01,
            "omega = {}, expected {expected}",
            m.omega
        );
        // At the no-load speed the back-EMF equals the supply: no current flows.
        assert!(m.current.abs() < 1e-6, "current = {}", m.current);
    }

    #[test]
    fn motor_loaded_steady_state_matches_closed_form() {
        // Steady state: i = τ_load / Kt and ω = (V − R·i) / Ke.
        let (voltage, load) = (12.0, 0.05);
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..100_000 {
            m.step(voltage, load, 1e-5);
        }
        let current = load / m.torque_constant();
        let omega = (voltage - m.resistance * current) / m.back_emf_constant();
        assert!(
            (m.current - current).abs() / current < 1e-3,
            "current = {}, expected {current}",
            m.current
        );
        assert!(
            (m.omega - omega).abs() / omega < 1e-3,
            "omega = {}, expected {omega}",
            m.omega
        );
    }

    #[test]
    fn motor_power_balance_closes_at_steady_state() {
        // V·i = R·i² + Kt·i·ω holds only if the back-EMF and torque constants
        // are the same physical constant (Ke = Kt).
        let (voltage, load) = (12.0, 0.05);
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..100_000 {
            m.step(voltage, load, 1e-5);
        }
        let p_in = m.power_in(voltage);
        let p_out = m.power_out();
        let p_copper = m.resistance * m.current * m.current;
        assert!(
            (p_in - p_out - p_copper).abs() < 1e-3 * p_in,
            "P_in = {p_in}, P_out = {p_out}, P_cu = {p_copper}"
        );
        // Efficiency of a DC motor at steady state: Ke·ω / V.
        let expected = m.back_emf_constant() * m.omega / voltage;
        assert!((m.efficiency(voltage) - expected).abs() < 1e-3);
        assert!(
            expected > 0.9,
            "a healthy motor is highly efficient: {expected}"
        );
    }

    #[test]
    fn motor_freewheels_without_regenerative_braking() {
        let mut m = BrushlessMotor::new(1000.0, 0.1);
        for _ in 0..200_000 {
            m.step(12.0, 0.0, 1e-5);
        }
        let spinning = m.omega;
        assert!(spinning > 1000.0);

        // Supply cut: the current would have to reverse to brake the rotor, which
        // the motoring-quadrant model does not allow — the rotor keeps its speed.
        for _ in 0..10_000 {
            m.step(0.0, 0.0, 1e-5);
        }
        assert_eq!(m.current, 0.0);
        assert!((m.omega - spinning).abs() < 1e-9 * spinning);

        // A 1e-4 N·m load on the 1e-5 kg·m² rotor decelerates it at 10 rad/s²:
        // 0.1 s costs exactly 1 rad/s.
        for _ in 0..10_000 {
            m.step(0.0, 1e-4, 1e-5);
        }
        assert_eq!(m.current, 0.0);
        assert!(
            (spinning - m.omega - 1.0).abs() < 1e-6,
            "omega dropped by {}",
            spinning - m.omega
        );
    }

    // ── required_battery_capacity ────────────────────────────────────────

    #[test]
    fn battery_capacity_positive() {
        let cap = required_battery_capacity(100.0, 0.8, 1.0, 14.8);
        assert!(cap > 0.0);
    }

    #[test]
    fn battery_capacity_scales_with_duration() {
        let c1 = required_battery_capacity(100.0, 0.8, 1.0, 14.8);
        let c2 = required_battery_capacity(100.0, 0.8, 2.0, 14.8);
        assert!((c2 - 2.0 * c1).abs() < 1e-6);
    }

    #[test]
    fn battery_capacity_increases_with_thrust() {
        let c1 = required_battery_capacity(100.0, 0.8, 1.0, 14.8);
        let c2 = required_battery_capacity(200.0, 0.8, 1.0, 14.8);
        assert!(c2 > c1);
    }

    #[test]
    fn battery_capacity_decreases_with_efficiency() {
        let c1 = required_battery_capacity(100.0, 0.5, 1.0, 14.8);
        let c2 = required_battery_capacity(100.0, 0.9, 1.0, 14.8);
        assert!(c1 > c2);
    }
}
