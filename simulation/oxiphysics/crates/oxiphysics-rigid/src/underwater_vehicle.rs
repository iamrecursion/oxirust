// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Underwater vehicle dynamics for AUVs, ROVs, and submarines.
//!
//! Implements hydrodynamic forces (buoyancy, drag, added mass), hydrostatic
//! restoring forces and moments, thruster allocation, acoustic propagation, and
//! a simple depth PID controller.

use crate::thruster_dynamics::wrench_pseudo_inverse;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n < 1e-300 {
        [0.0; 3]
    } else {
        scale3(a, 1.0 / n)
    }
}

/// Water density (kg/m³) at ~20 °C.
const RHO_WATER: f64 = 1025.0;
/// Gravitational acceleration (m/s²).
const GRAVITY: f64 = 9.81;

// ---------------------------------------------------------------------------
// UnderwaterVehicle
// ---------------------------------------------------------------------------

/// Full 6-DOF underwater vehicle (AUV / ROV / submarine).
///
/// State vector layout (12 elements):
/// `[x, y, z, phi, theta, psi, u, v, w, p, q, r]`
/// where `(x,y,z)` is position (world frame, `+z` up), `(phi,theta,psi)` are
/// ZYX Euler angles, `(u,v,w)` are body-frame linear velocities, and `(p,q,r)`
/// are body-frame angular rates.
///
/// The dynamics are the decoupled (diagonal) approximation of Fossen's
/// equations of motion: every degree of freedom has its own effective inertia
/// (rigid-body mass or moment of inertia plus the added-mass coefficient), its
/// own quadratic drag, and the hydrostatic restoring wrench acts on top.
/// Coriolis/centripetal coupling and off-diagonal added mass are neglected, and
/// the moments of inertia are taken about the body origin.
pub struct UnderwaterVehicle {
    /// Dry mass of the vehicle in kg.
    pub mass: f64,
    /// Moments of inertia about the body axes `[Ixx, Iyy, Izz]` in kg·m².
    pub inertia: [f64; 3],
    /// Displaced volume in m³.
    pub volume: f64,
    /// Linear (surge/sway/heave) and rotational (roll/pitch/yaw) drag
    /// coefficients `[Du, Dv, Dw, Dp, Dq, Dr]`.
    pub drag_coeff: [f64; 6],
    /// Added-mass coefficients along the same 6 axes `[Xu_dot, ..., Nr_dot]`:
    /// kg for surge/sway/heave, kg·m² for roll/pitch/yaw.
    pub added_mass: [f64; 6],
    /// Centre of buoyancy in body frame (m).
    pub center_of_buoyancy: [f64; 3],
    /// Centre of gravity in body frame (m).
    pub center_of_gravity: [f64; 3],
}

impl UnderwaterVehicle {
    /// Create a new underwater vehicle with the given parameters.
    ///
    /// # Arguments
    /// * `mass`   – dry mass (kg).
    /// * `inertia` – moments of inertia `[Ixx, Iyy, Izz]` about the body axes (kg·m²).
    /// * `volume` – displaced volume (m³).
    /// * `drag_coeff`          – quadratic drag coefficients for 6 DOF.
    /// * `added_mass`          – added-mass coefficients for 6 DOF.
    /// * `center_of_buoyancy`  – CoB in body frame (m).
    /// * `center_of_gravity`   – CoG in body frame (m).
    pub fn new(
        mass: f64,
        inertia: [f64; 3],
        volume: f64,
        drag_coeff: [f64; 6],
        added_mass: [f64; 6],
        center_of_buoyancy: [f64; 3],
        center_of_gravity: [f64; 3],
    ) -> Self {
        Self {
            mass,
            inertia,
            volume,
            drag_coeff,
            added_mass,
            center_of_buoyancy,
            center_of_gravity,
        }
    }

    /// Compute the buoyancy force vector in the world frame `[Fx, Fy, Fz]`.
    ///
    /// The buoyancy force equals `ρ_water * g * V` acting upward (+Z in NED
    /// or +Y in ENU).  Here we use an ENU convention: upward = +Z.
    pub fn buoyancy_force(&self) -> [f64; 3] {
        let fb = RHO_WATER * GRAVITY * self.volume;
        [0.0, 0.0, fb]
    }

    /// Hydrostatic (weight + buoyancy) wrench for a pose `pos = [x,y,z,φ,θ,ψ]`.
    ///
    /// Returns the 6-element wrench `[Fx, Fy, Fz, Mx, My, Mz]` in the body
    /// frame, valid for finite roll and pitch (no small-angle approximation).
    ///
    /// Gravity and buoyancy are both vertical in the world frame. With
    /// `k = [-sin θ, sin φ cos θ, cos φ cos θ]` the world `+Z` direction
    /// expressed in the body frame (ZYX Euler angles), buoyancy `B = ρ g V`
    /// acts at the centre of buoyancy `r_b` as `B·k` and the weight `W = m g`
    /// at the centre of gravity `r_g` as `−W·k`, so
    ///
    /// * force: `F = (B − W) k`,
    /// * moment: `M = r_b × (B k) − r_g × (W k)`.
    ///
    /// For `B = W` the force vanishes and `M = (r_b − r_g) × B k`; a centre of
    /// buoyancy above the centre of gravity then yields a restoring moment.
    /// The position components `x, y, z` do not enter (uniform fluid).
    pub fn hydrostatic_restoring(&self, pos: &[f64; 6]) -> [f64; 6] {
        let (sin_phi, cos_phi) = pos[3].sin_cos();
        let (sin_theta, cos_theta) = pos[4].sin_cos();

        let weight = self.mass * GRAVITY;
        let buoyancy = RHO_WATER * GRAVITY * self.volume;

        // World +Z (up) in the body frame: third row of the body→world rotation.
        let up_body = [-sin_theta, sin_phi * cos_theta, cos_phi * cos_theta];

        let force = scale3(up_body, buoyancy - weight);
        let moment = add3(
            cross3(self.center_of_buoyancy, scale3(up_body, buoyancy)),
            cross3(self.center_of_gravity, scale3(up_body, -weight)),
        );

        [
            force[0], force[1], force[2], moment[0], moment[1], moment[2],
        ]
    }

    /// Quadratic drag force/moment for body-frame velocity `vel = [u,v,w,p,q,r]`.
    ///
    /// Uses `F_i = -D_i * |vel_i| * vel_i` for each axis independently.
    pub fn drag_force(&self, vel: &[f64; 6]) -> [f64; 6] {
        let mut f = [0.0f64; 6];
        for i in 0..6 {
            f[i] = -self.drag_coeff[i] * vel[i].abs() * vel[i];
        }
        f
    }

    /// Added-mass reaction force/moment for body-frame acceleration `accel`.
    ///
    /// Returns `F_i = -Ma_i * accel_i` for each axis.
    pub fn added_mass_force(&self, accel: &[f64; 6]) -> [f64; 6] {
        let mut f = [0.0f64; 6];
        for i in 0..6 {
            f[i] = -self.added_mass[i] * accel[i];
        }
        f
    }

    /// Metacentric height GM (m).
    ///
    /// `GM = (CoB_z - CoG_z)` in the body frame.  Positive GM indicates
    /// positive static stability.
    pub fn metacentric_height(&self) -> f64 {
        self.center_of_buoyancy[2] - self.center_of_gravity[2]
    }

    /// Returns `true` if the vehicle is statically stable (GM > 0).
    pub fn static_stability(&self) -> bool {
        self.metacentric_height() > 0.0
    }

    /// Integrate the vehicle state by one timestep `dt` using semi-implicit
    /// (symplectic) Euler integration: the body velocities are advanced first
    /// from the forces at the current state, then the pose is advanced with the
    /// *updated* velocities.
    ///
    /// The force on each degree of freedom is the sum of
    ///
    /// * the control wrench,
    /// * the quadratic drag ([`UnderwaterVehicle::drag_force`]),
    /// * the hydrostatic wrench of weight and buoyancy
    ///   ([`UnderwaterVehicle::hydrostatic_restoring`]),
    ///
    /// divided by the effective inertia of that degree of freedom: mass plus
    /// added mass for surge/sway/heave, moment of inertia plus added inertia for
    /// roll/pitch/yaw.
    ///
    /// # Arguments
    /// * `state`   – current 12-element state `[x,y,z,φ,θ,ψ, u,v,w,p,q,r]`.
    /// * `control` – 4-element control input `[Fx, Fy, Fz, Mz]` in body frame.
    /// * `dt`      – timestep (s).
    ///
    /// Returns the new 12-element state.
    pub fn step(&self, state: &[f64; 12], control: &[f64; 4], dt: f64) -> [f64; 12] {
        let [x, y, z, phi, theta, psi, u, v, w, p, q, r] = *state;

        // Wrench on the vehicle in the body frame.
        let vel6 = [u, v, w, p, q, r];
        let drag = self.drag_force(&vel6);
        let restoring = self.hydrostatic_restoring(&[x, y, z, phi, theta, psi]);
        let ctrl6 = [control[0], control[1], control[2], 0.0, 0.0, control[3]];

        // Effective inertia per degree of freedom (including added mass).
        let m_eff = [
            self.mass + self.added_mass[0],
            self.mass + self.added_mass[1],
            self.mass + self.added_mass[2],
            self.inertia[0] + self.added_mass[3],
            self.inertia[1] + self.added_mass[4],
            self.inertia[2] + self.added_mass[5],
        ];

        let mut accel = [0.0f64; 6];
        for i in 0..6 {
            accel[i] = (ctrl6[i] + drag[i] + restoring[i]) / m_eff[i].max(1e-9);
        }

        // Semi-implicit Euler, velocity half.
        let new_u = u + accel[0] * dt;
        let new_v = v + accel[1] * dt;
        let new_w = w + accel[2] * dt;
        let new_p = p + accel[3] * dt;
        let new_q = q + accel[4] * dt;
        let new_r = r + accel[5] * dt;

        // Rotation matrix (body → world) — ZYX Euler.
        let (sphi, cphi) = phi.sin_cos();
        let (sth, cth) = theta.sin_cos();
        let (spsi, cpsi) = psi.sin_cos();

        // Updated body velocities → world velocities.
        let xd = cth * cpsi * new_u
            + (sphi * sth * cpsi - cphi * spsi) * new_v
            + (cphi * sth * cpsi + sphi * spsi) * new_w;
        let yd = cth * spsi * new_u
            + (sphi * sth * spsi + cphi * cpsi) * new_v
            + (cphi * sth * spsi - sphi * cpsi) * new_w;
        let zd = -sth * new_u + sphi * cth * new_v + cphi * cth * new_w;

        // Euler angle rates from the updated body rates.
        let phid = new_p + (new_q * sphi + new_r * cphi) * sth / cth.max(1e-6);
        let thetad = new_q * cphi - new_r * sphi;
        let psid = (new_q * sphi + new_r * cphi) / cth.max(1e-6);

        // Semi-implicit Euler, pose half.
        [
            x + xd * dt,
            y + yd * dt,
            z + zd * dt,
            phi + phid * dt,
            theta + thetad * dt,
            psi + psid * dt,
            new_u,
            new_v,
            new_w,
            new_p,
            new_q,
            new_r,
        ]
    }
}

// ---------------------------------------------------------------------------
// Thruster
// ---------------------------------------------------------------------------

/// A single thruster unit on an underwater vehicle.
pub struct Thruster {
    /// Mounting position in body frame (m).
    pub position: [f64; 3],
    /// Unit thrust direction vector in body frame.
    pub direction: [f64; 3],
    /// Maximum achievable thrust force (N).
    pub max_thrust: f64,
    /// Efficiency coefficient (0–1).
    pub efficiency: f64,
}

impl Thruster {
    /// Create a new thruster.
    pub fn new(position: [f64; 3], direction: [f64; 3], max_thrust: f64, efficiency: f64) -> Self {
        Self {
            position,
            direction: normalize3(direction),
            max_thrust,
            efficiency,
        }
    }

    /// Compute the thrust vector (N) for a normalised throttle in `[-1, 1]`.
    pub fn thrust_vector(&self, throttle: f64) -> [f64; 3] {
        let clamped = throttle.clamp(-1.0, 1.0);
        scale3(self.direction, self.max_thrust * clamped)
    }

    /// Estimate electrical power consumption (W) for a given thrust (N).
    ///
    /// Uses `P = |T| * V_exit / η` where the exit velocity is derived from
    /// the disk-actuator approximation `T = 2ρA V_exit²`.  For simplicity we
    /// use a linear approximation `P ≈ |T|² / (2 * max_thrust * efficiency)`.
    pub fn power_consumption(&self, thrust: f64) -> f64 {
        if self.efficiency < 1e-9 || self.max_thrust < 1e-9 {
            return 0.0;
        }
        (thrust * thrust) / (2.0 * self.max_thrust * self.efficiency)
    }
}

// ---------------------------------------------------------------------------
// ThrusterAllocation
// ---------------------------------------------------------------------------

/// Manages a collection of thrusters and allocates desired wrenches.
pub struct ThrusterAllocation {
    /// All thrusters belonging to this vehicle.
    pub thrusters: Vec<Thruster>,
}

impl ThrusterAllocation {
    /// Create a new allocation manager with the given thrusters.
    pub fn new(thrusters: Vec<Thruster>) -> Self {
        Self { thrusters }
    }

    /// Build the 6×N thruster configuration matrix B.
    ///
    /// Each column `i` is `[f_i; τ_i]` where `f_i = direction_i` and
    /// `τ_i = position_i × direction_i`.
    fn config_matrix(&self) -> Vec<[f64; 6]> {
        self.thrusters
            .iter()
            .map(|t| {
                let f = t.direction;
                let tau = cross3(t.position, t.direction);
                [f[0], f[1], f[2], tau[0], tau[1], tau[2]]
            })
            .collect()
    }

    /// Pseudo-inverse thrust allocation for a desired wrench `w = [Fx,Fy,Fz,Mx,My,Mz]`.
    ///
    /// Computes the minimum-norm least-squares command `u = B⁺ w` with the
    /// Moore–Penrose pseudo-inverse of the 6×N configuration matrix, obtained
    /// from its singular value decomposition (see
    /// [`thrust_allocation_pseudoinverse`](crate::thruster_dynamics::thrust_allocation_pseudoinverse)).
    /// Unlike the normal equations `B^T (B B^T)^{-1} w`, this stays well defined
    /// when `B` has rank below 6, which is the rule rather than the exception for
    /// ROVs with fewer than six (or coplanar) thrusters: wrench components the
    /// thrusters cannot produce are simply left out instead of corrupting the
    /// command, and the achievable part is allocated in the least-squares sense.
    ///
    /// The result is clamped to each thruster's `[-max_thrust, +max_thrust]`.
    /// A non-finite wrench or configuration (including a `NaN` thrust limit)
    /// yields `NaN` commands.
    pub fn pseudo_inverse_allocation(&self, wrench: &[f64; 6]) -> Vec<f64> {
        if self.thrusters.is_empty() {
            return Vec::new();
        }

        let mut t_alloc = wrench_pseudo_inverse(&self.config_matrix(), wrench);

        // Clamp to thruster limits.
        for (t, thruster) in t_alloc.iter_mut().zip(&self.thrusters) {
            let limit = thruster.max_thrust.abs();
            *t = if limit.is_nan() {
                f64::NAN
            } else {
                t.clamp(-limit, limit)
            };
        }
        t_alloc
    }

    /// Allocate the desired wrench and return the resulting actual wrench.
    ///
    /// First computes per-thruster throttles via pseudo-inverse, then
    /// re-accumulates the actual wrench from the clamped throttle values.
    pub fn allocate_wrench(&self, wrench: &[f64; 6]) -> [f64; 6] {
        let throttles = self.pseudo_inverse_allocation(wrench);
        let b = self.config_matrix();
        let mut actual = [0.0f64; 6];
        for (i, &thr) in throttles.iter().enumerate() {
            for j in 0..6 {
                actual[j] += b[i][j] * thr;
            }
        }
        actual
    }
}

// ---------------------------------------------------------------------------
// AcousticPropagation
// ---------------------------------------------------------------------------

/// Simplified underwater acoustic propagation model.
pub struct AcousticPropagation {
    /// Speed of sound in water (m/s).  Typical value ≈ 1500 m/s.
    pub speed_of_sound: f64,
    /// Attenuation coefficient (dB/km).  Typical value ≈ 0.1 dB/km at 10 kHz.
    pub attenuation: f64,
}

impl AcousticPropagation {
    /// Create a new acoustic propagation model.
    pub fn new(speed_of_sound: f64, attenuation: f64) -> Self {
        Self {
            speed_of_sound,
            attenuation,
        }
    }

    /// One-way travel time (s) for a given distance (m).
    pub fn travel_time(&self, distance: f64) -> f64 {
        distance / self.speed_of_sound.max(1e-9)
    }

    /// Maximum sonar detection range (m) using a simplified SONAR equation.
    ///
    /// `SL - TL ≥ NL`  ⟹  `TL = SL - NL`
    ///
    /// Two-way transmission loss:
    /// `TL = 20 log₁₀(R) + α·R·1e-3`  where α is in dB/km.
    ///
    /// Solved iteratively (fixed-point iteration, max 100 iterations).
    ///
    /// # Arguments
    /// * `source_level` – source level SL (dB re 1 µPa @ 1 m).
    /// * `noise`        – ambient noise level NL (dB).
    pub fn sonar_range(&self, source_level: f64, noise: f64) -> f64 {
        let tl_max = source_level - noise;
        if tl_max <= 0.0 {
            return 0.0;
        }
        let alpha_per_m = self.attenuation * 1e-3 / 20.0 * std::f64::consts::LN_10.recip();
        // Binary search on range.
        let mut lo = 1.0f64;
        let mut hi = 1_000_000.0f64;
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let tl = 20.0 * mid.log10() + self.attenuation * mid * 1e-3;
            if tl < tl_max {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let _ = alpha_per_m; // used in derivation above
        0.5 * (lo + hi)
    }
}

// ---------------------------------------------------------------------------
// DepthController
// ---------------------------------------------------------------------------

/// Simple PID depth controller for an underwater vehicle.
pub struct DepthController {
    /// PID gains `[Kp, Ki, Kd]`.
    pub pid: [f64; 3],
    /// Accumulated integral of the depth error.
    pub integral: f64,
    /// Previous error for derivative term.
    prev_error: f64,
}

impl DepthController {
    /// Create a new depth controller with gains `[Kp, Ki, Kd]`.
    pub fn new(pid: [f64; 3]) -> Self {
        Self {
            pid,
            integral: 0.0,
            prev_error: 0.0,
        }
    }

    /// Update the controller with the current depth error and timestep.
    ///
    /// Returns the heave thrust command (N, positive = ascend).
    pub fn update(&mut self, depth_error: f64, dt: f64) -> f64 {
        self.integral += depth_error * dt;
        let derivative = if dt > 1e-15 {
            (depth_error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = depth_error;
        self.pid[0] * depth_error + self.pid[1] * self.integral + self.pid[2] * derivative
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_vehicle() -> UnderwaterVehicle {
        UnderwaterVehicle::new(
            100.0,                             // mass kg
            [4.0, 6.0, 8.0],                   // inertia kg·m²
            0.1,                               // volume m³  (≈ 102.5 kg buoyancy)
            [10.0, 10.0, 10.0, 5.0, 5.0, 5.0], // drag
            [5.0, 5.0, 5.0, 1.0, 1.0, 1.0],    // added mass
            [0.0, 0.0, 0.05],                  // CoB above CoG
            [0.0, 0.0, 0.0],                   // CoG at origin
        )
    }

    /// Neutrally buoyant version of [`default_vehicle`]: weight equals buoyancy.
    fn neutral_vehicle() -> UnderwaterVehicle {
        let mut vehicle = default_vehicle();
        vehicle.mass = RHO_WATER * vehicle.volume;
        vehicle
    }

    /// Net buoyancy `B − W` of a vehicle (N, positive = floats up).
    fn net_buoyancy(vehicle: &UnderwaterVehicle) -> f64 {
        (RHO_WATER * vehicle.volume - vehicle.mass) * GRAVITY
    }

    // -----------------------------------------------------------------------
    // UnderwaterVehicle tests
    // -----------------------------------------------------------------------

    #[test]
    fn buoyancy_force_upward() {
        let v = default_vehicle();
        let fb = v.buoyancy_force();
        assert!(fb[2] > 0.0, "buoyancy should be upward (+Z)");
        let expected = RHO_WATER * GRAVITY * v.volume;
        assert!((fb[2] - expected).abs() < 1e-6);
    }

    #[test]
    fn drag_force_opposes_motion() {
        let v = default_vehicle();
        let vel = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let drag = v.drag_force(&vel);
        assert!(drag[0] < 0.0, "drag should oppose surge velocity");
    }

    #[test]
    fn drag_force_zero_for_zero_velocity() {
        let v = default_vehicle();
        let vel = [0.0; 6];
        let drag = v.drag_force(&vel);
        for (i, &d) in drag.iter().enumerate() {
            assert!(d.abs() < 1e-15, "drag[{i}] should be zero at rest");
        }
    }

    #[test]
    fn added_mass_force_proportional_to_accel() {
        let v = default_vehicle();
        let a1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a2 = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let f1 = v.added_mass_force(&a1);
        let f2 = v.added_mass_force(&a2);
        assert!((f2[0] - 2.0 * f1[0]).abs() < 1e-10);
    }

    #[test]
    fn metacentric_height_positive_stable() {
        let v = default_vehicle();
        assert!(v.metacentric_height() > 0.0);
        assert!(v.static_stability());
    }

    #[test]
    fn metacentric_height_negative_unstable() {
        let mut v = default_vehicle();
        v.center_of_buoyancy[2] = -0.1; // CoB below CoG
        assert!(v.metacentric_height() < 0.0);
        assert!(!v.static_stability());
    }

    #[test]
    fn step_advances_position() {
        let v = default_vehicle();
        let mut state = [0.0f64; 12];
        state[6] = 1.0; // surge velocity
        let control = [0.0; 4];
        let new_state = v.step(&state, &control, 0.1);
        assert!(new_state[0] > 0.0, "vehicle should move forward");
    }

    #[test]
    fn step_drag_slows_down_vehicle() {
        let v = default_vehicle();
        let mut state = [0.0f64; 12];
        state[6] = 10.0; // high surge velocity
        let control = [0.0; 4];
        let new_state = v.step(&state, &control, 0.1);
        assert!(new_state[6] < 10.0, "drag should reduce velocity");
    }

    #[test]
    fn hydrostatic_restoring_upright_has_fz() {
        let v = default_vehicle();
        let pos = [0.0; 6];
        let f = v.hydrostatic_restoring(&pos);
        // Net heave: buoyancy - weight
        let expected_fz = RHO_WATER * GRAVITY * v.volume - v.mass * GRAVITY;
        assert!((f[2] - expected_fz).abs() < 1e-6);
    }

    #[test]
    fn hydrostatic_restoring_nonzero_angle() {
        let v = default_vehicle();
        let phi = 0.1; // small roll
        let f = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, phi, 0.0, 0.0]);
        // CoB 5 cm above the CoG: M = r_b × (B k) − r_g × (W k) with r_g = 0, so
        // Mx = −0.05 · B · sin φ — negative for a positive roll (restoring).
        let buoyancy = RHO_WATER * GRAVITY * v.volume;
        let expected_mx = -0.05 * buoyancy * phi.sin();
        assert!((f[3] - expected_mx).abs() < 1e-9, "Mx = {}", f[3]);
        assert!(f[3] < 0.0, "roll moment must oppose the roll");
        assert!(f[4].abs() < 1e-9 && f[5].abs() < 1e-9);
    }

    #[test]
    fn hydrostatic_restoring_rotates_net_force_into_body_frame() {
        // Net buoyancy B − W acts along world +Z; in the body frame it is
        // (B − W)·[−sin θ, sin φ cos θ, cos φ cos θ].
        let v = default_vehicle();
        let net = net_buoyancy(&v);
        assert!(
            net > 20.0,
            "test needs a clearly positive net buoyancy: {net}"
        );

        let theta = 30.0_f64.to_radians();
        let f = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, 0.0, theta, 0.0]);
        assert!((f[0] + net * 0.5).abs() < 1e-9, "Fx = {}", f[0]);
        assert!(f[1].abs() < 1e-9, "Fy = {}", f[1]);
        assert!((f[2] - net * theta.cos()).abs() < 1e-9, "Fz = {}", f[2]);

        let phi = 20.0_f64.to_radians();
        let f = v.hydrostatic_restoring(&[1.0, 2.0, -3.0, phi, 0.0, 0.7]);
        assert!(f[0].abs() < 1e-9, "Fx = {}", f[0]);
        assert!((f[1] - net * phi.sin()).abs() < 1e-9, "Fy = {}", f[1]);
        assert!((f[2] - net * phi.cos()).abs() < 1e-9, "Fz = {}", f[2]);

        // The force is a rotation of (0,0,B−W), so its magnitude never changes.
        for &(phi, theta) in &[(0.4, -0.7), (-1.1, 0.3), (2.0, 0.9)] {
            let f = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, phi, theta, 0.0]);
            let magnitude = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            assert!((magnitude - net.abs()).abs() < 1e-9, "|F| = {magnitude}");
        }
    }

    #[test]
    fn hydrostatic_moment_accounts_for_weight_acting_at_the_cog() {
        // M = r_b × (B k) − r_g × (W k). With the CoG 10 cm ahead of the origin
        // and W ≠ B the weight term is not the same as r_bg × (B k).
        let mut v = default_vehicle();
        v.center_of_gravity = [0.1, 0.0, -0.02];
        let weight = v.mass * GRAVITY;
        let m = v.hydrostatic_restoring(&[0.0; 6]);
        assert!(m[3].abs() < 1e-9, "Mx = {}", m[3]);
        assert!((m[4] - 0.1 * weight).abs() < 1e-9, "My = {}", m[4]);
        assert!(m[5].abs() < 1e-9, "Mz = {}", m[5]);

        // Tilted: compare with the explicit cross products.
        let (phi, theta) = (0.35_f64, -0.2_f64);
        let k = [
            -theta.sin(),
            phi.sin() * theta.cos(),
            phi.cos() * theta.cos(),
        ];
        let buoyancy = RHO_WATER * GRAVITY * v.volume;
        let b_force = scale3(k, buoyancy);
        let w_force = scale3(k, -weight);
        let expected = add3(
            cross3(v.center_of_buoyancy, b_force),
            cross3(v.center_of_gravity, w_force),
        );
        let got = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, phi, theta, 0.0]);
        for i in 0..3 {
            assert!(
                (got[3 + i] - expected[i]).abs() < 1e-9,
                "M[{i}] = {} vs {}",
                got[3 + i],
                expected[i]
            );
        }
    }

    #[test]
    fn hydrostatic_roll_moment_flips_sign_when_cog_is_above_cob() {
        let mut v = neutral_vehicle();
        let phi = 0.2;
        let stable = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, phi, 0.0, 0.0]);
        assert!(stable[3] < 0.0, "CoG below CoB must restore: {}", stable[3]);
        v.center_of_buoyancy[2] = -0.05; // CoB below CoG
        let unstable = v.hydrostatic_restoring(&[0.0, 0.0, 0.0, phi, 0.0, 0.0]);
        assert!(
            unstable[3] > 0.0,
            "CoG above CoB must capsize: {}",
            unstable[3]
        );
        assert!((stable[3] + unstable[3]).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Dynamics with hydrostatics and rotational inertia
    // -----------------------------------------------------------------------

    #[test]
    fn neutral_vehicle_holds_depth_and_attitude() {
        let v = neutral_vehicle();
        let mut state = [0.0f64; 12];
        for _ in 0..1000 {
            state = v.step(&state, &[0.0; 4], 0.01);
        }
        for (i, s) in state.iter().enumerate() {
            assert!(s.abs() < 1e-9, "state[{i}] = {s} drifted");
        }
    }

    #[test]
    fn heavy_vehicle_sinks_and_light_vehicle_rises() {
        let dt = 0.01;
        let mut heavy = neutral_vehicle();
        heavy.mass *= 1.2;
        let mut light = neutral_vehicle();
        light.mass *= 0.8;

        // First step from rest: no drag yet, so w = (B − W) / (m + A_w) · dt.
        for vehicle in [&heavy, &light] {
            let first = vehicle.step(&[0.0; 12], &[0.0; 4], dt);
            let expected_w = net_buoyancy(vehicle) / (vehicle.mass + vehicle.added_mass[2]) * dt;
            assert!(
                (first[8] - expected_w).abs() < 1e-12,
                "w = {}, expected {expected_w}",
                first[8]
            );
        }

        let mut sinking = [0.0f64; 12];
        let mut rising = [0.0f64; 12];
        for _ in 0..200 {
            sinking = heavy.step(&sinking, &[0.0; 4], dt);
            rising = light.step(&rising, &[0.0; 4], dt);
        }
        assert!(
            sinking[2] < -0.5 && sinking[8] < 0.0,
            "heavy state {sinking:?}"
        );
        assert!(rising[2] > 0.5 && rising[8] > 0.0, "light state {rising:?}");
    }

    #[test]
    fn rotational_acceleration_uses_inertia_not_mass() {
        // Yaw moment 12 N·m on Izz = 8 kg·m², added inertia 1: r = 12/9·dt,
        // not 12/(m + 1) as when the mass was used as a moment of inertia.
        let v = neutral_vehicle();
        let dt = 0.01;
        let yawed = v.step(&[0.0; 12], &[0.0, 0.0, 0.0, 12.0], dt);
        let expected_r = 12.0 / (v.inertia[2] + v.added_mass[5]) * dt;
        assert!((yawed[11] - expected_r).abs() < 1e-12, "r = {}", yawed[11]);

        // Surge force 50 N on m + A_u: u = 50/(m + 5)·dt, and the pose already
        // sees the updated velocity (semi-implicit Euler): x = u·dt.
        let surged = v.step(&[0.0; 12], &[50.0, 0.0, 0.0, 0.0], dt);
        let expected_u = 50.0 / (v.mass + v.added_mass[0]) * dt;
        assert!((surged[6] - expected_u).abs() < 1e-12, "u = {}", surged[6]);
        assert!(
            (surged[0] - expected_u * dt).abs() < 1e-15,
            "x = {}",
            surged[0]
        );
    }

    #[test]
    fn roll_oscillation_period_is_set_by_inertia_and_metacentric_moment() {
        // Undamped small roll about Ixx: ω² = B·GM / Ixx with GM = 5 cm, so the
        // quarter period 2π/(4ω) is when φ first crosses zero. With the mass
        // (102.5 kg) used as the inertia the period would be about 5x longer.
        let mut v = neutral_vehicle();
        v.drag_coeff = [0.0; 6];
        v.added_mass = [0.0; 6];
        let buoyancy = RHO_WATER * GRAVITY * v.volume;
        let omega = (buoyancy * 0.05 / v.inertia[0]).sqrt();
        let quarter_period = std::f64::consts::FRAC_PI_2 / omega;

        let dt = 1e-4;
        let mut state = [0.0f64; 12];
        state[3] = 0.05;
        let mut t = 0.0;
        while state[3] > 0.0 && t < 3.0 * quarter_period {
            state = v.step(&state, &[0.0; 4], dt);
            t += dt;
        }
        assert!(
            (t - quarter_period).abs() < 0.01 * quarter_period,
            "zero crossing at {t} s, expected {quarter_period} s"
        );
    }

    #[test]
    fn roll_with_cog_below_cob_is_restored() {
        let v = neutral_vehicle();
        let phi0 = 0.3;
        let dt = 1e-3;

        // First step: p = Mx / (Ixx + A_p)·dt with Mx = −0.05·B·sin φ.
        let mut state = [0.0f64; 12];
        state[3] = phi0;
        let first = v.step(&state, &[0.0; 4], dt);
        let buoyancy = RHO_WATER * GRAVITY * v.volume;
        let expected_p = -0.05 * buoyancy * phi0.sin() / (v.inertia[0] + v.added_mass[3]) * dt;
        assert!((first[9] - expected_p).abs() < 1e-12, "p = {}", first[9]);
        assert!(first[3] < phi0, "roll angle must start to decrease");

        // Quadratic drag then damps the swing: it never exceeds the release
        // angle and the vehicle ends up (nearly) upright.
        let mut peak_after_release = 0.0f64;
        for step in 0..20_000 {
            state = v.step(&state, &[0.0; 4], dt);
            if step > 500 {
                peak_after_release = peak_after_release.max(state[3].abs());
            }
        }
        assert!(
            peak_after_release <= phi0 + 1e-3,
            "roll grew to {peak_after_release}"
        );
        assert!(state[3].abs() < 0.1, "final roll {}", state[3]);
    }

    #[test]
    fn roll_with_cog_above_cob_capsizes() {
        let mut v = neutral_vehicle();
        v.center_of_buoyancy[2] = -0.05;
        let mut state = [0.0f64; 12];
        state[3] = 0.05;
        for _ in 0..500 {
            state = v.step(&state, &[0.0; 4], 1e-3);
        }
        assert!(
            state[3] > 0.1,
            "an unstable vehicle must roll away: {}",
            state[3]
        );
    }

    // -----------------------------------------------------------------------
    // Thruster tests
    // -----------------------------------------------------------------------

    #[test]
    fn thruster_zero_throttle_zero_force() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        let f = t.thrust_vector(0.0);
        assert!(norm3(f) < 1e-12);
    }

    #[test]
    fn thruster_full_throttle_max_force() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        let f = t.thrust_vector(1.0);
        assert!((norm3(f) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn thruster_negative_throttle_reverses_force() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 50.0, 0.8);
        let f = t.thrust_vector(-1.0);
        assert!(f[0] < 0.0);
    }

    #[test]
    fn thruster_power_zero_for_zero_thrust() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        assert!((t.power_consumption(0.0)).abs() < 1e-15);
    }

    #[test]
    fn thruster_power_positive_for_nonzero_thrust() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        assert!(t.power_consumption(50.0) > 0.0);
    }

    // -----------------------------------------------------------------------
    // ThrusterAllocation tests
    // -----------------------------------------------------------------------

    #[test]
    fn allocation_single_thruster_surge() {
        let t = Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        let alloc = ThrusterAllocation::new(vec![t]);
        let wrench = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let thrusts = alloc.pseudo_inverse_allocation(&wrench);
        assert_eq!(thrusts.len(), 1);
        // Thruster should produce positive surge force.
        assert!(thrusts[0] > 0.0);
    }

    #[test]
    fn allocation_empty_thrusters() {
        let alloc = ThrusterAllocation::new(vec![]);
        let wrench = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let thrusts = alloc.pseudo_inverse_allocation(&wrench);
        assert!(thrusts.is_empty());
    }

    #[test]
    fn allocation_wrench_bounded_by_max_thrust() {
        let thrusters = vec![
            Thruster::new([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 50.0, 0.9),
            Thruster::new([-0.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 50.0, 0.9),
        ];
        let alloc = ThrusterAllocation::new(thrusters);
        let wrench = [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // saturated
        let thrusts = alloc.pseudo_inverse_allocation(&wrench);
        for (i, &t) in thrusts.iter().enumerate() {
            assert!(t <= 50.0, "thrust[{i}] = {t} exceeds max");
            assert!(t >= -50.0, "thrust[{i}] = {t} below min");
        }
    }

    #[test]
    fn allocate_wrench_returns_6_elements() {
        let thrusters = vec![
            Thruster::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8),
            Thruster::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 100.0, 0.8),
        ];
        let alloc = ThrusterAllocation::new(thrusters);
        let wrench = [5.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let actual = alloc.allocate_wrench(&wrench);
        assert_eq!(actual.len(), 6);
    }

    /// `B u` accumulated from the thruster wrench columns.
    fn produced_wrench(alloc: &ThrusterAllocation, u: &[f64]) -> [f64; 6] {
        let mut wrench = [0.0; 6];
        for (column, thrust) in alloc.config_matrix().iter().zip(u) {
            for (acc, entry) in wrench.iter_mut().zip(column) {
                *acc += entry * thrust;
            }
        }
        wrench
    }

    #[test]
    fn allocation_single_off_axis_thruster_is_least_squares() {
        // Thruster 1 m to the side pushing along x has the wrench column
        // b = [1,0,0, 0,0,-1]; least squares for w = [10,0,…] is bᵀw/bᵀb = 5.
        // (The old normal-equation solve returned 10.)
        let t = Thruster::new([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 100.0, 0.8);
        let alloc = ThrusterAllocation::new(vec![t]);
        let u = alloc.pseudo_inverse_allocation(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.len(), 1);
        assert!((u[0] - 5.0).abs() < 1e-10, "u = {u:?}");
    }

    #[test]
    fn allocation_single_diagonal_thruster_is_least_squares() {
        // Thruster at the origin along (1,1,0)/√2 (s = 1/√2) has no moment arm:
        // b = [s,s,0,0,0,0], bᵀb = 1 and bᵀw = 10·s, so u = 7.0711
        // (the old solve returned 14.14).
        let t = Thruster::new([0.0; 3], [1.0, 1.0, 0.0], 100.0, 0.8);
        let alloc = ThrusterAllocation::new(vec![t]);
        let u = alloc.pseudo_inverse_allocation(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((u[0] - 10.0 / 2.0_f64.sqrt()).abs() < 1e-10, "u = {u:?}");

        // Moving the thruster 1 m off-axis adds the yaw arm: b = [s,s,0,0,0,-s],
        // |b|² = 3s² = 1.5, so u = 10·s / 1.5 = 4.714.
        let t = Thruster::new([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 100.0, 0.8);
        let alloc = ThrusterAllocation::new(vec![t]);
        let u = alloc.pseudo_inverse_allocation(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((u[0] - 10.0 * s / 1.5).abs() < 1e-10, "u = {u:?}");
    }

    /// Four vectored horizontal thrusters `z` metres above the origin.
    fn horizontal_rov(z: f64) -> ThrusterAllocation {
        ThrusterAllocation::new(vec![
            Thruster::new([0.4, 0.3, z], [1.0, 1.0, 0.0], 80.0, 0.8),
            Thruster::new([0.4, -0.3, z], [1.0, -1.0, 0.0], 80.0, 0.8),
            Thruster::new([-0.4, 0.3, z], [1.0, -1.0, 0.0], 80.0, 0.8),
            Thruster::new([-0.4, -0.3, z], [1.0, 1.0, 0.0], 80.0, 0.8),
        ])
    }

    #[test]
    fn allocation_planar_rov_reproduces_surge_sway_and_yaw() {
        // Thrusters in the z = 0 plane: B has rank 3 (Fx, Fy, Mz) and B·Bᵀ is
        // structurally singular, yet every planar wrench must be reproduced.
        let alloc = horizontal_rov(0.0);
        for wrench in [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            [8.0, -4.0, 0.0, 0.0, 0.0, 1.5],
        ] {
            let u = alloc.pseudo_inverse_allocation(&wrench);
            assert!(u.iter().all(|x| x.is_finite()), "u = {u:?}");
            let produced = produced_wrench(&alloc, &u);
            for (p, w) in produced.iter().zip(&wrench) {
                assert!((p - w).abs() < 1e-9, "B·u = {produced:?}, w = {wrench:?}");
            }
            assert_eq!(alloc.allocate_wrench(&wrench).len(), 6);
        }
    }

    #[test]
    fn allocation_thrusters_below_origin_couple_surge_into_pitch() {
        // At z = −0.1 the pitch row is −0.1 × the surge row, so rank(B) = 3
        // without any structurally zero row (no pivot test can spot it).
        let alloc = horizontal_rov(-0.1);

        // 10 N of surge necessarily carries −1 N·m of pitch: reachable exactly.
        let reachable = [10.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let u = alloc.pseudo_inverse_allocation(&reachable);
        assert!(u.iter().all(|x| x.is_finite()), "u = {u:?}");
        let produced = produced_wrench(&alloc, &u);
        for (p, w) in produced.iter().zip(&reachable) {
            assert!((p - w).abs() < 1e-6, "B·u = {produced:?}");
        }

        // Pure surge is not reachable: the command must still be finite and
        // satisfy the least-squares optimality condition Bᵀ(B u − w) = 0.
        let pure_surge = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = alloc.pseudo_inverse_allocation(&pure_surge);
        assert!(u.iter().all(|x| x.is_finite()), "u = {u:?}");
        let produced = produced_wrench(&alloc, &u);
        let mut residual = [0.0; 6];
        for ((r, p), w) in residual.iter_mut().zip(&produced).zip(&pure_surge) {
            *r = p - w;
        }
        for column in alloc.config_matrix() {
            let gradient: f64 = column.iter().zip(&residual).map(|(a, b)| a * b).sum();
            assert!(gradient.abs() < 1e-9, "normal-equation residual {gradient}");
        }
        let miss: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(miss < 10.0, "residual {miss} must beat the zero command");
    }

    #[test]
    fn allocate_wrench_matches_request_for_full_rank_layout_until_saturation() {
        // Four vertical + four vectored horizontal thrusters: rank 6.
        let mut thrusters = Vec::new();
        for &(x, y) in &[(0.3, 0.2), (0.3, -0.2), (-0.3, 0.2), (-0.3, -0.2)] {
            thrusters.push(Thruster::new([x, y, 0.0], [0.0, 0.0, 1.0], 60.0, 0.8));
        }
        for &(x, y, dx, dy) in &[
            (0.4, 0.3, 1.0, 1.0),
            (0.4, -0.3, 1.0, -1.0),
            (-0.4, 0.3, 1.0, -1.0),
            (-0.4, -0.3, 1.0, 1.0),
        ] {
            thrusters.push(Thruster::new([x, y, 0.0], [dx, dy, 0.0], 60.0, 0.8));
        }
        let alloc = ThrusterAllocation::new(thrusters);

        let wrench = [12.0, -7.0, 30.0, 1.5, -2.5, 0.8];
        let actual = alloc.allocate_wrench(&wrench);
        for (a, w) in actual.iter().zip(&wrench) {
            assert!((a - w).abs() < 1e-9, "actual = {actual:?}");
        }

        // Far beyond the thrust limits the clamped command saturates and the
        // delivered wrench falls short of the request.
        let huge = [5000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delivered = alloc.allocate_wrench(&huge);
        assert!(
            delivered[0] > 0.0 && delivered[0] < huge[0],
            "{delivered:?}"
        );
    }

    #[test]
    fn allocation_reports_nan_for_non_finite_wrench() {
        let alloc = horizontal_rov(0.0);
        let u = alloc.pseudo_inverse_allocation(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.len(), 4);
        assert!(u.iter().all(|x| x.is_nan()), "u = {u:?}");
    }

    // -----------------------------------------------------------------------
    // AcousticPropagation tests
    // -----------------------------------------------------------------------

    #[test]
    fn travel_time_proportional_to_distance() {
        let ap = AcousticPropagation::new(1500.0, 0.1);
        let t1 = ap.travel_time(1500.0);
        let t2 = ap.travel_time(3000.0);
        assert!((t1 - 1.0).abs() < 1e-10);
        assert!((t2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn travel_time_zero_distance() {
        let ap = AcousticPropagation::new(1500.0, 0.1);
        assert!((ap.travel_time(0.0)).abs() < 1e-15);
    }

    #[test]
    fn sonar_range_positive_for_strong_source() {
        let ap = AcousticPropagation::new(1500.0, 0.1);
        let r = ap.sonar_range(200.0, 50.0);
        assert!(r > 0.0, "sonar range should be positive: {r}");
    }

    #[test]
    fn sonar_range_zero_when_noise_exceeds_source() {
        let ap = AcousticPropagation::new(1500.0, 0.1);
        let r = ap.sonar_range(50.0, 100.0);
        assert_eq!(r, 0.0, "sonar range should be zero when noise > SL");
    }

    #[test]
    fn sonar_range_increases_with_source_level() {
        let ap = AcousticPropagation::new(1500.0, 0.1);
        let r1 = ap.sonar_range(150.0, 50.0);
        let r2 = ap.sonar_range(200.0, 50.0);
        assert!(r2 > r1, "higher SL should give longer range");
    }

    // -----------------------------------------------------------------------
    // DepthController tests
    // -----------------------------------------------------------------------

    #[test]
    fn depth_controller_zero_error_zero_output() {
        let mut ctrl = DepthController::new([1.0, 0.1, 0.5]);
        let out = ctrl.update(0.0, 0.1);
        assert!(out.abs() < 1e-15);
    }

    #[test]
    fn depth_controller_positive_error_positive_output() {
        let mut ctrl = DepthController::new([1.0, 0.0, 0.0]);
        let out = ctrl.update(5.0, 0.1);
        assert!(out > 0.0);
    }

    #[test]
    fn depth_controller_integral_accumulates() {
        let mut ctrl = DepthController::new([0.0, 1.0, 0.0]);
        ctrl.update(1.0, 0.1);
        ctrl.update(1.0, 0.1);
        assert!(ctrl.integral > 0.0);
    }

    #[test]
    fn depth_controller_derivative_term() {
        let mut ctrl = DepthController::new([0.0, 0.0, 1.0]);
        ctrl.update(0.0, 0.1); // prime derivative
        let out = ctrl.update(1.0, 0.1);
        // Derivative of (1 - 0) / 0.1 = 10; output = Kd * 10 = 10
        assert!((out - 10.0).abs() < 1e-6, "derivative output = {out}");
    }

    #[test]
    fn depth_controller_reset_clears_state() {
        let mut ctrl = DepthController::new([1.0, 1.0, 1.0]);
        ctrl.update(5.0, 0.1);
        ctrl.reset();
        assert!(ctrl.integral.abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // Physics / helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn cross3_orthogonality() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross3(a, b);
        assert!((c[0]).abs() < 1e-15);
        assert!((c[1]).abs() < 1e-15);
        assert!((c[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn normalize3_unit_vector() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize3(v);
        assert!((norm3(n) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize3_zero_vector() {
        let v = [0.0, 0.0, 0.0];
        let n = normalize3(v);
        assert!(norm3(n) < 1e-15);
    }

    #[test]
    fn buoyancy_scales_with_volume() {
        let v1 =
            UnderwaterVehicle::new(100.0, [1.0; 3], 0.1, [0.0; 6], [0.0; 6], [0.0; 3], [0.0; 3]);
        let v2 =
            UnderwaterVehicle::new(100.0, [1.0; 3], 0.2, [0.0; 6], [0.0; 6], [0.0; 3], [0.0; 3]);
        let fb1 = v1.buoyancy_force();
        let fb2 = v2.buoyancy_force();
        assert!((fb2[2] - 2.0 * fb1[2]).abs() < 1e-6);
    }

    #[test]
    fn acoustic_propagation_custom_speed() {
        let ap = AcousticPropagation::new(1000.0, 0.0);
        assert!((ap.travel_time(2000.0) - 2.0).abs() < 1e-10);
    }
}
