//! Lumped interface structure: one damped oscillator per interface node,
//! integrated with the implicit Newmark method.
//!
//! The structure is an elastic (Winkler) foundation with a distributed mass: node
//! `k` carries the mass `m_k`, the viscous coefficient `c_k` and the stiffness
//! `k_k` of its tributary interface length and moves along its reference normal,
//!
//! m·a + c·v + k·u = F,
//!
//! where `F` is the normal component of the fluid force on the node.  The Newmark
//! update
//!
//! u⁺ = u + Δt·v + Δt²·[(½ − β)·a + β·a⁺],  v⁺ = v + Δt·[(1 − γ)·a + γ·a⁺]
//!
//! is implicit in the force, which is what makes the Dirichlet-Neumann coupling
//! iteration meaningful: the displacement predicted for the new time level
//! depends on the fluid force of the current iterate.  β = ¼, γ = ½ is the
//! undamped average-acceleration scheme; γ > ½ with β ≥ ¼(γ + ½)² adds
//! algorithmic damping of high frequencies.

/// Mass, damping and stiffness of one interface node (per unit depth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Oscillator {
    /// Lumped mass \[kg\] (per unit depth).
    pub(super) mass: f64,
    /// Viscous damping coefficient \[N·s/m\].
    pub(super) damping: f64,
    /// Spring stiffness \[N/m\].
    pub(super) stiffness: f64,
}

/// Displacement, velocity and acceleration of one structural degree of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ModalState {
    pub(super) displacement: f64,
    pub(super) velocity: f64,
    pub(super) acceleration: f64,
}

/// Newmark parameters and time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct NewmarkScheme {
    pub(super) beta: f64,
    pub(super) gamma: f64,
    pub(super) dt: f64,
}

impl Oscillator {
    /// Acceleration a = (F − c·v − k·u) / m of the dynamic equilibrium.
    pub(super) fn equilibrium_acceleration(
        &self,
        displacement: f64,
        velocity: f64,
        force: f64,
    ) -> f64 {
        (force - self.damping * velocity - self.stiffness * displacement) / self.mass
    }

    /// Advance the state over one Newmark step under the force `force` acting at
    /// the new time level.
    pub(super) fn advance(
        &self,
        now: &ModalState,
        force: f64,
        scheme: &NewmarkScheme,
    ) -> ModalState {
        let dt = scheme.dt;
        // Predictors that do not depend on the new acceleration.
        let u_star =
            now.displacement + dt * now.velocity + dt * dt * (0.5 - scheme.beta) * now.acceleration;
        let v_star = now.velocity + dt * (1.0 - scheme.gamma) * now.acceleration;
        let effective_mass =
            self.mass + scheme.gamma * dt * self.damping + scheme.beta * dt * dt * self.stiffness;
        let acceleration =
            (force - self.damping * v_star - self.stiffness * u_star) / effective_mass;
        ModalState {
            displacement: u_star + scheme.beta * dt * dt * acceleration,
            velocity: v_star + scheme.gamma * dt * acceleration,
            acceleration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn run(
        osc: &Oscillator,
        scheme: &NewmarkScheme,
        initial: ModalState,
        force: f64,
        steps: usize,
    ) -> ModalState {
        let mut state = initial;
        for _ in 0..steps {
            state = osc.advance(&state, force, scheme);
        }
        state
    }

    fn initial_state(osc: &Oscillator, u0: f64, v0: f64, force: f64) -> ModalState {
        ModalState {
            displacement: u0,
            velocity: v0,
            acceleration: osc.equilibrium_acceleration(u0, v0, force),
        }
    }

    #[test]
    fn equilibrium_acceleration_balances_the_forces() {
        let osc = Oscillator {
            mass: 2.0,
            damping: 3.0,
            stiffness: 50.0,
        };
        let a = osc.equilibrium_acceleration(0.1, 0.4, 10.0);
        assert!((osc.mass * a + osc.damping * 0.4 + osc.stiffness * 0.1 - 10.0).abs() < 1e-14);
    }

    #[test]
    fn undamped_average_acceleration_scheme_reproduces_the_period() {
        // ω = 100 rad/s; after one period the displacement returns to u0 and at
        // half a period it is −u0 (phase error O((ω·Δt)²)).
        let osc = Oscillator {
            mass: 1.0,
            damping: 0.0,
            stiffness: 1.0e4,
        };
        let period = 2.0 * PI / 100.0;
        let steps = 400;
        let scheme = NewmarkScheme {
            beta: 0.25,
            gamma: 0.5,
            dt: period / steps as f64,
        };
        let start = initial_state(&osc, 1e-3, 0.0, 0.0);
        let half = run(&osc, &scheme, start, 0.0, steps / 2);
        assert!((half.displacement + 1e-3).abs() < 2e-3 * 1e-3, "{half:?}");
        let full = run(&osc, &scheme, start, 0.0, steps);
        assert!((full.displacement - 1e-3).abs() < 2e-3 * 1e-3, "{full:?}");
        // The average-acceleration scheme conserves the energy exactly.
        let energy = |s: &ModalState| {
            0.5 * osc.mass * s.velocity * s.velocity
                + 0.5 * osc.stiffness * s.displacement * s.displacement
        };
        assert!((energy(&full) - energy(&start)).abs() < 1e-12 * energy(&start));
    }

    #[test]
    fn numerical_dissipation_damps_free_vibration() {
        let osc = Oscillator {
            mass: 1.0,
            damping: 0.0,
            stiffness: 1.0e4,
        };
        let steps = 200;
        let dt = 2.0 * PI / 100.0 / steps as f64;
        let gamma = 0.7;
        let scheme = NewmarkScheme {
            beta: 0.25 * (gamma + 0.5) * (gamma + 0.5),
            gamma,
            dt,
        };
        let start = initial_state(&osc, 1e-3, 0.0, 0.0);
        let end = run(&osc, &scheme, start, 0.0, 5 * steps);
        let amplitude = (end.displacement.powi(2) + (end.velocity / 100.0).powi(2)).sqrt();
        // Algorithmic damping ratio ≈ (γ − ½)·ω·Δt/2 gives ≈ 0.906 after five periods.
        assert!(
            amplitude < 0.93e-3 && amplitude > 0.85e-3,
            "dissipative Newmark must decay moderately, amplitude {amplitude}"
        );
    }

    #[test]
    fn viscous_damping_follows_the_analytic_decay() {
        // u(T_d) = u0·exp(−c T_d / 2m) after one damped period T_d.
        let (m, c, k) = (1.0, 20.0, 1.0e4);
        let osc = Oscillator {
            mass: m,
            damping: c,
            stiffness: k,
        };
        let omega_d = (k / m - (c / (2.0 * m)).powi(2)).sqrt();
        let period = 2.0 * PI / omega_d;
        let steps = 800;
        let scheme = NewmarkScheme {
            beta: 0.25,
            gamma: 0.5,
            dt: period / steps as f64,
        };
        let start = initial_state(&osc, 1e-3, 0.0, 0.0);
        let end = run(&osc, &scheme, start, 0.0, steps);
        let exact = 1e-3 * (-c * period / (2.0 * m)).exp();
        assert!(
            (end.displacement - exact).abs() < 2e-3 * 1e-3,
            "{} vs {exact}",
            end.displacement
        );
    }

    #[test]
    fn constant_force_converges_to_the_static_deflection() {
        let osc = Oscillator {
            mass: 1.0,
            damping: 2000.0,
            stiffness: 1.0e6,
        };
        let scheme = NewmarkScheme {
            beta: 0.25,
            gamma: 0.5,
            dt: 1e-3,
        };
        let force = -250.0;
        let start = initial_state(&osc, 0.0, 0.0, force);
        let end = run(&osc, &scheme, start, force, 60);
        assert!(
            (end.displacement - force / osc.stiffness).abs() < 1e-9 * (force / osc.stiffness).abs(),
            "{end:?}"
        );
        assert!(end.velocity.abs() < 1e-9);
    }
}
