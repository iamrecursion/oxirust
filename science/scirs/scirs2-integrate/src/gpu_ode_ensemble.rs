//! Batched ODE ensemble integration across parameter sets.
//!
//! This module provides a mechanism to solve many ODE initial-value problems
//! simultaneously, each with its own set of parameters and initial conditions
//! but sharing the same right-hand-side structure.  This pattern arises in
//! parameter sweeps, uncertainty quantification, and neural-ODE training.
//!
//! ## Algorithm
//!
//! Each ensemble member is integrated independently using the Dormand-Prince
//! adaptive RK45 scheme (the same pair used by `scipy.integrate.solve_ivp`
//! with `method='RK45'`).  Step-size control follows the standard PI-controller
//! formula:
//!
//! ```text
//! h_new = h * min(facmax, max(facmin, fac * (1/err)^(1/5)))
//! ```
//!
//! with `fac = 0.9`, `facmax = 10.0`, `facmin = 0.2`.
//!
//! ## Dispatch
//!
//! [`EnsembleDispatch::Sequential`] processes members one at a time on the CPU.
//! [`EnsembleDispatch::Simulated`] represents a conceptual GPU batched dispatch
//! (same numerics, different conceptual path) and is provided for API
//! compatibility with future hardware acceleration.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::gpu_ode_ensemble::{
//!     OdeEnsemble, OdeEnsembleConfig, EnsembleMember, EnsembleDispatch,
//! };
//!
//! // Solve y' = -k * y for several values of k
//! let config = OdeEnsembleConfig {
//!     t_span: [0.0, 1.0],
//!     rtol: 1e-6,
//!     atol: 1e-9,
//!     max_steps: 10_000,
//!     dispatch: EnsembleDispatch::Sequential,
//! };
//! let members: Vec<EnsembleMember> = (1..=5)
//!     .map(|k| EnsembleMember {
//!         params: vec![k as f64],
//!         y0: vec![1.0],
//!     })
//!     .collect();
//!
//! let ensemble = OdeEnsemble::new(config);
//! let result = ensemble.integrate(&members, &|t, y, p| vec![-p[0] * y[0]]);
//! assert!(result.success.iter().all(|&s| s));
//! ```

/// Convenience type alias for the right-hand-side function signature.
///
/// The arguments are `(t, y, params) -> dydt`.
pub type OdeRhsFn = Box<dyn Fn(f64, &[f64], &[f64]) -> Vec<f64> + Send + Sync>;

/// Execution strategy for the ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleDispatch {
    /// Integrate members one at a time on the CPU.
    Sequential,
    /// Simulated GPU-batched execution (same numerics as `Sequential`).
    Simulated,
}

/// Configuration for [`OdeEnsemble`].
#[derive(Debug, Clone)]
pub struct OdeEnsembleConfig {
    /// Integration interval `[t_start, t_end]`.
    pub t_span: [f64; 2],
    /// Relative tolerance for the adaptive stepper.
    pub rtol: f64,
    /// Absolute tolerance for the adaptive stepper.
    pub atol: f64,
    /// Maximum number of steps per member before declaring failure.
    pub max_steps: usize,
    /// Execution dispatch strategy.
    pub dispatch: EnsembleDispatch,
}

/// One member of the ensemble: its parameters and initial conditions.
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    /// Parameters passed to the RHS as the third argument.
    pub params: Vec<f64>,
    /// Initial condition `y(t_start)`.
    pub y0: Vec<f64>,
}

/// Result of integrating a full ensemble.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final state `y(t_end)` for each member.
    pub solutions: Vec<Vec<f64>>,
    /// Number of steps taken per member.
    pub n_steps: Vec<usize>,
    /// Whether each member converged within `max_steps`.
    pub success: Vec<bool>,
    /// Final time reached by each member.
    pub t_final: Vec<f64>,
}

/// Ensemble ODE integrator.
pub struct OdeEnsemble {
    config: OdeEnsembleConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Dormand-Prince RK45 Butcher tableau
// ─────────────────────────────────────────────────────────────────────────────
//
//   0    |
//  1/5   | 1/5
//  3/10  | 3/40        9/40
//  4/5   | 44/45      -56/15      32/9
//  8/9   | 19372/6561 -25360/2187  64448/6561  -212/729
//  1     | 9017/3168  -355/33      46732/5247   49/176   -5103/18656
//  1     | 35/384      0           500/1113     125/192  -2187/6784   11/84
//
//  Order-4 error estimate coefficients (difference: 5th − 4th order)
//  e = y5 − y4
//  e1 = 71/57600,  e3 = -71/16695, e4 = 71/1920, e5 = -17253/339200, e6 = 22/525, e7 = -1/40

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order solution weights
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// Error coefficients (5th − 4th order)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// Node positions (c values)
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

// ─────────────────────────────────────────────────────────────────────────────
// Core RK45 step
// ─────────────────────────────────────────────────────────────────────────────

/// Dormand-Prince RK45 adaptive step.
///
/// Advances the state from `(t, y)` by step `h` using the Dormand-Prince
/// pair.  Returns `(y_order5, y_order4_error_estimate, error_norm)`.
///
/// The error norm is the RMS of the componentwise scaled errors:
/// `err_i / (atol + rtol * max(|y_i|, |y5_i|))`.
///
/// # Arguments
///
/// * `t`      — current time.
/// * `y`      — current state (length `n`).
/// * `params` — parameters forwarded verbatim to `rhs`.
/// * `h`      — step size (may be positive or negative).
/// * `rhs`    — right-hand side `f(t, y, params) -> dydt`.
/// * `rtol`   — relative tolerance (for error scaling).
/// * `atol`   — absolute tolerance (for error scaling).
///
/// Returns `(y5, err_norm)` where `y5` is the 5th-order solution.
pub fn rk45_step(
    t: f64,
    y: &[f64],
    params: &[f64],
    h: f64,
    rhs: &dyn Fn(f64, &[f64], &[f64]) -> Vec<f64>,
    rtol: f64,
    atol: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = y.len();

    // Stage 1
    let k1 = rhs(t, y, params);

    // Stage 2
    let y2: Vec<f64> = (0..n).map(|i| y[i] + h * A21 * k1[i]).collect();
    let k2 = rhs(t + C2 * h, &y2, params);

    // Stage 3
    let y3: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
        .collect();
    let k3 = rhs(t + C3 * h, &y3, params);

    // Stage 4
    let y4: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
        .collect();
    let k4 = rhs(t + C4 * h, &y4, params);

    // Stage 5
    let y5_tmp: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
        .collect();
    let k5 = rhs(t + C5 * h, &y5_tmp, params);

    // Stage 6
    let y6_tmp: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
        .collect();
    let k6 = rhs(t + h, &y6_tmp, params);

    // 5th-order solution
    let y_new: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]))
        .collect();

    // Stage 7 (FSAL: first same as last)
    let k7 = rhs(t + h, &y_new, params);

    // Error estimate: e = y5 - y4  (using the E coefficients)
    let err_vec: Vec<f64> = (0..n)
        .map(|i| h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]))
        .collect();

    // RMS error norm (scaled)
    let err_norm = {
        let sum_sq: f64 = (0..n)
            .map(|i| {
                let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
                let e = err_vec[i] / sc;
                e * e
            })
            .sum::<f64>();
        (sum_sq / n as f64).sqrt()
    };

    (y_new, err_vec, err_norm)
}

// ─────────────────────────────────────────────────────────────────────────────
// OdeEnsemble implementation
// ─────────────────────────────────────────────────────────────────────────────

impl OdeEnsemble {
    /// Create a new ensemble integrator with the given configuration.
    pub fn new(config: OdeEnsembleConfig) -> Self {
        Self { config }
    }

    /// Integrate all members from `t_span[0]` to `t_span[1]`.
    ///
    /// # Arguments
    ///
    /// * `members` — slice of ensemble members (parameters + initial conditions).
    /// * `rhs`     — right-hand side `f(t, y, params) -> dydt`.
    ///
    /// # Returns
    ///
    /// An [`EnsembleResult`] containing the final state for each member.
    pub fn integrate(
        &self,
        members: &[EnsembleMember],
        rhs: &dyn Fn(f64, &[f64], &[f64]) -> Vec<f64>,
    ) -> EnsembleResult {
        let n = members.len();
        let mut solutions = Vec::with_capacity(n);
        let mut n_steps_vec = Vec::with_capacity(n);
        let mut success_vec = Vec::with_capacity(n);
        let mut t_final_vec = Vec::with_capacity(n);

        for member in members {
            let (y_final, n_steps, ok) = self.integrate_single(member, rhs);
            let t_reached = if ok {
                self.config.t_span[1]
            } else {
                // Report partial progress: we don't track intermediate times in
                // the current implementation, so report t_start on failure.
                self.config.t_span[0]
            };
            solutions.push(y_final);
            n_steps_vec.push(n_steps);
            success_vec.push(ok);
            t_final_vec.push(t_reached);
        }

        EnsembleResult {
            solutions,
            n_steps: n_steps_vec,
            success: success_vec,
            t_final: t_final_vec,
        }
    }

    /// Integrate a single ensemble member using adaptive RK45.
    ///
    /// Returns `(final_y, n_steps, converged)`.
    fn integrate_single(
        &self,
        member: &EnsembleMember,
        rhs: &dyn Fn(f64, &[f64], &[f64]) -> Vec<f64>,
    ) -> (Vec<f64>, usize, bool) {
        let t_start = self.config.t_span[0];
        let t_end = self.config.t_span[1];
        let rtol = self.config.rtol;
        let atol = self.config.atol;
        let max_steps = self.config.max_steps;

        let mut t = t_start;
        let mut y = member.y0.clone();
        let n = y.len();

        if n == 0 {
            return (y, 0, true);
        }

        // Initial step size heuristic
        let span = (t_end - t_start).abs();
        let mut h = span * 1e-3;
        // Clamp to avoid overshooting on the first step
        h = h.min(span);

        let direction = if t_end >= t_start { 1.0_f64 } else { -1.0 };
        h *= direction;

        let fac = 0.9_f64;
        let fac_max = 10.0_f64;
        let fac_min = 0.2_f64;

        let mut steps = 0_usize;
        let mut converged = false;

        while (direction * (t_end - t)).abs() > 1e-12 * span.max(f64::EPSILON) {
            if steps >= max_steps {
                break;
            }

            // Don't overshoot the endpoint
            if direction * (t + h - t_end) > 0.0 {
                h = t_end - t;
            }
            if h.abs() < f64::EPSILON * span {
                // Step size collapsed — declare failure
                break;
            }

            let (y_new, _err_vec, err_norm) = rk45_step(t, &y, &member.params, h, rhs, rtol, atol);

            // Accept or reject step
            if err_norm <= 1.0 || err_norm.is_nan() {
                // Accept
                t += h;
                y = y_new;
                steps += 1;

                if (direction * (t_end - t)).abs() < 1e-12 * span.max(f64::EPSILON) {
                    converged = true;
                    break;
                }
            }

            // Adjust step size
            let err_safe = err_norm.max(f64::EPSILON);
            let factor = fac * err_safe.powf(-0.2);
            let factor = factor.clamp(fac_min, fac_max);
            h *= factor;

            // Safety: if we accepted, count this step towards the limit already
            // (done above via `steps += 1`).
        }

        // If we've reached t_end within tolerance, mark as converged
        if (t - t_end).abs() < 1e-8 * span.max(f64::EPSILON) {
            converged = true;
        }

        (y, steps, converged)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> OdeEnsembleConfig {
        OdeEnsembleConfig {
            t_span: [0.0, 1.0],
            rtol: 1e-7,
            atol: 1e-9,
            max_steps: 100_000,
            dispatch: EnsembleDispatch::Sequential,
        }
    }

    /// Five members with identical parameters and initial conditions must
    /// produce identical solutions.
    #[test]
    fn test_identical_params_same_solution() {
        let config = default_config();
        let ensemble = OdeEnsemble::new(config);
        let members: Vec<EnsembleMember> = (0..5)
            .map(|_| EnsembleMember {
                params: vec![2.0],
                y0: vec![1.0],
            })
            .collect();
        let result = ensemble.integrate(&members, &|_t, y, p| vec![-p[0] * y[0]]);
        let y0 = &result.solutions[0];
        for (i, sol) in result.solutions.iter().enumerate().skip(1) {
            assert!(
                (sol[0] - y0[0]).abs() < 1e-14,
                "member {i} diverges from member 0: {:.6e} vs {:.6e}",
                sol[0],
                y0[0]
            );
        }
    }

    /// Members with different decay rates must give different final values.
    #[test]
    fn test_different_params_different_solutions() {
        let config = default_config();
        let ensemble = OdeEnsemble::new(config);
        let ks: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let members: Vec<EnsembleMember> = ks
            .iter()
            .map(|&k| EnsembleMember {
                params: vec![k],
                y0: vec![1.0],
            })
            .collect();
        let result = ensemble.integrate(&members, &|_t, y, p| vec![-p[0] * y[0]]);
        // Higher k → smaller y(1)
        for i in 1..ks.len() {
            let y_prev = result.solutions[i - 1][0];
            let y_curr = result.solutions[i][0];
            assert!(
                y_curr < y_prev,
                "k={} solution ({:.6e}) should be < k={} solution ({:.6e})",
                ks[i],
                y_curr,
                ks[i - 1],
                y_prev
            );
        }
    }

    /// Exponential decay: y' = -k*y, y(0) = y0.
    /// Analytical solution: y(t) = y0 * exp(-k*t).
    #[test]
    fn test_exponential_decay_analytical() {
        let config = OdeEnsembleConfig {
            t_span: [0.0, 2.0],
            rtol: 1e-8,
            atol: 1e-10,
            max_steps: 100_000,
            dispatch: EnsembleDispatch::Sequential,
        };
        let ensemble = OdeEnsemble::new(config);
        let k = 3.0_f64;
        let y0 = 2.5_f64;
        let members = vec![EnsembleMember {
            params: vec![k],
            y0: vec![y0],
        }];
        let result = ensemble.integrate(&members, &|_t, y, p| vec![-p[0] * y[0]]);
        let y_numerical = result.solutions[0][0];
        let y_analytical = y0 * (-k * 2.0_f64).exp();
        assert!(
            (y_numerical - y_analytical).abs() < 1e-6,
            "y_numerical = {y_numerical:.8e}, y_analytical = {y_analytical:.8e}"
        );
    }

    /// All members of a well-behaved system must converge.
    #[test]
    fn test_all_converge() {
        let config = default_config();
        let ensemble = OdeEnsemble::new(config);
        let members: Vec<EnsembleMember> = (1..=5)
            .map(|k| EnsembleMember {
                params: vec![k as f64],
                y0: vec![1.0],
            })
            .collect();
        let result = ensemble.integrate(&members, &|_t, y, p| vec![-p[0] * y[0]]);
        for (i, &ok) in result.success.iter().enumerate() {
            assert!(ok, "member {i} did not converge");
        }
    }

    /// Number of steps must be positive for all members.
    #[test]
    fn test_n_steps_positive() {
        let config = default_config();
        let ensemble = OdeEnsemble::new(config);
        let members: Vec<EnsembleMember> = (1..=5)
            .map(|k| EnsembleMember {
                params: vec![k as f64],
                y0: vec![1.0],
            })
            .collect();
        let result = ensemble.integrate(&members, &|_t, y, p| vec![-p[0] * y[0]]);
        for (i, &ns) in result.n_steps.iter().enumerate() {
            assert!(ns > 0, "member {i} took 0 steps");
        }
    }

    /// 2-D system: van-der-Pol oscillator at low μ must be stable.
    #[test]
    fn test_2d_system_vanderpol() {
        let config = OdeEnsembleConfig {
            t_span: [0.0, 5.0],
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 500_000,
            dispatch: EnsembleDispatch::Sequential,
        };
        let ensemble = OdeEnsemble::new(config);
        // mu = 0.1  (weak non-linearity)
        let member = EnsembleMember {
            params: vec![0.1],
            y0: vec![2.0, 0.0],
        };
        let result = ensemble.integrate(&[member], &|_t, y, p| {
            let mu = p[0];
            vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
        });
        assert!(result.success[0], "van-der-Pol did not converge");
        // Final state should be finite
        for &v in &result.solutions[0] {
            assert!(v.is_finite(), "van-der-Pol solution is non-finite");
        }
    }

    /// Simulated dispatch produces the same solutions as sequential.
    #[test]
    fn test_simulated_dispatch_matches_sequential() {
        let config_seq = OdeEnsembleConfig {
            t_span: [0.0, 1.0],
            rtol: 1e-7,
            atol: 1e-9,
            max_steps: 50_000,
            dispatch: EnsembleDispatch::Sequential,
        };
        let config_sim = OdeEnsembleConfig {
            dispatch: EnsembleDispatch::Simulated,
            ..config_seq.clone()
        };
        let members: Vec<EnsembleMember> = vec![
            EnsembleMember {
                params: vec![1.0],
                y0: vec![1.0],
            },
            EnsembleMember {
                params: vec![2.0],
                y0: vec![3.0],
            },
        ];
        let ens_seq = OdeEnsemble::new(config_seq);
        let ens_sim = OdeEnsemble::new(config_sim);
        let rhs = &|_t: f64, y: &[f64], p: &[f64]| vec![-p[0] * y[0]];
        let res_seq = ens_seq.integrate(&members, rhs);
        let res_sim = ens_sim.integrate(&members, rhs);
        for i in 0..members.len() {
            assert!(
                (res_seq.solutions[i][0] - res_sim.solutions[i][0]).abs() < 1e-14,
                "member {i}: sequential={:.6e}, simulated={:.6e}",
                res_seq.solutions[i][0],
                res_sim.solutions[i][0]
            );
        }
    }
}
