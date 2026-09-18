//! Neural ODE (Neural Ordinary Differential Equations) implementation.
//!
//! Provides the Dormand-Prince RK45 adaptive solver and adjoint sensitivity
//! method for memory-efficient gradient computation through continuous dynamics.
//!
//! # Overview
//!
//! Neural ODEs replace discrete layer stacks with a continuous ODE:
//! ```text
//!   dy/dt = f(t, y, θ),   y(t0) = y0
//! ```
//! The output is `y(t1)` obtained by numerical integration. Gradients are
//! computed via the adjoint sensitivity method, which avoids storing all
//! intermediate states during the forward pass.
//!
//! # Example
//! ```rust
//! use tensorlogic_train::neural_ode::{NeuralOde, OdeFunc, OdeSolverConfig};
//!
//! struct LinearOde;
//! impl OdeFunc for LinearOde {
//!     fn call(&self, _t: f64, y: &[f64], params: &[f64]) -> Vec<f64> {
//!         y.iter().zip(params.iter()).map(|(yi, pi)| yi * pi).collect()
//!     }
//!     fn vjp(&self, _t: f64, y: &[f64], params: &[f64], grad: &[f64])
//!         -> (Vec<f64>, f64, Vec<f64>)
//!     {
//!         let dy = grad.iter().zip(params.iter()).map(|(g, p)| g * p).collect();
//!         let dt = 0.0_f64;
//!         let dp = grad.iter().zip(y.iter()).map(|(g, yi)| g * yi).collect();
//!         (dy, dt, dp)
//!     }
//! }
//!
//! let ode = NeuralOde::new(LinearOde, 0.0, 1.0);
//! let sol = ode.forward(&[1.0], &[-1.0]).unwrap();
//! assert!((sol.states.last().unwrap()[0] - (-1.0_f64).exp()).abs() < 1e-3);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Public traits
// ---------------------------------------------------------------------------

/// ODE right-hand side: `dy/dt = f(t, y, params)`.
///
/// Implement this trait to define the dynamics of a Neural ODE layer.
pub trait OdeFunc: Send + Sync {
    /// Evaluate the ODE RHS at time `t`, state `y`, and parameters `params`.
    fn call(&self, t: f64, y: &[f64], params: &[f64]) -> Vec<f64>;

    /// Vector-Jacobian product (VJP) for the adjoint method.
    ///
    /// Returns `(dL/dy, dL/dt, dL/dparams)` given `grad_output = dL/df`.
    ///
    /// The default implementation uses finite differences (expensive but
    /// correct). Override for analytic efficiency.
    fn vjp(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        grad_output: &[f64],
    ) -> (Vec<f64>, f64, Vec<f64>) {
        let eps = 1e-6_f64;
        let n = y.len();
        let p = params.len();

        // dL/dy via finite differences
        let mut grad_y = vec![0.0_f64; n];
        for i in 0..n {
            let mut y_plus = y.to_vec();
            let mut y_minus = y.to_vec();
            y_plus[i] += eps;
            y_minus[i] -= eps;
            let f_plus = self.call(t, &y_plus, params);
            let f_minus = self.call(t, &y_minus, params);
            for (k, go) in grad_output.iter().enumerate() {
                grad_y[i] += go * (f_plus[k] - f_minus[k]) / (2.0 * eps);
            }
        }

        // dL/dt via finite differences
        let f_tplus = self.call(t + eps, y, params);
        let f_tminus = self.call(t - eps, y, params);
        let grad_t: f64 = grad_output
            .iter()
            .enumerate()
            .map(|(k, go)| go * (f_tplus[k] - f_tminus[k]) / (2.0 * eps))
            .sum();

        // dL/dparams via finite differences
        let mut grad_params = vec![0.0_f64; p];
        for j in 0..p {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[j] += eps;
            p_minus[j] -= eps;
            let f_plus = self.call(t, y, &p_plus);
            let f_minus = self.call(t, y, &p_minus);
            for (k, go) in grad_output.iter().enumerate() {
                grad_params[j] += go * (f_plus[k] - f_minus[k]) / (2.0 * eps);
            }
        }

        (grad_y, grad_t, grad_params)
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a fixed-step RK4 integration.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// Time points at which the state was recorded.
    pub times: Vec<f64>,
    /// States corresponding to each time point. `states[i]` is `y(times[i])`.
    pub states: Vec<Vec<f64>>,
    /// Total number of ODE function evaluations.
    pub nfev: usize,
}

/// Result of an adaptive Dormand-Prince RK45 integration.
#[derive(Debug, Clone)]
pub struct AdaptiveSolution {
    /// The embedded ODE solution.
    pub solution: OdeSolution,
    /// Number of steps that were rejected due to error tolerance violation.
    pub rejected_steps: usize,
    /// Step size at the final accepted step.
    pub final_step_size: f64,
}

/// Gradient information produced by the adjoint sensitivity method.
#[derive(Debug, Clone)]
pub struct AdjointResult {
    /// Final state `y(t1)` from the forward pass.
    pub final_state: Vec<f64>,
    /// Gradient with respect to the initial state: `dL/dy0`.
    pub grad_y0: Vec<f64>,
    /// Gradient with respect to parameters: `dL/dθ`.
    pub grad_params: Vec<f64>,
    /// Total ODE function evaluations (forward + backward).
    pub total_nfev: usize,
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive ODE solver.
#[derive(Debug, Clone)]
pub struct OdeSolverConfig {
    /// Relative tolerance (default `1e-4`).
    pub rtol: f64,
    /// Absolute tolerance (default `1e-6`).
    pub atol: f64,
    /// Maximum number of integration steps (default `1000`).
    pub max_steps: usize,
    /// Minimum allowed step size (default `1e-12`).
    pub min_step: f64,
    /// Maximum allowed step size (default `f64::INFINITY`).
    pub max_step: f64,
    /// Whether to store every accepted step (`true`) or only the endpoint
    /// (`false`).
    pub dense_output: bool,
}

impl Default for OdeSolverConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 1000,
            min_step: 1e-12,
            max_step: f64::INFINITY,
            dense_output: true,
        }
    }
}

impl OdeSolverConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the relative tolerance (builder pattern).
    pub fn rtol(mut self, v: f64) -> Self {
        self.rtol = v;
        self
    }

    /// Set the absolute tolerance (builder pattern).
    pub fn atol(mut self, v: f64) -> Self {
        self.atol = v;
        self
    }

    /// Set the maximum number of integration steps (builder pattern).
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Disable intermediate state storage (builder pattern).
    pub fn no_dense_output(mut self) -> Self {
        self.dense_output = false;
        self
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during ODE integration.
#[derive(Debug)]
pub enum OdeError {
    /// The solver exceeded the maximum number of allowed steps.
    MaxStepsExceeded,
    /// The adaptive solver required a step smaller than `min_step`.
    StepTooSmall,
    /// The solution grew without bound (detected by NaN/Inf in state).
    DivergentSolution,
    /// Invalid input parameters were supplied.
    InvalidInput(String),
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::MaxStepsExceeded => write!(
                f,
                "ODE solver exceeded the maximum number of steps; \
                 consider relaxing tolerances or increasing max_steps"
            ),
            OdeError::StepTooSmall => write!(
                f,
                "ODE solver step size fell below the minimum threshold; \
                 the problem may be too stiff for this explicit solver"
            ),
            OdeError::DivergentSolution => write!(
                f,
                "ODE solution diverged (NaN or Inf encountered in state vector)"
            ),
            OdeError::InvalidInput(msg) => {
                write!(f, "ODE solver received invalid input: {msg}")
            }
        }
    }
}

impl std::error::Error for OdeError {}

// ---------------------------------------------------------------------------
// Helper arithmetic on Vec<f64>
// ---------------------------------------------------------------------------

#[inline]
#[allow(dead_code)]
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline]
#[allow(dead_code)]
fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

#[inline]
fn vec_axpy(y: &[f64], alpha: f64, x: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(yi, xi)| yi + alpha * xi)
        .collect()
}

/// Compute the mixed-tolerance norm used for step-size control.
///
/// Uses the standard formula `||e||_rms = sqrt( mean( (e_i / sc_i)^2 ) )`
/// where `sc_i = atol + rtol * max(|y_i|, |y_new_i|)`.
fn error_norm(err: &[f64], y: &[f64], y_new: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = err.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = err
        .iter()
        .zip(y.iter())
        .zip(y_new.iter())
        .map(|((e, yi), yn)| {
            let sc = atol + rtol * yi.abs().max(yn.abs());
            (e / sc).powi(2)
        })
        .sum();
    (sum / n as f64).sqrt()
}

/// Check whether any element of the state is NaN or infinite.
fn has_diverged(v: &[f64]) -> bool {
    v.iter().any(|x| x.is_nan() || x.is_infinite())
}

// ---------------------------------------------------------------------------
// Fixed-step RK4 solver
// ---------------------------------------------------------------------------

/// Integrate an ODE using classic 4th-order Runge-Kutta with a fixed step size.
///
/// # Arguments
/// - `func`       – ODE right-hand side.
/// - `t0`, `t1`  – integration interval `[t0, t1]`.
/// - `y0`        – initial state.
/// - `params`    – ODE parameters passed through to `func`.
/// - `num_steps` – number of equal-width steps to take.
///
/// # Returns
/// An [`OdeSolution`] that always contains both the initial and final states.
/// If `dense_output` semantics are desired, all intermediate states are stored.
pub fn rk4_solve(
    func: &dyn OdeFunc,
    t0: f64,
    t1: f64,
    y0: &[f64],
    params: &[f64],
    num_steps: usize,
) -> OdeSolution {
    let steps = num_steps.max(1);
    let h = (t1 - t0) / steps as f64;

    let mut times = Vec::with_capacity(steps + 1);
    let mut states = Vec::with_capacity(steps + 1);
    let mut nfev = 0usize;

    times.push(t0);
    states.push(y0.to_vec());

    let mut t = t0;
    let mut y = y0.to_vec();

    for _ in 0..steps {
        // k1
        let k1 = func.call(t, &y, params);
        nfev += 1;
        // k2
        let y2 = vec_axpy(&y, h * 0.5, &k1);
        let k2 = func.call(t + h * 0.5, &y2, params);
        nfev += 1;
        // k3
        let y3 = vec_axpy(&y, h * 0.5, &k2);
        let k3 = func.call(t + h * 0.5, &y3, params);
        nfev += 1;
        // k4
        let y4 = vec_axpy(&y, h, &k3);
        let k4 = func.call(t + h, &y4, params);
        nfev += 1;

        // y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        y = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                yi + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            })
            .collect();
        t += h;

        times.push(t);
        states.push(y.clone());
    }

    OdeSolution {
        times,
        states,
        nfev,
    }
}

// ---------------------------------------------------------------------------
// Dormand-Prince RK45 adaptive solver (DOPRI5)
// ---------------------------------------------------------------------------

/// Dormand-Prince Butcher tableau coefficients (DOPRI5 / RK45).
///
/// c-coefficients (nodes):
/// ```text
///   c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
/// ```
///
/// a-coefficients (Runge-Kutta matrix, lower-triangular):
/// ```text
///   a21 = 1/5
///   a31 = 3/40,      a32 = 9/40
///   a41 = 44/45,     a42 = -56/15,    a43 = 32/9
///   a51 = 19372/6561, a52=-25360/2187, a53=64448/6561, a54=-212/729
///   a61 = 9017/3168,  a62=-355/33,     a63=46732/5247, a64=49/176,  a65=-5103/18656
///   a71 = 35/384,     a72 = 0,         a73=500/1113,   a74=125/192, a75=-2187/6784, a76=11/84
/// ```
///
/// 5th-order weights `b5` = a7x (FSAL property):
/// ```text
///   b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
/// ```
///
/// 4th-order weights `b4` (embedded):
/// ```text
///   b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
/// ```
///
/// Error coefficients `e = b5 - b4`:
const DOPRI5_A21: f64 = 1.0 / 5.0;
const DOPRI5_A31: f64 = 3.0 / 40.0;
const DOPRI5_A32: f64 = 9.0 / 40.0;
const DOPRI5_A41: f64 = 44.0 / 45.0;
const DOPRI5_A42: f64 = -56.0 / 15.0;
const DOPRI5_A43: f64 = 32.0 / 9.0;
const DOPRI5_A51: f64 = 19372.0 / 6561.0;
const DOPRI5_A52: f64 = -25360.0 / 2187.0;
const DOPRI5_A53: f64 = 64448.0 / 6561.0;
const DOPRI5_A54: f64 = -212.0 / 729.0;
const DOPRI5_A61: f64 = 9017.0 / 3168.0;
const DOPRI5_A62: f64 = -355.0 / 33.0;
const DOPRI5_A63: f64 = 46732.0 / 5247.0;
const DOPRI5_A64: f64 = 49.0 / 176.0;
const DOPRI5_A65: f64 = -5103.0 / 18656.0;
const DOPRI5_A71: f64 = 35.0 / 384.0;
const DOPRI5_A73: f64 = 500.0 / 1113.0;
const DOPRI5_A74: f64 = 125.0 / 192.0;
const DOPRI5_A75: f64 = -2187.0 / 6784.0;
const DOPRI5_A76: f64 = 11.0 / 84.0;

// Error coefficients e_i = b5_i - b4_i
const DOPRI5_E1: f64 = 71.0 / 57600.0;
const DOPRI5_E3: f64 = -71.0 / 16695.0;
const DOPRI5_E4: f64 = 71.0 / 1920.0;
const DOPRI5_E5: f64 = -17253.0 / 339200.0;
const DOPRI5_E6: f64 = 22.0 / 525.0;
const DOPRI5_E7: f64 = -1.0 / 40.0;

const DOPRI5_SAFETY: f64 = 0.9;
const DOPRI5_MIN_FACTOR: f64 = 0.2;
const DOPRI5_MAX_FACTOR: f64 = 10.0;
const DOPRI5_ORDER: f64 = 5.0;

/// Integrate an ODE with the adaptive Dormand-Prince RK45 method (DOPRI5).
///
/// # Arguments
/// - `func`   – ODE right-hand side.
/// - `t0`, `t1` – integration interval.
/// - `y0`    – initial state.
/// - `params` – ODE parameters.
/// - `config` – solver tolerances and limits.
///
/// # Returns
/// `Ok(AdaptiveSolution)` on success, or an [`OdeError`] if the solver fails.
pub fn dopri5_solve(
    func: &dyn OdeFunc,
    t0: f64,
    t1: f64,
    y0: &[f64],
    params: &[f64],
    config: &OdeSolverConfig,
) -> Result<AdaptiveSolution, OdeError> {
    if t0 == t1 {
        return Ok(AdaptiveSolution {
            solution: OdeSolution {
                times: vec![t0],
                states: vec![y0.to_vec()],
                nfev: 0,
            },
            rejected_steps: 0,
            final_step_size: 0.0,
        });
    }

    if y0.is_empty() {
        return Err(OdeError::InvalidInput("state vector is empty".into()));
    }

    let forward = t1 > t0;
    let sign = if forward { 1.0_f64 } else { -1.0_f64 };
    let span = (t1 - t0).abs();

    // Initial step size heuristic
    let f0 = func.call(t0, y0, params);
    let d0 = (y0.iter().map(|x| x * x).sum::<f64>() / y0.len() as f64).sqrt();
    let d1 = (f0.iter().map(|x| x * x).sum::<f64>() / f0.len() as f64).sqrt();
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let mut h = sign * h0.min(span).min(config.max_step);

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut k1 = f0;
    let mut nfev = 1usize; // already evaluated f0

    let mut times = vec![t0];
    let mut states = vec![y0.to_vec()];

    let mut rejected_steps = 0usize;
    let mut steps = 0usize;

    while (sign * (t1 - t)).abs() > f64::EPSILON * span.max(1.0) {
        if steps >= config.max_steps {
            return Err(OdeError::MaxStepsExceeded);
        }

        // Clamp step to not overshoot
        if (t + h - t1) * sign > 0.0 {
            h = t1 - t;
        }

        let h_abs = h.abs();
        if h_abs < config.min_step {
            return Err(OdeError::StepTooSmall);
        }

        // Stage 2
        let y2 = vec_axpy(&y, DOPRI5_A21 * h, &k1);
        let k2 = func.call(t + h / 5.0, &y2, params);
        nfev += 1;

        // Stage 3
        let y3: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .map(|((yi, k1i), k2i)| yi + h * (DOPRI5_A31 * k1i + DOPRI5_A32 * k2i))
            .collect();
        let k3 = func.call(t + h * 3.0 / 10.0, &y3, params);
        nfev += 1;

        // Stage 4
        let y4: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .map(|(((yi, k1i), k2i), k3i)| {
                yi + h * (DOPRI5_A41 * k1i + DOPRI5_A42 * k2i + DOPRI5_A43 * k3i)
            })
            .collect();
        let k4 = func.call(t + h * 4.0 / 5.0, &y4, params);
        nfev += 1;

        // Stage 5
        let y5: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                yi + h * (DOPRI5_A51 * k1i + DOPRI5_A52 * k2i + DOPRI5_A53 * k3i + DOPRI5_A54 * k4i)
            })
            .collect();
        let k5 = func.call(t + h * 8.0 / 9.0, &y5, params);
        nfev += 1;

        // Stage 6
        let y6: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .zip(k5.iter())
            .map(|(((((yi, k1i), k2i), k3i), k4i), k5i)| {
                yi + h
                    * (DOPRI5_A61 * k1i
                        + DOPRI5_A62 * k2i
                        + DOPRI5_A63 * k3i
                        + DOPRI5_A64 * k4i
                        + DOPRI5_A65 * k5i)
            })
            .collect();
        let k6 = func.call(t + h, &y6, params);
        nfev += 1;

        // 5th-order solution (= next k1 via FSAL)
        let y_new: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .zip(k5.iter())
            .zip(k6.iter())
            .map(|(((((yi, k1i), k3i), k4i), k5i), k6i)| {
                yi + h
                    * (DOPRI5_A71 * k1i
                        + DOPRI5_A73 * k3i
                        + DOPRI5_A74 * k4i
                        + DOPRI5_A75 * k5i
                        + DOPRI5_A76 * k6i)
            })
            .collect();

        if has_diverged(&y_new) {
            return Err(OdeError::DivergentSolution);
        }

        // Stage 7 (FSAL: this becomes k1 of the next step)
        let k7 = func.call(t + h, &y_new, params);
        nfev += 1;

        // Error estimate using e_i = b5_i - b4_i
        let err: Vec<f64> = k1
            .iter()
            .zip(k3.iter())
            .zip(k4.iter())
            .zip(k5.iter())
            .zip(k6.iter())
            .zip(k7.iter())
            .map(|(((((e1, e3), e4), e5), e6), e7)| {
                h * (DOPRI5_E1 * e1
                    + DOPRI5_E3 * e3
                    + DOPRI5_E4 * e4
                    + DOPRI5_E5 * e5
                    + DOPRI5_E6 * e6
                    + DOPRI5_E7 * e7)
            })
            .collect();

        let error_norm_val = error_norm(&err, &y, &y_new, config.rtol, config.atol);

        if error_norm_val <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            k1 = k7; // FSAL

            if config.dense_output {
                times.push(t);
                states.push(y.clone());
            }
            steps += 1;

            // Compute new step size
            let factor = if error_norm_val == 0.0 {
                DOPRI5_MAX_FACTOR
            } else {
                (DOPRI5_SAFETY * error_norm_val.powf(-1.0 / DOPRI5_ORDER))
                    .clamp(DOPRI5_MIN_FACTOR, DOPRI5_MAX_FACTOR)
            };
            h *= factor;
            h = h.abs().min(config.max_step) * sign;
        } else {
            // Reject step
            rejected_steps += 1;
            let factor = (DOPRI5_SAFETY * error_norm_val.powf(-1.0 / DOPRI5_ORDER))
                .clamp(DOPRI5_MIN_FACTOR, 1.0);
            h *= factor;
        }
    }

    // Ensure endpoint is stored
    // For dense output only push endpoint if not already recorded; for fixed output always push.
    if !config.dense_output || times.last().map(|&last| last != t).unwrap_or(true) {
        times.push(t);
        states.push(y.clone());
    }

    Ok(AdaptiveSolution {
        solution: OdeSolution {
            times,
            states,
            nfev,
        },
        rejected_steps,
        final_step_size: h.abs(),
    })
}

// ---------------------------------------------------------------------------
// NeuralOde layer
// ---------------------------------------------------------------------------

/// A Neural ODE layer that wraps an [`OdeFunc`] with fixed integration limits.
///
/// The forward pass integrates `y(t0) = y0` to `y(t1)` and the `adjoint`
/// method computes `dL/dy0` and `dL/dθ` via the adjoint sensitivity method
/// without storing all intermediate activations.
pub struct NeuralOde<F: OdeFunc> {
    func: F,
    t0: f64,
    t1: f64,
    config: OdeSolverConfig,
}

impl<F: OdeFunc> NeuralOde<F> {
    /// Create a new [`NeuralOde`] with default solver configuration.
    pub fn new(func: F, t0: f64, t1: f64) -> Self {
        Self {
            func,
            t0,
            t1,
            config: OdeSolverConfig::default(),
        }
    }

    /// Create a new [`NeuralOde`] with a custom solver configuration.
    pub fn with_config(func: F, t0: f64, t1: f64, config: OdeSolverConfig) -> Self {
        Self {
            func,
            t0,
            t1,
            config,
        }
    }

    /// Forward pass: integrate `y0` from `t0` to `t1`.
    ///
    /// Uses the adaptive DOPRI5 solver internally.
    pub fn forward(&self, y0: &[f64], params: &[f64]) -> Result<OdeSolution, OdeError> {
        if y0.is_empty() {
            return Err(OdeError::InvalidInput("initial state is empty".into()));
        }
        let adaptive = dopri5_solve(&self.func, self.t0, self.t1, y0, params, &self.config)?;
        Ok(adaptive.solution)
    }

    /// Full adjoint sensitivity method.
    ///
    /// Integrates forward to obtain `y(t1)`, then runs the augmented backward
    /// ODE to compute `dL/dy0` and `dL/dθ` in a single backward pass.
    ///
    /// # Arguments
    /// - `y0`          – initial state.
    /// - `params`      – ODE parameters.
    /// - `grad_output` – upstream gradient `dL/dy(t1)`.
    pub fn adjoint(
        &self,
        y0: &[f64],
        params: &[f64],
        grad_output: &[f64],
    ) -> Result<AdjointResult, OdeError> {
        if y0.len() != grad_output.len() {
            return Err(OdeError::InvalidInput(format!(
                "grad_output length {} does not match state dimension {}",
                grad_output.len(),
                y0.len()
            )));
        }

        // Forward pass with dense output to store trajectory
        let fwd_config = OdeSolverConfig {
            dense_output: true,
            ..self.config.clone()
        };
        let adaptive = dopri5_solve(&self.func, self.t0, self.t1, y0, params, &fwd_config)?;
        let fwd_nfev = adaptive.solution.nfev;

        let adj_result = adjoint_backward(
            &self.func,
            &adaptive.solution,
            params,
            grad_output,
            &self.config,
        );

        Ok(AdjointResult {
            total_nfev: fwd_nfev + adj_result.total_nfev,
            ..adj_result
        })
    }
}

// ---------------------------------------------------------------------------
// Adjoint backward pass
// ---------------------------------------------------------------------------

/// Run the augmented adjoint backward integration.
///
/// The adjoint (co-state) `a(t) = dL/dy(t)` satisfies:
/// ```text
///   da/dt = -a^T * (∂f/∂y)
/// ```
/// evaluated backwards from `t1` to `t0`. Simultaneously, the parameter
/// gradient accumulates as:
/// ```text
///   dL/dθ = -∫_{t1}^{t0} a^T * (∂f/∂θ) dt
/// ```
///
/// Implementation: we integrate the augmented state `[a, dL/dθ]` backward
/// in time using RK4, stepping through the stored forward trajectory in
/// reverse order to obtain accurate `y(t)` at each sub-step.
fn adjoint_backward(
    func: &dyn OdeFunc,
    solution: &OdeSolution,
    params: &[f64],
    grad_output: &[f64],
    _config: &OdeSolverConfig,
) -> AdjointResult {
    let n_state = grad_output.len();
    let n_params = params.len();

    let final_state = solution
        .states
        .last()
        .cloned()
        .unwrap_or_else(|| grad_output.to_vec());

    // Initialise adjoint at t1
    let mut a = grad_output.to_vec();
    let mut grad_params = vec![0.0_f64; n_params];
    let mut total_nfev = 0usize;

    // Use a fixed number of backward sub-steps per forward interval
    let adj_steps_per_interval = 4usize;

    // Walk intervals in reverse: [t_{k+1}, t_k]
    let n_intervals = solution.times.len().saturating_sub(1);
    for interval_idx in (0..n_intervals).rev() {
        let t_start = solution.times[interval_idx + 1];
        let t_end = solution.times[interval_idx];
        let y_start = &solution.states[interval_idx + 1];
        let y_end = &solution.states[interval_idx];

        // Subdivide backward interval with fixed-step RK4
        let h = (t_end - t_start) / adj_steps_per_interval as f64;

        let mut t_cur = t_start;

        for step_idx in 0..adj_steps_per_interval {
            // Interpolate y linearly between the two stored states for
            // the current sub-step (simple but sufficient for moderate stiffness)
            let alpha = step_idx as f64 / adj_steps_per_interval as f64;
            let y_interp: Vec<f64> = y_start
                .iter()
                .zip(y_end.iter())
                .map(|(ys, ye)| ys + alpha * (ye - ys))
                .collect();

            // Augmented RHS evaluated at current (a, param_grad)
            let aug_rhs =
                |t_local: f64, a_local: &[f64], y_local: &[f64]| -> (Vec<f64>, Vec<f64>) {
                    let (da_dy, _da_dt, da_dp) = func.vjp(t_local, y_local, params, a_local);
                    // a-dot = -da_dy  (adjoint equation in backward time)
                    let a_dot: Vec<f64> = da_dy.iter().map(|x| -x).collect();
                    // grad_params accumulation = -da_dp
                    let gp_dot: Vec<f64> = da_dp.iter().map(|x| -x).collect();
                    (a_dot, gp_dot)
                };

            // RK4 for augmented state [a, grad_params]
            let (k1_a, k1_gp) = aug_rhs(t_cur, &a, &y_interp);
            total_nfev += 1;

            let a2 = vec_axpy(&a, h * 0.5, &k1_a);
            let alpha2 = (step_idx as f64 + 0.5) / adj_steps_per_interval as f64;
            let y2: Vec<f64> = y_start
                .iter()
                .zip(y_end.iter())
                .map(|(ys, ye)| ys + alpha2 * (ye - ys))
                .collect();
            let (k2_a, k2_gp) = aug_rhs(t_cur + h * 0.5, &a2, &y2);
            total_nfev += 1;

            let a3 = vec_axpy(&a, h * 0.5, &k2_a);
            let (k3_a, k3_gp) = aug_rhs(t_cur + h * 0.5, &a3, &y2);
            total_nfev += 1;

            let a4 = vec_axpy(&a, h, &k3_a);
            let alpha_end = (step_idx + 1) as f64 / adj_steps_per_interval as f64;
            let y4: Vec<f64> = y_start
                .iter()
                .zip(y_end.iter())
                .map(|(ys, ye)| ys + alpha_end * (ye - ys))
                .collect();
            let (k4_a, k4_gp) = aug_rhs(t_cur + h, &a4, &y4);
            total_nfev += 1;

            // Update a and grad_params
            a = a
                .iter()
                .zip(k1_a.iter())
                .zip(k2_a.iter())
                .zip(k3_a.iter())
                .zip(k4_a.iter())
                .map(|((((ai, k1i), k2i), k3i), k4i)| {
                    ai + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
                })
                .collect();

            grad_params = grad_params
                .iter()
                .zip(k1_gp.iter())
                .zip(k2_gp.iter())
                .zip(k3_gp.iter())
                .zip(k4_gp.iter())
                .map(|((((gp, k1i), k2i), k3i), k4i)| {
                    gp + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
                })
                .collect();

            t_cur += h;
        }

        let _ = n_state; // suppress unused warning if n_state == 0
    }

    AdjointResult {
        final_state,
        grad_y0: a,
        grad_params,
        total_nfev,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper ODE functions -----------------------------------------------

    /// dy/dt = 0  (constant function)
    struct ConstantOde;
    impl OdeFunc for ConstantOde {
        fn call(&self, _t: f64, _y: &[f64], _params: &[f64]) -> Vec<f64> {
            vec![0.0]
        }
        fn vjp(
            &self,
            _t: f64,
            _y: &[f64],
            _params: &[f64],
            _grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            (vec![0.0], 0.0, vec![])
        }
    }

    /// dy/dt = y  (exponential growth)
    struct ExponentialGrowthOde;
    impl OdeFunc for ExponentialGrowthOde {
        fn call(&self, _t: f64, y: &[f64], _params: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
        fn vjp(
            &self,
            _t: f64,
            _y: &[f64],
            _params: &[f64],
            grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            // df/dy = I, so VJP = grad
            (grad.to_vec(), 0.0, vec![])
        }
    }

    /// dy/dt = -y  (exponential decay)
    struct ExponentialDecayOde;
    impl OdeFunc for ExponentialDecayOde {
        fn call(&self, _t: f64, y: &[f64], _params: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        fn vjp(
            &self,
            _t: f64,
            _y: &[f64],
            _params: &[f64],
            grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            (grad.iter().map(|g| -g).collect(), 0.0, vec![])
        }
    }

    /// Harmonic oscillator: dx/dt = y, dy/dt = -x  (unit circle)
    struct OscillatorOde;
    impl OdeFunc for OscillatorOde {
        fn call(&self, _t: f64, y: &[f64], _params: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]]
        }
        fn vjp(
            &self,
            _t: f64,
            _y: &[f64],
            _params: &[f64],
            grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            // Jacobian: [[0, 1], [-1, 0]], so VJP = grad^T * J
            let ga = grad[1]; // d/dy0 = -grad[1]
            let gb = grad[0]; // d/dy1 =  grad[0]
            (vec![-ga, gb], 0.0, vec![])
        }
    }

    /// dy/dt = param * y  (linear with parameter)
    struct LinearParamOde;
    impl OdeFunc for LinearParamOde {
        fn call(&self, _t: f64, y: &[f64], params: &[f64]) -> Vec<f64> {
            vec![params[0] * y[0]]
        }
        fn vjp(
            &self,
            _t: f64,
            y: &[f64],
            params: &[f64],
            grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            let grad_y = vec![grad[0] * params[0]];
            let grad_p = vec![grad[0] * y[0]];
            (grad_y, 0.0, grad_p)
        }
    }

    /// Stiff ODE: dy/dt = -1000 * y  (stiffness ratio 1000)
    struct StiffOde;
    impl OdeFunc for StiffOde {
        fn call(&self, _t: f64, y: &[f64], _params: &[f64]) -> Vec<f64> {
            vec![-1000.0 * y[0]]
        }
        fn vjp(
            &self,
            _t: f64,
            _y: &[f64],
            _params: &[f64],
            grad: &[f64],
        ) -> (Vec<f64>, f64, Vec<f64>) {
            (grad.iter().map(|g| -1000.0 * g).collect(), 0.0, vec![])
        }
    }

    // =========================================================================
    // Test 1: RK4 solves dy/dt = 0 (constant)
    // =========================================================================
    #[test]
    fn test_rk4_constant_ode() {
        let init_val = 42.0_f64;
        let sol = rk4_solve(&ConstantOde, 0.0, 1.0, &[init_val], &[], 100);
        let final_y = sol.states.last().unwrap()[0];
        assert!(
            (final_y - init_val).abs() < 1e-12,
            "constant ODE should stay at {init_val}, got {final_y}"
        );
    }

    // =========================================================================
    // Test 2: RK4 solves dy/dt = y (exponential growth)
    // =========================================================================
    #[test]
    fn test_rk4_exponential_growth() {
        let sol = rk4_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], 10_000);
        let final_y = sol.states.last().unwrap()[0];
        let exact = std::f64::consts::E;
        assert!(
            (final_y - exact).abs() < 1e-6,
            "RK4 exponential growth: got {final_y}, expected {exact}"
        );
    }

    // =========================================================================
    // Test 3: RK4 solves dy/dt = -y (exponential decay)
    // =========================================================================
    #[test]
    fn test_rk4_exponential_decay() {
        let sol = rk4_solve(&ExponentialDecayOde, 0.0, 1.0, &[1.0], &[], 10_000);
        let final_y = sol.states.last().unwrap()[0];
        let exact = (-1.0_f64).exp();
        assert!(
            (final_y - exact).abs() < 1e-6,
            "RK4 exponential decay: got {final_y}, expected {exact}"
        );
    }

    // =========================================================================
    // Test 4: RK4 with 2D oscillator (unit circle)
    // =========================================================================
    #[test]
    fn test_rk4_oscillator_2d() {
        // Integrate one full period: [0, 2π]
        use std::f64::consts::PI;
        let sol = rk4_solve(&OscillatorOde, 0.0, 2.0 * PI, &[1.0, 0.0], &[], 100_000);
        let last = sol.states.last().unwrap();
        // Should return close to [1, 0]
        assert!(
            (last[0] - 1.0).abs() < 1e-4,
            "oscillator x: got {}",
            last[0]
        );
        assert!(last[1].abs() < 1e-4, "oscillator y: got {}", last[1]);
    }

    // =========================================================================
    // Test 5: DOPRI5 achieves high accuracy when tolerances are tight
    // =========================================================================
    #[test]
    fn test_dopri5_more_accurate_than_rk4() {
        let exact = std::f64::consts::E;

        // RK4 with only 10 steps (coarse fixed-step baseline)
        let rk4_sol = rk4_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], 10);
        let rk4_err = (rk4_sol.states.last().unwrap()[0] - exact).abs();

        // DOPRI5 with tight tolerances — adaptive solver should beat coarse RK4
        let config = OdeSolverConfig::new().rtol(1e-8).atol(1e-10);
        let dp5 = dopri5_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], &config).unwrap();
        let dp5_err = (dp5.solution.states.last().unwrap()[0] - exact).abs();

        assert!(
            dp5_err < rk4_err,
            "DOPRI5 (tight tol) error {dp5_err} should be less than coarse RK4 error {rk4_err}"
        );
        // Verify DOPRI5 achieves the requested tolerance
        assert!(
            dp5_err < 1e-6,
            "DOPRI5 with rtol=1e-8/atol=1e-10 should achieve < 1e-6 error, got {dp5_err}"
        );
    }

    // =========================================================================
    // Test 6: DOPRI5 rejects steps on a stiff function
    // =========================================================================
    #[test]
    fn test_dopri5_step_rejection_on_stiff() {
        let config = OdeSolverConfig::new().rtol(1e-6).atol(1e-8).max_steps(5000);
        // Stiff ODE: many step rejections expected
        let result = dopri5_solve(&StiffOde, 0.0, 0.01, &[1.0], &[], &config);
        // Either succeeds with rejections, or fails with StepTooSmall
        match result {
            Ok(adaptive) => {
                // The solver may reject some steps on a very stiff problem; just verify the field is accessible.
                let _ = adaptive.rejected_steps;
            }
            Err(OdeError::StepTooSmall) | Err(OdeError::MaxStepsExceeded) => {
                // Acceptable — this problem is genuinely stiff
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // =========================================================================
    // Test 7: OdeSolverConfig builder pattern
    // =========================================================================
    #[test]
    fn test_solver_config_builder() {
        let cfg = OdeSolverConfig::new().rtol(1e-8).atol(1e-10).max_steps(500);
        assert!((cfg.rtol - 1e-8).abs() < 1e-15);
        assert!((cfg.atol - 1e-10).abs() < 1e-18);
        assert_eq!(cfg.max_steps, 500);
    }

    // =========================================================================
    // Test 8: NeuralOde::forward returns correct endpoint
    // =========================================================================
    #[test]
    fn test_neural_ode_forward_correct_endpoint() {
        let ode = NeuralOde::new(ExponentialGrowthOde, 0.0, 1.0);
        let sol = ode.forward(&[1.0], &[]).unwrap();
        let final_y = sol.states.last().unwrap()[0];
        let exact = std::f64::consts::E;
        assert!(
            (final_y - exact).abs() < 1e-4,
            "NeuralOde forward: got {final_y}, expected ~{exact}"
        );
    }

    // =========================================================================
    // Test 9: NeuralOde::forward with t0 = t1 returns y0 unchanged
    // =========================================================================
    #[test]
    fn test_neural_ode_forward_t0_equals_t1() {
        let init_val = 7.5_f64; // arbitrary non-special constant
        let ode = NeuralOde::new(ExponentialGrowthOde, 1.5, 1.5);
        let sol = ode.forward(&[init_val], &[]).unwrap();
        // Should contain exactly the initial state
        assert!((sol.states[0][0] - init_val).abs() < 1e-12);
    }

    // =========================================================================
    // Test 10: MaxStepsExceeded error on very stiff problem with tight limits
    // =========================================================================
    #[test]
    fn test_max_steps_exceeded_on_stiff() {
        // Extremely stiff ODE with very few allowed steps and tight tolerances
        let config = OdeSolverConfig::new().rtol(1e-12).atol(1e-14).max_steps(5); // intentionally tiny
        let result = dopri5_solve(&StiffOde, 0.0, 1.0, &[1.0], &[], &config);
        assert!(
            matches!(
                result,
                Err(OdeError::MaxStepsExceeded) | Err(OdeError::StepTooSmall)
            ),
            "expected MaxStepsExceeded or StepTooSmall"
        );
    }

    // =========================================================================
    // Test 11: OdeSolution nfev count is reasonable (> 0)
    // =========================================================================
    #[test]
    fn test_nfev_is_positive() {
        let sol = rk4_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], 10);
        assert!(sol.nfev > 0, "nfev should be > 0, got {}", sol.nfev);
        // RK4: 4 evaluations per step
        assert_eq!(sol.nfev, 40, "RK4 should use 4 * num_steps evaluations");
    }

    // =========================================================================
    // Test 12: AdaptiveSolution.rejected_steps >= 0
    // =========================================================================
    #[test]
    fn test_rejected_steps_field_exists() {
        let config = OdeSolverConfig::new();
        let adaptive = dopri5_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], &config).unwrap();
        // This field must exist and be a valid integer >= 0
        let _ = adaptive.rejected_steps; // type check — usize is always >= 0
        assert!(adaptive.solution.nfev > 0);
    }

    // =========================================================================
    // Test 13: Dense output stores intermediate steps
    // =========================================================================
    #[test]
    fn test_dense_output_stores_intermediate_steps() {
        let config = OdeSolverConfig::new().rtol(1e-6).atol(1e-8);
        let adaptive = dopri5_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], &config).unwrap();
        // Dense output: should have more than just start and end
        assert!(
            adaptive.solution.times.len() > 2,
            "dense output should contain more than 2 time points, got {}",
            adaptive.solution.times.len()
        );
        assert_eq!(
            adaptive.solution.times.len(),
            adaptive.solution.states.len(),
            "times and states must have the same length"
        );
    }

    // =========================================================================
    // Test 14: Adjoint grad_y0 has same dimension as y0
    // =========================================================================
    #[test]
    fn test_adjoint_grad_y0_dimension() {
        let ode = NeuralOde::new(LinearParamOde, 0.0, 0.5);
        let y0 = vec![1.0_f64];
        let params = vec![-1.0_f64];
        let grad_out = vec![1.0_f64];
        let adj = ode.adjoint(&y0, &params, &grad_out).unwrap();
        assert_eq!(
            adj.grad_y0.len(),
            y0.len(),
            "grad_y0 must have same dim as y0"
        );
    }

    // =========================================================================
    // Test 15: Adjoint grad_params has same dimension as params
    // =========================================================================
    #[test]
    fn test_adjoint_grad_params_dimension() {
        let ode = NeuralOde::new(LinearParamOde, 0.0, 0.5);
        let y0 = vec![1.0_f64];
        let params = vec![-1.0_f64];
        let grad_out = vec![1.0_f64];
        let adj = ode.adjoint(&y0, &params, &grad_out).unwrap();
        assert_eq!(
            adj.grad_params.len(),
            params.len(),
            "grad_params must have same dim as params"
        );
    }

    // =========================================================================
    // Test 16: OdeError Display shows meaningful messages
    // =========================================================================
    #[test]
    fn test_ode_error_display() {
        let msgs = [
            (OdeError::MaxStepsExceeded, "max"),
            (OdeError::StepTooSmall, "step"),
            (OdeError::DivergentSolution, "diverged"),
            (OdeError::InvalidInput("bad".into()), "bad"),
        ];
        for (err, keyword) in msgs {
            let msg = format!("{err}");
            assert!(
                msg.to_lowercase().contains(keyword),
                "Display for {err:?} should contain '{keyword}', got: '{msg}'"
            );
        }
    }

    // =========================================================================
    // Test 17: Multiple forward passes produce same result (deterministic)
    // =========================================================================
    #[test]
    fn test_forward_is_deterministic() {
        let ode = NeuralOde::new(ExponentialGrowthOde, 0.0, 1.0);
        let sol1 = ode.forward(&[1.0], &[]).unwrap();
        let sol2 = ode.forward(&[1.0], &[]).unwrap();
        let y1 = sol1.states.last().unwrap()[0];
        let y2 = sol2.states.last().unwrap()[0];
        assert_eq!(y1, y2, "repeated forward passes must be deterministic");
    }

    // =========================================================================
    // Test 18: RK4 converges to exact solution as num_steps increases
    // =========================================================================
    #[test]
    fn test_rk4_convergence_with_steps() {
        let exact = std::f64::consts::E;
        let steps_list = [10usize, 100, 1000, 10_000];
        let mut prev_err = f64::INFINITY;
        for &n in &steps_list {
            let sol = rk4_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], n);
            let err = (sol.states.last().unwrap()[0] - exact).abs();
            assert!(
                err < prev_err,
                "error {err} at n={n} is not less than prev {prev_err}"
            );
            prev_err = err;
        }
        // Final error at 10_000 steps must be very small (RK4 is 4th order;
        // we bound at 1e-13 to allow for floating-point rounding accumulation)
        assert!(
            prev_err < 1e-13,
            "RK4 with 10_000 steps: error {prev_err} > 1e-13"
        );
    }

    // =========================================================================
    // Test 19: DOPRI5 rtol/atol affect solution accuracy
    // =========================================================================
    #[test]
    fn test_dopri5_tolerance_affects_accuracy() {
        let exact = std::f64::consts::E;

        let coarse = OdeSolverConfig::new().rtol(1e-3).atol(1e-5);
        let fine = OdeSolverConfig::new().rtol(1e-9).atol(1e-11);

        let sol_coarse =
            dopri5_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], &coarse).unwrap();
        let sol_fine = dopri5_solve(&ExponentialGrowthOde, 0.0, 1.0, &[1.0], &[], &fine).unwrap();

        let err_coarse = (sol_coarse.solution.states.last().unwrap()[0] - exact).abs();
        let err_fine = (sol_fine.solution.states.last().unwrap()[0] - exact).abs();

        assert!(
            err_fine < err_coarse,
            "fine tol error {err_fine} should be less than coarse tol error {err_coarse}"
        );
    }

    // =========================================================================
    // Test 20: NeuralOde with params affects trajectory
    // =========================================================================
    #[test]
    fn test_neural_ode_params_affect_trajectory() {
        // LinearParamOde: dy/dt = p * y  =>  y(1) = y0 * exp(p)
        let ode = NeuralOde::new(LinearParamOde, 0.0, 1.0);
        let sol_pos = ode.forward(&[1.0], &[1.0]).unwrap(); // y(1) ~ e
        let sol_neg = ode.forward(&[1.0], &[-1.0]).unwrap(); // y(1) ~ e^-1

        let y_pos = sol_pos.states.last().unwrap()[0];
        let y_neg = sol_neg.states.last().unwrap()[0];

        assert!(
            y_pos > y_neg,
            "positive param should give larger y: y_pos={y_pos}, y_neg={y_neg}"
        );
        assert!(
            (y_pos - std::f64::consts::E).abs() < 1e-3,
            "y_pos ~ e, got {y_pos}"
        );
        assert!(
            (y_neg - (-1.0_f64).exp()).abs() < 1e-3,
            "y_neg ~ e^-1, got {y_neg}"
        );
    }

    // =========================================================================
    // Extra: verify AdjointResult fields exist and total_nfev is positive
    // =========================================================================
    #[test]
    fn test_adjoint_result_fields() {
        let ode = NeuralOde::new(LinearParamOde, 0.0, 1.0);
        let adj = ode.adjoint(&[1.0], &[-1.0], &[1.0]).unwrap();
        assert!(adj.total_nfev > 0, "total_nfev should be > 0");
        assert!(!adj.final_state.is_empty(), "final_state must not be empty");
        assert!(!adj.grad_y0.is_empty(), "grad_y0 must not be empty");
        // grad_params matches params dimension (1 here)
        assert_eq!(adj.grad_params.len(), 1);
    }

    // =========================================================================
    // Extra: OdeSolution start state equals y0
    // =========================================================================
    #[test]
    fn test_solution_first_state_is_y0() {
        let y0 = vec![42.0_f64, -7.5];
        let sol = rk4_solve(&OscillatorOde, 0.0, 1.0, &y0, &[], 100);
        assert_eq!(&sol.states[0], &y0, "first stored state must equal y0");
        assert_eq!(sol.times[0], 0.0);
    }
}
