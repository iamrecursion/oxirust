//! SIMD-accelerated explicit ODE solver methods
//!
//! This module provides SIMD-optimized versions of explicit ODE solvers,
//! offering significant performance improvements for large systems of ODEs
//! on modern processors with SIMD instruction sets.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::ode::types::{ODEOptions, ODEResult};
use crate::ode::utils::common::{estimate_initial_step, ODEState, StepResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

#[cfg(feature = "simd")]
use crate::ode::utils::simd_ops::SimdOdeOps;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-accelerated 4th-order Runge-Kutta method
///
/// This implementation uses SIMD instructions to accelerate vector operations
/// in the RK4 integration steps, providing significant performance improvements
/// for large systems of ODEs.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_rk4_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine step size
    let h = opts.h0.unwrap_or_else(|| {
        let dy0 = f(t_start, y0.view());
        let tol = opts.atol + opts.rtol;
        estimate_initial_step(&f, t_start, &y0, &dy0, tol, t_end)
    });

    // Storage for solution
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    let mut t = t_start;
    let mut y = y0;
    let mut steps = 0;
    let mut func_evals = 0;

    while t < t_end {
        // Adjust step size near the end
        let h_current = if t + h > t_end { t_end - t } else { h };

        // SIMD-accelerated RK4 step
        let (y_new, n_evals) = simd_rk4_step(&f, t, &y.view(), h_current)?;
        func_evals += n_evals;

        // Update state
        t += h_current;
        y = y_new;
        steps += 1;

        // Store solution
        t_values.push(t);
        y_values.push(y.clone());

        // Safety check
        if steps > 1_000_000 {
            return Err(crate::error::IntegrateError::ConvergenceError(
                "Maximum number of steps exceeded in SIMD RK4 method".to_string(),
            ));
        }
    }

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        n_steps: steps,
        n_eval: func_evals,
        n_accepted: steps,
        n_rejected: 0,
        n_lu: 0,
        n_jac: 0,
        method: crate::ode::types::ODEMethod::RK4,
        success: true,
        message: Some("Integration completed successfully".to_string()),
    })
}

/// SIMD-accelerated adaptive Runge-Kutta method (RK45)
///
/// This method uses embedded Runge-Kutta formulas with SIMD acceleration
/// for both the integration steps and error estimation.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options including tolerances
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_rk45_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;

    // Initial step size
    let mut h = opts.h0.unwrap_or_else(|| {
        let dy0 = f(t_start, y0.view());
        let tol = opts.atol + opts.rtol;
        estimate_initial_step(&f, t_start, &y0, &dy0, tol, t_end)
    });

    let min_step = opts
        .min_step
        .unwrap_or(F::from_f64(1e-12).expect("Operation failed"));
    let max_step = opts
        .max_step
        .unwrap_or((t_end - t_start) / F::from_f64(10.0).expect("Operation failed"));
    let abs_tol = opts.atol;
    let rel_tol = opts.rtol;

    // Storage for solution
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    let mut t = t_start;
    let mut y = y0;
    let mut steps = 0;
    let mut func_evals = 0;
    let mut rejected_steps = 0;

    while t < t_end {
        // Adjust step size near the end
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // SIMD-accelerated RK45 step with error estimation
        let (y_new, y_star, n_evals) = simd_rk45_step(&f, t, &y.view(), h)?;
        func_evals += n_evals;

        // Compute scaled error norm (matching non-SIMD version)
        let mut err_norm = F::zero();
        for i in 0..y_new.len() {
            let sc = abs_tol + rel_tol * y_new[i].abs();
            let err = (y_new[i] - y_star[i]).abs() / sc;
            err_norm = err_norm.max(err);
        }

        // Step size control (matching non-SIMD version)
        let order = F::from_f64(5.0).expect("Operation failed");
        let exponent = F::one() / (order + F::one());
        let safety = F::from_f64(0.9).expect("Operation failed");
        let factor = safety * (F::one() / err_norm).powf(exponent);
        let factor_min = F::from_f64(0.2).expect("Operation failed");
        let factor_max = F::from_f64(5.0).expect("Operation failed");
        let factor = factor.min(factor_max).max(factor_min);

        if err_norm <= F::one() {
            // Accept step
            t += h;
            y = y_new;
            steps += 1;

            // Store solution
            t_values.push(t);
            y_values.push(y.clone());

            // Adjust step size for next step
            if err_norm <= F::from_f64(0.1).expect("Operation failed") {
                h *= factor.max(F::from_f64(2.0).expect("Operation failed"));
            } else {
                h *= factor;
            }
        } else {
            // Reject step
            rejected_steps += 1;
            h *= factor.min(F::one());

            // Check minimum step size
            if h < min_step {
                return Err(crate::error::IntegrateError::StepSizeTooSmall(
                    "Step size became too small in SIMD RK45 method".to_string(),
                ));
            }
        }

        // Safety check
        if steps > 100_000 {
            return Err(crate::error::IntegrateError::ConvergenceError(
                "Maximum number of steps exceeded in SIMD RK45 method".to_string(),
            ));
        }
    }

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        n_steps: steps,
        n_eval: func_evals,
        n_accepted: steps - rejected_steps,
        n_rejected: rejected_steps,
        n_lu: 0,
        n_jac: 0,
        method: crate::ode::types::ODEMethod::RK45,
        success: true,
        message: Some("Integration completed successfully".to_string()),
    })
}

/// Perform a single SIMD-accelerated RK4 step
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn simd_rk4_step<F, Func>(
    f: &Func,
    t: F,
    y: &ArrayView1<F>,
    h: F,
) -> IntegrateResult<(Array1<F>, usize)>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let h_half = h * F::from_f64(0.5).expect("Operation failed");

    // k1 = f(t, y)
    let k1 = f(t, y.to_owned().view());

    // k2 = f(t + h/2, y + h/2 * k1)
    let y_temp1 = F::simd_add(y, &F::simd_scalar_mul(&k1.view(), h_half).view());
    let k2 = f(t + h_half, y_temp1.view());

    // k3 = f(t + h/2, y + h/2 * k2)
    let y_temp2 = F::simd_add(y, &F::simd_scalar_mul(&k2.view(), h_half).view());
    let k3 = f(t + h_half, y_temp2.view());

    // k4 = f(t + h, y + h * k3)
    let y_temp3 = F::simd_add(y, &F::simd_scalar_mul(&k3.view(), h).view());
    let k4 = f(t + h, y_temp3.view());

    // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    let c1 = F::one() / F::from_f64(6.0).expect("Operation failed");
    let c2 =
        F::from_f64(2.0).expect("Operation failed") / F::from_f64(6.0).expect("Operation failed");

    // Compute weighted sum: k1/6 + k2/3 + k3/3 + k4/6
    let term1 = F::simd_scalar_mul(&k1.view(), c1 * h);
    let term2 = F::simd_scalar_mul(&k2.view(), c2 * h);
    let term3 = F::simd_scalar_mul(&k3.view(), c2 * h);
    let term4 = F::simd_scalar_mul(&k4.view(), c1 * h);

    let sum12 = F::simd_add(&term1.view(), &term2.view());
    let sum34 = F::simd_add(&term3.view(), &term4.view());
    let weighted_sum = F::simd_add(&sum12.view(), &sum34.view());

    let y_new = F::simd_add(y, &weighted_sum.view());

    Ok((y_new, 4)) // 4 function evaluations
}

/// Perform a single SIMD-accelerated RK45 step with error estimation
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn simd_rk45_step<F, Func>(
    f: &Func,
    t: F,
    y: &ArrayView1<F>,
    h: F,
) -> IntegrateResult<(Array1<F>, Array1<F>, usize)>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Dormand-Prince coefficients
    let a21 = F::from_f64(1.0 / 5.0).expect("Operation failed");
    let a31 = F::from_f64(3.0 / 40.0).expect("Operation failed");
    let a32 = F::from_f64(9.0 / 40.0).expect("Operation failed");
    let a41 = F::from_f64(44.0 / 45.0).expect("Operation failed");
    let a42 = F::from_f64(-56.0 / 15.0).expect("Operation failed");
    let a43 = F::from_f64(32.0 / 9.0).expect("Operation failed");
    let a51 = F::from_f64(19372.0 / 6561.0).expect("Operation failed");
    let a52 = F::from_f64(-25360.0 / 2187.0).expect("Operation failed");
    let a53 = F::from_f64(64448.0 / 6561.0).expect("Operation failed");
    let a54 = F::from_f64(-212.0 / 729.0).expect("Operation failed");
    let a61 = F::from_f64(9017.0 / 3168.0).expect("Operation failed");
    let a62 = F::from_f64(-355.0 / 33.0).expect("Operation failed");
    let a63 = F::from_f64(46732.0 / 5247.0).expect("Operation failed");
    let a64 = F::from_f64(49.0 / 176.0).expect("Operation failed");
    let a65 = F::from_f64(-5103.0 / 18656.0).expect("Operation failed");

    // k1 = f(t, y)
    let k1 = f(t, y.to_owned().view());

    // k2 = f(t + h/5, y + h/5 * k1)
    let y2 = F::simd_add(y, &F::simd_scalar_mul(&k1.view(), h * a21).view());
    let k2 = f(t + h * a21, y2.view());

    // k3 = f(t + 3h/10, y + h * (3/40 * k1 + 9/40 * k2))
    let term1 = F::simd_scalar_mul(&k1.view(), a31 * h);
    let term2 = F::simd_scalar_mul(&k2.view(), a32 * h);
    let y3 = F::simd_add(y, &F::simd_add(&term1.view(), &term2.view()).view());
    let k3 = f(
        t + h * F::from_f64(3.0 / 10.0).expect("Operation failed"),
        y3.view(),
    );

    // k4 = f(t + 4h/5, y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
    let t1 = F::simd_scalar_mul(&k1.view(), a41 * h);
    let t2 = F::simd_scalar_mul(&k2.view(), a42 * h);
    let t3 = F::simd_scalar_mul(&k3.view(), a43 * h);
    let y4 = F::simd_add(
        y,
        &F::simd_add(&F::simd_add(&t1.view(), &t2.view()).view(), &t3.view()).view(),
    );
    let k4 = f(
        t + h * F::from_f64(4.0 / 5.0).expect("Operation failed"),
        y4.view(),
    );

    // k5
    let r1 = F::simd_scalar_mul(&k1.view(), a51 * h);
    let r2 = F::simd_scalar_mul(&k2.view(), a52 * h);
    let r3 = F::simd_scalar_mul(&k3.view(), a53 * h);
    let r4 = F::simd_scalar_mul(&k4.view(), a54 * h);
    let sum1 = F::simd_add(&r1.view(), &r2.view());
    let sum2 = F::simd_add(&r3.view(), &r4.view());
    let y5 = F::simd_add(y, &F::simd_add(&sum1.view(), &sum2.view()).view());
    let k5 = f(
        t + h * F::from_f64(8.0 / 9.0).expect("Operation failed"),
        y5.view(),
    );

    // k6
    let s1 = F::simd_scalar_mul(&k1.view(), a61 * h);
    let s2 = F::simd_scalar_mul(&k2.view(), a62 * h);
    let s3 = F::simd_scalar_mul(&k3.view(), a63 * h);
    let s4 = F::simd_scalar_mul(&k4.view(), a64 * h);
    let s5 = F::simd_scalar_mul(&k5.view(), a65 * h);
    let ssum1 = F::simd_add(&s1.view(), &s2.view());
    let ssum2 = F::simd_add(&s3.view(), &s4.view());
    let ssum3 = F::simd_add(&ssum1.view(), &ssum2.view());
    let y6 = F::simd_add(y, &F::simd_add(&ssum3.view(), &s5.view()).view());
    let k6 = f(t + h, y6.view());

    // 5th order solution (y_stage is same as y_new for FSAL property)
    let b1 = F::from_f64(35.0 / 384.0).expect("Operation failed");
    let b3 = F::from_f64(500.0 / 1113.0).expect("Operation failed");
    let b4 = F::from_f64(125.0 / 192.0).expect("Operation failed");
    let b5 = F::from_f64(-2187.0 / 6784.0).expect("Operation failed");
    let b6 = F::from_f64(11.0 / 84.0).expect("Operation failed");

    let w1 = F::simd_scalar_mul(&k1.view(), b1 * h);
    let w3 = F::simd_scalar_mul(&k3.view(), b3 * h);
    let w4 = F::simd_scalar_mul(&k4.view(), b4 * h);
    let w5 = F::simd_scalar_mul(&k5.view(), b5 * h);
    let w6 = F::simd_scalar_mul(&k6.view(), b6 * h);
    let wsum1 = F::simd_add(&w1.view(), &w3.view());
    let wsum2 = F::simd_add(&w4.view(), &w5.view());
    let wsum3 = F::simd_add(&wsum1.view(), &wsum2.view());
    let y_new = F::simd_add(y, &F::simd_add(&wsum3.view(), &w6.view()).view());

    // k7 = f(t + h, y_new) - needed for 4th order solution
    let k7 = f(t + h, y_new.view());

    // 4th order solution for error estimation (includes k7)
    let b1_star = F::from_f64(5179.0 / 57600.0).expect("Operation failed");
    let b3_star = F::from_f64(7571.0 / 16695.0).expect("Operation failed");
    let b4_star = F::from_f64(393.0 / 640.0).expect("Operation failed");
    let b5_star = F::from_f64(-92097.0 / 339200.0).expect("Operation failed");
    let b6_star = F::from_f64(187.0 / 2100.0).expect("Operation failed");
    let b7_star = F::from_f64(1.0 / 40.0).expect("Operation failed");

    let v1 = F::simd_scalar_mul(&k1.view(), b1_star * h);
    let v3 = F::simd_scalar_mul(&k3.view(), b3_star * h);
    let v4 = F::simd_scalar_mul(&k4.view(), b4_star * h);
    let v5 = F::simd_scalar_mul(&k5.view(), b5_star * h);
    let v6 = F::simd_scalar_mul(&k6.view(), b6_star * h);
    let v7 = F::simd_scalar_mul(&k7.view(), b7_star * h);
    let vsum1 = F::simd_add(&v1.view(), &v3.view());
    let vsum2 = F::simd_add(&v4.view(), &v5.view());
    let vsum3 = F::simd_add(&v6.view(), &v7.view());
    let vsum4 = F::simd_add(&vsum1.view(), &vsum2.view());
    let y_star = F::simd_add(y, &F::simd_add(&vsum4.view(), &vsum3.view()).view());

    // Return both 5th and 4th order solutions for error estimation
    Ok((y_new, y_star, 7)) // 7 function evaluations
}

/// Fallback methods when SIMD is not available
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
pub fn simd_rk4_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Fallback to regular RK4 method
    let h = opts.h0.unwrap_or_else(|| {
        let dy0 = f(t_span[0], y0.view());
        let tol = opts.atol + opts.rtol;
        estimate_initial_step(&f, t_span[0], &y0, &dy0, tol, t_span[1])
    });
    crate::ode::methods::explicit::rk4_method(f, t_span, y0, h, opts)
}

#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
pub fn simd_rk45_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Fallback to regular RK45 method
    crate::ode::methods::adaptive::rk45_method(f, t_span, y0, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr1;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_rk4_simple() {
        // Test on simple exponential decay: dy/dt = -y, y(0) = 1
        let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { -y.to_owned() };

        let y0 = arr1(&[1.0]);
        let t_span = [0.0, 1.0];
        let opts = ODEOptions {
            h0: Some(0.1),
            ..Default::default()
        };

        let result = simd_rk4_method(f, t_span, y0, opts).expect("Operation failed");

        // Exact solution at t=1 is exp(-1) ≈ 0.36788
        let final_value = result.y.last().expect("Operation failed")[0];
        let exact = (-1.0_f64).exp();

        assert_relative_eq!(final_value, exact, epsilon = 1e-3);
        assert!(result.success);
        // Check that it's using SIMD RK4 method (would need method tracking)
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_rk45_adaptive() {
        // Test on harmonic oscillator: d²y/dt² + y = 0
        // Convert to system: dy₁/dt = y₂, dy₂/dt = -y₁
        let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { arr1(&[y[1], -y[0]]) };

        let y0 = arr1(&[1.0, 0.0]); // y(0) = 1, dy/dt(0) = 0
        let t_span = [0.0, std::f64::consts::PI]; // Half period
        let opts = ODEOptions {
            atol: 1e-8,
            rtol: 1e-8,
            h0: Some(0.1),
            ..Default::default()
        };

        let result = simd_rk45_method(f, t_span, y0, opts).expect("Operation failed");

        // At t = π, exact solution is y₁ = -1, y₂ = 0
        let final_y = result.y.last().expect("Operation failed");
        assert_relative_eq!(final_y[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(final_y[1], 0.0, epsilon = 1e-6);
        assert!(result.success);
        // Check that it's using SIMD RK45 method (would need method tracking)
    }
}
