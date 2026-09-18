//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    BSplineTrajectory, BoundaryConditions, CostFunction, DynamicsModel, KeepOutZone,
    ShootingResult, SqpStepResult, Trajectory, TrajectoryKnot, TrustRegionConfig,
    TrustRegionResult,
};

/// Dot product of two equal-length slices.
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
/// Euclidean norm of a slice.
pub(super) fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
/// Element-wise a + b.
pub(super) fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
/// Element-wise a - b.
pub(super) fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
/// Scalar multiply.
pub(super) fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}
/// Linear interpolation between two vectors.
pub(super) fn vec_lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x + (y - x) * t)
        .collect()
}
/// Multiply an n x n row-major matrix by a vector of length n.
pub(super) fn mat_vec(mat: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += mat[i * n + j] * x[j];
        }
    }
    y
}
/// Multiply two n x n row-major matrices.
#[cfg(test)]
pub(super) fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}
/// Transpose an n x n row-major matrix.
pub(super) fn mat_transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}
/// Identity n x n matrix (row-major).
#[cfg(test)]
pub(super) fn mat_eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}
/// Solve a small n x n linear system Ax = b via Gaussian elimination with
/// partial pivoting. Returns `None` if singular.
#[cfg(test)]
pub(super) fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-15 {
            return None;
        }
        if pivot != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, pivot * (n + 1) + j);
            }
        }
        let diag = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / diag;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }
    Some(x)
}
/// Perform single-shooting forward simulation with given controls.
///
/// Integrates the dynamics from `x0` over `num_steps` steps using RK4.
pub fn direct_shooting(
    model: &DynamicsModel,
    x0: &[f64],
    controls: &[Vec<f64>],
    t0: f64,
    dt: f64,
    cost: &CostFunction,
) -> ShootingResult {
    let num_steps = controls.len();
    let mut traj = Trajectory::new(model.n_state, model.n_control);
    let mut x = x0.to_vec();
    let mut t = t0;
    let mut total_cost = 0.0;
    traj.knots.push(TrajectoryKnot {
        time: t,
        state: x.clone(),
        control: if controls.is_empty() {
            vec![0.0; model.n_control]
        } else {
            controls[0].clone()
        },
    });
    for i in 0..num_steps {
        let u = &controls[i];
        total_cost += cost.running_cost(&x, u) * dt;
        x = model.rk4_step(&x, u, t, dt);
        t += dt;
        traj.knots.push(TrajectoryKnot {
            time: t,
            state: x.clone(),
            control: if i + 1 < num_steps {
                controls[i + 1].clone()
            } else {
                vec![0.0; model.n_control]
            },
        });
    }
    total_cost += cost.terminal_cost(&x);
    ShootingResult {
        trajectory: traj,
        final_state: x,
        cost: total_cost,
        final_error: 0.0,
    }
}
/// Perform multiple shooting with `num_segments` segments.
///
/// Returns defects at segment boundaries and the trajectory.
pub fn multiple_shooting(
    model: &DynamicsModel,
    segment_states: &[Vec<f64>],
    controls: &[Vec<f64>],
    t0: f64,
    dt: f64,
    steps_per_segment: usize,
) -> (Vec<Vec<f64>>, Trajectory) {
    let num_segments = segment_states.len() - 1;
    let mut defects = Vec::with_capacity(num_segments);
    let mut traj = Trajectory::new(model.n_state, model.n_control);
    for seg in 0..num_segments {
        let mut x = segment_states[seg].clone();
        let mut t = t0 + (seg * steps_per_segment) as f64 * dt;
        for step in 0..steps_per_segment {
            let ctrl_idx = seg * steps_per_segment + step;
            let u = if ctrl_idx < controls.len() {
                &controls[ctrl_idx]
            } else {
                &controls[controls.len() - 1]
            };
            traj.knots.push(TrajectoryKnot {
                time: t,
                state: x.clone(),
                control: u.clone(),
            });
            x = model.rk4_step(&x, u, t, dt);
            t += dt;
        }
        let defect = vec_sub(&x, &segment_states[seg + 1]);
        defects.push(defect);
    }
    let final_t = t0 + (num_segments * steps_per_segment) as f64 * dt;
    traj.knots.push(TrajectoryKnot {
        time: final_t,
        state: segment_states[num_segments].clone(),
        control: vec![0.0; model.n_control],
    });
    (defects, traj)
}
/// Defect constraints for trapezoidal transcription.
///
/// For each segment \[k, k+1\], the defect is:
///   d_k = x_{k+1} - x_k - (h/2)*(f_k + f_{k+1})
///
/// Returns a vector of defects (one per segment, each of length n_state).
pub fn trapezoidal_defects(model: &DynamicsModel, trajectory: &Trajectory) -> Vec<Vec<f64>> {
    let n = trajectory.knots.len();
    if n < 2 {
        return Vec::new();
    }
    let mut defects = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let xk = &trajectory.knots[k].state;
        let uk = &trajectory.knots[k].control;
        let tk = trajectory.knots[k].time;
        let xk1 = &trajectory.knots[k + 1].state;
        let uk1 = &trajectory.knots[k + 1].control;
        let tk1 = trajectory.knots[k + 1].time;
        let h = tk1 - tk;
        let fk = model.eval(xk, uk, tk);
        let fk1 = model.eval(xk1, uk1, tk1);
        let defect: Vec<f64> = (0..model.n_state)
            .map(|i| xk1[i] - xk[i] - 0.5 * h * (fk[i] + fk1[i]))
            .collect();
        defects.push(defect);
    }
    defects
}
/// Maximum defect norm across all segments.
pub fn max_defect_norm(defects: &[Vec<f64>]) -> f64 {
    defects.iter().map(|d| norm(d)).fold(0.0_f64, f64::max)
}
/// Hermite-Simpson collocation defects.
///
/// For each segment \[k, k+1\], uses a cubic Hermite interpolant with midpoint
/// collocation:
///
///   x_c = 0.5*(x_k + x_{k+1}) + (h/8)*(f_k - f_{k+1})
///   d_k = x_{k+1} - x_k - (h/6)*(f_k + 4*f_c + f_{k+1})
///
/// where f_c = f(x_c, u_c, t_c) and u_c = 0.5*(u_k + u_{k+1}).
pub fn hermite_simpson_defects(model: &DynamicsModel, trajectory: &Trajectory) -> Vec<Vec<f64>> {
    let n = trajectory.knots.len();
    if n < 2 {
        return Vec::new();
    }
    let mut defects = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let xk = &trajectory.knots[k].state;
        let uk = &trajectory.knots[k].control;
        let tk = trajectory.knots[k].time;
        let xk1 = &trajectory.knots[k + 1].state;
        let uk1 = &trajectory.knots[k + 1].control;
        let tk1 = trajectory.knots[k + 1].time;
        let h = tk1 - tk;
        let tc = 0.5 * (tk + tk1);
        let fk = model.eval(xk, uk, tk);
        let fk1 = model.eval(xk1, uk1, tk1);
        let xc: Vec<f64> = (0..model.n_state)
            .map(|i| 0.5 * (xk[i] + xk1[i]) + (h / 8.0) * (fk[i] - fk1[i]))
            .collect();
        let uc: Vec<f64> = (0..model.n_control)
            .map(|i| 0.5 * (uk[i] + uk1[i]))
            .collect();
        let fc = model.eval(&xc, &uc, tc);
        let defect: Vec<f64> = (0..model.n_state)
            .map(|i| xk1[i] - xk[i] - (h / 6.0) * (fk[i] + 4.0 * fc[i] + fk1[i]))
            .collect();
        defects.push(defect);
    }
    defects
}
/// Backward sweep costate estimation using the adjoint method.
///
/// Given a trajectory, cost function, and dynamics model, computes the
/// costate (lambda) at each knot by integrating backward:
///
///   d(lambda)/dt = -(df/dx)^T * lambda - dL/dx
///
/// Terminal condition: lambda(T) = dPhi/dx(x_T) = 2*Qf*x_T.
pub fn adjoint_costate(
    model: &DynamicsModel,
    trajectory: &Trajectory,
    cost: &CostFunction,
) -> Vec<Vec<f64>> {
    let n_knots = trajectory.knots.len();
    let ns = model.n_state;
    if n_knots == 0 {
        return Vec::new();
    }
    let mut costates = vec![vec![0.0; ns]; n_knots];
    let xf = &trajectory.knots[n_knots - 1].state;
    let qf_xf = mat_vec(&cost.qf_matrix, xf, ns);
    costates[n_knots - 1] = vec_scale(&qf_xf, 2.0);
    for k in (0..n_knots - 1).rev() {
        let xk = &trajectory.knots[k].state;
        let uk = &trajectory.knots[k].control;
        let tk = trajectory.knots[k].time;
        let dt = trajectory.knots[k + 1].time - tk;
        let a_mat = model.state_jacobian(xk, uk, tk, 1e-7);
        let q_x = mat_vec(&cost.q_matrix, xk, ns);
        let dl_dx = vec_scale(&q_x, 2.0);
        let at = mat_transpose(&a_mat, ns);
        let at_lam = mat_vec(&at, &costates[k + 1], ns);
        let rhs = vec_add(&at_lam, &dl_dx);
        costates[k] = vec_add(&costates[k + 1], &vec_scale(&rhs, dt));
    }
    costates
}
/// Relative step of the central finite differences used by [`sqp_step`].
///
/// An entry `u` is perturbed by `GRADIENT_FD_STEP * max(1, |u|)` in both directions.
const GRADIENT_FD_STEP: f64 = 1e-5;
/// Quadratic-penalty objective of a trajectory.
///
/// ```text
/// J_pen = J(trajectory) + penalty * ( sum_k |d_k|^2 + |x_N - x_f|^2 )
/// ```
///
/// `J` is [`Trajectory::evaluate_cost`], `d_k` are the [`trapezoidal_defects`] of the
/// trajectory and `x_f` is the final state of `boundary` (the last term is absent when the
/// final state is free). Squaring the violations makes the objective smooth in the
/// controls, which the max-norm exact penalty is not, so finite-difference gradients of it
/// are meaningful. [`sqp_step`] decreases this merit function on the forward rollout of the
/// controls, so for a trajectory that already is such a rollout (for instance the result of
/// a previous step) it is the value to compare before and after a step.
pub fn penalty_cost(
    model: &DynamicsModel,
    trajectory: &Trajectory,
    cost: &CostFunction,
    boundary: &BoundaryConditions,
    penalty: f64,
) -> f64 {
    let defect_sq: f64 = trapezoidal_defects(model, trajectory)
        .iter()
        .map(|defect| dot(defect, defect))
        .sum();
    let final_err = trajectory
        .knots
        .last()
        .map_or(0.0, |knot| boundary.final_violation(&knot.state));
    trajectory.evaluate_cost(cost) + penalty * (defect_sq + final_err * final_err)
}
/// Check that the dimensions of `trajectory`, `model` and `cost` agree.
///
/// Only the data [`sqp_step`] actually reads is checked: the state of the first knot, the
/// controls of every knot that starts an interval, and the cost dimensions.
fn shapes_consistent(model: &DynamicsModel, cost: &CostFunction, trajectory: &Trajectory) -> bool {
    let intervals = trajectory.knots.len().saturating_sub(1);
    cost.n_state == model.n_state
        && cost.n_control == model.n_control
        && trajectory
            .knots
            .first()
            .is_some_and(|knot| knot.state.len() == model.n_state)
        && trajectory
            .knots
            .iter()
            .take(intervals)
            .all(|knot| knot.control.len() == model.n_control)
}
/// Forward rollout of `controls` from the first knot of `template`.
///
/// Takes one RK4 step per knot interval of `template` with the control held constant over
/// the interval (the discretisation of [`direct_shooting`]), so non-uniform time grids are
/// honoured. The states of `template` after the first knot are ignored. The final knot
/// carries the last control held over, so that the trapezoidal defect of the last segment
/// is not biased by an arbitrary terminal control.
fn shooting_rollout(
    model: &DynamicsModel,
    template: &Trajectory,
    controls: &[Vec<f64>],
) -> Trajectory {
    let mut rolled = Trajectory::new(template.n_state, template.n_control);
    let Some(first) = template.knots.first() else {
        return rolled;
    };
    let mut state = first.state.clone();
    let mut time = first.time;
    for (window, control) in template.knots.windows(2).zip(controls) {
        let [start, end] = window else {
            continue;
        };
        rolled.knots.push(TrajectoryKnot {
            time: start.time,
            state: state.clone(),
            control: control.clone(),
        });
        state = model.rk4_step(&state, control, start.time, end.time - start.time);
        time = end.time;
    }
    let held = controls
        .last()
        .cloned()
        .unwrap_or_else(|| first.control.clone());
    rolled.knots.push(TrajectoryKnot {
        time,
        state,
        control: held,
    });
    rolled
}
/// Central-difference gradient of the penalty objective of the shooting rollout with
/// respect to every entry of `controls`.
///
/// Each entry is perturbed by `+-delta`, the objective is evaluated at both points and the
/// entry is restored to its exact original value before the next entry is touched, so every
/// gradient component is taken at the same, unperturbed control vector.
fn penalty_gradient(
    model: &DynamicsModel,
    template: &Trajectory,
    cost: &CostFunction,
    boundary: &BoundaryConditions,
    penalty: f64,
    controls: &mut [Vec<f64>],
) -> Vec<Vec<f64>> {
    let objective = |candidate: &[Vec<f64>]| {
        let rolled = shooting_rollout(model, template, candidate);
        penalty_cost(model, &rolled, cost, boundary, penalty)
    };
    let mut gradient: Vec<Vec<f64>> = controls.iter().map(|row| vec![0.0; row.len()]).collect();
    for k in 0..controls.len() {
        for j in 0..controls[k].len() {
            let original = controls[k][j];
            let delta = GRADIENT_FD_STEP * original.abs().max(1.0);
            let upper = original + delta;
            let lower = original - delta;
            controls[k][j] = upper;
            let cost_upper = objective(controls);
            controls[k][j] = lower;
            let cost_lower = objective(controls);
            controls[k][j] = original;
            gradient[k][j] = (cost_upper - cost_lower) / (upper - lower);
        }
    }
    gradient
}
/// Result of [`sqp_step`] when there is nothing to optimise.
///
/// The input is returned untouched, no step is applied (`step_size` is `0.0`), nothing is
/// evaluated (`cost` is `0.0`) and the violation is infinite, so the result can never be
/// mistaken for a converged one.
fn unchanged_step(trajectory: &Trajectory) -> SqpStepResult {
    SqpStepResult {
        trajectory: trajectory.clone(),
        cost: 0.0,
        max_violation: f64::INFINITY,
        step_size: 0.0,
        converged: false,
    }
}
/// Perform a single quadratic-penalty gradient step on the controls of a trajectory.
///
/// This is a first-order penalty method, **not** sequential quadratic programming: there
/// is no QP subproblem, no Hessian approximation and no multiplier update. The name is
/// kept for API compatibility.
///
/// The decision variables are the controls of the knots that start an interval (every knot
/// but the last). For a control sequence `u` the merit function is
///
/// ```text
/// merit(u) = penalty_cost(rollout(u))
/// ```
///
/// where `rollout(u)` integrates the dynamics forward from the state of the first knot with
/// one RK4 step per knot interval and `u` held constant over each interval (see
/// [`direct_shooting`]). The states of all later knots of `trajectory` are therefore
/// ignored, and a violation of the final state feeds back into the controls through the
/// dynamics. `penalty` weights the squared defects and the squared final-state error.
///
/// 1. The gradient of `merit` is computed by central differences on the unmodified
///    controls: every entry is perturbed by `+-eps`, `merit` is evaluated at both points and
///    the entry is restored before the next one is touched, so no component is polluted by
///    the update of another.
/// 2. Only then is the step `u -= step_size * gradient` applied to all entries.
///
/// No line search is performed: for a small enough `step_size` the merit decreases, a
/// large one can overshoot. The returned trajectory is `rollout(u_new)` (its final knot
/// carries the last control held over), `cost` is its [`Trajectory::evaluate_cost`] and
/// `max_violation` is the largest trapezoidal defect norm plus the final-state violation.
/// `converged` is set when `max_violation < tol` and the cost changed by less than `tol`
/// relative to the rollout of the input controls.
///
/// The rollout holds each control constant over its interval, whereas the trapezoidal
/// defects interpolate the control linearly between knots. A control that jumps between
/// adjacent intervals therefore contributes a defect proportional to the jump and to the
/// interval length, both to the merit and to `max_violation`.
///
/// # Degenerate input
///
/// A trajectory with fewer than two knots, or with state, control or cost dimensions that
/// disagree with `model`, cannot be optimised. It is returned unchanged with
/// `converged == false`, `step_size == 0.0`, `cost == 0.0` and an infinite `max_violation`.
pub fn sqp_step(
    model: &DynamicsModel,
    trajectory: &Trajectory,
    cost: &CostFunction,
    boundary: &BoundaryConditions,
    step_size: f64,
    penalty: f64,
    tol: f64,
) -> SqpStepResult {
    let n_knots = trajectory.knots.len();
    if n_knots < 2 || !shapes_consistent(model, cost, trajectory) {
        return unchanged_step(trajectory);
    }
    let mut controls: Vec<Vec<f64>> = trajectory
        .knots
        .iter()
        .take(n_knots - 1)
        .map(|knot| knot.control.clone())
        .collect();
    let cost_before = shooting_rollout(model, trajectory, &controls).evaluate_cost(cost);
    let gradient = penalty_gradient(model, trajectory, cost, boundary, penalty, &mut controls);
    for (row, gradient_row) in controls.iter_mut().zip(&gradient) {
        for (entry, slope) in row.iter_mut().zip(gradient_row) {
            *entry -= step_size * slope;
        }
    }
    let updated = shooting_rollout(model, trajectory, &controls);
    let new_cost = updated.evaluate_cost(cost);
    let final_err = updated
        .knots
        .last()
        .map_or(0.0, |knot| boundary.final_violation(&knot.state));
    let max_violation = max_defect_norm(&trapezoidal_defects(model, &updated)) + final_err;
    let converged = max_violation < tol && (new_cost - cost_before).abs() < tol;
    SqpStepResult {
        trajectory: updated,
        cost: new_cost,
        max_violation,
        step_size,
        converged,
    }
}
/// Evaluate a trust region step.
///
/// Given a proposed step `delta_u` (control perturbation), evaluates the
/// actual vs predicted cost reduction and updates the trust region radius.
pub fn trust_region_evaluate(
    config: &TrustRegionConfig,
    current_radius: f64,
    current_cost: f64,
    proposed_cost: f64,
    predicted_reduction: f64,
    step_norm: f64,
) -> TrustRegionResult {
    let actual_reduction = current_cost - proposed_cost;
    let ratio = if predicted_reduction.abs() < 1e-15 {
        0.0
    } else {
        actual_reduction / predicted_reduction
    };
    let accepted = ratio > config.accept_ratio && step_norm <= current_radius * 1.01;
    let new_radius = if ratio > 0.75 && step_norm > 0.9 * current_radius {
        (current_radius * config.expand_factor).min(config.max_radius)
    } else if ratio < 0.25 {
        (current_radius * config.shrink_factor).max(config.min_radius)
    } else {
        current_radius
    };
    TrustRegionResult {
        accepted,
        new_radius,
        actual_reduction,
        predicted_reduction,
        ratio,
    }
}
/// Uniform cubic B-spline basis functions.
///
/// Returns the 4 basis function values at parameter `t` in `[0, 1)`.
pub(super) fn bspline_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    let b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    let b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    let b3 = t3 / 6.0;
    [b0, b1, b2, b3]
}
/// Derivative of uniform cubic B-spline basis functions at parameter `t`.
pub(super) fn bspline_basis_derivative(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let db0 = (-3.0 + 6.0 * t - 3.0 * t2) / 6.0;
    let db1 = (-12.0 * t + 9.0 * t2) / 6.0;
    let db2 = (3.0 + 6.0 * t - 9.0 * t2) / 6.0;
    let db3 = 3.0 * t2 / 6.0;
    [db0, db1, db2, db3]
}
/// Smooth a trajectory using B-spline fitting.
///
/// Extracts positions from the trajectory, fits a B-spline, then re-samples.
pub fn smooth_trajectory(
    trajectory: &Trajectory,
    num_control_points: usize,
    num_output_knots: usize,
) -> Trajectory {
    let waypoints: Vec<Vec<f64>> = trajectory.knots.iter().map(|k| k.state.clone()).collect();
    let spline = BSplineTrajectory::fit_to_waypoints(&waypoints, num_control_points);
    let samples = spline.sample(num_output_knots);
    let t0 = trajectory.knots.first().map(|k| k.time).unwrap_or(0.0);
    let tf = trajectory.knots.last().map(|k| k.time).unwrap_or(1.0);
    let mut new_traj = Trajectory::new(trajectory.n_state, trajectory.n_control);
    for (i, state) in samples.into_iter().enumerate() {
        let alpha = if num_output_knots > 1 {
            i as f64 / (num_output_knots - 1) as f64
        } else {
            0.0
        };
        let t = t0 + alpha * (tf - t0);
        new_traj.knots.push(TrajectoryKnot {
            time: t,
            state,
            control: vec![0.0; trajectory.n_control],
        });
    }
    new_traj
}
/// Scale a trajectory's time grid by a factor (for free final time problems).
///
/// Returns a new trajectory with all times multiplied by `scale_factor`.
pub fn scale_time(trajectory: &Trajectory, scale_factor: f64) -> Trajectory {
    let mut traj = trajectory.clone();
    for knot in &mut traj.knots {
        knot.time *= scale_factor;
    }
    traj
}
/// Gradient of cost with respect to final time (for free final time).
///
/// Approximates dJ/dT by perturbing the time scale.
pub fn free_time_gradient(
    model: &DynamicsModel,
    trajectory: &Trajectory,
    cost: &CostFunction,
    eps: f64,
) -> f64 {
    let duration = trajectory.duration();
    if duration.abs() < 1e-15 {
        return 0.0;
    }
    let scale_plus = (duration + eps) / duration;
    let scale_minus = (duration - eps) / duration;
    let traj_plus = scale_time(trajectory, scale_plus);
    let traj_minus = scale_time(trajectory, scale_minus);
    let cost_plus = traj_plus.evaluate_cost(cost);
    let cost_minus = traj_minus.evaluate_cost(cost);
    let def_plus = max_defect_norm(&trapezoidal_defects(model, &traj_plus));
    let def_minus = max_defect_norm(&trapezoidal_defects(model, &traj_minus));
    let aug_plus = cost_plus + 100.0 * def_plus;
    let aug_minus = cost_minus + 100.0 * def_minus;
    (aug_plus - aug_minus) / (2.0 * eps)
}
/// Compute the total obstacle avoidance penalty for a trajectory.
///
/// Uses a smooth penalty: max(0, radius - distance)^2 for each knot.
pub fn obstacle_penalty(trajectory: &Trajectory, obstacles: &[KeepOutZone]) -> f64 {
    let mut total = 0.0;
    for knot in &trajectory.knots {
        for obs in obstacles {
            let v = obs.violation(&knot.state);
            if v < 0.0 {
                total += v * v;
            }
        }
    }
    total
}
/// Check if the trajectory is obstacle-free.
pub fn is_obstacle_free(trajectory: &Trajectory, obstacles: &[KeepOutZone]) -> bool {
    for knot in &trajectory.knots {
        for obs in obstacles {
            if obs.violation(&knot.state) < 0.0 {
                return false;
            }
        }
    }
    true
}
/// Compute the jerk (third derivative) cost of a trajectory.
///
/// Uses finite differences on the acceleration (which itself is finite-
/// differenced from velocity).
pub fn jerk_cost(trajectory: &Trajectory) -> f64 {
    let n_knots = trajectory.knots.len();
    if n_knots < 4 {
        return 0.0;
    }
    let ns = trajectory.n_state;
    let mut total = 0.0;
    let mut accels = Vec::with_capacity(n_knots - 2);
    for k in 1..n_knots - 1 {
        let dt1 = trajectory.knots[k].time - trajectory.knots[k - 1].time;
        let dt2 = trajectory.knots[k + 1].time - trajectory.knots[k].time;
        let dt_avg = 0.5 * (dt1 + dt2);
        if dt_avg.abs() < 1e-15 {
            accels.push(vec![0.0; ns]);
            continue;
        }
        let acc: Vec<f64> = (0..ns)
            .map(|i| {
                let v1 = (trajectory.knots[k].state[i] - trajectory.knots[k - 1].state[i]) / dt1;
                let v2 = (trajectory.knots[k + 1].state[i] - trajectory.knots[k].state[i]) / dt2;
                (v2 - v1) / dt_avg
            })
            .collect();
        accels.push(acc);
    }
    for k in 1..accels.len() {
        let dt = trajectory.knots[k + 1].time - trajectory.knots[k].time;
        if dt.abs() < 1e-15 {
            continue;
        }
        let jerk_sq: f64 = (0..ns)
            .map(|i| {
                let j = (accels[k][i] - accels[k - 1][i]) / dt;
                j * j
            })
            .sum();
        total += jerk_sq * dt;
    }
    total
}
/// Compute the gradient of total cost w.r.t. all control variables.
///
/// Controls are flattened into a single vector; the gradient is returned in
/// the same layout.
pub fn cost_gradient_controls(
    model: &DynamicsModel,
    x0: &[f64],
    controls_flat: &[f64],
    t0: f64,
    dt: f64,
    n_knots: usize,
    cost: &CostFunction,
    eps: f64,
) -> Vec<f64> {
    let nc = model.n_control;
    let num_controls = (n_knots - 1) * nc;
    let base_controls: Vec<Vec<f64>> = controls_flat.chunks(nc).map(|c| c.to_vec()).collect();
    let base_result = direct_shooting(model, x0, &base_controls, t0, dt, cost);
    let base_cost = base_result.cost;
    let mut grad = vec![0.0; num_controls];
    let mut perturbed = controls_flat.to_vec();
    for i in 0..num_controls {
        perturbed[i] += eps;
        let pert_controls: Vec<Vec<f64>> = perturbed.chunks(nc).map(|c| c.to_vec()).collect();
        let pert_result = direct_shooting(model, x0, &pert_controls, t0, dt, cost);
        grad[i] = (pert_result.cost - base_cost) / eps;
        perturbed[i] = controls_flat[i];
    }
    grad
}
