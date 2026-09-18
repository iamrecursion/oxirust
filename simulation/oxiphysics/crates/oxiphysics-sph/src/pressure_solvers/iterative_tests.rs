//! Behavioural tests of the iterative pressure solvers: the DFSPH passes, the
//! PCISPH loops with a density predictor and the full PCISPH
//! prediction-correction loop.
//!
//! The solvers are run on small cubic particle lattices (spacing `s = 0.02 m`,
//! rest density 1000 kg/m³, smoothing length `h = 1.3 s`) with the exact
//! cubic-spline kernel gradients, so that the neighbour data is real and the
//! iterations have something to converge on: an iteration that does not feed
//! its correction back into the state it evaluates cannot reduce the residual
//! of these configurations.

use super::functions::{pcisph_delta, pcisph_lattice_response};
use super::support::{max_propagating_nan, projected_density_error};
use super::types_advanced::{
    DfsphPassResult, DfsphSolverSimple, PciSph, PcisphSolver, SolverComparison,
};
use super::types_core::{
    AdaptiveSolverSwitch, PcisphIterativeLoop, PcisphSolverSimple, PcisphState, SolverType,
};
use crate::kernel::{CubicSplineKernel, SphKernel};

type Vec3 = [f64; 3];

/// Rest density \[kg/m³\].
const RHO0: f64 = 1000.0;
/// Lattice spacing at rest density \[m\].
const SPACING: f64 = 0.02;
/// Smoothing length: 1.3 lattice spacings (support radius 2.6 spacings).
const H: f64 = 1.3 * SPACING;
/// Time step \[s\].
const DT: f64 = 1.0e-3;

/// Particle mass of the rest lattice \[kg\].
fn particle_mass() -> f64 {
    RHO0 * SPACING * SPACING * SPACING
}

/// `n_side³` lattice points with spacing `spacing`, centred on the origin and
/// ordered `x`-major; the middle point of an odd lattice has index `n/2`.
fn lattice(n_side: usize, spacing: f64) -> Vec<Vec3> {
    let half = (n_side as f64 - 1.0) / 2.0;
    let mut points = Vec::with_capacity(n_side * n_side * n_side);
    for i in 0..n_side {
        for j in 0..n_side {
            for k in 0..n_side {
                points.push([
                    (i as f64 - half) * spacing,
                    (j as f64 - half) * spacing,
                    (k as f64 - half) * spacing,
                ]);
            }
        }
    }
    points
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn norm(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Density summation `ρᵢ = Σⱼ mⱼ W(rᵢⱼ, h)` including the particle itself.
fn summation_densities(positions: &[Vec3], masses: &[f64], h: f64) -> Vec<f64> {
    positions
        .iter()
        .map(|&x_i| {
            positions
                .iter()
                .zip(masses)
                .map(|(&x_j, &m_j)| m_j * CubicSplineKernel.w(distance(x_i, x_j), h))
                .sum()
        })
        .collect()
}

/// All `j ≠ i` closer than the support radius `2h`, by brute force.
fn index_lists(positions: &[Vec3], h: f64) -> Vec<Vec<usize>> {
    positions
        .iter()
        .enumerate()
        .map(|(i, &x_i)| {
            (0..positions.len())
                .filter(|&j| j != i && distance(x_i, positions[j]) < 2.0 * h)
                .collect()
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, rel: f64, what: &str) {
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{what}: {actual} vs {expected}"
    );
}

// ---------------------------------------------------------------------------
// DFSPH
// ---------------------------------------------------------------------------

/// A lattice with everything the DFSPH passes need.
struct Block {
    positions: Vec<Vec3>,
    masses: Vec<f64>,
    densities: Vec<f64>,
    neighbors: Vec<Vec<(usize, f64, Vec3)>>,
}

/// A cubic block of `n_side³` particles whose spacing is `compression` times the
/// rest spacing, with summation densities and exact kernel gradients.
fn block(n_side: usize, compression: f64) -> Block {
    let positions = lattice(n_side, compression * SPACING);
    let masses = vec![particle_mass(); positions.len()];
    let densities = summation_densities(&positions, &masses, H);
    let neighbors = DfsphSolverSimple::neighbor_lists(&positions, H);
    Block {
        positions,
        masses,
        densities,
        neighbors,
    }
}

fn dfsph(omega: f64) -> DfsphSolverSimple {
    DfsphSolverSimple::new(RHO0, omega, 200, 1.0e-6, 1.0e-4)
}

/// The velocity field `v = rate · x`.
fn linear_flow(block: &Block, rate: f64) -> Vec<Vec3> {
    block
        .positions
        .iter()
        .map(|x| [rate * x[0], rate * x[1], rate * x[2]])
        .collect()
}

/// `max_i max(−∇·vᵢ, 0)` \[1/s\]: the quantity the divergence-free pass bounds.
fn compression_rate(solver: &DfsphSolverSimple, block: &Block, velocities: &[Vec3]) -> f64 {
    (0..block.positions.len())
        .map(|i| {
            solver
                .velocity_divergence(
                    i,
                    velocities,
                    &block.masses,
                    &block.densities,
                    &block.neighbors,
                )
                .max(0.0)
        })
        .fold(0.0_f64, f64::max)
}

/// `max_i max(ρᵢ + Δt Dρᵢ/Dt − ρ₀, 0) / ρ₀`: the quantity the density-invariant
/// pass bounds.
fn predicted_density_excess(solver: &DfsphSolverSimple, block: &Block, velocities: &[Vec3]) -> f64 {
    (0..block.positions.len())
        .map(|i| {
            let rate = solver.compute_source_term(i, velocities, &block.masses, &block.neighbors);
            ((block.densities[i] + DT * rate - RHO0) / RHO0).max(0.0)
        })
        .fold(0.0_f64, f64::max)
}

/// `Σᵢ mᵢ (vᵢ − uᵢ)`.
fn momentum_change(masses: &[f64], after: &[Vec3], before: &[Vec3]) -> Vec3 {
    let mut total = [0.0_f64; 3];
    for ((m, a), b) in masses.iter().zip(after).zip(before) {
        for k in 0..3 {
            total[k] += m * (a[k] - b[k]);
        }
    }
    total
}

/// Check that the velocity change of a pass is the standard symmetric SPH
/// pressure acceleration of the returned pressures,
/// `Δv = Δt a`, `a = −Σⱼ mⱼ (pᵢ/ρᵢ² + pⱼ/ρⱼ²) ∇ᵢWᵢⱼ`, computed here
/// independently of the solver.
fn assert_pressures_match_velocity_change(
    block: &Block,
    before: &[Vec3],
    result: &DfsphPassResult,
) {
    for (i, row) in block.neighbors.iter().enumerate() {
        let mut acceleration = [0.0_f64; 3];
        for &(j, _, grad_w) in row {
            let weight = block.masses[j]
                * (result.pressures[i] / block.densities[i].powi(2)
                    + result.pressures[j] / block.densities[j].powi(2));
            for k in 0..3 {
                acceleration[k] -= weight * grad_w[k];
            }
        }
        for k in 0..3 {
            let change = result.velocities[i][k] - before[i][k];
            assert!(
                (change - DT * acceleration[k]).abs() <= 1e-9 * change.abs() + 1e-12,
                "particle {i} component {k}: {change} vs {}",
                DT * acceleration[k]
            );
        }
    }
}

#[test]
fn neighbour_lists_hold_the_exact_kernel_gradients() {
    let block = block(3, 0.9);
    let n = block.positions.len();
    assert_eq!(block.neighbors.len(), n);
    for (i, row) in block.neighbors.iter().enumerate() {
        let expected: Vec<usize> = (0..n)
            .filter(|&j| j != i && distance(block.positions[i], block.positions[j]) < 2.0 * H)
            .collect();
        let listed: Vec<usize> = row.iter().map(|&(j, _, _)| j).collect();
        assert_eq!(listed, expected, "row {i} lists the neighbours in order");
        for &(j, r, grad_w) in row {
            let r_ij = [
                block.positions[i][0] - block.positions[j][0],
                block.positions[i][1] - block.positions[j][1],
                block.positions[i][2] - block.positions[j][2],
            ];
            assert_close(r, norm(r_ij), 1e-14, "stored distance");
            let slope = CubicSplineKernel.grad_w(r, H);
            assert!(slope < 0.0, "the cubic spline decreases inside its support");
            for k in 0..3 {
                assert_close(grad_w[k], slope * r_ij[k] / r, 1e-13, "gradient component");
            }
            // ∇ⱼWⱼᵢ = −∇ᵢWᵢⱼ.
            let mirrored = block.neighbors[j]
                .iter()
                .find(|&&(k, _, _)| k == i)
                .map(|&(_, _, g)| g);
            let mirrored = mirrored.expect("neighbour lists are symmetric");
            for k in 0..3 {
                assert_eq!(mirrored[k], -grad_w[k]);
            }
        }
    }
}

#[test]
fn neighbour_lists_are_empty_for_isolated_particles_and_bad_radii() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    for row in DfsphSolverSimple::neighbor_lists(&positions, H) {
        assert!(row.is_empty());
    }
    for h in [0.0, -1.0, f64::NAN] {
        for row in DfsphSolverSimple::neighbor_lists(&lattice(2, SPACING), h) {
            assert!(row.is_empty(), "h = {h}");
        }
    }
    assert!(DfsphSolverSimple::neighbor_lists(&[], H).is_empty());
}

#[test]
fn alpha_is_the_inverse_diagonal_of_the_stiffness_system() {
    // A unit stiffness at particle i changes the density rate of i by
    // −Δt (|Σ mⱼ∇Wᵢⱼ|² + Σ mⱼ m_i |∇Wᵢⱼ|²)/ρᵢ, which is −Δt/(ρᵢ αᵢ) for equal
    // masses: the Jacobi update with αᵢ removes the violation of particle i
    // exactly when the other particles do not move.
    let solver = dfsph(0.5);
    let block = block(3, 0.9);
    let n = block.positions.len();
    let alphas = solver.compute_alpha(&block.positions, &block.masses, &block.neighbors, H);
    assert_eq!(alphas.len(), n);
    for i in 0..n {
        let mut stiffness = vec![0.0; n];
        stiffness[i] = 1.0;
        let accelerations = solver.stiffness_accelerations(
            &stiffness,
            &block.masses,
            &block.densities,
            &block.neighbors,
        );
        let dv: Vec<Vec3> = accelerations
            .iter()
            .map(|a| [DT * a[0], DT * a[1], DT * a[2]])
            .collect();
        let rate = solver.compute_source_term(i, &dv, &block.masses, &block.neighbors);
        let expected = -DT / (block.densities[i] * alphas[i]);
        assert_close(rate, expected, 1e-10, "diagonal response");
    }
}

#[test]
fn alpha_vanishes_for_particles_without_neighbours() {
    let solver = dfsph(0.5);
    let neighbors: Vec<Vec<(usize, f64, Vec3)>> = (0..3).map(|_| Vec::new()).collect();
    let alphas = solver.compute_alpha(&[[0.0; 3]; 3], &[1.0; 3], &neighbors, H);
    assert_eq!(alphas, vec![0.0; 3]);
}

#[test]
fn stiffness_accelerations_push_neighbours_apart_and_conserve_momentum() {
    let solver = dfsph(0.5);
    let block = block(3, 0.9);
    let n = block.positions.len();
    let centre = n / 2;
    let mut stiffness = vec![0.0; n];
    stiffness[centre] = 2.5;
    let accelerations = solver.stiffness_accelerations(
        &stiffness,
        &block.masses,
        &block.densities,
        &block.neighbors,
    );
    for (j, a) in accelerations.iter().enumerate() {
        if j == centre {
            // The neighbour gradients of the middle particle cancel.
            assert!(norm(*a) < 1e-9 * norm(accelerations[0]));
            continue;
        }
        let outward = [
            block.positions[j][0] - block.positions[centre][0],
            block.positions[j][1] - block.positions[centre][1],
            block.positions[j][2] - block.positions[centre][2],
        ];
        assert!(dot(*a, outward) > 0.0, "particle {j} is pushed inwards");
    }
    // Arbitrary non-negative stiffnesses: total momentum stays zero.
    let stiffness: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7) % 5) as f64).collect();
    let accelerations = solver.stiffness_accelerations(
        &stiffness,
        &block.masses,
        &block.densities,
        &block.neighbors,
    );
    let mut momentum = [0.0_f64; 3];
    let mut scale = 0.0_f64;
    for (a, m) in accelerations.iter().zip(&block.masses) {
        for k in 0..3 {
            momentum[k] += m * a[k];
        }
        scale = scale.max(m * norm(*a));
    }
    assert!(scale > 0.0);
    assert!(norm(momentum) < 1e-12 * scale, "momentum {momentum:?}");
}

#[test]
fn divergence_free_solve_removes_the_compression_of_a_converging_flow() {
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    let initial = compression_rate(&solver, &block, &velocities);
    assert!(initial > 5.0, "the flow must compress, got {initial}");

    let result = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(result.converged);
    assert!(result.residual < solver.tol_divergence);
    assert_eq!(result.iterations, result.residual_history.len());
    assert!(result.iterations > 2, "a real iteration is needed");

    // The history starts at the residual of the input and decreases in
    // every pass.
    assert_close(
        result.residual_history[0],
        initial,
        1e-12,
        "initial residual",
    );
    for pair in result.residual_history.windows(2) {
        assert!(pair[1] < pair[0], "residual rose: {pair:?}");
    }
    // The reported residual is the one of the returned velocities.
    let recomputed = compression_rate(&solver, &block, &result.velocities);
    assert_close(recomputed, result.residual, 1e-9, "recomputed residual");

    assert_pressures_match_velocity_change(&block, &velocities, &result);
    assert!(result.pressures.iter().all(|&p| p >= 0.0 && p.is_finite()));
    assert!(result.pressures.iter().any(|&p| p > 0.0));

    // The pressure pushes the corners of the block outwards and conserves
    // the momentum.
    let corner_change = [
        result.velocities[0][0] - velocities[0][0],
        result.velocities[0][1] - velocities[0][1],
        result.velocities[0][2] - velocities[0][2],
    ];
    assert!(dot(corner_change, block.positions[0]) > 0.0);
    let momentum = momentum_change(&block.masses, &result.velocities, &velocities);
    let scale = block.masses[0] * norm(corner_change);
    assert!(norm(momentum) < 1e-9 * scale, "momentum {momentum:?}");
}

#[test]
fn divergence_free_solve_leaves_expanding_and_uniform_flows_alone() {
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let uniform = vec![[1.5, -0.5, 2.0]; block.positions.len()];
    for velocities in [linear_flow(&block, 5.0), uniform] {
        let result = solver.divergence_free_solve(
            &velocities,
            &block.masses,
            &block.densities,
            &block.neighbors,
            DT,
        );
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.residual, 0.0);
        assert_eq!(result.residual_history, vec![0.0]);
        assert_eq!(result.velocities, velocities);
        assert!(result.pressures.iter().all(|&p| p == 0.0), "no suction");
    }
}

#[test]
fn divergence_free_solve_ignores_expanding_particles_next_to_compressing_ones() {
    // v_x = −c x² has ∇·v = −2 c x: the half x > 0 compresses and the half
    // x < 0 expands.  The pressure of the compressing half only has to remove
    // its compression; the expanding half must not be pulled back.
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let curvature = 140.0;
    let velocities: Vec<Vec3> = block
        .positions
        .iter()
        .map(|x| [-curvature * x[0] * x[0], 0.0, 0.0])
        .collect();
    let initial = compression_rate(&solver, &block, &velocities);
    assert!(initial > 0.5, "the x > 0 half must compress, got {initial}");
    let result = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(result.converged);
    assert!(compression_rate(&solver, &block, &result.velocities) < solver.tol_divergence);
    for (x, p) in block.positions.iter().zip(&result.pressures) {
        assert!(*p >= 0.0);
        if x[0] < -0.5 * SPACING {
            assert_eq!(*p, 0.0, "expanding particle at x = {} has pressure", x[0]);
        }
    }
    assert!(
        result
            .pressures
            .iter()
            .zip(&block.positions)
            .any(|(p, x)| x[0] > 0.5 * SPACING && *p > 0.0),
        "the compressing half needs pressure"
    );
}

#[test]
fn density_invariant_solve_decompresses_a_compressed_lattice() {
    let solver = dfsph(0.5);
    let block = block(3, 0.9);
    let n = block.positions.len();
    let centre = n / 2;
    let velocities = vec![[0.0; 3]; n];
    let initial = predicted_density_excess(&solver, &block, &velocities);
    assert!(
        initial > 0.1,
        "the lattice must be compressed, got {initial}"
    );

    let result = solver.density_invariant_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(result.converged);
    assert!(result.residual < solver.tol_density);
    assert_eq!(result.iterations, result.residual_history.len());
    assert!(result.iterations > 2, "a real iteration is needed");
    assert_close(
        result.residual_history[0],
        initial,
        1e-12,
        "initial residual",
    );
    for pair in result.residual_history.windows(2) {
        assert!(pair[1] < pair[0], "residual rose: {pair:?}");
    }
    assert_pressures_match_velocity_change(&block, &velocities, &result);
    // Independent check: the density predicted with the returned velocities
    // exceeds the rest density by less than the tolerance everywhere.
    let remaining = predicted_density_excess(&solver, &block, &result.velocities);
    assert_close(remaining, result.residual, 1e-9, "recomputed residual");
    assert!(remaining < solver.tol_density);

    // The most compressed particle (the middle one) carries the highest
    // pressure, all pressures are non-negative, and the block expands: the
    // corners move outwards while the total momentum stays zero.
    assert!(result.pressures.iter().all(|&p| p >= 0.0 && p.is_finite()));
    let p_max = result.pressures.iter().cloned().fold(0.0_f64, f64::max);
    assert!(p_max > 0.0);
    assert_eq!(result.pressures[centre], p_max);
    assert!(dot(result.velocities[0], block.positions[0]) > 0.0);
    let momentum = momentum_change(&block.masses, &result.velocities, &velocities);
    let scale = block.masses[0] * norm(result.velocities[0]);
    assert!(norm(momentum) < 1e-9 * scale, "momentum {momentum:?}");
}

#[test]
fn density_invariant_solve_corrects_the_velocities_it_is_given() {
    // A flow that would compress the lattice further needs a stronger pressure
    // than the lattice at rest: the density is predicted from the velocities.
    let solver = dfsph(0.5);
    let block = block(3, 0.9);
    let at_rest = vec![[0.0; 3]; block.positions.len()];
    let converging = linear_flow(&block, -2.0);
    let rest = solver.density_invariant_solve(
        &at_rest,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    let squeezed = solver.density_invariant_solve(
        &converging,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(rest.converged && squeezed.converged);
    let centre = block.positions.len() / 2;
    assert!(
        squeezed.pressures[centre] > rest.pressures[centre],
        "{} vs {}",
        squeezed.pressures[centre],
        rest.pressures[centre]
    );
}

#[test]
fn density_invariant_solve_leaves_a_block_at_rest_spacing_alone() {
    // All densities of a 3x3x3 block at the rest spacing are below the rest
    // density (missing neighbours), so there is nothing to correct.
    let solver = dfsph(0.5);
    let block = block(3, 1.0);
    assert!(block.densities.iter().all(|&rho| rho < RHO0));
    let velocities = vec![[0.0; 3]; block.positions.len()];
    let result = solver.density_invariant_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.pressures.iter().all(|&p| p == 0.0));
    assert_eq!(result.velocities, velocities);
}

#[test]
fn an_overshooting_relaxation_factor_never_returns_a_worse_state() {
    // ω far beyond the stability limit of the Jacobi iteration: the result
    // must still be the best state that was evaluated, never worse than the
    // input, and finite.
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    for omega in [1.0, 1.9, 4.0, 50.0] {
        let solver = dfsph(omega);
        let initial = compression_rate(&solver, &block, &velocities);
        let result = solver.divergence_free_solve(
            &velocities,
            &block.masses,
            &block.densities,
            &block.neighbors,
            DT,
        );
        assert!(result.residual <= initial * (1.0 + 1e-12), "omega {omega}");
        let best = result
            .residual_history
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert_eq!(result.residual, best, "omega {omega}");
        assert!(result.pressures.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!(result.velocities.iter().all(|v| norm(*v).is_finite()));
        assert_close(
            compression_rate(&solver, &block, &result.velocities),
            result.residual,
            1e-9,
            "recomputed residual",
        );
        let dens = solver.density_invariant_solve(
            &vec![[0.0; 3]; block.positions.len()],
            &block.masses,
            &block.densities,
            &block.neighbors,
            DT,
        );
        assert!(dens.residual <= dens.residual_history[0] * (1.0 + 1e-12));
    }
}

#[test]
fn the_solver_stops_at_the_iteration_limit() {
    let mut solver = dfsph(0.05);
    solver.max_iter_density = 4;
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    let result = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(!result.converged);
    assert_eq!(result.iterations, 4);
    assert_eq!(
        result.residual_history.len(),
        5,
        "input plus four corrections"
    );
    assert!(result.residual < result.residual_history[0]);
}

#[test]
fn the_step_wrappers_return_the_pressures_and_passes_of_the_solves() {
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    let solve = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    let (pressures, iterations) = solver.divergence_free_step(
        &block.positions,
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        H,
        DT,
    );
    assert_eq!(pressures, solve.pressures);
    assert_eq!(iterations, solve.iterations);
    assert!(pressures.iter().any(|&p| p > 0.0));

    let at_rest = vec![[0.0; 3]; block.positions.len()];
    let solve = solver.density_invariant_solve(
        &at_rest,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    let (pressures, iterations) = solver.density_invariant_step(
        &block.positions,
        &at_rest,
        &block.masses,
        &block.densities,
        &block.neighbors,
        H,
        DT,
    );
    assert_eq!(pressures, solve.pressures);
    assert_eq!(iterations, solve.iterations);
    assert!(pressures.iter().any(|&p| p > 0.0));
}

#[test]
fn invalid_time_step_or_relaxation_gives_a_neutral_result() {
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    for (omega, dt) in [
        (0.5, 0.0),
        (0.5, -1.0),
        (0.5, f64::NAN),
        (0.5, f64::INFINITY),
        (0.0, DT),
        (-1.0, DT),
        (f64::NAN, DT),
    ] {
        let solver = dfsph(omega);
        for result in [
            solver.divergence_free_solve(
                &velocities,
                &block.masses,
                &block.densities,
                &block.neighbors,
                dt,
            ),
            solver.density_invariant_solve(
                &velocities,
                &block.masses,
                &block.densities,
                &block.neighbors,
                dt,
            ),
        ] {
            assert!(!result.converged, "omega {omega} dt {dt}");
            assert_eq!(result.iterations, 0);
            assert!(result.residual.is_nan());
            assert!(result.residual_history.is_empty());
            assert_eq!(result.velocities, velocities);
            assert!(result.pressures.iter().all(|&p| p == 0.0));
        }
    }
}

#[test]
fn non_finite_velocities_are_reported_instead_of_converging() {
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let mut velocities = linear_flow(&block, -5.0);
    velocities[4] = [f64::NAN, 0.0, 0.0];
    let result = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert!(!result.converged, "NaN must not pass the convergence test");
    assert!(result.residual.is_nan());
    assert_eq!(result.residual_history.len(), 1);
    assert!(result.pressures.iter().all(|&p| p == 0.0));
}

#[test]
fn inconsistent_array_lengths_are_tolerated() {
    // Neighbour entries outside the arrays and missing rows are ignored, as
    // are particles with a vanishing density.
    let solver = dfsph(0.3);
    let block = block(3, 0.9);
    let velocities = linear_flow(&block, -5.0);
    let reference = solver.divergence_free_solve(
        &velocities,
        &block.masses,
        &block.densities,
        &block.neighbors,
        DT,
    );

    let mut padded = block.neighbors.clone();
    for row in padded.iter_mut() {
        row.push((10_000, 0.01, [1.0, 2.0, 3.0]));
    }
    let result =
        solver.divergence_free_solve(&velocities, &block.masses, &block.densities, &padded, DT);
    assert_eq!(result.pressures, reference.pressures);
    assert_eq!(result.velocities, reference.velocities);

    let rows = block.neighbors.len();
    let truncated = &block.neighbors[..rows - 3];
    let result =
        solver.divergence_free_solve(&velocities, &block.masses, &block.densities, truncated, DT);
    assert_eq!(result.pressures.len(), rows);
    assert!(result.pressures.iter().all(|p| p.is_finite()));

    let mut densities = block.densities.clone();
    densities[0] = 0.0;
    let result =
        solver.divergence_free_solve(&velocities, &block.masses, &densities, &block.neighbors, DT);
    assert_eq!(result.pressures[0], 0.0);
    assert!(result.velocities.iter().all(|v| norm(*v).is_finite()));

    let result = solver.divergence_free_solve(
        &velocities,
        &block.masses[..5],
        &block.densities,
        &block.neighbors,
        DT,
    );
    assert_eq!(result.pressures.len(), rows);
    assert!(result.velocities.iter().all(|v| norm(*v).is_finite()));
}

#[test]
fn the_projected_density_error_ignores_expansion_without_pressure() {
    assert_eq!(projected_density_error(-30.0, 0.0), 0.0);
    assert_eq!(projected_density_error(30.0, 0.0), 30.0);
    assert_eq!(projected_density_error(-30.0, 5.0), 30.0);
    assert_eq!(projected_density_error(30.0, 5.0), 30.0);
    assert!(projected_density_error(f64::NAN, 5.0).is_nan());
    assert!(projected_density_error(f64::NAN, 0.0).is_nan());
}

#[test]
fn the_nan_propagating_maximum_does_not_drop_a_nan() {
    assert_eq!(max_propagating_nan([1.0, 3.0, 2.0]), 3.0);
    assert_eq!(max_propagating_nan(std::iter::empty::<f64>()), 0.0);
    assert!(max_propagating_nan([1.0, f64::NAN, 2.0]).is_nan());
    assert!(max_propagating_nan([f64::NAN]).is_nan());
}

// ---------------------------------------------------------------------------
// PCISPH with a density predictor
// ---------------------------------------------------------------------------

/// A predictor with a linear response: `ρ*ᵢ(p) = ρ₀ + eᵢ − a pᵢ`.  With the
/// update `p ← p + δ (ρ* − ρ₀)` the error of particle `i` after `k` passes is
/// `eᵢ (1 − δ a)^k`.
fn linear_predictor(excess: &[f64], slope: f64) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
    move |pressures| {
        excess
            .iter()
            .zip(pressures)
            .map(|(e, p)| RHO0 + e - slope * p)
            .collect()
    }
}

#[test]
fn the_predictor_sees_the_pressures_of_the_previous_pass() {
    let mut solver = PcisphSolverSimple::new(RHO0, 0.5);
    solver.tolerance = 1.0e-3;
    let excess = [80.0, 40.0, -30.0];
    let predictor = linear_predictor(&excess, 1.0);
    let mut seen: Vec<Vec<f64>> = Vec::new();
    let (pressures, iterations) = solver.solve_with_predictor(3, |p| {
        seen.push(p.to_vec());
        predictor(p)
    });
    assert_eq!(seen.len(), iterations, "one prediction per pass");
    assert_eq!(
        seen[0],
        vec![0.0; 3],
        "the first prediction has no pressure"
    );
    // δ a = 0.5: the error of the compressed particles halves in every pass,
    // the pressure approaches e/a, the expanded particle keeps p = 0.
    for (k, p) in seen.iter().enumerate() {
        let factor = 1.0 - 0.5_f64.powi(k as i32);
        assert_close(p[0], 80.0 * factor, 1e-12, "pressure of particle 0");
        assert_close(p[1], 40.0 * factor, 1e-12, "pressure of particle 1");
        assert_eq!(p[2], 0.0);
    }
    // 80 · 0.5^k < 0.001 · 1000 for k ≥ 7: the eighth prediction converges.
    assert_eq!(iterations, 8);
    assert_close(pressures[0], 80.0 * (1.0 - 0.5_f64.powi(7)), 1e-12, "p0");
    assert_close(pressures[1], 40.0 * (1.0 - 0.5_f64.powi(7)), 1e-12, "p1");
    assert_eq!(pressures[2], 0.0);
}

#[test]
fn a_predictor_without_feedback_cannot_converge() {
    let mut solver = PcisphSolverSimple::new(RHO0, 0.1);
    solver.max_iter = 7;
    let mut calls = 0_usize;
    let (pressures, iterations) = solver.solve_with_predictor(2, |_| {
        calls += 1;
        vec![1050.0, 1000.0]
    });
    assert_eq!(iterations, 7);
    assert_eq!(calls, 7);
    assert_close(pressures[0], 7.0 * 0.1 * 50.0, 1e-12, "p grows linearly");
    assert_eq!(pressures[1], 0.0);
}

#[test]
fn an_expanded_particle_without_pressure_is_not_a_violation() {
    let solver = PcisphSolverSimple::new(RHO0, 0.5);
    let (pressures, iterations) = solver.solve_with_predictor(2, |_| vec![600.0, 900.0]);
    assert_eq!(iterations, 1);
    assert_eq!(pressures, vec![0.0, 0.0]);
}

#[test]
fn an_overshooting_update_oscillates_but_a_damped_one_converges() {
    // ρ*(p) = ρ₀ + 100 − 10 p: the update p ← p + δ (ρ* − ρ₀) has the
    // multiplier 1 − 10 δ.  With δ = 0.5 it is −4: the pressure alternates
    // between 50 Pa and 0 and never converges.
    let mut solver = PcisphSolverSimple::new(RHO0, 0.5);
    solver.max_iter = 100;
    let excess = [100.0];
    let (pressures, iterations) = solver.solve_with_predictor(1, linear_predictor(&excess, 10.0));
    // The loop runs to the limit; the pressure stays non-negative.
    assert_eq!(iterations, 100);
    assert!(pressures[0] >= 0.0);

    // With δ = 0.15 the multiplier is −0.5: the pressure oscillates around
    // 10 Pa and contracts.
    let excess = [100.0];
    let solver = PcisphSolverSimple::new(RHO0, 0.15);
    let (pressures, iterations) = solver.solve_with_predictor(1, linear_predictor(&excess, 10.0));
    assert!(iterations < solver.max_iter, "oscillating but convergent");
    assert_close(pressures[0], 10.0, 2e-2, "p = e/a");
}

#[test]
fn a_missing_or_surplus_prediction_counts_as_rest_density() {
    let solver = PcisphSolverSimple::new(RHO0, 0.5);
    let (pressures, _) = solver.solve_with_predictor(3, |_| vec![1100.0]);
    assert_eq!(pressures.len(), 3);
    assert!(pressures[0] > 0.0);
    assert_eq!(pressures[1], 0.0);
    assert_eq!(pressures[2], 0.0);
    let (pressures, _) = solver.solve_with_predictor(1, |_| vec![1100.0, 1200.0, 1300.0]);
    assert_eq!(pressures.len(), 1);
}

#[test]
fn a_non_finite_prediction_stops_the_loop() {
    let solver = PcisphSolverSimple::new(RHO0, 0.5);
    let (pressures, iterations) = solver.solve_with_predictor(1, |_| vec![f64::NAN]);
    assert_eq!(iterations, 1);
    assert_eq!(pressures, vec![0.0]);
}

#[test]
fn solving_from_densities_uses_the_lattice_response_of_delta() {
    let solver = PcisphSolverSimple::new(RHO0, 0.5);
    let densities = [1050.0, 1000.0, 950.0];
    let (pressures, iterations) = solver.solve(&densities);
    // p = δ (ρ − ρ₀) removes the error of the compressed particle in one
    // correction; the expanded one stays free.
    assert_close(pressures[0], 25.0, 1e-12, "p0");
    assert_eq!(pressures[1], 0.0);
    assert_eq!(pressures[2], 0.0);
    assert_eq!(iterations, 2);
    let corrected = pcisph_lattice_response(&densities, &pressures, solver.delta);
    assert_close(corrected[0], RHO0, 1e-12, "corrected density");
    assert_eq!(corrected[2], 950.0);
}

#[test]
fn the_lattice_response_needs_a_positive_delta() {
    assert_eq!(
        pcisph_lattice_response(&[1100.0, 900.0], &[10.0, 20.0], 0.0),
        vec![1100.0, 900.0]
    );
    assert_eq!(
        pcisph_lattice_response(&[1100.0, 900.0], &[10.0, 20.0], f64::NAN),
        vec![1100.0, 900.0]
    );
    assert_eq!(
        pcisph_lattice_response(&[1100.0, 900.0], &[10.0, 20.0], 2.0),
        vec![1095.0, 890.0]
    );
    assert_eq!(
        pcisph_lattice_response(&[1100.0, 900.0], &[10.0], 2.0).len(),
        1
    );
}

#[test]
fn the_iterative_loop_records_a_decreasing_density_error() {
    let looper = PcisphIterativeLoop {
        rho0: RHO0,
        delta: 0.5,
        min_iter: 1,
        max_iter: 100,
        eta: 1.0e-3,
    };
    let excess = [80.0, 40.0, -30.0];
    let (pressures, iterations, diagnostics) =
        looper.solve_with_predictor(3, linear_predictor(&excess, 1.0));
    assert_eq!(iterations, 8);
    assert_eq!(diagnostics.len(), iterations);
    for (k, entry) in diagnostics.history.iter().enumerate() {
        assert_eq!(entry.iteration, k + 1);
        // The expanded particle is not counted, the others halve every pass.
        assert_close(
            entry.max_density_error,
            80.0 * 0.5_f64.powi(k as i32),
            1e-12,
            "max error",
        );
        assert_close(
            entry.mean_density_error,
            (80.0 + 40.0) * 0.5_f64.powi(k as i32) / 3.0,
            1e-12,
            "mean error",
        );
        assert_eq!(entry.converged, k == 7);
    }
    for pair in diagnostics.history.windows(2) {
        assert!(pair[1].max_density_error < pair[0].max_density_error);
        assert!(pair[1].max_pressure > pair[0].max_pressure);
    }
    assert_eq!(diagnostics.convergence_iteration(), Some(8));
    assert!(pressures[0] > 79.0 && pressures[1] > 39.0 && pressures[2] == 0.0);
}

#[test]
fn the_iterative_loop_respects_the_minimum_number_of_passes() {
    let looper = PcisphIterativeLoop {
        rho0: RHO0,
        delta: 0.5,
        min_iter: 5,
        max_iter: 100,
        eta: 1.0e-3,
    };
    let excess = [0.0, 0.0];
    let (pressures, iterations, diagnostics) =
        looper.solve_with_predictor(2, linear_predictor(&excess, 1.0));
    assert_eq!(iterations, 5);
    assert!(diagnostics.history.iter().all(|entry| entry.converged));
    assert_eq!(pressures, vec![0.0, 0.0]);
}

#[test]
fn the_iterative_loop_from_densities_converges_in_two_passes() {
    let looper = PcisphIterativeLoop::new(RHO0, 0.05, 1.0e-3);
    let densities = [1100.0, 900.0, 1000.0];
    let (pressures, iterations, diagnostics) = looper.solve_from_predicted_densities(&densities);
    assert_eq!(iterations, looper.min_iter);
    assert_eq!(diagnostics.len(), 2);
    assert_close(
        diagnostics.history[0].max_density_error,
        100.0,
        1e-12,
        "first error",
    );
    assert!(diagnostics.history[1].max_density_error < 1e-9);
    assert_close(pressures[0], 5.0, 1e-12, "p0");
    assert_eq!(pressures[1], 0.0);
    assert_eq!(pressures[2], 0.0);
}

#[test]
fn the_comparison_reports_the_error_left_after_the_pcisph_correction() {
    let comparison = SolverComparison::new(RHO0, 100.0);
    let result = comparison.run_pcisph(&[1100.0, 1000.0, 900.0]);
    assert_eq!(result.solver_type, SolverType::Pcisph);
    assert_eq!(result.iterations, 2);
    assert_close(
        result.pressures[0],
        comparison.pcisph_delta * 100.0,
        1e-12,
        "p0",
    );
    assert_eq!(result.pressures[1], 0.0);
    assert_eq!(result.pressures[2], 0.0);
    // The compression is removed; the 100 kg/m³ of expansion remains.
    assert_close(result.max_density_error, 100.0, 1e-12, "max error");
    assert_close(result.avg_density_error, 100.0 / 3.0, 1e-12, "mean error");
}

#[test]
fn the_adaptive_switch_corrects_a_strongly_compressed_state() {
    let mut switch = AdaptiveSolverSwitch::new(RHO0, 100.0, 0.05);
    let pressures = switch.compute_pressures(&[1000.0, 1200.0, 800.0]);
    assert_eq!(switch.current_solver, SolverType::Pcisph);
    assert_close(pressures[1], switch.delta * 200.0, 1e-12, "p1");
    assert_eq!(pressures[0], 0.0);
    assert_eq!(pressures[2], 0.0);
}

// ---------------------------------------------------------------------------
// The PCISPH stiffness delta
// ---------------------------------------------------------------------------

/// `Σⱼ |∇Wᵢⱼ|²` over a full lattice of spacing `s`, summed independently of
/// the library with the kernel trait.
fn lattice_gradient_square_sum(spacing: f64, h: f64) -> f64 {
    let reach = (2.0 * h / spacing).ceil() as i32 + 1;
    let mut total = 0.0;
    for ix in -reach..=reach {
        for iy in -reach..=reach {
            for iz in -reach..=reach {
                let r = spacing * ((ix * ix + iy * iy + iz * iz) as f64).sqrt();
                if r > 0.0 && r < 2.0 * h {
                    total += CubicSplineKernel.grad_w(r, h).powi(2);
                }
            }
        }
    }
    total
}

#[test]
fn pcisph_delta_is_the_inverse_gradient_square_sum_of_the_prototype_lattice() {
    // The gradients of the full lattice cancel, so δ = ρ₀² / (2 Δt² m² Σ|∇W|²).
    let mass = particle_mass();
    let expected =
        RHO0 * RHO0 / (2.0 * DT * DT * mass * mass * lattice_gradient_square_sum(SPACING, H));
    let delta = pcisph_delta(RHO0, H, DT, mass);
    assert!(delta > 0.0 && delta.is_finite());
    assert_close(delta, expected, 1e-12, "delta");
}

#[test]
fn pcisph_delta_scales_with_the_inverse_square_of_the_time_step() {
    let mass = particle_mass();
    let base = pcisph_delta(RHO0, H, DT, mass);
    assert_close(
        pcisph_delta(RHO0, H, 2.0 * DT, mass),
        base / 4.0,
        1e-12,
        "2 dt",
    );
    assert_close(
        pcisph_delta(RHO0, H, 0.5 * DT, mass),
        base * 4.0,
        1e-12,
        "dt / 2",
    );
}

#[test]
fn pcisph_delta_depends_on_the_lattice_and_not_on_the_density_scale() {
    // Doubling density and mass keeps the lattice spacing; δ ∝ ρ₀²/m² is unchanged.
    let mass = particle_mass();
    let base = pcisph_delta(RHO0, H, DT, mass);
    assert_close(
        pcisph_delta(2.0 * RHO0, H, DT, 2.0 * mass),
        base,
        1e-12,
        "scaled",
    );
    // A larger kernel at the same lattice spreads the gradients and raises δ.
    assert!(pcisph_delta(RHO0, 2.0 * H, DT, mass) > base);
}

#[test]
fn pcisph_delta_matches_the_reference_solver_for_a_half_h_lattice() {
    // crate::pcisph builds its prototype with the lattice spacing h/2.
    let h = 0.1;
    let mass = RHO0 * (h / 2.0_f64).powi(3);
    let mut reference = crate::pcisph::PcisphSolver::new(RHO0, h);
    reference.compute_delta(DT, mass);
    assert!(reference.delta > 0.0);
    assert_close(
        pcisph_delta(RHO0, h, DT, mass),
        reference.delta,
        1e-12,
        "delta",
    );
}

#[test]
fn pcisph_delta_is_zero_for_invalid_input() {
    let mass = particle_mass();
    assert_eq!(pcisph_delta(0.0, H, DT, mass), 0.0);
    assert_eq!(pcisph_delta(RHO0, 0.0, DT, mass), 0.0);
    assert_eq!(pcisph_delta(RHO0, H, 0.0, mass), 0.0);
    assert_eq!(pcisph_delta(RHO0, H, DT, 0.0), 0.0);
    assert_eq!(pcisph_delta(-RHO0, H, DT, mass), 0.0);
    assert_eq!(pcisph_delta(RHO0, H, f64::NAN, mass), 0.0);
    assert_eq!(pcisph_delta(RHO0, f64::INFINITY, DT, mass), 0.0);
    // A support radius of more than 64 spacings is refused.
    assert_eq!(pcisph_delta(RHO0, 40.0 * SPACING, DT, mass), 0.0);
    // A kernel that reaches no lattice neighbour has no gradient sum.
    assert_eq!(pcisph_delta(RHO0, 0.1 * SPACING, DT, mass), 0.0);
}

#[test]
fn the_pcisph_structs_use_the_lattice_delta() {
    let mass = particle_mass();
    let delta = pcisph_delta(RHO0, H, DT, mass);
    assert_eq!(PciSph::new(RHO0, H, DT, mass).delta, delta);
    assert_eq!(PciSph::compute_delta(H, RHO0, DT, mass), delta);
    let mut solver = PcisphSolver::new(RHO0, H, DT, mass, 50, 1.0e-3);
    assert_eq!(solver.delta, delta);
    solver.update_delta(H, 2.0 * DT, mass);
    assert_close(solver.delta, delta / 4.0, 1e-12, "updated delta");
}

#[test]
fn the_own_pressure_response_is_half_the_paper_slope() {
    // Pressurise only the middle particle of a lattice with a full
    // neighbourhood and move all particles by their pressure accelerations:
    // the density of the middle particle drops by Δρ = P / (2 δ) (ρ₀/ρ_c)².
    // δ of the paper assumes that the neighbours carry the pressure as well,
    // which doubles the neighbour accelerations, so the paper's slope is P/δ:
    // the response to the own pressure alone is half of it.  This ties the
    // force, the density prediction and δ together.
    let positions = lattice(7, SPACING);
    let n = positions.len();
    let centre = n / 2;
    let masses = vec![particle_mass(); n];
    let neighbors = index_lists(&positions, H);
    let solver = PcisphSolver::new(RHO0, H, DT, particle_mass(), 50, 1.0e-3);

    let mut state = PcisphState::new(n);
    state.predicted_positions = positions.clone();
    solver.predict_densities(&masses, &neighbors, H, &mut state);
    let rho_centre = state.predicted_densities[centre];
    assert!((rho_centre / RHO0 - 1.0).abs() < 0.01, "ρ_c = {rho_centre}");
    let reference = state.predicted_densities.clone();

    let pressure = 100.0;
    let mut pressures = vec![0.0; n];
    pressures[centre] = pressure;
    let accelerations = solver.pressure_accelerations(&masses, &pressures, &neighbors, H, &state);
    // The gradients around the middle particle cancel: it does not move itself,
    // its neighbours are pushed away.
    let neighbour = centre + 1;
    assert!(neighbors[centre].contains(&neighbour));
    assert!(norm(accelerations[neighbour]) > 0.1);
    assert!(norm(accelerations[centre]) < 1e-9 * norm(accelerations[neighbour]));
    for (x, a) in state.predicted_positions.iter_mut().zip(&accelerations) {
        for k in 0..3 {
            x[k] += DT * DT * a[k];
        }
    }
    solver.predict_densities(&masses, &neighbors, H, &mut state);
    let drop = reference[centre] - state.predicted_densities[centre];
    let expected = pressure / (2.0 * solver.delta) * (RHO0 / rho_centre).powi(2);
    assert_close(
        drop,
        expected,
        2e-3,
        "density drop of the pressurised particle",
    );
}

// ---------------------------------------------------------------------------
// The full PCISPH loop
// ---------------------------------------------------------------------------

struct Cluster {
    positions: Vec<Vec3>,
    masses: Vec<f64>,
    neighbors: Vec<Vec<usize>>,
}

fn cluster(n_side: usize, compression: f64) -> Cluster {
    let positions = lattice(n_side, compression * SPACING);
    let masses = vec![particle_mass(); positions.len()];
    let neighbors = index_lists(&positions, H);
    Cluster {
        positions,
        masses,
        neighbors,
    }
}

fn pcisph(max_iter: usize) -> PcisphSolver {
    PcisphSolver::new(RHO0, H, DT, particle_mass(), max_iter, 1.0e-3)
}

#[test]
fn predict_correct_decompresses_a_compressed_block() {
    let cluster = cluster(3, 0.95);
    let n = cluster.positions.len();
    let centre = n / 2;
    let velocities = vec![[0.0; 3]; n];
    let forces = vec![[0.0; 3]; n];
    let solver = pcisph(100);
    assert!(solver.delta > 0.0);
    let result = solver.predict_correct(
        &cluster.positions,
        &velocities,
        &cluster.masses,
        &forces,
        &cluster.neighbors,
        H,
        DT,
    );

    // The block starts 8 % over-dense in the middle and ends below the
    // threshold, with a smaller error after every pass.
    let history = &result.density_error_history;
    assert!(history[0] > 50.0, "initial error {}", history[0]);
    assert!(result.converged);
    assert!(*history.last().expect("history") < solver.eta * RHO0);
    assert_eq!(result.iterations, history.len());
    assert!(result.iterations > 3, "a real iteration is needed");
    for pair in history.windows(2) {
        assert!(pair[1] < pair[0], "density error rose: {pair:?}");
    }

    // Pressures are non-negative and largest in the most compressed particle;
    // the pressure accelerations conserve the momentum.
    assert!(result.pressures.iter().all(|&p| p >= 0.0 && p.is_finite()));
    let p_max = result.pressures.iter().cloned().fold(0.0_f64, f64::max);
    assert!(p_max > 0.0);
    assert_eq!(result.pressures[centre], p_max);
    let mut momentum = [0.0_f64; 3];
    let mut scale = 0.0_f64;
    for (a, m) in result.accelerations.iter().zip(&cluster.masses) {
        for k in 0..3 {
            momentum[k] += m * a[k];
        }
        scale = scale.max(m * norm(*a));
    }
    assert!(scale > 0.0);
    assert!(norm(momentum) < 1e-9 * scale, "momentum {momentum:?}");

    // The returned state is the prediction with these accelerations: the block
    // has expanded relative to the start, v* = Δt a, x* = x + Δt v*, and its
    // densities are those of x*.
    let state = &result.state;
    assert_eq!(state.len(), n);
    for i in 0..n {
        for k in 0..3 {
            let v = DT * result.accelerations[i][k];
            assert!((state.predicted_velocities[i][k] - v).abs() <= 1e-12 * v.abs() + 1e-300);
            assert!(
                (state.predicted_positions[i][k] - (cluster.positions[i][k] + DT * v)).abs()
                    <= 1e-15
            );
        }
    }
    assert!(
        norm(state.predicted_positions[0]) > norm(cluster.positions[0]),
        "the corner must move outwards"
    );
    let densities = summation_densities(&state.predicted_positions, &cluster.masses, H);
    for (rho, expected) in state.predicted_densities.iter().zip(&densities) {
        assert_close(*rho, *expected, 1e-12, "predicted density");
    }
    assert!(state.pressure_corrections.iter().any(|&c| c != 0.0));
}

#[test]
fn predict_correct_converges_for_a_strongly_compressed_block() {
    // At a 10 % smaller spacing (22 % over-dense) the max-norm error is not
    // monotone in every pass, but the loop converges within its budget.
    let cluster = cluster(3, 0.9);
    let n = cluster.positions.len();
    let solver = pcisph(100);
    let result = solver.predict_correct(
        &cluster.positions,
        &vec![[0.0; 3]; n],
        &cluster.masses,
        &vec![[0.0; 3]; n],
        &cluster.neighbors,
        H,
        DT,
    );
    let history = &result.density_error_history;
    assert!(history[0] > 150.0, "initial error {}", history[0]);
    assert!(result.converged);
    assert!(*history.last().expect("history") < solver.eta * RHO0);
    assert!(result.iterations < 100);
    assert!(history[history.len() - 1] < history[0] / 100.0);
    assert!(result.pressures.iter().all(|&p| p >= 0.0 && p.is_finite()));
    assert_eq!(
        result.pressures[n / 2],
        result.pressures.iter().cloned().fold(0.0_f64, f64::max)
    );
}

#[test]
fn predict_correct_step_returns_the_pressures_and_passes() {
    let cluster = cluster(3, 0.9);
    let n = cluster.positions.len();
    let velocities = vec![[0.0; 3]; n];
    let forces = vec![[0.0; 3]; n];
    let solver = pcisph(100);
    let full = solver.predict_correct(
        &cluster.positions,
        &velocities,
        &cluster.masses,
        &forces,
        &cluster.neighbors,
        H,
        DT,
    );
    let (pressures, iterations) = solver.predict_correct_step(
        &cluster.positions,
        &velocities,
        &cluster.masses,
        &forces,
        &cluster.neighbors,
        H,
        DT,
    );
    assert_eq!(pressures, full.pressures);
    assert_eq!(iterations, full.iterations);
    assert!((2..=100).contains(&iterations));
    assert!(pressures.iter().any(|&p| p > 0.0));
}

#[test]
fn a_block_at_rest_spacing_needs_no_pressure() {
    // The 3x3x3 block at the rest spacing is under-dense everywhere (its
    // surface particles miss neighbours): nothing to correct.
    let cluster = cluster(3, 1.0);
    let n = cluster.positions.len();
    let result = pcisph(100).predict_correct(
        &cluster.positions,
        &vec![[0.0; 3]; n],
        &cluster.masses,
        &vec![[0.0; 3]; n],
        &cluster.neighbors,
        H,
        DT,
    );
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.density_error_history, vec![0.0]);
    assert!(result.pressures.iter().all(|&p| p == 0.0));
    assert!(result.accelerations.iter().all(|a| *a == [0.0; 3]));
}

#[test]
fn predict_correct_applies_the_non_pressure_forces_to_the_prediction() {
    // A uniform force F on every particle moves the block rigidly; pressures
    // stay zero and the prediction is x + Δt (v + Δt F/m).
    let cluster = cluster(3, 1.0);
    let n = cluster.positions.len();
    let force = [0.0, -particle_mass() * 9.81, 0.0];
    let velocities = vec![[0.5, 0.0, 0.0]; n];
    let result = pcisph(20).predict_correct(
        &cluster.positions,
        &velocities,
        &cluster.masses,
        &vec![force; n],
        &cluster.neighbors,
        H,
        DT,
    );
    assert!(result.pressures.iter().all(|&p| p == 0.0));
    for (x, x0) in result
        .state
        .predicted_positions
        .iter()
        .zip(&cluster.positions)
    {
        assert_close(x[0], x0[0] + DT * 0.5, 1e-12, "x");
        assert_close(x[1], x0[1] - DT * DT * 9.81, 1e-12, "y");
        assert_close(x[2], x0[2], 1e-12, "z");
    }
}

#[test]
fn predict_correct_balances_a_compressing_force_with_pressure() {
    // A force towards the middle, F = −m k x, squeezes a block at the rest
    // spacing by 10 % within the step (Δt² k = 0.1): the prediction without
    // pressure is over-dense, the loop builds the pressure that carries the
    // squeeze and the predicted block ends up less compressed.
    let cluster = cluster(3, 1.0);
    let n = cluster.positions.len();
    let confinement = 0.1 / (DT * DT);
    let forces: Vec<Vec3> = cluster
        .positions
        .iter()
        .map(|x| {
            let m = particle_mass();
            [
                -m * confinement * x[0],
                -m * confinement * x[1],
                -m * confinement * x[2],
            ]
        })
        .collect();
    let velocities = vec![[0.0; 3]; n];
    let solver = pcisph(100);
    let result = solver.predict_correct(
        &cluster.positions,
        &velocities,
        &cluster.masses,
        &forces,
        &cluster.neighbors,
        H,
        DT,
    );
    let history = &result.density_error_history;
    assert!(
        history[0] > 100.0,
        "the squeeze must over-compress: {}",
        history[0]
    );
    assert!(result.converged);
    assert!(*history.last().expect("history") < solver.eta * RHO0);
    assert!(result.pressures.iter().all(|&p| p >= 0.0));
    assert!(result.pressures[n / 2] > 0.0);
    // The block is squeezed (the corner moved inwards) but less than by the
    // confining force alone.
    let corner = cluster.positions[0];
    let squeezed_alone = [0.9 * corner[0], 0.9 * corner[1], 0.9 * corner[2]];
    let with_pressure = result.state.predicted_positions[0];
    assert!(norm(with_pressure) < norm(corner));
    assert!(norm(with_pressure) > norm(squeezed_alone));
}

#[test]
fn predict_correct_gives_an_empty_result_for_inconsistent_arrays() {
    let cluster = cluster(2, 1.0);
    let n = cluster.positions.len();
    let solver = pcisph(10);
    let velocities = vec![[0.0; 3]; n];
    let forces = vec![[0.0; 3]; n];
    for (v, m, f) in [
        (&velocities[..n - 1], &cluster.masses[..], &forces[..]),
        (&velocities[..], &cluster.masses[..n - 1], &forces[..]),
        (&velocities[..], &cluster.masses[..], &forces[..n - 1]),
    ] {
        let result = solver.predict_correct(&cluster.positions, v, m, f, &cluster.neighbors, H, DT);
        assert_eq!(result.pressures, vec![0.0; n]);
        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert!(result.density_error_history.is_empty());
    }
}

#[test]
fn predict_correct_does_nothing_without_a_usable_delta_time_step_or_kernel() {
    let cluster = cluster(3, 0.9);
    let n = cluster.positions.len();
    let velocities = vec![[0.0; 3]; n];
    let forces = vec![[0.0; 3]; n];
    let mut no_delta = pcisph(10);
    no_delta.delta = 0.0;
    let valid = pcisph(10);
    for (solver, h, dt) in [
        (&no_delta, H, DT),
        (&valid, H, 0.0),
        (&valid, H, f64::NAN),
        (&valid, 0.0, DT),
        (&valid, f64::NAN, DT),
    ] {
        let result = solver.predict_correct(
            &cluster.positions,
            &velocities,
            &cluster.masses,
            &forces,
            &cluster.neighbors,
            h,
            dt,
        );
        assert!(result.pressures.iter().all(|&p| p == 0.0));
        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert!(result.density_error_history.len() <= 1);
    }
}

#[test]
fn predict_correct_of_an_empty_system_converges_immediately() {
    let result = pcisph(10).predict_correct(&[], &[], &[], &[], &[], H, DT);
    assert!(result.pressures.is_empty());
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
}

#[test]
fn predict_correct_stops_at_the_iteration_limit() {
    let cluster = cluster(3, 0.9);
    let n = cluster.positions.len();
    let result = pcisph(2).predict_correct(
        &cluster.positions,
        &vec![[0.0; 3]; n],
        &cluster.masses,
        &vec![[0.0; 3]; n],
        &cluster.neighbors,
        H,
        DT,
    );
    assert!(!result.converged);
    assert_eq!(result.iterations, 2);
    assert_eq!(
        result.density_error_history.len(),
        3,
        "start plus two corrections"
    );
}

#[test]
fn isolated_particles_never_receive_pressure() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let masses = [particle_mass(); 2];
    let neighbors: Vec<Vec<usize>> = (0..2).map(|_| Vec::new()).collect();
    let result = pcisph(10).predict_correct(
        &positions,
        &[[0.0; 3]; 2],
        &masses,
        &[[0.0; 3]; 2],
        &neighbors,
        H,
        DT,
    );
    // ρ* = m W(0, h) is far below the rest density.
    assert!(result.converged);
    assert_eq!(result.pressures, vec![0.0, 0.0]);
    assert_eq!(result.iterations, 1);
}
