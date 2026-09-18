//! Tests of the Dirichlet-Neumann ALE-FSI solver (F1104).

use super::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;

const RHO_G: f64 = 1000.0 * 9.81;

/// Closed container `[0, length] × [0, height]`: bottom = interface, no-slip side
/// walls, open top (the two upper corners are wall nodes).
fn container(nx: usize, ny: usize, length: f64, height: f64) -> AleFsiSolver {
    let mut solver = AleFsiSolver::rectangular_channel(nx, ny, length, height).unwrap();
    solver.inlet_nodes.clear();
    solver.wall_nodes = (1..=ny)
        .flat_map(|j| {
            [
                AleFsiSolver::grid_node(nx, 0, j),
                AleFsiSolver::grid_node(nx, nx, j),
            ]
        })
        .collect();
    solver
}

/// Light, critically damped plate: areal mass 1 kg/m², foundation stiffness
/// `kappa`, damping `alpha` (1/s).
fn set_plate(solver: &mut AleFsiSolver, kappa: f64, alpha: f64) {
    solver.struct_rho = 1.0;
    solver.thickness = 1.0;
    solver.struct_stiffness = kappa;
    solver.struct_damping = alpha;
}

/// Tributary length of interface node `k` of a uniform interface with `n` nodes.
fn tributary(k: usize, n: usize, length: f64) -> f64 {
    let ds = length / (n - 1) as f64;
    if k == 0 || k + 1 == n { 0.5 * ds } else { ds }
}

/// Outward flux ∮ v·n ds through every boundary edge of the current mesh:
/// (edge nodes, flux) pairs.
fn boundary_fluxes(solver: &AleFsiSolver) -> Vec<([usize; 2], f64)> {
    let mut edges: BTreeMap<(usize, usize), EdgeOwners> = BTreeMap::new();
    for e in &solver.elements {
        for (u, v, o) in [(e[0], e[1], e[2]), (e[1], e[2], e[0]), (e[2], e[0], e[1])] {
            edges
                .entry((u.min(v), u.max(v)))
                .or_default()
                .push((u, v, o));
        }
    }
    let mut out = Vec::new();
    for owners in edges.values().filter(|o| o.len() == 1) {
        let (u, v, o) = owners[0];
        let (a, b, c) = (
            solver.nodes[u].spatial_pos,
            solver.nodes[v].spatial_pos,
            solver.nodes[o].spatial_pos,
        );
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let mut normal = [dy, -dx];
        if normal[0] * (c[0] - a[0]) + normal[1] * (c[1] - a[1]) > 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        // `normal` has the length of the edge, so the product is the flux.
        let (va, vb) = (
            solver.nodes[u].fluid_velocity,
            solver.nodes[v].fluid_velocity,
        );
        let mean = [0.5 * (va[0] + vb[0]), 0.5 * (va[1] + vb[1])];
        out.push(([u, v], mean[0] * normal[0] + mean[1] * normal[1]));
    }
    out
}

fn debug_state(solver: &AleFsiSolver) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}",
        solver.nodes, solver.struct_velocity, solver.struct_acceleration, solver.interface_force
    )
}

// ---------------------------------------------------------------------------
// Fluid problem
// ---------------------------------------------------------------------------

#[test]
fn rest_pressure_is_hydrostatic_on_the_current_mesh() {
    // Gravity on, fluid at rest, open top at p₀ = 0: the Stokes solve (not a hard-coded
    // formula) must return p = ρ g (H − y) and a plate force equal to the weight.
    let mut solver = container(6, 4, 1.5, 1.0);
    solver.gravity = [0.0, -9.81];
    solver.solve_fluid(0.0).unwrap();
    for node in &solver.nodes {
        let exact = RHO_G * (1.0 - node.spatial_pos[1]);
        assert!(
            (node.pressure - exact).abs() < 1e-8 * RHO_G,
            "p = {}, exact {exact}",
            node.pressure
        );
        assert!(node.fluid_velocity[0].abs() < 1e-6 && node.fluid_velocity[1].abs() < 1e-6);
    }
    let force = solver.total_fluid_force();
    assert!(
        (force[1] + RHO_G * 1.5).abs() < 1e-8 * RHO_G * 1.5,
        "{force:?}"
    );
    assert!(force[0].abs() < 1e-6, "{force:?}");

    // Push the plate 0.1 m away from the fluid: the column is 1.1 m deep, the
    // ALE mesh follows and the pressure on the plate is ρ g · 1.1.
    for &node in &solver.interface.node_indices.clone() {
        solver.nodes[node].displacement = [0.0, -0.1];
    }
    solver.solve_fluid(0.0).unwrap();
    for node in &solver.nodes {
        let exact = RHO_G * (1.0 - node.spatial_pos[1]);
        assert!((node.pressure - exact).abs() < 1e-8 * RHO_G);
    }
    let plate = solver.interface.node_indices[3];
    assert!((solver.nodes[plate].spatial_pos[1] + 0.1).abs() < 1e-14);
    assert!((solver.nodes[plate].pressure - 1.1 * RHO_G).abs() < 1e-8 * RHO_G);
    assert!((solver.total_fluid_force()[1] + 1.1 * RHO_G * 1.5).abs() < 1e-7 * RHO_G);
}

#[test]
fn positive_pressure_pushes_the_plate_away_from_the_fluid() {
    // The interface normal points into the fluid (+y) and the fluid sits above the
    // plate, so a positive pressure must load the plate along −n.
    let mut solver = container(5, 3, 1.0, 1.0);
    solver.outlet_pressure = 250.0;
    solver.solve_fluid(0.0).unwrap();
    let n = solver.interface.node_indices.len();
    for k in 0..n {
        let f = solver.interface_force[k];
        let expected = -250.0 * tributary(k, n, 1.0);
        assert!((f[1] - expected).abs() < 1e-9 * 250.0, "node {k}: {f:?}");
        let normal_component = f[1] * solver.interface.normals[k][1];
        assert!(
            normal_component < 0.0,
            "the force must point against the normal"
        );
        if k == 0 || k + 1 == n {
            // The corner nodes also carry half of the pressure force on the adjacent
            // side-wall edge (length 1/3); only the normal component loads the plate.
            let wall = 250.0 * (1.0 / 3.0) / 2.0;
            let sign = if k == 0 { -1.0 } else { 1.0 };
            assert!(
                (f[0] - sign * wall).abs() < 1e-9 * 250.0,
                "corner {k}: {f:?}"
            );
        } else {
            assert!(
                f[0].abs() < 1e-9 * 250.0,
                "no tangential force at rest: {f:?}"
            );
        }
    }
    assert!((solver.total_fluid_force()[1] + 250.0).abs() < 1e-9 * 250.0);
    assert!(
        solver.total_fluid_force()[0].abs() < 1e-9 * 250.0,
        "the corner forces cancel"
    );
    // The plate-side pressure is the applied one everywhere.
    assert!(
        solver
            .nodes
            .iter()
            .all(|node| (node.pressure - 250.0).abs() < 1e-8)
    );
}

#[test]
fn moving_plate_conserves_mass_through_the_open_boundary() {
    // Plate rising with 0.05 m/s: the discrete continuity equation makes the net
    // boundary flux vanish exactly; the volume the plate displaces leaves through
    // the open top.
    let mut solver = container(8, 4, 2.0, 1.0);
    let n = solver.interface.node_indices.len();
    solver.struct_velocity = vec![0.05; n];
    solver.mu = 0.1;
    solver.solve_fluid(0.0).unwrap();
    let fluxes = boundary_fluxes(&solver);
    let net: f64 = fluxes.iter().map(|(_, q)| q).sum();
    let interface_flux: f64 = fluxes
        .iter()
        .filter(|(nodes, _)| {
            nodes
                .iter()
                .all(|node| solver.interface.node_indices.contains(node))
        })
        .map(|(_, q)| q)
        .sum();
    // Outward flux through the plate is −w·L (the fluid is pushed away from it).
    assert!(
        (interface_flux + 0.05 * 2.0).abs() < 1e-12,
        "interface flux {interface_flux}"
    );
    assert!(net.abs() < 1e-10, "net boundary flux {net}");
    let open_flux: f64 = fluxes
        .iter()
        .filter(|(nodes, _)| {
            nodes
                .iter()
                .any(|node| solver.nodes[*node].spatial_pos[1] >= 1.0 - 1e-12)
        })
        .map(|(_, q)| q)
        .sum();
    assert!(
        open_flux > 0.0,
        "fluid leaves through the open top: {open_flux}"
    );
    // The moving plate drags the fluid up, no-slip walls hold it back: the force
    // on the plate resists the motion (points against the velocity, i.e. down).
    assert!(solver.total_fluid_force()[1] < 0.0);
}

/// Pressure slope between x = 1 and x = 2.5 on the interface of a channel with
/// plug inflow, normalised by the Poiseuille value 12·μ·Q/H³ for the flux Q that
/// actually enters through the discrete inlet.
fn poiseuille_slope_ratio(nx: usize, ny: usize) -> f64 {
    let (length, height, mu) = (4.0, 1.0, 1.0);
    let mut solver = AleFsiSolver::rectangular_channel(nx, ny, length, height).unwrap();
    solver.mu = mu;
    solver.solve_fluid(1.0).unwrap();
    // Plug inflow on the inlet nodes only (the corner nodes are no-slip): Q = H (ny−1)/ny.
    let q = height * (ny - 1) as f64 / ny as f64;
    let fluxes = boundary_fluxes(&solver);
    let outflow: f64 = fluxes
        .iter()
        .filter(|(nodes, _)| {
            nodes
                .iter()
                .all(|n| solver.nodes[*n].spatial_pos[0] > length - 1e-9)
        })
        .map(|(_, f)| f)
        .sum();
    assert!(
        (outflow - q).abs() < 1e-10,
        "outflow {outflow} vs inflow {q}"
    );
    let node_at = |x: f64| {
        let i = (x / length * nx as f64).round() as usize;
        solver.interface.node_indices[i]
    };
    let (p1, p2) = (
        solver.nodes[node_at(1.0)].pressure,
        solver.nodes[node_at(2.5)].pressure,
    );
    (p1 - p2) / 1.5 / (12.0 * mu * q / height.powi(3))
}

#[test]
fn channel_pressure_gradient_converges_to_the_poiseuille_value() {
    // Stokes flow between plates: dp/dx = −12 μ Q / H³ (parabolic profile).  P1
    // elements underestimate the gradient by O(1/ny²).
    let coarse = poiseuille_slope_ratio(16, 4);
    let fine = poiseuille_slope_ratio(24, 6);
    assert!((coarse - 1.0).abs() < 0.07, "ny = 4: ratio {coarse}");
    assert!((fine - 1.0).abs() < 0.04, "ny = 6: ratio {fine}");
    assert!(
        (fine - 1.0).abs() < (coarse - 1.0).abs(),
        "no convergence: {coarse} → {fine}"
    );
    assert!(
        coarse < 1.0 && fine < 1.0,
        "the pressure gradient converges from below"
    );
}

#[test]
fn channel_pressure_is_proportional_to_the_viscosity() {
    let mut a = AleFsiSolver::rectangular_channel(12, 4, 3.0, 1.0).unwrap();
    a.mu = 0.5;
    a.solve_fluid(2.0).unwrap();
    let mut b = a.clone();
    b.mu = 1.5;
    b.solve_fluid(2.0).unwrap();
    let scale = a
        .nodes
        .iter()
        .fold(0.0_f64, |m, node| m.max(node.pressure.abs()));
    assert!(scale > 1.0);
    for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
        assert!((nb.pressure - 3.0 * na.pressure).abs() < 1e-9 * 3.0 * scale);
        assert!((nb.fluid_velocity[0] - na.fluid_velocity[0]).abs() < 1e-9);
    }
    // Viscous drag on the (rigid) plate: the flow pushes it downstream (+x).
    assert!(a.total_fluid_force()[0] > 0.0);
    assert!((b.total_fluid_force()[0] - 3.0 * a.total_fluid_force()[0]).abs() < 1e-8);
}

// ---------------------------------------------------------------------------
// Coupled time stepping
// ---------------------------------------------------------------------------

#[test]
fn quiescent_fluid_loads_the_plate_to_the_static_deflection_p_over_k() {
    // Applied pressure P on the open top, fluid at rest: the structure must settle at
    // δ = −P/κ (per node: k·δ = −P·A with k = κ·ds, A = ds).
    let (p0, kappa) = (1000.0, 1.0e6);
    let mut solver = container(6, 3, 1.0, 1.0);
    solver.outlet_pressure = p0;
    set_plate(&mut solver, kappa, 2000.0);
    solver.tol = 1e-13;
    let mut max_iterations = 0;
    for step in 0..60 {
        let report = solver.step(1e-3, 0.0).unwrap();
        assert!(report.converged, "step {step}: {report:?}");
        max_iterations = max_iterations.max(report.iterations);
    }
    assert!(
        (1..=8).contains(&max_iterations),
        "iterations {max_iterations}"
    );
    for (k, delta) in solver.interface_deflection().iter().enumerate() {
        assert!(
            (delta + p0 / kappa).abs() < 1e-9 * (p0 / kappa),
            "node {k}: δ = {delta:e}, static {:e}",
            -p0 / kappa
        );
    }
    for k in 0..solver.interface.node_indices.len() {
        assert!(solver.struct_velocity[k].abs() < 1e-8);
        assert!(solver.struct_acceleration[k].abs() < 1e-3);
    }
    assert!((solver.total_fluid_force()[1] + p0).abs() < 1e-8 * p0);
    assert!(
        solver
            .nodes
            .iter()
            .all(|n| (n.pressure - p0).abs() < 1e-6 * p0)
    );
    // The moved mesh is consistent with the deflection.
    let plate = solver.interface.node_indices[2];
    assert!((solver.nodes[plate].spatial_pos[1] + p0 / kappa).abs() < 1e-12);
    assert!(solver.max_deflection() > 0.99 * p0 / kappa);
}

#[test]
fn hydrostatic_head_deflection_includes_the_geometric_coupling() {
    // The column above the plate is H − δ deep once the plate sank by |δ|, so the
    // static state of κ δ = −ρ g (H − δ) is δ = −ρ g H / (κ − ρ g), not −ρ g H / κ.
    let kappa = 1.0e6;
    let mut solver = container(6, 3, 1.0, 1.0);
    solver.gravity = [0.0, -9.81];
    set_plate(&mut solver, kappa, 2000.0);
    solver.tol = 1e-13;
    for _ in 0..60 {
        let report = solver.step(1e-3, 0.0).unwrap();
        assert!(report.converged, "{report:?}");
    }
    let expected = -RHO_G / (kappa - RHO_G);
    let naive = -RHO_G / kappa;
    assert!(
        (expected - naive).abs() > 5e-5 * naive.abs(),
        "the test must see the coupling"
    );
    for delta in solver.interface_deflection() {
        assert!(
            (delta - expected).abs() < 1e-8 * expected.abs(),
            "δ = {delta:e}, coupled static {expected:e}, naive {naive:e}"
        );
    }
}

#[test]
fn transient_follows_the_analytic_damped_single_dof_response() {
    // Negligible fluid viscosity: the plate is a damped oscillator under the constant
    // force −P·ds.  Newmark with the dynamic-equilibrium initial acceleration must
    // follow u(t) = −(P/κ)·[1 − e^{−ζωt}(cos ω_d t + ζω/ω_d sin ω_d t)]; a wrong
    // initial acceleration (a₀ = 0) is off by about 3 % of the static value.
    let (p0, kappa, alpha) = (50.0, 1.0e4, 10.0);
    let mut solver = container(6, 3, 1.0, 1.0);
    solver.outlet_pressure = p0;
    solver.mu = 1e-9;
    set_plate(&mut solver, kappa, alpha);
    let omega = kappa.sqrt();
    let zeta = alpha / (2.0 * omega);
    let omega_d = omega * (1.0 - zeta * zeta).sqrt();
    let dt = 2.0 * PI / omega / 100.0;
    let static_deflection = -p0 / kappa;
    let mut worst = 0.0_f64;
    for step in 1..=200 {
        solver.step(dt, 0.0).unwrap();
        let t = step as f64 * dt;
        let exact = static_deflection
            * (1.0
                - (-zeta * omega * t).exp()
                    * ((omega_d * t).cos() + zeta * omega / omega_d * (omega_d * t).sin()));
        let delta = solver.interface_deflection()[3];
        worst = worst.max((delta - exact).abs() / static_deflection.abs());
    }
    assert!(
        worst < 1e-2,
        "worst deviation {worst:.4} of the static deflection"
    );
    // The very first acceleration is the force over the mass: a₀ = F/m.
    let mut fresh = container(6, 3, 1.0, 1.0);
    fresh.outlet_pressure = p0;
    fresh.mu = 1e-9;
    set_plate(&mut fresh, kappa, alpha);
    fresh.step(dt / 1000.0, 0.0).unwrap();
    let a0 = fresh.struct_acceleration[3];
    assert!(
        (a0 + p0).abs() < 1e-2 * p0,
        "a = {a0}, expected −P/(ρh) = {}",
        -p0
    );
}

#[test]
fn coupled_solution_satisfies_the_structural_equilibrium_and_the_kinematic_condition() {
    // Strongly coupled case (heavy viscous drag, light plate).  After a converged
    // step the Newmark equilibrium m a + c v + k u = F must hold with the interface
    // force of the fluid, and the fluid must move with the interface.
    let mut solver = container(6, 3, 1.0, 1.0);
    solver.outlet_pressure = 10.0;
    solver.mu = 20.0;
    set_plate(&mut solver, 1.0e3, 5.0);
    solver.struct_rho = 1e-2;
    solver.tol = 1e-12;
    solver.max_iter = 120;
    let dt = 0.01;
    let before = solver.interface_deflection();
    let report = solver.step(dt, 0.0).unwrap();
    assert!(report.converged, "{report:?}");
    let n = solver.interface.node_indices.len();
    let after = solver.interface_deflection();
    for (k, &delta_before) in before.iter().enumerate() {
        let ds = tributary(k, n, 1.0);
        let (m, c, stiffness) = (1e-2 * ds, 5.0 * 1e-2 * ds, 1.0e3 * ds);
        let delta = after[k];
        let normal_force = solver.interface_force[k][1];
        let residual =
            m * solver.struct_acceleration[k] + c * solver.struct_velocity[k] + stiffness * delta
                - normal_force;
        assert!(
            residual.abs() < 1e-6 * normal_force.abs().max(1e-3),
            "node {k}: equilibrium residual {residual:e}, F = {normal_force:e}"
        );
        // Kinematic condition: the fluid at the interface moves with the plate.
        let node = &solver.nodes[solver.interface.node_indices[k]];
        let plate_velocity = (delta - delta_before) / dt;
        assert!(
            (node.fluid_velocity[1] - plate_velocity).abs() < 1e-6 * plate_velocity.abs().max(1e-6),
            "node {k}: fluid {} vs plate {plate_velocity}",
            node.fluid_velocity[1]
        );
        assert!(
            (node.mesh_velocity[1] - plate_velocity).abs() < 1e-9 * plate_velocity.abs().max(1.0)
        );
        assert!(node.fluid_velocity[0].abs() < 1e-9 && node.mesh_velocity[0].abs() < 1e-12);
    }
}

#[test]
fn aitken_relaxation_handles_strong_viscous_coupling() {
    // Heavy viscous drag on a light plate makes the plain fixed-point iteration
    // diverge; Aitken's dynamic relaxation shrinks ω to a stable value.
    let mut solver = container(6, 3, 1.0, 1.0);
    solver.outlet_pressure = 10.0;
    solver.mu = 20.0;
    set_plate(&mut solver, 1.0e3, 0.0);
    solver.struct_rho = 1e-2;
    solver.max_iter = 120;
    let report = solver.step(0.01, 0.0).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(
        report.iterations > 5,
        "strong coupling needs many iterations: {report:?}"
    );
    assert!(report.omega < 0.2, "ω must have been reduced: {report:?}");
    assert!(report.interface_residual < 1e-8 * (1.0 + solver.max_deflection()));
    // With too few iterations the step is reported as not converged but still
    // committed (no error): the last iterate is kept.
    let mut limited = container(6, 3, 1.0, 1.0);
    limited.outlet_pressure = 10.0;
    limited.mu = 20.0;
    set_plate(&mut limited, 1.0e3, 0.0);
    limited.struct_rho = 1e-2;
    limited.max_iter = 3;
    let capped = limited.step(0.01, 0.0).unwrap();
    assert_eq!(capped.iterations, 3);
    assert!(!capped.converged);
    assert!(capped.interface_residual > report.interface_residual);
    assert!(limited.nodes.iter().all(|n| n.pressure.is_finite()));
}

#[test]
fn dissipative_newmark_parameters_do_not_change_the_static_state() {
    let (p0, kappa) = (500.0, 1.0e6);
    let mut solver = container(5, 3, 1.0, 1.0);
    solver.outlet_pressure = p0;
    set_plate(&mut solver, kappa, 2000.0);
    solver.newmark_gamma = 0.7;
    solver.newmark_beta = 0.25 * (0.7 + 0.5) * (0.7 + 0.5);
    solver.tol = 1e-13;
    for _ in 0..80 {
        assert!(solver.step(1e-3, 0.0).unwrap().converged);
    }
    for delta in solver.interface_deflection() {
        assert!(
            (delta + p0 / kappa).abs() < 1e-8 * (p0 / kappa),
            "{delta:e}"
        );
    }
}

#[test]
fn stepping_is_deterministic() {
    let build = || {
        let mut solver = container(5, 3, 1.0, 1.0);
        solver.outlet_pressure = 120.0;
        solver.mu = 0.5;
        set_plate(&mut solver, 5.0e4, 40.0);
        solver
    };
    let (mut a, mut b) = (build(), build());
    for _ in 0..5 {
        a.step(2e-3, 0.0).unwrap();
        b.step(2e-3, 0.0).unwrap();
    }
    assert_eq!(debug_state(&a), debug_state(&b));
}

// ---------------------------------------------------------------------------
// Configuration and error handling
// ---------------------------------------------------------------------------

#[test]
fn rectangular_channel_builds_a_consistent_mesh() {
    let solver = AleFsiSolver::rectangular_channel(5, 4, 2.0, 1.0).unwrap();
    assert_eq!(solver.nodes.len(), 6 * 5);
    assert_eq!(solver.elements.len(), 2 * 5 * 4);
    assert_eq!(solver.interface.node_indices.len(), 6);
    assert!(solver.interface.normals.iter().all(|n| *n == [0.0, 1.0]));
    assert_eq!(solver.inlet_nodes.len(), 3);
    assert_eq!(solver.wall_nodes.len(), 6);
    let area: f64 = solver
        .elements
        .iter()
        .map(|e| {
            let p: Vec<[f64; 2]> = e.iter().map(|&n| solver.nodes[n].material_pos).collect();
            0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
        })
        .sum();
    assert!(
        (area - 2.0).abs() < 1e-12,
        "counter-clockwise elements tile the channel: {area}"
    );
    solver.validate().unwrap();
    assert!(AleFsiSolver::rectangular_channel(0, 3, 1.0, 1.0).is_err());
    assert!(AleFsiSolver::rectangular_channel(3, 0, 1.0, 1.0).is_err());
    assert!(AleFsiSolver::rectangular_channel(3, 3, -1.0, 1.0).is_err());
    assert!(AleFsiSolver::rectangular_channel(3, 3, 1.0, f64::NAN).is_err());
}

#[test]
fn invalid_configurations_are_rejected_and_leave_the_solver_untouched() {
    type Corrupt = Box<dyn Fn(&mut AleFsiSolver)>;
    let cases: Vec<(&str, Corrupt)> = vec![
        ("viscosity", Box::new(|s| s.mu = 0.0)),
        (
            "negative stiffness",
            Box::new(|s| s.struct_stiffness = -1.0),
        ),
        ("zero density", Box::new(|s| s.struct_rho = 0.0)),
        ("Newmark gamma", Box::new(|s| s.newmark_gamma = 0.4)),
        ("Newmark beta", Box::new(|s| s.newmark_beta = 0.0)),
        ("omega zero", Box::new(|s| s.omega = 0.0)),
        ("omega above one", Box::new(|s| s.omega = 1.5)),
        ("tolerance", Box::new(|s| s.tol = 0.0)),
        ("max_iter", Box::new(|s| s.max_iter = 0)),
        ("gravity", Box::new(|s| s.gravity = [f64::NAN, 0.0])),
        ("no elements", Box::new(|s| s.elements.clear())),
        (
            "element out of range",
            Box::new(|s| s.elements[0][2] = 10_000),
        ),
        (
            "repeated node",
            Box::new(|s| s.elements[1][1] = s.elements[1][0]),
        ),
        (
            "inlet out of range",
            Box::new(|s| s.inlet_nodes.push(10_000)),
        ),
        ("wall out of range", Box::new(|s| s.wall_nodes.push(10_000))),
        (
            "orphan node",
            Box::new(|s| {
                s.nodes.push(AleNode::new(5.0, 5.0));
            }),
        ),
        (
            "duplicate interface node",
            Box::new(|s| {
                let first = s.interface.node_indices[0];
                s.interface.node_indices[1] = first;
            }),
        ),
        (
            "interior interface node",
            Box::new(|s| {
                let interior = AleFsiSolver::grid_node(4, 2, 2);
                s.interface.node_indices[2] = interior;
            }),
        ),
        (
            "zero normal",
            Box::new(|s| s.interface.normals[2] = [0.0, 0.0]),
        ),
        (
            "short velocity state",
            Box::new(|s| {
                s.struct_velocity.pop();
            }),
        ),
        (
            "tangential displacement",
            Box::new(|s| {
                let node = s.interface.node_indices[2];
                s.nodes[node].displacement = [0.05, 0.01];
            }),
        ),
        (
            "no open boundary",
            Box::new(|s| {
                s.wall_nodes = (0..s.nodes.len()).collect();
            }),
        ),
    ];
    for (name, corrupt) in cases {
        let mut solver = AleFsiSolver::rectangular_channel(4, 3, 1.0, 1.0).unwrap();
        corrupt(&mut solver);
        let before = debug_state(&solver);
        let result = solver.step(1e-3, 0.5);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "{name}: expected InvalidInput, got {result:?}"
        );
        assert_eq!(
            debug_state(&solver),
            before,
            "{name}: the solver changed on error"
        );
        assert!(solver.solve_fluid(0.5).is_err(), "{name}: solve_fluid");
        assert!(
            solver.validate().is_err() || name == "tangential displacement",
            "{name}"
        );
    }
}

#[test]
fn invalid_step_arguments_are_rejected() {
    let mut solver = AleFsiSolver::rectangular_channel(4, 3, 1.0, 1.0).unwrap();
    let before = debug_state(&solver);
    for dt in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(solver.step(dt, 0.0), Err(Error::InvalidInput(_))),
            "dt = {dt}"
        );
    }
    assert!(solver.step(1e-3, f64::NAN).is_err());
    assert!(solver.solve_fluid(f64::INFINITY).is_err());
    assert_eq!(debug_state(&solver), before);
    // A solver created without a fluid mesh explains what is missing.
    let nodes: Vec<AleNode> = (0..5).map(|i| AleNode::new(i as f64 * 0.1, 0.0)).collect();
    let interface = AleInterface::new((0..5).collect(), vec![[0.0, 1.0]; 5]);
    let mut bare = AleFsiSolver::new(nodes, interface);
    match bare.step(1e-3, 1.0) {
        Err(Error::InvalidInput(message)) => assert!(message.contains("no elements"), "{message}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    assert_eq!(bare.total_fluid_force(), [0.0, 0.0]);
}

#[test]
fn tangled_mesh_is_reported_and_leaves_the_state_unchanged() {
    let mut solver = container(4, 3, 1.0, 1.0);
    // Push the plate through the lid: the harmonic extension cannot keep the
    // elements valid.
    for &node in &solver.interface.node_indices.clone() {
        solver.nodes[node].displacement = [0.0, 3.0];
    }
    let before = debug_state(&solver);
    match solver.solve_fluid(0.0) {
        Err(Error::General(message)) => assert!(message.contains("tangled"), "{message}"),
        other => panic!("expected a tangled-mesh error, got {other:?}"),
    }
    assert!(solver.step(1e-3, 0.0).is_err());
    assert_eq!(debug_state(&solver), before);
}

#[test]
fn deflection_sign_and_mesh_follow_the_interface_normal() {
    // Positive deflection moves the plate into the fluid (along the normal).
    let mut solver = container(4, 3, 1.0, 1.0);
    for &node in &solver.interface.node_indices.clone() {
        solver.nodes[node].displacement = [0.0, 0.2];
    }
    solver.solve_fluid(0.0).unwrap();
    assert!(
        solver
            .interface_deflection()
            .iter()
            .all(|d| (d - 0.2).abs() < 1e-14)
    );
    let plate = solver.interface.node_indices[2];
    assert!((solver.nodes[plate].spatial_pos[1] - 0.2).abs() < 1e-14);
    // Interior nodes move less than the plate and in the same direction.
    for (index, node) in solver.nodes.iter().enumerate() {
        let dy = node.displacement[1];
        assert!(
            (-1e-12..=0.2 + 1e-12).contains(&dy),
            "node {index}: dy = {dy}"
        );
        assert!(node.displacement[0].abs() < 1e-12);
        let moved = [
            node.material_pos[0] + node.displacement[0],
            node.material_pos[1] + node.displacement[1],
        ];
        assert!(
            (moved[0] - node.spatial_pos[0]).abs() < 1e-15
                && (moved[1] - node.spatial_pos[1]).abs() < 1e-15
        );
    }
    let above = solver.nodes[AleFsiSolver::grid_node(4, 2, 1)].displacement[1];
    assert!(above > 0.0 && above < 0.2, "{above}");
    assert!((solver.max_deflection() - 0.2).abs() < 1e-14);
    // A compressed column (plate into the fluid) raises the pressure on the plate.
    solver.outlet_pressure = 100.0;
    solver.gravity = [0.0, -9.81];
    solver.solve_fluid(0.0).unwrap();
    let depth = 1.0 - 0.2;
    assert!((solver.nodes[plate].pressure - (100.0 + RHO_G * depth)).abs() < 1e-7 * RHO_G);
}

#[test]
fn interface_traction_sign_convention_of_the_pressure_only_helper() {
    // f = −p·n·ds with n from the structure into the fluid: positive pressure pushes
    // the structure away from the fluid.
    let mut nodes = vec![AleNode::new(0.0, 0.0), AleNode::new(1.0, 0.0)];
    nodes[0].pressure = 100.0;
    nodes[1].pressure = 40.0;
    let interface = AleInterface::new(vec![0, 1], vec![[0.0, 1.0]; 2]);
    let traction = interface.compute_traction(&nodes, &[0.5, 0.25]);
    assert!((traction[0][1] + 50.0).abs() < 1e-12 && traction[0][0].abs() < 1e-12);
    assert!((traction[1][1] + 10.0).abs() < 1e-12 && traction[1][0].abs() < 1e-12);
}
