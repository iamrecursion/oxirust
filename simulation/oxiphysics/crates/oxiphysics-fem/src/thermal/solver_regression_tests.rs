//! Regression tests for the 1-D thermal solvers.
//!
//! The implicit, steady-state and Crank-Nicolson solves copied only
//! `K[i][i-1]`, `K[i][i]`, `K[i][i+1]` into Thomas' algorithm.  Any coupling with
//! `|i − j| > 1` (a thermal contact resistance between non-adjacent nodes, a 2-D or
//! 3-D conduction matrix) was silently discarded in the left-hand side while the
//! explicit step and the right-hand sides used the full matrix, so the integrators
//! disagreed.  Every reference below is an independent dense LU solve.

use super::functions::*;
use super::linear_solve::{BandedSystem, half_bandwidths};
use super::types::ThermalMesh1D;
use crate::dense_lu::LuFactor;
use crate::error::{Error, Result};

/// Factorise and solve in one call.
fn solve_system(a: Vec<Vec<f64>>, rhs: &[f64]) -> Result<Vec<f64>> {
    BandedSystem::new(a)?.solve(rhs)
}

/// Deterministic generator (64-bit LCG, high bits).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn signed_unit(&mut self) -> f64 {
        (self.next() % 20_001) as f64 / 10_000.0 - 1.0
    }
}

fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    LuFactor::new(a)
        .expect("reference matrix must be nonsingular")
        .solve(b)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Dense reference of one Dirichlet-constrained solve `A x = b` by row replacement.
fn constrained_dense_solve(a: &[Vec<f64>], b: &[f64], bcs: &[(usize, f64)]) -> Vec<f64> {
    let mut matrix = a.to_vec();
    let mut rhs = b.to_vec();
    for &(node, value) in bcs {
        for entry in matrix[node].iter_mut() {
            *entry = 0.0;
        }
        matrix[node][node] = 1.0;
        rhs[node] = value;
    }
    dense_solve(&matrix, &rhs)
}

/// Graph Laplacian of an `nx × ny` grid with unit conductances (5-point stencil).
fn grid_conduction_matrix(nx: usize, ny: usize) -> Vec<Vec<f64>> {
    let n = nx * ny;
    let mut k = vec![vec![0.0; n]; n];
    for row in 0..ny {
        for col in 0..nx {
            let i = row * nx + col;
            let mut link = |j: usize| {
                k[i][i] += 1.0;
                k[i][j] -= 1.0;
            };
            if col > 0 {
                link(i - 1);
            }
            if col + 1 < nx {
                link(i + 1);
            }
            if row > 0 {
                link(i - nx);
            }
            if row + 1 < ny {
                link(i + nx);
            }
        }
    }
    k
}

/// One dense Crank-Nicolson trajectory, the reference for the banded solver.
fn dense_crank_nicolson(
    k: &[Vec<f64>],
    c: &[f64],
    t0: &[f64],
    f: &[f64],
    bcs: &[(usize, f64)],
    dt: f64,
    steps: usize,
) -> Vec<Vec<f64>> {
    let n = t0.len();
    let mut lhs = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            lhs[i][j] = 0.5 * k[i][j];
        }
        lhs[i][i] += c[i] / dt;
    }
    let mut history = vec![t0.to_vec()];
    for _ in 0..steps {
        let t = &history[history.len() - 1];
        let rhs: Vec<f64> = (0..n)
            .map(|i| {
                let kt: f64 = (0..n).map(|j| k[i][j] * t[j]).sum();
                c[i] / dt * t[i] - 0.5 * kt + f[i]
            })
            .collect();
        let next = constrained_dense_solve(&lhs, &rhs, bcs);
        history.push(next);
    }
    history
}

// ---------------------------------------------------------------------------
// The banded solver itself
// ---------------------------------------------------------------------------

#[test]
fn half_bandwidths_measure_the_non_zero_pattern() {
    let diagonal = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    assert_eq!(half_bandwidths(&diagonal), (0, 0));
    let tridiagonal = vec![
        vec![2.0, -1.0, 0.0, 0.0],
        vec![-1.0, 2.0, -1.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0],
        vec![0.0, 0.0, -1.0, 2.0],
    ];
    assert_eq!(half_bandwidths(&tridiagonal), (1, 1));
    let mut bridged = tridiagonal.clone();
    bridged[0][3] = -0.5;
    bridged[3][0] = -0.5;
    assert_eq!(half_bandwidths(&bridged), (3, 3));
    let mut lower_only = diagonal.clone();
    lower_only[2][0] = 1.0;
    assert_eq!(half_bandwidths(&lower_only), (2, 0));
    assert_eq!(half_bandwidths(&[]), (0, 0));
}

#[test]
fn banded_solve_matches_the_dense_lu_for_random_band_patterns() {
    let mut rng = Lcg(2025);
    for &(n, kl, ku) in &[
        (1, 0, 0),
        (6, 0, 2),
        (7, 2, 0),
        (12, 1, 1),
        (12, 2, 1),
        (15, 3, 2),
        (20, 4, 4),
        (9, 8, 8),
    ] {
        let mut a = vec![vec![0.0; n]; n];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                if i <= j + kl && j <= i + ku {
                    *entry = rng.signed_unit();
                }
            }
        }
        for (i, row) in a.iter_mut().enumerate() {
            // Weak diagonals force row exchanges (and sometimes make Thomas fail) ...
            row[i] *= 0.05;
            row[i] += if i.is_multiple_of(3) { 0.0 } else { 0.4 };
            // ... while the row sum keeps the matrix non-singular and moderately
            // conditioned.
            let row_sum: f64 = row.iter().map(|v| v.abs()).sum();
            row[i] += 0.8 * row_sum * if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        }
        let b: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
        let x = solve_system(a.clone(), &b).expect("banded solve");
        let reference = dense_solve(&a, &b);
        // Random matrices can be ill conditioned: compare relative to the size of
        // the solution, and judge the solve by its backward error (the residual).
        let x_scale = 1.0 + reference.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_abs_diff(&x, &reference) < 1e-7 * x_scale,
            "n={n} kl={kl} ku={ku}: {x:?} vs {reference:?}"
        );
        let residual: f64 = (0..n)
            .map(|i| ((0..n).map(|j| a[i][j] * x[j]).sum::<f64>() - b[i]).abs())
            .fold(0.0, f64::max);
        let a_norm = a
            .iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let x_norm = x.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            residual < 1e-12 * (1.0 + a_norm * x_norm),
            "n={n}: residual {residual:e} for |A| = {a_norm}, |x| = {x_norm}"
        );
    }
}

#[test]
fn the_factorisation_matches_the_matrix_structure() {
    // A conduction-like tridiagonal matrix takes the O(n) Thomas path ...
    let mut chain = vec![vec![0.0; 6]; 6];
    for (i, row) in chain.iter_mut().enumerate() {
        row[i] = 2.5;
        if i > 0 {
            row[i - 1] = -1.0;
        }
        if i + 1 < 6 {
            row[i + 1] = -1.0;
        }
    }
    let thomas = BandedSystem::new(chain.clone()).expect("chain");
    assert!(matches!(thomas, BandedSystem::Tridiagonal { .. }));
    // ... a bridged one the banded LU ...
    let mut bridged = chain.clone();
    bridged[0][4] = -0.5;
    bridged[4][0] = -0.5;
    let banded = BandedSystem::new(bridged).expect("bridged");
    assert!(matches!(banded, BandedSystem::Banded { kl: 4, ku: 4, .. }));
    // ... and a tridiagonal matrix that is not diagonally dominant is pivoted.
    let weak = vec![
        vec![0.1, 1.0, 0.0],
        vec![1.0, 0.1, 1.0],
        vec![0.0, 1.0, 0.1],
    ];
    let pivoted = BandedSystem::new(weak.clone()).expect("weak");
    assert!(matches!(pivoted, BandedSystem::Banded { .. }));
    let b = [1.0, 2.0, 3.0];
    assert!(max_abs_diff(&pivoted.solve(&b).expect("solve"), &dense_solve(&weak, &b)) < 1e-12);
    // Dominance by columns alone (row 0 is not row dominant: 2 < 3) is enough for Thomas.
    let column_dominant = vec![
        vec![2.0, 3.0, 0.0, 0.0],
        vec![1.0, 4.0, 0.5, 0.0],
        vec![0.0, 0.5, 2.0, 0.5],
        vec![0.0, 0.0, 1.0, 1.0],
    ];
    let system = BandedSystem::new(column_dominant.clone()).expect("columns");
    assert!(matches!(system, BandedSystem::Tridiagonal { .. }));
    let rhs = [1.0, -2.0, 3.0, 0.5];
    assert!(
        max_abs_diff(
            &system.solve(&rhs).expect("solve"),
            &dense_solve(&column_dominant, &rhs)
        ) < 1e-12
    );
}

#[test]
fn pivoting_rescues_a_tridiagonal_matrix_with_a_vanishing_pivot() {
    // [[0, 1], [1, 0]] is non-singular but Thomas' algorithm divides by zero.
    let x = solve_system(vec![vec![0.0, 1.0], vec![1.0, 0.0]], &[3.0, 5.0]).expect("pivoted");
    assert!(
        (x[0] - 5.0).abs() < 1e-14 && (x[1] - 3.0).abs() < 1e-14,
        "{x:?}"
    );
    // The same with an interior zero pivot in a larger tridiagonal system.
    let a = vec![
        vec![2.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.5, 1.0, 0.0],
        vec![0.0, 2.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 3.0],
    ];
    let b = [1.0, 2.0, 3.0, 4.0];
    let x = solve_system(a.clone(), &b).expect("pivoted");
    assert!(max_abs_diff(&x, &dense_solve(&a, &b)) < 1e-12);
}

#[test]
fn degenerate_systems_are_handled_or_reported() {
    // Empty system.
    assert!(solve_system(Vec::new(), &[]).expect("empty").is_empty());
    assert!(BandedSystem::new(Vec::new()).is_ok());
    // Singular and non-finite matrices.
    let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    assert!(matches!(
        solve_system(singular, &[1.0, 2.0]),
        Err(Error::SingularSystem)
    ));
    assert!(matches!(
        solve_system(vec![vec![0.0; 3]; 3], &[1.0, 1.0, 1.0]),
        Err(Error::SingularSystem)
    ));
    let mut bridged_singular = vec![vec![0.0; 4]; 4];
    for (i, row) in bridged_singular.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    bridged_singular[0][3] = 1.0;
    bridged_singular[3][3] = 0.0; // row 3 becomes zero: singular with a wide band
    bridged_singular[3][0] = 0.0;
    assert!(matches!(
        solve_system(bridged_singular, &[1.0; 4]),
        Err(Error::SingularSystem)
    ));
    assert!(matches!(
        solve_system(vec![vec![1.0, f64::NAN], vec![0.0, 1.0]], &[1.0, 1.0]),
        Err(Error::InvalidInput(_))
    ));
    // Shape errors.
    assert!(matches!(
        solve_system(vec![vec![1.0, 0.0]], &[1.0]),
        Err(Error::InvalidInput(_))
    ));
    let system = BandedSystem::new(vec![vec![2.0, 0.0], vec![0.0, 2.0]]).expect("diagonal");
    assert!(matches!(system.solve(&[1.0]), Err(Error::InvalidInput(_))));
    assert_eq!(system.solve(&[2.0, 4.0]).expect("solve"), vec![1.0, 2.0]);
}

#[test]
fn thomas_algorithm_accepts_an_empty_system() {
    // `x[n - 1]` used to underflow for n = 0.
    assert!(thomas_algorithm(&[], &[], &[], &[]).is_empty());
    // And still solves a regular system.
    let x = thomas_algorithm(
        &[0.0, -1.0, -1.0],
        &[2.0, 2.0, 2.0],
        &[-1.0, -1.0, 0.0],
        &[1.0, 0.0, 1.0],
    );
    for (got, want) in x.iter().zip([1.0, 1.0, 1.0]) {
        assert!((got - want).abs() < 1e-14, "{x:?}");
    }
}

// ---------------------------------------------------------------------------
// crank_nicolson_transient with a non-tridiagonal conduction matrix
// ---------------------------------------------------------------------------

/// The pre-fix algorithm: Thomas on the three diagonals of `K`, full `K` on the
/// right-hand side.
fn truncated_crank_nicolson(
    k: &[Vec<f64>],
    c: &[f64],
    t0: &[f64],
    f: &[f64],
    bcs: &[(usize, f64)],
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    let n = t0.len();
    let mut t = t0.to_vec();
    for _ in 0..steps {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut cv = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            b[i] = c[i] / dt + 0.5 * k[i][i];
            if i > 0 {
                a[i] = 0.5 * k[i][i - 1];
            }
            if i < n - 1 {
                cv[i] = 0.5 * k[i][i + 1];
            }
            let kt: f64 = (0..n).map(|j| k[i][j] * t[j]).sum();
            rhs[i] = c[i] / dt * t[i] - 0.5 * kt + f[i];
        }
        for &(node, temp) in bcs {
            a[node] = 0.0;
            b[node] = 1.0;
            cv[node] = 0.0;
            rhs[node] = temp;
        }
        t = thomas_algorithm(&a, &b, &cv, &rhs);
    }
    t
}

#[test]
fn crank_nicolson_on_a_two_dimensional_grid_matches_the_dense_reference() {
    let (nx, ny) = (3, 3);
    let n = nx * ny;
    let k = grid_conduction_matrix(nx, ny);
    assert_eq!(
        half_bandwidths(&k),
        (3, 3),
        "the grid matrix is not tridiagonal"
    );
    let c = vec![1.0; n];
    let t0 = vec![0.0; n];
    let mut f = vec![0.0; n];
    f[4] = 5.0;
    let bcs = [(0, 100.0), (8, 0.0)];
    let (dt, steps) = (0.05, 30);
    let result = crank_nicolson_transient(&k, &c, &t0, &f, &bcs, dt, steps).expect("CN");
    let reference = dense_crank_nicolson(&k, &c, &t0, &f, &bcs, dt, steps);
    assert_eq!(result.history.len(), steps + 1);
    for (step, (got, want)) in result.history.iter().zip(reference.iter()).enumerate() {
        assert!(
            max_abs_diff(got, want) < 1e-9,
            "step {step}: {got:?} vs dense {want:?}"
        );
    }
    assert_eq!(result.final_temperatures, result.history[steps]);
    // The tridiagonal truncation of the old implementation is far off.
    let truncated = truncated_crank_nicolson(&k, &c, &t0, &f, &bcs, dt, steps);
    assert!(
        max_abs_diff(&truncated, &result.final_temperatures) > 1.0,
        "the truncated solve must differ visibly: {truncated:?} vs {:?}",
        result.final_temperatures
    );
}

#[test]
fn crank_nicolson_steps_satisfy_the_full_matrix_equation() {
    // (C/dt + K/2) T⁺ − (C/dt − K/2) T − f = 0 at every free node, with the full K.
    let k = grid_conduction_matrix(4, 3);
    let n = 12;
    let c: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();
    let t0: Vec<f64> = (0..n).map(|i| 10.0 * ((i * 7) % 5) as f64).collect();
    let f: Vec<f64> = (0..n).map(|i| if i == 5 { 3.0 } else { 0.0 }).collect();
    let bcs = [(0, 50.0), (11, 20.0)];
    let dt = 0.02;
    let result = crank_nicolson_transient(&k, &c, &t0, &f, &bcs, dt, 12).expect("CN");
    for pair in result.history.windows(2) {
        let (before, after) = (&pair[0], &pair[1]);
        for i in 0..n {
            if bcs.iter().any(|&(node, _)| node == i) {
                continue;
            }
            let lhs: f64 =
                c[i] / dt * after[i] + 0.5 * (0..n).map(|j| k[i][j] * after[j]).sum::<f64>();
            let rhs: f64 = c[i] / dt * before[i]
                - 0.5 * (0..n).map(|j| k[i][j] * before[j]).sum::<f64>()
                + f[i];
            assert!((lhs - rhs).abs() < 1e-9, "node {i}: {lhs} vs {rhs}");
        }
    }
    assert!((result.final_temperatures[0] - 50.0).abs() < 1e-9);
    assert!((result.final_temperatures[11] - 20.0).abs() < 1e-9);
}

#[test]
fn crank_nicolson_reports_invalid_input_instead_of_panicking() {
    let k = [vec![1.0, -1.0], vec![-1.0, 1.0]];
    let c = [1.0, 1.0];
    let t = [0.0, 0.0];
    let q = [0.0, 0.0];
    let bc = [(0, 1.0)];
    let ok = |dt: f64| crank_nicolson_transient(&k, &c, &t, &q, &bc, dt, 3);
    assert!(ok(0.1).is_ok());
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(ok(dt), Err(Error::InvalidInput(_))), "dt = {dt}");
    }
    // Wrong sizes.
    assert!(matches!(
        crank_nicolson_transient(&k[..1], &c, &t, &q, &bc, 0.1, 1),
        Err(Error::InvalidInput(_))
    ));
    let ragged = [vec![1.0, -1.0], vec![-1.0]];
    assert!(matches!(
        crank_nicolson_transient(&ragged, &c, &t, &q, &bc, 0.1, 1),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        crank_nicolson_transient(&k, &c[..1], &t, &q, &bc, 0.1, 1),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        crank_nicolson_transient(&k, &c, &t, &q[..1], &bc, 0.1, 1),
        Err(Error::InvalidInput(_))
    ));
    // Dirichlet node out of range, and a non-finite prescribed temperature.
    match crank_nicolson_transient(&k, &c, &t, &q, &[(2, 1.0)], 0.1, 1) {
        Err(Error::DofOutOfRange { dof, n }) => assert_eq!((dof, n), (2, 2)),
        other => panic!("expected DofOutOfRange, got {other:?}"),
    }
    assert!(matches!(
        crank_nicolson_transient(&k, &c, &t, &q, &[(0, f64::NAN)], 0.1, 1),
        Err(Error::InvalidInput(_))
    ));
    // A pure conduction problem with no capacitance and no constraint is singular.
    assert!(matches!(
        crank_nicolson_transient(&k, &[0.0, 0.0], &t, &q, &[], 0.1, 1),
        Err(Error::SingularSystem)
    ));
    // Zero steps and the empty problem are fine.
    let none = crank_nicolson_transient(&k, &c, &t, &q, &bc, 0.1, 0).expect("0 steps");
    assert_eq!(none.history.len(), 1);
    assert_eq!(none.final_temperatures, vec![0.0, 0.0]);
    let empty = crank_nicolson_transient(&[], &[], &[], &[], &[], 0.1, 4).expect("empty");
    assert_eq!(empty.history.len(), 5);
    assert!(empty.final_temperatures.is_empty());
}

// ---------------------------------------------------------------------------
// ThermalMesh1D with a contact resistance between non-adjacent nodes
// ---------------------------------------------------------------------------

/// `n`-node rod, optionally bridged by a contact between `bridge` nodes.
fn rod(n: usize, bridge: Option<(usize, usize, f64)>) -> ThermalMesh1D {
    let mut mesh = ThermalMesh1D::new_uniform(n, 1.0, 3.0, 2.0, 1.5);
    if let Some((a, b, h_c)) = bridge {
        mesh.add_contact_resistance(a, b, h_c, 1.0);
    }
    mesh
}

#[test]
fn steady_state_keeps_a_non_adjacent_contact_resistance() {
    let n = 5;
    let build = |bridge| {
        let mut mesh = ThermalMesh1D::new_uniform(n, 1.0, 1.0, 1.0, 1.0);
        mesh.set_temperature_bc(0, 100.0);
        mesh.set_temperature_bc(n - 1, 0.0);
        if bridge {
            mesh.add_contact_resistance(0, 3, 2.0, 1.0);
        }
        mesh
    };
    let q = vec![0.0; n];
    let mut bridged = build(true);
    bridged.steady_state(&q).expect("steady state");
    // Independent dense solve of the assembled system.
    let k = bridged.assemble_conductance_matrix();
    assert_eq!(half_bandwidths(&k), (3, 3));
    let reference = constrained_dense_solve(
        &k,
        &bridged.assemble_load_vector(&q),
        &[(0, 100.0), (4, 0.0)],
    );
    assert!(
        max_abs_diff(&bridged.temperatures, &reference) < 1e-10,
        "{:?} vs dense {reference:?}",
        bridged.temperatures
    );
    // The bridge pulls node 3 towards the hot end (a linear profile gives 25).
    let mut plain = build(false);
    plain.steady_state(&q).expect("steady state");
    assert!((plain.temperatures[3] - 25.0).abs() < 1e-9);
    assert!(
        bridged.temperatures[3] > plain.temperatures[3] + 5.0,
        "T3 with bridge {} vs without {}",
        bridged.temperatures[3],
        plain.temperatures[3]
    );
}

#[test]
fn steady_state_handles_contacts_in_either_node_order_and_other_conditions() {
    let n = 6;
    let q = [0.0, 4.0, 0.0, 0.0, 1.0, 0.0];
    for &(a, b) in &[(1, 4), (4, 1), (0, 5), (2, 5)] {
        let mut mesh = rod(n, Some((a, b, 3.0)));
        mesh.set_temperature_bc(0, 90.0);
        mesh.add_convection_bc(n - 1, 8.0, 0.5, 15.0);
        mesh.steady_state(&q).expect("steady state");
        let k = mesh.assemble_conductance_matrix();
        let reference = constrained_dense_solve(&k, &mesh.assemble_load_vector(&q), &[(0, 90.0)]);
        assert!(
            max_abs_diff(&mesh.temperatures, &reference) < 1e-9,
            "contact ({a},{b}): {:?} vs {reference:?}",
            mesh.temperatures
        );
    }
}

#[test]
fn implicit_step_matches_a_dense_backward_euler_step_with_a_contact() {
    let n = 5;
    let mut mesh = rod(n, Some((0, 3, 6.0)));
    for (i, t) in mesh.temperatures.iter_mut().enumerate() {
        *t = 80.0 - 7.0 * i as f64;
    }
    mesh.set_temperature_bc(0, 80.0);
    mesh.add_convection_bc(n - 1, 25.0, 0.5, 15.0);
    let q = [0.0, 0.0, 40.0, 25.0, 0.0];
    let dt = 0.01;
    let k = mesh.assemble_conductance_matrix();
    let c = mesh.assemble_capacitance_vector();
    let load = mesh.assemble_load_vector(&q);
    let mut lhs = k.clone();
    for i in 0..n {
        lhs[i][i] += c[i] / dt;
    }
    let rhs: Vec<f64> = (0..n)
        .map(|i| c[i] / dt * mesh.temperatures[i] + load[i])
        .collect();
    let reference = constrained_dense_solve(&lhs, &rhs, &[(0, 80.0)]);
    mesh.step_implicit(dt, &q).expect("implicit step");
    assert!(
        max_abs_diff(&mesh.temperatures, &reference) < 1e-9,
        "{:?} vs dense {reference:?}",
        mesh.temperatures
    );
}

#[test]
fn explicit_and_implicit_agree_for_tiny_steps_with_a_non_adjacent_contact() {
    // The explicit step always used the full K; the implicit one dropped the bridge.
    let n = 5;
    let dt = 1.0e-6;
    let build = || {
        let mut mesh = rod(n, Some((0, 3, 6.0)));
        for (i, t) in mesh.temperatures.iter_mut().enumerate() {
            *t = 80.0 - 7.0 * i as f64;
        }
        mesh.set_temperature_bc(0, 80.0);
        mesh.add_convection_bc(n - 1, 25.0, 0.5, 15.0);
        mesh
    };
    let mut explicit = build();
    let mut implicit = build();
    let mut q = vec![0.0; n];
    q[2] = 40.0;
    for _ in 0..100 {
        explicit.step_explicit(dt, &q);
        implicit.step_implicit(dt, &q).expect("implicit step");
    }
    for i in 0..n {
        let (te, ti) = (explicit.temperatures[i], implicit.temperatures[i]);
        assert!(
            (te - ti).abs() <= 1e-6 * ti.abs(),
            "node {i}: explicit {te} vs implicit {ti}"
        );
    }
    // The contact matters: without it node 3 would not move like this.
    let mut unbridged = ThermalMesh1D::new_uniform(n, 1.0, 3.0, 2.0, 1.5);
    for (i, t) in unbridged.temperatures.iter_mut().enumerate() {
        *t = 80.0 - 7.0 * i as f64;
    }
    unbridged.set_temperature_bc(0, 80.0);
    unbridged.add_convection_bc(n - 1, 25.0, 0.5, 15.0);
    for _ in 0..100 {
        unbridged.step_implicit(dt, &q).expect("implicit step");
    }
    assert!((unbridged.temperatures[3] - implicit.temperatures[3]).abs() > 1e-3);
}

#[test]
fn theta_method_with_a_contact_matches_the_dense_crank_nicolson_reference() {
    let n = 5;
    let mut mesh = rod(n, Some((1, 4, 5.0)));
    mesh.set_temperature_bc(0, 100.0);
    mesh.set_temperature_bc(n - 1, 0.0);
    let k = mesh.assemble_conductance_matrix();
    let c = mesh.assemble_capacitance_vector();
    let q = vec![0.0; n];
    let t0 = mesh.temperatures.clone();
    let (dt, steps) = (0.01, 10);
    let reference = dense_crank_nicolson(&k, &c, &t0, &q, &[(0, 100.0), (4, 0.0)], dt, steps);
    let history = transient_theta_method(&mut mesh, dt, steps, 0.5, &q).expect("theta method");
    assert_eq!(history.len(), steps + 1);
    for (step, (got, want)) in history.iter().zip(reference.iter()).enumerate() {
        assert!(
            max_abs_diff(got, want) < 1e-9,
            "step {step}: {got:?} vs {want:?}"
        );
    }
    assert_eq!(mesh.temperatures, history[steps]);
}

#[test]
fn backward_euler_theta_method_equals_repeated_implicit_steps_with_all_conditions() {
    let build = || {
        let mut mesh = rod(6, Some((0, 2, 4.0)));
        for (i, t) in mesh.temperatures.iter_mut().enumerate() {
            *t = 400.0 - 20.0 * i as f64;
        }
        mesh.set_temperature_bc(0, 400.0);
        mesh.add_convection_bc(5, 12.0, 0.4, 300.0);
        mesh.add_radiation_bc(4, 0.8, 0.3, 280.0);
        mesh
    };
    let q = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let dt = 0.05;
    let mut stepped = build();
    for _ in 0..8 {
        stepped.step_implicit(dt, &q).expect("implicit step");
    }
    let mut marched = build();
    let history = transient_theta_method(&mut marched, dt, 8, 1.0, &q).expect("theta method");
    assert!(
        max_abs_diff(&stepped.temperatures, &marched.temperatures) < 1e-9,
        "{:?} vs {:?}",
        stepped.temperatures,
        marched.temperatures
    );
    assert_eq!(history.len(), 9);
}

#[test]
fn theta_method_includes_the_convection_load_of_the_mesh() {
    // Marching to steady state must give the steady solution with the ambient
    // temperature; the old right-hand side had no `h A T_inf` term.
    let n = 5;
    let build = || {
        let mut mesh = ThermalMesh1D::new_uniform(n, 1.0, 1.0, 1.0, 1.0);
        mesh.set_temperature_bc(0, 100.0);
        mesh.add_convection_bc(n - 1, 10.0, 1.0, 20.0);
        mesh
    };
    let q = vec![0.0; n];
    let mut steady = build();
    steady.steady_state(&q).expect("steady state");
    // Analytic: conduction 4 (100 - T1)... the end node balances 10 (T4 - 20).
    let mut marched = build();
    transient_theta_method(&mut marched, 0.5, 400, 1.0, &q).expect("theta method");
    assert!(
        max_abs_diff(&marched.temperatures, &steady.temperatures) < 1e-6,
        "{:?} vs {:?}",
        marched.temperatures,
        steady.temperatures
    );
    let mut crank = build();
    transient_theta_method(&mut crank, 0.5, 400, 0.5, &q).expect("theta method");
    assert!(
        max_abs_diff(&crank.temperatures, &steady.temperatures) < 1e-6,
        "{:?} vs {:?}",
        crank.temperatures,
        steady.temperatures
    );
    // The end node sits strictly between the ambient and the hot end.
    assert!(steady.temperatures[n - 1] > 20.0 && steady.temperatures[n - 1] < 100.0);
}

#[test]
fn steady_state_includes_the_radiation_stiffness() {
    // Radiation was added to the load as h_rad A T_surr without the matching
    // h_rad A on the diagonal, i.e. as a pure heat source.
    let n = 5;
    let mut mesh = ThermalMesh1D::new_uniform(n, 1.0, 1.0, 1.0, 1.0);
    mesh.set_temperature_bc(0, 500.0);
    mesh.add_radiation_bc(n - 1, 0.9, 1.0, 300.0);
    for t in mesh.temperatures.iter_mut() {
        *t = 450.0;
    }
    let q = vec![0.0; n];
    // The linearisation is re-evaluated at the new temperatures until it settles.
    for _ in 0..40 {
        mesh.steady_state(&q).expect("steady state");
    }
    let t_end = mesh.temperatures[n - 1];
    assert!(
        t_end > 300.0 && t_end < 500.0,
        "radiating end must lie between surroundings and the hot end: {t_end}"
    );
    // Fixed point: conduction into the end node equals the linearised radiation loss.
    let k_cond = 4.0;
    let h_rad = 4.0 * 0.9 * 5.670374419e-8 * t_end.powi(3);
    let conduction = k_cond * (mesh.temperatures[n - 2] - t_end);
    let radiation = h_rad * (t_end - 300.0);
    assert!(
        (conduction - radiation).abs() < 1e-6 * conduction.abs(),
        "conduction {conduction} vs radiation {radiation}"
    );
    // Time marching converges to the same fixed point.
    let mut marched = ThermalMesh1D::new_uniform(n, 1.0, 1.0, 1.0, 1.0);
    marched.set_temperature_bc(0, 500.0);
    marched.add_radiation_bc(n - 1, 0.9, 1.0, 300.0);
    for t in marched.temperatures.iter_mut() {
        *t = 450.0;
    }
    for _ in 0..600 {
        marched.step_implicit(0.5, &q).expect("implicit step");
    }
    assert!(max_abs_diff(&marched.temperatures, &mesh.temperatures) < 1e-6);
}

#[test]
fn unconstrained_and_malformed_solves_report_errors_and_leave_the_mesh_alone() {
    let n = 4;
    let mut mesh = ThermalMesh1D::new_uniform(n, 1.0, 1.0, 1.0, 1.0);
    for (i, t) in mesh.temperatures.iter_mut().enumerate() {
        *t = 10.0 * i as f64;
    }
    let before = mesh.temperatures.clone();
    let q = vec![0.0; n];
    // Pure Neumann conduction has no unique steady state.
    assert!(matches!(mesh.steady_state(&q), Err(Error::SingularSystem)));
    assert_eq!(mesh.temperatures, before);
    // Wrong source length, bad step, bad theta.
    assert!(matches!(
        mesh.steady_state(&[0.0; 3]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        mesh.step_implicit(0.1, &[0.0; 5]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        mesh.step_implicit(0.1, &[0.0, f64::NAN, 0.0, 0.0]),
        Err(Error::InvalidInput(_))
    ));
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(mesh.step_implicit(dt, &q), Err(Error::InvalidInput(_))),
            "dt = {dt}"
        );
    }
    for theta in [-0.1, 1.1, f64::NAN] {
        assert!(
            matches!(
                transient_theta_method(&mut mesh, 0.1, 2, theta, &q),
                Err(Error::InvalidInput(_))
            ),
            "theta = {theta}"
        );
    }
    assert_eq!(mesh.temperatures, before);
    // With capacitance the implicit step is solvable even without constraints,
    // and conserves the heat content of an insulated rod.
    let heat_before = mesh.total_heat_content();
    mesh.step_implicit(0.1, &q)
        .expect("insulated implicit step");
    assert!((mesh.total_heat_content() - heat_before).abs() < 1e-9 * heat_before.abs());
}
