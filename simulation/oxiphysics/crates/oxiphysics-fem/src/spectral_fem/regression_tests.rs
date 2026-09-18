//! Regression tests for the 2-D boundary element influence matrices.
//!
//! `SpectralBoundaryIntegral::h_matrix` / `g_matrix` used to return the raw
//! point kernels `∂G/∂n(xᵢ, yⱼ)` and `G(xᵢ, yⱼ)`: no panel integration, zero
//! diagonals (the free term `½` and the logarithmic self integral were missing),
//! no solve, and `green_normal_derivative_2d` returned `−∂G/∂n_y`.  `H u = G q`
//! was therefore not a boundary integral equation.

use super::functions::{green_function_2d, green_normal_derivative_2d};
use super::types::{GAUSS4_NODES, GAUSS4_WEIGHTS, SpectralBoundaryIntegral};
use crate::error::Error;
use std::f64::consts::PI;

/// Vertices of a regular `n`-gon inscribed in the circle of radius `radius`
/// (counter-clockwise).
fn circle_vertices(radius: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n as f64;
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect()
}

fn ellipse_vertices(a: f64, b: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n as f64;
            [a * angle.cos(), b * angle.sin()]
        })
        .collect()
}

fn polygon(vertices: &[[f64; 2]]) -> SpectralBoundaryIntegral {
    SpectralBoundaryIntegral::from_polygon(vertices).expect("valid polygon")
}

/// Largest `|q_i − f(x_i, n_i)|` over the panels accepted by `keep`.
fn flux_error(
    bem: &SpectralBoundaryIntegral,
    q: &[f64],
    exact: impl Fn([f64; 2], [f64; 2]) -> f64,
    keep: impl Fn([f64; 2]) -> bool,
) -> f64 {
    (0..bem.n_panels())
        .filter(|&i| keep(bem.nodes[i]))
        .map(|i| (q[i] - exact(bem.nodes[i], bem.normals[i])).abs())
        .fold(0.0_f64, f64::max)
}

fn potential(bem: &SpectralBoundaryIntegral, u: impl Fn([f64; 2]) -> f64) -> Vec<f64> {
    bem.nodes.iter().map(|&x| u(x)).collect()
}

// ---------------------------------------------------------------------------
// Kernels and quadrature
// ---------------------------------------------------------------------------

#[test]
fn double_layer_kernel_is_the_normal_derivative_of_the_fundamental_solution() {
    // The old kernel returned −∂G/∂n_y: positive for a normal pointing away from x.
    let x = [0.3, -0.2];
    let y = [1.1, 0.4];
    let norm = 0.6_f64.hypot(-0.3);
    let n = [0.6 / norm, -0.3 / norm];
    let h = 1.0e-6;
    let plus = [y[0] + h * n[0], y[1] + h * n[1]];
    let minus = [y[0] - h * n[0], y[1] - h * n[1]];
    let numeric = (green_function_2d(x, plus) - green_function_2d(x, minus)) / (2.0 * h);
    let analytic = green_normal_derivative_2d(x, y, n);
    assert!(
        (numeric - analytic).abs() <= 1e-7 * analytic.abs(),
        "FD {numeric} vs kernel {analytic}"
    );
    // A normal pointing away from the source lets G decay: dG/dn < 0.
    let r = ((y[0] - x[0]).powi(2) + (y[1] - x[1]).powi(2)).sqrt();
    let away = [(y[0] - x[0]) / r, (y[1] - x[1]) / r];
    let outward = green_normal_derivative_2d(x, y, away);
    assert!(outward < 0.0);
    // Magnitude on the axis: |dG/dn| = 1 / (2 pi r).
    assert!((outward + 1.0 / (2.0 * PI * r)).abs() < 1e-14);
    // Pointing towards the source it grows; tangential directions give zero.
    assert!(green_normal_derivative_2d(x, y, [-away[0], -away[1]]) > 0.0);
    assert!(green_normal_derivative_2d(x, y, [-away[1], away[0]]).abs() < 1e-15);
    assert_eq!(green_normal_derivative_2d(x, x, n), 0.0);
}

#[test]
fn four_point_gauss_rule_integrates_polynomials_up_to_degree_seven() {
    assert!((GAUSS4_WEIGHTS.iter().sum::<f64>() - 2.0).abs() < 1e-15);
    for degree in 0..=7_i32 {
        let approx: f64 = GAUSS4_NODES
            .iter()
            .zip(GAUSS4_WEIGHTS.iter())
            .map(|(&x, &w)| w * x.powi(degree))
            .sum();
        let exact = if degree % 2 == 0 {
            2.0 / (degree + 1) as f64
        } else {
            0.0
        };
        assert!(
            (approx - exact).abs() < 1e-15,
            "degree {degree}: {approx} vs {exact}"
        );
    }
}

#[test]
fn self_and_neighbour_integrals_match_the_closed_forms() {
    // Three collinear panels of length l centred at 0, l and 2 l on the x axis.
    let l = 0.5;
    let nodes = vec![[0.0, 0.0], [l, 0.0], [2.0 * l, 0.0]];
    let normals = vec![[0.0, 1.0]; 3];
    let bem = SpectralBoundaryIntegral::with_lengths(nodes, normals, vec![l; 3]).expect("mesh");
    let g = bem.g_matrix().expect("G");
    let h = bem.h_matrix().expect("H");
    // ∫ −ln s ds / (2π) = −(s ln s − s) / (2π) between the panel end points.
    let primitive = |s: f64| -(s * s.ln() - s) / (2.0 * PI);
    let self_term = l / (2.0 * PI) * (1.0 - (0.5 * l).ln());
    assert!((g[1][1] - self_term).abs() < 1e-14);
    assert!((g[0][0] - self_term).abs() < 1e-14);
    // Adjacent panels: distances 0.5 l .. 1.5 l from the collocation point.
    let adjacent = primitive(1.5 * l) - primitive(0.5 * l);
    for (i, j) in [(1, 2), (1, 0), (0, 1), (2, 1)] {
        assert!(
            (g[i][j] - adjacent).abs() < 1e-5 * adjacent.abs(),
            "G[{i}][{j}] = {} vs {adjacent}",
            g[i][j]
        );
    }
    // Next-nearest panels: distances 1.5 l .. 2.5 l.  The integral is a small
    // difference of O(1) terms (ln s changes sign at s = 1), so compare absolutely.
    let distant = primitive(2.5 * l) - primitive(1.5 * l);
    for (i, j) in [(0, 2), (2, 0)] {
        assert!(
            (g[i][j] - distant).abs() < 5e-9,
            "G[{i}][{j}] = {} vs {distant}",
            g[i][j]
        );
    }
    // Straight panels in a line: (y − x) ⟂ n, so the double-layer kernel vanishes.
    for row in &h {
        for &value in row {
            assert!(value.abs() < 1e-14, "{h:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Influence matrices
// ---------------------------------------------------------------------------

#[test]
fn h_rows_sum_to_zero_and_the_diagonal_carries_the_free_term() {
    for n in [32, 64, 128] {
        let bem = polygon(&circle_vertices(1.0, n));
        let h = bem.h_matrix().expect("H");
        for (i, row) in h.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-13, "row {i} of the {n}-gon sums to {sum}");
            // cᵢ = ½ on a smooth boundary (the straight panel adds no self integral).
            assert!((row[i] - 0.5).abs() < 1e-5, "H[{i}][{i}] = {}", row[i]);
            assert!(row.iter().enumerate().all(|(j, &v)| j == i || v < 0.0));
        }
    }
}

#[test]
fn g_is_a_symmetric_positive_diagonal_for_a_regular_polygon() {
    let bem = polygon(&circle_vertices(0.4, 32));
    let g = bem.g_matrix().expect("G");
    assert_eq!(g.len(), 32);
    for (i, row) in g.iter().enumerate() {
        assert!(row[i] > 0.0);
        for (j, &value) in row.iter().enumerate() {
            assert!(
                (value - g[j][i]).abs() < 1e-7,
                "G not symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn g_has_a_null_direction_on_the_unit_circle_scale() {
    // ∮ −ln|x − y| dΓ(y) = −2π R ln R: zero at R = 1 (the degenerate scale of the
    // logarithmic kernel), so G·1 ≈ 0 there while it is clearly non-zero elsewhere.
    for (radius, expected) in [
        (1.0, 0.0),
        (0.5, 0.5 * 2.0_f64.ln()),
        (2.0, -2.0 * 2.0_f64.ln()),
    ] {
        let bem = polygon(&circle_vertices(radius, 128));
        let g = bem.g_matrix().expect("G");
        let row_sum: f64 = g[0].iter().sum();
        assert!(
            (row_sum - expected).abs() < 2e-3,
            "R = {radius}: {row_sum} vs {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// solve_dirichlet
// ---------------------------------------------------------------------------

#[test]
fn unit_circle_with_64_panels_recovers_the_flux_of_a_linear_potential() {
    // u = x is harmonic; its outward flux on the unit circle is q = n_x.
    let bem = polygon(&circle_vertices(1.0, 64));
    let u = potential(&bem, |x| x[0]);
    let q = bem.solve_dirichlet(&u).expect("Dirichlet solve");
    let error = flux_error(&bem, &q, |_, n| n[0], |_| true);
    assert!(error < 1.2e-3, "max |q - n_x| = {error:e}");
    // Second-order convergence of the constant-panel collocation.
    let fine = polygon(&circle_vertices(1.0, 128));
    let q_fine = fine
        .solve_dirichlet(&potential(&fine, |x| x[0]))
        .expect("Dirichlet solve");
    let error_fine = flux_error(&fine, &q_fine, |_, n| n[0], |_| true);
    assert!(error_fine < 3.0e-4, "{error_fine:e}");
    assert!(
        error / error_fine > 3.5,
        "no h² convergence: {error:e} -> {error_fine:e}"
    );
    // The flux balance holds exactly.
    let net: f64 = q
        .iter()
        .zip(bem.lengths.iter())
        .map(|(qi, li)| qi * li)
        .sum();
    assert!(net.abs() < 1e-12, "net flux {net}");
}

#[test]
fn nodes_and_normals_on_a_circle_recover_the_flux_too() {
    // `new` derives the panel lengths from the node spacing (tangent panels).
    let n = 64;
    let nodes: Vec<[f64; 2]> = circle_vertices(1.0, n);
    let normals = nodes.clone();
    let bem = SpectralBoundaryIntegral::new(nodes, normals).expect("mesh");
    let chord = 2.0 * (PI / n as f64).sin();
    for &l in &bem.lengths {
        assert!((l - chord).abs() < 1e-14);
    }
    let q = bem
        .solve_dirichlet(&potential(&bem, |x| x[0]))
        .expect("Dirichlet solve");
    let error = flux_error(&bem, &q, |_, n| n[0], |_| true);
    assert!(error < 1.2e-3, "{error:e}");
}

#[test]
fn the_solution_is_scale_invariant_including_the_degenerate_unit_radius() {
    // For R = 1 the plain system G q = H u is (nearly) singular; the bordered
    // system is exactly scale covariant, so the flux of u = x is the same for every R.
    let mut reference: Option<Vec<f64>> = None;
    for radius in [0.5, 1.0, 2.0, 7.0] {
        let bem = polygon(&circle_vertices(radius, 64));
        let q = bem
            .solve_dirichlet(&potential(&bem, |x| x[0]))
            .expect("Dirichlet solve");
        let error = flux_error(&bem, &q, |_, n| n[0], |_| true);
        assert!(error < 1.2e-3, "R = {radius}: {error:e}");
        match &reference {
            None => reference = Some(q),
            Some(first) => {
                for (a, b) in first.iter().zip(q.iter()) {
                    assert!((a - b).abs() < 1e-9, "R = {radius}: {a} vs {b}");
                }
            }
        }
    }
}

#[test]
fn an_ellipse_of_logarithmic_capacity_one_is_solved() {
    // The capacity of an ellipse with semi-axes a, b is (a + b) / 2: a = 1.5,
    // b = 0.5 is again a degenerate scale.
    let bem = polygon(&ellipse_vertices(1.5, 0.5, 128));
    let q = bem
        .solve_dirichlet(&potential(&bem, |x| x[0]))
        .expect("Dirichlet solve");
    let error = flux_error(&bem, &q, |_, n| n[0], |_| true);
    assert!(error < 3.0e-3, "{error:e}");
    // Linear and bilinear harmonic data.
    let q_linear = bem
        .solve_dirichlet(&potential(&bem, |x| 2.0 * x[0] - 3.0 * x[1] + 1.0))
        .expect("Dirichlet solve");
    let error_linear = flux_error(&bem, &q_linear, |_, n| 2.0 * n[0] - 3.0 * n[1], |_| true);
    assert!(error_linear < 2.0e-2, "{error_linear:e}");
    let q_xy = bem
        .solve_dirichlet(&potential(&bem, |x| x[0] * x[1]))
        .expect("Dirichlet solve");
    let error_xy = flux_error(&bem, &q_xy, |x, n| x[1] * n[0] + x[0] * n[1], |_| true);
    assert!(error_xy < 6.0e-3, "{error_xy:e}");
}

#[test]
fn a_constant_potential_carries_no_flux_and_constants_do_not_change_the_flux() {
    let bem = polygon(&ellipse_vertices(1.2, 0.7, 48));
    let constant = bem
        .solve_dirichlet(&vec![3.7; 48])
        .expect("Dirichlet solve");
    assert!(constant.iter().all(|q| q.abs() < 1e-12), "{constant:?}");
    let base = bem
        .solve_dirichlet(&potential(&bem, |x| x[0] - 0.5 * x[1]))
        .expect("Dirichlet solve");
    let shifted = bem
        .solve_dirichlet(&potential(&bem, |x| x[0] - 0.5 * x[1] + 100.0))
        .expect("Dirichlet solve");
    for (a, b) in base.iter().zip(shifted.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

#[test]
fn square_flux_is_accurate_away_from_the_corners() {
    // u = x² − y² has q = 2 x n_x − 2 y n_y; constant panels are inaccurate next to
    // the corners (q jumps there) but converge elsewhere.
    let per_side = 16;
    let mut vertices = Vec::new();
    for k in 0..per_side {
        vertices.push([-1.0 + 2.0 * k as f64 / per_side as f64, -1.0]);
    }
    for k in 0..per_side {
        vertices.push([1.0, -1.0 + 2.0 * k as f64 / per_side as f64]);
    }
    for k in 0..per_side {
        vertices.push([1.0 - 2.0 * k as f64 / per_side as f64, 1.0]);
    }
    for k in 0..per_side {
        vertices.push([-1.0, 1.0 - 2.0 * k as f64 / per_side as f64]);
    }
    let bem = polygon(&vertices);
    let q = bem
        .solve_dirichlet(&potential(&bem, |x| x[0] * x[0] - x[1] * x[1]))
        .expect("Dirichlet solve");
    let away = |x: [f64; 2]| {
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
            .iter()
            .all(|&(cx, cy)| (x[0] - cx).hypot(x[1] - cy) > 0.3)
    };
    let exact = |x: [f64; 2], n: [f64; 2]| 2.0 * x[0] * n[0] - 2.0 * x[1] * n[1];
    let error = flux_error(&bem, &q, exact, away);
    assert!(error < 3.0e-3, "{error:e}");
    // The corner panels are still the right order of magnitude (|q| <= 2).
    let corner_error = flux_error(&bem, &q, exact, |_| true);
    assert!(corner_error < 0.5, "{corner_error}");
}

// ---------------------------------------------------------------------------
// Construction and validation
// ---------------------------------------------------------------------------

#[test]
fn polygon_normals_point_outward_for_either_orientation() {
    let counter_clockwise = circle_vertices(1.0, 16);
    let mut clockwise = counter_clockwise.clone();
    clockwise.reverse();
    for vertices in [counter_clockwise, clockwise] {
        let bem = polygon(&vertices);
        assert_eq!(bem.n_panels(), 16);
        for i in 0..16 {
            let outward = bem.nodes[i][0] * bem.normals[i][0] + bem.nodes[i][1] * bem.normals[i][1];
            assert!(outward > 0.98, "panel {i}: n . x = {outward}");
            let norm = bem.normals[i][0].hypot(bem.normals[i][1]);
            assert!((norm - 1.0).abs() < 1e-14);
        }
        let perimeter: f64 = bem.lengths.iter().sum();
        assert!((perimeter - 32.0 * (PI / 16.0).sin()).abs() < 1e-12);
    }
}

#[test]
fn constructors_reject_malformed_meshes() {
    let circle = circle_vertices(1.0, 8);
    let unit_normals = circle.clone();
    let invalid =
        |r: Result<SpectralBoundaryIntegral, Error>| matches!(r, Err(Error::InvalidInput(_)));
    // Length mismatches.
    assert!(invalid(SpectralBoundaryIntegral::new(
        circle.clone(),
        unit_normals[..7].to_vec()
    )));
    assert!(invalid(SpectralBoundaryIntegral::with_lengths(
        circle.clone(),
        unit_normals.clone(),
        vec![0.1; 7]
    )));
    // Too few nodes for a closed boundary; empty mesh.
    assert!(invalid(SpectralBoundaryIntegral::new(
        circle[..2].to_vec(),
        unit_normals[..2].to_vec()
    )));
    assert!(invalid(SpectralBoundaryIntegral::with_lengths(
        Vec::new(),
        Vec::new(),
        Vec::new()
    )));
    // Non-unit and non-finite normals, non-finite nodes.
    let mut bad_normals = unit_normals.clone();
    bad_normals[3] = [1.0, 1.0];
    assert!(invalid(SpectralBoundaryIntegral::new(
        circle.clone(),
        bad_normals
    )));
    let mut nan_normals = unit_normals.clone();
    nan_normals[0] = [f64::NAN, 0.0];
    assert!(invalid(SpectralBoundaryIntegral::new(
        circle.clone(),
        nan_normals
    )));
    let mut nan_nodes = circle.clone();
    nan_nodes[2] = [f64::INFINITY, 0.0];
    assert!(invalid(SpectralBoundaryIntegral::new(
        nan_nodes,
        unit_normals.clone()
    )));
    // Coincident nodes give a zero panel length.
    assert!(invalid(SpectralBoundaryIntegral::new(
        vec![[0.0, 0.0]; 4],
        vec![[1.0, 0.0]; 4]
    )));
    // Non-positive lengths.
    for bad in [0.0, -0.1, f64::NAN] {
        let mut lengths = vec![0.3; 8];
        lengths[5] = bad;
        assert!(invalid(SpectralBoundaryIntegral::with_lengths(
            circle.clone(),
            unit_normals.clone(),
            lengths
        )));
    }
    // Polygons: too few vertices, collinear (zero area), repeated vertex, NaN.
    assert!(invalid(SpectralBoundaryIntegral::from_polygon(
        &circle[..2]
    )));
    assert!(invalid(SpectralBoundaryIntegral::from_polygon(&[
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0]
    ])));
    assert!(invalid(SpectralBoundaryIntegral::from_polygon(&[
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])));
    assert!(invalid(SpectralBoundaryIntegral::from_polygon(&[
        [0.0, 0.0],
        [1.0, f64::NAN],
        [1.0, 1.0]
    ])));
}

#[test]
fn solves_and_assembly_report_invalid_input() {
    let bem = polygon(&circle_vertices(1.0, 12));
    assert!(matches!(
        bem.solve_dirichlet(&[0.0; 11]),
        Err(Error::InvalidInput(_))
    ));
    let mut with_nan = [0.0; 12];
    with_nan[4] = f64::NAN;
    assert!(matches!(
        bem.solve_dirichlet(&with_nan),
        Err(Error::InvalidInput(_))
    ));
    // The public fields can be edited into an inconsistent mesh: report, don't panic.
    let mut broken = polygon(&circle_vertices(1.0, 12));
    broken.lengths.pop();
    assert!(matches!(broken.h_matrix(), Err(Error::InvalidInput(_))));
    assert!(matches!(broken.g_matrix(), Err(Error::InvalidInput(_))));
    assert!(matches!(
        broken.solve_dirichlet(&[0.0; 12]),
        Err(Error::InvalidInput(_))
    ));
    let mut bad_normal = polygon(&circle_vertices(1.0, 12));
    bad_normal.normals[2] = [2.0, 0.0];
    assert!(matches!(bad_normal.h_matrix(), Err(Error::InvalidInput(_))));
}
