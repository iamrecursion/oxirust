// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the singular-metric handling of the N-dimensional
//! differential-geometry API (`MetricTensorND::try_inverse` and every function
//! that inverts a metric) and for `RiemannianMetric::christoffel_111`.
//!
//! Split out of `functions.rs` to keep that file under the 2000-line limit.

use super::*;
use crate::error::PhysicsError;

/// Polar metric `diag(1, r²)`, singular at the origin `r = 0`.
fn polar_metric(p: &[f64; MAX_DIM]) -> [[f64; MAX_DIM]; MAX_DIM] {
    let mut g = [[0.0; MAX_DIM]; MAX_DIM];
    g[0][0] = 1.0;
    g[1][1] = p[0] * p[0];
    g
}

/// Polar metric whose angular row vanishes identically for `r <= 0`, so the
/// singularity is hit exactly (no rounding to a tiny but non-zero `r²`).
fn clamped_polar_metric(p: &[f64; MAX_DIM]) -> [[f64; MAX_DIM]; MAX_DIM] {
    let mut g = [[0.0; MAX_DIM]; MAX_DIM];
    g[0][0] = 1.0;
    g[1][1] = if p[0] > 0.0 { p[0] * p[0] } else { 0.0 };
    g
}

/// Spherical metric `diag(1, r², r² sin²θ)` of flat 3-space; coordinates
/// `(r, θ, φ)`.  Singular at `r = 0` and on the polar axis `sin θ = 0`.
fn spherical_metric(p: &[f64; MAX_DIM]) -> [[f64; MAX_DIM]; MAX_DIM] {
    let (r, theta) = (p[0], p[1]);
    let mut g = [[0.0; MAX_DIM]; MAX_DIM];
    g[0][0] = 1.0;
    g[1][1] = r * r;
    g[2][2] = r * r * theta.sin() * theta.sin();
    g
}

fn flat_2d_metric(_p: &[f64; MAX_DIM]) -> [[f64; MAX_DIM]; MAX_DIM] {
    let mut g = [[0.0; MAX_DIM]; MAX_DIM];
    g[0][0] = 1.0;
    g[1][1] = 1.0;
    g
}

fn metric_from_rows(rows: &[&[f64]]) -> MetricTensorND {
    let dim = rows.len();
    let mut g = [[0.0; MAX_DIM]; MAX_DIM];
    for (g_row, row) in g.iter_mut().zip(rows.iter()) {
        for (cell, &value) in g_row.iter_mut().zip(row.iter()) {
            *cell = value;
        }
    }
    MetricTensorND { dim, g }
}

fn assert_invalid_input(error: &PhysicsError, expected_field: &str) {
    match error {
        PhysicsError::InvalidInput { field, .. } => {
            assert_eq!(field, expected_field, "wrong field in {error}");
        }
        other => panic!("expected InvalidInput({expected_field}), got {other:?}"),
    }
}

fn assert_singular_metric(error: &PhysicsError) {
    assert_invalid_input(error, "metric");
    assert!(
        error.to_string().contains("singular"),
        "message should say singular: {error}"
    );
}

/// `max |g · g⁻¹ − I|` over the leading `dim × dim` block.
fn inverse_residual(metric: &MetricTensorND, inverse: &[[f64; MAX_DIM]; MAX_DIM]) -> f64 {
    let n = metric.dim;
    let mut worst = 0.0_f64;
    for (i, g_row) in metric.g.iter().enumerate().take(n) {
        for j in 0..n {
            let product: f64 = g_row
                .iter()
                .zip(inverse.iter())
                .map(|(&g_ik, inv_k)| g_ik * inv_k[j])
                .take(n)
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((product - expected).abs());
        }
    }
    worst
}

// ── MetricTensorND::try_inverse ───────────────────────────────────────────────

#[test]
fn test_try_inverse_rejects_zero_row() {
    // The polar metric at the origin: diag(1, r²) with r = 0.
    let metric = metric_from_rows(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let error = metric.try_inverse().expect_err("zero row is singular");
    assert_singular_metric(&error);
}

#[test]
fn test_try_inverse_rejects_zero_matrix() {
    let metric = metric_from_rows(&[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]]);
    let error = metric.try_inverse().expect_err("zero matrix is singular");
    assert_singular_metric(&error);
}

#[test]
fn test_try_inverse_rejects_rank_deficient_metrics() {
    let duplicate_rows = metric_from_rows(&[&[1.0, 1.0], &[1.0, 1.0]]);
    assert_singular_metric(&duplicate_rows.try_inverse().expect_err("rank 1"));
    let proportional_rows = metric_from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
    assert_singular_metric(&proportional_rows.try_inverse().expect_err("rank 1"));
    // Rank 2 in three dimensions: the third row is the sum of the first two.
    let rank_two = metric_from_rows(&[&[2.0, 0.5, 2.5], &[0.5, 3.0, 3.5], &[2.5, 3.5, 6.0]]);
    assert_singular_metric(&rank_two.try_inverse().expect_err("rank 2"));
    // Singular to working precision: the pivot after elimination is ~1e-15.
    let nearly = metric_from_rows(&[&[1.0, 1.0], &[1.0, 1.0 + 1e-15]]);
    assert_singular_metric(&nearly.try_inverse().expect_err("numerically rank 1"));
}

#[test]
fn test_try_inverse_keeps_badly_scaled_regular_metrics_invertible() {
    // SI Minkowski metric ds² = -c² dt² + dx² + dy² + dz²: the largest and
    // smallest components differ by 17 orders of magnitude.  A pivot test
    // relative to the global maximum |g_ij| would call this singular.
    let c = 299_792_458.0_f64;
    let si_minkowski = metric_from_rows(&[
        &[-c * c, 0.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]);
    let inverse = si_minkowski.try_inverse().expect("regular metric");
    assert!(((inverse[0][0] + 1.0 / (c * c)) * c * c).abs() < 1e-12);
    for (k, row) in inverse.iter().enumerate().skip(1) {
        assert!((row[k] - 1.0).abs() < 1e-12, "g^{k}{k} = {}", row[k]);
    }
    assert!(inverse_residual(&si_minkowski, &inverse) < 1e-9);

    let wide = metric_from_rows(&[&[1e20, 0.0], &[0.0, 1.0]]);
    let inverse = wide.try_inverse().expect("diag(1e20, 1) is regular");
    assert!((inverse[0][0] * 1e20 - 1.0).abs() < 1e-12);
    assert!((inverse[1][1] - 1.0).abs() < 1e-12);

    // The polar metric a hair away from the origin is a legitimate (if
    // extreme) regular metric: diag(1, 1e-16).
    let near_origin = metric_from_rows(&[&[1.0, 0.0], &[0.0, 1e-16]]);
    let inverse = near_origin.try_inverse().expect("regular metric");
    assert!((inverse[1][1] * 1e-16 - 1.0).abs() < 1e-12);

    // A uniformly tiny metric is as regular as the identity.
    let tiny = metric_from_rows(&[&[3e-30, 1e-30], &[1e-30, 2e-30]]);
    let inverse = tiny.try_inverse().expect("regular metric");
    assert!(inverse_residual(&tiny, &inverse) < 1e-12);
}

#[test]
fn test_try_inverse_matches_inverse_and_round_trips() {
    // Needs row exchanges: zero (0, 0) entry.
    let swap = metric_from_rows(&[&[0.0, 1.0], &[1.0, 0.0]]);
    let inverse = swap.try_inverse().expect("regular metric");
    assert_eq!(inverse[0][1], 1.0);
    assert_eq!(inverse[1][0], 1.0);
    assert_eq!(inverse[0][0], 0.0);

    let general = metric_from_rows(&[
        &[2.0, 0.5, -0.3, 0.1],
        &[0.5, 3.0, 0.2, -0.4],
        &[-0.3, 0.2, 1.5, 0.6],
        &[0.1, -0.4, 0.6, 2.5],
    ]);
    let inverse = general.try_inverse().expect("regular metric");
    assert!(inverse_residual(&general, &inverse) < 1e-12);
    assert_eq!(inverse, general.inverse());

    let pair = metric_from_rows(&[&[2.0, 0.5], &[0.5, 3.0]]);
    let inverse = pair.try_inverse().expect("regular metric");
    let det = 2.0 * 3.0 - 0.5 * 0.5;
    assert!((inverse[0][0] - 3.0 / det).abs() < 1e-14);
    assert!((inverse[0][1] + 0.5 / det).abs() < 1e-14);
    assert!((inverse[1][0] + 0.5 / det).abs() < 1e-14);
    assert!((inverse[1][1] - 2.0 / det).abs() < 1e-14);
    // Components outside the leading dim x dim block are zero.
    assert_eq!(inverse[2], [0.0; MAX_DIM]);
    assert_eq!(inverse[0][2], 0.0);
}

#[test]
fn test_try_inverse_rejects_non_finite_components() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let metric = metric_from_rows(&[&[1.0, 0.0], &[0.0, bad]]);
        let error = metric.try_inverse().expect_err("non-finite metric");
        assert_invalid_input(&error, "metric");
        assert!(error.to_string().contains("not finite"), "{error}");
    }
}

#[test]
fn test_try_inverse_validates_dimension() {
    let mut oversized = MetricTensorND::euclidean(MAX_DIM);
    oversized.dim = MAX_DIM + 1;
    assert_invalid_input(&oversized.try_inverse().expect_err("dim > MAX_DIM"), "dim");
    // dim == 0 is the empty metric: its inverse is the empty (zero) matrix.
    let mut empty = MetricTensorND::euclidean(MAX_DIM);
    empty.dim = 0;
    assert_eq!(
        empty.try_inverse().expect("empty metric"),
        [[0.0; MAX_DIM]; MAX_DIM]
    );
}

#[test]
fn test_determinant_of_degenerate_dimensions_does_not_panic() {
    let mut metric = MetricTensorND::euclidean(MAX_DIM);
    metric.dim = 0;
    assert_eq!(metric.determinant(), 1.0, "empty determinant is 1");
    assert_eq!(metric.volume_element(), 1.0);
    metric.dim = MAX_DIM + 1;
    assert!(
        metric.determinant().is_nan(),
        "oversized dim has no determinant"
    );
    // The supported dimensions are unchanged.
    assert!((MetricTensorND::euclidean(3).determinant() - 1.0).abs() < 1e-15);
    assert!((MetricTensorND::minkowski().determinant() + 1.0).abs() < 1e-15);
}

#[test]
#[should_panic(expected = "singular metric tensor")]
fn test_inverse_still_panics_on_singular_metric_as_documented() {
    let metric = metric_from_rows(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let _ = metric.inverse();
}

#[test]
fn test_raise_index_reports_singular_metric() {
    let singular = metric_from_rows(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let mut v = [0.0; MAX_DIM];
    v[0] = 1.0;
    v[1] = 2.0;
    assert_singular_metric(&singular.raise_index(&v).expect_err("singular"));
    let regular = metric_from_rows(&[&[2.0, 0.5], &[0.5, 3.0]]);
    let raised = regular
        .raise_index(&regular.lower_index(&v))
        .expect("regular");
    assert!((raised[0] - 1.0).abs() < 1e-13 && (raised[1] - 2.0).abs() < 1e-13);
}

// ── Functions built on the inverse metric ─────────────────────────────────────

#[test]
fn test_polar_metric_at_origin_is_an_error_not_a_panic() {
    let origin = [0.0; MAX_DIM];
    let h = 1e-5;
    assert_singular_metric(
        &LeviCivitaConnection::from_metric_fn(&polar_metric, &origin, 2, h)
            .map(|_| ())
            .expect_err("Levi-Civita connection at r = 0"),
    );
    assert_singular_metric(
        &RiemannTensor::from_metric_fn(&polar_metric, &origin, 2, h)
            .map(|_| ())
            .expect_err("Riemann tensor at r = 0"),
    );
    assert_singular_metric(
        &RicciTensor::from_metric_fn(&polar_metric, &origin, 2, h)
            .map(|_| ())
            .expect_err("Ricci tensor at r = 0"),
    );
    assert_singular_metric(
        &ricci_scalar_from_metric_fn(&polar_metric, &origin, 2, h).expect_err("Ricci scalar"),
    );
    assert_singular_metric(
        &kretschner_scalar(&polar_metric, &origin, 2, h).expect_err("Kretschner scalar"),
    );
    assert_singular_metric(
        &gauss_bonnet_2d_integrand(&polar_metric, &origin, h).expect_err("Gauss-Bonnet integrand"),
    );
}

#[test]
fn test_polar_metric_away_from_origin_still_works() {
    let r = 2.0;
    let mut point = [0.0; MAX_DIM];
    point[0] = r;
    point[1] = 1.0;
    let conn = LeviCivitaConnection::from_metric_fn(&polar_metric, &point, 2, 1e-5)
        .expect("regular point");
    assert!(
        (conn.gamma(0, 1, 1) + r).abs() < 1e-6,
        "Γ^r_θθ = {}",
        conn.gamma(0, 1, 1)
    );
    assert!(
        (conn.gamma(1, 0, 1) - 1.0 / r).abs() < 1e-6,
        "Γ^θ_rθ = {}",
        conn.gamma(1, 0, 1)
    );
    // The plane is flat: R = 0 and the Kretschner scalar vanishes.
    let ricci = ricci_scalar_from_metric_fn(&polar_metric, &point, 2, 1e-4).expect("regular");
    assert!(ricci.abs() < 1e-3, "R = {ricci}");
    let kretschner = kretschner_scalar(&polar_metric, &point, 2, 1e-4).expect("regular");
    assert!(kretschner.abs() < 1e-3, "K = {kretschner}");
}

#[test]
fn test_singular_neighbour_of_finite_difference_stencil_is_reported() {
    // At r = h the metric is regular but the stencil point r - h is the
    // origin: the Riemann tensor needs the connection there.
    let h = 1e-3;
    let mut point = [0.0; MAX_DIM];
    point[0] = h;
    // The connection only inverts the metric at `point` itself.
    assert!(LeviCivitaConnection::from_metric_fn(&clamped_polar_metric, &point, 2, h).is_ok());
    let riemann = RiemannTensor::from_metric_fn(&clamped_polar_metric, &point, 2, h);
    assert_singular_metric(&riemann.map(|_| ()).expect_err("stencil reaches r = 0"));
}

#[test]
fn test_connection_validates_dimension_and_step() {
    let point = [0.5, 0.5, 0.0, 0.0];
    // dim > MAX_DIM used to index past the fixed-size arrays and panic.
    for dim in [MAX_DIM + 1, 12] {
        let error = LeviCivitaConnection::from_metric_fn(&flat_2d_metric, &point, dim, 1e-4)
            .map(|_| ())
            .expect_err("dim > MAX_DIM");
        assert_invalid_input(&error, "dim");
        let error = RiemannTensor::from_metric_fn(&flat_2d_metric, &point, dim, 1e-4)
            .map(|_| ())
            .expect_err("dim > MAX_DIM");
        assert_invalid_input(&error, "dim");
    }
    for bad_h in [0.0, f64::NAN, f64::INFINITY] {
        let error = LeviCivitaConnection::from_metric_fn(&flat_2d_metric, &point, 2, bad_h)
            .map(|_| ())
            .expect_err("bad step");
        assert_invalid_input(&error, "h");
    }
    // A negative step is harmless: the central difference is symmetric.
    let backwards =
        LeviCivitaConnection::from_metric_fn(&polar_metric, &[2.0, 1.0, 0.0, 0.0], 2, -1e-5)
            .expect("negative step is fine");
    assert!((backwards.gamma(0, 1, 1) + 2.0).abs() < 1e-6);
}

#[test]
fn test_ricci_scalar_and_einstein_tensor_report_singular_metric() {
    let singular = metric_from_rows(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let ricci = RicciTensor {
        components: [[0.0; MAX_DIM]; MAX_DIM],
        dim: 2,
    };
    assert_singular_metric(&ricci_scalar(&singular, &ricci).expect_err("singular"));
    assert_singular_metric(&einstein_tensor(&singular, &ricci).expect_err("singular"));
    // dim above MAX_DIM through the public field is an error, not an index panic.
    let mut oversized = MetricTensorND::euclidean(MAX_DIM);
    oversized.dim = MAX_DIM + 1;
    assert_invalid_input(&ricci_scalar(&oversized, &ricci).expect_err("dim"), "dim");
}

#[test]
fn test_weyl_tensor_reports_degenerate_inputs() {
    // The spherical metric of flat 3-space is singular at r = 0 ...
    let at_centre = [0.0, 1.0, 0.5, 0.0];
    assert_singular_metric(
        &weyl_tensor(&spherical_metric, &at_centre, 3, 1e-5).expect_err("r = 0"),
    );
    // ... and on the polar axis θ = 0 (sin θ = 0 zeroes g_φφ).
    let on_axis = [2.0, 0.0, 0.5, 0.0];
    assert_singular_metric(&weyl_tensor(&spherical_metric, &on_axis, 3, 1e-5).expect_err("θ = 0"));
    // The Weyl tensor needs dim >= 3: this used to be an assert! panic.
    let point = [0.5, 0.5, 0.0, 0.0];
    assert_invalid_input(
        &weyl_tensor(&flat_2d_metric, &point, 2, 1e-4).expect_err("dim 2"),
        "dim",
    );
    // Away from the singularities flat space has vanishing Weyl curvature.
    let regular = [2.0, 1.0, 0.5, 0.0];
    let weyl = weyl_tensor(&spherical_metric, &regular, 3, 1e-4).expect("regular point");
    let largest = weyl
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    assert!(largest < 1e-5, "flat space has C = 0, got {largest}");
}

#[test]
fn test_geodesic_integration_stops_at_the_singularity() {
    let mut start = [0.0; MAX_DIM];
    start[0] = 0.15;
    let mut velocity = [0.0; MAX_DIM];
    velocity[0] = -1.0;
    let cfg = GeodesicConfig {
        dim: 2,
        dt: 0.1,
        steps: 5,
        h_christoffel: 1e-4,
    };
    let error = geodesic_equation_solve(&clamped_polar_metric, &start, &velocity, &cfg)
        .expect_err("the radial geodesic reaches r <= 0 within two steps");
    assert_singular_metric(&error);
    // The same integration that stays in r > 0 succeeds.
    let mut outward = [0.0; MAX_DIM];
    outward[0] = 1.0;
    start[0] = 1.0;
    let trajectory = geodesic_equation_solve(&clamped_polar_metric, &start, &outward, &cfg)
        .expect("outgoing radial geodesic");
    assert_eq!(trajectory.len(), cfg.steps + 1);
    let (final_x, _) = trajectory[cfg.steps];
    assert!((final_x[0] - 1.5).abs() < 1e-6, "r = {}", final_x[0]);
    // An invalid dimension in the configuration is an error, not an index panic.
    let bad_cfg = GeodesicConfig {
        dim: MAX_DIM + 1,
        ..cfg
    };
    assert_invalid_input(
        &geodesic_equation_solve(&clamped_polar_metric, &start, &outward, &bad_cfg)
            .expect_err("dim > MAX_DIM"),
        "dim",
    );
}

#[test]
fn test_parallel_transport_and_killing_check_report_singular_points() {
    let origin = [0.0; MAX_DIM];
    let mut ahead = [0.0; MAX_DIM];
    ahead[0] = 0.5;
    let curve = [origin, ahead];
    let mut vector = [0.0; MAX_DIM];
    vector[0] = 1.0;
    // The connection used for the first segment is evaluated at the origin.
    assert_singular_metric(
        &parallel_transport_nd(&polar_metric, 2, &curve, &vector, 1e-5).expect_err("origin"),
    );
    let xi_fn = |_p: &[f64; MAX_DIM]| -> [f64; MAX_DIM] {
        let mut xi = [0.0; MAX_DIM];
        xi[1] = 1.0;
        xi
    };
    assert_singular_metric(
        &killing_equation_violation(&polar_metric, &xi_fn, &origin, 2, 1e-5).expect_err("origin"),
    );
    // ∂/∂θ is a Killing field of the polar metric away from the origin.
    let mut regular = [0.0; MAX_DIM];
    regular[0] = 1.5;
    regular[1] = 0.3;
    let violation = killing_equation_violation(&polar_metric, &xi_fn, &regular, 2, 1e-5)
        .expect("regular point");
    assert!(violation < 1e-4, "∂θ is Killing, violation = {violation}");
}

#[test]
fn test_geodesic_deviation_reports_singular_points() {
    let origin = [0.0; MAX_DIM];
    let mut velocity = [0.0; MAX_DIM];
    velocity[0] = 1.0;
    let geodesic = [(origin, velocity), (origin, velocity)];
    let xi0 = [0.0; MAX_DIM];
    let dxi0 = [0.0; MAX_DIM];
    let cfg = DeviationConfig {
        dim: 2,
        dt: 0.1,
        h: 1e-5,
    };
    assert_singular_metric(
        &geodesic_deviation(&polar_metric, &geodesic, &xi0, &dxi0, &cfg).expect_err("origin"),
    );
}

// ── RiemannianMetric::christoffel_111 ─────────────────────────────────────────

/// A parametrised surface together with its analytic derivatives.
struct Surface {
    r: fn(f64, f64) -> Vec3,
    ru: fn(f64, f64) -> Vec3,
    rv: fn(f64, f64) -> Vec3,
    ruu: fn(f64, f64) -> Vec3,
    ruv: fn(f64, f64) -> Vec3,
    rvv: fn(f64, f64) -> Vec3,
}

/// The skewed surface of the audit: `r_uu = r_vv = 0`, `F = 0.5 + u v ≠ 0`.
fn skewed_surface() -> Surface {
    Surface {
        r: |u, v| [u + 0.5 * v, v, u * v],
        ru: |_, v| [1.0, 0.0, v],
        rv: |u, _| [0.5, 1.0, u],
        ruu: |_, _| [0.0, 0.0, 0.0],
        ruv: |_, _| [0.0, 0.0, 1.0],
        rvv: |_, _| [0.0, 0.0, 0.0],
    }
}

/// `z = u² + u v`: here `r_uu = (0, 0, 2) ≠ 0`, so Γ^1_11 is not identically
/// zero (Γ_{11,1} = 2(2u + v), Γ_{11,2} = 2u).
fn quadratic_surface() -> Surface {
    Surface {
        r: |u, v| [u, v, u * u + u * v],
        ru: |u, v| [1.0, 0.0, 2.0 * u + v],
        rv: |u, _| [0.0, 1.0, u],
        ruu: |_, _| [0.0, 0.0, 2.0],
        ruv: |_, _| [0.0, 0.0, 1.0],
        rvv: |_, _| [0.0, 0.0, 0.0],
    }
}

/// A surface for which every derivative is non-zero.
fn bumpy_surface() -> Surface {
    Surface {
        r: |u, v| [u + 0.2 * v.sin(), v + 0.3 * u * u, 0.5 * u * v + u.sin()],
        ru: |u, v| [1.0, 0.6 * u, 0.5 * v + u.cos()],
        rv: |u, v| [0.2 * v.cos(), 1.0, 0.5 * u],
        ruu: |u, _| [0.0, 0.6, -u.sin()],
        ruv: |_, _| [0.0, 0.0, 0.5],
        rvv: |_, v| [-0.2 * v.sin(), 0.0, 0.0],
    }
}

fn first_fundamental_form(surface: &Surface, u: f64, v: f64) -> RiemannianMetric {
    RiemannianMetric::from_tangents((surface.ru)(u, v), (surface.rv)(u, v))
}

/// Γ^1_11 = ½ g^11 E_u + g^12 (F_u − ½ E_v) from central differences of the
/// first fundamental form (step `h`), independent of `christoffel_111`.
fn christoffel_111_from_fundamental_form(surface: &Surface, u: f64, v: f64, h: f64) -> f64 {
    let at = |du: f64, dv: f64| first_fundamental_form(surface, u + du, v + dv);
    let e_u = (at(h, 0.0).e - at(-h, 0.0).e) / (2.0 * h);
    let e_v = (at(0.0, h).e - at(0.0, -h).e) / (2.0 * h);
    let f_u = (at(h, 0.0).f - at(-h, 0.0).f) / (2.0 * h);
    let (g11, g12, _g22) = at(0.0, 0.0).inverse();
    0.5 * g11 * e_u + g12 * (f_u - 0.5 * e_v)
}

fn christoffel_111_of(surface: &Surface, u: f64, v: f64) -> f64 {
    first_fundamental_form(surface, u, v).christoffel_111(
        (surface.ruu)(u, v),
        (surface.ruv)(u, v),
        (surface.rvv)(u, v),
        (surface.ru)(u, v),
        (surface.rv)(u, v),
    )
}

const SAMPLE_POINTS: [(f64, f64); 5] = [
    (0.7, -0.4),
    (-0.35, 0.8),
    (1.2, 0.3),
    (0.1, 1.5),
    (-0.9, -0.6),
];

#[test]
fn test_christoffel_111_matches_central_differences_of_the_fundamental_form() {
    for surface in [skewed_surface(), quadratic_surface(), bumpy_surface()] {
        for &(u, v) in &SAMPLE_POINTS {
            let expected = christoffel_111_from_fundamental_form(&surface, u, v, 1e-5);
            let actual = christoffel_111_of(&surface, u, v);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Γ^1_11({u}, {v}) = {actual}, central differences give {expected}"
            );
        }
    }
}

#[test]
fn test_christoffel_111_agrees_with_the_surface_symbols_helper() {
    // christoffel_symbols_numerical raises the last index of the first-kind
    // symbols with the inverse metric; Γ^1_11 must be its [0][0][0] entry.
    for surface in [quadratic_surface(), bumpy_surface()] {
        for &(u, v) in &SAMPLE_POINTS {
            let gamma = christoffel_symbols_numerical(&surface.r, u, v, 1e-4);
            let actual = christoffel_111_of(&surface, u, v);
            assert!(
                (actual - gamma[0][0][0]).abs() < 1e-5,
                "Γ^1_11({u}, {v}) = {actual}, helper gives {}",
                gamma[0][0][0]
            );
        }
    }
}

#[test]
fn test_christoffel_111_closed_form_on_the_quadratic_surface() {
    // E = 1 + (2u+v)², F = u (2u+v), G = 1 + u²;  Γ_{11,1} = 2(2u+v),
    // Γ_{11,2} = 2u, so Γ^1_11 = g^11 · 2(2u+v) + g^12 · 2u with
    // g^11 = G/D and g^12 = -F/D, D = EG − F² = 1 + u² + (2u+v)².
    let surface = quadratic_surface();
    for &(u, v) in &SAMPLE_POINTS {
        let p = 2.0 * u + v;
        let d = 1.0 + u * u + p * p;
        let expected = (1.0 + u * u) / d * 2.0 * p - u * p / d * 2.0 * u;
        let actual = christoffel_111_of(&surface, u, v);
        assert!(
            (actual - expected).abs() < 1e-12,
            "Γ^1_11({u}, {v}) = {actual}, closed form {expected}"
        );
        assert!(
            actual.abs() > 1e-3,
            "must not be identically zero at ({u}, {v})"
        );
    }
}

#[test]
fn test_christoffel_111_vanishes_on_the_skewed_surface() {
    // r_uu = 0, so both first-kind symbols Γ_{11,1} and Γ_{11,2} vanish.  The
    // old formula returned g^12 · r_u·r_uv = g^12 · v here, i.e. not zero.
    let surface = skewed_surface();
    for &(u, v) in &SAMPLE_POINTS {
        assert!(
            christoffel_111_of(&surface, u, v).abs() < 1e-14,
            "at ({u}, {v})"
        );
        let old_formula_value = {
            let metric = first_fundamental_form(&surface, u, v);
            let (_, g12, _) = metric.inverse();
            g12 * v
        };
        assert!(
            old_formula_value.abs() > 1e-3,
            "the sample point must distinguish the correct value from g^12·v"
        );
    }
}
