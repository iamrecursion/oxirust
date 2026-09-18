//! Regression tests for the B-spline / NURBS constructors, the derivative
//! guards, least-squares surface fitting and knot refinement.

use crate::Error;
use crate::bspline::{
    BsplineBasis, BsplineCurve, BsplineSurface, KnotVector, NurbsCurve, NurbsSurface,
};
use std::f64::consts::PI;

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

// ── derivatives above the degree ────────────────────────────────────────────

#[test]
fn curve_derivatives_above_the_degree_are_zero() {
    // Degree 1: curvature (k = 2) used to index `ders[2]` of a 2-row table.
    let line =
        BsplineCurve::clamped(1, vec![[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 3.0, 1.0]]).unwrap();
    assert_eq!(line.curvature(0.5), 0.0);
    assert_eq!(line.eval_deriv(0.3, 2), [0.0; 3]);
    assert_eq!(line.eval_deriv(0.3, 7), [0.0; 3]);
    assert_eq!(line.eval_deriv(0.3, usize::MAX), [0.0; 3]);
    // The first derivative is still the (piecewise constant) velocity: the knot
    // vector is [0, 0, 0.5, 1, 1], so each segment spans half the domain.
    let d_first = line.eval_deriv(0.25, 1);
    let d_second = line.eval_deriv(0.75, 1);
    for (got, want) in d_first.iter().zip([4.0, 2.0, 0.0]) {
        assert!(
            (got - want).abs() < 1e-12,
            "first segment velocity {d_first:?}"
        );
    }
    for (got, want) in d_second.iter().zip([2.0, 4.0, 2.0]) {
        assert!(
            (got - want).abs() < 1e-12,
            "second segment velocity {d_second:?}"
        );
    }

    // Degree 2: torsion (k = 3) used to index `ders[3]` of a 3-row table.
    let quadratic = BsplineCurve::clamped(
        2,
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.5],
            [2.0, 0.0, 1.0],
            [3.0, 2.0, 0.0],
            [4.0, 0.0, 0.3],
        ],
    )
    .unwrap();
    for t in [0.0, 0.2, 0.5, 0.8, 1.0] {
        assert_eq!(
            quadratic.torsion(t),
            0.0,
            "torsion of a degree-2 curve at {t}"
        );
        assert_eq!(quadratic.eval_deriv(t, 3), [0.0; 3]);
        assert_eq!(quadratic.eval_deriv(t, 100), [0.0; 3]);
        assert!(quadratic.curvature(t).is_finite());
    }

    // A twisted cubic keeps a genuine, finite torsion.
    let cubic = BsplineCurve::clamped(
        3,
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
    )
    .unwrap();
    let tau = cubic.torsion(0.5);
    assert!(tau.is_finite() && tau.abs() > 1e-6, "cubic torsion {tau}");
    assert_eq!(cubic.eval_deriv(0.5, 4), [0.0; 3]);
}

#[test]
fn basis_derivatives_above_the_degree_are_zero() {
    let basis = BsplineBasis::clamped(2, 5).unwrap();
    for t in [0.0, 0.3, 0.5, 0.9, 1.0] {
        for i in 0..5 {
            assert_eq!(basis.eval_deriv(i, t, 3), 0.0, "N_{i}'''({t})");
            assert_eq!(basis.eval_deriv(i, t, usize::MAX), 0.0);
        }
        // The partition of unity is constant, so the derivatives sum to zero.
        for k in 1..=2 {
            let sum: f64 = (0..5).map(|i| basis.eval_deriv(i, t, k)).sum();
            assert!(
                sum.abs() < 1e-9,
                "sum of order-{k} derivatives at {t} = {sum}"
            );
        }
    }
    // The order-2 derivative of a quadratic basis function is not identically 0.
    let second: f64 = (0..5).map(|i| basis.eval_deriv(i, 0.2, 2).abs()).sum();
    assert!(second > 1.0, "second derivatives vanished: {second}");

    // Every N_i'(t) is the derivative of that very function (not of N_{i-p}):
    // compare against central differences of `eval`, and with the closed forms
    // of the three quadratic Bezier basis functions.
    let h = 1e-6;
    for t in [0.1, 0.35, 0.5, 0.8] {
        for i in 0..5 {
            let numeric = (basis.eval(i, t + h) - basis.eval(i, t - h)) / (2.0 * h);
            let analytic = basis.eval_deriv(i, t, 1);
            assert!(
                (numeric - analytic).abs() < 1e-6,
                "N_{i}'({t}): analytic {analytic} vs numeric {numeric}"
            );
        }
    }
    let bezier = BsplineBasis::clamped(2, 3).unwrap();
    for t in [0.0, 0.25, 0.6, 0.99] {
        assert!((bezier.eval_deriv(0, t, 1) + 2.0 * (1.0 - t)).abs() < 1e-12);
        assert!((bezier.eval_deriv(1, t, 1) - (2.0 - 4.0 * t)).abs() < 1e-12);
        assert!((bezier.eval_deriv(2, t, 1) - 2.0 * t).abs() < 1e-12);
        assert!((bezier.eval_deriv(0, t, 2) - 2.0).abs() < 1e-12);
        assert!((bezier.eval_deriv(1, t, 2) + 4.0).abs() < 1e-12);
        assert!((bezier.eval_deriv(2, t, 2) - 2.0).abs() < 1e-12);
    }

    let constant = BsplineBasis::new(0, KnotVector::new(vec![0.0, 0.5, 1.0]), 2);
    assert_eq!(constant.eval_deriv(0, 0.25, 1), 0.0);
    assert_eq!(constant.eval_deriv(1, 0.75, 1), 0.0);
}

#[test]
fn degree_zero_directions_have_zero_derivatives() {
    let curve = BsplineCurve::new(
        0,
        KnotVector::new(vec![0.0, 0.5, 1.0]),
        vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    );
    assert_eq!(curve.eval(0.25), [1.0, 0.0, 0.0]);
    assert_eq!(curve.eval(0.75), [0.0, 1.0, 0.0]);
    assert_eq!(curve.tangent(0.25), [0.0; 3]);
    assert_eq!(curve.curvature(0.25), 0.0);
    assert_eq!(curve.torsion(0.25), 0.0);
    assert_eq!(curve.principal_normal(0.25), [0.0; 3]);

    // Degree 0 in both directions: both partial derivatives vanish (the
    // implementation used to index row 1 of a one-row table).
    let kv = KnotVector::new(vec![0.0, 0.5, 1.0]);
    let net = vec![
        vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    ];
    let flat = BsplineSurface::new(0, 0, kv.clone(), kv.clone(), net.clone());
    assert_eq!(flat.eval_du(0.25, 0.25), [0.0; 3]);
    assert_eq!(flat.eval_dv(0.25, 0.25), [0.0; 3]);
    assert_eq!(flat.normal(0.25, 0.25), [0.0, 0.0, 1.0]);
    assert!(flat.gaussian_curvature(0.25, 0.25).is_finite());

    // Mixed degrees: only the degree-0 direction has a zero partial.
    let linear_v = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]);
    let mixed = BsplineSurface::new(0, 1, kv, linear_v, net);
    assert_eq!(mixed.eval_du(0.25, 0.4), [0.0; 3]);
    let dv = mixed.eval_dv(0.25, 0.4);
    for (got, want) in dv.iter().zip([0.0, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-12, "dS/dv = {dv:?}");
    }
}

// ── constructors ────────────────────────────────────────────────────────────

#[test]
fn clamped_uniform_rejects_too_few_control_points() {
    assert!(matches!(
        KnotVector::clamped_uniform(3, 3),
        Err(Error::TooFewPoints {
            required: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        KnotVector::clamped_uniform(0, 0),
        Err(Error::TooFewPoints {
            required: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        KnotVector::clamped_uniform(0, 3),
        Err(Error::TooFewPoints {
            required: 4,
            actual: 0
        })
    ));
    assert!(matches!(
        KnotVector::clamped_uniform(2, usize::MAX),
        Err(Error::TooFewPoints { .. })
    ));

    // Valid sizes keep the documented layout.
    let bezier = KnotVector::clamped_uniform(4, 3).unwrap();
    assert_eq!(bezier.knots, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let uniform = KnotVector::clamped_uniform(5, 2).unwrap();
    let want = [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0];
    assert_eq!(uniform.knots.len(), want.len());
    for (got, w) in uniform.knots.iter().zip(want) {
        assert!((got - w).abs() < 1e-15, "{:?}", uniform.knots);
    }
    let single = KnotVector::clamped_uniform(1, 0).unwrap();
    assert_eq!(single.knots, vec![0.0, 1.0]);
}

#[test]
fn clamped_basis_and_curve_constructors_report_errors() {
    assert!(matches!(
        BsplineBasis::clamped(3, 3),
        Err(Error::TooFewPoints {
            required: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        BsplineBasis::clamped(1, 0),
        Err(Error::TooFewPoints {
            required: 2,
            actual: 0
        })
    ));
    let three = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]];
    assert!(matches!(
        BsplineCurve::clamped(3, three.clone()),
        Err(Error::TooFewPoints {
            required: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        BsplineCurve::clamped(2, Vec::new()),
        Err(Error::TooFewPoints {
            required: 3,
            actual: 0
        })
    ));
    // The boundary case degree + 1 control points is a Bezier curve.
    let bezier = BsplineCurve::clamped(2, three).unwrap();
    assert_eq!(bezier.basis.n_ctrl, 3);
    let mid = bezier.eval(0.5);
    assert!((mid[0] - 1.0).abs() < 1e-12 && (mid[1] - 0.5).abs() < 1e-12);
}

#[test]
fn nurbs_curve_new_validates_its_input() {
    let kv = || KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let cps = || vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]];
    assert!(NurbsCurve::new(2, kv(), cps(), vec![1.0, 0.7, 1.0]).is_ok());

    // Weights that do not match the control points used to hit `assert_eq!`.
    assert!(matches!(
        NurbsCurve::new(2, kv(), cps(), vec![1.0, 1.0]),
        Err(Error::DimensionMismatch {
            lhs: "weights",
            lhs_len: 2,
            rhs: "control_points",
            rhs_len: 3
        })
    ));
    // Fewer control points than degree + 1.
    assert!(matches!(
        NurbsCurve::new(
            3,
            KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            cps(),
            vec![1.0; 3]
        ),
        Err(Error::TooFewPoints {
            required: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        NurbsCurve::new(0, KnotVector::new(vec![0.0]), Vec::new(), Vec::new()),
        Err(Error::TooFewPoints {
            required: 1,
            actual: 0
        })
    ));
    // Knot vector of the wrong length.
    assert!(matches!(
        NurbsCurve::new(
            2,
            KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0]),
            cps(),
            vec![1.0; 3]
        ),
        Err(Error::LengthMismatch {
            expected: 6,
            actual: 5
        })
    ));
    // Weights must be finite and positive.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                NurbsCurve::new(2, kv(), cps(), vec![1.0, bad, 1.0]),
                Err(Error::OutOfRange { name: "weight", .. })
            ),
            "weight {bad} must be rejected"
        );
    }
}

#[test]
fn from_points_and_weights_reports_errors_instead_of_underflowing() {
    // `m - 2 * (degree + 1)` underflowed for fewer than degree + 1 points.
    let two = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(matches!(
        NurbsCurve::from_points_and_weights(two.clone(), vec![1.0, 1.0], 2),
        Err(Error::TooFewPoints {
            required: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        NurbsCurve::from_points_and_weights(Vec::new(), Vec::new(), 1),
        Err(Error::TooFewPoints {
            required: 2,
            actual: 0
        })
    ));
    assert!(matches!(
        NurbsCurve::from_points_and_weights(two.clone(), vec![1.0], 1),
        Err(Error::DimensionMismatch { .. })
    ));

    // With unit weights the NURBS is the clamped uniform B-spline.
    let pts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 1.0],
        [3.0, 1.0, 0.0],
        [4.0, -1.0, 2.0],
    ];
    let nurbs = NurbsCurve::from_points_and_weights(pts.clone(), vec![1.0; 5], 3).unwrap();
    let bspline = BsplineCurve::clamped(3, pts).unwrap();
    assert_eq!(
        nurbs.basis.knot_vector.knots,
        bspline.basis.knot_vector.knots
    );
    for i in 0..=10 {
        let t = i as f64 / 10.0;
        assert!(
            dist(nurbs.eval(t), bspline.eval(t)) < 1e-12,
            "NURBS and B-spline differ at t = {t}"
        );
    }
}

#[test]
fn knot_vector_try_new_validates_knots() {
    let ok = KnotVector::try_new(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]).unwrap();
    assert_eq!(ok.len(), 6);
    assert!(KnotVector::try_new(Vec::new()).unwrap().is_empty());
    assert!(matches!(
        KnotVector::try_new(vec![0.0, 0.6, 0.5, 1.0]),
        Err(Error::OutOfRange { name: "knot", .. })
    ));
    assert!(matches!(
        KnotVector::try_new(vec![0.0, f64::NAN, 1.0]),
        Err(Error::OutOfRange { name: "knot", .. })
    ));
    assert!(matches!(
        KnotVector::try_new(vec![0.0, 0.5, f64::INFINITY]),
        Err(Error::OutOfRange { name: "knot", .. })
    ));
}

#[test]
fn circle_constructors_build_exact_circles_for_every_arc_count() {
    let center = [1.0, -2.0, 0.5];
    let radius = 1.75;
    // 1, 2, 3 and 4 quadratic arcs are chosen from the sweep.
    for sweep in [PI / 3.0, PI, 1.4 * PI, 2.0 * PI] {
        let arc = NurbsCurve::circle_arc(center, radius, 0.2, 0.2 + sweep);
        assert_eq!(arc.control_points.len(), arc.weights.len());
        assert_eq!(
            arc.basis.knot_vector.len(),
            arc.control_points.len() + arc.basis.degree + 1
        );
        for i in 0..=40 {
            let p = arc.eval(i as f64 / 40.0);
            let r = ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2)).sqrt();
            assert!((r - radius).abs() < 1e-9, "sweep {sweep}: r = {r}");
            assert!((p[2] - center[2]).abs() < 1e-12);
        }
    }
    let circle = NurbsCurve::circle(2.0);
    assert_eq!(circle.control_points.len(), 9);
    for i in 0..=36 {
        let p = circle.eval(i as f64 / 36.0);
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!((r - 2.0).abs() < 1e-9, "circle radius {r}");
    }
}

// ── least-squares surface fitting ───────────────────────────────────────────

/// A 5 x 6 control net with deliberately wrong positions and weights 2.0.
fn scrambled_nurbs_surface(degree_u: usize, degree_v: usize) -> NurbsSurface {
    let (n_u, n_v) = (5, 6);
    let net = (0..n_u)
        .map(|i| {
            (0..n_v)
                .map(|j| [10.0 + i as f64, -7.0 * j as f64, (i * j) as f64])
                .collect()
        })
        .collect();
    NurbsSurface::new(
        degree_u,
        degree_v,
        KnotVector::clamped_uniform(n_u, degree_u).unwrap(),
        KnotVector::clamped_uniform(n_v, degree_v).unwrap(),
        net,
        vec![vec![2.0; n_v]; n_u],
    )
}

fn lerp_bilinear(corners: [[f64; 3]; 4], s: f64, t: f64) -> [f64; 3] {
    let [p00, p10, p01, p11] = corners;
    let mut out = [0.0; 3];
    for d in 0..3 {
        out[d] = (1.0 - s) * (1.0 - t) * p00[d]
            + s * (1.0 - t) * p10[d]
            + (1.0 - s) * t * p01[d]
            + s * t * p11[d];
    }
    out
}

#[test]
fn fit_to_grid_reproduces_a_bilinear_patch_from_a_ten_by_ten_grid() {
    let corners = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 3.0, -1.0],
        [2.0, 3.0, 2.0],
    ];
    let grid: Vec<Vec<[f64; 3]>> = (0..10)
        .map(|k| {
            (0..10)
                .map(|l| lerp_bilinear(corners, k as f64 / 9.0, l as f64 / 9.0))
                .collect()
        })
        .collect();
    let mut surf = scrambled_nurbs_surface(3, 3);
    surf.fit_to_grid(&grid).unwrap();
    assert_eq!(surf.control_points.len(), 5);
    assert_eq!(surf.control_points[0].len(), 6);
    for i in 0..=20 {
        for j in 0..=20 {
            let (u, v) = (i as f64 / 20.0, j as f64 / 20.0);
            let got = surf.eval(u, v);
            let want = lerp_bilinear(corners, u, v);
            assert!(
                dist(got, want) < 1e-8,
                "({u}, {v}): {got:?} vs {want:?} (error {})",
                dist(got, want)
            );
        }
    }
}

#[test]
fn fit_to_grid_reproduces_a_twisted_graph_which_nearest_sample_copying_cannot() {
    // z = x * y is bilinear in the (uniform) chord-length parameters, so a
    // bicubic x biquadratic least-squares fit reproduces it exactly.  Copying
    // the nearest grid samples into the control net does not.
    let (k_pts, l_pts) = (12, 9);
    let point = |k: usize, l: usize| {
        let x = -1.0 + 2.0 * k as f64 / (k_pts - 1) as f64;
        let y = 0.5 + l as f64 / (l_pts - 1) as f64;
        [x, y, x * y]
    };
    let grid: Vec<Vec<[f64; 3]>> = (0..k_pts)
        .map(|k| (0..l_pts).map(|l| point(k, l)).collect())
        .collect();
    let mut surf = scrambled_nurbs_surface(3, 2);
    surf.fit_to_grid(&grid).unwrap();
    for i in 0..=16 {
        for j in 0..=16 {
            let (u, v) = (i as f64 / 16.0, j as f64 / 16.0);
            let (x, y) = (-1.0 + 2.0 * u, 0.5 + v);
            let got = surf.eval(u, v);
            assert!(
                dist(got, [x, y, x * y]) < 1e-9,
                "({u}, {v}): {got:?} vs {:?}",
                [x, y, x * y]
            );
        }
    }
}

#[test]
fn fit_to_grid_satisfies_the_normal_equations_for_curved_data() {
    // A half cylinder sampled uniformly in angle and height: the averaged
    // chord-length parameters are exactly uniform, so the residual of the
    // least-squares solution must be orthogonal to every tensor-product basis
    // function evaluated at those parameters.
    let (k_pts, l_pts) = (14, 11);
    let point = |k: usize, l: usize| {
        let theta = PI * k as f64 / (k_pts - 1) as f64;
        [
            theta.cos(),
            theta.sin(),
            2.0 * l as f64 / (l_pts - 1) as f64,
        ]
    };
    let grid: Vec<Vec<[f64; 3]>> = (0..k_pts)
        .map(|k| (0..l_pts).map(|l| point(k, l)).collect())
        .collect();
    let mut surf = scrambled_nurbs_surface(3, 2);
    surf.fit_to_grid(&grid).unwrap();

    let u_params: Vec<f64> = (0..k_pts).map(|k| k as f64 / (k_pts - 1) as f64).collect();
    let v_params: Vec<f64> = (0..l_pts).map(|l| l as f64 / (l_pts - 1) as f64).collect();
    let mut max_residual = 0.0_f64;
    let mut gradient = vec![vec![[0.0_f64; 3]; 6]; 5];
    for (k, &u) in u_params.iter().enumerate() {
        let nu = surf.basis_u.eval_all(u);
        for (l, &v) in v_params.iter().enumerate() {
            let nv = surf.basis_v.eval_all(v);
            let s = surf.eval(u, v);
            let target = grid[k][l];
            let residual = [s[0] - target[0], s[1] - target[1], s[2] - target[2]];
            max_residual = max_residual.max(dist(s, target));
            for (i, &nu_i) in nu.iter().enumerate() {
                for (j, &nv_j) in nv.iter().enumerate() {
                    for (g, r) in gradient[i][j].iter_mut().zip(residual) {
                        *g += nu_i * nv_j * r;
                    }
                }
            }
        }
    }
    assert!(
        max_residual > 1e-4,
        "a 5 x 6 net cannot interpolate a 14 x 11 cylinder grid, residual {max_residual}"
    );
    assert!(max_residual < 0.1, "fit is poor: residual {max_residual}");
    for (i, row) in gradient.iter().enumerate() {
        for (j, g) in row.iter().enumerate() {
            for (d, value) in g.iter().enumerate() {
                assert!(
                    value.abs() < 1e-9,
                    "normal equation ({i}, {j}) coordinate {d} violated: {value}"
                );
            }
        }
    }
}

#[test]
fn fit_to_grid_resets_all_weights_to_one() {
    let grid: Vec<Vec<[f64; 3]>> = (0..8)
        .map(|k| {
            (0..8)
                .map(|l| [k as f64, l as f64, 0.1 * (k * l) as f64])
                .collect()
        })
        .collect();
    let mut surf = scrambled_nurbs_surface(2, 2);
    assert!(surf.weights.iter().flatten().all(|&w| w == 2.0));
    surf.fit_to_grid(&grid).unwrap();
    assert_eq!(surf.weights.len(), 5);
    for row in &surf.weights {
        assert_eq!(row, &vec![1.0; 6]);
    }
}

#[test]
fn fit_to_grid_of_identical_points_is_that_point() {
    let grid = vec![vec![[1.5, -2.0, 0.25]; 7]; 7];
    let mut surf = scrambled_nurbs_surface(3, 3);
    surf.fit_to_grid(&grid).unwrap();
    for &(u, v) in &[(0.0, 0.0), (0.3, 0.8), (1.0, 1.0)] {
        assert!(dist(surf.eval(u, v), [1.5, -2.0, 0.25]) < 1e-10);
    }
}

#[test]
fn fit_to_grid_rejects_invalid_grids_and_keeps_the_surface() {
    let mut surf = scrambled_nurbs_surface(3, 3);
    let before_cps = surf.control_points.clone();
    let before_weights = surf.weights.clone();
    let good_row = |k: usize| -> Vec<[f64; 3]> {
        (0..8)
            .map(|l| [k as f64, l as f64, (k + l) as f64])
            .collect()
    };

    // Empty grid, and fewer rows or columns than the 5 x 6 control net.
    assert!(matches!(
        surf.fit_to_grid(&[]),
        Err(Error::TooFewPoints {
            required: 5,
            actual: 0
        })
    ));
    let few_rows: Vec<Vec<[f64; 3]>> = (0..4).map(good_row).collect();
    assert!(matches!(
        surf.fit_to_grid(&few_rows),
        Err(Error::TooFewPoints {
            required: 5,
            actual: 4
        })
    ));
    let few_cols: Vec<Vec<[f64; 3]>> = (0..8)
        .map(|k| good_row(k).into_iter().take(5).collect())
        .collect();
    assert!(matches!(
        surf.fit_to_grid(&few_cols),
        Err(Error::TooFewPoints {
            required: 6,
            actual: 5
        })
    ));
    // Ragged rows.
    let mut ragged: Vec<Vec<[f64; 3]>> = (0..8).map(good_row).collect();
    ragged[3].pop();
    assert!(matches!(
        surf.fit_to_grid(&ragged),
        Err(Error::LengthMismatch {
            expected: 8,
            actual: 7
        })
    ));
    // Non-finite data.
    let mut nan: Vec<Vec<[f64; 3]>> = (0..8).map(good_row).collect();
    nan[2][5][1] = f64::NAN;
    assert!(matches!(surf.fit_to_grid(&nan), Err(Error::General(_))));
    // A surface without control points cannot be fitted.
    let mut empty = NurbsSurface::new(
        1,
        1,
        KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]),
        KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]),
        Vec::new(),
        Vec::new(),
    );
    assert!(matches!(
        empty.fit_to_grid(&vec![vec![[0.0; 3]; 2]; 2]),
        Err(Error::TooFewPoints { .. })
    ));

    assert_eq!(surf.control_points, before_cps);
    assert_eq!(surf.weights, before_weights);
}

#[test]
fn fit_to_grid_reports_singular_systems() {
    // Rows 1 and 2 coincide, so two chord-length parameters are equal and the
    // 4 x 4 collocation matrix is singular.
    let xs = [0.0, 1.0, 1.0, 2.0];
    let grid: Vec<Vec<[f64; 3]>> = xs
        .iter()
        .map(|&x| (0..4).map(|l| [x, l as f64, 0.0]).collect())
        .collect();
    let kv = KnotVector::clamped_uniform(4, 3).unwrap();
    let net = vec![vec![[0.0; 3]; 4]; 4];
    let mut surf = NurbsSurface::new(3, 3, kv.clone(), kv, net.clone(), vec![vec![3.0; 4]; 4]);
    let err = surf.fit_to_grid(&grid).unwrap_err();
    assert!(
        matches!(&err, Error::General(msg) if msg.contains("singular")),
        "{err}"
    );
    assert_eq!(surf.control_points, net);
    assert_eq!(surf.weights, vec![vec![3.0; 4]; 4]);
}

// ── knot refinement ─────────────────────────────────────────────────────────

/// A genuinely curved bicubic x biquadratic surface with interior knots.
fn curved_surface() -> BsplineSurface {
    let (n_u, n_v) = (6, 5);
    let net = (0..n_u)
        .map(|i| {
            (0..n_v)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    [
                        x + 0.3 * (1.7 * y).sin(),
                        y + 0.2 * (0.9 * x).cos(),
                        0.8 * (0.6 * x).sin() * (0.5 * y).cos() + 0.1 * x * y,
                    ]
                })
                .collect()
        })
        .collect();
    BsplineSurface::new(
        3,
        2,
        KnotVector::clamped_uniform(n_u, 3).unwrap(),
        KnotVector::clamped_uniform(n_v, 2).unwrap(),
        net,
    )
}

const SAMPLES: [f64; 5] = [0.0, 0.3, 0.5, 0.77, 1.0];

fn assert_same_geometry(a: &BsplineSurface, b: &BsplineSurface, tol: f64) {
    for &u in &SAMPLES {
        for &v in &SAMPLES {
            let (pa, pb) = (a.eval(u, v), b.eval(u, v));
            assert!(
                dist(pa, pb) < tol,
                "({u}, {v}): {pa:?} vs {pb:?} (error {})",
                dist(pa, pb)
            );
        }
    }
}

fn sorted_merge(old: &[f64], new: &[f64]) -> Vec<f64> {
    let mut all: Vec<f64> = old.iter().chain(new.iter()).copied().collect();
    all.sort_by(f64::total_cmp);
    all
}

#[test]
fn refine_knots_with_three_knots_per_direction_preserves_the_surface() {
    let surf = curved_surface();
    let new_knots = [0.25, 0.5, 0.75];
    let refined = surf.refine_knots(&new_knots, &new_knots).unwrap();

    // One control row / column per inserted knot, and the sorted merge of knots.
    assert_eq!(refined.control_points.len(), 6 + 3);
    assert!(refined.control_points.iter().all(|row| row.len() == 5 + 3));
    assert_eq!(refined.basis_u.n_ctrl, 9);
    assert_eq!(refined.basis_v.n_ctrl, 8);
    assert_eq!(
        refined.basis_u.knot_vector.knots,
        sorted_merge(&surf.basis_u.knot_vector.knots, &new_knots)
    );
    assert_eq!(
        refined.basis_v.knot_vector.knots,
        sorted_merge(&surf.basis_v.knot_vector.knots, &new_knots)
    );
    assert_same_geometry(&surf, &refined, 1e-10);

    // Derivatives agree as well: the refined surface is the same function.
    for &(u, v) in &[(0.3, 0.5), (0.61, 0.17), (0.9, 0.83)] {
        let (a, b) = (surf.eval_du(u, v), refined.eval_du(u, v));
        assert!(dist(a, b) < 1e-8, "dS/du at ({u}, {v}): {a:?} vs {b:?}");
        let (a, b) = (surf.eval_dv(u, v), refined.eval_dv(u, v));
        assert!(dist(a, b) < 1e-8, "dS/dv at ({u}, {v}): {a:?} vs {b:?}");
    }
}

#[test]
fn refine_knots_inserting_only_one_direction_keeps_the_other_untouched() {
    let surf = curved_surface();
    let along_u = surf.refine_knots(&[0.1, 0.6, 0.9], &[]).unwrap();
    assert_eq!(along_u.control_points.len(), 9);
    assert_eq!(along_u.control_points[0].len(), 5);
    assert_eq!(
        along_u.basis_v.knot_vector.knots,
        surf.basis_v.knot_vector.knots
    );
    assert_same_geometry(&surf, &along_u, 1e-10);
    let along_v = surf.refine_knots(&[], &[0.1, 0.6, 0.9]).unwrap();
    assert_eq!(along_v.control_points.len(), 6);
    assert_eq!(along_v.control_points[0].len(), 8);
    assert_same_geometry(&surf, &along_v, 1e-10);
    // Nothing to insert: the very same surface.
    let same = surf.refine_knots(&[], &[]).unwrap();
    assert_eq!(same.control_points, surf.control_points);
}

#[test]
fn refine_knots_inserting_an_existing_interior_knot_raises_its_multiplicity() {
    let surf = curved_surface();
    // The clamped uniform vectors have interior knots at 1/3 and 2/3.
    let refined = surf
        .refine_knots(&[1.0 / 3.0, 0.4], &[2.0 / 3.0, 1.0 / 3.0])
        .unwrap();
    assert_eq!(
        refined.basis_u.knot_vector.knots,
        sorted_merge(&surf.basis_u.knot_vector.knots, &[1.0 / 3.0, 0.4])
    );
    assert_same_geometry(&surf, &refined, 1e-10);
}

#[test]
fn refine_knots_repeated_knots_raise_the_multiplicity_up_to_the_degree() {
    let surf = curved_surface();
    // Degree 3 in u: the knot may reach multiplicity 3; degree 2 in v: up to 2.
    let refined = surf.refine_knots(&[0.4, 0.4, 0.4], &[0.4, 0.4]).unwrap();
    assert_eq!(refined.control_points.len(), 9);
    assert_eq!(refined.control_points[0].len(), 7);
    assert_eq!(
        refined
            .basis_u
            .knot_vector
            .knots
            .iter()
            .filter(|&&k| k == 0.4)
            .count(),
        3
    );
    assert_same_geometry(&surf, &refined, 1e-10);
    for &v in &SAMPLES {
        let (a, b) = (surf.eval(0.4, v), refined.eval(0.4, v));
        assert!(dist(a, b) < 1e-10, "at the multiple knot u = 0.4, v = {v}");
    }
    // One more would exceed the degree.
    assert!(matches!(
        surf.refine_knots(&[0.4, 0.4, 0.4, 0.4], &[]),
        Err(Error::Unsupported {
            operation: "knot insertion",
            ..
        })
    ));
    assert!(matches!(
        surf.refine_knots(&[], &[0.4, 0.4, 0.4]),
        Err(Error::Unsupported {
            operation: "knot insertion",
            ..
        })
    ));
}

#[test]
fn refine_knots_result_does_not_depend_on_the_insertion_order() {
    let surf = curved_surface();
    let ascending = surf
        .refine_knots(&[0.25, 0.5, 0.75], &[0.2, 0.55, 0.9])
        .unwrap();
    let shuffled = surf
        .refine_knots(&[0.75, 0.25, 0.5], &[0.9, 0.2, 0.55])
        .unwrap();
    assert_eq!(
        ascending.basis_u.knot_vector.knots,
        shuffled.basis_u.knot_vector.knots
    );
    assert_eq!(
        ascending.basis_v.knot_vector.knots,
        shuffled.basis_v.knot_vector.knots
    );
    for (ra, rs) in ascending
        .control_points
        .iter()
        .zip(&shuffled.control_points)
    {
        for (pa, ps) in ra.iter().zip(rs) {
            assert!(dist(*pa, *ps) < 1e-12, "{pa:?} vs {ps:?}");
        }
    }
}

#[test]
fn refine_knots_rejects_invalid_knots() {
    let surf = curved_surface();
    for bad in [f64::NAN, f64::INFINITY, -0.1, 1.5] {
        assert!(
            matches!(
                surf.refine_knots(&[bad], &[]),
                Err(Error::OutOfRange { name: "knot", .. })
            ),
            "u knot {bad}"
        );
        assert!(
            matches!(
                surf.refine_knots(&[], &[bad]),
                Err(Error::OutOfRange { name: "knot", .. })
            ),
            "v knot {bad}"
        );
    }
    // The end knots already have multiplicity degree + 1.
    for boundary in [0.0, 1.0] {
        assert!(matches!(
            surf.refine_knots(&[boundary], &[]),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            surf.refine_knots(&[], &[boundary]),
            Err(Error::Unsupported { .. })
        ));
    }
    // A surface without control points cannot be refined.
    let empty = BsplineSurface {
        basis_u: BsplineBasis::new(1, KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]), 0),
        basis_v: BsplineBasis::new(1, KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]), 0),
        control_points: Vec::new(),
    };
    assert!(matches!(
        empty.refine_knots(&[0.5], &[]),
        Err(Error::TooFewPoints { .. })
    ));
}

#[test]
fn refine_knots_on_a_linear_surface_splits_the_spans() {
    // Degree 1 x 1: every inserted knot may only be new (multiplicity < 1 + 1).
    let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]);
    let net = vec![
        vec![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        vec![[1.0, 0.0, 2.0], [1.0, 1.0, 0.0]],
    ];
    let surf = BsplineSurface::new(1, 1, kv.clone(), kv, net);
    let refined = surf.refine_knots(&[0.3, 0.7], &[0.5]).unwrap();
    assert_eq!(refined.control_points.len(), 4);
    assert_eq!(refined.control_points[0].len(), 3);
    assert_same_geometry(&surf, &refined, 1e-12);
    assert!(matches!(
        refined.refine_knots(&[0.3], &[]),
        Err(Error::Unsupported { .. })
    ));
}
