use oximedia_compat_cv2::contour::{
    fit_ellipse, fit_line, hu_moments, min_area_rect, moments, point_polygon_test,
};
use oximedia_compat_cv2::mat::{Point, Point2f};
use std::f64::consts::PI;

fn pt(x: i32, y: i32) -> Point {
    Point { x, y }
}

/// 100×100 square vertices (CCW)
fn unit_square_100() -> Vec<Point> {
    vec![pt(0, 0), pt(100, 0), pt(100, 100), pt(0, 100)]
}

#[test]
fn test_moments_square() {
    let pts = unit_square_100();
    let m = moments(&pts);
    // m00 = area = 10 000
    assert!((m.m00 - 10000.0).abs() < 10.0, "m00 = {}", m.m00);
    // centroid should be (50, 50)
    if m.m00.abs() > 1e-6 {
        let xc = m.m10 / m.m00;
        let yc = m.m01 / m.m00;
        assert!((xc - 50.0).abs() < 2.0, "centroid x = {}", xc);
        assert!((yc - 50.0).abs() < 2.0, "centroid y = {}", yc);
    }
}

#[test]
fn test_moments_square_m21_numerical() {
    // Verify Steger 1996 coefficients produce the correct integral value.
    // For a 100×100 square: m21 = ∫₀¹⁰⁰∫₀¹⁰⁰ x²y dx dy = (100³/3)·(100²/2)
    let pts = unit_square_100();
    let m = moments(&pts);
    let expected = 100.0_f64.powi(3) / 3.0 * 100.0_f64.powi(2) / 2.0;
    assert!(
        (m.m21 - expected).abs() / expected < 1e-4,
        "m21 = {} expected ≈ {}",
        m.m21,
        expected
    );
    // Symmetry: m12 should equal m21 for a square
    assert!(
        (m.m12 - expected).abs() / expected < 1e-4,
        "m12 = {} expected ≈ {}",
        m.m12,
        expected
    );
}

#[test]
fn test_moments_m30() {
    let pts = unit_square_100();
    let m = moments(&pts);
    // m30 = ∫₀¹⁰⁰∫₀¹⁰⁰ x³ dx dy = (100⁴/4)·100 = 2_500_000_000
    let expected = 100.0_f64.powi(4) / 4.0 * 100.0;
    assert!(
        (m.m30 - expected).abs() / expected < 1e-4,
        "m30 = {} expected ≈ {}",
        m.m30,
        expected
    );
}

#[test]
fn test_hu_moments_finite() {
    let pts = unit_square_100();
    let m = moments(&pts);
    let hu = hu_moments(&m);
    for (i, h) in hu.iter().enumerate() {
        assert!(h.is_finite(), "Hu[{}] is not finite: {}", i, h);
    }
    // hu[0] = nu20 + nu02 >= 0 for a non-degenerate shape
    assert!(hu[0] >= 0.0, "hu[0] = {}", hu[0]);
}

#[test]
fn test_hu_moments_all_computed() {
    // Ensure all 7 invariants are actually computed (not trivially zero).
    let pts = unit_square_100();
    let m = moments(&pts);
    let hu = hu_moments(&m);
    // hu[0] = nu20 + nu02 should be positive for any non-degenerate shape.
    assert!(hu[0] > 0.0, "hu[0] should be positive, got {}", hu[0]);
    // hu[1] = (nu20 - nu02)^2 + 4*nu11^2 = 0 for a symmetric square (nu20==nu02, nu11==0).
    // That's correct: h1 measures departure from circular symmetry, square has h1=0.
    // Verify it's finite and non-negative.
    assert!(
        hu[1] >= 0.0 && hu[1].is_finite(),
        "hu[1] should be >= 0 and finite, got {}",
        hu[1]
    );

    // Use a rectangle (not square) to verify h1 > 0.
    let rect_pts = vec![
        Point { x: 0, y: 0 },
        Point { x: 200, y: 0 },
        Point { x: 200, y: 50 },
        Point { x: 0, y: 50 },
    ];
    let m2 = moments(&rect_pts);
    let hu2 = hu_moments(&m2);
    assert!(hu2[0] > 0.0, "rect hu[0] = {}", hu2[0]);
    assert!(
        hu2[1] > 0.0,
        "rect hu[1] should be > 0 for non-square rectangle, got {}",
        hu2[1]
    );
}

#[test]
fn test_fit_ellipse_circle() {
    // 32 points on a circle of radius 50 centred at (100, 100)
    let pts: Vec<Point> = (0..32)
        .map(|i| {
            let theta = 2.0 * PI * i as f64 / 32.0;
            Point {
                x: (100.0 + 50.0 * theta.cos()).round() as i32,
                y: (100.0 + 50.0 * theta.sin()).round() as i32,
            }
        })
        .collect();
    let r = fit_ellipse(&pts).unwrap();
    assert!(
        (r.center.x - 100.0).abs() < 3.0,
        "center.x = {}",
        r.center.x
    );
    assert!(
        (r.center.y - 100.0).abs() < 3.0,
        "center.y = {}",
        r.center.y
    );
    // Both axes should be approximately 100 (diameter = 2 × radius)
    let a = r.size.0.max(r.size.1);
    let b = r.size.0.min(r.size.1);
    assert!((a - 100.0).abs() < 5.0, "major axis = {}", a);
    assert!((b - 100.0).abs() < 5.0, "minor axis = {}", b);
}

#[test]
fn test_fit_ellipse_too_few_points() {
    let pts = vec![pt(0, 0), pt(1, 0), pt(0, 1)];
    assert!(fit_ellipse(&pts).is_err());
}

#[test]
fn test_fit_ellipse_elongated() {
    // Non-circular ellipse: semi-axes 80×40, centred at (100, 100), no rotation.
    let pts: Vec<Point> = (0..32)
        .map(|i| {
            let theta = 2.0 * PI * i as f64 / 32.0;
            Point {
                x: (100.0 + 80.0 * theta.cos()).round() as i32,
                y: (100.0 + 40.0 * theta.sin()).round() as i32,
            }
        })
        .collect();
    let r = fit_ellipse(&pts).unwrap();
    assert!((r.center.x - 100.0).abs() < 5.0, "cx = {}", r.center.x);
    assert!((r.center.y - 100.0).abs() < 5.0, "cy = {}", r.center.y);
    let major = r.size.0.max(r.size.1);
    let minor = r.size.0.min(r.size.1);
    // major axis ≈ 2*80 = 160, minor axis ≈ 2*40 = 80
    assert!(
        (major - 160.0).abs() < 10.0,
        "major = {} expected ≈ 160",
        major
    );
    assert!(
        (minor - 80.0).abs() < 10.0,
        "minor = {} expected ≈ 80",
        minor
    );
}

#[test]
fn test_fit_line_l2_collinear() {
    use oximedia_compat_cv2::constants::DIST_L2;
    // y = 2x + 3, 10 points
    let pts: Vec<Point> = (0..10)
        .map(|i| Point {
            x: i * 10,
            y: 2 * i * 10 + 3,
        })
        .collect();
    let result = fit_line(&pts, DIST_L2).unwrap();
    let [vx, vy, _x0, _y0] = result;
    // Direction should be approximately (1, 2) / √5
    let expected_vx = 1.0f32 / 5.0f32.sqrt();
    let expected_vy = 2.0f32 / 5.0f32.sqrt();
    // Allow for ±direction (vx might be negative)
    let dot = (vx * expected_vx + vy * expected_vy).abs();
    assert!(dot > 0.95, "direction dot product = {}", dot);
}

#[test]
fn test_fit_line_l1_robust() {
    use oximedia_compat_cv2::constants::{DIST_L1, DIST_L2};
    // 10 points on y = 0.  No outlier: both L1 and L2 should give horizontal direction.
    let pts: Vec<Point> = (0..10).map(|i| Point { x: i * 10, y: 0 }).collect();
    let result_l1 = fit_line(&pts, DIST_L1).unwrap();
    let result_l2 = fit_line(&pts, DIST_L2).unwrap();
    // Both should be horizontal (vy near 0)
    assert!(result_l1[1].abs() < 0.2, "L1 vy = {}", result_l1[1]);
    assert!(result_l2[1].abs() < 0.2, "L2 vy = {}", result_l2[1]);
}

#[test]
fn test_fit_line_l1_collinear() {
    use oximedia_compat_cv2::constants::DIST_L1;
    // Collinear points on y = 2x, 10 points
    let pts: Vec<Point> = (0..10)
        .map(|i| Point {
            x: i * 10,
            y: 2 * i * 10,
        })
        .collect();
    let result = fit_line(&pts, DIST_L1).unwrap();
    let [vx, vy, _x0, _y0] = result;
    let expected_vx = 1.0f32 / 5.0f32.sqrt();
    let expected_vy = 2.0f32 / 5.0f32.sqrt();
    let dot = (vx * expected_vx + vy * expected_vy).abs();
    assert!(dot > 0.90, "L1 collinear dot = {}", dot);
}

#[test]
fn test_fit_line_unsupported_dist_type() {
    let pts = vec![pt(0, 0), pt(1, 1), pt(2, 2)];
    assert!(fit_line(&pts, 99).is_err());
}

#[test]
fn test_fit_line_too_few_points() {
    use oximedia_compat_cv2::constants::DIST_L2;
    assert!(fit_line(&[pt(5, 5)], DIST_L2).is_err());
}

#[test]
fn test_min_area_rect_axis_aligned() {
    // Axis-aligned rectangle 20×10
    let pts = vec![pt(0, 0), pt(20, 0), pt(20, 10), pt(0, 10)];
    let r = min_area_rect(&pts);
    let w = r.size.0.max(r.size.1);
    let h = r.size.0.min(r.size.1);
    assert!((w - 20.0).abs() < 2.0, "width = {}", w);
    assert!((h - 10.0).abs() < 2.0, "height = {}", h);
    assert!((r.center.x - 10.0).abs() < 2.0, "cx = {}", r.center.x);
    assert!((r.center.y - 5.0).abs() < 2.0, "cy = {}", r.center.y);
}

#[test]
fn test_min_area_rect_empty() {
    let r = min_area_rect(&[]);
    assert_eq!(r.size.0, 0.0);
    assert_eq!(r.size.1, 0.0);
}

#[test]
fn test_point_polygon_test_inside_outside() {
    let square = unit_square_100();
    let inside = point_polygon_test(&square, Point2f { x: 50.0, y: 50.0 }, false);
    assert!(inside > 0.0, "inside should be positive, got {}", inside);
    let outside = point_polygon_test(&square, Point2f { x: 150.0, y: 150.0 }, false);
    assert!(outside < 0.0, "outside should be negative, got {}", outside);
}

#[test]
fn test_point_polygon_test_distance() {
    let square = unit_square_100();
    let result = point_polygon_test(&square, Point2f { x: 50.0, y: 50.0 }, true);
    // Centre of 100×100 square → distance to nearest edge = 50
    assert!(result > 0.0, "should be inside (positive), got {}", result);
    assert!(
        (result - 50.0).abs() < 5.0,
        "distance should be ≈50, got {}",
        result
    );
}

#[test]
fn test_point_polygon_test_on_edge() {
    let square = unit_square_100();
    // Point exactly on the bottom edge
    let result = point_polygon_test(&square, Point2f { x: 50.0, y: 0.0 }, false);
    assert!(result == 0.0, "on edge should be 0, got {}", result);
}

#[test]
fn test_moments_degenerate_empty() {
    let m = moments(&[]);
    assert_eq!(m.m00, 0.0);
    assert_eq!(m.nu20, 0.0);
}

#[test]
fn test_moments_degenerate_two_points() {
    let m = moments(&[pt(0, 0), pt(1, 0)]);
    assert_eq!(m.m00, 0.0);
}
