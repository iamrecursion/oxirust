//! Tests for fitLine DIST_HUBER and DIST_FAIR M-estimator IRLS variants.

use oximedia_compat_cv2::contour::fit_line;
use oximedia_compat_cv2::mat::Point;

const DIST_L2: i32 = 2;
const DIST_HUBER: i32 = 7;
const DIST_FAIR: i32 = 5;

fn pt(x: i32, y: i32) -> Point {
    Point { x, y }
}

/// Build a nearly-horizontal line y ≈ slope * x + intercept, then add `n_outliers`
/// large-magnitude outliers at random x positions.
///
/// Uses a deterministic pseudo-random pattern so tests are reproducible without
/// an external RNG dependency.
fn line_with_outliers(
    n_inliers: usize,
    slope: f64,
    intercept: f64,
    n_outliers: usize,
    outlier_magnitude: f64,
) -> Vec<Point> {
    let mut pts: Vec<Point> = (0..n_inliers)
        .map(|i| {
            let x = i as f64;
            let y = slope * x + intercept;
            pt(x.round() as i32, y.round() as i32)
        })
        .collect();

    // Deterministic "pseudo-random" outlier positions using simple LCG pattern.
    let mut seed: u64 = 42;
    for _ in 0..n_outliers {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (seed >> 33) as usize % n_inliers;
        // Alternating sign for outliers.
        let sign = if (seed >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        pts.push(pt(x as i32, (sign * outlier_magnitude) as i32));
    }
    pts
}

/// Recover the fitted slope from [vx, vy, x0, y0].
/// The line passes through (x0, y0) with direction (vx, vy); slope = vy / vx.
fn slope_from_fit(fit: [f32; 4]) -> f64 {
    let (vx, vy) = (fit[0] as f64, fit[1] as f64);
    if vx.abs() < 1e-6 {
        f64::INFINITY
    } else {
        vy / vx
    }
}

/// Perpendicular-distance L1 error from fit line.
fn l1_error(pts: &[Point], fit: [f32; 4]) -> f64 {
    let vx = fit[0] as f64;
    let vy = fit[1] as f64;
    let x0 = fit[2] as f64;
    let y0 = fit[3] as f64;
    pts.iter()
        .map(|p| ((p.x as f64 - x0) * vy - (p.y as f64 - y0) * vx).abs())
        .sum::<f64>()
        / pts.len() as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// DIST_HUBER should recover the true slope within 0.05 even when 10% of
/// points are gross outliers.
#[test]
fn test_fit_line_huber_robust() {
    let true_slope = 0.5_f64;
    let n_inliers = 90;
    let n_outliers = 10; // 10%
    let pts = line_with_outliers(n_inliers, true_slope, 20.0, n_outliers, 200.0);

    let fit = fit_line(&pts, DIST_HUBER).expect("DIST_HUBER fit should succeed");
    let fitted_slope = slope_from_fit(fit);

    assert!(
        (fitted_slope - true_slope).abs() < 0.05,
        "HUBER slope error too large: fitted={:.4} true={:.4}",
        fitted_slope,
        true_slope
    );
}

/// DIST_FAIR should also recover the true slope within 0.05 with 10% outliers.
#[test]
fn test_fit_line_fair_robust() {
    let true_slope = 0.5_f64;
    let n_inliers = 90;
    let n_outliers = 10;
    let pts = line_with_outliers(n_inliers, true_slope, 20.0, n_outliers, 200.0);

    let fit = fit_line(&pts, DIST_FAIR).expect("DIST_FAIR fit should succeed");
    let fitted_slope = slope_from_fit(fit);

    assert!(
        (fitted_slope - true_slope).abs() < 0.05,
        "FAIR slope error too large: fitted={:.4} true={:.4}",
        fitted_slope,
        true_slope
    );
}

/// With outliers present, DIST_HUBER should have strictly smaller mean
/// perpendicular error than DIST_L2 on the inlier subset.
#[test]
fn test_fit_line_huber_vs_l2_with_outliers() {
    let true_slope = 1.0_f64;
    let n_inliers = 80;
    let n_outliers = 20; // 20% – more extreme
    let pts = line_with_outliers(n_inliers, true_slope, 0.0, n_outliers, 300.0);

    let fit_huber = fit_line(&pts, DIST_HUBER).expect("HUBER fit should succeed");
    let fit_l2 = fit_line(&pts, DIST_L2).expect("L2 fit should succeed");

    // Evaluate on inliers only (first n_inliers points).
    let inliers = &pts[..n_inliers];
    let err_huber = l1_error(inliers, fit_huber);
    let err_l2 = l1_error(inliers, fit_l2);

    assert!(
        err_huber < err_l2,
        "HUBER should outperform L2 with 20% outliers: err_huber={:.3} err_l2={:.3}",
        err_huber,
        err_l2
    );
}

/// All four distance types should complete without panic on the same dataset.
#[test]
fn test_fit_line_all_dist_types_no_panic() {
    let pts: Vec<Point> = (0..50).map(|i| pt(i, i / 2 + (i % 3))).collect();

    const DIST_L1: i32 = 1;

    let r_l2 = fit_line(&pts, DIST_L2);
    let r_l1 = fit_line(&pts, DIST_L1);
    let r_huber = fit_line(&pts, DIST_HUBER);
    let r_fair = fit_line(&pts, DIST_FAIR);

    assert!(r_l2.is_ok(), "DIST_L2 failed: {:?}", r_l2);
    assert!(r_l1.is_ok(), "DIST_L1 failed: {:?}", r_l1);
    assert!(r_huber.is_ok(), "DIST_HUBER failed: {:?}", r_huber);
    assert!(r_fair.is_ok(), "DIST_FAIR failed: {:?}", r_fair);

    // All results should be finite.
    for (name, r) in [
        ("L2", r_l2),
        ("L1", r_l1),
        ("HUBER", r_huber),
        ("FAIR", r_fair),
    ] {
        let v = r.unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{} produced non-finite result: {:?}",
            name,
            v
        );
    }
}
