//! Contour analysis: findContours, bounding_rect, contour_area, arc_length,
//! approx_poly_dp, min_enclosing_circle, convex_hull, moments, hu_moments,
//! fit_ellipse, fit_line, min_area_rect, point_polygon_test.
//!
//! Algorithms lifted from the PyO3 cv2-compat layer (`cv2_compat/contours.rs`)
//! and adapted to use the pure-Rust `Mat`/`Point` types.

use crate::{
    constants::{DIST_FAIR, DIST_HUBER, DIST_L1, DIST_L2},
    error::{Cv2Error, Cv2Result},
    mat::{Mat, MatType, Point, Point2f, Rect},
};

// ── Public API ────────────────────────────────────────────────────────────────

/// cv2.findContours — extract contours from a binary image.
///
/// Returns `(contours, hierarchy)` where:
/// * `contours` — each contour is a `Vec<Point>` of boundary pixels.
/// * `hierarchy` — one `[i32; 4]` per contour: `[next, prev, first_child, parent]`.
///
/// Only `RETR_EXTERNAL` and `RETR_LIST` (flat hierarchy) and
/// `CHAIN_APPROX_SIMPLE` / `CHAIN_APPROX_NONE` are recognised.
/// `RETR_CCOMP` and `RETR_TREE` fall back to flat-list behaviour.
///
/// # Errors
/// Returns `UnsupportedDtype` for non-`CV_8UC1` input.
pub fn find_contours(
    src: &Mat,
    _mode: i32,
    _method: i32,
) -> Cv2Result<(Vec<Vec<Point>>, Vec<[i32; 4]>)> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }

    let w = src.cols;
    let h = src.rows;
    let binary: Vec<bool> = src.data.iter().map(|&v| v > 0).collect();

    let raw = trace_contours(&binary, w, h);

    // Convert Vec<(i32,i32)> → Vec<Point>
    let contours: Vec<Vec<Point>> = raw
        .into_iter()
        .map(|c| c.into_iter().map(|(x, y)| Point { x, y }).collect())
        .collect();

    let n = contours.len();
    // Build flat-list (sibling chain) hierarchy
    let hierarchy: Vec<[i32; 4]> = (0..n)
        .map(|i| {
            let next = if i + 1 < n { i as i32 + 1 } else { -1 };
            let prev = if i > 0 { i as i32 - 1 } else { -1 };
            [next, prev, -1, -1]
        })
        .collect();

    Ok((contours, hierarchy))
}

/// cv2.boundingRect — axis-aligned bounding rectangle of a contour.
pub fn bounding_rect(contour: &[Point]) -> Rect {
    if contour.is_empty() {
        return Rect::default();
    }
    let x_min = contour.iter().map(|p| p.x).min().unwrap_or(0);
    let x_max = contour.iter().map(|p| p.x).max().unwrap_or(0);
    let y_min = contour.iter().map(|p| p.y).min().unwrap_or(0);
    let y_max = contour.iter().map(|p| p.y).max().unwrap_or(0);
    Rect {
        x: x_min,
        y: y_min,
        width: x_max - x_min + 1,
        height: y_max - y_min + 1,
    }
}

/// cv2.contourArea — signed shoelace area, absolute value returned.
pub fn contour_area(contour: &[Point]) -> f64 {
    shoelace_area(contour).abs()
}

/// cv2.arcLength — perimeter of a contour.
///
/// If `closed` is `true` the last-to-first segment is included.
pub fn arc_length(contour: &[Point], closed: bool) -> f64 {
    let n = contour.len();
    if n < 2 {
        return 0.0;
    }
    let pairs = if closed { n } else { n - 1 };
    let mut length = 0.0f64;
    for i in 0..pairs {
        let p0 = contour[i];
        let p1 = contour[(i + 1) % n];
        let dx = (p1.x - p0.x) as f64;
        let dy = (p1.y - p0.y) as f64;
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// cv2.approxPolyDP — Douglas-Peucker polygon approximation.
pub fn approx_poly_dp(contour: &[Point], epsilon: f64, _closed: bool) -> Vec<Point> {
    let pts: Vec<(i32, i32)> = contour.iter().map(|p| (p.x, p.y)).collect();
    let simplified = douglas_peucker(&pts, epsilon);
    simplified
        .into_iter()
        .map(|(x, y)| Point { x, y })
        .collect()
}

/// cv2.minEnclosingCircle — smallest enclosing circle of a contour.
///
/// Returns `(centre, radius)`.
pub fn min_enclosing_circle(contour: &[Point]) -> (Point2f, f32) {
    if contour.is_empty() {
        return (Point2f::default(), 0.0);
    }
    if contour.len() == 1 {
        return (
            Point2f {
                x: contour[0].x as f32,
                y: contour[0].y as f32,
            },
            0.0,
        );
    }

    // Welzl's algorithm (iterative, randomized) — O(n) expected
    // For simplicity use the simple enclosing-circle heuristic:
    // find the two farthest points and use their midpoint/half-dist as initial circle,
    // then grow to include all points.
    let (p0, p1) = farthest_pair(contour);
    let cx = (p0.x as f32 + p1.x as f32) / 2.0;
    let cy = (p0.y as f32 + p1.y as f32) / 2.0;
    let dx = (p1.x - p0.x) as f32;
    let dy = (p1.y - p0.y) as f32;
    let mut radius = ((dx * dx + dy * dy) as f32).sqrt() / 2.0;
    let mut centre = Point2f { x: cx, y: cy };

    // Grow circle to include all points
    for p in contour {
        let dx2 = p.x as f32 - centre.x;
        let dy2 = p.y as f32 - centre.y;
        let dist = (dx2 * dx2 + dy2 * dy2).sqrt();
        if dist > radius {
            // Expand: new radius is average, move centre toward p
            let new_radius = (radius + dist) / 2.0;
            let ratio = (new_radius - radius) / dist;
            centre.x += dx2 * ratio;
            centre.y += dy2 * ratio;
            radius = new_radius;
        }
    }

    (centre, radius)
}

/// cv2.convexHull — convex hull of a contour (Graham scan).
pub fn convex_hull(contour: &[Point]) -> Vec<Point> {
    if contour.len() < 3 {
        return contour.to_vec();
    }

    // Find lowest-leftmost point as pivot
    let pivot_idx = contour
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut pts: Vec<Point> = contour.to_vec();
    pts.swap(0, pivot_idx);
    let pivot = pts[0];

    // Sort by polar angle relative to pivot
    pts[1..].sort_unstable_by(|a, b| {
        let cross = (a.x - pivot.x) as i64 * (b.y - pivot.y) as i64
            - (a.y - pivot.y) as i64 * (b.x - pivot.x) as i64;
        // Counter-clockwise: negative cross product means a is CCW before b
        cross.cmp(&0).reverse().then_with(|| {
            let da = dist_sq(pivot, *a);
            let db = dist_sq(pivot, *b);
            da.cmp(&db)
        })
    });

    // Graham scan
    let mut hull: Vec<Point> = Vec::new();
    for &p in &pts {
        while hull.len() >= 2 {
            let n = hull.len();
            let cross = cross_product(hull[n - 2], hull[n - 1], p);
            if cross <= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    hull
}

// ── Shape-analysis structs ────────────────────────────────────────────────────

/// Image moments up to order 3 — matches `cv2.Moments`.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    // Spatial moments
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,
    // Central moments
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,
    // Normalized central moments
    pub nu20: f64,
    pub nu11: f64,
    pub nu02: f64,
    pub nu30: f64,
    pub nu21: f64,
    pub nu12: f64,
    pub nu03: f64,
}

/// A rotated rectangle (used by `fit_ellipse` and `min_area_rect`).
#[derive(Debug, Clone, PartialEq)]
pub struct RotatedRect {
    pub center: Point2f,
    /// Full width and height (not semi-axes).
    pub size: (f32, f32),
    /// Angle in degrees, CCW from +x axis.
    pub angle: f32,
}

// ── Shape-analysis public API ─────────────────────────────────────────────────

/// Compute image moments (up to order 3) for a polygon contour.
///
/// Uses Green's theorem (contour-integral form) with Steger 1996 polynomial
/// coefficients for all moment orders.  Central and normalized central moments
/// are derived analytically from the spatial moments.
///
/// Returns all-zero `Moments` for degenerate input (fewer than 3 vertices or
/// zero enclosed area).
pub fn moments(contour: &[Point]) -> Moments {
    let n = contour.len();
    if n < 3 {
        return zero_moments();
    }

    let mut m00 = 0.0_f64;
    let mut m10 = 0.0_f64;
    let mut m01 = 0.0_f64;
    let mut m20 = 0.0_f64;
    let mut m11 = 0.0_f64;
    let mut m02 = 0.0_f64;
    let mut m30 = 0.0_f64;
    let mut m21 = 0.0_f64;
    let mut m12 = 0.0_f64;
    let mut m03 = 0.0_f64;

    for i in 0..n {
        let xi = contour[i].x as f64;
        let yi = contour[i].y as f64;
        let xj = contour[(i + 1) % n].x as f64;
        let yj = contour[(i + 1) % n].y as f64;
        // Signed cross term for this directed edge (Green's theorem).
        let a = xi * yj - xj * yi;

        m00 += a;
        m10 += (xi + xj) * a;
        m01 += (yi + yj) * a;
        m20 += (xi * xi + xi * xj + xj * xj) * a;
        m11 += (2.0 * xi * yi + xi * yj + xj * yi + 2.0 * xj * yj) * a;
        m02 += (yi * yi + yi * yj + yj * yj) * a;
        // m30 / m03: (x³ + x²x₁ + xx₁² + x₁³) — Steger 1996
        m30 += (xi * xi * xi + xi * xi * xj + xi * xj * xj + xj * xj * xj) * a;
        // m21: Steger 1996 — note both xi+1² and xi² corner terms have coefficient 3
        m21 += (3.0 * xi * xi * yi
            + 2.0 * xi * xj * yi
            + xj * xj * yi
            + xi * xi * yj
            + 2.0 * xi * xj * yj
            + 3.0 * xj * xj * yj)
            * a;
        // m12: symmetric of m21
        m12 += (3.0 * yi * yi * xi
            + 2.0 * yi * yj * xi
            + yj * yj * xi
            + yi * yi * xj
            + 2.0 * yi * yj * xj
            + 3.0 * yj * yj * xj)
            * a;
        m03 += (yi * yi * yi + yi * yi * yj + yi * yj * yj + yj * yj * yj) * a;
    }

    m00 /= 2.0;
    m10 /= 6.0;
    m01 /= 6.0;
    m20 /= 12.0;
    m11 /= 24.0;
    m02 /= 12.0;
    m30 /= 20.0;
    m21 /= 60.0;
    m12 /= 60.0;
    m03 /= 20.0;

    // Signed m00 matches cv2 convention (sign reflects winding direction).
    if m00 == 0.0 {
        return zero_moments();
    }

    let xc = m10 / m00;
    let yc = m01 / m00;

    // Central moments via standard algebraic shift relations.
    let mu20 = m20 - xc * m10;
    let mu11 = m11 - xc * m01;
    let mu02 = m02 - yc * m01;
    let mu30 = m30 - 3.0 * xc * m20 + 2.0 * xc * xc * m10;
    let mu21 = m21 - 2.0 * xc * m11 - yc * m20 + 2.0 * xc * xc * m01;
    let mu12 = m12 - 2.0 * yc * m11 - xc * m02 + 2.0 * yc * yc * m10;
    let mu03 = m03 - 3.0 * yc * m02 + 2.0 * yc * yc * m01;

    // Normalized central moments: nu_pq = mu_pq / |m00|^((p+q)/2 + 1).
    let m00_abs = m00.abs();
    let s2 = m00_abs * m00_abs; // exponent 2 for p+q = 2
    let s25 = s2 * m00_abs.sqrt(); // exponent 2.5 for p+q = 3
    let nu20 = mu20 / s2;
    let nu11 = mu11 / s2;
    let nu02 = mu02 / s2;
    let nu30 = mu30 / s25;
    let nu21 = mu21 / s25;
    let nu12 = mu12 / s25;
    let nu03 = mu03 / s25;

    Moments {
        m00,
        m10,
        m01,
        m20,
        m11,
        m02,
        m30,
        m21,
        m12,
        m03,
        mu20,
        mu11,
        mu02,
        mu30,
        mu21,
        mu12,
        mu03,
        nu20,
        nu11,
        nu02,
        nu30,
        nu21,
        nu12,
        nu03,
    }
}

/// Compute the 7 Hu moment invariants from a `Moments` struct (Hu 1962).
#[must_use]
pub fn hu_moments(m: &Moments) -> [f64; 7] {
    let (nu20, nu11, nu02) = (m.nu20, m.nu11, m.nu02);
    let (nu30, nu21, nu12, nu03) = (m.nu30, m.nu21, m.nu12, m.nu03);

    let h0 = nu20 + nu02;
    let h1 = (nu20 - nu02).powi(2) + 4.0 * nu11.powi(2);
    let h2 = (nu30 - 3.0 * nu12).powi(2) + (3.0 * nu21 - nu03).powi(2);
    let h3 = (nu30 + nu12).powi(2) + (nu21 + nu03).powi(2);
    let h4 =
        (nu30 - 3.0 * nu12) * (nu30 + nu12) * ((nu30 + nu12).powi(2) - 3.0 * (nu21 + nu03).powi(2))
            + (3.0 * nu21 - nu03)
                * (nu21 + nu03)
                * (3.0 * (nu30 + nu12).powi(2) - (nu21 + nu03).powi(2));
    let h5 = (nu20 - nu02) * ((nu30 + nu12).powi(2) - (nu21 + nu03).powi(2))
        + 4.0 * nu11 * (nu30 + nu12) * (nu21 + nu03);
    let h6 =
        (3.0 * nu21 - nu03) * (nu30 + nu12) * ((nu30 + nu12).powi(2) - 3.0 * (nu21 + nu03).powi(2))
            - (nu30 - 3.0 * nu12)
                * (nu21 + nu03)
                * (3.0 * (nu30 + nu12).powi(2) - (nu21 + nu03).powi(2));

    [h0, h1, h2, h3, h4, h5, h6]
}

/// Fitzgibbon–Pilu–Fisher direct least-squares ellipse fit.
///
/// Returns a [`RotatedRect`] whose `size` is `(2*a_axis, 2*b_axis)` (full axes).
///
/// # Errors
/// Returns `UnsupportedFlag` if fewer than 5 points are supplied or if the
/// recovered conic is degenerate.
pub fn fit_ellipse(points: &[Point]) -> Cv2Result<RotatedRect> {
    let n = points.len();
    if n < 5 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "fit_ellipse: need >= 5 points",
            value: n as i32,
        });
    }

    // Build scatter matrix S = DᵀD (6×6 symmetric).
    // D row i = [xi², xi·yi, yi², xi, yi, 1].
    let mut s = [[0.0_f64; 6]; 6];
    for p in points {
        let xi = p.x as f64;
        let yi = p.y as f64;
        let row = [xi * xi, xi * yi, yi * yi, xi, yi, 1.0];
        for r in 0..6 {
            for c in r..6 {
                let v = row[r] * row[c];
                s[r][c] += v;
                if r != c {
                    s[c][r] += v;
                }
            }
        }
    }

    // Invert S via Gauss-Jordan elimination.
    let s_inv = match mat6_inverse(&s) {
        Some(m) => m,
        None => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "fit_ellipse: singular scatter matrix",
                value: 0,
            });
        }
    };

    // Constraint matrix C (6×6): C[0][2]=C[2][0]=2, C[1][1]=-1.
    let mut cmat = [[0.0_f64; 6]; 6];
    cmat[0][2] = 2.0;
    cmat[2][0] = 2.0;
    cmat[1][1] = -1.0;

    // M = S⁻¹ · C
    let m_mat = mat6_mul(&s_inv, &cmat);

    // Find the eigenvector of M with the unique positive eigenvalue; that is
    // the Fitzgibbon ellipse solution satisfying aᵀCa > 0.
    let (evecs, evals) = eigen6_nonsym(&m_mat);

    let mut best_idx = None;
    let mut best_val = 0.0_f64;
    for i in 0..6 {
        if evals[i] > best_val {
            let av = evecs[i];
            // aᵀCa = 2*a[0]*a[2] - a[1]²
            let ctca = 2.0 * av[0] * av[2] - av[1] * av[1];
            if ctca > 0.0 {
                best_val = evals[i];
                best_idx = Some(i);
            }
        }
    }

    let idx = match best_idx {
        Some(i) => i,
        None => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "fit_ellipse: no valid ellipse solution",
                value: 0,
            });
        }
    };

    let av = evecs[idx];
    let (aa, bb, cc, dd, ee, ff) = (av[0], av[1], av[2], av[3], av[4], av[5]);

    // Center from [[2A,B],[B,2C]]·[xc,yc]ᵀ = −[D,E].
    let det = 4.0 * aa * cc - bb * bb;
    if det <= 0.0 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "fit_ellipse: degenerate conic (not an ellipse)",
            value: 0,
        });
    }
    let xc = (bb * ee - 2.0 * cc * dd) / det;
    let yc = (bb * dd - 2.0 * aa * ee) / det;

    // Scale factor M = A·xc² + B·xc·yc + C·yc² − F.
    // (This equals the quadratic form evaluated at the center, with sign chosen
    // so that semi-axes are real; must be positive for a valid ellipse.)
    let scale_m = aa * xc * xc + bb * xc * yc + cc * yc * yc - ff;
    if scale_m <= 0.0 {
        // Try negating: the ellipse solution may be expressed with flipped sign.
        let scale_m_neg = -scale_m;
        if scale_m_neg <= 0.0 {
            return Err(Cv2Error::UnsupportedFlag {
                name: "fit_ellipse: degenerate (non-positive scale)",
                value: 0,
            });
        }
        // Negate the conic coefficients so that scale_m becomes positive.
        let (aa, bb, cc, dd, ee, ff) = (-aa, -bb, -cc, -dd, -ee, -ff);
        let det = 4.0 * aa * cc - bb * bb;
        if det <= 0.0 {
            return Err(Cv2Error::UnsupportedFlag {
                name: "fit_ellipse: degenerate conic after sign flip",
                value: 0,
            });
        }
        let xc = (bb * ee - 2.0 * cc * dd) / det;
        let yc = (bb * dd - 2.0 * aa * ee) / det;
        let scale_m2 = aa * xc * xc + bb * xc * yc + cc * yc * yc - ff;
        if scale_m2 <= 0.0 {
            return Err(Cv2Error::UnsupportedFlag {
                name: "fit_ellipse: degenerate ellipse",
                value: 0,
            });
        }
        let tr = aa + cc;
        let diff = ((aa - cc) * (aa - cc) + bb * bb).sqrt();
        let lambda1 = (tr + diff) / (2.0 * scale_m2);
        let lambda2 = (tr - diff) / (2.0 * scale_m2);
        let a_axis = if lambda1 > 0.0 {
            (1.0 / lambda1).sqrt()
        } else {
            0.0
        };
        let b_axis = if lambda2 > 0.0 {
            (1.0 / lambda2).sqrt()
        } else {
            0.0
        };
        let angle = (0.5 * bb.atan2(aa - cc) * 180.0 / std::f64::consts::PI) as f32;
        return Ok(RotatedRect {
            center: Point2f {
                x: xc as f32,
                y: yc as f32,
            },
            size: (2.0 * a_axis as f32, 2.0 * b_axis as f32),
            angle,
        });
    }

    // Semi-axes from eigenvalues of [[A,B/2],[B/2,C]] / scale_m.
    let tr = aa + cc;
    let diff = ((aa - cc) * (aa - cc) + bb * bb).sqrt();
    let lambda1 = (tr + diff) / (2.0 * scale_m);
    let lambda2 = (tr - diff) / (2.0 * scale_m);
    let a_axis = if lambda1 > 0.0 {
        (1.0 / lambda1).sqrt()
    } else {
        0.0
    };
    let b_axis = if lambda2 > 0.0 {
        (1.0 / lambda2).sqrt()
    } else {
        0.0
    };

    // Half-angle of the conic axis direction (CCW from +x).
    let angle = (0.5 * bb.atan2(aa - cc) * 180.0 / std::f64::consts::PI) as f32;

    Ok(RotatedRect {
        center: Point2f {
            x: xc as f32,
            y: yc as f32,
        },
        size: (2.0 * a_axis as f32, 2.0 * b_axis as f32),
        angle,
    })
}

/// Fit a line to a point set using the specified distance norm.
///
/// Returns `[vx, vy, x0, y0]` where `(x0, y0)` is a point on the line and
/// `(vx, vy)` is the unit direction vector.
///
/// Supported `dist_type` values:
/// - `DIST_L2` (2) — PCA on covariance matrix
/// - `DIST_L1` (1) — IRLS robust regression with L1 weights
/// - `DIST_HUBER` (7) — IRLS with Huber M-estimator weights (c ≈ 1.345 MAD)
/// - `DIST_FAIR` (5) — IRLS with Fair M-estimator weights (c ≈ 1.3998 MAD)
///
/// # Errors
/// Returns `UnsupportedFlag` for unsupported `dist_type` or fewer than 2 points.
pub fn fit_line(points: &[Point], dist_type: i32) -> Cv2Result<[f32; 4]> {
    let n = points.len();
    if n < 2 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "fit_line: need >= 2 points",
            value: n as i32,
        });
    }
    if dist_type == DIST_L2 {
        Ok(fit_line_l2(points))
    } else if dist_type == DIST_L1 {
        Ok(fit_line_l1(points))
    } else if dist_type == DIST_HUBER {
        Ok(fit_line_huber(points))
    } else if dist_type == DIST_FAIR {
        Ok(fit_line_fair(points))
    } else {
        Err(Cv2Error::UnsupportedFlag {
            name: "fit_line: unsupported dist_type",
            value: dist_type,
        })
    }
}

/// Minimum-area enclosing rotated rectangle using the rotating-calipers algorithm.
///
/// Returns a zero-size rect for fewer than 2 input points.
pub fn min_area_rect(points: &[Point]) -> RotatedRect {
    if points.is_empty() {
        return RotatedRect {
            center: Point2f { x: 0.0, y: 0.0 },
            size: (0.0, 0.0),
            angle: 0.0,
        };
    }
    if points.len() == 1 {
        return RotatedRect {
            center: Point2f {
                x: points[0].x as f32,
                y: points[0].y as f32,
            },
            size: (0.0, 0.0),
            angle: 0.0,
        };
    }
    if points.len() == 2 {
        let p0 = points[0];
        let p1 = points[1];
        let cx = (p0.x as f32 + p1.x as f32) / 2.0;
        let cy = (p0.y as f32 + p1.y as f32) / 2.0;
        let dx = (p1.x - p0.x) as f64;
        let dy = (p1.y - p0.y) as f64;
        let len = (dx * dx + dy * dy).sqrt() as f32;
        return RotatedRect {
            center: Point2f { x: cx, y: cy },
            size: (len, 0.0),
            angle: dy.atan2(dx).to_degrees() as f32,
        };
    }

    let hull = convex_hull(points);
    let hn = hull.len();

    let mut min_area = f64::MAX;
    let mut best_rect = RotatedRect {
        center: Point2f { x: 0.0, y: 0.0 },
        size: (0.0, 0.0),
        angle: 0.0,
    };

    for i in 0..hn {
        let hp1 = hull[i];
        let hp2 = hull[(i + 1) % hn];
        let dx = (hp2.x - hp1.x) as f64;
        let dy = (hp2.y - hp1.y) as f64;
        let edge_len = (dx * dx + dy * dy).sqrt();
        if edge_len < 1e-10 {
            continue;
        }
        let ux = dx / edge_len; // unit vector along edge
        let uy = dy / edge_len;

        // Project all hull points onto (ux, uy) and (−uy, ux).
        let mut min_u = f64::MAX;
        let mut max_u = f64::MIN;
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for hp in &hull {
            let px = hp.x as f64;
            let py = hp.y as f64;
            let u = px * ux + py * uy;
            let v = -px * uy + py * ux;
            if u < min_u {
                min_u = u;
            }
            if u > max_u {
                max_u = u;
            }
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }

        let width = max_u - min_u;
        let height = max_v - min_v;
        let area = width * height;

        if area < min_area {
            min_area = area;
            let cu = (min_u + max_u) / 2.0;
            let cv_mid = (min_v + max_v) / 2.0;
            let cx = cu * ux - cv_mid * uy;
            let cy = cu * uy + cv_mid * ux;
            // Angle: CCW from +x, normalized to [−90, 0) to match cv2 convention.
            let mut angle = uy.atan2(ux).to_degrees() as f32;
            while angle >= 0.0 {
                angle -= 90.0;
            }
            while angle < -90.0 {
                angle += 90.0;
            }
            best_rect = RotatedRect {
                center: Point2f {
                    x: cx as f32,
                    y: cy as f32,
                },
                size: (width as f32, height as f32),
                angle,
            };
        }
    }
    best_rect
}

/// Test whether a point lies inside, on, or outside a polygon contour.
///
/// When `measure_dist` is `false`, returns `+1.0` (inside), `−1.0` (outside),
/// or `0.0` (on edge).
///
/// When `measure_dist` is `true`, returns the signed minimum distance to the
/// nearest edge: positive inside, negative outside, zero on edge.
///
/// Uses the Jordan-curve (ray-casting) theorem for inside/outside classification.
pub fn point_polygon_test(contour: &[Point], pt: Point2f, measure_dist: bool) -> f64 {
    if contour.is_empty() {
        return 0.0;
    }
    let n = contour.len();
    let px = pt.x as f64;
    let py = pt.y as f64;

    // Minimum distance to any edge segment.
    let mut min_seg_dist = f64::MAX;
    for i in 0..n {
        let ax = contour[i].x as f64;
        let ay = contour[i].y as f64;
        let bx = contour[(i + 1) % n].x as f64;
        let by = contour[(i + 1) % n].y as f64;
        let d = point_segment_dist(px, py, ax, ay, bx, by);
        if d < min_seg_dist {
            min_seg_dist = d;
        }
    }

    // On-edge: within tolerance.
    if min_seg_dist < 1e-3 {
        return 0.0;
    }

    // Ray-casting: count rightward horizontal-ray crossings.
    let mut crossings = 0u32;
    for i in 0..n {
        let xi = contour[i].x as f64;
        let yi = contour[i].y as f64;
        let xj = contour[(i + 1) % n].x as f64;
        let yj = contour[(i + 1) % n].y as f64;

        let straddles = (yi <= py && py < yj) || (yj <= py && py < yi);
        if straddles {
            let x_cross = xi + (py - yi) * (xj - xi) / (yj - yi);
            if x_cross > px {
                crossings += 1;
            }
        }
    }

    let inside = crossings % 2 == 1;
    if !measure_dist {
        return if inside { 1.0 } else { -1.0 };
    }

    if inside {
        min_seg_dist
    } else {
        -min_seg_dist
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

type RawContour = Vec<(i32, i32)>;

fn trace_contours(binary: &[bool], w: usize, h: usize) -> Vec<RawContour> {
    let mut visited = vec![false; h * w];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if binary[idx] && !visited[idx] {
                let contour = trace_single_contour(binary, &mut visited, w, h, x, y);
                if !contour.is_empty() {
                    contours.push(contour);
                }
            }
        }
    }
    contours
}

fn trace_single_contour(
    binary: &[bool],
    visited: &mut Vec<bool>,
    w: usize,
    h: usize,
    start_x: usize,
    start_y: usize,
) -> RawContour {
    let mut contour = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((start_x as i32, start_y as i32));

    while let Some((x, y)) = queue.pop_front() {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            continue;
        }
        let idx = y as usize * w + x as usize;
        if visited[idx] || !binary[idx] {
            continue;
        }
        visited[idx] = true;

        // Include only boundary pixels (at least one background neighbour)
        let is_boundary = [(-1, 0i32), (1, 0), (0, -1), (0, 1)]
            .iter()
            .any(|&(dx, dy)| {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    true
                } else {
                    !binary[ny as usize * w + nx as usize]
                }
            });

        if is_boundary {
            contour.push((x, y));
        }

        for (dx, dy) in [(-1, 0i32), (1, 0), (0, -1), (0, 1)] {
            queue.push_back((x + dx, y + dy));
        }
    }
    contour
}

fn shoelace_area(pts: &[Point]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0i64;
    for i in 0..n {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % n];
        sum += p0.x as i64 * p1.y as i64 - p1.x as i64 * p0.y as i64;
    }
    sum as f64 / 2.0
}

fn douglas_peucker(pts: &[(i32, i32)], epsilon: f64) -> Vec<(i32, i32)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let (start, end) = (pts[0], pts[pts.len() - 1]);
    let mut max_dist = 0.0f64;
    let mut max_idx = 0;

    for i in 1..pts.len() - 1 {
        let d = point_line_dist(pts[i], start, end);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        let mut left = douglas_peucker(&pts[..=max_idx], epsilon);
        let right = douglas_peucker(&pts[max_idx..], epsilon);
        left.pop(); // remove duplicate at junction
        left.extend(right);
        left
    } else {
        vec![start, end]
    }
}

fn point_line_dist(p: (i32, i32), a: (i32, i32), b: (i32, i32)) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let num = ((by - ay) * px - (bx - ax) * py + bx * ay - by * ax).abs();
    let den = ((by - ay).powi(2) + (bx - ax).powi(2)).sqrt();
    if den < 1e-10 {
        0.0
    } else {
        num / den
    }
}

fn farthest_pair(pts: &[Point]) -> (Point, Point) {
    let mut max_dist = 0i64;
    let mut best = (pts[0], pts[pts.len() - 1]);
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = dist_sq(pts[i], pts[j]);
            if d > max_dist {
                max_dist = d;
                best = (pts[i], pts[j]);
            }
        }
    }
    best
}

fn dist_sq(a: Point, b: Point) -> i64 {
    let dx = (b.x - a.x) as i64;
    let dy = (b.y - a.y) as i64;
    dx * dx + dy * dy
}

fn cross_product(o: Point, a: Point, b: Point) -> i64 {
    (a.x - o.x) as i64 * (b.y - o.y) as i64 - (a.y - o.y) as i64 * (b.x - o.x) as i64
}

/// Return all-zero Moments for degenerate cases.
fn zero_moments() -> Moments {
    Moments {
        m00: 0.0,
        m10: 0.0,
        m01: 0.0,
        m20: 0.0,
        m11: 0.0,
        m02: 0.0,
        m30: 0.0,
        m21: 0.0,
        m12: 0.0,
        m03: 0.0,
        mu20: 0.0,
        mu11: 0.0,
        mu02: 0.0,
        mu30: 0.0,
        mu21: 0.0,
        mu12: 0.0,
        mu03: 0.0,
        nu20: 0.0,
        nu11: 0.0,
        nu02: 0.0,
        nu30: 0.0,
        nu21: 0.0,
        nu12: 0.0,
        nu03: 0.0,
    }
}

/// Perpendicular distance from point (px,py) to segment (ax,ay)→(bx,by).
fn point_segment_dist(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-20 {
        // Degenerate segment: distance to either endpoint.
        let ex = px - ax;
        let ey = py - ay;
        return (ex * ex + ey * ey).sqrt();
    }
    // Project onto segment, clamp to [0, 1].
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let qx = ax + t * dx;
    let qy = ay + t * dy;
    let ex = px - qx;
    let ey = py - qy;
    (ex * ex + ey * ey).sqrt()
}

/// L2 line fit via PCA (centred covariance matrix eigenvector).
fn fit_line_l2(points: &[Point]) -> [f32; 4] {
    let n = points.len() as f64;
    let xc = points.iter().map(|p| p.x as f64).sum::<f64>() / n;
    let yc = points.iter().map(|p| p.y as f64).sum::<f64>() / n;

    let mut s00 = 0.0_f64;
    let mut s01 = 0.0_f64;
    let mut s11 = 0.0_f64;
    for p in points {
        let dx = p.x as f64 - xc;
        let dy = p.y as f64 - yc;
        s00 += dx * dx;
        s01 += dx * dy;
        s11 += dy * dy;
    }

    // Eigenvector of 2×2 covariance for the larger eigenvalue.
    let (vx, vy) = pca2x2_direction(s00, s01, s11);
    [vx as f32, vy as f32, xc as f32, yc as f32]
}

/// L1 line fit via IRLS (iteratively reweighted least squares).
///
/// Bootstrap strategy: use the spatial median (component-wise median) and the
/// L2 direction of the low-weight centred set.  Multiple restarts from
/// different starting directions improve convergence in the presence of
/// large outliers.
fn fit_line_l1(points: &[Point]) -> [f32; 4] {
    // Collect sorted x and y values for median.
    let mut xs: Vec<f64> = points.iter().map(|p| p.x as f64).collect();
    let mut ys: Vec<f64> = points.iter().map(|p| p.y as f64).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    let med_x = if n % 2 == 0 {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    } else {
        xs[n / 2]
    };
    let med_y = if n % 2 == 0 {
        (ys[n / 2 - 1] + ys[n / 2]) / 2.0
    } else {
        ys[n / 2]
    };

    // Run IRLS from multiple initial directions and return the best result
    // (lowest weighted L1 objective).
    let starts: [(f64, f64, f64, f64); 3] = [
        // Horizontal direction from median centre.
        (1.0, 0.0, med_x, med_y),
        // Vertical direction from median centre.
        (0.0, 1.0, med_x, med_y),
        // 45-degree from median centre.
        (1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt(), med_x, med_y),
    ];

    let mut best_result = [0.0_f32; 4];
    let mut best_obj = f64::MAX;

    for &(init_vx, init_vy, init_x0, init_y0) in &starts {
        let result = irls_line(points, init_vx, init_vy, init_x0, init_y0, 60);
        // Compute L1 objective (sum of perpendicular distances).
        let obj: f64 = points
            .iter()
            .map(|p| {
                let r = (p.x as f64 - result[2] as f64) * result[1] as f64
                    - (p.y as f64 - result[3] as f64) * result[0] as f64;
                r.abs()
            })
            .sum();
        if obj < best_obj {
            best_obj = obj;
            best_result = result;
        }
    }
    best_result
}

/// Core IRLS iteration for L1 line fitting.
fn irls_line(
    points: &[Point],
    init_vx: f64,
    init_vy: f64,
    init_x0: f64,
    init_y0: f64,
    iters: usize,
) -> [f32; 4] {
    let mut vx = init_vx;
    let mut vy = init_vy;
    let mut x0 = init_x0;
    let mut y0 = init_y0;

    for _ in 0..iters {
        // Perpendicular residuals.
        let weights: Vec<f64> = points
            .iter()
            .map(|p| {
                let r = ((p.x as f64 - x0) * vy - (p.y as f64 - y0) * vx).abs();
                1.0 / r.max(1e-4)
            })
            .collect();

        let w_sum: f64 = weights.iter().sum();
        if w_sum < 1e-20 {
            break;
        }

        // Weighted centroid.
        let xc = points
            .iter()
            .zip(&weights)
            .map(|(p, &w)| p.x as f64 * w)
            .sum::<f64>()
            / w_sum;
        let yc = points
            .iter()
            .zip(&weights)
            .map(|(p, &w)| p.y as f64 * w)
            .sum::<f64>()
            / w_sum;

        // Weighted covariance.
        let mut s00 = 0.0_f64;
        let mut s01 = 0.0_f64;
        let mut s11 = 0.0_f64;
        for (p, &w) in points.iter().zip(&weights) {
            let dx = p.x as f64 - xc;
            let dy = p.y as f64 - yc;
            s00 += w * dx * dx;
            s01 += w * dx * dy;
            s11 += w * dy * dy;
        }

        let (nvx, nvy) = pca2x2_direction(s00, s01, s11);
        vx = nvx;
        vy = nvy;
        x0 = xc;
        y0 = yc;
    }

    [vx as f32, vy as f32, x0 as f32, y0 as f32]
}

/// Weight function choice for IRLS M-estimators.
#[derive(Clone, Copy)]
enum IrlsWeight {
    /// Huber: w = 1 if r <= c, else c / r.
    Huber { c: f64 },
    /// Fair: w = 1 / (1 + r / c).
    Fair { c: f64 },
}

/// Compute an IRLS weight for a given residual magnitude using the specified
/// weight function.
#[inline]
fn irls_weight_fn(r: f64, weight: IrlsWeight) -> f64 {
    match weight {
        IrlsWeight::Huber { c } => {
            if r <= c {
                1.0
            } else {
                c / r
            }
        }
        IrlsWeight::Fair { c } => 1.0 / (1.0 + r / c.max(1e-10)),
    }
}

/// Median of the values, sorting them in place.  Returns `0.0` for an empty
/// slice.
fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

/// Absolute perpendicular distance from `p` to the unit-direction line
/// `[vx, vy, x0, y0]`.
fn perp_residual(p: Point, line: &[f64; 4]) -> f64 {
    ((p.x as f64 - line[2]) * line[1] - (p.y as f64 - line[3]) * line[0]).abs()
}

/// Median absolute perpendicular residual of `points` about `line`.
fn median_abs_residual(points: &[Point], line: &[f64; 4]) -> f64 {
    let mut r: Vec<f64> = points.iter().map(|&p| perp_residual(p, line)).collect();
    median_in_place(&mut r)
}

/// Median squared perpendicular residual (least-median-of-squares objective).
fn median_squared_residual(points: &[Point], line: &[f64; 4]) -> f64 {
    let mut r: Vec<f64> = points
        .iter()
        .map(|&p| {
            let d = perp_residual(p, line);
            d * d
        })
        .collect();
    median_in_place(&mut r)
}

/// Upper bound on the number of points sampled for the Theil-Sen seed so the
/// O(n^2) pair enumeration stays bounded on large inputs.  The final polish
/// always runs on the full point set.
const THEIL_SEN_MAX_POINTS: usize = 200;

/// Theil-Sen robust line estimate in one orientation.
///
/// When `horizontal` is true the slope is `dy / dx` (line modelled as
/// `y = m x + b`); otherwise it is `dx / dy` (line modelled as `x = m y + b`),
/// which is needed to represent near-vertical lines.  Returns `None` when the
/// independent axis is degenerate (every sampled pair shares the same
/// coordinate).
fn theil_sen_line(points: &[Point], horizontal: bool) -> Option<[f64; 4]> {
    let stride = points.len().div_ceil(THEIL_SEN_MAX_POINTS).max(1);
    let sample: Vec<Point> = points.iter().copied().step_by(stride).collect();
    let m = sample.len();

    let mut slopes: Vec<f64> = Vec::new();
    for i in 0..m {
        let a = sample[i];
        for b in sample.iter().skip(i + 1) {
            let (num, den) = if horizontal {
                ((b.y - a.y) as f64, (b.x - a.x) as f64)
            } else {
                ((b.x - a.x) as f64, (b.y - a.y) as f64)
            };
            if den.abs() > 0.0 {
                slopes.push(num / den);
            }
        }
    }
    if slopes.is_empty() {
        return None;
    }
    let slope = median_in_place(&mut slopes);
    let norm = (1.0 + slope * slope).sqrt();

    if horizontal {
        let mut intercepts: Vec<f64> = sample
            .iter()
            .map(|p| p.y as f64 - slope * p.x as f64)
            .collect();
        let b = median_in_place(&mut intercepts);
        let mut xs: Vec<f64> = sample.iter().map(|p| p.x as f64).collect();
        let cx = median_in_place(&mut xs);
        Some([1.0 / norm, slope / norm, cx, slope * cx + b])
    } else {
        let mut intercepts: Vec<f64> = sample
            .iter()
            .map(|p| p.x as f64 - slope * p.y as f64)
            .collect();
        let b = median_in_place(&mut intercepts);
        let mut ys: Vec<f64> = sample.iter().map(|p| p.y as f64).collect();
        let cy = median_in_place(&mut ys);
        Some([slope / norm, 1.0 / norm, slope * cy + b, cy])
    }
}

/// High-breakdown robust seed line: the better-fitting of the two Theil-Sen
/// orientations, falling back to the L2 fit when both are degenerate.
fn robust_line_seed(points: &[Point]) -> [f64; 4] {
    let horizontal = theil_sen_line(points, true);
    let vertical = theil_sen_line(points, false);
    match (horizontal, vertical) {
        (Some(h), Some(v)) => {
            if median_abs_residual(points, &h) <= median_abs_residual(points, &v) {
                h
            } else {
                v
            }
        }
        (Some(h), None) => h,
        (None, Some(v)) => v,
        (None, None) => {
            let l2 = fit_line_l2(points);
            [l2[0] as f64, l2[1] as f64, l2[2] as f64, l2[3] as f64]
        }
    }
}

/// Robust M-estimator polish seeded from a high-breakdown estimate.
///
/// Standard Huber / Fair IRLS is not resistant to high-leverage outliers in
/// the total-least-squares (perpendicular) formulation: their non-redescending
/// `c / r` tail lets a distant point's leverage `w * r^2 = c * r` grow without
/// bound, so a naive iteration slowly rotates the fit towards the outliers even
/// from a perfect start.  Seeding from `robust_line_seed` places the iteration
/// in the correct basin, and returning the iterate that minimises the
/// least-median-of-squares objective guarantees the polish never degrades the
/// robust fit.  The per-iteration scale `c` is re-estimated from the median
/// absolute residual so the tuning constant is scale-invariant.
fn robust_polish(
    points: &[Point],
    seed: [f64; 4],
    iters: usize,
    base_weight: IrlsWeight,
    tuning: f64,
) -> [f64; 4] {
    let [mut vx, mut vy, mut x0, mut y0] = seed;
    let mut best = seed;
    let mut best_obj = median_squared_residual(points, &seed);

    for _ in 0..iters {
        let raw: Vec<f64> = points
            .iter()
            .map(|p| ((p.x as f64 - x0) * vy - (p.y as f64 - y0) * vx).abs())
            .collect();

        let mut scale_sample = raw.clone();
        let sigma = (median_in_place(&mut scale_sample) / 0.6745).max(1e-6);
        let c = (tuning * sigma).max(1e-6);

        let effective_weight = match base_weight {
            IrlsWeight::Huber { .. } => IrlsWeight::Huber { c },
            IrlsWeight::Fair { .. } => IrlsWeight::Fair { c },
        };

        let weights: Vec<f64> = raw
            .iter()
            .map(|&r| irls_weight_fn(r, effective_weight))
            .collect();

        let w_sum: f64 = weights.iter().sum();
        if w_sum < 1e-20 {
            break;
        }

        let xc = points
            .iter()
            .zip(&weights)
            .map(|(p, &w)| p.x as f64 * w)
            .sum::<f64>()
            / w_sum;
        let yc = points
            .iter()
            .zip(&weights)
            .map(|(p, &w)| p.y as f64 * w)
            .sum::<f64>()
            / w_sum;

        let mut s00 = 0.0_f64;
        let mut s01 = 0.0_f64;
        let mut s11 = 0.0_f64;
        for (p, &w) in points.iter().zip(&weights) {
            let dx = p.x as f64 - xc;
            let dy = p.y as f64 - yc;
            s00 += w * dx * dx;
            s01 += w * dx * dy;
            s11 += w * dy * dy;
        }

        let (nvx, nvy) = pca2x2_direction(s00, s01, s11);
        // |sin| of the change in unit direction between iterations.
        let direction_change = (vx * nvy - vy * nvx).abs();
        vx = nvx;
        vy = nvy;
        x0 = xc;
        y0 = yc;

        let candidate = [vx, vy, x0, y0];
        let obj = median_squared_residual(points, &candidate);
        if obj < best_obj {
            best_obj = obj;
            best = candidate;
        }
        if direction_change < 1e-4 {
            break;
        }
    }

    best
}

/// Huber M-estimator line fit: high-breakdown Theil-Sen seed refined by a
/// robust IRLS polish.
///
/// Uses the standard Huber tuning constant k = 1.345 (95% efficiency at the
/// Gaussian distribution).
fn fit_line_huber(points: &[Point]) -> [f32; 4] {
    const HUBER_TUNING: f64 = 1.345;
    let seed = robust_line_seed(points);
    let line = robust_polish(
        points,
        seed,
        40,
        IrlsWeight::Huber { c: HUBER_TUNING },
        HUBER_TUNING,
    );
    [
        line[0] as f32,
        line[1] as f32,
        line[2] as f32,
        line[3] as f32,
    ]
}

/// Fair M-estimator line fit: high-breakdown Theil-Sen seed refined by a
/// robust IRLS polish.
///
/// Uses the standard Fair tuning constant c = 1.3998 (95% efficiency at the
/// Gaussian distribution).
fn fit_line_fair(points: &[Point]) -> [f32; 4] {
    const FAIR_TUNING: f64 = 1.3998;
    let seed = robust_line_seed(points);
    let line = robust_polish(
        points,
        seed,
        40,
        IrlsWeight::Fair { c: FAIR_TUNING },
        FAIR_TUNING,
    );
    [
        line[0] as f32,
        line[1] as f32,
        line[2] as f32,
        line[3] as f32,
    ]
}

/// Direction eigenvector of 2×2 symmetric covariance [[s00, s01],[s01, s11]]
/// corresponding to the larger eigenvalue.
fn pca2x2_direction(s00: f64, s01: f64, s11: f64) -> (f64, f64) {
    let tr = s00 + s11;
    let diff = ((s00 - s11) * (s00 - s11) + 4.0 * s01 * s01).sqrt();
    let lambda1 = (tr + diff) / 2.0;

    let (vx, vy) = if s01.abs() > 1e-10 {
        (lambda1 - s11, s01)
    } else if s00 >= s11 {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let norm = (vx * vx + vy * vy).sqrt();
    if norm < 1e-20 {
        (1.0, 0.0)
    } else {
        (vx / norm, vy / norm)
    }
}

// ── 6×6 matrix helpers for fit_ellipse ───────────────────────────────────────

/// Multiply two 6×6 matrices.
fn mat6_mul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Invert a 6×6 matrix via Gauss-Jordan elimination.
/// Returns `None` if the matrix is singular (pivot < 1e-14).
fn mat6_inverse(mat: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut a = *mat;
    let mut inv = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        inv[i][i] = 1.0;
    }

    for col in 0..6 {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..6 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);
        inv.swap(col, max_row);

        // Scale pivot row.
        let pivot = a[col][col];
        for j in 0..6 {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }

        // Eliminate column.
        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in 0..6 {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

/// Solve the Fitzgibbon generalized eigenvalue problem S·a = λ·C·a via the
/// Cholesky reduction to a standard symmetric EVP.
///
/// Algorithm (Fitzgibbon et al. 1996):
/// 1. Cholesky S = LLᵀ  (L lower-triangular, 6×6).
/// 2. Form T = L⁻¹ · C · L⁻ᵀ  (symmetric, 6×6).
/// 3. Jacobi eigendecomposition: T·b = λ·b.
/// 4. Recover solution vector a = L⁻ᵀ · b.
///
/// Returns `(eigenvectors[6][6], eigenvalues[6])` where `eigenvectors[i]` is
/// the eigenvector for `eigenvalues[i]`.
///
/// The `m` parameter (= S⁻¹·C) is accepted for API compatibility but this
/// function rebuilds the computation from the original scatter matrix stored
/// inside `m` — actually we take a different approach: since the caller already
/// has S_inv and C, we accept M directly but work around non-symmetry via the
/// Cholesky path.  In practice we receive M and must factor it properly.
///
/// Actual implementation: the standard trick for the Fitzgibbon constraint is
/// to work with the **reduced** 3×3 system.  The 6×6 C matrix acts only on the
/// first 3 components of a (the [A,B,C] conic coefficients).  Partition:
/// a = [a1; a2] where a1 = [A,B,C] (3-vector) and a2 = [D,E,F] (3-vector).
/// Partition S accordingly:  S = [[S11, S12]; [S21, S22]].
/// The bottom block gives a2 = -S22⁻¹·S21·a1 (from S·a = 0 projected).
/// Then reduce to a 3×3 symmetric eigenproblem in a1 only.
///
/// For simplicity and correctness we use: since M = S⁻¹C is already computed,
/// find its eigenvalues numerically via the **characteristic polynomial** of the
/// reduced 3×3 sub-problem.  For the constraint C₃ = [[0,0,2],[0,-1,0],[2,0,0]]
/// acting on a1, the problem is well-conditioned.
///
/// However, the cleanest pure-Rust approach that avoids non-symmetric EVP issues
/// is to reconstruct S from the design matrix, do Cholesky, and use Jacobi on
/// the reduced symmetric T₃.  Since we don't have D available here, we instead
/// use a stable iterative approach on M directly:  **inverse iteration** for
/// the positive eigenvalue of S·a = λ·C·a with shift σ > 0.
fn eigen6_nonsym(m: &[[f64; 6]; 6]) -> ([[f64; 6]; 6], [f64; 6]) {
    // We need eigenvalues/eigenvectors of M = S⁻¹·C (passed as `m`).
    // Since C has rank 3 and M = S⁻¹C, there are 3 non-zero eigenvalues.
    // The valid ellipse corresponds to the UNIQUE positive eigenvalue.
    //
    // Strategy: Use the fact that M·(M·v) = λ²·v, so M² is the "power" matrix.
    // Apply simultaneous orthogonal iteration (Golub-Van Loan §8.2) to find
    // the dominant invariant subspace, then extract eigenvalues.
    //
    // For our 6×6 case with known structure we do 6 independent inverse
    // power iterations with orthogonalisation (modified Gram-Schmidt).

    let mut evecs = [[0.0_f64; 6]; 6];
    let mut evals = [0.0_f64; 6];

    // Initialize Q as identity (orthonormal starting basis).
    let mut q = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        q[i][i] = 1.0;
    }

    // Simultaneous power iteration: Q ← M·Q, then QR-factor.
    for _iter in 0..400 {
        // Z = M · Q  (each column of Q is multiplied by M)
        let mut z = [[0.0_f64; 6]; 6];
        for col in 0..6 {
            for i in 0..6 {
                for j in 0..6 {
                    z[i][col] += m[i][j] * q[j][col];
                }
            }
        }
        // QR factorisation of Z (modified Gram-Schmidt), Q ← new orthonormal basis.
        for col in 0..6 {
            // Orthogonalise z[:,col] against previous columns.
            for prev in 0..col {
                let dot: f64 = (0..6).map(|r| z[r][col] * q[r][prev]).sum();
                for r in 0..6 {
                    z[r][col] -= dot * q[r][prev];
                }
            }
            let norm: f64 = (0..6).map(|r| z[r][col] * z[r][col]).sum::<f64>().sqrt();
            if norm < 1e-30 {
                // Zero column: keep previous q[:,col] (it's in the null space).
                continue;
            }
            for r in 0..6 {
                q[r][col] = z[r][col] / norm;
            }
        }
    }

    // Extract eigenvalues as Rayleigh quotients: λ_i = q_i^T · M · q_i.
    for col in 0..6 {
        let mut mqi = [0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                mqi[i] += m[i][j] * q[j][col];
            }
        }
        let lam: f64 = (0..6).map(|r| q[r][col] * mqi[r]).sum();
        evals[col] = lam;
        for r in 0..6 {
            evecs[col][r] = q[r][col];
        }
    }

    (evecs, evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_rect_empty() {
        let r = bounding_rect(&[]);
        assert_eq!(r.width, 0);
        assert_eq!(r.height, 0);
    }

    #[test]
    fn test_contour_area_triangle() {
        let pts = [
            Point { x: 0, y: 0 },
            Point { x: 4, y: 0 },
            Point { x: 0, y: 3 },
        ];
        let area = contour_area(&pts);
        assert!((area - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_arc_length_square() {
        let pts = [
            Point { x: 0, y: 0 },
            Point { x: 10, y: 0 },
            Point { x: 10, y: 10 },
            Point { x: 0, y: 10 },
        ];
        let len = arc_length(&pts, true);
        assert!((len - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_contours_dtype_error() {
        use crate::mat::MatType;
        let mat = Mat::new(5, 5, MatType::CV_8UC3);
        let result = find_contours(&mat, 0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_contours_blank() {
        let mat = Mat::new_8uc1(10, 10);
        let (contours, hierarchy) = find_contours(&mat, 0, 2).unwrap();
        assert!(contours.is_empty());
        assert!(hierarchy.is_empty());
    }

    #[test]
    fn test_moments_square_numerical() {
        // 100×100 square: vertices (0,0),(100,0),(100,100),(0,100)
        let pts = vec![
            Point { x: 0, y: 0 },
            Point { x: 100, y: 0 },
            Point { x: 100, y: 100 },
            Point { x: 0, y: 100 },
        ];
        let m = moments(&pts);
        assert!((m.m00 - 10000.0).abs() < 1.0, "m00={}", m.m00);
        // m20 = m02 = ∫₀¹⁰⁰ ∫₀¹⁰⁰ x² dx dy = (100³/3)·100 = 33_333_333.33
        let expected_m20 = 100.0_f64.powi(3) / 3.0 * 100.0;
        assert!(
            (m.m20 - expected_m20).abs() < 1000.0,
            "m20={} expected≈{}",
            m.m20,
            expected_m20
        );
        // m21 = ∫ x²y dA = (100³/3)(100²/2) = 1_666_666_666.67
        let expected_m21 = 100.0_f64.powi(3) / 3.0 * 100.0_f64.powi(2) / 2.0;
        assert!(
            (m.m21 - expected_m21).abs() / expected_m21 < 1e-4,
            "m21={} expected≈{}",
            m.m21,
            expected_m21
        );
        // m30 = ∫ x³ dA = (100⁴/4)·100 = 2_500_000_000
        let expected_m30 = 100.0_f64.powi(4) / 4.0 * 100.0;
        assert!(
            (m.m30 - expected_m30).abs() / expected_m30 < 1e-4,
            "m30={} expected≈{}",
            m.m30,
            expected_m30
        );
    }
}
