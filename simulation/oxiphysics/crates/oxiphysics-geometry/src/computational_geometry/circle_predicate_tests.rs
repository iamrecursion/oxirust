//! Regression tests for [`circumcircle_2d`] and [`in_circumcircle_2d`]: both
//! used an absolute tolerance (`1e-12`), which silently broke small-scale
//! input; they now rely on the exact `orient2d` / `incircle` predicates.

use super::{Point2, circumcircle_2d, delaunay_2d, in_circumcircle_2d, voronoi_from_delaunay};

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y)
}

fn scaled(p: Point2, s: f64) -> Point2 {
    pt(p.x * s, p.y * s)
}

#[test]
fn circumcircle_of_a_tiny_triangle_is_found() {
    // The unit right triangle has circumcentre (0.5, 0.5) and r² = 0.5.  At the
    // scale 1e-7 the old `|d| < 1e-12` test (d = 2 * area * 2 = 2e-14) gave None.
    for s in [1.0, 1e-3, 1e-7, 1e-9, 1e6] {
        let (a, b, c) = (pt(0.0, 0.0), pt(s, 0.0), pt(0.0, s));
        let (center, r2) = circumcircle_2d(a, b, c).expect("a proper triangle has a circle");
        assert!((center.x - 0.5 * s).abs() <= 1e-12 * s, "scale {s}");
        assert!((center.y - 0.5 * s).abs() <= 1e-12 * s, "scale {s}");
        assert!((r2 - 0.5 * s * s).abs() <= 1e-12 * s * s, "scale {s}");
    }
}

#[test]
fn collinear_points_have_no_circumcircle_at_any_scale() {
    for s in [1.0, 1e-9, 1e9] {
        assert!(circumcircle_2d(pt(0.0, 0.0), pt(s, s), pt(2.0 * s, 2.0 * s)).is_none());
        assert!(circumcircle_2d(pt(0.0, 0.0), pt(s, 0.0), pt(s, 0.0)).is_none());
    }
    // Coincident points are collinear too.
    assert!(circumcircle_2d(pt(1.0, 1.0), pt(1.0, 1.0), pt(1.0, 1.0)).is_none());
}

#[test]
fn in_circumcircle_is_exact_orientation_free_and_scale_free() {
    let (a, b, c) = (pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0));
    let inside = pt(0.5, 0.5);
    let on_circle = pt(1.0, 1.0);
    let outside = pt(1.1, 1.1);
    for s in [1.0, 1e-6, 1e-9, 1e8] {
        let (a, b, c) = (scaled(a, s), scaled(b, s), scaled(c, s));
        for (name, p, want) in [
            ("inside", scaled(inside, s), true),
            ("on circle", scaled(on_circle, s), false),
            ("outside", scaled(outside, s), false),
        ] {
            assert_eq!(in_circumcircle_2d(a, b, c, p), want, "{name} at scale {s}");
            // The same triangle listed clockwise gives the same answer.
            assert_eq!(
                in_circumcircle_2d(a, c, b, p),
                want,
                "{name} cw at scale {s}"
            );
        }
    }
    // A degenerate triangle has no circle, so nothing is inside it.
    assert!(!in_circumcircle_2d(
        pt(0.0, 0.0),
        pt(1.0, 1.0),
        pt(2.0, 2.0),
        pt(1.0, 1.0)
    ));
}

#[test]
fn voronoi_cells_of_a_tiny_point_set_keep_their_vertices() {
    // The old absolute tolerance dropped every circumcentre of a 1e-7 sized
    // triangulation, leaving the Voronoi cells empty.
    let s = 1e-7;
    let points: Vec<Point2> = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        .iter()
        .map(|&(x, y)| pt(x * s, y * s))
        .collect();
    let tris = delaunay_2d(&points);
    assert_eq!(tris.len(), 2);
    let cells = voronoi_from_delaunay(&points, &tris);
    for cell in &cells {
        assert!(!cell.circumcenters.is_empty(), "site {}", cell.site);
        for center in &cell.circumcenters {
            // Both triangles of the square share its circle, centred at (s, s).
            assert!(center.x.is_finite() && center.y.is_finite());
            assert!((center.x - s).abs() < 1e-3 * s && (center.y - s).abs() < 1e-3 * s);
        }
    }
}
