//! Regression tests for the marching-squares contour extraction of
//! [`extract_contour`].

use super::*;
use std::collections::BTreeMap;

/// `true` when the two points coincide to within rounding.
fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    dist3(a, b) < 1e-12
}

/// Every consecutive point pair of every contour line.
fn segments_of(lines: &[ContourLine]) -> Vec<([f64; 3], [f64; 3])> {
    lines
        .iter()
        .flat_map(|line| line.points.windows(2).map(|w| (w[0], w[1])))
        .collect()
}

/// `true` when `segs` holds a segment joining `p` and `q` (in either order).
fn has_segment(segs: &[([f64; 3], [f64; 3])], p: [f64; 3], q: [f64; 3]) -> bool {
    segs.iter()
        .any(|&(a, b)| (close(a, p) && close(b, q)) || (close(a, q) && close(b, p)))
}

/// Grid with a single cell spanning the local square `[-0.5, 0.5]²` of the XY
/// plane; the node values are `(v00, v10, v01, v11)` with the first index
/// running along local `u` and the second along local `v`.
fn single_cell(v00: f64, v10: f64, v01: f64, v11: f64) -> ScalarGrid2D {
    let mut grid = ScalarGrid2D::new(&CuttingPlane::xy(0.0), 0.5, 1);
    grid.set(0, 0, v00);
    grid.set(1, 0, v10);
    grid.set(0, 1, v01);
    grid.set(1, 1, v11);
    grid
}

/// World position of the local point `(u, v)` of the grid's plane.
fn world(grid: &ScalarGrid2D, u: f64, v: f64) -> [f64; 3] {
    grid.plane.from_local_2d([u, v])
}

#[test]
fn left_column_high_crosses_from_the_bottom_to_the_top_edge() {
    let grid = single_cell(1.0, 0.0, 1.0, 0.0);
    let segs = segments_of(&extract_contour(&grid, 0.5));
    assert_eq!(segs.len(), 1, "got {segs:?}");
    assert!(
        has_segment(&segs, world(&grid, 0.0, -0.5), world(&grid, 0.0, 0.5)),
        "got {segs:?}"
    );
}

#[test]
fn right_column_high_crosses_from_the_bottom_to_the_top_edge() {
    let grid = single_cell(0.0, 1.0, 0.0, 1.0);
    let segs = segments_of(&extract_contour(&grid, 0.5));
    assert_eq!(segs.len(), 1, "got {segs:?}");
    assert!(
        has_segment(&segs, world(&grid, 0.0, -0.5), world(&grid, 0.0, 0.5)),
        "got {segs:?}"
    );
}

#[test]
fn saddle_with_high_main_diagonal_follows_the_cell_centre_average() {
    // v00 = v11 = 1 and v10 = v01 = 0, so the centre average is 0.5.
    let grid = single_cell(1.0, 0.0, 0.0, 1.0);

    // Centre above the iso-value: the high nodes are joined, and the contour
    // cuts off the low corners (0.5, -0.5) and (-0.5, 0.5) one by one.
    let joined = segments_of(&extract_contour(&grid, 0.4));
    assert_eq!(joined.len(), 2, "got {joined:?}");
    assert!(has_segment(
        &joined,
        world(&grid, -0.5, 0.1),
        world(&grid, -0.1, 0.5)
    ));
    assert!(has_segment(
        &joined,
        world(&grid, 0.1, -0.5),
        world(&grid, 0.5, -0.1)
    ));

    // Centre below the iso-value: the low nodes are joined, and the contour
    // cuts off the high corners (-0.5, -0.5) and (0.5, 0.5) one by one.
    let split = segments_of(&extract_contour(&grid, 0.6));
    assert_eq!(split.len(), 2, "got {split:?}");
    assert!(has_segment(
        &split,
        world(&grid, -0.5, -0.1),
        world(&grid, -0.1, -0.5)
    ));
    assert!(has_segment(
        &split,
        world(&grid, 0.1, 0.5),
        world(&grid, 0.5, 0.1)
    ));
}

#[test]
fn saddle_with_high_anti_diagonal_follows_the_cell_centre_average() {
    // v10 = v01 = 1 and v00 = v11 = 0, so the centre average is 0.5.
    let grid = single_cell(0.0, 1.0, 1.0, 0.0);

    // Centre above the iso-value: cut off the low corners (-0.5, -0.5) and
    // (0.5, 0.5).
    let joined = segments_of(&extract_contour(&grid, 0.4));
    assert_eq!(joined.len(), 2, "got {joined:?}");
    assert!(has_segment(
        &joined,
        world(&grid, -0.5, -0.1),
        world(&grid, -0.1, -0.5)
    ));
    assert!(has_segment(
        &joined,
        world(&grid, 0.1, 0.5),
        world(&grid, 0.5, 0.1)
    ));

    // Centre below the iso-value: cut off the high corners (0.5, -0.5) and
    // (-0.5, 0.5).
    let split = segments_of(&extract_contour(&grid, 0.6));
    assert_eq!(split.len(), 2, "got {split:?}");
    assert!(has_segment(
        &split,
        world(&grid, 0.1, -0.5),
        world(&grid, 0.5, -0.1)
    ));
    assert!(has_segment(
        &split,
        world(&grid, -0.5, 0.1),
        world(&grid, -0.1, 0.5)
    ));
}

#[test]
fn linear_field_contour_lies_on_the_iso_line_and_spans_the_grid() {
    let mut grid = ScalarGrid2D::new(&CuttingPlane::xy(0.0), 1.0, 10);
    grid.sample_field(|p| p[0] - 0.3);
    let lines = extract_contour(&grid, 0.0);
    assert!(!lines.is_empty(), "the iso-line x = 0.3 crosses the grid");

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for line in &lines {
        assert_eq!(line.value, 0.0);
        for pt in &line.points {
            assert!(
                (pt[0] - 0.3).abs() < 1e-9,
                "contour point {pt:?} is off the line x = 0.3"
            );
            assert!(pt[2].abs() < 1e-12, "contour point {pt:?} left the plane");
            y_min = y_min.min(pt[1]);
            y_max = y_max.max(pt[1]);
        }
    }
    assert!((y_min + 1.0).abs() < 1e-9, "contour starts at y = {y_min}");
    assert!((y_max - 1.0).abs() < 1e-9, "contour ends at y = {y_max}");
}

#[test]
fn circle_contour_hugs_the_circle_and_every_vertex_joins_two_segments() {
    // The nodes are 0.2 apart, so a radius of 0.95 passes through no node
    // (a contour through a node meets four cells and has degree four).
    let radius = 0.95;
    let mut grid = ScalarGrid2D::new(&CuttingPlane::xy(0.0), 2.0, 20);
    grid.sample_field(|p| (p[0] * p[0] + p[1] * p[1]).sqrt());
    let lines = extract_contour(&grid, radius);
    assert!(!lines.is_empty());

    let mut degree: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for (a, b) in segments_of(&lines) {
        for pt in [a, b] {
            let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
            assert!((r - radius).abs() < 0.05, "vertex {pt:?} at radius {r}");
            let key = ((pt[0] * 1e9).round() as i64, (pt[1] * 1e9).round() as i64);
            *degree.entry(key).or_insert(0) += 1;
        }
    }
    assert!(
        degree.values().all(|&d| d == 2),
        "a closed contour joins exactly two segments at every vertex: {degree:?}"
    );
}
