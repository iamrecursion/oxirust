//! Regression tests for the obstacle visibility test behind
//! [`VisibilityGraph`] and the art-gallery coverage helpers: segments may graze
//! an obstacle, but must not pass through its interior, and gallery chords must
//! stay inside the gallery.

use super::functions::{
    segment_extent, segment_inside_polygon, segment_visible, segments_properly_intersect,
};
use super::{
    Obstacle2D, Point2, VisibilityGraph, art_gallery_greedy, check_full_coverage, polygon_area,
};

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y)
}

fn square(x0: f64, y0: f64, side: f64) -> Obstacle2D {
    Obstacle2D::new(vec![
        pt(x0, y0),
        pt(x0 + side, y0),
        pt(x0 + side, y0 + side),
        pt(x0, y0 + side),
    ])
}

fn path_length(graph: &VisibilityGraph, path: &[usize]) -> f64 {
    path.windows(2)
        .map(|w| graph.nodes[w[0]].dist(graph.nodes[w[1]]))
        .sum()
}

/// L-shaped gallery with the reflex vertex `(1, 1)` (index 3).
fn l_shape() -> Vec<Point2> {
    vec![
        pt(0.0, 0.0),
        pt(2.0, 0.0),
        pt(2.0, 1.0),
        pt(1.0, 1.0),
        pt(1.0, 2.0),
        pt(0.0, 2.0),
    ]
}

/// U-shaped gallery: two towers joined by a base, notch between `x = 1` and `x = 2`.
fn u_shape() -> Vec<Point2> {
    vec![
        pt(0.0, 0.0),
        pt(3.0, 0.0),
        pt(3.0, 3.0),
        pt(2.0, 3.0),
        pt(2.0, 1.0),
        pt(1.0, 1.0),
        pt(1.0, 3.0),
        pt(0.0, 3.0),
    ]
}

#[test]
fn square_diagonals_are_blocked_but_sides_are_visible() {
    let graph = VisibilityGraph::build(&[square(0.0, 0.0, 1.0)]);
    assert_eq!(graph.num_nodes(), 4);
    // Every vertex sees its two neighbours along the sides, not the opposite
    // corner across the interior (all four sides share an endpoint with it).
    assert_eq!(graph.edges[0], vec![1, 3]);
    assert_eq!(graph.edges[1], vec![0, 2]);
    assert_eq!(graph.edges[2], vec![1, 3]);
    assert_eq!(graph.edges[3], vec![0, 2]);
}

#[test]
fn shortest_path_goes_around_the_obstacle() {
    let obstacles = [square(0.0, 0.0, 1.0)];
    let mut graph = VisibilityGraph::build(&obstacles);
    graph.add_query_point(pt(-1.0, 0.5), &obstacles);
    graph.add_query_point(pt(2.0, 0.5), &obstacles);
    let (start, goal) = (4, 5);
    // The straight line crosses the square.
    assert!(!graph.edges[start].contains(&goal));
    let path = graph
        .shortest_path(start, goal)
        .expect("the goal is reachable around the square");
    assert_eq!(path.len(), 4, "start, two corners, goal: {path:?}");
    let detour = 2.0 * (1.0_f64 + 0.25).sqrt() + 1.0;
    let length = path_length(&graph, &path);
    assert!(length > 3.0 + 0.2, "detour must exceed the blocked 3.0");
    assert!(
        (length - detour).abs() < 1e-6,
        "length {length} vs {detour}"
    );
}

#[test]
fn clear_line_of_sight_is_a_direct_edge() {
    let obstacles = [square(0.0, 0.0, 1.0)];
    let mut graph = VisibilityGraph::build(&obstacles);
    graph.add_query_point(pt(-1.0, 2.0), &obstacles);
    graph.add_query_point(pt(2.0, 2.0), &obstacles);
    let path = graph.shortest_path(4, 5).expect("direct path exists");
    assert_eq!(path, vec![4, 5]);
    assert!((path_length(&graph, &path) - 3.0).abs() < 1e-12);
}

#[test]
fn segment_touching_two_obstacles_only_at_a_shared_corner_is_visible() {
    // Two squares meet at (1, 1); the anti-diagonal from (0, 2) to (2, 0)
    // passes between them and touches both only at that corner.
    let obstacles = [square(0.0, 0.0, 1.0), square(1.0, 1.0, 1.0)];
    assert!(segment_visible(pt(0.0, 2.0), pt(2.0, 0.0), &obstacles));
    // The other diagonal crosses both squares.
    assert!(!segment_visible(pt(0.0, 0.0), pt(2.0, 2.0), &obstacles));
    assert!(!segment_visible(pt(0.5, 0.5), pt(1.5, 1.5), &obstacles));
}

#[test]
fn segment_through_vertices_of_a_diamond_is_blocked() {
    // The horizontal through the left and right corners of a diamond runs
    // through its interior without crossing any edge in a proper point.
    let diamond = Obstacle2D::new(vec![pt(0.0, 1.0), pt(1.0, 0.0), pt(2.0, 1.0), pt(1.0, 2.0)]);
    let obstacles = [diamond];
    assert!(!segment_visible(pt(-1.0, 1.0), pt(3.0, 1.0), &obstacles));
    assert!(!segment_visible(pt(0.0, 1.0), pt(2.0, 1.0), &obstacles));
    // Skimming the bottom corner from outside is fine.
    assert!(segment_visible(pt(-1.0, 0.0), pt(3.0, 0.0), &obstacles));
    let touch = segment_extent(pt(0.0, 0.0), pt(2.0, 0.0), &obstacles[0].vertices);
    assert!(!touch.inside && touch.outside);
}

#[test]
fn segment_running_along_an_edge_is_visible_even_when_longer_than_the_edge() {
    let obstacles = [square(0.0, 0.0, 1.0)];
    assert!(segment_visible(pt(-1.0, 0.0), pt(2.0, 0.0), &obstacles));
    assert!(segment_visible(pt(0.25, 0.0), pt(0.75, 0.0), &obstacles));
    assert!(segment_visible(pt(1.0, -1.0), pt(1.0, 2.0), &obstacles));
    // Starting on the boundary and heading inside is blocked.
    assert!(!segment_visible(pt(0.0, 0.0), pt(0.5, 0.5), &obstacles));
    // Starting inside is blocked as well.
    assert!(!segment_visible(pt(0.5, 0.5), pt(3.0, 0.5), &obstacles));
}

#[test]
fn degenerate_inputs_do_not_block() {
    assert!(segment_visible(pt(0.0, 0.0), pt(1.0, 1.0), &[]));
    let two_vertices = Obstacle2D::new(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
    assert!(segment_visible(pt(0.0, 1.0), pt(1.0, 0.0), &[two_vertices]));
    // A zero-length segment has no interior to cross.
    let obstacles = [square(0.0, 0.0, 1.0)];
    assert!(segment_visible(pt(0.5, 0.5), pt(0.5, 0.5), &obstacles));
}

#[test]
fn proper_intersection_is_exact_and_scale_free() {
    // Plain crossing.
    assert!(segments_properly_intersect(
        pt(0.0, 0.0),
        pt(2.0, 2.0),
        pt(0.0, 2.0),
        pt(2.0, 0.0)
    ));
    // Shared endpoint, T-junction and collinear overlap are not proper.
    assert!(!segments_properly_intersect(
        pt(0.0, 0.0),
        pt(1.0, 1.0),
        pt(1.0, 1.0),
        pt(2.0, 0.0)
    ));
    assert!(!segments_properly_intersect(
        pt(0.0, 0.0),
        pt(2.0, 0.0),
        pt(1.0, 0.0),
        pt(1.0, 1.0)
    ));
    assert!(!segments_properly_intersect(
        pt(0.0, 0.0),
        pt(2.0, 0.0),
        pt(1.0, 0.0),
        pt(3.0, 0.0)
    ));
    // The same crossing at a tiny scale (all endpoint distances far below the
    // old absolute 1e-6 tolerance) is still detected.
    let s = 1e-7;
    assert!(segments_properly_intersect(
        pt(0.0, 0.0),
        pt(2.0 * s, 2.0 * s),
        pt(0.0, 2.0 * s),
        pt(2.0 * s, 0.0)
    ));
}

#[test]
fn gallery_chords_may_not_cut_across_a_notch() {
    let l = l_shape();
    // Through the inside of the L.
    assert!(segment_inside_polygon(l[0], l[2], &l));
    assert!(segment_inside_polygon(l[0], l[4], &l));
    // Along the boundary, including through the reflex vertex.
    assert!(segment_inside_polygon(l[2], l[3], &l));
    assert!(segment_inside_polygon(l[5], l[1], &l));
    // Across the empty corner: no edge is crossed properly, yet the chord is outside.
    assert!(!segment_inside_polygon(l[2], l[4], &l));
    assert!(!segment_inside_polygon(l[1], l[4], &l));
}

#[test]
fn check_full_coverage_rejects_guards_that_only_see_through_the_notch() {
    let l = l_shape();
    // From (0, 0) every vertex of the L is visible.
    assert!(check_full_coverage(&l, &[0]));
    // From (2, 1) the vertex (1, 2) lies across the notch.
    assert!(!check_full_coverage(&l, &[2]));
    // Two guards at the ends of the notch see everything between them.
    assert!(check_full_coverage(&l, &[2, 4]));
    assert!(!check_full_coverage(&l, &[]));
    // Guard indices that are not vertices cover nothing.
    assert!(!check_full_coverage(&l, &[17]));
}

#[test]
fn greedy_art_gallery_covers_the_l_shape_with_a_vertex_that_really_sees_everything() {
    let l = l_shape();
    let result = art_gallery_greedy(&l, 6);
    assert!(result.covered.iter().all(|&c| c));
    assert_eq!(result.guards.len(), 1, "guards {:?}", result.guards);
    assert!(
        [0, 3].contains(&result.guards[0]),
        "only the vertices (0,0) and (1,1) see the whole L, got {:?}",
        result.guards
    );
    assert!(check_full_coverage(&l, &result.guards));
}

#[test]
fn greedy_art_gallery_needs_several_guards_for_a_u_shape() {
    let u = u_shape();
    assert!(polygon_area(&u) > 0.0);
    // No vertex of the U sees all of it (the notch hides the far tower tips).
    for guard in 0..u.len() {
        assert!(!check_full_coverage(&u, &[guard]), "guard {guard}");
    }
    let one = art_gallery_greedy(&u, 1);
    assert_eq!(one.guards.len(), 1);
    assert!(
        one.covered.iter().any(|&c| !c),
        "one guard cannot cover a U"
    );
    let all = art_gallery_greedy(&u, u.len());
    assert!(all.guards.len() >= 2);
    assert!(all.covered.iter().all(|&c| c));
    assert!(check_full_coverage(&u, &all.guards));
}

#[test]
fn convex_gallery_and_empty_polygon_edge_cases() {
    let sq = vec![pt(0.0, 0.0), pt(4.0, 0.0), pt(4.0, 4.0), pt(0.0, 4.0)];
    let result = art_gallery_greedy(&sq, 4);
    assert_eq!(result.guards.len(), 1);
    assert!(result.covered.iter().all(|&c| c));
    let empty = art_gallery_greedy(&[], 3);
    assert!(empty.guards.is_empty() && empty.covered.is_empty());
    assert!(check_full_coverage(&[], &[]));
    // max_guards = 0 places no guard and covers nothing.
    let none = art_gallery_greedy(&sq, 0);
    assert!(none.guards.is_empty());
    assert!(none.covered.iter().all(|&c| !c));
}
