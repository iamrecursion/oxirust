//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::computational_geometry::HalfEdgeMesh;
    use crate::computational_geometry::Line2D;
    use crate::computational_geometry::LineArrangement;
    use crate::computational_geometry::Obstacle2D;
    use crate::computational_geometry::Point2;
    use crate::computational_geometry::Segment2D;
    use crate::computational_geometry::SlabPointLocator;
    use crate::computational_geometry::VisibilityGraph;
    use crate::computational_geometry::*;
    #[test]
    fn test_point2_add_sub() {
        let a = Point2::new(1.0, 2.0);
        let b = Point2::new(3.0, 4.0);
        let s = a + b;
        assert!((s.x - 4.0).abs() < 1e-12);
        assert!((s.y - 6.0).abs() < 1e-12);
        let d = b - a;
        assert!((d.x - 2.0).abs() < 1e-12);
    }
    /// Verifies the `std::ops` operator impls (`Add`, `Sub`, `Mul<f64>`)
    /// give the same results as the former `add`/`sub`/`scale` methods.
    #[test]
    fn test_point2_operators() {
        let a = Point2::new(1.0, 2.0);
        let b = Point2::new(3.0, 4.0);
        // `Add`: former `a.add(b)`.
        let sum = a + b;
        assert!((sum.x - 4.0).abs() < 1e-12);
        assert!((sum.y - 6.0).abs() < 1e-12);
        // `Sub`: former `b.sub(a)`.
        let diff = b - a;
        assert!((diff.x - 2.0).abs() < 1e-12);
        assert!((diff.y - 2.0).abs() < 1e-12);
        // `Mul<f64>`: former `a.scale(t)`.
        let scaled = a * 2.5;
        assert!((scaled.x - 2.5).abs() < 1e-12);
        assert!((scaled.y - 5.0).abs() < 1e-12);
    }
    #[test]
    fn test_point2_cross() {
        let a = Point2::new(1.0, 0.0);
        let b = Point2::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < 1e-12);
        assert!((b.cross(a) + 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_point2_dist() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.dist(b) - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_cross2_ccw() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1.0);
        assert!(Point2::cross2(a, b, c) > 0.0);
    }
    #[test]
    fn test_cross3_orthogonal() {
        let x: Point3 = [1.0, 0.0, 0.0];
        let y: Point3 = [0.0, 1.0, 0.0];
        let z = cross3(x, y);
        assert!((z[0]).abs() < 1e-12);
        assert!((z[1]).abs() < 1e-12);
        assert!((z[2] - 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_normalize3() {
        let v: Point3 = [3.0, 4.0, 0.0];
        let n = normalize3(v);
        assert!((mag3(n) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_dot3() {
        let a: Point3 = [1.0, 0.0, 0.0];
        let b: Point3 = [0.0, 1.0, 0.0];
        assert!((dot3(a, b)).abs() < 1e-12);
        let c: Point3 = [2.0, 3.0, 4.0];
        assert!((dot3(c, c) - 29.0).abs() < 1e-12);
    }
    #[test]
    fn test_halfedge_add_triangle() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0]);
        let v2 = mesh.add_vertex([0.0, 1.0, 0.0]);
        let fid = mesh.add_triangle(v0, v1, v2);
        assert_eq!(mesh.num_faces(), 1);
        let verts = mesh.face_vertices(fid);
        assert_eq!(verts.len(), 3);
    }
    #[test]
    fn test_halfedge_face_normal_up() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0]);
        let v2 = mesh.add_vertex([0.0, 1.0, 0.0]);
        let fid = mesh.add_triangle(v0, v1, v2);
        let n = mesh.face_normal(fid);
        assert!(n[2] > 0.9);
    }
    #[test]
    fn test_halfedge_face_centroid() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0]);
        let v1 = mesh.add_vertex([3.0, 0.0, 0.0]);
        let v2 = mesh.add_vertex([0.0, 3.0, 0.0]);
        let fid = mesh.add_triangle(v0, v1, v2);
        let c = mesh.face_centroid(fid);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_halfedge_twin_links() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0]);
        let v2 = mesh.add_vertex([0.0, 1.0, 0.0]);
        let v3 = mesh.add_vertex([1.0, 1.0, 0.0]);
        mesh.add_triangle(v0, v1, v2);
        mesh.add_triangle(v1, v3, v2);
        mesh.build_twin_links();
        let has_twin = mesh.half_edges.iter().any(|he| he.twin != usize::MAX);
        assert!(has_twin);
    }
    #[test]
    fn test_convex_hull_3d_tetrahedron() {
        let pts: Vec<Point3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let hull = convex_hull_3d(&pts).expect("Hull should succeed");
        assert_eq!(hull.faces.len(), 4);
    }
    #[test]
    fn test_convex_hull_3d_cube_faces() {
        let pts: Vec<Point3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let hull = convex_hull_3d(&pts).expect("Cube hull should succeed");
        assert!(hull.faces.len() >= 12);
    }
    #[test]
    fn test_convex_hull_3d_volume_tetrahedron() {
        let pts: Vec<Point3> = vec![
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0],
        ];
        let hull = convex_hull_3d(&pts).expect("Hull should succeed");
        let vol = hull.volume();
        assert!((vol - 36.0).abs() < 1.0, "Volume was {vol}");
    }
    #[test]
    fn test_convex_hull_3d_surface_area_tetrahedron() {
        let pts: Vec<Point3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let hull = convex_hull_3d(&pts).expect("Hull should succeed");
        let sa = hull.surface_area();
        assert!(sa > 2.0 && sa < 3.0, "Surface area was {sa}");
    }
    #[test]
    fn test_sutherland_hodgman_unit_squares_overlap() {
        let sq: Vec<Point2> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = sutherland_hodgman(&sq, &sq);
        assert!(result.len() >= 4);
        let area = polygon_area(&result).abs();
        assert!((area - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_sutherland_hodgman_half_overlap() {
        let subject = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let clip = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = sutherland_hodgman(&subject, &clip);
        let area = polygon_area(&result).abs();
        assert!((area - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_sutherland_hodgman_no_overlap() {
        let sq1 = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let sq2 = vec![
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(2.0, 1.0),
        ];
        let result = sutherland_hodgman(&sq1, &sq2);
        assert!(result.is_empty());
    }
    #[test]
    fn test_polygon_area_square() {
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        assert!((polygon_area(&sq) - 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_minkowski_sum_two_unit_squares() {
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = minkowski_sum(&sq, &sq);
        assert!(result.len() >= 4);
        let area = polygon_area(&result).abs();
        assert!(area > 3.5 && area < 4.5, "Area was {area}");
    }
    #[test]
    fn test_minkowski_sum_nonempty() {
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = minkowski_sum(&tri, &sq);
        assert!(!result.is_empty());
    }
    #[test]
    fn test_line_arrangement_three_lines() {
        let lines = vec![
            Line2D::through(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)),
            Line2D::through(Point2::new(0.0, 1.0), Point2::new(1.0, 2.0)),
            Line2D::through(Point2::new(0.0, 3.0), Point2::new(3.0, 0.0)),
        ];
        let arr = LineArrangement::build(&lines);
        assert_eq!(arr.num_vertices(), 3);
    }
    #[test]
    fn test_line_arrangement_two_lines_one_vertex() {
        let lines = vec![
            Line2D::through(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)),
            Line2D::through(Point2::new(0.5, -1.0), Point2::new(0.5, 1.0)),
        ];
        let arr = LineArrangement::build(&lines);
        assert_eq!(arr.num_vertices(), 1);
        let v = &arr.vertices[0];
        assert!((v.point.x - 0.5).abs() < 1e-9);
        assert!((v.point.y).abs() < 1e-9);
    }
    #[test]
    fn test_line_intersect_2d_basic() {
        let l1 = Line2D {
            a: 1.0,
            b: 0.0,
            c: 2.0,
        };
        let l2 = Line2D {
            a: 0.0,
            b: 1.0,
            c: 3.0,
        };
        let pt = line_intersect_2d(&l1, &l2).expect("Should intersect");
        assert!((pt.x - 2.0).abs() < 1e-10);
        assert!((pt.y - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_line_intersect_parallel() {
        let l1 = Line2D {
            a: 1.0,
            b: 0.0,
            c: 1.0,
        };
        let l2 = Line2D {
            a: 1.0,
            b: 0.0,
            c: 2.0,
        };
        assert!(line_intersect_2d(&l1, &l2).is_none());
    }
    #[test]
    fn test_slab_locator_build() {
        let segs = vec![
            Segment2D::new(Point2::new(0.0, 1.0), Point2::new(4.0, 1.0)),
            Segment2D::new(Point2::new(0.0, 2.0), Point2::new(4.0, 2.0)),
        ];
        let locator = SlabPointLocator::build(&segs, &[None, None]);
        assert!(!locator.slab_xs.is_empty());
    }
    #[test]
    fn test_slab_locator_query_above_first() {
        let segs = vec![Segment2D::new(Point2::new(0.0, 1.0), Point2::new(4.0, 1.0))];
        let locator = SlabPointLocator::build(&segs, &[None]);
        let result = locator.locate(Point2::new(2.0, 0.5));
        assert_eq!(result, Some(0));
    }
    #[test]
    fn test_slab_locator_query_above_all() {
        let segs = vec![Segment2D::new(Point2::new(0.0, 1.0), Point2::new(4.0, 1.0))];
        let locator = SlabPointLocator::build(&segs, &[None]);
        let result = locator.locate(Point2::new(2.0, 5.0));
        assert!(result.is_none());
    }
    #[test]
    fn test_visibility_graph_no_obstacles() {
        let vg = VisibilityGraph::build(&[]);
        assert_eq!(vg.num_nodes(), 0);
    }
    #[test]
    fn test_visibility_graph_square_obstacle() {
        let obs = Obstacle2D::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        let vg = VisibilityGraph::build(&[obs]);
        assert_eq!(vg.num_nodes(), 4);
    }
    #[test]
    fn test_visibility_graph_adjacent_vertices_visible() {
        let obs = Obstacle2D::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        let vg = VisibilityGraph::build(&[obs]);
        let v0_sees_v1 = vg.edges[0].contains(&1);
        assert!(v0_sees_v1);
    }
    #[test]
    fn test_art_gallery_triangle() {
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 4.0),
        ];
        let result = art_gallery_greedy(&tri, 3);
        assert!(!result.guards.is_empty());
        assert!(result.covered.iter().all(|&c| c));
    }
    #[test]
    fn test_art_gallery_square_one_guard() {
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(0.0, 4.0),
        ];
        let result = art_gallery_greedy(&sq, 4);
        assert!(result.covered.iter().all(|&c| c));
    }
    #[test]
    fn test_check_full_coverage_with_guard() {
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(check_full_coverage(&tri, &[0]));
    }
    #[test]
    fn test_delaunay_four_points() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let tris = delaunay_2d(&pts);
        assert_eq!(tris.len(), 2);
    }
    #[test]
    fn test_delaunay_triangle_count() {
        let pts: Vec<Point2> = (0..6)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 6.0;
                Point2::new(angle.cos(), angle.sin())
            })
            .collect();
        let tris = delaunay_2d(&pts);
        assert!(!tris.is_empty());
    }
    #[test]
    fn test_circumcircle_2d_basic() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1.0);
        let result = circumcircle_2d(a, b, c);
        assert!(result.is_some());
        let (center, r2) = result.unwrap();
        let r2a = center.dist_sq(a);
        let r2b = center.dist_sq(b);
        assert!((r2a - r2b).abs() < 1e-9);
        assert!((r2a - r2).abs() < 1e-9);
    }
    #[test]
    fn test_in_circumcircle_2d() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(2.0, 0.0);
        let c = Point2::new(1.0, 2.0);
        let inside = Point2::new(1.0, 0.5);
        let outside = Point2::new(5.0, 5.0);
        assert!(in_circumcircle_2d(a, b, c, inside));
        assert!(!in_circumcircle_2d(a, b, c, outside));
    }
    #[test]
    fn test_voronoi_from_delaunay_four_points() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let tris = delaunay_2d(&pts);
        let cells = voronoi_from_delaunay(&pts, &tris);
        assert_eq!(cells.len(), 4);
        for cell in &cells {
            assert!(!cell.circumcenters.is_empty());
        }
    }
    #[test]
    fn test_ccw_positive() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1.0);
        assert!(ccw(a, b, c));
        assert!(!ccw(a, c, b));
    }
    #[test]
    fn test_collinear() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 1.0);
        let c = Point2::new(2.0, 2.0);
        assert!(collinear(a, b, c));
        let d = Point2::new(1.0, 1.5);
        assert!(!collinear(a, b, d));
    }
    #[test]
    fn test_point_in_triangle() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 0.0);
        let c = Point2::new(0.0, 3.0);
        assert!(point_in_triangle(Point2::new(0.5, 0.5), a, b, c));
        assert!(!point_in_triangle(Point2::new(3.0, 3.0), a, b, c));
    }
    #[test]
    fn test_point_in_convex_polygon() {
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(0.0, 4.0),
        ];
        assert!(point_in_convex_polygon(Point2::new(2.0, 2.0), &sq));
        assert!(!point_in_convex_polygon(Point2::new(5.0, 2.0), &sq));
    }
    #[test]
    fn test_convex_hull_2d_square() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 4);
    }
    #[test]
    fn test_convex_hull_2d_triangle() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(1.0, 0.5),
        ];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 3);
    }
    #[test]
    fn test_segments_properly_intersect_yes() {
        let p = Point2::new(0.0, 0.0);
        let q = Point2::new(2.0, 2.0);
        let a = Point2::new(0.0, 2.0);
        let b = Point2::new(2.0, 0.0);
        assert!(segments_properly_intersect(p, q, a, b));
    }
    #[test]
    fn test_segments_properly_intersect_no() {
        let p = Point2::new(0.0, 0.0);
        let q = Point2::new(1.0, 0.0);
        let a = Point2::new(2.0, 0.0);
        let b = Point2::new(3.0, 0.0);
        assert!(!segments_properly_intersect(p, q, a, b));
    }
    #[test]
    fn test_segment2d_y_at() {
        let s = Segment2D::new(Point2::new(0.0, 0.0), Point2::new(4.0, 4.0));
        assert!((s.y_at(2.0).unwrap() - 2.0).abs() < 1e-10);
        assert!(s.y_at(5.0).is_none());
    }
    #[test]
    fn test_line2d_through_signed_dist() {
        let l = Line2D::through(Point2::new(0.0, 2.0), Point2::new(1.0, 2.0));
        let above = Point2::new(0.5, 5.0);
        let below = Point2::new(0.5, 0.0);
        assert!(l.signed_dist(above) * l.signed_dist(below) < 0.0);
    }
    #[test]
    fn test_convex_hull_3d_insufficient_points() {
        let pts: Vec<Point3> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(convex_hull_3d(&pts).is_none());
    }
    #[test]
    fn test_obstacle_edges_count() {
        let obs = Obstacle2D::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        assert_eq!(obs.edges().len(), 4);
    }
    #[test]
    fn test_euler_face_count() {
        let lines = vec![
            Line2D::through(Point2::new(0.0, 0.0), Point2::new(1.0, 0.5)),
            Line2D::through(Point2::new(0.0, 1.0), Point2::new(1.0, 0.0)),
            Line2D::through(Point2::new(0.0, 0.5), Point2::new(1.0, 1.0)),
        ];
        let arr = LineArrangement::build(&lines);
        assert_eq!(arr.euler_face_count(), 7);
    }
    #[test]
    fn test_halfedge_vertex_faces() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0]);
        let v2 = mesh.add_vertex([0.0, 1.0, 0.0]);
        mesh.add_triangle(v0, v1, v2);
        mesh.build_twin_links();
        let faces = mesh.vertex_faces(v0);
        assert!(!faces.is_empty());
    }
    #[test]
    fn test_convex_hull_2d_circle_approx() {
        let n = 8usize;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / n as f64;
                Point2::new(t.cos(), t.sin())
            })
            .collect();
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), n);
    }
    #[test]
    fn test_sutherland_hodgman_triangle_clip() {
        let subject = vec![
            Point2::new(-1.0, -1.0),
            Point2::new(3.0, -1.0),
            Point2::new(1.0, 3.0),
        ];
        let clip = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = sutherland_hodgman(&subject, &clip);
        assert!(!result.is_empty());
        let area = polygon_area(&result).abs();
        assert!(area > 0.9 && area <= 1.0 + 1e-9, "area={area}");
    }
    #[test]
    fn test_voronoi_circumcenters_equidistant() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        ];
        let tris = delaunay_2d(&pts);
        let cells = voronoi_from_delaunay(&pts, &tris);
        for t in &tris {
            if let Some((center, r2)) = circumcircle_2d(pts[t.a], pts[t.b], pts[t.c]) {
                let ra = center.dist_sq(pts[t.a]);
                let rb = center.dist_sq(pts[t.b]);
                assert!((ra - r2).abs() < 1e-8);
                assert!((rb - r2).abs() < 1e-8);
            }
        }
        assert_eq!(cells.len(), 3);
    }

    // ── Minkowski sum of convex polygons ──

    /// Brute-force Minkowski sum: the convex hull of all pairwise vertex sums.
    fn minkowski_by_hull(p: &[Point2], q: &[Point2]) -> Vec<Point2> {
        let sums: Vec<Point2> = p
            .iter()
            .flat_map(|&a| q.iter().map(move |&b| a + b))
            .collect();
        convex_hull_2d(&sums)
    }

    /// `actual` is a counter-clockwise polygon with the area of `hull` that
    /// contains every vertex of `hull`.
    fn assert_matches_hull(actual: &[Point2], hull: &[Point2]) {
        let area = polygon_area(actual);
        let expected = polygon_area(hull);
        assert!(area > 0.0, "result must be counter-clockwise, area {area}");
        assert!(
            (area - expected).abs() <= 1e-12 * expected.max(1.0),
            "area {area} expected {expected}"
        );
        for h in hull {
            assert!(
                actual.iter().any(|p| p.dist(*h) < 1e-9),
                "hull vertex ({}, {}) missing from the Minkowski sum",
                h.x,
                h.y
            );
        }
    }

    fn unit_square() -> Vec<Point2> {
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]
    }

    /// Deterministic xorshift64 generator mapped to `[0, 1)`.
    struct Uniform(u64);

    impl Uniform {
        fn sample(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// A random strictly convex polygon (counter-clockwise): `k` points in
    /// sorted angular order on a randomly stretched, rotated and shifted ellipse.
    fn random_convex_polygon(rng: &mut Uniform, k: usize) -> Vec<Point2> {
        let (rx, ry) = (0.5 + 2.5 * rng.sample(), 0.5 + 2.5 * rng.sample());
        let tilt = std::f64::consts::TAU * rng.sample();
        let (cx, cy) = (10.0 * rng.sample() - 5.0, 10.0 * rng.sample() - 5.0);
        (0..k)
            .map(|i| {
                let angle =
                    std::f64::consts::TAU * (i as f64 + 0.2 + 0.6 * rng.sample()) / k as f64;
                let (x, y) = (rx * angle.cos(), ry * angle.sin());
                Point2::new(
                    cx + x * tilt.cos() - y * tilt.sin(),
                    cy + x * tilt.sin() + y * tilt.cos(),
                )
            })
            .collect()
    }

    #[test]
    fn test_minkowski_sum_terminates_when_one_polygon_runs_out_of_edges() {
        // The edges of Q turn through angles beyond the last edge of P, so the
        // merge must keep consuming Q after P is exhausted.
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(20.0, 0.0),
            Point2::new(10.0, 0.63),
        ];
        let q = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 0.325),
        ];
        let result = minkowski_sum(&p, &q);
        let expected = [
            Point2::new(0.0, 0.0),
            Point2::new(22.0, 0.0),
            Point2::new(21.0, 0.325),
            Point2::new(11.0, 0.955),
            Point2::new(1.0, 0.325),
        ];
        assert_eq!(result.len(), expected.len(), "vertices: {result:?}");
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(got.dist(*want) < 1e-12, "got {got:?} want {want:?}");
        }
        assert_matches_hull(&result, &minkowski_by_hull(&p, &q));
        // The sum is symmetric in its operands.
        assert_matches_hull(&minkowski_sum(&q, &p), &minkowski_by_hull(&p, &q));
    }

    #[test]
    fn test_minkowski_sum_fuses_parallel_edges() {
        let sq = unit_square();
        let result = minkowski_sum(&sq, &sq);
        let expected = [
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        assert_eq!(result.len(), 4, "vertices: {result:?}");
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(got.dist(*want) < 1e-12, "got {got:?} want {want:?}");
        }
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        let result = minkowski_sum(&tri, &sq);
        assert_eq!(result.len(), 5, "vertices: {result:?}");
        assert!((polygon_area(&result) - 3.5).abs() < 1e-12);
        assert_matches_hull(&result, &minkowski_by_hull(&tri, &sq));
    }

    #[test]
    fn test_minkowski_sum_matches_hull_of_pairwise_sums_for_random_polygons() {
        let mut rng = Uniform(0x2545_F491_4F6C_DD1D);
        for case in 0..300 {
            let n = 3 + (rng.sample() * 8.0) as usize;
            let m = 3 + (rng.sample() * 8.0) as usize;
            let p = random_convex_polygon(&mut rng, n);
            let q = random_convex_polygon(&mut rng, m);
            let expected = minkowski_by_hull(&p, &q);
            let result = minkowski_sum(&p, &q);
            assert!(
                result.len() <= n + m,
                "case {case}: {} vertices",
                result.len()
            );
            assert_eq!(result.len(), expected.len(), "case {case}");
            assert_matches_hull(&result, &expected);
        }
    }

    #[test]
    fn test_minkowski_sum_ignores_start_vertex_and_accepts_clockwise_input() {
        let mut rng = Uniform(0xDEAD_BEEF_CAFE_F00D);
        for case in 0..60 {
            let p = random_convex_polygon(&mut rng, 7);
            let q = random_convex_polygon(&mut rng, 5);
            let expected = minkowski_by_hull(&p, &q);
            let shift_p = (rng.sample() * 7.0) as usize;
            let shift_q = (rng.sample() * 5.0) as usize;
            let mut p_rot = p.clone();
            p_rot.rotate_left(shift_p);
            let mut q_cw = q.clone();
            q_cw.rotate_left(shift_q);
            q_cw.reverse();
            let result = minkowski_sum(&p_rot, &q_cw);
            assert_eq!(result.len(), expected.len(), "case {case}");
            assert_matches_hull(&result, &expected);
        }
    }

    #[test]
    fn test_minkowski_sum_of_degenerate_polygons() {
        let sq = unit_square();
        // A single point translates the other polygon.
        let moved = minkowski_sum(&[Point2::new(3.0, 4.0)], &sq);
        let expected_moved = [
            Point2::new(3.0, 4.0),
            Point2::new(4.0, 4.0),
            Point2::new(4.0, 5.0),
            Point2::new(3.0, 5.0),
        ];
        assert_eq!(moved.len(), 4);
        for (got, want) in moved.iter().zip(expected_moved.iter()) {
            assert!(got.dist(*want) < 1e-12, "got {got:?} want {want:?}");
        }
        // A segment sweeps the square into a rectangle.
        let seg = [Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let rect = minkowski_sum(&seg, &sq);
        assert_eq!(rect.len(), 4, "vertices: {rect:?}");
        assert!((polygon_area(&rect) - 3.0).abs() < 1e-12);
        // Two non-parallel segments give a parallelogram.
        let diag = [Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)];
        let para = minkowski_sum(&seg, &diag);
        let expected_para = [
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        assert_eq!(para.len(), 4, "vertices: {para:?}");
        for (got, want) in para.iter().zip(expected_para.iter()) {
            assert!(got.dist(*want) < 1e-12, "got {got:?} want {want:?}");
        }
        // Repeated vertices and a closing copy of the first vertex are ignored.
        let mut noisy = sq.clone();
        noisy.insert(1, sq[0]);
        noisy.push(sq[0]);
        let result = minkowski_sum(&noisy, &sq);
        assert_eq!(result.len(), 4, "vertices: {result:?}");
        assert!((polygon_area(&result) - 4.0).abs() < 1e-12);
        // Empty operands give an empty sum.
        assert!(minkowski_sum(&[], &sq).is_empty());
        assert!(minkowski_sum(&sq, &[]).is_empty());
    }

    #[test]
    fn test_minkowski_sum_keeps_collinear_input_vertices() {
        // A square with an extra vertex in the middle of its bottom edge.
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let q = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let result = minkowski_sum(&p, &q);
        assert!(result.len() <= p.len() + q.len());
        assert_matches_hull(&result, &minkowski_by_hull(&p, &q));
    }
}
