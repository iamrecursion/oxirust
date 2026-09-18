// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the scale-invariant degeneracy test and the OBJ import.

use super::functions::{
    obj_to_stl, remove_degenerate_triangles, scale_uniform, stl_to_obj, validate_mesh, validate_stl,
};
use super::functions_2::compute_quality_metrics;
use super::types::{StlMesh, StlTriangle};
use crate::Error;

/// Unit cube, outward normals, 12 triangles.
fn unit_cube() -> StlMesh {
    let mut m = StlMesh::new("cube");
    let quads: [[[f32; 3]; 4]; 6] = [
        [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
        [[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
        [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
        [[0., 1., 0.], [0., 1., 1.], [1., 1., 1.], [1., 1., 0.]],
        [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
    ];
    for q in quads {
        m.add_triangle(q[0], q[1], q[2]);
        m.add_triangle(q[0], q[2], q[3]);
    }
    m
}

fn parse_error_message(result: crate::Result<StlMesh>) -> String {
    match result {
        Err(Error::Parse(message)) => message,
        Err(other) => panic!("expected a parse error, got {other}"),
        Ok(mesh) => panic!(
            "expected a parse error, got a mesh of {} triangles",
            mesh.triangles.len()
        ),
    }
}

// --- degeneracy -------------------------------------------------------------------

#[test]
fn cube_scaled_by_a_millionth_keeps_all_its_triangles() {
    let mut cube = unit_cube();
    scale_uniform(&mut cube, 1e-6);
    assert_eq!(cube.triangles.len(), 12);
    remove_degenerate_triangles(&mut cube);
    assert_eq!(
        cube.triangles.len(),
        12,
        "micro-scale triangles were deleted"
    );
}

#[test]
fn degenerate_counts_do_not_depend_on_the_scale_of_the_mesh() {
    for scale in [1e-9_f32, 1e-6, 1e-3, 1.0, 1e3] {
        let mut cube = unit_cube();
        scale_uniform(&mut cube, scale);
        assert_eq!(
            validate_stl(&cube).degenerate_count,
            0,
            "validate_stl, scale {scale}"
        );
        assert_eq!(
            validate_mesh(&cube).degenerate_count,
            0,
            "validate_mesh, scale {scale}"
        );
        assert_eq!(
            compute_quality_metrics(&cube).degenerate_fraction,
            0.0,
            "quality metrics, scale {scale}"
        );
    }
}

#[test]
fn collinear_triangles_are_degenerate_at_every_scale() {
    for scale in [1e-9_f32, 1e-6, 1.0, 1e4] {
        let mut mesh = unit_cube();
        scale_uniform(&mut mesh, scale);
        // Three points on a line and a triangle that collapsed to one point.
        mesh.add_triangle([0.0, 0.0, 0.0], [scale, 0.0, 0.0], [2.0 * scale, 0.0, 0.0]);
        mesh.add_triangle(
            [scale, scale, scale],
            [scale, scale, scale],
            [scale, scale, scale],
        );
        assert_eq!(
            validate_stl(&mesh).degenerate_count,
            2,
            "validate_stl, scale {scale}"
        );
        assert_eq!(
            validate_mesh(&mesh).degenerate_count,
            2,
            "validate_mesh, scale {scale}"
        );
        let fraction = compute_quality_metrics(&mesh).degenerate_fraction;
        assert!(
            (fraction - 2.0 / 14.0).abs() < 1e-6,
            "fraction {fraction}, scale {scale}"
        );
        remove_degenerate_triangles(&mut mesh);
        assert_eq!(mesh.triangles.len(), 12, "scale {scale}");
    }
}

#[test]
fn slivers_are_judged_by_their_height_over_the_longest_edge() {
    let mut mesh = StlMesh::new("slivers");
    // Height 1e-7 over a unit base: below the 1e-6 tolerance, degenerate.
    mesh.add_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-7, 0.0]);
    // Height 1e-3 over a unit base: thin but a real triangle.
    mesh.add_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-3, 0.0]);
    assert_eq!(validate_stl(&mesh).degenerate_count, 1);
    assert_eq!(validate_mesh(&mesh).degenerate_count, 1);
    remove_degenerate_triangles(&mut mesh);
    assert_eq!(mesh.triangles.len(), 1);
    assert!((mesh.triangles[0].v2[1] - 1e-3).abs() < 1e-9);
}

#[test]
fn ten_micrometre_triangles_in_metres_are_not_degenerate() {
    // Edges of 10 micrometres in a metre-unit file have an area of 5e-11, below
    // the old absolute cut-off of 1e-10 that erased them.
    let mut mesh = StlMesh::new("micro");
    mesh.add_triangle([0.0, 0.0, 0.0], [1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0]);
    mesh.add_triangle([1e-5, 0.0, 0.0], [1e-5, 1e-5, 0.0], [0.0, 1e-5, 0.0]);
    assert_eq!(validate_stl(&mesh).degenerate_count, 0);
    assert_eq!(validate_mesh(&mesh).degenerate_count, 0);
    remove_degenerate_triangles(&mut mesh);
    assert_eq!(mesh.triangles.len(), 2);
}

#[test]
fn small_triangles_far_from_the_origin_are_not_degenerate() {
    // Corners are exact f32 values (2^-10 is eight ulps at 1000); the edge
    // vectors must survive the subtraction.
    let step = 1.0_f32 / 1024.0;
    let mut mesh = StlMesh::new("far");
    mesh.add_triangle(
        [1000.0, 1000.0, 1000.0],
        [1000.0 + step, 1000.0, 1000.0],
        [1000.0, 1000.0 + step, 1000.0],
    );
    assert_eq!(validate_stl(&mesh).degenerate_count, 0);
    assert_eq!(validate_mesh(&mesh).degenerate_count, 0);
}

#[test]
fn triangles_with_nan_corners_are_still_removed_and_not_counted_as_degenerate() {
    let mut mesh = unit_cube();
    mesh.triangles.push(StlTriangle {
        normal: [0.0, 0.0, 1.0],
        v0: [f32::NAN, 0.0, 0.0],
        v1: [1.0, 0.0, 0.0],
        v2: [0.0, 1.0, 0.0],
    });
    let report = validate_stl(&mesh);
    assert_eq!(report.nan_inf_count, 1);
    assert_eq!(report.degenerate_count, 0);
    assert_eq!(validate_mesh(&mesh).degenerate_count, 0);
    remove_degenerate_triangles(&mut mesh);
    assert_eq!(mesh.triangles.len(), 12);
}

// --- OBJ import -------------------------------------------------------------------

#[test]
fn obj_quad_becomes_two_triangles() {
    let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
    let mesh = obj_to_stl(obj).expect("valid quad");
    assert_eq!(mesh.triangles.len(), 2);
    assert_eq!(mesh.triangles[0].v0, [0.0, 0.0, 0.0]);
    assert_eq!(mesh.triangles[0].v1, [1.0, 0.0, 0.0]);
    assert_eq!(mesh.triangles[0].v2, [1.0, 1.0, 0.0]);
    assert_eq!(mesh.triangles[1].v0, [0.0, 0.0, 0.0]);
    assert_eq!(mesh.triangles[1].v1, [1.0, 1.0, 0.0]);
    assert_eq!(mesh.triangles[1].v2, [0.0, 1.0, 0.0]);
    assert!((mesh.surface_area() - 1.0).abs() < 1e-6);
}

#[test]
fn obj_polygon_is_fan_triangulated_about_its_first_vertex() {
    let obj = "v 0 0 0\nv 2 0 0\nv 3 1 0\nv 1.5 2 0\nv 0 1 0\nf 1 2 3 4 5\n";
    let mesh = obj_to_stl(obj).expect("valid pentagon");
    assert_eq!(mesh.triangles.len(), 3);
    for tri in &mesh.triangles {
        assert_eq!(tri.v0, [0.0, 0.0, 0.0]);
    }
    assert_eq!(mesh.triangles[2].v2, [0.0, 1.0, 0.0]);
}

#[test]
fn obj_negative_indices_count_back_from_the_last_defined_vertex() {
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nf -3 -2 -1\n";
    let mesh = obj_to_stl(obj).expect("relative indices");
    assert_eq!(mesh.triangles.len(), 1);
    assert_eq!(mesh.triangles[0].v0, [1.0, 0.0, 0.0]);
    assert_eq!(mesh.triangles[0].v1, [0.0, 1.0, 0.0]);
    assert_eq!(mesh.triangles[0].v2, [5.0, 5.0, 5.0]);
}

#[test]
fn obj_negative_indices_are_relative_to_the_vertices_defined_so_far() {
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -3\nv 0 0 1\nf -1 -3 -4\n";
    let mesh = obj_to_stl(obj).expect("relative indices");
    assert_eq!(mesh.triangles.len(), 2);
    assert_eq!(mesh.triangles[0].v0, [0.0, 1.0, 0.0]);
    assert_eq!(mesh.triangles[0].v2, [0.0, 0.0, 0.0]);
    // After the fourth vertex: -1 is (0,0,1), -3 is (1,0,0), -4 is (0,0,0).
    assert_eq!(mesh.triangles[1].v0, [0.0, 0.0, 1.0]);
    assert_eq!(mesh.triangles[1].v1, [1.0, 0.0, 0.0]);
    assert_eq!(mesh.triangles[1].v2, [0.0, 0.0, 0.0]);
}

#[test]
fn obj_index_zero_is_an_error_not_a_panic() {
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"));
    assert!(message.starts_with("line 4:"), "{message}");
    assert!(message.contains("bad index"), "{message}");
}

#[test]
fn obj_index_beyond_the_vertex_list_is_an_error_not_the_origin() {
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n"));
    assert!(message.starts_with("line 4:"), "{message}");
    assert!(message.contains("bad index 4"), "{message}");
}

#[test]
fn obj_relative_index_before_the_first_vertex_is_an_error() {
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 0\nf -1 -2 -3\n"));
    assert!(message.starts_with("line 3:"), "{message}");
}

#[test]
fn obj_malformed_records_report_their_line() {
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0\nf 1 1 1\n"));
    assert!(message.starts_with("line 2:"), "{message}");
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 x\n"));
    assert!(message.starts_with("line 2:"), "{message}");
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 three\n"));
    assert!(message.starts_with("line 4:"), "{message}");
    let message = parse_error_message(obj_to_stl("v 0 0 0\nv 1 0 0\nf 1 2\n"));
    assert!(message.starts_with("line 3:"), "{message}");
    assert!(message.contains("at least three"), "{message}");
}

#[test]
fn obj_face_reference_forms_and_whitespace_are_accepted() {
    let obj = "# header\n\
               o my part\n\
               v\t0 0 0\n\
               v 1 0 0 1.0\n\
               v 0 1 0   # trailing comment\n\
               vt 0 0\n\
               vn 0 0 1\n\
               s off\n\
               f 1/1/1 2//1 3/1\n\
               f\t1\t2\t3\n";
    let mesh = obj_to_stl(obj).expect("valid obj");
    assert_eq!(mesh.name, "my part");
    assert_eq!(mesh.triangles.len(), 2);
    assert_eq!(mesh.triangles[0].v2, [0.0, 1.0, 0.0]);
}

#[test]
fn obj_default_name_and_empty_input() {
    let mesh = obj_to_stl("").expect("empty input is a valid empty mesh");
    assert_eq!(mesh.name, "imported");
    assert!(mesh.triangles.is_empty());
    // An `o` record without a name does not erase the previous name.
    let mesh = obj_to_stl("o first\no\no   \n").expect("valid");
    assert_eq!(mesh.name, "first");
}

#[test]
fn obj_faces_may_precede_the_vertices_they_reference() {
    let obj = "f 1 2 3\nv 0 0 0\nv 1 0 0\nv 0 1 0\n";
    let mesh = obj_to_stl(obj).expect("forward references are resolved at the end");
    assert_eq!(mesh.triangles.len(), 1);
    assert_eq!(mesh.triangles[0].v1, [1.0, 0.0, 0.0]);
}

#[test]
fn stl_to_obj_and_back_preserves_every_vertex_bit_for_bit() {
    let mut mesh = StlMesh::new("round_trip");
    mesh.add_triangle(
        [0.1, 0.2, 0.3],
        [1.0 / 3.0, -2.5e-7, 7.0e12],
        [0.0, -0.0, 1.5],
    );
    mesh.add_triangle([0.1, 0.2, 0.3], [0.0, -0.0, 1.5], [9.0, 8.0, 7.0]);
    let again = obj_to_stl(&stl_to_obj(&mesh)).expect("own output parses");
    assert_eq!(again.triangles.len(), 2);
    for (a, b) in mesh.triangles.iter().zip(again.triangles.iter()) {
        for (va, vb) in [(a.v0, b.v0), (a.v1, b.v1), (a.v2, b.v2)] {
            assert_eq!(va.map(f32::to_bits), vb.map(f32::to_bits));
        }
    }
}
