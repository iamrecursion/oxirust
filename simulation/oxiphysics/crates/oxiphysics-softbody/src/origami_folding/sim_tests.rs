// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the bar-and-hinge folding simulators.

use super::*;

/// Two triangles sharing the crease (0, 1) along +x; wing 2 at +y, wing 3 at −y.
fn single_crease(mv: MountainValley, stiffness: f64, pin_axis: bool) -> CreasePattern {
    let mut p = CreasePattern::new();
    if pin_axis {
        p.add_pinned_vertex([0.0_f64, 0.0_f64]);
        p.add_pinned_vertex([1.0_f64, 0.0_f64]);
    } else {
        p.add_vertex([0.0_f64, 0.0_f64], MountainValley::Boundary, 0.01_f64);
        p.add_vertex([1.0_f64, 0.0_f64], MountainValley::Boundary, 0.01_f64);
    }
    p.add_vertex([0.4_f64, 0.8_f64], MountainValley::Boundary, 0.01_f64);
    p.add_vertex([0.6_f64, -0.7_f64], MountainValley::Boundary, 0.01_f64);
    p.add_fold_line(0, 1, mv, stiffness);
    p.add_panel(vec![0, 1, 2]);
    p.add_panel(vec![1, 0, 3]);
    p
}

#[test]
fn dynamic_single_crease_reaches_target_with_rigid_bars() {
    let p = single_crease(MountainValley::Valley, 5.0_f64, false);
    let mut sim = FoldingSimulation::new(p, 8.0_f64, 20, 0.0_f64);
    // Valley target is π; fold half way → commanded angle π/2.
    sim.target_fold_param = 0.5_f64;
    sim.fold_speed = 0.5_f64;
    sim.begin_folding();
    for _ in 0..4000 {
        sim.step(0.005_f64);
    }
    let fl = &sim.pattern.fold_lines[0];
    let target = PI * 0.5_f64;
    assert!(
        (fl.current_angle - target).abs() < 1e-3_f64,
        "angle {} vs target {target}",
        fl.current_angle
    );
    // current_angle is measured from the geometry.
    let v = &sim.pattern.vertices;
    let measured = signed_fold_angle(v[2].position, v[0].position, v[1].position, v[3].position);
    assert!((measured - fl.current_angle).abs() < 1e-12_f64);
    assert!(
        sim.max_bar_error() < 1e-6_f64,
        "bar error {}",
        sim.max_bar_error()
    );
    // Geometry actually moved out of plane.
    assert!(v.iter().any(|x| x.position[2].abs() > 0.1_f64));
}

#[test]
fn dynamic_mountain_crease_folds_negative() {
    let p = single_crease(MountainValley::Mountain, 5.0_f64, true);
    let mut sim = FoldingSimulation::new(p, 8.0_f64, 20, 0.0_f64);
    sim.target_fold_param = 0.25_f64;
    sim.fold_speed = 0.5_f64;
    sim.begin_folding();
    for _ in 0..3000 {
        sim.step(0.005_f64);
    }
    let angle = sim.pattern.fold_lines[0].current_angle;
    assert!((angle + PI * 0.25_f64).abs() < 1e-3_f64, "angle {angle}");
    // Pinned axis vertices never move.
    assert_eq!(
        sim.pattern.vertices[0].position,
        [0.0_f64, 0.0_f64, 0.0_f64]
    );
    assert_eq!(
        sim.pattern.vertices[1].position,
        [1.0_f64, 0.0_f64, 0.0_f64]
    );
    // Wings rise below the sheet for a mountain fold seen from +z.
    assert!(sim.pattern.vertices[2].position[2] < 0.0_f64);
    assert!(sim.pattern.vertices[3].position[2] < 0.0_f64);
    assert!(sim.max_bar_error() < 1e-6_f64);
}

#[test]
fn dynamic_gravity_sags_pinned_flap() {
    // Pin one panel completely; the free wing falls under gravity and the
    // inactive (unactuated) crease rotates.
    let mut p = single_crease(MountainValley::Valley, 5.0_f64, true);
    p.vertices[2].is_pinned = true;
    p.vertices[2].inv_mass = 0.0_f64;
    let mut sim = FoldingSimulation::new(p, 2.0_f64, 10, 9.81_f64);
    for _ in 0..200 {
        sim.step(0.005_f64);
    }
    assert!(sim.pattern.vertices[3].position[2] < -0.05_f64);
    assert!(sim.max_bar_error() < 1e-6_f64);
    assert!(sim.pattern.fold_lines[0].current_angle.abs() > 0.05_f64);
}

#[test]
fn rigid_folding_reaches_target_and_preserves_bars() {
    let mut p = single_crease(MountainValley::Valley, 1.0_f64, true);
    p.fold_lines[0].target_angle = 2.0_f64;
    let mut rfold = OrigamiRigidFolding::new(p, 10, 4.0_f64, 0.0_f64);
    rfold.activate_all();
    rfold.run_to_convergence(0.05_f64, 2000, 1e-6_f64);
    let fl = &rfold.pattern.fold_lines[0];
    assert!(
        (fl.current_angle - 2.0_f64).abs() < 1e-3_f64,
        "angle {}",
        fl.current_angle
    );
    assert!(rfold.max_bar_error() < 1e-6_f64);
    assert_eq!(rfold.converged_count(), 1);
    assert!((rfold.fold_param - 1.0_f64).abs() < 1e-12_f64);
}

#[test]
fn rigid_folding_quad_panels_stay_flat() {
    // Two unit squares sharing the crease (1, 4).
    let mut p = CreasePattern::new();
    p.add_pinned_vertex([0.0_f64, 0.0_f64]);
    p.add_pinned_vertex([1.0_f64, 0.0_f64]);
    p.add_vertex([2.0_f64, 0.0_f64], MountainValley::Boundary, 1.0_f64);
    p.add_pinned_vertex([0.0_f64, 1.0_f64]);
    p.add_pinned_vertex([1.0_f64, 1.0_f64]);
    p.add_vertex([2.0_f64, 1.0_f64], MountainValley::Boundary, 1.0_f64);
    let crease = p.add_fold_line(1, 4, MountainValley::Valley, 1.0_f64);
    p.fold_lines[crease].target_angle = 1.0_f64;
    p.add_panel(vec![0, 1, 4, 3]);
    p.add_panel(vec![1, 2, 5, 4]);
    let mut rfold = OrigamiRigidFolding::new(p, 10, 4.0_f64, 0.0_f64);
    rfold.activate(crease);
    rfold.run_to_convergence(0.05_f64, 2000, 1e-6_f64);
    let v = &rfold.pattern.vertices;
    assert!((rfold.pattern.fold_lines[crease].current_angle - 1.0_f64).abs() < 1e-3_f64);
    assert!(rfold.max_bar_error() < 1e-6_f64);
    // The free panel stays planar: its flatness hinge is ~0.
    for h in &rfold.model().panel_hinges {
        let a = h.angle(v).unwrap_or(f64::NAN);
        assert!(a.abs() < 1e-5_f64, "panel hinge angle {a}");
    }
    // The folded panel's free edge rotated about the crease by the fold
    // angle: z = sin(1), x = 1 + cos(1).
    assert!((v[2].position[2] - 1.0_f64.sin()).abs() < 1e-3_f64);
    assert!((v[2].position[0] - (1.0_f64 + 1.0_f64.cos())).abs() < 1e-3_f64);
}

#[test]
fn boundary_fold_lines_have_no_hinge() {
    let mut p = single_crease(MountainValley::Valley, 1.0_f64, false);
    p.add_fold_line(1, 2, MountainValley::Mountain, 1.0_f64);
    let sim = FoldingSimulation::new(p, 1.0_f64, 5, 0.0_f64);
    assert!(sim.model().fold_line_hinge[0].is_some());
    assert!(sim.model().fold_line_hinge[1].is_none());
}

#[test]
fn miura_crease_pattern_has_panels_and_interior_hinges() {
    let m = MiuraOri::new(2, 2, 0.1_f64, 0.1_f64, PI / 3.0_f64);
    let pat = m.to_crease_pattern(1.0_f64, 0.01_f64);
    let nx = 2 * 2 + 1;
    let ny = 2 + 1;
    assert_eq!(pat.panels.len(), (nx - 1) * (ny - 1));
    let model = BarHingeModel::from_pattern(&pat);
    // Interior creases: horizontal (nx−1)(ny−2) + vertical (nx−2)(ny−1).
    let interior = (nx - 1) * (ny - 2) + (nx - 2) * (ny - 1);
    assert_eq!(model.fold_hinges.len(), interior);
}

#[test]
fn step_ignores_non_positive_dt() {
    let p = single_crease(MountainValley::Valley, 1.0_f64, false);
    let mut sim = FoldingSimulation::new(p, 1.0_f64, 5, 9.81_f64);
    sim.begin_folding();
    sim.step(0.0_f64);
    sim.step(f64::NAN);
    assert_eq!(sim.time, 0.0_f64);
    assert!(
        sim.pattern
            .vertices
            .iter()
            .all(|v| v.position[2] == 0.0_f64)
    );
}
