// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the pressure load and the inflation of
//! [`InflatableShell`].
//!
//! `balloon_model` used to wind its triangles clockwise seen from outside
//! although [`InflatableShell::tris`] is documented as counter-clockwise, and
//! `InflatableShell::pressure_force` trusted the raw winding. A positive
//! (under-inflated) pressure therefore pushed the generated balloon inward and
//! the inflation step shrank it.

use super::*;

/// The same shell with every triangle wound the other way (inward normals).
fn reversed(shell: &InflatableShell) -> InflatableShell {
    let mut flipped = shell.clone();
    for tri in &mut flipped.tris {
        tri.swap(1, 2);
    }
    flipped
}

/// `shell` with every vertex displaced by a smooth, deterministic offset, so
/// that the mesh is no longer symmetric.
fn perturbed(mut shell: InflatableShell, amplitude: f64) -> InflatableShell {
    for (index, vertex) in shell.vertices.iter_mut().enumerate() {
        let s = index as f64;
        vertex[0] += amplitude * (1.3 * s).sin();
        vertex[1] += amplitude * (0.7 * s).cos();
        vertex[2] += amplitude * (2.1 * s).sin();
    }
    shell
}

#[test]
fn balloon_model_encloses_a_positive_signed_volume() {
    for panels in [2_usize, 3, 4, 6, 9] {
        let shell = balloon_model(panels, 1.5);
        let signed = shell.signed_enclosed_volume();
        assert!(
            signed > 0.0,
            "balloon_model({panels}): signed volume {signed}"
        );
        assert!(signed < 4.0 / 3.0 * std::f64::consts::PI * 1.5_f64.powi(3));
        assert_eq!(shell.compute_enclosed_volume(), signed);

        let flipped = reversed(&shell);
        assert_eq!(flipped.signed_enclosed_volume(), -signed);
        assert_eq!(flipped.compute_enclosed_volume(), signed);
    }
}

#[test]
fn positive_pressure_pushes_every_vertex_outward_whichever_way_the_mesh_winds() {
    let shell = balloon_model(4, 1.0);
    for (name, mut shell) in [("generated", shell.clone()), ("reversed", reversed(&shell))] {
        shell.pressure = 250.0;
        for (index, vertex) in shell.vertices.iter().enumerate() {
            let force = shell.pressure_force(index);
            let outward = vec_dot(force, *vertex);
            assert!(
                outward > 0.0,
                "{name}: vertex {index} at {vertex:?} is pushed inward by {force:?}"
            );
            // The sphere is symmetric, so the push is radial up to the mesh.
            assert!(
                outward > 0.8 * vec_len(force) * vec_len(*vertex),
                "{name}: vertex {index}: the force {force:?} is far from radial"
            );
        }
    }
}

#[test]
fn negative_pressure_pulls_every_vertex_inward() {
    let shell = balloon_model(4, 1.0);
    for (name, mut shell) in [("generated", shell.clone()), ("reversed", reversed(&shell))] {
        shell.pressure = -250.0;
        for (index, vertex) in shell.vertices.iter().enumerate() {
            let force = shell.pressure_force(index);
            assert!(
                vec_dot(force, *vertex) < 0.0,
                "{name}: vertex {index} at {vertex:?} is pushed outward by {force:?}"
            );
        }
    }
}

#[test]
fn the_pressure_force_does_not_depend_on_the_winding() {
    let shell = perturbed(balloon_model(4, 1.0), 0.05);
    let flipped = reversed(&shell);
    for pressure in [250.0, -40.0] {
        let mut a = shell.clone();
        let mut b = flipped.clone();
        a.pressure = pressure;
        b.pressure = pressure;
        for index in 0..a.vertices.len() {
            let fa = a.pressure_force(index);
            let fb = b.pressure_force(index);
            for axis in 0..3 {
                assert!(
                    (fa[axis] - fb[axis]).abs() < 1e-12 * pressure.abs().max(1.0),
                    "vertex {index} axis {axis}: {} vs {}",
                    fa[axis],
                    fb[axis]
                );
            }
        }
    }
}

#[test]
fn the_pressure_force_is_the_pressure_times_the_gradient_of_the_volume() {
    // The virtual work of a uniform pressure is p dV, so the force on vertex i
    // is p times the gradient of the enclosed volume with respect to x_i.
    let shell = perturbed(balloon_model(3, 1.0), 0.05);
    let step = 1e-6;
    for (name, mut shell) in [("generated", shell.clone()), ("reversed", reversed(&shell))] {
        shell.pressure = 120.0;
        for index in 0..shell.vertices.len() {
            let force = shell.pressure_force(index);
            for (axis, analytic) in force.iter().enumerate() {
                let mut plus = shell.clone();
                plus.vertices[index][axis] += step;
                let mut minus = shell.clone();
                minus.vertices[index][axis] -= step;
                let numeric = shell.pressure
                    * (plus.compute_enclosed_volume() - minus.compute_enclosed_volume())
                    / (2.0 * step);
                assert!(
                    (analytic - numeric).abs() < 1e-5,
                    "{name}: vertex {index} axis {axis}: {analytic} vs {numeric}"
                );
            }
        }
    }
}

#[test]
fn a_uniform_pressure_has_no_net_force_and_the_virial_of_three_pv() {
    let shell = perturbed(balloon_model(5, 1.2), 0.04);
    for (name, mut shell) in [("generated", shell.clone()), ("reversed", reversed(&shell))] {
        shell.pressure = 75.0;
        let mut net = [0.0_f64; 3];
        let mut virial = 0.0;
        for (index, vertex) in shell.vertices.iter().enumerate() {
            let force = shell.pressure_force(index);
            net = vec_add(net, force);
            virial += vec_dot(force, *vertex);
        }
        assert!(
            vec_len(net) < 1e-9 * shell.pressure,
            "{name}: a closed shell under uniform pressure has no net force, got {net:?}"
        );
        // The volume is homogeneous of degree three: sum x_i . grad_i V = 3 V.
        let expected = 3.0 * shell.pressure * shell.compute_enclosed_volume();
        assert!(
            (virial - expected).abs() < 1e-9 * expected,
            "{name}: virial {virial} vs 3 p V = {expected}"
        );
    }
}

#[test]
fn pressure_forces_returns_the_force_on_every_vertex() {
    let shell = perturbed(balloon_model(4, 1.0), 0.05);
    for (name, mut shell) in [("generated", shell.clone()), ("reversed", reversed(&shell))] {
        shell.pressure = 310.0;
        let all = shell.pressure_forces();
        assert_eq!(all.len(), shell.vertices.len(), "{name}");
        for (index, force) in all.iter().enumerate() {
            let single = shell.pressure_force(index);
            for axis in 0..3 {
                assert!(
                    (force[axis] - single[axis]).abs() < 1e-12 * shell.pressure,
                    "{name}: vertex {index} axis {axis}: {} vs {}",
                    force[axis],
                    single[axis]
                );
            }
        }
    }
}

#[test]
fn an_under_inflated_balloon_grows_toward_its_target_volume() {
    let mut shell = balloon_model(3, 0.5);
    let initial = shell.compute_enclosed_volume();
    shell.target_volume = 2.0 * initial;
    let mut volumes = vec![initial];
    for _ in 0..50 {
        shell.step(1.0 / 600.0, 50.0, 0.1);
        volumes.push(shell.compute_enclosed_volume());
    }
    let final_volume = volumes[volumes.len() - 1];
    let peak = volumes.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        shell.signed_enclosed_volume() > 0.0,
        "the balloon turned inside out"
    );
    // The membrane stops the balloon far short of the doubled volume and the
    // undamped shell then rings, but the pressure first drives it outward.
    assert!(
        volumes[..=25].windows(2).all(|pair| pair[1] > pair[0]),
        "the volume must grow at every one of the first steps: {volumes:?}"
    );
    assert!(
        peak > 1.1 * initial,
        "peak volume {peak} vs initial {initial}"
    );
    assert!(
        final_volume > 1.05 * initial,
        "the balloon should end up inflated: {initial} -> {final_volume}"
    );
    assert!(shell.pressure > 0.0, "pressure {}", shell.pressure);
    assert!(
        shell
            .vertices
            .iter()
            .all(|v| v.iter().all(|x| x.is_finite())),
        "the vertices must stay finite"
    );
}

#[test]
fn a_balloon_without_membrane_stiffness_inflates_freely() {
    let mut shell = balloon_model(3, 0.5);
    let initial = shell.compute_enclosed_volume();
    shell.target_volume = 2.0 * initial;
    let mut previous = initial;
    for step in 0..50 {
        shell.step(1.0 / 600.0, 50.0, 0.0);
        let volume = shell.compute_enclosed_volume();
        assert!(volume > previous, "step {step}: {previous} -> {volume}");
        previous = volume;
    }
    assert!(
        previous > 2.0 * initial * 0.9,
        "without a membrane the balloon should be near its target: {previous} vs {}",
        2.0 * initial
    );
}

#[test]
fn an_over_inflated_balloon_shrinks_toward_its_target_volume() {
    let mut shell = balloon_model(3, 0.5);
    let initial = shell.compute_enclosed_volume();
    shell.target_volume = 0.5 * initial;
    for _ in 0..50 {
        shell.step(1.0 / 600.0, 50.0, 0.1);
    }
    let final_volume = shell.compute_enclosed_volume();
    assert!(
        final_volume < initial,
        "negative pressure should deflate the balloon: {initial} -> {final_volume}"
    );
    assert!(shell.pressure < 0.0, "pressure {}", shell.pressure);
}

#[test]
fn inflation_does_not_depend_on_the_winding_of_the_mesh() {
    let shell = balloon_model(3, 0.5);
    let mut generated = shell.clone();
    let mut flipped = reversed(&shell);
    let target = 2.0 * shell.compute_enclosed_volume();
    generated.target_volume = target;
    flipped.target_volume = target;
    for _ in 0..50 {
        generated.step(1.0 / 600.0, 50.0, 0.1);
        flipped.step(1.0 / 600.0, 50.0, 0.1);
    }
    assert!(flipped.compute_enclosed_volume() > shell.compute_enclosed_volume());
    for (a, b) in generated.vertices.iter().zip(flipped.vertices.iter()) {
        assert!(
            vec_len(vec_sub(*a, *b)) < 1e-12,
            "the two windings must follow the same trajectory: {a:?} vs {b:?}"
        );
    }
}
