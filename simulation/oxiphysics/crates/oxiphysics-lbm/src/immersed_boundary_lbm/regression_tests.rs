// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the immersed-boundary delta kernels, the velocity
//! interpolation weights and the bending model of [`FlexibleIbmFiber`].

use super::*;

/// Sub-grid offsets `s` at which the kernel moment conditions are checked.
const OFFSETS: [f64; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// A discrete delta kernel `φ(r)`.
type Kernel = fn(f64) -> f64;

/// Zeroth and first discrete moments `(Σ_k φ(s − k), Σ_k (s − k) φ(s − k))`.
fn kernel_moments(kernel: Kernel, s: f64) -> (f64, f64) {
    let mut m0 = 0.0;
    let mut m1 = 0.0;
    for k in -4_i64..=5 {
        let r = s - k as f64;
        let w = kernel(r);
        m0 += w;
        m1 += r * w;
    }
    (m0, m1)
}

// ---------------------------------------------------------------------------
// Delta kernels (F2581)
// ---------------------------------------------------------------------------

#[test]
fn delta_kernels_satisfy_partition_of_unity_and_first_moment() {
    let kernels: [(&str, Kernel); 2] = [("delta_phi", delta_phi), ("delta_peskin4", delta_peskin4)];
    for (name, kernel) in kernels {
        for s in OFFSETS {
            let (m0, m1) = kernel_moments(kernel, s);
            assert!((m0 - 1.0).abs() < 1e-12, "{name}: sum at s={s} is {m0}");
            assert!(m1.abs() < 1e-12, "{name}: first moment at s={s} is {m1}");
        }
    }
}

#[test]
fn delta_kernels_are_continuous_at_support_edges_and_joins() {
    // Roma 3-point kernel: smooth join at |r| = 1/2 and zero at |r| = 3/2.
    let eps = 1e-9;
    assert!((delta_phi(0.5 - eps) - delta_phi(0.5 + eps)).abs() < 1e-6);
    assert!(
        delta_phi(1.5 - eps) < 1e-8,
        "phi(1.5-) = {}",
        delta_phi(1.5 - eps)
    );
    assert_eq!(delta_phi(1.5 + eps), 0.0);
    // Peskin 4-point kernel: join at |r| = 1 and vanishing at |r| = 2
    // (the former cosine kernel jumped from 1/8 to 0 there).
    assert!((delta_peskin4(1.0 - eps) - delta_peskin4(1.0 + eps)).abs() < 1e-6);
    assert!(
        delta_peskin4(2.0 - eps) < 1e-8,
        "peskin4(2-) = {}",
        delta_peskin4(2.0 - eps)
    );
    assert_eq!(delta_peskin4(2.0 + eps), 0.0);
}

#[test]
fn delta_kernels_are_non_negative_and_even() {
    let mut r = -2.5;
    while r <= 2.5 {
        for kernel in [delta_phi as Kernel, delta_peskin4] {
            assert!(kernel(r) >= 0.0, "negative weight at r={r}");
            assert!(
                (kernel(r) - kernel(-r)).abs() < 1e-14,
                "asymmetric at r={r}"
            );
        }
        r += 0.01;
    }
}

#[test]
fn delta_kernel_values_at_the_origin() {
    // Roma: (1/3)(1 + 1) = 2/3.  Peskin 4-point: (1/8)(3 + 1) = 1/2.
    assert!((delta_phi(0.0) - 2.0 / 3.0).abs() < 1e-14);
    assert!((delta_peskin4(0.0) - 0.5).abs() < 1e-14);
}

#[test]
fn spreading_conserves_the_total_force_for_both_kernels() {
    let ds = 0.25;
    let force = [2.0, -1.5];
    let mut p = IbmParticle::with_rest_and_ds(7.3, 8.6, 7.3, 8.6, ds);
    p.force = force;

    let mut roma = IbmFluidGrid::new(16, 16);
    roma.spread_forces(std::slice::from_ref(&p));
    let t = roma.total_force();
    assert!((t[0] - force[0] * ds).abs() < 1e-12, "Roma Fx = {}", t[0]);
    assert!((t[1] - force[1] * ds).abs() < 1e-12, "Roma Fy = {}", t[1]);

    let mut peskin = IbmFluidGrid::new(16, 16);
    peskin.spread_forces_peskin4(std::slice::from_ref(&p));
    let t = peskin.total_force();
    assert!((t[0] - force[0] * ds).abs() < 1e-12, "Peskin Fx = {}", t[0]);
    assert!((t[1] - force[1] * ds).abs() < 1e-12, "Peskin Fy = {}", t[1]);
}

#[test]
fn legacy_spread_force_conserves_the_total_force() {
    let (nx, ny) = (16, 16);
    let mut marker = IbmMarker::new(6.77, 9.21);
    marker.force = [1.0, 0.5];
    let mut fx = vec![0.0; nx * ny];
    let mut fy = vec![0.0; nx * ny];
    spread_force(&[marker], &mut fx, &mut fy, nx, ny, 0.4);
    let sx: f64 = fx.iter().sum();
    let sy: f64 = fy.iter().sum();
    assert!((sx - 0.4).abs() < 1e-12, "Fx = {sx}");
    assert!((sy - 0.2).abs() < 1e-12, "Fy = {sy}");
}

// ---------------------------------------------------------------------------
// Velocity interpolation carries no Lagrangian quadrature weight (F2582)
// ---------------------------------------------------------------------------

#[test]
fn interpolation_reproduces_a_uniform_field_for_any_arc_length() {
    let mut grid = IbmFluidGrid::new(16, 16);
    grid.set_uniform_velocity(0.3, -0.2);
    let mut particles = vec![
        IbmParticle::with_rest_and_ds(7.3, 8.6, 7.3, 8.6, 0.25),
        IbmParticle::with_rest_and_ds(4.0, 4.0, 4.0, 4.0, 3.0),
        IbmParticle::with_rest_and_ds(10.75, 3.5, 10.75, 3.5, 0.01),
    ];
    grid.interpolate_velocities(&mut particles);
    for p in &particles {
        assert!(
            (p.velocity[0] - 0.3).abs() < 1e-12,
            "ux = {}",
            p.velocity[0]
        );
        assert!(
            (p.velocity[1] + 0.2).abs() < 1e-12,
            "uy = {}",
            p.velocity[1]
        );
    }
}

#[test]
fn interpolation_reproduces_a_linear_field() {
    let (nx, ny) = (16, 16);
    let mut grid = IbmFluidGrid::new(nx, ny);
    for iy in 0..ny {
        for ix in 0..nx {
            grid.ux[iy * nx + ix] = 0.2 + 0.01 * ix as f64;
            grid.uy[iy * nx + ix] = 0.1 - 0.02 * iy as f64;
        }
    }
    let mut particles = vec![IbmParticle::with_rest_and_ds(6.37, 9.11, 6.37, 9.11, 0.5)];
    grid.interpolate_velocities(&mut particles);
    assert!((particles[0].velocity[0] - (0.2 + 0.01 * 6.37)).abs() < 1e-12);
    assert!((particles[0].velocity[1] - (0.1 - 0.02 * 9.11)).abs() < 1e-12);
}

#[test]
fn legacy_interpolation_reproduces_a_uniform_field() {
    let (nx, ny) = (16, 16);
    let ux = vec![0.3; nx * ny];
    let uy = vec![-0.2; nx * ny];
    let mut markers = vec![IbmMarker::new(7.3, 8.6)];
    interpolate_velocity(&mut markers, &ux, &uy, nx, ny);
    assert!((markers[0].velocity[0] - 0.3).abs() < 1e-12);
    assert!((markers[0].velocity[1] + 0.2).abs() < 1e-12);
}

#[test]
fn legacy_step_advects_markers_with_the_fluid_velocity_regardless_of_ds() {
    let (nx, ny) = (16, 16);
    let ux = vec![0.1; nx * ny];
    let uy = vec![0.0; nx * ny];
    for ds in [0.25, 1.0, 4.0] {
        let mut markers = vec![IbmMarker::new(7.3, 8.6)];
        let mut fx = vec![0.0; nx * ny];
        let mut fy = vec![0.0; nx * ny];
        ib_lbm_step(
            &mut markers,
            &mut fx,
            &mut fy,
            &ux,
            &uy,
            nx,
            ny,
            ds,
            0.0,
            2.0,
        );
        assert!(
            (markers[0].position[0] - 7.5).abs() < 1e-12,
            "ds={ds}: x = {}",
            markers[0].position[0]
        );
        assert!((markers[0].position[1] - 8.6).abs() < 1e-12);
    }
}

#[test]
fn direct_forcing_uses_the_unweighted_fluid_velocity() {
    let markers = make_circle_markers(8.0, 8.0, 3.0, 12);
    let ds = markers[0].ds;
    assert!((ds - 2.0 * PI * 3.0 / 12.0).abs() < 1e-14);
    let dt = 0.5;
    let mut df = DirectForcingIbm::new(markers, dt, 2.0);
    let mut grid = IbmFluidGrid::new(16, 16);
    grid.set_uniform_velocity(0.05, 0.0);
    let targets = df.set_target_velocity([0.0, 0.0]);
    df.step(&mut grid, &targets);
    for p in &df.particles {
        assert!(
            (p.velocity[0] - 0.05).abs() < 1e-12,
            "u_fluid = {}",
            p.velocity[0]
        );
        // f = rho (u_target - u_fluid) / dt = 2 * (-0.05) / 0.5
        assert!((p.force[0] + 0.2).abs() < 1e-11, "force = {}", p.force[0]);
    }
}

// ---------------------------------------------------------------------------
// FlexibleIbmFiber bending (F2583)
// ---------------------------------------------------------------------------

/// Deterministic bumpy 9-marker fiber (rest spacing 1).
fn bumpy_fiber(k_tension: f64, k_bending: f64, kappa0: f64) -> FlexibleIbmFiber {
    let mut fiber = FlexibleIbmFiber::straight(0.0, 5.0, 9, 1.0, k_tension, k_bending);
    fiber.kappa0 = kappa0;
    let offsets: [[f64; 2]; 9] = [
        [0.0, 0.0],
        [0.05, 0.12],
        [-0.04, 0.21],
        [0.03, -0.1],
        [0.1, 0.18],
        [-0.07, -0.12],
        [0.02, 0.3],
        [-0.06, 0.05],
        [0.04, -0.02],
    ];
    for (p, o) in fiber.particles.iter_mut().zip(offsets.iter()) {
        p.position[0] += o[0];
        p.position[1] += o[1];
    }
    fiber
}

/// Central-difference derivative of the elastic energy with respect to one coordinate.
fn energy_gradient(fiber: &mut FlexibleIbmFiber, marker: usize, axis: usize, h: f64) -> f64 {
    let x0 = fiber.particles[marker].position[axis];
    fiber.particles[marker].position[axis] = x0 + h;
    let e_plus = fiber.elastic_energy();
    fiber.particles[marker].position[axis] = x0 - h;
    let e_minus = fiber.elastic_energy();
    fiber.particles[marker].position[axis] = x0;
    (e_plus - e_minus) / (2.0 * h)
}

#[test]
fn bumped_marker_is_pushed_back_toward_the_line() {
    let kb = 2.0;
    let h = 0.1;
    let k = 4;
    let mut fiber = FlexibleIbmFiber::straight(0.0, 5.0, 9, 1.0, 0.0, kb);
    fiber.particles[k].position[1] += h;
    fiber.compute_elastic_forces();
    // 5-point stencil of the bump: -6, +4, -1 times (k_b/ds0^3) * h.
    let c = kb * h;
    let expected = [(k, -6.0 * c), (k - 1, 4.0 * c), (k + 1, 4.0 * c)];
    for (i, fy) in expected {
        assert!(
            (fiber.particles[i].force[1] - fy).abs() < 1e-12,
            "marker {i}: fy = {}, expected {fy}",
            fiber.particles[i].force[1]
        );
    }
    assert!((fiber.particles[k - 2].force[1] + c).abs() < 1e-12);
    assert!((fiber.particles[k + 2].force[1] + c).abs() < 1e-12);
    assert!(
        fiber.particles[k].force[1] < 0.0,
        "bumped marker must be pulled back"
    );
    for p in &fiber.particles {
        assert!(
            p.force[0].abs() < 1e-14,
            "no x force for a purely vertical bump"
        );
    }
}

#[test]
fn bending_energy_of_a_bent_fiber_matches_the_hand_value() {
    // x0=(0,0), x1=(1,0.5), x2=(2,0): D_1=(0,-1), kappa_1=1, ds0=1, kb=2.
    let mut fiber = FlexibleIbmFiber::straight(0.0, 0.0, 3, 1.0, 0.0, 2.0);
    fiber.particles[1].position[1] = 0.5;
    assert!((fiber.bending_energy() - 1.0).abs() < 1e-14); // 1/2 * 2 * 1^2
    fiber.kappa0 = 0.4;
    // 1/2 * kb * (kappa - kappa0)^2 * ds0 = 0.36, not (kappa^2 - kappa0^2) * ...
    assert!((fiber.bending_energy() - 0.36).abs() < 1e-14);
    assert!((fiber.elastic_energy() - 0.36).abs() < 1e-14);
    fiber.compute_elastic_forces();
    // G = kb (D - kappa0 D_hat) = 2 * (0,-0.6); F_0 = F_2 = -G, F_1 = 2 G.
    let expected = [[0.0, 1.2], [0.0, -2.4], [0.0, 1.2]];
    for (p, e) in fiber.particles.iter().zip(expected.iter()) {
        assert!((p.force[0] - e[0]).abs() < 1e-12, "fx = {}", p.force[0]);
        assert!((p.force[1] - e[1]).abs() < 1e-12, "fy = {}", p.force[1]);
    }
}

#[test]
fn elastic_forces_are_the_negative_energy_gradient() {
    let h = 1e-5;
    for &(k_tension, k_bending, kappa0) in &[
        (0.0, 2.0, 0.0),
        (50.0, 2.0, 0.0),
        (50.0, 2.0, 0.35),
        (10.0, 0.5, 1.2),
    ] {
        let mut fiber = bumpy_fiber(k_tension, k_bending, kappa0);
        fiber.compute_elastic_forces();
        let forces: Vec<[f64; 2]> = fiber.particles.iter().map(|p| p.force).collect();
        for (marker, force) in forces.iter().enumerate() {
            for (axis, &f) in force.iter().enumerate() {
                let grad = energy_gradient(&mut fiber, marker, axis, h);
                assert!(
                    (f + grad).abs() < 1e-6,
                    "kt={k_tension} kb={k_bending} k0={kappa0} marker={marker} axis={axis}: \
                     force {f} vs -grad {}",
                    -grad
                );
            }
        }
    }
}

#[test]
fn elastic_forces_carry_no_net_force_or_torque() {
    for &(k_tension, kappa0) in &[(0.0, 0.0), (80.0, 0.0), (80.0, 0.3)] {
        let mut fiber = bumpy_fiber(k_tension, 3.0, kappa0);
        fiber.compute_elastic_forces();
        let mut total = [0.0_f64; 2];
        let mut torque = 0.0_f64;
        for p in &fiber.particles {
            total[0] += p.force[0];
            total[1] += p.force[1];
            torque += p.position[0] * p.force[1] - p.position[1] * p.force[0];
        }
        assert!(
            total[0].abs() < 1e-11 && total[1].abs() < 1e-11,
            "net force {total:?}"
        );
        assert!(torque.abs() < 1e-10, "net torque {torque}");
    }
}

#[test]
fn bending_energy_is_minimal_for_the_straight_fiber() {
    let straight = FlexibleIbmFiber::straight(0.0, 5.0, 9, 1.0, 100.0, 2.0);
    assert_eq!(straight.elastic_energy(), 0.0);
    let bent = bumpy_fiber(100.0, 2.0, 0.0);
    assert!(bent.elastic_energy() > 0.0);
    assert!(bent.bending_energy() > 0.0);
}

#[test]
fn straight_fiber_with_natural_curvature_has_finite_zero_bending_force() {
    let mut fiber = FlexibleIbmFiber::straight(0.0, 5.0, 6, 1.0, 100.0, 2.0);
    fiber.kappa0 = 0.5;
    fiber.compute_elastic_forces();
    for p in &fiber.particles {
        assert!(p.force[0].is_finite() && p.force[1].is_finite());
        assert!(p.force[0].abs() < 1e-14 && p.force[1].abs() < 1e-14);
    }
}

#[test]
fn bent_fiber_relaxes_toward_the_straight_line_under_its_own_forces() {
    // Gradient flow with the elastic force must reduce the energy monotonically.
    let mut fiber = bumpy_fiber(40.0, 2.0, 0.0);
    let mut energy = fiber.elastic_energy();
    let first = energy;
    for _ in 0..200 {
        fiber.compute_elastic_forces();
        for p in fiber.particles.iter_mut() {
            p.position[0] += 0.002 * p.force[0];
            p.position[1] += 0.002 * p.force[1];
        }
        let next = fiber.elastic_energy();
        assert!(
            next <= energy + 1e-14,
            "energy rose from {energy} to {next}"
        );
        energy = next;
    }
    assert!(
        energy < 0.5 * first,
        "energy {energy} did not decay from {first}"
    );
}
