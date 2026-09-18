// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use super::*;

const RHO_F: f64 = 1000.0;
const RHO_S: f64 = 2650.0;

/// Fill a closed 2-D (x–z) box `[0, length] × [0, depth]` with fluid particles
/// on a square lattice of spacing `dx`, surrounded by three layers of wall
/// particles. `conc(x, z)` gives the initial concentration.
fn closed_box_2d(
    sim: &mut TurbidityCurrentSph,
    length: f64,
    depth: f64,
    dx: f64,
    h: f64,
    conc: impl Fn(f64, f64) -> f64,
) {
    let mass = sim.rho_fluid * dx * dx;
    let nx = (length / dx).round() as i64;
    let nz = (depth / dx).round() as i64;
    for i in 0..nx {
        for k in 0..nz {
            let x = (i as f64 + 0.5) * dx;
            let z = (k as f64 + 0.5) * dx;
            sim.add_particle(TurbidityParticle::new(
                [x, 0.0, z],
                [0.0; 3],
                conc(x, z),
                mass,
                h,
            ));
        }
    }
    let layers = 3;
    for i in -layers..(nx + layers) {
        let x = (i as f64 + 0.5) * dx;
        for l in 0..layers {
            let off = (l as f64 + 0.5) * dx;
            sim.add_wall_particle(TurbidityWallParticle::new(
                [x, 0.0, -off],
                [0.0, 0.0, 1.0],
                mass,
                h,
            ));
            sim.add_wall_particle(TurbidityWallParticle::new(
                [x, 0.0, depth + off],
                [0.0, 0.0, -1.0],
                mass,
                h,
            ));
        }
    }
    for k in 0..nz {
        let z = (k as f64 + 0.5) * dx;
        for l in 0..layers {
            let off = (l as f64 + 0.5) * dx;
            sim.add_wall_particle(TurbidityWallParticle::new(
                [-off, 0.0, z],
                [1.0, 0.0, 0.0],
                mass,
                h,
            ));
            sim.add_wall_particle(TurbidityWallParticle::new(
                [length + off, 0.0, z],
                [-1.0, 0.0, 0.0],
                mass,
                h,
            ));
        }
    }
}

/// Concentration giving reduced gravity `g′` for the test densities.
fn conc_for_g_prime(g_prime: f64) -> f64 {
    g_prime * RHO_F / (GRAVITY * (RHO_S - RHO_F))
}

/// Least-squares slope of `ys` against `xs`.
fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    sxy / sxx
}

// --- TurbidityParticle -------------------------------------------------------

#[test]
fn test_turbidity_bulk_density() {
    let p = TurbidityParticle::new([0.0; 3], [0.0; 3], 0.1, 1.0, 0.01);
    let rho = p.bulk_density(RHO_F, RHO_S);
    // 0.9 * 1000 + 0.1 * 2650 = 900 + 265 = 1165
    assert!((rho - 1165.0).abs() < 1e-9, "rho={rho}");
}

#[test]
fn test_turbidity_settle_reduces_concentration() {
    let mut p = TurbidityParticle::new([0.0; 3], [0.0; 3], 0.5, 1.0, 0.1);
    p.settle(0.01, 1.0);
    // c = 0.5 exp(-0.01 * 1 / 0.1)
    let expected = 0.5 * (-0.1_f64).exp();
    assert!((p.concentration - expected).abs() < 1e-12);
}

#[test]
fn test_turbidity_settle_scales_inversely_with_h() {
    // The settling loss rate is w_s c / h: halving h doubles the exponent.
    let mut small = TurbidityParticle::new([0.0; 3], [0.0; 3], 0.4, 1.0, 0.05);
    let mut large = TurbidityParticle::new([0.0; 3], [0.0; 3], 0.4, 1.0, 0.1);
    small.settle(0.002, 1.0);
    large.settle(0.002, 1.0);
    let k_small = -(small.concentration / 0.4).ln();
    let k_large = -(large.concentration / 0.4).ln();
    assert!((k_small / k_large - 2.0).abs() < 1e-9);
}

#[test]
fn test_turbidity_concentration_non_negative() {
    let mut p = TurbidityParticle::new([0.0; 3], [0.0; 3], 1e-5, 1.0, 0.01);
    p.settle(100.0, 1000.0);
    assert!(p.concentration >= 0.0);
}

#[test]
fn test_turbidity_particle_drag_has_acceleration_units() {
    // Neutrally buoyant particle: only drag acts, dv = -dt cd |v| v / h.
    let mut p = TurbidityParticle::new([0.0; 3], [2.0, 0.0, 0.0], 0.0, 1.0, 0.5);
    p.step(0.01, RHO_F, RHO_S, 0.1);
    let expected = 2.0 - 0.01 * 0.1 * 2.0 * 2.0 / 0.5;
    assert!(
        (p.velocity[0] - expected).abs() < 1e-12,
        "{}",
        p.velocity[0]
    );
    assert!(p.velocity[2].abs() < 1e-15);
}

// --- TurbidityCurrentSph -----------------------------------------------------

#[test]
fn test_turbidity_sph_add_prune() {
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S);
    tc.add_particle(TurbidityParticle::new([0.0; 3], [0.0; 3], 1e-8, 1.0, 0.01));
    tc.add_particle(TurbidityParticle::new([0.0; 3], [0.0; 3], 0.5, 1.0, 0.01));
    tc.prune(1e-6);
    assert_eq!(tc.num_particles(), 1);
}

#[test]
fn test_turbidity_sph_isolated_particle_advects_and_sinks() {
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S);
    tc.add_particle(TurbidityParticle::new(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        0.2,
        1.0,
        0.1,
    ));
    tc.step(0.01, 0.0, 0.0).expect("valid step");
    let p = &tc.particles[0];
    let g_r = GRAVITY * 0.2 * (RHO_S - RHO_F) / RHO_F;
    assert!((p.position[0] - 0.01).abs() < 1e-12);
    assert!((p.velocity[2] + 0.01 * g_r).abs() < 1e-12);
}

#[test]
fn test_turbidity_sph_rejects_invalid_parameters() {
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S);
    tc.add_particle(TurbidityParticle::new([0.0; 3], [0.0; 3], 0.1, 1.0, 0.1));
    assert!(tc.step(0.0, 0.0, 0.0).is_err());
    assert!(tc.step(f64::NAN, 0.0, 0.0).is_err());
    assert!(tc.step(1e-3, -1.0, 0.0).is_err());
    assert!(tc.step(1e-3, 0.0, -0.1).is_err());
    tc.particles[0].h = 0.0;
    assert!(matches!(
        tc.step(1e-3, 0.0, 0.0),
        Err(Error::NonPositiveSmoothingLength { .. })
    ));
    // State untouched by rejected steps.
    assert_eq!(tc.time(), 0.0);
    assert_eq!(tc.particles[0].position, [0.0; 3]);
}

#[test]
fn test_turbidity_sph_density_summation_on_lattice() {
    let dx = 0.05;
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0);
    closed_box_2d(&mut tc, 1.0, 1.0, dx, 1.3 * dx, |_, _| 0.0);
    tc.step(1e-4, 0.0, 0.0).expect("valid step");
    // Every particle (interior and wall-adjacent) has full kernel support.
    for &rho in tc.densities() {
        assert!((rho / RHO_F - 1.0).abs() < 0.02, "rho = {rho}");
    }
}

#[test]
fn test_turbidity_sph_pair_forces_conserve_momentum() {
    // No walls, no buoyancy (rho_s = rho_f): pressure and viscosity forces
    // are antisymmetric, so total momentum is conserved exactly.
    let dx = 0.1;
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_F)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0)
        .with_artificial_viscosity(0.1);
    let mass = RHO_F * dx * dx;
    for i in 0..8 {
        for k in 0..8 {
            let x = i as f64 * dx * (1.0 + 0.03 * ((i * 7 + k * 3) % 5) as f64);
            let z = k as f64 * dx;
            let v = [
                0.3 * ((i + 2 * k) % 3) as f64 - 0.3,
                0.0,
                0.2 * ((3 * i + k) % 4) as f64 - 0.3,
            ];
            tc.add_particle(TurbidityParticle::new([x, 0.0, z], v, 0.0, mass, 1.3 * dx));
        }
    }
    let v_init: Vec<[f64; 3]> = tc.particles.iter().map(|p| p.velocity).collect();
    let m0 = tc.total_momentum();
    for _ in 0..50 {
        tc.step(5e-4, 0.0, 0.0).expect("valid step");
    }
    let m1 = tc.total_momentum();
    let scale: f64 = tc.particles.iter().map(|p| p.mass).sum::<f64>();
    for k in 0..3 {
        assert!(
            (m1[k] - m0[k]).abs() < 1e-9 * scale,
            "k={k}: {m0:?} -> {m1:?}"
        );
    }
    // ... and the particles did interact (velocities changed).
    let max_dv = tc
        .particles
        .iter()
        .zip(&v_init)
        .map(|(p, v0)| norm(sub(p.velocity, *v0)))
        .fold(0.0, f64::max);
    assert!(max_dv > 1e-3, "no interaction: max |dv| = {max_dv}");
}

#[test]
fn test_turbidity_sph_diffusion_conserves_and_smooths() {
    let dx = 0.05;
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_F)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0)
        .with_sediment_diffusivity(1e-3);
    closed_box_2d(&mut tc, 0.5, 0.5, dx, 1.3 * dx, |x, z| {
        if (x - 0.25).abs() < 0.06 && (z - 0.25).abs() < 0.06 {
            0.2
        } else {
            0.0
        }
    });
    let v0 = tc.suspended_sediment_volume();
    let max0 = tc
        .particles
        .iter()
        .map(|p| p.concentration)
        .fold(0.0, f64::max);
    let dt = tc.max_stable_dt().min(1e-3);
    for _ in 0..200 {
        tc.step(dt, 0.0, 0.0).expect("valid step");
    }
    let v1 = tc.suspended_sediment_volume();
    assert!((v1 - v0).abs() < 1e-12 * v0.max(1e-30), "{v0} -> {v1}");
    let max1 = tc
        .particles
        .iter()
        .map(|p| p.concentration)
        .fold(0.0, f64::max);
    assert!(max1 < 0.9 * max0, "peak not flattened: {max0} -> {max1}");
    assert!(tc.particles.iter().all(|p| p.concentration >= 0.0));
    // Sediment spread into initially clear particles next to the patch.
    let spread = tc
        .particles
        .iter()
        .filter(|p| (p.position[0] - 0.25).abs() > 0.08 && p.concentration > 1e-4)
        .count();
    assert!(spread > 0);
}

#[test]
fn test_turbidity_sph_settling_conserves_sediment_and_deposits() {
    // Neutrally buoyant mixture (rho_s = rho_f) isolates the settling flux.
    let dx = 0.05;
    let depth = 0.5;
    let c_init = 0.05;
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_F)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0);
    closed_box_2d(&mut tc, 0.5, depth, dx, 1.3 * dx, |_, _| c_init);
    let v0 = tc.suspended_sediment_volume();
    let w_s = 0.01;
    for _ in 0..500 {
        tc.step(1e-3, 0.0, w_s).expect("valid step");
    }
    let suspended = tc.suspended_sediment_volume();
    let deposited = tc.deposited_sediment_volume();
    assert!(deposited > 0.0);
    assert!(
        ((suspended + deposited) - v0).abs() < 1e-10 * v0,
        "budget: {v0} vs {suspended} + {deposited}"
    );
    // Deposition rate onto the floor ≈ w_s c L (2-D, per unit width): after
    // t = 0.5 s roughly w_s c L t of sediment has left the suspension.
    let expected = w_s * c_init * 0.5 * 0.5;
    assert!(
        (deposited / expected - 1.0).abs() < 0.3,
        "deposited {deposited}, expected ≈ {expected}"
    );
    // A clear-water layer forms beneath the lid, sediment stays near the bed.
    let mean_c = |lo: f64, hi: f64| {
        let sel: Vec<f64> = tc
            .particles
            .iter()
            .filter(|p| p.position[2] >= lo && p.position[2] < hi)
            .map(|p| p.concentration)
            .collect();
        sel.iter().sum::<f64>() / sel.len() as f64
    };
    let top = mean_c(depth - dx, depth + dx);
    let mid = mean_c(0.2, 0.3);
    assert!(top < mid, "top {top} should be clearer than mid {mid}");
}

#[test]
fn test_turbidity_sph_uniform_suspension_stays_near_rest() {
    // Uniform concentration: buoyancy is balanced by a hydrostatic pressure
    // gradient supplied by the fluid and the Adami walls.
    let dx = 0.05;
    let depth = 0.5;
    let g_prime = 1.0;
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0)
        .with_artificial_viscosity(0.05);
    closed_box_2d(&mut tc, 0.5, depth, dx, 1.3 * dx, |_, _| {
        conc_for_g_prime(g_prime)
    });
    let dt = 0.2 * 1.3 * dx / 10.0;
    // Run for ~13 acoustic periods 4H/c₀ so the initial pressure transient
    // (fluid starts at uniform pressure) has decayed.
    for _ in 0..2000 {
        tc.step(dt, 0.0, 0.0).expect("valid step");
    }
    let u_ref = (g_prime * depth).sqrt();
    let v_max = tc
        .particles
        .iter()
        .map(|p| norm(p.velocity))
        .fold(0.0, f64::max);
    assert!(v_max < 0.02 * u_ref, "spurious flow {v_max}");
    for p in &tc.particles {
        assert!(
            p.position[2] > 0.0 && p.position[2] < depth,
            "{:?}",
            p.position
        );
        assert!(
            p.position[0] > 0.0 && p.position[0] < 0.5,
            "{:?}",
            p.position
        );
    }
    // Hydrostatic wall pressure: the floor carries ≈ ρ₀ g′ H more than the lid.
    let mut floor = Vec::new();
    let mut lid = Vec::new();
    for (w, &p) in tc.walls.iter().zip(tc.wall_pressures()) {
        if (w.position[2] + 0.5 * dx).abs() < 1e-9 && w.position[0] > 0.1 && w.position[0] < 0.4 {
            floor.push(p);
        }
        if (w.position[2] - depth - 0.5 * dx).abs() < 1e-9
            && w.position[0] > 0.1
            && w.position[0] < 0.4
        {
            lid.push(p);
        }
    }
    let p_floor = floor.iter().sum::<f64>() / floor.len() as f64;
    let p_lid = lid.iter().sum::<f64>() / lid.len() as f64;
    let expected = RHO_F * g_prime * (depth + dx);
    assert!(
        ((p_floor - p_lid) / expected - 1.0).abs() < 0.15,
        "Δp = {}, expected ≈ {expected}",
        p_floor - p_lid
    );
}

#[test]
fn test_turbidity_sph_lock_exchange_front_speed() {
    // Full-depth Boussinesq lock exchange in a closed 2-D tank: the heavy
    // front advances at ≈ 0.5 √(g′H) (Benjamin 1968; Shin et al. 2004).
    let dx = 0.05;
    let depth = 1.0;
    let length = 3.0;
    let x_lock = 1.5;
    let g_prime = 1.0;
    let c_heavy = conc_for_g_prime(g_prime);
    let mut tc = TurbidityCurrentSph::new(RHO_F, RHO_S)
        .with_dimension(TurbidityDimension::Two)
        .with_sound_speed(10.0)
        .with_artificial_viscosity(0.02);
    closed_box_2d(&mut tc, length, depth, dx, 1.3 * dx, |x, _| {
        if x < x_lock { c_heavy } else { 0.0 }
    });
    let dt = 0.2 * 1.3 * dx / 10.0;
    let mut ts = Vec::new();
    let mut xs = Vec::new();
    let t_end = 2.0;
    let mut n = 0usize;
    while tc.time() < t_end {
        tc.step(dt, 0.0, 0.0).expect("valid step");
        n += 1;
        if tc.time() >= 0.6 && n.is_multiple_of(20) {
            let front = tc
                .front_position(0.5 * c_heavy, 0.5 * depth)
                .expect("heavy fluid present");
            ts.push(tc.time());
            xs.push(front);
        }
    }
    let u_front = slope(&ts, &xs);
    let fr = u_front / (g_prime * depth).sqrt();
    assert!(
        (fr - 0.5).abs() < 0.1,
        "front Froude number {fr} (speed {u_front} m/s)"
    );
    // Heavy fluid stays in suspension (no settling) and buoyancy has been
    // converted into motion.
    let v_final = tc.suspended_sediment_volume();
    assert!(v_final > 0.0);
}

#[test]
fn test_turbidity_kernel_normalisation_2d_and_3d() {
    // Σ W(r) Δx^d over a fine lattice ≈ 1 for both dimensions.
    let h: f64 = 1.0;
    let dx: f64 = 0.05;
    let n = (2.0 * h / dx).ceil() as i64 + 1;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    for i in -n..=n {
        for k in -n..=n {
            let (x, z) = (i as f64 * dx, k as f64 * dx);
            sum2 += TurbidityDimension::Two.w((x * x + z * z).sqrt(), h) * dx * dx;
            for j in -n..=n {
                let y = j as f64 * dx;
                sum3 +=
                    TurbidityDimension::Three.w((x * x + y * y + z * z).sqrt(), h) * dx * dx * dx;
            }
        }
    }
    assert!((sum2 - 1.0).abs() < 0.01, "2-D: {sum2}");
    assert!((sum3 - 1.0).abs() < 0.01, "3-D: {sum3}");
    // dW/dr is the derivative of W (central difference check).
    for dim in [TurbidityDimension::Two, TurbidityDimension::Three] {
        for &r in &[0.3, 0.9, 1.5] {
            let e = 1e-6;
            let fd = (dim.w(r + e, h) - dim.w(r - e, h)) / (2.0 * e);
            assert!((fd - dim.dw_dr(r, h)).abs() < 1e-6, "{dim:?} r={r}");
        }
    }
}
