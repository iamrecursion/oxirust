//! F4178: the P2G elastic force uses the Kirchhoff stress `P Fᵀ`.

use super::*;

/// A lone particle in the interior of a 16³ grid, with `f` as both the total and
/// the elastic deformation gradient.
fn lone_particle(material: MaterialType, f: [[f64; 3]; 3]) -> MpmParticle {
    let mut p = MpmParticle::new(
        [0.83, 0.77, 0.81],
        [0.0; 3],
        1.0,
        VOLUME0,
        material,
        1.0e5,
        2.0e5,
    );
    p.deformation_gradient = f;
    p.fe = f;
    p
}

/// Runs P2G for `p` alone, without gravity, and returns the grid.
fn p2g_of(p: &MpmParticle) -> MpmGrid {
    let mut grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    p2g_transfer(&mut grid, std::slice::from_ref(p), 1.0e-4, [0.0; 3]);
    grid
}

/// Moments of the nodal forces about the particle position.
struct ForceMoments {
    net_force: [f64; 3],
    torque: [f64; 3],
    /// `sum |f_i|`, the scale against which the net force is judged.
    force_scale: f64,
    /// `sum |r_i| |f_i|`, the scale against which the torque is judged.
    torque_scale: f64,
    /// `sum f_i . r_i`.
    virial: f64,
}

fn force_moments(grid: &MpmGrid, x_p: [f64; 3]) -> ForceMoments {
    let [nx, ny, nz] = grid.dims;
    let mut m = ForceMoments {
        net_force: [0.0; 3],
        torque: [0.0; 3],
        force_scale: 0.0,
        torque_scale: 0.0,
        virial: 0.0,
    };
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let force = grid.nodes[grid.idx(ix, iy, iz)].force;
                let r = sub3(grid.node_pos(ix, iy, iz), x_p);
                m.net_force = add3(m.net_force, force);
                m.torque = add3(m.torque, cross3(r, force));
                m.force_scale += len3(force);
                m.torque_scale += len3(r) * len3(force);
                m.virial += dot3(force, r);
            }
        }
    }
    m
}

/// `F = R(30 deg about z) · diag(1.1, 1, 1)`: stretched *and* rotated.
fn rotated_stretched_gradient() -> [[f64; 3]; 3] {
    mat3_mul(
        rot_z(std::f64::consts::FRAC_PI_6),
        mat3_diag([1.1, 1.0, 1.0]),
    )
}

#[test]
fn f4178_neo_hookean_force_matches_the_closed_form_kirchhoff_stress() {
    let (mu, lambda) = (1.0e5, 2.0e5);
    let stretch = 1.1_f64;
    let rotation = rot_z(std::f64::consts::FRAC_PI_6);
    let f = rotated_stretched_gradient();
    let p = lone_particle(MaterialType::NeoHookean, f);

    // P is not symmetric for this F, so P and P Fᵀ really differ.
    let pk1 = neo_hookean_pk1(f, mu, lambda);
    assert!(
        symmetry_error(pk1) > 0.1 * frobenius(pk1),
        "test would pass vacuously: P is (nearly) symmetric"
    );

    // tau = mu (F Fᵀ - 1) + lambda ln(J) 1 = R diag(mu (s² - 1) + lambda ln s, ...) Rᵀ.
    let ln_j = stretch.ln();
    let tau_exact = rotate_diagonal(
        rotation,
        [
            mu * (stretch * stretch - 1.0) + lambda * ln_j,
            lambda * ln_j,
            lambda * ln_j,
        ],
    );
    let tau = particle_kirchhoff_stress(&p);
    assert!(
        max_diff(tau, tau_exact) < 1e-9 * frobenius(tau_exact),
        "tau = {tau:?}"
    );
    assert!(symmetry_error(tau) < 1e-9 * frobenius(tau));
    assert!(max_diff(tau, pk1_to_kirchhoff(pk1, f)) < 1e-9 * frobenius(tau));

    // Nodal force f_i = -V0 tau (4/h²) w_i (x_i - x_p).
    let grid = p2g_of(&p);
    let (base, frac) = grid.base_and_frac(p.position);
    let (w, _) = compute_weights(base, frac, GRID_H);
    let mut checked = 0;
    for di in 0..3 {
        for dj in 0..3 {
            for dk in 0..3 {
                let (ix, iy, iz) = (
                    (base[0] + di as i64) as usize,
                    (base[1] + dj as i64) as usize,
                    (base[2] + dk as i64) as usize,
                );
                let r = sub3(grid.node_pos(ix, iy, iz), p.position);
                let weight = w[0][di] * w[1][dj] * w[2][dk];
                let expected = scale3(
                    mat3_vec(tau_exact, r),
                    -VOLUME0 * 4.0 / (GRID_H * GRID_H) * weight,
                );
                let force = grid.nodes[grid.idx(ix, iy, iz)].force;
                assert!(
                    vec_diff(force, expected) < 1e-9 * (1.0 + len3(expected)),
                    "node ({ix},{iy},{iz}): force {force:?}, expected {expected:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 27);
}

#[test]
fn f4178_snow_stress_matches_the_closed_form_fixed_corotated_kirchhoff_stress() {
    // tau = 2 mu (F - R) Fᵀ + lambda J (J - 1) 1 = R diag(2 mu (s - 1) s + lambda J (J - 1), ...) Rᵀ
    // with the hardened Lame parameters mu_e = mu0 exp(xi (1 - Jp)).
    let rotation = rot_z(std::f64::consts::FRAC_PI_6);
    let stretch = 1.1_f64;
    let f = rotated_stretched_gradient();
    for plastic_jacobian in [1.0_f64, 0.9] {
        let mut p = lone_particle(MaterialType::Snow, f);
        p.fp = mat3_diag([plastic_jacobian, 1.0, 1.0]);
        let hardening = (p.hardening * (1.0 - plastic_jacobian)).exp();
        let (mu, lambda) = (p.mu0 * hardening, p.lambda0 * hardening);
        let j = stretch;
        let volumetric = lambda * j * (j - 1.0);
        let tau_exact = rotate_diagonal(
            rotation,
            [
                2.0 * mu * (stretch - 1.0) * stretch + volumetric,
                volumetric,
                volumetric,
            ],
        );
        let tau = particle_kirchhoff_stress(&p);
        assert!(
            max_diff(tau, tau_exact) < 1e-9 * frobenius(tau_exact),
            "Jp = {plastic_jacobian}: tau = {tau:?}, expected {tau_exact:?}"
        );
        assert!(symmetry_error(tau) < 1e-9 * frobenius(tau));
    }
}

#[test]
fn f4178_nodal_forces_of_a_rotated_stretched_particle_exert_no_net_force_or_torque() {
    let f = rotated_stretched_gradient();
    for material in [
        MaterialType::NeoHookean,
        MaterialType::VonMises,
        MaterialType::Snow,
        MaterialType::Sand,
    ] {
        let p = lone_particle(material, f);
        let grid = p2g_of(&p);
        let m = force_moments(&grid, p.position);
        assert!(
            m.force_scale > 1.0,
            "{material:?}: particle exerts no force"
        );
        for k in 0..3 {
            assert!(
                m.net_force[k].abs() < 1e-10 * m.force_scale,
                "{material:?}: net force {:?}",
                m.net_force
            );
            assert!(
                m.torque[k].abs() < 1e-10 * m.torque_scale,
                "{material:?}: net torque {:?} (scale {})",
                m.torque,
                m.torque_scale
            );
        }
        // Absolute criterion as well: the torque vanishes to rounding, not just
        // relative to the (large) force moments.
        assert!(
            len3(m.torque) < 1e-10,
            "{material:?}: net torque {:?}",
            m.torque
        );
    }
}

#[test]
fn f4178_pk1_force_would_exert_a_torque() {
    // Documents why the Fᵀ factor matters: feeding P instead of tau into the same
    // stencil gives a large spurious torque for this F.
    let f = rotated_stretched_gradient();
    let pk1 = neo_hookean_pk1(f, 1.0e5, 2.0e5);
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let x_p = [0.83, 0.77, 0.81];
    let (base, frac) = grid.base_and_frac(x_p);
    let (w, _) = compute_weights(base, frac, GRID_H);
    let mut torque = [0.0f64; 3];
    let mut scale = 0.0f64;
    for di in 0..3 {
        for dj in 0..3 {
            for dk in 0..3 {
                let node = grid.node_pos(
                    (base[0] + di as i64) as usize,
                    (base[1] + dj as i64) as usize,
                    (base[2] + dk as i64) as usize,
                );
                let r = sub3(node, x_p);
                let weight = w[0][di] * w[1][dj] * w[2][dk];
                let force = scale3(
                    mat3_vec(pk1, r),
                    -VOLUME0 * 4.0 / (GRID_H * GRID_H) * weight,
                );
                torque = add3(torque, cross3(r, force));
                scale += len3(r) * len3(force);
            }
        }
    }
    assert!(
        len3(torque) > 0.05 * scale,
        "P torque {torque:?} vs scale {scale}"
    );
}

#[test]
fn f4178_each_material_reads_the_deformation_gradient_it_is_defined_on() {
    let stretched = rotated_stretched_gradient();
    let identity = mat3_identity();
    for material in [
        MaterialType::NeoHookean,
        MaterialType::VonMises,
        MaterialType::Snow,
        MaterialType::Sand,
    ] {
        // NeoHookean is a function of the total gradient F, the others of Fe.
        let reads_total = material == MaterialType::NeoHookean;
        let mut p = lone_particle(material, identity);
        if reads_total {
            p.deformation_gradient = stretched;
        } else {
            p.fe = stretched;
        }
        let stressed = frobenius(particle_kirchhoff_stress(&p));
        assert!(
            stressed > 1.0e3,
            "{material:?}: no stress from the gradient it reads"
        );

        let mut q = lone_particle(material, stretched);
        if reads_total {
            q.deformation_gradient = identity;
        } else {
            q.fe = identity;
        }
        let relaxed = frobenius(particle_kirchhoff_stress(&q));
        assert!(
            relaxed < 1.0e-6,
            "{material:?}: stress {relaxed} from the wrong gradient"
        );
    }
    let rigid = lone_particle(MaterialType::Rigid, stretched);
    assert_eq!(particle_kirchhoff_stress(&rigid), mat3_zero());
}

#[test]
fn f4178_compressed_fluid_pushes_the_nodes_outward() {
    let lambda = 2.0e4;
    for (j, outward) in [(0.9_f64, true), (1.1, false), (1.0, true)] {
        let mut p = lone_particle(MaterialType::Fluid, mat3_diag([j, 1.0, 1.0]));
        p.lambda0 = lambda;
        // tau = lambda J (J - 1) 1: negative (compressive) for J < 1.
        let tau = particle_kirchhoff_stress(&p);
        let expected = lambda * j * (j - 1.0);
        assert!(max_diff(tau, mat3_scale(mat3_identity(), expected)) < 1e-9);

        let grid = p2g_of(&p);
        let m = force_moments(&grid, p.position);
        // sum f_i . (x_i - x_p) = -V0 tau (4/h²) sum w |r|² = -3 V0 tau.
        let expected_virial = -3.0 * VOLUME0 * expected;
        assert!(
            (m.virial - expected_virial).abs() < 1e-9 * (1.0 + expected_virial.abs()),
            "J = {j}: virial {} vs {expected_virial}",
            m.virial
        );
        if j == 1.0 {
            assert!(m.force_scale < 1e-9);
        } else {
            assert_eq!(
                m.virial > 0.0,
                outward,
                "J = {j}: wrong sign of the fluid force"
            );
        }
        for k in 0..3 {
            assert!(m.net_force[k].abs() < 1e-10 * (1.0 + m.force_scale));
        }
    }
}

fn total_momentum(particles: &[MpmParticle]) -> [f64; 3] {
    particles
        .iter()
        .fold([0.0f64; 3], |acc, p| add3(acc, scale3(p.velocity, p.mass)))
}

#[test]
fn f4178_mpm_step_conserves_linear_momentum_for_every_material() {
    for material in [
        MaterialType::NeoHookean,
        MaterialType::VonMises,
        MaterialType::Snow,
        MaterialType::Sand,
        MaterialType::Fluid,
    ] {
        let mut grid = MpmGrid::new([20, 20, 20], GRID_H, [0.0; 3]);
        let mut particles = init_particle_block(
            [0.8, 0.8, 0.8],
            [1.1, 1.1, 1.1],
            0.05,
            0.05,
            material,
            5.0e4,
            1.0e5,
        );
        let centre = [0.95; 3];
        for p in &mut particles {
            // A rotation plus a shear, so that the stress is non-trivial.
            let r = sub3(p.position, centre);
            p.velocity = add3(
                [0.3, -0.1, 0.2],
                add3(cross3([0.0, 0.0, 2.0], r), scale3(r, 0.5)),
            );
            p.yield_param = if material == MaterialType::Sand {
                30.0
            } else {
                1.0e4
            };
        }
        let scale: f64 = particles.iter().map(|p| p.mass * len3(p.velocity)).sum();
        let before = total_momentum(&particles);
        let mass_before: f64 = particles.iter().map(|p| p.mass).sum();
        for _ in 0..20 {
            mpm_step(
                &mut grid,
                &mut particles,
                2.0e-4,
                [0.0; 3],
                THETA_C,
                THETA_S,
                2,
            );
        }
        let after = total_momentum(&particles);
        let mass_after: f64 = particles.iter().map(|p| p.mass).sum();
        assert_eq!(mass_before, mass_after);
        assert!(
            vec_diff(before, after) < 1e-9 * scale,
            "{material:?}: momentum {before:?} -> {after:?}"
        );
        assert!(
            particles.iter().all(|p| p
                .position
                .iter()
                .chain(p.velocity.iter())
                .all(|x| x.is_finite())),
            "{material:?}: non-finite state"
        );
        assert!(
            particles.iter().all(|p| p
                .deformation_gradient
                .iter()
                .flatten()
                .all(|x| x.is_finite())),
            "{material:?}: non-finite deformation gradient"
        );
    }
}

#[test]
fn f4178_neo_hookean_block_at_rest_stays_at_rest_and_a_stretched_block_relaxes() {
    // An unstrained block with zero velocity feels no elastic force at all.
    let mut grid = MpmGrid::new([20, 20, 20], GRID_H, [0.0; 3]);
    let mut at_rest = init_particle_block(
        [0.8, 0.8, 0.8],
        [1.1, 1.1, 1.1],
        0.05,
        0.05,
        MaterialType::NeoHookean,
        5.0e4,
        1.0e5,
    );
    for _ in 0..5 {
        mpm_step(
            &mut grid,
            &mut at_rest,
            2.0e-4,
            [0.0; 3],
            THETA_C,
            THETA_S,
            2,
        );
    }
    let drift = at_rest.iter().map(|p| len3(p.velocity)).fold(0.0, f64::max);
    assert!(drift < 1e-9, "unstrained block moved: |v| = {drift}");

    // A uniformly stretched block (F = 1.05 1) contracts towards its centre.
    let mut grid = MpmGrid::new([20, 20, 20], GRID_H, [0.0; 3]);
    let mut stretched = init_particle_block(
        [0.8, 0.8, 0.8],
        [1.1, 1.1, 1.1],
        0.05,
        0.05,
        MaterialType::NeoHookean,
        5.0e4,
        1.0e5,
    );
    let centre = [0.95; 3];
    for p in &mut stretched {
        p.deformation_gradient = mat3_scale(mat3_identity(), 1.05);
        p.fe = p.deformation_gradient;
    }
    mpm_step(
        &mut grid,
        &mut stretched,
        2.0e-4,
        [0.0; 3],
        THETA_C,
        THETA_S,
        2,
    );
    // Outer particles are pulled inwards: v . (x - c) < 0 on the block surface.
    let corner = stretched
        .iter()
        .max_by(|a, b| len3(sub3(a.position, centre)).total_cmp(&len3(sub3(b.position, centre))))
        .map(|p| dot3(p.velocity, sub3(p.position, centre)));
    let inward = corner.unwrap_or(0.0);
    assert!(
        inward < 0.0,
        "stretched block does not contract: v.r = {inward}"
    );
}

#[test]
fn f4178_compressed_fluid_block_expands_and_stretched_fluid_contracts() {
    let centre = [0.95; 3];
    for (scale, expands) in [(0.95_f64, true), (1.05, false)] {
        let mut grid = MpmGrid::new([20, 20, 20], GRID_H, [0.0; 3]);
        let mut block = init_particle_block(
            [0.8, 0.8, 0.8],
            [1.1, 1.1, 1.1],
            0.05,
            0.05,
            MaterialType::Fluid,
            0.0,
            1.0e5,
        );
        for p in &mut block {
            p.deformation_gradient = mat3_scale(mat3_identity(), scale);
            p.fe = p.deformation_gradient;
        }
        mpm_step(&mut grid, &mut block, 2.0e-4, [0.0; 3], THETA_C, THETA_S, 2);
        // The particle farthest from the centre sits on the block surface.
        let outward = block
            .iter()
            .max_by(|a, b| {
                len3(sub3(a.position, centre)).total_cmp(&len3(sub3(b.position, centre)))
            })
            .map(|p| dot3(p.velocity, sub3(p.position, centre)))
            .unwrap_or(0.0);
        assert_eq!(
            outward > 0.0,
            expands,
            "F = {scale} 1: surface particle v.r = {outward}"
        );
        assert!(outward.abs() > 1e-6, "no fluid force at all");
    }
}
