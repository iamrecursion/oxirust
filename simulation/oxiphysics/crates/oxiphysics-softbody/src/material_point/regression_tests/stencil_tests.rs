//! F4181: the quadratic B-spline stencil covers `base..=base + 2`, its weights sum to
//! one, and the P2G/G2P transfers conserve and reproduce what they should.

use super::*;

#[test]
fn f4181_stencil_weights_form_a_partition_of_unity_for_every_fraction() {
    let grid = MpmGrid::new([16, 16, 16], 0.1, [0.0; 3]);
    for step in 0..200 {
        // Fractional part of the position in cells sweeps the whole [0, 1) range,
        // including the (0.5, 1) half where the old stencil dropped a node.
        let offset = step as f64 / 200.0;
        let pos = [
            (6.0 + offset) * 0.1,
            (7.0 + offset) * 0.1,
            (5.0 + offset) * 0.1,
        ];
        let (base, frac) = grid.base_and_frac(pos);
        let (w, dw) = compute_weights(base, frac, grid.h);
        for d in 0..3 {
            let sum: f64 = w[d].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "axis {d}, offset {offset}: weights sum to {sum}"
            );
            let dsum: f64 = dw[d].iter().sum();
            assert!(
                dsum.abs() < 1e-9,
                "axis {d}, offset {offset}: gradient weights sum to {dsum}"
            );
            // Linear reproduction: sum_k w_k (x_k - x_p) = 0 (in cells).
            let first_moment: f64 = (0..3).map(|k| w[d][k] * (k as f64 - frac[d])).sum();
            assert!(
                first_moment.abs() < 1e-12,
                "axis {d}, offset {offset}: first moment {first_moment}"
            );
            // Second moment of the quadratic B-spline is exactly 1/4 cell^2.
            let second_moment: f64 = (0..3)
                .map(|k| w[d][k] * (k as f64 - frac[d]) * (k as f64 - frac[d]))
                .sum();
            assert!(
                (second_moment - 0.25).abs() < 1e-12,
                "axis {d}, offset {offset}: second moment {second_moment}"
            );
            assert!(w[d].iter().all(|&wk| wk >= 0.0));
        }
    }
}

#[test]
fn f4181_base_and_frac_places_fraction_in_half_open_interval() {
    let grid = MpmGrid::new([32, 32, 32], 0.25, [-1.0, 0.5, 2.0]);
    let mut rng = TestRng::new(0xF4181);
    for _ in 0..500 {
        let pos = [
            rng.range(-0.5, 5.0),
            rng.range(1.5, 6.0),
            rng.range(3.0, 8.0),
        ];
        let (base, frac) = grid.base_and_frac(pos);
        for d in 0..3 {
            let x = (pos[d] - grid.origin[d]) / grid.h;
            assert!(
                (0.5..1.5).contains(&frac[d]),
                "frac {} outside [0.5, 1.5)",
                frac[d]
            );
            assert!((base[d] as f64 + frac[d] - x).abs() < 1e-12);
            // The middle stencil node is the nearest grid node.
            let nearest = x.round() as i64;
            assert_eq!(base[d] + 1, nearest, "x = {x}");
        }
    }
}

#[test]
fn f4181_p2g_conserves_mass_and_linear_momentum_at_arbitrary_positions() {
    let mut rng = TestRng::new(0xABCD_0001);
    let mut grid = MpmGrid::new([16, 16, 16], 0.1, [0.0; 3]);
    let mut particles = Vec::new();
    for _ in 0..300 {
        let mut p = MpmParticle::new(
            rng.vec3(0.45, 1.05),
            rng.vec3(-1.0, 1.0),
            rng.range(0.5, 2.0),
            1.0e-3,
            MaterialType::NeoHookean,
            1.0e4,
            2.0e4,
        );
        p.affine_c = rng.mat3(-5.0, 5.0);
        particles.push(p);
    }
    p2g_transfer(&mut grid, &particles, 1.0e-4, [0.0; 3]);

    let particle_mass: f64 = particles.iter().map(|p| p.mass).sum();
    let grid_mass: f64 = grid.nodes.iter().map(|n| n.mass).sum();
    assert!(
        ((grid_mass - particle_mass) / particle_mass).abs() < 1e-12,
        "mass {grid_mass} vs {particle_mass}"
    );

    let mut particle_momentum = [0.0f64; 3];
    let mut momentum_scale = 0.0f64;
    for p in &particles {
        particle_momentum = add3(particle_momentum, scale3(p.velocity, p.mass));
        momentum_scale += p.mass * len3(p.velocity);
    }
    let grid_momentum = grid
        .nodes
        .iter()
        .fold([0.0f64; 3], |acc, n| add3(acc, n.momentum));
    // The affine term sum_i w_i (x_i - x_p) vanishes only for a complete stencil.
    assert!(
        vec_diff(grid_momentum, particle_momentum) < 1e-12 * momentum_scale,
        "momentum {grid_momentum:?} vs {particle_momentum:?}"
    );
}

#[test]
fn f4181_p2g_g2p_round_trip_reproduces_uniform_velocity() {
    let mut rng = TestRng::new(0x5EED_4181);
    let mut grid = MpmGrid::new([16, 16, 16], 0.1, [0.0; 3]);
    let velocity = [0.3, -0.2, 0.1];
    let mut particles = Vec::new();
    for _ in 0..400 {
        particles.push(MpmParticle::new(
            rng.vec3(0.5, 1.0),
            velocity,
            1.0,
            1.0e-3,
            MaterialType::NeoHookean,
            0.0,
            0.0,
        ));
    }
    p2g_transfer(&mut grid, &particles, 1.0e-3, [0.0; 3]);
    g2p_transfer(&grid, &mut particles);
    for p in &particles {
        assert!(
            vec_diff(p.velocity, velocity) < 1e-12,
            "velocity {:?} drifted from {velocity:?}",
            p.velocity
        );
        assert!(
            max_diff(p.affine_c, mat3_zero()) < 1e-9,
            "uniform flow must have C = 0, got {:?}",
            p.affine_c
        );
    }
}

/// Fills the grid velocity with the affine field `v(x) = a + B x`.
fn fill_affine_velocity(grid: &mut MpmGrid, a: [f64; 3], b: [[f64; 3]; 3]) {
    let [nx, ny, nz] = grid.dims;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let x = grid.node_pos(ix, iy, iz);
                let idx = grid.idx(ix, iy, iz);
                grid.nodes[idx].velocity = add3(a, mat3_vec(b, x));
            }
        }
    }
}

#[test]
fn f4181_g2p_reproduces_affine_velocity_field_in_every_transfer() {
    let mut rng = TestRng::new(0x0630_4181);
    let mut grid = MpmGrid::new([16, 16, 16], 0.1, [0.0; 3]);
    let a = [0.4, -0.3, 0.2];
    let b = [[0.5, -0.2, 0.1], [0.3, 0.7, -0.4], [-0.6, 0.2, 0.9]];
    fill_affine_velocity(&mut grid, a, b);
    let template: Vec<MpmParticle> = (0..100)
        .map(|_| {
            MpmParticle::new(
                rng.vec3(0.45, 1.05),
                [0.0; 3],
                1.0,
                1.0e-3,
                MaterialType::NeoHookean,
                1.0e4,
                2.0e4,
            )
        })
        .collect();

    let check = |label: &str, particles: &[MpmParticle]| {
        for p in particles {
            let exact = add3(a, mat3_vec(b, p.position));
            assert!(
                vec_diff(p.velocity, exact) < 1e-12,
                "{label}: v = {:?}, exact {exact:?}",
                p.velocity
            );
            assert!(
                max_diff(p.affine_c, b) < 1e-9,
                "{label}: C = {:?}, exact {b:?}",
                p.affine_c
            );
        }
    };

    let mut with_dt = template.clone();
    g2p_transfer_dt(&grid, &mut with_dt, 1.0e-3);
    check("g2p_transfer_dt", &with_dt);

    let mut plain = template.clone();
    g2p_transfer(&grid, &mut plain);
    check("g2p_transfer", &plain);

    let mut flip = template.clone();
    let old_velocity = vec![[0.0f64; 3]; grid.num_nodes()];
    g2p_flip(&grid, &old_velocity, &mut flip, 0.0);
    check("g2p_flip(PIC)", &flip);
}
