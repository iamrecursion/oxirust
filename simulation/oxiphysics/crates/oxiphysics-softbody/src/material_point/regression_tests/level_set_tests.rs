//! F4179: level-set coupling computes the gradient, pushes particles out and
//! applies the restitution.

use super::*;

/// Samples `f` at every grid node (flat node order).
fn sample_field(grid: &MpmGrid, f: impl Fn([f64; 3]) -> f64) -> Vec<f64> {
    let [nx, ny, nz] = grid.dims;
    let mut field = vec![0.0; grid.num_nodes()];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                field[grid.idx(ix, iy, iz)] = f(grid.node_pos(ix, iy, iz));
            }
        }
    }
    field
}

fn probe(position: [f64; 3], velocity: [f64; 3]) -> MpmParticle {
    MpmParticle::new(
        position,
        velocity,
        1.0,
        VOLUME0,
        MaterialType::NeoHookean,
        1.0e4,
        2.0e4,
    )
}

#[test]
fn f4179_floor_level_set_pushes_the_particle_out_and_applies_the_restitution() {
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    // Solid half-space y < 0.5: phi = y - 0.5 is negative inside.
    let phi = sample_field(&grid, |x| x[1] - 0.5);
    // v_n' = -e v_n: e = 0 removes the inward velocity, e = 1 bounces elastically.
    for (restitution, expected_vy) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
        let mut particles = [probe([0.8, 0.45, 0.8], [0.2, -1.0, 0.1])];
        apply_level_set_coupling(&mut particles, &phi, &grid, restitution);
        let p = &particles[0];
        assert!(p.position[1] >= 0.5 - 1e-6, "y = {}", p.position[1]);
        assert!((p.position[1] - 0.5).abs() < 1e-9, "y = {}", p.position[1]);
        assert!((p.position[0] - 0.8).abs() < 1e-12 && (p.position[2] - 0.8).abs() < 1e-12);
        assert!(
            p.velocity[1] >= -1e-12,
            "e = {restitution}: v_y = {}",
            p.velocity[1]
        );
        assert!(
            (p.velocity[1] - expected_vy).abs() < 1e-9,
            "e = {restitution}: v_y = {}",
            p.velocity[1]
        );
        // The tangential velocity is untouched.
        assert!((p.velocity[0] - 0.2).abs() < 1e-12 && (p.velocity[2] - 0.1).abs() < 1e-12);
        // The sampled signed distance is recorded on the particle.
        assert!(
            (p.level_set_phi + 0.05).abs() < 1e-9,
            "phi = {}",
            p.level_set_phi
        );
    }
}

#[test]
fn f4179_only_penetrating_particles_are_moved_and_outgoing_ones_keep_their_velocity() {
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let phi = sample_field(&grid, |x| x[1] - 0.5);

    // Inside the solid but already moving out: repositioned, velocity unchanged.
    let mut inside_leaving = [probe([0.8, 0.45, 0.8], [0.3, 1.0, -0.2])];
    apply_level_set_coupling(&mut inside_leaving, &phi, &grid, 0.8);
    assert!((inside_leaving[0].position[1] - 0.5).abs() < 1e-9);
    assert_eq!(inside_leaving[0].velocity, [0.3, 1.0, -0.2]);

    // Outside the solid: untouched, only the sampled distance is recorded.
    let mut outside = [probe([0.8, 0.62, 0.8], [0.3, -2.0, 0.0])];
    apply_level_set_coupling(&mut outside, &phi, &grid, 0.8);
    assert_eq!(outside[0].position, [0.8, 0.62, 0.8]);
    assert_eq!(outside[0].velocity, [0.3, -2.0, 0.0]);
    assert!((outside[0].level_set_phi - 0.12).abs() < 1e-9);

    // Inactive particles are ignored entirely.
    let mut inactive = [probe([0.8, 0.45, 0.8], [0.0, -1.0, 0.0])];
    inactive[0].active = false;
    apply_level_set_coupling(&mut inactive, &phi, &grid, 0.8);
    assert_eq!(inactive[0].position, [0.8, 0.45, 0.8]);
    assert_eq!(inactive[0].velocity, [0.0, -1.0, 0.0]);
    assert_eq!(inactive[0].level_set_phi, f64::MAX);
}

#[test]
fn f4179_curved_level_set_pushes_along_the_gradient() {
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let centre = [0.8, 0.8, 0.8];
    let radius = 0.3;
    // Solid sphere: phi = |x - c| - R.
    let phi = sample_field(&grid, |x| len3(sub3(x, centre)) - radius);
    let normal = norm3([1.0, 1.0, 0.0]);
    let start = add3(centre, scale3(normal, 0.27));
    let velocity = add3(scale3(normal, -2.0), [0.0, 0.0, 0.5]);
    let mut particles = [probe(start, velocity)];
    apply_level_set_coupling(&mut particles, &phi, &grid, 0.5);
    let p = &particles[0];

    // Moved outward along the normal, close to the surface (the quadratic B-spline
    // interpolant of a curved field is biased by about h²/(8 R) · 2).
    let shift = sub3(p.position, start);
    assert!(len3(shift) > 0.01, "particle was not pushed: {shift:?}");
    assert!(
        dot3(norm3(shift), normal) > 0.999,
        "shift {shift:?} not along the normal"
    );
    let distance = len3(sub3(p.position, centre));
    assert!(
        (distance - radius).abs() < 0.02,
        "distance from centre {distance}"
    );
    assert!(distance > 0.27);

    // v_n' = -e v_n = +1 along the normal, tangential part kept.
    let v_n = dot3(p.velocity, normal);
    assert!((v_n - 1.0).abs() < 0.02, "v_n = {v_n}");
    assert!(
        (p.velocity[2] - 0.5).abs() < 0.02,
        "v_z = {}",
        p.velocity[2]
    );
}

#[test]
fn f4179_particles_without_level_set_data_are_left_alone() {
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let make = || {
        [
            probe([0.8, 0.45, 0.8], [0.2, -1.0, 0.1]),
            probe([0.02, 0.45, 0.8], [0.2, -1.0, 0.1]),
        ]
    };

    // The compute_level_set sentinel f64::MAX means "no data".
    let sentinel = vec![f64::MAX; grid.num_nodes()];
    let mut particles = make();
    apply_level_set_coupling(&mut particles, &sentinel, &grid, 0.5);
    for p in &particles {
        assert_eq!(p.position[1], 0.45);
        assert_eq!(p.velocity, [0.2, -1.0, 0.1]);
    }

    // Non-finite samples never leak NaN into the particles.
    let mut broken = sample_field(&grid, |x| x[1] - 0.5);
    broken[grid.idx(8, 5, 8)] = f64::NAN;
    let mut particles = make();
    apply_level_set_coupling(&mut particles, &broken, &grid, 0.5);
    for p in &particles {
        assert!(
            p.position
                .iter()
                .chain(p.velocity.iter())
                .all(|x| x.is_finite())
        );
    }

    // A stencil that leaves the grid cannot be evaluated: the second particle (x = 0.02)
    // keeps its state even though the field would put it inside the solid.
    let phi = sample_field(&grid, |x| x[1] - 0.5);
    let mut particles = make();
    apply_level_set_coupling(&mut particles, &phi, &grid, 0.5);
    assert!((particles[0].position[1] - 0.5).abs() < 1e-9);
    assert_eq!(particles[1].position, [0.02, 0.45, 0.8]);
    assert_eq!(particles[1].velocity, [0.2, -1.0, 0.1]);

    // A field of the wrong length is rejected without panicking.
    let short = vec![-1.0; grid.num_nodes() - 1];
    let mut particles = make();
    apply_level_set_coupling(&mut particles, &short, &grid, 0.5);
    assert_eq!(particles[0].position, [0.8, 0.45, 0.8]);
}

#[test]
fn f4179_compute_level_set_feeds_the_coupling() {
    // A "solid" made of one particle: the field is |x - x_s| - r, negative inside.
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let solid = probe([0.8, 0.8, 0.8], [0.0; 3]);
    let phi = compute_level_set(std::slice::from_ref(&solid), &grid, 0.3);
    let mut particles = [probe([0.8, 0.8, 0.55], [0.0, 0.0, 1.0])];
    apply_level_set_coupling(&mut particles, &phi, &grid, 1.0);
    let p = &particles[0];
    // phi at the probe is about |0.25| - 0.3 = -0.05: it is pushed away from the
    // sphere centre (down, -z) and bounces back out.
    assert!(p.position[2] < 0.55, "z = {}", p.position[2]);
    assert!(p.velocity[2] < 0.0, "v_z = {}", p.velocity[2]);
}

#[test]
fn f4181_level_set_splat_covers_every_node_near_the_particle() {
    let grid = MpmGrid::new([16, 16, 16], GRID_H, [0.0; 3]);
    let particle = probe([0.77, 0.83, 0.81], [0.0; 3]);
    let radius = 0.1;
    let phi = compute_level_set(std::slice::from_ref(&particle), &grid, radius);
    let [nx, ny, nz] = grid.dims;
    let mut covered = 0;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let d = len3(sub3(grid.node_pos(ix, iy, iz), particle.position));
                let value = phi[grid.idx(ix, iy, iz)];
                if d <= 2.5 * GRID_H {
                    assert!(
                        (value - (d - radius)).abs() < 1e-12,
                        "node ({ix},{iy},{iz}) at distance {d}: phi = {value}"
                    );
                    covered += 1;
                }
            }
        }
    }
    assert!(covered > 30, "only {covered} nodes checked");
    assert_eq!(phi[grid.idx(0, 0, 0)], f64::MAX);
    assert_eq!(phi[grid.idx(15, 15, 15)], f64::MAX);
}
