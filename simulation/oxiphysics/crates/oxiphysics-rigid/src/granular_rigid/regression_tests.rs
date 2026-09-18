// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Hertz-Mindlin contact model and the persistent
//! contact list of the granular solver.
//!
//! They pin the sign conventions (contact-point velocity, damping, friction),
//! Newton's third law for every contribution of a contact (normal, shear,
//! rolling resistance, cohesion), the Mindlin stiffness law, the restitution
//! calibration of the normal dashpot and the persistence of the tangential
//! spring between steps.

use super::*;

/// Grain of radius 0.05 m and density 2500 kg/m³ (E = 1e8 Pa, ν = 0.3,
/// μ = 0.4, e = 0.7) centred at `position`.
fn grain(id: usize, position: [f64; 3]) -> GranularParticle {
    let mut p = GranularParticle::new(id, 0.05, 2500.0);
    p.position = position;
    p
}

/// Two grains whose centres are `0.1 - overlap` apart along x.
fn touching_pair(overlap: f64) -> (GranularParticle, GranularParticle) {
    (grain(0, [0.0; 3]), grain(1, [0.1 - overlap, 0.0, 0.0]))
}

fn zero_gravity_system(dt: f64) -> GranularSystem {
    GranularSystem::new([0.0; 3], [[-100.0, 100.0]; 3], dt)
}

fn momentum(particles: &[GranularParticle]) -> [f64; 3] {
    particles
        .iter()
        .fold([0.0; 3], |acc, p| add(acc, scale(p.velocity, p.mass)))
}

// ── Mindlin tangential stiffness ─────────────────────────────────────────

#[test]
fn mindlin_stiffness_scales_with_the_square_root_of_the_normal_overlap() {
    let k1 = mindlin_tangential_stiffness(1.0e7, 0.025, 1.0e-4);
    let k4 = mindlin_tangential_stiffness(1.0e7, 0.025, 4.0e-4);
    assert!((k4 / k1 - 2.0).abs() < 1e-12, "ratio = {}", k4 / k1);
    let expected = 8.0 * 1.0e7 * (0.025_f64 * 1.0e-4).sqrt();
    assert!((k1 - expected).abs() < 1e-9 * expected);
    // No overlap, no contact patch, no stiffness.
    assert_eq!(mindlin_tangential_stiffness(1.0e7, 0.025, 0.0), 0.0);
    assert_eq!(mindlin_tangential_stiffness(1.0e7, 0.025, -1.0e-4), 0.0);
}

#[test]
fn mindlin_force_is_linear_in_the_tangential_displacement_below_the_cap() {
    let (g_star, r_star, delta_n) = (1.0e7, 0.025, 1.0e-4);
    let kt = mindlin_tangential_stiffness(g_star, r_star, delta_n);
    let f1 = mindlin_tangential_force(1.0e-6, g_star, r_star, delta_n, 1.0e3, 0.4);
    let f2 = mindlin_tangential_force(2.0e-6, g_star, r_star, delta_n, 1.0e3, 0.4);
    assert!(f1 > 0.0 && f1 < 400.0, "f1 = {f1}");
    assert!((f1 - kt * 1.0e-6).abs() < 1e-12 * f1);
    assert!((f2 / f1 - 2.0).abs() < 1e-12, "ratio = {}", f2 / f1);
}

#[test]
fn mindlin_force_saturates_at_the_coulomb_limit() {
    let f = mindlin_tangential_force(1.0, 1.0e7, 0.025, 1.0e-4, 1.0e3, 0.4);
    assert!((f - 400.0).abs() < 1e-12, "f = {f}");
    // A tensile (negative) normal force provides no friction.
    assert_eq!(
        mindlin_tangential_force(1.0, 1.0e7, 0.025, 1.0e-4, -5.0, 0.4),
        0.0
    );
}

// ── Normal force: Tsuji dashpot ──────────────────────────────────────────

#[test]
fn normal_damping_dissipates_energy_instead_of_adding_it() {
    // Same overlap, particle i approaching (+x) versus separating (-x).
    let speed = 0.2;
    let result_for = |vx: f64| {
        let (mut pi, pj) = touching_pair(1.0e-3);
        pi.velocity = [vx, 0.0, 0.0];
        resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-6, None, None)
            .expect("overlapping particles are in contact")
    };
    let resting = result_for(0.0);
    let approaching = result_for(speed);
    let separating = result_for(-speed);

    // The force on i points along -x (repulsion); its size is fn_mag.
    assert!(resting.force_i[0] < 0.0);
    assert!(
        approaching.fn_mag > resting.fn_mag,
        "approach must add dashpot repulsion: {} vs {}",
        approaching.fn_mag,
        resting.fn_mag
    );
    assert!(
        separating.fn_mag < resting.fn_mag,
        "separation must remove repulsion: {} vs {}",
        separating.fn_mag,
        resting.fn_mag
    );
    // The dashpot term is odd in the normal velocity.
    let up = approaching.fn_mag - resting.fn_mag;
    let down = resting.fn_mag - separating.fn_mag;
    assert!((up - down).abs() < 1e-9 * up, "{up} vs {down}");
    // The stored elastic energy does not depend on the velocity.
    assert!((approaching.strain_energy - separating.strain_energy).abs() < 1e-12);
    assert!(resting.strain_energy > 0.0);
}

#[test]
fn elastic_particles_have_no_normal_damping() {
    let (mut pi, mut pj) = touching_pair(1.0e-3);
    pi.restitution = 1.0;
    pj.restitution = 1.0;
    pi.velocity = [0.3, 0.0, 0.0];
    let moving = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-6, None, None)
        .expect("overlapping particles are in contact");
    pi.velocity = [0.0; 3];
    let resting = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-6, None, None)
        .expect("overlapping particles are in contact");
    assert_eq!(moving.fn_mag, resting.fn_mag);
}

/// Head-on collision of two identical grains approaching at `closing` m/s in
/// total.  The system is stepped until the grains have touched and separated
/// again (at most 0.2 s).  Returns the ratio of the final separation speed to
/// the approach speed and whether the grains separated.
fn head_on_collision(restitution: f64, closing: f64) -> (f64, bool) {
    let mut sys = zero_gravity_system(1.0e-5);
    let mut a = grain(0, [-0.0505, 0.0, 0.0]);
    let mut b = grain(1, [0.0505, 0.0, 0.0]);
    a.restitution = restitution;
    b.restitution = restitution;
    a.velocity = [0.5 * closing, 0.0, 0.0];
    b.velocity = [-0.5 * closing, 0.0, 0.0];
    sys.add_particle(a);
    sys.add_particle(b);
    let mut touched = false;
    for _ in 0..20_000 {
        sys.step();
        touched |= !sys.contacts.is_empty();
        if touched && sys.contacts.is_empty() {
            break;
        }
    }
    assert!(touched, "the grains never touched");
    let ratio = (sys.particles[1].velocity[0] - sys.particles[0].velocity[0]) / closing;
    (ratio, sys.contacts.is_empty())
}

/// Rebound ratio of a collision in which the grains do separate.
fn head_on_restitution(restitution: f64, closing: f64) -> f64 {
    let (ratio, separated) = head_on_collision(restitution, closing);
    assert!(separated, "e = {restitution}: the grains never separated");
    ratio
}

#[test]
fn head_on_collision_rebounds_with_the_requested_restitution() {
    for &e in &[0.3, 0.5, 0.7, 0.9] {
        let ratio = head_on_restitution(e, 0.5);
        assert!((ratio - e).abs() < 0.01, "e = {e}: rebound ratio {ratio}");
    }
}

#[test]
fn head_on_rebound_does_not_depend_on_the_impact_speed() {
    // The Hertz-Tsuji contact is self-similar in the approach speed.
    let slow = head_on_restitution(0.5, 0.2);
    let fast = head_on_restitution(0.5, 1.0);
    assert!((slow - 0.5).abs() < 0.01, "slow = {slow}");
    assert!((fast - 0.5).abs() < 0.01, "fast = {fast}");
}

#[test]
fn perfectly_elastic_collision_returns_the_approach_speed() {
    let elastic = head_on_restitution(1.0, 0.5);
    assert!((elastic - 1.0).abs() < 0.01, "e = 1 rebound {elastic}");
}

#[test]
fn perfectly_inelastic_collision_does_not_rebound() {
    // A critically damped contact creeps apart asymptotically, so the grains
    // need not separate within the observation time; they must in any case not
    // move apart (or into each other) with an appreciable speed.
    let (ratio, _) = head_on_collision(0.0, 0.5);
    assert!(
        (0.0..0.01).contains(&ratio),
        "e = 0 must not rebound, got {ratio}"
    );
}

// ── Contact kinematics and friction ──────────────────────────────────────

#[test]
fn slip_velocity_includes_the_spin_of_particle_i() {
    // i spins about +z, j (to the +x side) is at rest: the contact point on i
    // moves with ω × (R_i x̂) = +R_i ŷ, so the slip is +R_i ŷ.
    let (mut pi, mut pj) = touching_pair(1.0e-3);
    // Isolate the friction torque from the rolling-resistance couple.
    pi.rolling_coeff = 0.0;
    pj.rolling_coeff = 0.0;
    pi.angular_velocity = [0.0, 0.0, 1.0];
    let dt = 1.0e-6;
    let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], dt, None, None)
        .expect("overlapping particles are in contact");
    let spring = res.new_tangential_spring;
    assert!(
        (spring[1] - pi.radius * dt).abs() < 1e-15,
        "shear displacement {spring:?} must be +R_i ŷ · dt"
    );
    assert!(spring[0].abs() < 1e-15 && spring[2].abs() < 1e-15);

    // Friction opposes the slip: it pulls i along -y and brakes its spin.
    assert!(res.force_i[1] < 0.0, "force_i = {:?}", res.force_i);
    assert!(res.torque_i[2] < 0.0, "torque_i = {:?}", res.torque_i);
    // j is dragged along +y and starts to counter-rotate like a gear.
    assert!(res.force_j[1] > 0.0, "force_j = {:?}", res.force_j);
    assert!(res.torque_j[2] < 0.0, "torque_j = {:?}", res.torque_j);
}

#[test]
fn slip_velocity_includes_the_spin_of_particle_j() {
    // j spins about +z; its contact point lies at -R_j x̂ from its centre and
    // moves with ω × (-R_j x̂) = -R_j ŷ, so the slip of i relative to j is +R_j ŷ.
    let (mut pi, mut pj) = touching_pair(1.0e-3);
    pi.rolling_coeff = 0.0;
    pj.rolling_coeff = 0.0;
    pj.angular_velocity = [0.0, 0.0, 1.0];
    let dt = 1.0e-6;
    let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], dt, None, None)
        .expect("overlapping particles are in contact");
    assert!((res.new_tangential_spring[1] - pj.radius * dt).abs() < 1e-15);
    assert!(res.force_i[1] < 0.0, "force_i = {:?}", res.force_i);
    // The friction torque on j opposes its own spin.
    assert!(res.torque_j[2] < 0.0, "torque_j = {:?}", res.torque_j);
}

#[test]
fn rolling_contact_is_free_of_slip() {
    // i rolls over the resting j without slipping: v_i + ω_i × (R_i x̂) = 0.
    let (mut pi, pj) = touching_pair(1.0e-3);
    pi.angular_velocity = [0.0, 0.0, 2.0];
    pi.velocity = [0.0, -pi.radius * 2.0, 0.0];
    let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-4, None, None)
        .expect("overlapping particles are in contact");
    assert!(norm(res.new_tangential_spring) < 1e-18);
    assert!(norm(res.force_i) > 0.0, "the normal force must remain");
    assert!(res.force_i[1].abs() < 1e-9 && res.force_i[2].abs() < 1e-9);
}

#[test]
fn sliding_contact_force_is_capped_by_coulomb_friction() {
    let (mut pi, pj) = touching_pair(1.0e-3);
    pi.velocity = [0.0, 10.0, 0.0];
    let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0, None, None)
        .expect("overlapping particles are in contact");
    let mu = 0.4;
    let tangential = res.force_i[1];
    assert!(
        tangential < 0.0,
        "friction must oppose the slip: {tangential}"
    );
    assert!(
        (tangential.abs() - mu * res.fn_mag).abs() < 1e-9 * res.fn_mag,
        "|F_t| = {} vs mu * F_n = {}",
        tangential.abs(),
        mu * res.fn_mag
    );
    // The stored spring is limited so that k_t * |spring| = mu * F_n, with the
    // shear modulus G* = E / (4 (2 - nu) (1 + nu)) of two identical grains.
    let g_star = 1.0e8 / (4.0 * (2.0 - 0.3) * (1.0 + 0.3));
    let kt = mindlin_tangential_stiffness(g_star, 0.025, 1.0e-3);
    assert!(
        (kt * norm(res.new_tangential_spring) - mu * res.fn_mag).abs() < 1e-9 * res.fn_mag,
        "spring = {:?}",
        res.new_tangential_spring
    );
}

#[test]
fn shear_history_is_projected_onto_the_tangent_plane_of_the_contact() {
    // A carried spring with a normal component must not push along the normal.
    let (pi, pj) = touching_pair(1.0e-3);
    let spring = [3.0e-6, 2.0e-6, -1.0e-6];
    let res = resolve_hertz_mindlin(&pi, &pj, spring, 0.0, None, None)
        .expect("overlapping particles are in contact");
    assert!(res.new_tangential_spring[0].abs() < 1e-18);
    assert!((res.new_tangential_spring[1] - 2.0e-6).abs() < 1e-18);
    assert!((res.new_tangential_spring[2] + 1.0e-6).abs() < 1e-18);
}

// ── Newton's third law ───────────────────────────────────────────────────

#[test]
fn hertz_mindlin_conserves_momentum_for_a_sliding_spinning_contact() {
    let (mut pi, mut pj) = touching_pair(2.0e-3);
    pi.velocity = [0.4, 0.3, -0.2];
    pj.velocity = [-0.1, -0.2, 0.15];
    pi.angular_velocity = [1.0, -2.0, 3.0];
    pj.angular_velocity = [-0.5, 0.7, 1.5];
    let spring = [0.0, 2.0e-6, -1.0e-6];
    let res = resolve_hertz_mindlin(&pi, &pj, spring, 1.0e-5, None, None)
        .expect("overlapping particles are in contact");
    assert!(
        norm(res.force_i) > 1.0,
        "the contact must carry load: {:?}",
        res.force_i
    );
    let net = add(res.force_i, res.force_j);
    assert!(norm(net) < 1e-12, "force_i + force_j = {net:?}");
}

#[test]
fn hertz_mindlin_conserves_momentum_with_cohesion() {
    let (mut pi, mut pj) = touching_pair(1.0e-3);
    pi.velocity = [0.0, 0.5, 0.0];
    pj.velocity = [0.0, -0.5, 0.0];
    for (capillary, hamaker) in [(Some((0.072, 1.0e-12)), None), (None, Some(1.0e-19))] {
        let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-5, capillary, hamaker)
            .expect("overlapping particles are in contact");
        let net = add(res.force_i, res.force_j);
        assert!(
            norm(net) < 1e-12,
            "{capillary:?}/{hamaker:?}: net = {net:?}"
        );
    }
}

#[test]
fn rolling_resistance_torques_are_equal_and_opposite() {
    // Without Coulomb friction only the rolling-resistance couple remains.
    let (mut pi, mut pj) = touching_pair(1.0e-3);
    pi.friction = 0.0;
    pj.friction = 0.0;
    pi.angular_velocity = [0.0, 0.0, 4.0];
    pj.angular_velocity = [0.0, 0.0, -1.0];
    let res = resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-5, None, None)
        .expect("overlapping particles are in contact");
    assert!(res.torque_i[2] < 0.0, "torque_i = {:?}", res.torque_i);
    let net = add(res.torque_i, res.torque_j);
    assert!(norm(net) < 1e-15, "torque_i + torque_j = {net:?}");
}

#[test]
fn granular_system_conserves_linear_momentum_without_external_forces() {
    for capillary in [None, Some((0.072, 2.0e-12))] {
        let mut sys = zero_gravity_system(1.0e-5);
        sys.capillary_params = capillary;
        // A grain cluster in which every grain touches at least one other.
        sys.tensile_strength = 1.0e12; // keep the grains from being crushed (frozen)
        let layout = [
            (
                [0.00, 0.00, 0.00],
                0.05,
                [0.30, 0.10, -0.20],
                [1.0, 2.0, 0.5],
            ),
            (
                [0.08, 0.02, 0.00],
                0.04,
                [-0.20, 0.05, 0.10],
                [0.0, -1.0, 2.0],
            ),
            (
                [0.03, 0.08, 0.02],
                0.06,
                [0.10, -0.30, 0.00],
                [-2.0, 0.5, 1.0],
            ),
            (
                [-0.07, 0.04, 0.05],
                0.05,
                [0.25, 0.20, 0.30],
                [0.3, 0.3, -0.3],
            ),
        ];
        for (idx, (pos, radius, vel, spin)) in layout.into_iter().enumerate() {
            let mut p = GranularParticle::new(idx, radius, 2500.0 + 300.0 * idx as f64);
            p.position = pos;
            p.velocity = vel;
            p.angular_velocity = spin;
            sys.add_particle(p);
        }
        let before = momentum(&sys.particles);
        for _ in 0..50 {
            sys.step();
        }
        assert!(!sys.contacts.is_empty(), "the cluster must be in contact");
        let after = momentum(&sys.particles);
        let scale_ref = norm(before);
        for k in 0..3 {
            assert!(
                (after[k] - before[k]).abs() < 1e-9 * scale_ref,
                "capillary = {capillary:?}, axis {k}: {before:?} -> {after:?}"
            );
        }
    }
}

/// Translational plus rotational kinetic energy of all grains.
fn total_kinetic_energy(particles: &[GranularParticle]) -> f64 {
    particles
        .iter()
        .map(|p| p.kinetic_energy() + p.rotational_energy())
        .sum()
}

#[test]
fn oblique_spinning_collision_dissipates_energy_through_friction() {
    // Elastic normal contact and no rolling resistance: friction is the only
    // dissipation mechanism.  A friction law with the wrong sign (or a wrong
    // contact-point velocity) would pump energy into the spin instead.
    let mut sys = zero_gravity_system(1.0e-5);
    let mut a = grain(0, [0.0; 3]);
    let mut b = grain(1, [0.15, 0.07, 0.0]);
    for g in [&mut a, &mut b] {
        g.restitution = 1.0;
        g.rolling_coeff = 0.0;
    }
    a.velocity = [0.5, 0.0, 0.0];
    a.angular_velocity = [0.0, 0.0, 20.0];
    sys.add_particle(a);
    sys.add_particle(b);

    let energy_before = total_kinetic_energy(&sys.particles);
    let momentum_before = momentum(&sys.particles);
    let mut touched = false;
    for _ in 0..20_000 {
        sys.step();
        touched |= !sys.contacts.is_empty();
        if touched && sys.contacts.is_empty() {
            break;
        }
    }
    assert!(touched && sys.contacts.is_empty(), "no complete collision");

    let energy_after = total_kinetic_energy(&sys.particles);
    assert!(
        energy_after < energy_before * (1.0 - 1.0e-3),
        "friction must dissipate energy: {energy_before} -> {energy_after}"
    );
    let momentum_after = momentum(&sys.particles);
    for k in 0..3 {
        assert!(
            (momentum_after[k] - momentum_before[k]).abs() < 1e-9 * norm(momentum_before),
            "momentum axis {k}: {momentum_before:?} -> {momentum_after:?}"
        );
    }
    // The spin of the hitting grain is partly transferred to the target.
    assert!(sys.particles[1].angular_velocity[2].abs() > 1.0);
}

// ── Persistent tangential spring ─────────────────────────────────────────

#[test]
fn sheared_resting_pair_accumulates_its_tangential_spring() {
    let mut sys = zero_gravity_system(1.0e-5);
    let (a, mut b) = touching_pair(1.0e-3);
    b.velocity = [0.0, 0.1, 0.0];
    sys.add_particle(a);
    sys.add_particle(b);

    let mut norms = Vec::new();
    for _ in 0..10 {
        sys.step();
        assert_eq!(sys.contacts.len(), 1, "the contact must persist");
        norms.push(norm(sys.contacts[0].tangential_spring));
    }
    assert!(norms[0] > 0.0);
    for pair in norms.windows(2) {
        assert!(
            pair[1] > pair[0],
            "the shear displacement must keep growing: {norms:?}"
        );
    }
    // 10 steps of ~0.1 m/s * 1e-5 s, slightly reduced by the friction itself.
    assert!(
        norms[9] > 8.0e-6 && norms[9] < 1.0e-5 + 1e-12,
        "norms = {norms:?}"
    );
    // The friction drags the resting grain along with the sliding one.
    assert!(sys.particles[0].velocity[1] > 0.0);
    assert!(sys.particles[1].velocity[1] < 0.1);
}

#[test]
fn detect_contacts_projects_the_carried_spring_onto_the_new_tangent_plane() {
    let mut sys = zero_gravity_system(1.0e-5);
    sys.add_particle(grain(0, [0.0; 3]));
    sys.add_particle(grain(1, [0.099, 0.0, 0.0]));
    let mut previous = ContactPair::new(0, 1);
    previous.tangential_spring = [0.3, 0.2, -0.1];
    sys.contacts = vec![previous];
    sys.detect_contacts();
    assert_eq!(sys.contacts.len(), 1);
    let spring = sys.contacts[0].tangential_spring;
    assert!(spring[0].abs() < 1e-15, "normal component left: {spring:?}");
    assert!((spring[1] - 0.2).abs() < 1e-15 && (spring[2] + 0.1).abs() < 1e-15);

    // An oblique contact normal n = (1, 1, 0)/√2 projects (1, 0, 0) onto (½, -½, 0).
    let mut oblique = zero_gravity_system(1.0e-5);
    oblique.add_particle(grain(0, [0.0; 3]));
    oblique.add_particle(grain(1, [0.06, 0.06, 0.0]));
    let mut previous = ContactPair::new(0, 1);
    previous.tangential_spring = [1.0, 0.0, 0.0];
    oblique.contacts = vec![previous];
    oblique.detect_contacts();
    let spring = oblique.contacts[0].tangential_spring;
    assert!((spring[0] - 0.5).abs() < 1e-12, "{spring:?}");
    assert!((spring[1] + 0.5).abs() < 1e-12, "{spring:?}");
    assert!(spring[2].abs() < 1e-12);
}

#[test]
fn detect_contacts_drops_lost_contacts_and_starts_new_ones_at_zero() {
    let mut sys = zero_gravity_system(1.0e-5);
    sys.add_particle(grain(0, [0.0; 3]));
    sys.add_particle(grain(1, [0.5, 0.0, 0.0])); // far from 0
    sys.add_particle(grain(2, [0.0, 0.099, 0.0])); // newly touching 0
    let mut stale = ContactPair::new(0, 1);
    stale.tangential_spring = [0.0, 1.0e-5, 0.0];
    let mut unrelated = ContactPair::new(7, 9);
    unrelated.tangential_spring = [1.0, 1.0, 1.0];
    sys.contacts = vec![stale, unrelated];
    sys.detect_contacts();
    assert_eq!(sys.contacts.len(), 1);
    let c = &sys.contacts[0];
    assert_eq!((c.i, c.j), (0, 2));
    assert_eq!(c.tangential_spring, [0.0; 3]);
    assert!((c.overlap - 0.001).abs() < 1e-12);
}

#[test]
fn contact_list_stays_sorted_by_particle_pair() {
    let mut sys = zero_gravity_system(1.0e-5);
    for idx in 0..4 {
        sys.add_particle(grain(idx, [0.09 * idx as f64, 0.0, 0.0]));
    }
    sys.detect_contacts();
    sys.detect_contacts();
    let pairs: Vec<(usize, usize)> = sys.contacts.iter().map(|c| (c.i, c.j)).collect();
    assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
}

// ── Cohesion beyond contact ──────────────────────────────────────────────

/// Two grains with a surface gap of `gap` (negative = overlap) along x.
fn pair_with_gap(sys: &mut GranularSystem, gap: f64) {
    sys.add_particle(grain(0, [0.0; 3]));
    sys.add_particle(grain(1, [0.1 + gap, 0.0, 0.0]));
}

#[test]
fn capillary_bridge_range_follows_the_bridge_volume() {
    assert!((capillary_rupture_distance(1.0e-12) - 1.0e-4).abs() < 1e-16);
    assert!((capillary_rupture_distance(8.0e-9) - 2.0e-3).abs() < 1e-15);
    assert_eq!(capillary_rupture_distance(0.0), 0.0);
    assert_eq!(capillary_rupture_distance(-1.0), 0.0);
}

#[test]
fn capillary_bridge_keeps_separated_pairs_in_the_contact_list() {
    let (gamma, volume) = (0.072, 1.0e-12); // rupture distance 1e-4 m
    let mut sys = zero_gravity_system(1.0e-5);
    sys.capillary_params = Some((gamma, volume));
    pair_with_gap(&mut sys, 5.0e-5);
    sys.detect_contacts();
    assert_eq!(sys.contacts.len(), 1, "the bridge must be listed");
    assert!((sys.contacts[0].overlap + 5.0e-5).abs() < 1e-12);
    assert_eq!(sys.contacts[0].tangential_spring, [0.0; 3]);

    // The bridge pulls the grains together with the module's capillary force.
    sys.resolve_contacts();
    let r_star = 0.025;
    let expected = capillary_force(gamma, r_star, 5.0e-5, volume);
    let f0 = sys.particles[0].force;
    let f1 = sys.particles[1].force;
    assert!(expected > 1.0e-3, "expected = {expected}");
    assert!((f0[0] - expected).abs() < 1e-9 * expected, "f0 = {f0:?}");
    assert!((f1[0] + expected).abs() < 1e-9 * expected, "f1 = {f1:?}");
    assert!(norm(add(f0, f1)) < 1e-15);
    assert!((sys.contacts[0].fn_mag - expected).abs() < 1e-9 * expected);

    // Beyond the rupture distance the bridge is gone.
    let mut far = zero_gravity_system(1.0e-5);
    far.capillary_params = Some((gamma, volume));
    pair_with_gap(&mut far, 2.0e-4);
    far.detect_contacts();
    assert!(far.contacts.is_empty());
}

#[test]
fn capillary_bridge_pulls_separated_grains_together() {
    let mut sys = zero_gravity_system(1.0e-5);
    sys.capillary_params = Some((0.072, 1.0e-12));
    pair_with_gap(&mut sys, 5.0e-5);
    let gap_before = sys.particles[1].position[0] - sys.particles[0].position[0] - 0.1;
    for _ in 0..20 {
        sys.step();
    }
    let gap_after = sys.particles[1].position[0] - sys.particles[0].position[0] - 0.1;
    assert!(
        gap_after < gap_before,
        "the bridge must close the gap: {gap_before} -> {gap_after}"
    );
    assert!(sys.particles[0].velocity[0] > 0.0 && sys.particles[1].velocity[0] < 0.0);
}

#[test]
fn van_der_waals_attraction_acts_within_its_cutoff_only() {
    let hamaker = 1.0e-19;
    let mut sys = zero_gravity_system(1.0e-5);
    sys.hamaker = Some(hamaker);
    pair_with_gap(&mut sys, 5.0e-8);
    sys.detect_contacts();
    assert_eq!(sys.contacts.len(), 1);
    sys.resolve_contacts();
    let expected = van_der_waals_force(hamaker, 0.025, 5.0e-8, 1.0e-10);
    // The 5e-8 m gap is the difference of two numbers of size 0.1, hence the
    // looser relative tolerance.
    assert!((sys.particles[0].force[0] - expected).abs() < 1e-6 * expected);
    assert!((sys.particles[1].force[0] + expected).abs() < 1e-6 * expected);

    let mut far = zero_gravity_system(1.0e-5);
    far.hamaker = Some(hamaker);
    pair_with_gap(&mut far, 2.0 * VAN_DER_WAALS_CUTOFF_GAP);
    far.detect_contacts();
    assert!(far.contacts.is_empty());
}

#[test]
fn separated_pair_without_cohesion_is_not_a_contact() {
    let pi = grain(0, [0.0; 3]);
    let pj = grain(1, [0.1 + 1.0e-6, 0.0, 0.0]);
    assert!(resolve_hertz_mindlin(&pi, &pj, [0.0; 3], 1.0e-5, None, None).is_none());

    let mut sys = zero_gravity_system(1.0e-5);
    pair_with_gap(&mut sys, 1.0e-6);
    sys.detect_contacts();
    assert!(sys.contacts.is_empty());
}

#[test]
fn cohesion_is_cut_off_beyond_its_range_in_the_pair_force() {
    let pi = grain(0, [0.0; 3]);
    let near = grain(1, [0.1 + 5.0e-5, 0.0, 0.0]);
    let far = grain(2, [0.1 + 2.0e-4, 0.0, 0.0]);
    let bridge = Some((0.072, 1.0e-12));
    assert!(resolve_hertz_mindlin(&pi, &near, [0.0; 3], 1.0e-5, bridge, None).is_some());
    assert!(resolve_hertz_mindlin(&pi, &far, [0.0; 3], 1.0e-5, bridge, None).is_none());

    let near_vdw = grain(3, [0.1 + 5.0e-8, 0.0, 0.0]);
    let far_vdw = grain(4, [0.1 + 5.0e-7, 0.0, 0.0]);
    assert!(resolve_hertz_mindlin(&pi, &near_vdw, [0.0; 3], 1.0e-5, None, Some(1e-19)).is_some());
    assert!(resolve_hertz_mindlin(&pi, &far_vdw, [0.0; 3], 1.0e-5, None, Some(1e-19)).is_none());
}

#[test]
fn shear_history_is_released_when_the_surfaces_part_but_the_bridge_remains() {
    let mut sys = zero_gravity_system(1.0e-5);
    sys.capillary_params = Some((0.072, 1.0e-12));
    pair_with_gap(&mut sys, -1.0e-3);
    let mut sheared = ContactPair::new(0, 1);
    sheared.tangential_spring = [0.0, 3.0e-6, 0.0];
    sys.contacts = vec![sheared];
    sys.detect_contacts();
    assert!((sys.contacts[0].tangential_spring[1] - 3.0e-6).abs() < 1e-18);

    // The grains part but stay inside the bridge range: the pair remains listed
    // while its shear history is lost.
    sys.particles[1].position[0] = 0.1 + 5.0e-5;
    sys.detect_contacts();
    assert_eq!(sys.contacts.len(), 1);
    assert_eq!(sys.contacts[0].tangential_spring, [0.0; 3]);
    sys.resolve_contacts();
    assert_eq!(sys.contacts[0].tangential_spring, [0.0; 3]);

    // Touching again starts a fresh contact: the old spring does not return.
    sys.particles[1].position[0] = 0.1 - 1.0e-3;
    sys.detect_contacts();
    assert_eq!(sys.contacts[0].tangential_spring, [0.0; 3]);
    assert!(sys.contacts[0].overlap > 0.0);
}

#[test]
fn separated_branch_resets_a_stale_shear_spring() {
    let pi = grain(0, [0.0; 3]);
    let pj = grain(1, [0.1 + 5.0e-5, 0.0, 0.0]);
    let res = resolve_hertz_mindlin(
        &pi,
        &pj,
        [0.0, 1.0e-6, 0.0],
        1.0e-5,
        Some((0.072, 1.0e-12)),
        None,
    )
    .expect("the bridge is intact");
    assert_eq!(res.new_tangential_spring, [0.0; 3]);
    assert!(res.force_i[0] > 0.0, "the bridge attracts i towards j");
    assert!(norm(add(res.force_i, res.force_j)) < 1e-15);
}
