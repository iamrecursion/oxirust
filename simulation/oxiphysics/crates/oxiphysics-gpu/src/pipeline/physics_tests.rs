//! Behavioural tests of the sphere physics pipeline: sort-and-sweep broad phase,
//! sphere-sphere narrow phase, sequential-impulse solver and state rejection.

use crate::pipeline::{
    DEFAULT_BAUMGARTE, PhysicsPipeline, PipelineBuilder, PipelineConfig, PipelineError,
    PipelineStage, SphereContact, StateQuantity, WorldState,
};

const DT: f64 = 1.0 / 60.0;
const GRAVITY: f64 = 9.81;

/// Deterministic uniform samples in `[0, 1)` (64-bit LCG, top 53 bits).
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// A pipeline that only detects: broad phase and narrow phase, no solver and no
/// integration, so the bodies stay where they are.
fn detection_only() -> PhysicsPipeline {
    PipelineBuilder::new()
        .disable_stage(PipelineStage::ConstraintSolve)
        .disable_stage(PipelineStage::Integration)
        .disable_stage(PipelineStage::PostProcess)
        .build()
}

/// A dynamic unit-mass sphere at rest.
fn add_sphere(world: &mut WorldState, centre: [f64; 3], radius: f64) -> usize {
    world.add_body(centre, [0.0; 3], 1.0, radius)
}

/// Spheres of radius 1 spaced 1.5 apart along `direction`: only neighbours overlap.
fn chain_along(direction: [f64; 3], count: usize) -> WorldState {
    let mut world = WorldState::default();
    for i in 0..count {
        let step = 1.5 * i as f64;
        add_sphere(
            &mut world,
            [
                direction[0] * step,
                direction[1] * step,
                direction[2] * step,
            ],
            1.0,
        );
    }
    world
}

fn neighbour_pairs(count: usize) -> Vec<(usize, usize)> {
    (0..count - 1).map(|i| (i, i + 1)).collect()
}

/// O(n^2) reference: every pair of collidable bodies, not both immovable, whose
/// axis-aligned boxes (centre +- radius) overlap.
fn brute_force_pairs(world: &WorldState) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for a in 0..world.body_count() {
        for b in a + 1..world.body_count() {
            let (Some(ra), Some(rb)) = (world.collision_radius(a), world.collision_radius(b))
            else {
                continue;
            };
            let (Some(pa), Some(pb)) = (world.position(a), world.position(b)) else {
                continue;
            };
            if !(world.is_dynamic(a) || world.is_dynamic(b)) {
                continue;
            }
            let overlap = (0..3).all(|k| pa[k] - ra <= pb[k] + rb && pb[k] - rb <= pa[k] + ra);
            if overlap {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

/// O(n^2) reference: `(a, b, depth)` of every pair of spheres that penetrate.
fn brute_force_contacts(world: &WorldState) -> Vec<(usize, usize, f64)> {
    brute_force_pairs(world)
        .into_iter()
        .filter_map(|(a, b)| {
            let (pa, pb) = (world.position(a)?, world.position(b)?);
            let reach = world.collision_radius(a)? + world.collision_radius(b)?;
            let distance = (0..3).map(|k| (pb[k] - pa[k]).powi(2)).sum::<f64>().sqrt();
            (distance < reach).then_some((a, b, reach - distance))
        })
        .collect()
}

/// Random cloud of spheres (some immovable, some without a collision shape).
fn random_scene(seed: u64, bodies: usize, extent: f64) -> WorldState {
    let mut rng = seed;
    let mut world = WorldState::default();
    for body in 0..bodies {
        let centre = [
            extent * next_unit(&mut rng),
            extent * next_unit(&mut rng),
            extent * next_unit(&mut rng),
        ];
        let radius = if body % 17 == 5 {
            0.0
        } else {
            0.4 + 0.8 * next_unit(&mut rng)
        };
        let inverse_mass = if body % 5 == 0 {
            0.0
        } else {
            0.5 + next_unit(&mut rng)
        };
        world.add_body(centre, [0.0; 3], inverse_mass, radius);
    }
    world
}

/// Two spheres of radius 0.5 on the x axis, `depth` deep inside each other, flying
/// at each other with the given x velocities.  Returns the world after one step of
/// `dt` and the pipeline that ran it.
fn head_on(
    restitution: f64,
    depth: f64,
    velocities: (f64, f64),
    inverse_masses: (f64, f64),
    dt: f64,
) -> (WorldState, PhysicsPipeline) {
    let half_gap = 0.5 * (1.0 - depth);
    let mut world = WorldState::default();
    world.add_body(
        [-half_gap, 0.0, 0.0],
        [velocities.0, 0.0, 0.0],
        inverse_masses.0,
        0.5,
    );
    world.add_body(
        [half_gap, 0.0, 0.0],
        [velocities.1, 0.0, 0.0],
        inverse_masses.1,
        0.5,
    );
    let mut pipeline = PipelineBuilder::new().restitution(restitution).build();
    pipeline
        .step(&mut world, dt)
        .expect("a head-on collision is a valid step");
    (world, pipeline)
}

// ---------------------------------------------------------------------------
// Broad phase and narrow phase
// ---------------------------------------------------------------------------

#[test]
fn two_separated_bodies_give_zero_pairs() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [10.0, 0.0, 0.0], 1.0);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 0);
    assert_eq!(stats.contacts, 0);
    assert_eq!(stats.solved_constraints, 0);
    assert!(pipeline.pairs().is_empty());
    assert!(pipeline.contacts().is_empty());
}

#[test]
fn one_overlapping_pair_among_three_bodies_is_not_three_pairs() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [1.5, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [50.0, 0.0, 0.0], 1.0);
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(pipeline.pairs(), &[(0, 1)]);
    assert_eq!(stats.collision_pairs, 1);
    assert_eq!(stats.contacts, 1);
}

#[test]
fn overlapping_pair_reports_normal_depth_and_contact_point() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [1.5, 0.0, 0.0], 1.0);
    let mut pipeline = detection_only();
    pipeline.step(&mut world, DT).expect("valid step");
    let contacts = pipeline.contacts();
    assert_eq!(contacts.len(), 1);
    let contact = contacts[0];
    assert_eq!((contact.body_a, contact.body_b), (0, 1));
    assert_eq!(contact.normal, [1.0, 0.0, 0.0]);
    assert!((contact.depth - 0.5).abs() < 1e-12);
    // The overlap spans x in [0.5, 1.0]; the contact point is its middle.
    assert!((contact.point[0] - 0.75).abs() < 1e-12);
    assert_eq!(contact.normal_impulse, 0.0);
}

#[test]
fn contact_normal_points_from_the_lower_to_the_higher_body_index() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 3.0, 0.0], 1.0);
    add_sphere(&mut world, [0.0, 1.0, 1.0], 1.5);
    let mut pipeline = detection_only();
    pipeline.step(&mut world, DT).expect("valid step");
    let contact = pipeline.contacts()[0];
    let offset: [f64; 3] = [0.0, -2.0, 1.0];
    let length = (offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
    for (component, expected) in contact.normal.iter().zip(offset) {
        assert!((component - expected / length).abs() < 1e-12);
    }
    assert!((contact.depth - (2.5 - length)).abs() < 1e-12);
}

#[test]
fn coincident_centres_use_the_fallback_normal() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [2.0, 2.0, 2.0], 1.0);
    add_sphere(&mut world, [2.0, 2.0, 2.0], 0.5);
    let mut pipeline = detection_only();
    pipeline.step(&mut world, DT).expect("valid step");
    let contact = pipeline.contacts()[0];
    assert_eq!(contact.normal, [1.0, 0.0, 0.0]);
    assert!((contact.depth - 1.5).abs() < 1e-12);
}

#[test]
fn box_overlap_without_sphere_contact_is_rejected_by_the_narrow_phase() {
    // The boxes overlap in the corner region, the spheres are 2.69 apart (> 2).
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [1.9, 1.9, 0.0], 1.0);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 1);
    assert_eq!(stats.contacts, 0);
    assert_eq!(stats.solved_constraints, 0);
    assert_eq!(pipeline.pairs(), &[(0, 1)]);
    assert!(pipeline.contacts().is_empty());
}

#[test]
fn touching_spheres_are_a_pair_but_not_a_contact() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [2.0, 0.0, 0.0], 1.0);
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 1);
    assert_eq!(stats.contacts, 0);
}

#[test]
fn a_chain_reports_only_the_neighbouring_pairs() {
    // Five spheres 1.5 apart: 4 neighbouring pairs, not 5 * 4 / 2 = 10.
    let mut world = chain_along([1.0, 0.0, 0.0], 5);
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(pipeline.pairs(), neighbour_pairs(5).as_slice());
    assert_eq!(stats.collision_pairs, 4);
    assert_eq!(stats.contacts, 4);
}

#[test]
fn pairs_do_not_depend_on_the_axis_the_bodies_are_spread_along() {
    for direction in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.0, 0.8],
        [0.0, 0.6, -0.8],
    ] {
        for use_gpu in [false, true] {
            let mut world = chain_along(direction, 24);
            let mut pipeline = detection_only();
            pipeline.config.use_gpu = use_gpu;
            pipeline.step(&mut world, DT).expect("valid step");
            assert_eq!(
                pipeline.pairs(),
                neighbour_pairs(24).as_slice(),
                "direction {direction:?}, use_gpu {use_gpu}"
            );
        }
    }
}

#[test]
fn a_pile_of_overlapping_bodies_reports_every_pair() {
    // Eight spheres around the same point all overlap each other: 8 * 7 / 2 pairs.
    let mut world = WorldState::default();
    for i in 0..8 {
        let offset = 0.01 * i as f64;
        add_sphere(&mut world, [offset, -offset, 0.5 * offset], 1.0);
    }
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 28);
    assert_eq!(stats.contacts, 28);
}

#[test]
fn sort_and_sweep_matches_the_brute_force_oracle_on_random_scenes() {
    for seed in 1..=8u64 {
        let world = random_scene(0xC0FF_EE00 + seed, 160, 10.0);
        let expected_pairs = brute_force_pairs(&world);
        let expected_contacts = brute_force_contacts(&world);
        let all_pairs = world.body_count() * (world.body_count() - 1) / 2;
        assert!(
            expected_pairs.len() >= 20 && expected_pairs.len() * 10 < all_pairs,
            "seed {seed}: the scene must be neither empty nor dense ({} pairs)",
            expected_pairs.len()
        );
        assert!(
            expected_contacts.len() >= 5 && expected_contacts.len() < expected_pairs.len(),
            "seed {seed}: some box overlaps must fail the sphere test ({} of {})",
            expected_contacts.len(),
            expected_pairs.len()
        );

        let mut cpu_world = world.clone();
        let mut cpu = detection_only();
        let cpu_stats = cpu.step(&mut cpu_world, DT).expect("valid step");
        assert_eq!(cpu.pairs(), expected_pairs.as_slice(), "seed {seed}");
        assert_eq!(cpu_stats.collision_pairs as usize, expected_pairs.len());
        assert_eq!(cpu_stats.contacts as usize, expected_contacts.len());
        for (contact, &(a, b, depth)) in cpu.contacts().iter().zip(&expected_contacts) {
            assert_eq!((contact.body_a, contact.body_b), (a, b), "seed {seed}");
            assert!((contact.depth - depth).abs() < 1e-12, "seed {seed}");
        }

        let mut gpu_world = world.clone();
        let mut gpu = detection_only();
        gpu.config.use_gpu = true;
        gpu.step(&mut gpu_world, DT).expect("valid step");
        assert_eq!(gpu.pairs(), cpu.pairs(), "seed {seed}");
        assert_eq!(gpu.contacts(), cpu.contacts(), "seed {seed}");
    }
}

#[test]
fn the_gpu_sort_order_is_exact_for_bounds_that_collide_in_f32() {
    // Bounds that are different as f64 but equal after rounding to f32 must keep
    // their f64 order, or the sweep would end early and lose pairs.
    let base = 1.0e8;
    let mut world = WorldState::default();
    for i in 0..40 {
        // All of 1e8 + k * 1e-3, k < 40, round to one f32 value (the f32 spacing at
        // 1e8 is 8).  The bodies are stored in a shuffled order, so a sort that
        // stops at the f32 key would leave them in the wrong f64 order.
        let k = (i * 17) % 40;
        add_sphere(&mut world, [base + 1.0e-3 * k as f64, 0.0, 0.0], 1.0e-3);
    }
    let expected = brute_force_pairs(&world);
    let mut cpu_world = world.clone();
    let mut cpu = detection_only();
    cpu.step(&mut cpu_world, DT).expect("valid step");
    assert_eq!(cpu.pairs(), expected.as_slice());
    let mut gpu_world = world;
    let mut gpu = detection_only();
    gpu.config.use_gpu = true;
    gpu.step(&mut gpu_world, DT).expect("valid step");
    assert_eq!(gpu.pairs(), expected.as_slice());
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn use_gpu_routes_the_sort_through_the_gpu_radix_sort() {
    use crate::gpu_primitives::shared;
    if shared::shared_backend().is_none() {
        eprintln!("SKIPPED: no GPU adapter");
        return;
    }
    let mut world = random_scene(7, 300, 12.0);
    let mut pipeline = detection_only();
    pipeline.config.use_gpu = true;
    let runs_before = shared::gpu_runs();
    pipeline.step(&mut world, DT).expect("valid step");
    // The 64-bit sweep key is sorted in two 32-bit radix passes.
    assert!(
        shared::gpu_runs() >= runs_before + 2,
        "the broad phase must have run two GPU radix sorts"
    );
    assert_eq!(pipeline.pairs(), brute_force_pairs(&world).as_slice());
}

#[test]
fn pairs_of_immovable_bodies_are_not_reported() {
    let mut world = WorldState::default();
    world.add_body([0.0; 3], [0.0; 3], 0.0, 1.0);
    world.add_body([1.0, 0.0, 0.0], [0.0; 3], 0.0, 1.0);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 0);
    assert_eq!(stats.solved_constraints, 0);

    // An immovable body against a movable one is a pair.
    world.add_body([0.5, 0.5, 0.0], [0.0; 3], 1.0, 1.0);
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(pipeline.pairs(), &[(0, 2), (1, 2)]);
    assert_eq!(stats.contacts, 2);
}

#[test]
fn bodies_without_a_collision_shape_never_collide() {
    // Without a radii array nothing has a shape, even if all bodies coincide.
    let mut world = WorldState {
        positions: vec![0.0; 9],
        velocities: vec![0.0; 9],
        inverse_masses: vec![1.0; 3],
        radii: Vec::new(),
    };
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 0);

    // Zero and negative radii are shapeless as well.
    world.radii = vec![0.0, -1.0, 2.0];
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 0);
    assert_eq!(world.collision_radius(0), None);
    assert_eq!(world.collision_radius(1), None);
    assert_eq!(world.collision_radius(2), Some(2.0));
}

#[test]
fn bodies_with_a_non_finite_position_are_left_out_of_the_broad_phase() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [f64::NAN, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [f64::INFINITY, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [0.5, 0.0, 0.0], 1.0);
    // Post-processing is disabled in `detection_only`, so nothing rejects the NaN.
    let mut pipeline = detection_only();
    let stats = pipeline.step(&mut world, DT).expect("no state validation");
    assert_eq!(pipeline.pairs(), &[(0, 3)]);
    assert_eq!(stats.contacts, 1);
}

#[test]
fn disabled_upstream_stages_leave_nothing_for_the_stages_after_them() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [1.0, 0.0, 0.0], 1.0);

    let mut no_broadphase = PipelineBuilder::new()
        .disable_stage(PipelineStage::BroadPhase)
        .build();
    let stats = no_broadphase.step(&mut world.clone(), DT).expect("valid");
    assert_eq!(stats.collision_pairs, 0);
    assert_eq!(stats.contacts, 0);
    assert_eq!(stats.solved_constraints, 0);

    let mut no_narrowphase = PipelineBuilder::new()
        .disable_stage(PipelineStage::NarrowPhase)
        .build();
    let stats = no_narrowphase.step(&mut world.clone(), DT).expect("valid");
    assert_eq!(stats.collision_pairs, 1);
    assert_eq!(stats.contacts, 0);
    assert_eq!(stats.solved_constraints, 0);
}

#[test]
fn substeps_run_the_collision_stages_every_substep() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    add_sphere(&mut world, [1.5, 0.0, 0.0], 1.0);
    let mut pipeline = PipelineBuilder::new()
        .substeps(3)
        .disable_stage(PipelineStage::ConstraintSolve)
        .disable_stage(PipelineStage::Integration)
        .build();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(stats.collision_pairs, 3);
    assert_eq!(stats.contacts, 3);
    pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(pipeline.stats.collision_pairs, 6);
    assert_eq!(pipeline.stats.contacts, 6);
}

// ---------------------------------------------------------------------------
// Constraint solver
// ---------------------------------------------------------------------------

#[test]
fn elastic_head_on_collision_of_equal_masses_swaps_the_velocities() {
    let (world, pipeline) = head_on(1.0, 0.02, (2.0, -2.0), (1.0, 1.0), 1.0e-3);
    assert!((world.velocities[0] + 2.0).abs() < 1e-12);
    assert!((world.velocities[3] - 2.0).abs() < 1e-12);
    // Gravity only touches the y component.
    assert!((world.velocities[1] + GRAVITY * 1.0e-3).abs() < 1e-12);
    assert!((world.velocities[4] + GRAVITY * 1.0e-3).abs() < 1e-12);
    let contact = pipeline.contacts()[0];
    assert!((contact.normal_impulse - 4.0).abs() < 1e-12);
}

#[test]
fn inelastic_head_on_collision_removes_the_relative_normal_velocity() {
    // 1e-4 deep is inside the slop, so there is no push-out either.
    let (world, pipeline) = head_on(0.0, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3);
    assert!(world.velocities[0].abs() < 1e-12);
    assert!(world.velocities[3].abs() < 1e-12);
    let contact = pipeline.contacts()[0];
    assert!((contact.normal_impulse - 2.0).abs() < 1e-12);
}

#[test]
fn unequal_masses_conserve_momentum_and_follow_the_restitution_law() {
    // Masses 1 and 4; velocities 3 and -1: momentum -1, closing speed 4.
    let (world, _) = head_on(0.5, 1.0e-4, (3.0, -1.0), (1.0, 0.25), 1.0e-3);
    let (va, vb) = (world.velocities[0], world.velocities[3]);
    assert!(
        (1.0 * va + 4.0 * vb + 1.0).abs() < 1e-12,
        "momentum {va} {vb}"
    );
    // Separation speed = restitution * closing speed = 0.5 * 4.
    assert!((vb - va - 2.0).abs() < 1e-12, "separation {}", vb - va);
    assert!((va + 1.8).abs() < 1e-12);
    assert!((vb - 0.2).abs() < 1e-12);
}

#[test]
fn a_light_body_is_dragged_along_by_a_heavy_one() {
    // Body 0 is a million times lighter than body 1, which flies towards it.
    let (world, _) = head_on(0.0, 1.0e-4, (0.0, -1.0), (1.0e6, 1.0), 1.0e-3);
    let (light, heavy) = (world.velocities[0], world.velocities[3]);
    assert!((heavy + 1.0).abs() < 2.0e-6, "heavy body slowed to {heavy}");
    assert!((light + 1.0).abs() < 2.0e-6, "light body moves at {light}");
}

#[test]
fn more_solver_iterations_propagate_the_impulse_along_a_chain() {
    // Three equal spheres in a row, barely touching; the first flies at the others
    // with 3 m/s and the contacts are inelastic.  Momentum 3 ends up as 1 each.
    let run = |iterations: u32| {
        let mut world = WorldState::default();
        world.add_body([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0, 0.5);
        world.add_body([0.99999, 0.0, 0.0], [0.0; 3], 1.0, 0.5);
        world.add_body([1.99998, 0.0, 0.0], [0.0; 3], 1.0, 0.5);
        let mut pipeline = PipelineBuilder::new().solver_iterations(iterations).build();
        pipeline.step(&mut world, 1.0e-3).expect("valid step");
        [
            world.velocities[0],
            world.velocities[3],
            world.velocities[6],
        ]
    };
    // One Gauss-Seidel sweep solves the contacts in order and leaves the first one
    // still closing (v1 - v0 = -0.75).
    let single = run(1);
    for (speed, expected) in single.iter().zip([1.5, 0.75, 0.75]) {
        assert!((speed - expected).abs() < 1e-12, "{single:?}");
    }
    // Zero iterations are treated as one.
    assert_eq!(run(0), single);
    let converged = run(64);
    for speed in converged {
        assert!((speed - 1.0).abs() < 1e-6, "{converged:?}");
    }
    assert!((converged.iter().sum::<f64>() - 3.0).abs() < 1e-12);
}

#[test]
fn builder_sets_the_solver_parameters() {
    let pipeline = PipelineBuilder::new()
        .solver_iterations(3)
        .restitution(0.25)
        .baumgarte(0.5)
        .use_gpu(true)
        .build();
    assert_eq!(pipeline.config.solver_iterations, 3);
    assert_eq!(pipeline.config.restitution, 0.25);
    assert_eq!(pipeline.config.baumgarte, 0.5);
    assert!(pipeline.config.use_gpu);
}

#[test]
fn separating_bodies_are_not_pulled_back_together() {
    // The spheres overlap but fly apart fast: the contact must not apply any
    // (negative) impulse.
    let (world, pipeline) = head_on(0.0, 1.0e-4, (-3.0, 3.0), (1.0, 1.0), 1.0e-3);
    assert_eq!(world.velocities[0], -3.0);
    assert_eq!(world.velocities[3], 3.0);
    assert_eq!(pipeline.contacts()[0].normal_impulse, 0.0);
}

#[test]
fn resting_speed_contacts_do_not_bounce() {
    // Closing at 0.01 m/s is far below two steps of gravity: no rebound even for a
    // perfectly elastic contact.
    let (world, _) = head_on(1.0, 1.0e-4, (0.005, -0.005), (1.0, 1.0), DT);
    assert!(world.velocities[0].abs() < 1e-12);
    assert!(world.velocities[3].abs() < 1e-12);
}

#[test]
fn restitution_and_baumgarte_are_sanitised() {
    let reference = head_on(0.0, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3).0;
    let nan = head_on(f64::NAN, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3).0;
    assert_eq!(nan.velocities, reference.velocities);
    let negative = head_on(-3.0, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3).0;
    assert_eq!(negative.velocities, reference.velocities);
    let elastic = head_on(1.0, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3).0;
    let too_elastic = head_on(7.0, 1.0e-4, (2.0, -2.0), (1.0, 1.0), 1.0e-3).0;
    assert_eq!(too_elastic.velocities, elastic.velocities);

    // A NaN Baumgarte factor means no push-out: the deep overlap is left alone.
    let mut config = PipelineConfig::new();
    config.baumgarte = f64::NAN;
    let mut pipeline = PhysicsPipeline::new(config);
    let mut world = WorldState::default();
    world.add_body([-0.4, 0.0, 0.0], [0.0; 3], 1.0, 0.5);
    world.add_body([0.4, 0.0, 0.0], [0.0; 3], 1.0, 0.5);
    pipeline.step(&mut world, DT).expect("valid step");
    assert_eq!(world.velocities[0], 0.0);
    assert_eq!(world.velocities[3], 0.0);
}

#[test]
fn baumgarte_pushes_a_deep_overlap_apart_at_a_bounded_speed() {
    // Two radius-0.5 spheres 0.3 deep in each other, at rest: the push-out speed
    // is baumgarte / dt * min(depth - slop, 0.2 * radius) = 0.2 * 60 * 0.1 = 1.2.
    let (world, _) = head_on(0.0, 0.3, (0.0, 0.0), (1.0, 1.0), DT);
    assert!((world.velocities[3] - world.velocities[0] - 1.2).abs() < 1e-9);
    assert!((world.velocities[0] + 0.6).abs() < 1e-9);
    assert!((world.velocities[3] - 0.6).abs() < 1e-9);
}

#[test]
fn dynamic_bodies_conserve_momentum_in_a_multi_body_pile() {
    let mut rng = 0xBEEF_u64;
    let mut world = WorldState::default();
    for _ in 0..40 {
        let centre = [
            3.0 * next_unit(&mut rng),
            3.0 * next_unit(&mut rng),
            3.0 * next_unit(&mut rng),
        ];
        let velocity = [
            2.0 * next_unit(&mut rng) - 1.0,
            2.0 * next_unit(&mut rng) - 1.0,
            2.0 * next_unit(&mut rng) - 1.0,
        ];
        world.add_body(centre, velocity, 0.25 + 2.0 * next_unit(&mut rng), 0.5);
    }
    let momentum = |w: &WorldState| {
        let mut total = [0.0f64; 3];
        for body in 0..w.body_count() {
            let mass = 1.0 / w.inverse_mass(body);
            let velocity = w.velocity(body).expect("finite");
            for (sum, component) in total.iter_mut().zip(velocity) {
                *sum += mass * component;
            }
        }
        total
    };
    let total_mass: f64 = (0..world.body_count())
        .map(|body| 1.0 / world.inverse_mass(body))
        .sum();
    let before = momentum(&world);
    let mut pipeline = PipelineBuilder::new()
        .restitution(0.3)
        .solver_iterations(12)
        .build();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert!(stats.solved_constraints >= 20, "{stats:?}");
    let after = momentum(&world);
    // Contact impulses are equal and opposite; only gravity changes the momentum.
    assert!((after[0] - before[0]).abs() < 1e-9);
    assert!((after[2] - before[2]).abs() < 1e-9);
    assert!((after[1] - (before[1] - GRAVITY * DT * total_mass)).abs() < 1e-9);
}

#[test]
fn contact_impulses_are_never_negative() {
    let mut world = random_scene(99, 120, 6.0);
    for body in 0..world.body_count() {
        let angle = body as f64;
        world.velocities[3 * body] = angle.sin();
        world.velocities[3 * body + 1] = angle.cos();
    }
    let mut pipeline = PipelineBuilder::new().restitution(0.4).build();
    let stats = pipeline.step(&mut world, DT).expect("valid step");
    assert!(stats.contacts > 10);
    assert!(pipeline.contacts().iter().all(|c| c.normal_impulse >= 0.0));
    assert!(pipeline.contacts().iter().any(|c| c.normal_impulse > 0.0));
}

#[test]
fn zero_time_step_changes_no_position_and_stays_finite() {
    let mut world = WorldState::default();
    add_sphere(&mut world, [0.0, 0.0, 0.0], 1.0);
    world.add_body([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 1.0, 1.0);
    let positions = world.positions.clone();
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let stats = pipeline.step(&mut world, 0.0).expect("dt = 0 is allowed");
    assert_eq!(stats.contacts, 1);
    assert_eq!(world.positions, positions);
    assert!(world.velocities.iter().all(|v| v.is_finite()));
}

/// Ball of radius 0.5 above a huge static sphere whose top is at y = 0.
fn ball_over_ground(height: f64) -> WorldState {
    let mut world = WorldState::default();
    world.add_body([0.0, -1000.0, 0.0], [0.0; 3], 0.0, 1000.0);
    world.add_body([0.0, height, 0.0], [0.0; 3], 1.0, 0.5);
    world
}

#[test]
fn a_falling_ball_does_not_tunnel_through_a_static_sphere() {
    let mut world = ball_over_ground(3.0);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let mut lowest = f64::INFINITY;
    let mut touched = 0;
    for _ in 0..120 {
        let stats = pipeline.step(&mut world, DT).expect("valid step");
        lowest = lowest.min(world.positions[4]);
        touched += stats.solved_constraints;
    }
    assert!(touched > 0, "the ball never reached the ground");
    // At 7 m/s one step is 0.12 m; the ball must never sink deeper than that.
    assert!(lowest > 0.5 - 0.2, "lowest centre {lowest}");
}

#[test]
fn a_resting_ball_settles_at_the_baumgarte_equilibrium_depth() {
    let mut world = ball_over_ground(2.0);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    for _ in 0..600 {
        pipeline.step(&mut world, DT).expect("valid step");
    }
    // Integration runs after the solver, so the ball sinks g dt^2 per step and the
    // solver pushes out at baumgarte / dt * (depth - slop): they balance at
    // depth = slop + g dt^2 / baumgarte.
    let slop = 0.01 * 0.5;
    let equilibrium_depth = slop + GRAVITY * DT * DT / DEFAULT_BAUMGARTE;
    let depth = 0.5 - world.positions[4];
    assert!(
        (depth - equilibrium_depth).abs() < 1e-6,
        "depth {depth}, expected {equilibrium_depth}"
    );
    assert!(world.velocities[4].abs() < 1e-6, "{}", world.velocities[4]);
    assert_eq!(world.positions[3], 0.0);
    assert_eq!(world.positions[5], 0.0);
}

#[test]
fn a_dropped_ball_rebounds_with_the_restitution_speed() {
    let mut world = ball_over_ground(1.5);
    let mut pipeline = PipelineBuilder::new().restitution(0.5).build();
    let mut rebounded = false;
    for _ in 0..200 {
        let speed_before = world.velocities[4];
        let stats = pipeline.step(&mut world, DT).expect("valid step");
        if stats.contacts == 0 {
            continue;
        }
        // The solver turns the closing speed v into e * v; integration then adds
        // one step of gravity.
        let rebound_speed = world.velocities[4] + GRAVITY * DT;
        assert!(speed_before < -2.0, "first contact at {speed_before} m/s");
        assert!(
            (rebound_speed - 0.5 * -speed_before).abs() < 1e-9,
            "rebound {rebound_speed} for impact {speed_before}"
        );
        rebounded = true;
        break;
    }
    assert!(rebounded, "the ball never hit the ground");
}

#[test]
fn a_ball_stack_on_the_ground_keeps_its_order_and_does_not_collapse() {
    let mut world = ball_over_ground(0.5);
    add_sphere(&mut world, [0.0, 1.5, 0.0], 0.5);
    add_sphere(&mut world, [0.0, 2.5, 0.0], 0.5);
    let mut pipeline = PipelineBuilder::new().solver_iterations(16).build();
    for _ in 0..300 {
        pipeline.step(&mut world, DT).expect("valid step");
    }
    let heights: Vec<f64> = [1usize, 2, 3]
        .iter()
        .map(|&b| world.positions[3 * b + 1])
        .collect();
    for (lower, upper) in heights.iter().zip(heights.iter().skip(1)) {
        let gap = upper - lower;
        assert!(
            gap > 0.85 && gap < 1.05,
            "neighbouring balls {lower} and {upper} are {gap} apart"
        );
    }
    // The bottom ball rests on the ground within a few centimetres of penetration.
    assert!(heights[0] > 0.45 && heights[0] < 0.5 + 1e-9, "{heights:?}");
    for body in [1usize, 2, 3] {
        assert!(world.positions[3 * body].abs() < 1e-9);
        assert!(world.positions[3 * body + 2].abs() < 1e-9);
        assert!(world.velocities[3 * body + 1].abs() < 0.05);
    }
}

#[test]
fn identical_runs_give_identical_results() {
    let scene = random_scene(2024, 100, 6.0);
    let run = |use_gpu: bool| {
        let mut world = scene.clone();
        let mut pipeline = PipelineBuilder::new()
            .restitution(0.2)
            .use_gpu(use_gpu)
            .substeps(2)
            .build();
        for _ in 0..10 {
            pipeline.step(&mut world, DT).expect("valid step");
        }
        let contacts: Vec<SphereContact> = pipeline.contacts().to_vec();
        (world, contacts)
    };
    let first = run(false);
    assert_eq!(first, run(false));
    // The GPU sort changes nothing about the simulation.
    assert_eq!(first, run(true));
}

// ---------------------------------------------------------------------------
// Rejection of invalid input and state
// ---------------------------------------------------------------------------

/// Damages one quantity of a [`WorldState`].
type Corruption = fn(&mut WorldState);

fn two_bodies() -> WorldState {
    let mut world = WorldState::default();
    world.add_body([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.5);
    world.add_body([4.0, 5.0, 0.0], [-1.0, 0.0, 0.0], 2.0, 0.5);
    world
}

#[test]
fn a_step_that_produces_a_non_finite_position_is_rejected_and_rolled_back() {
    let mut world = two_bodies();
    world.velocities[3] = 1.0e308;
    let before = world.clone();
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let error = pipeline
        .step(&mut world, 10.0)
        .expect_err("the position overflows");
    assert_eq!(
        error,
        PipelineError::NonFiniteState {
            body: 1,
            quantity: StateQuantity::Position
        }
    );
    assert_eq!(
        world, before,
        "the world must be exactly as before the step"
    );
    assert_eq!(pipeline.stats.total_time_ms, 0.0);
    assert_eq!(pipeline.stats.collision_pairs, 0);
    assert!(pipeline.pairs().is_empty());
    assert!(pipeline.contacts().is_empty());
    assert_eq!(error.to_string(), "body 1 has a non-finite position");
}

#[test]
fn a_rejection_in_a_late_substep_rolls_back_all_substeps() {
    let mut world = two_bodies();
    // Each of the 4 sub-steps moves body 0 by 6e307: finite after two, infinite
    // after three.
    world.velocities[0] = 6.0e307;
    let before = world.clone();
    let mut pipeline = PipelineBuilder::new().substeps(4).build();
    let error = pipeline
        .step(&mut world, 4.0)
        .expect_err("overflow in substep 3");
    assert_eq!(
        error,
        PipelineError::NonFiniteState {
            body: 0,
            quantity: StateQuantity::Position
        }
    );
    assert_eq!(world, before);
}

#[test]
fn non_finite_input_state_is_rejected_per_quantity() {
    let cases: [(Corruption, PipelineError); 4] = [
        (
            |w| w.positions[4] = f64::NAN,
            PipelineError::NonFiniteState {
                body: 1,
                quantity: StateQuantity::Position,
            },
        ),
        (
            |w| {
                // An immovable body is not integrated, so its position stays finite.
                w.inverse_masses[0] = 0.0;
                w.velocities[0] = f64::INFINITY;
            },
            PipelineError::NonFiniteState {
                body: 0,
                quantity: StateQuantity::Velocity,
            },
        ),
        (
            |w| w.radii[1] = f64::NAN,
            PipelineError::NonFiniteState {
                body: 1,
                quantity: StateQuantity::Radius,
            },
        ),
        (
            |w| w.inverse_masses[0] = f64::NAN,
            PipelineError::NonFiniteState {
                body: 0,
                quantity: StateQuantity::InverseMass,
            },
        ),
    ];
    for (corrupt, expected) in cases {
        let mut world = two_bodies();
        corrupt(&mut world);
        let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
        let error = pipeline
            .step(&mut world, DT)
            .expect_err("invalid state must be rejected");
        assert_eq!(error, expected);
    }
}

#[test]
fn a_negative_inverse_mass_is_rejected() {
    let mut world = two_bodies();
    world.inverse_masses[1] = -0.5;
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    let error = pipeline
        .step(&mut world, DT)
        .expect_err("negative inverse mass");
    assert_eq!(
        error,
        PipelineError::NegativeInverseMass {
            body: 1,
            value: -0.5
        }
    );
    assert_eq!(world.inverse_masses[1], -0.5);
}

#[test]
fn nan_input_does_not_panic_or_spread_to_other_bodies() {
    let mut world = two_bodies();
    world.positions[0] = f64::NAN;
    // Bodies 1 and 2 overlap, so the solver changes their velocities before the
    // post-process stage rejects the NaN of body 0; the rollback must undo that.
    world.add_body([4.2, 5.0, 0.0], [0.0; 3], 1.0, 0.5);
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
    assert!(pipeline.step(&mut world, DT).is_err());
    assert_eq!(world.positions[6..], [4.2, 5.0, 0.0]);
    assert_eq!(world.velocities[6..], [0.0, 0.0, 0.0]);
    assert!(world.positions[3..6].iter().all(|v| v.is_finite()));
}

#[test]
fn the_post_process_stage_decides_whether_state_is_validated() {
    let mut world = two_bodies();
    world.positions[0] = f64::NAN;
    let mut pipeline = PipelineBuilder::new()
        .disable_stage(PipelineStage::PostProcess)
        .build();
    pipeline
        .step(&mut world, DT)
        .expect("without post-processing nothing is validated");
    assert!(world.positions[0].is_nan());
}

#[test]
fn invalid_time_steps_are_rejected_before_anything_changes() {
    for dt in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -DT] {
        let mut world = two_bodies();
        let before = world.clone();
        let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());
        let error = pipeline.step(&mut world, dt).expect_err("invalid dt");
        assert!(matches!(error, PipelineError::InvalidTimeStep { .. }));
        assert_eq!(world, before);
    }
}

#[test]
fn inconsistent_array_lengths_are_rejected() {
    let mut pipeline = PhysicsPipeline::new(PipelineConfig::new());

    let mut world = two_bodies();
    world.positions.pop();
    assert_eq!(
        pipeline.step(&mut world, DT).expect_err("short positions"),
        PipelineError::LayoutMismatch {
            array: "positions",
            expected: 6,
            actual: 5,
            bodies: 2
        }
    );

    let mut world = two_bodies();
    world.velocities.extend_from_slice(&[0.0; 3]);
    assert_eq!(
        pipeline.step(&mut world, DT).expect_err("long velocities"),
        PipelineError::LayoutMismatch {
            array: "velocities",
            expected: 6,
            actual: 9,
            bodies: 2
        }
    );

    let mut world = two_bodies();
    world.radii.pop();
    assert_eq!(
        pipeline.step(&mut world, DT).expect_err("short radii"),
        PipelineError::LayoutMismatch {
            array: "radii",
            expected: 2,
            actual: 1,
            bodies: 2
        }
    );
}

#[test]
fn world_state_helpers_describe_the_bodies() {
    let mut world = WorldState::default();
    let first = world.add_body([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.5, 0.25);
    // Appending to a world whose radii array is shorter pads it with shapeless entries.
    world.radii.clear();
    let second = world.add_body([4.0, 5.0, 6.0], [0.0; 3], 0.0, 2.0);
    assert_eq!((first, second), (0, 1));
    assert_eq!(world.radii, vec![0.0, 2.0]);
    assert_eq!(world.validate_layout(), Ok(()));
    assert_eq!(world.position(0), Some([1.0, 2.0, 3.0]));
    assert_eq!(world.velocity(0), Some([0.1, 0.2, 0.3]));
    assert_eq!(world.position(2), None);
    assert!(world.is_dynamic(0));
    assert!(!world.is_dynamic(1));
    assert_eq!(world.inverse_mass(1), 0.0);
    assert_eq!(world.inverse_mass(9), 0.0);
    world.positions[1] = f64::NAN;
    assert_eq!(world.position(0), None);
}
