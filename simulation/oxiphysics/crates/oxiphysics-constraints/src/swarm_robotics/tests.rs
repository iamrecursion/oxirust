//! Tests of the swarm robotics constraints.

use super::*;

// Helper: make a line of agents along X axis.
fn make_agents(n: usize, spacing: f64, sensor_radius: f64) -> Vec<SwarmAgent> {
    (0..n)
        .map(|i| {
            let mut a = SwarmAgent::new(i, [i as f64 * spacing, 0.0, 0.0], sensor_radius);
            a.velocity = [1.0, 0.0, 0.0];
            a
        })
        .collect()
}

// ── SwarmAgent ──────────────────────────────────────────────────────────

#[test]
fn test_agent_creation() {
    let a = SwarmAgent::new(0, [1.0, 2.0, 3.0], 5.0);
    assert!((a.position[0] - 1.0).abs() < 1e-10);
    assert!((a.sensor_radius - 5.0).abs() < 1e-10);
}

#[test]
fn test_agent_integrate() {
    let mut a = SwarmAgent::new(0, [0.0; 3], 5.0);
    a.velocity = [2.0, 0.0, 0.0];
    a.integrate(0.5);
    assert!(
        (a.position[0] - 1.0).abs() < 1e-10,
        "pos={:.6}",
        a.position[0]
    );
}

#[test]
fn test_agent_neighbors_all_in_range() {
    let agents = make_agents(3, 1.0, 10.0);
    let nbs = agents[1].neighbors(&agents);
    assert_eq!(nbs.len(), 2);
}

#[test]
fn test_agent_neighbors_none_in_range() {
    let agents = make_agents(3, 100.0, 1.0);
    let nbs = agents[0].neighbors(&agents);
    assert!(nbs.is_empty());
}

#[test]
fn test_agent_heading_update() {
    let mut a = SwarmAgent::new(0, [0.0; 3], 5.0);
    a.velocity = [1.0, 0.0, 0.0];
    a.integrate(0.1);
    assert!((a.heading - 0.0).abs() < 1e-10, "heading={:.6}", a.heading);
}

// ── FlockingBehavior ────────────────────────────────────────────────────

#[test]
fn test_flocking_cohesion_nonempty() {
    let flock = FlockingBehavior::new(1.0);
    let agent = SwarmAgent::new(0, [0.0, 0.0, 0.0], 100.0);
    let nb1 = SwarmAgent::new(1, [4.0, 0.0, 0.0], 100.0);
    let nb2 = SwarmAgent::new(2, [-4.0, 0.0, 0.0], 100.0);
    let nbs = vec![&nb1, &nb2];
    let coh = flock.cohesion(&agent, &nbs);
    // Center is at [0,0,0] so cohesion should be near zero in X.
    assert!(coh[0].abs() < 1e-3, "cohesion_x={:.6}", coh[0]);
}

#[test]
fn test_flocking_alignment_matches_average() {
    let flock = FlockingBehavior::new(1.0);
    let mut agent = SwarmAgent::new(0, [0.0; 3], 100.0);
    agent.velocity = [0.0; 3];
    let mut nb = SwarmAgent::new(1, [2.0, 0.0, 0.0], 100.0);
    nb.velocity = [4.0, 0.0, 0.0];
    let nbs = vec![&nb];
    let ali = flock.alignment(&agent, &nbs);
    // Should steer toward [4,0,0] from [0,0,0], so X component positive.
    assert!(ali[0] > 0.0, "alignment_x={:.6}", ali[0]);
}

#[test]
fn test_flocking_separation_pushes_away() {
    let flock = FlockingBehavior::new(5.0); // r_sep = 5
    let agent = SwarmAgent::new(0, [0.0, 0.0, 0.0], 100.0);
    let nb = SwarmAgent::new(1, [1.0, 0.0, 0.0], 100.0); // within r_sep
    let nbs = vec![&nb];
    let sep = flock.separation(&agent, &nbs);
    // Should push agent in -X direction (away from nb at +X).
    assert!(sep[0] < 0.0, "sep_x={:.6}", sep[0]);
}

#[test]
fn test_flocking_step_swarm_runs() {
    let flock = FlockingBehavior::new(2.0);
    let mut agents = make_agents(5, 1.5, 10.0);
    flock.step_swarm(&mut agents, 0.01);
    // Just checking it doesn't panic.
    assert_eq!(agents.len(), 5);
}

#[test]
fn test_flocking_no_collisions_after_steps() {
    let flock = FlockingBehavior::new(2.0);
    let mut agents = make_agents(5, 1.5, 10.0);
    for _ in 0..20 {
        flock.step_swarm(&mut agents, 0.05);
    }
    // Check no two agents are at exactly the same position.
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let d = vec3_dist(agents[i].position, agents[j].position);
            assert!(d > 1e-6, "collision between {} and {}: d={:.6}", i, j, d);
        }
    }
}

#[test]
fn test_flocking_cohesion_radius() {
    // After many steps, agents should stay within a bounded region.
    let flock = FlockingBehavior::new(1.5);
    let mut agents = make_agents(5, 2.0, 15.0);
    for _ in 0..100 {
        flock.step_swarm(&mut agents, 0.02);
    }
    // Compute centroid
    let cx: f64 = agents.iter().map(|a| a.position[0]).sum::<f64>() / 5.0;
    let cy: f64 = agents.iter().map(|a| a.position[1]).sum::<f64>() / 5.0;
    // All agents within 50 units of centroid (generous bound).
    for a in &agents {
        let d = ((a.position[0] - cx).powi(2) + (a.position[1] - cy).powi(2)).sqrt();
        assert!(d < 50.0, "agent {} too far from centroid: d={:.6}", a.id, d);
    }
}

// ── FormationControl ────────────────────────────────────────────────────

#[test]
fn test_formation_desired_positions() {
    let offsets = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    let fc = FormationControl::new(FormationType::VirtualStructure, offsets, 1.0);
    let desired = fc.desired_positions();
    assert!((desired[0][0] - 1.0).abs() < 1e-10);
    assert!((desired[1][0] + 1.0).abs() < 1e-10);
}

#[test]
fn test_formation_tracking_error_decreases() {
    let offsets = vec![[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]];
    let mut fc = FormationControl::new(FormationType::VirtualStructure, offsets, 2.0);
    let mut agents = vec![
        SwarmAgent::new(0, [5.0, 0.0, 0.0], 10.0),
        SwarmAgent::new(1, [-5.0, 0.0, 0.0], 10.0),
    ];
    let err0 = fc.tracking_error(&agents);
    for _ in 0..50 {
        fc.step(&mut agents, 0.05);
    }
    let err1 = fc.tracking_error(&agents);
    assert!(
        err1 < err0,
        "error did not decrease: {:.6} → {:.6}",
        err0,
        err1
    );
}

#[test]
fn test_formation_advance_center() {
    let mut fc = FormationControl::new(FormationType::VirtualStructure, vec![], 1.0);
    fc.virtual_velocity = [1.0, 0.0, 0.0];
    fc.advance_center(2.0);
    assert!((fc.virtual_center[0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_formation_steering_out_of_range() {
    let fc = FormationControl::new(FormationType::VirtualStructure, vec![], 1.0);
    let agents = vec![SwarmAgent::new(0, [0.0; 3], 5.0)];
    let s = fc.formation_steering(&agents, 99);
    for c in s {
        assert!(c.abs() < 1e-10);
    }
}

// ── ConsensusProtocol ───────────────────────────────────────────────────

#[test]
fn test_consensus_convergence() {
    let states = vec![0.0, 5.0, 10.0, 15.0];
    let mut proto = ConsensusProtocol::new_fully_connected(states, 0.1);
    for _ in 0..5000 {
        proto.step(0.01);
    }
    assert!(
        proto.is_converged(0.01),
        "not converged: {:?}",
        proto.states
    );
}

#[test]
fn test_consensus_mean_preserved() {
    let states = vec![1.0, 3.0, 5.0, 7.0];
    let mean0 = 4.0;
    let mut proto = ConsensusProtocol::new_fully_connected(states, 0.05);
    for _ in 0..100 {
        proto.step(0.01);
    }
    let mean1 = proto.mean_state();
    assert!((mean1 - mean0).abs() < 0.01, "mean changed: {:.6}", mean1);
}

#[test]
fn test_consensus_two_agents() {
    let mut proto = ConsensusProtocol::new_fully_connected(vec![0.0, 10.0], 0.5);
    for _ in 0..2000 {
        proto.step(0.01);
    }
    assert!((proto.states[0] - 5.0).abs() < 0.01);
    assert!((proto.states[1] - 5.0).abs() < 0.01);
}

#[test]
fn test_consensus_laplacian_eigenvalue_positive() {
    let proto = ConsensusProtocol::new_fully_connected(vec![0.0, 1.0, 2.0, 3.0], 0.1);
    let lam = proto.laplacian_second_eigenvalue();
    // The complete graph on 4 agents has the Laplacian eigenvalues 0, 4, 4, 4.
    assert!((lam - 4.0).abs() < 1e-9, "lambda2={:.12}", lam);
}

// ── CoverageControl ─────────────────────────────────────────────────────

#[test]
fn test_coverage_quality_improves_with_agents() {
    let bounds = [[0.0, 10.0], [0.0, 10.0]];
    let ctrl = CoverageControl::new(bounds, 100, 0.5);
    let one_agent = vec![SwarmAgent::new(0, [5.0, 5.0, 0.0], 20.0)];
    let two_agents = vec![
        SwarmAgent::new(0, [2.5, 5.0, 0.0], 20.0),
        SwarmAgent::new(1, [7.5, 5.0, 0.0], 20.0),
    ];
    let q1 = ctrl.coverage_quality(&one_agent);
    let q2 = ctrl.coverage_quality(&two_agents);
    assert!(
        q2 < q1,
        "more agents should improve coverage: {:.6} vs {:.6}",
        q1,
        q2
    );
}

#[test]
fn test_coverage_step_moves_agents() {
    let bounds = [[0.0, 10.0], [0.0, 10.0]];
    let ctrl = CoverageControl::new(bounds, 100, 1.0);
    let mut agents = vec![
        SwarmAgent::new(0, [1.0, 1.0, 0.0], 20.0),
        SwarmAgent::new(1, [2.0, 2.0, 0.0], 20.0),
    ];
    let pos0 = agents[0].position;
    ctrl.step(&mut agents, 0.1);
    let moved = vec3_dist(agents[0].position, pos0) > 1e-6
        || vec3_dist(agents[1].position, [2.0, 2.0, 0.0]) > 1e-6;
    assert!(moved, "agents should have moved");
}

#[test]
fn test_coverage_quality_no_agents() {
    let bounds = [[0.0, 10.0], [0.0, 10.0]];
    let ctrl = CoverageControl::new(bounds, 100, 0.5);
    let q = ctrl.coverage_quality(&[]);
    assert!(q.is_infinite());
}

// ── TaskAllocation ──────────────────────────────────────────────────────

#[test]
fn test_task_allocation_all_assigned_when_enough_agents() {
    let alloc = TaskAllocation::new(1.0);
    let agents = vec![
        SwarmAgent::new(0, [0.0, 0.0, 0.0], 10.0),
        SwarmAgent::new(1, [5.0, 0.0, 0.0], 10.0),
        SwarmAgent::new(2, [10.0, 0.0, 0.0], 10.0),
    ];
    let tasks = vec![
        Task {
            id: 0,
            position: [1.0, 0.0, 0.0],
            utility: 10.0,
        },
        Task {
            id: 1,
            position: [6.0, 0.0, 0.0],
            utility: 10.0,
        },
    ];
    let result = alloc.allocate(&agents, &tasks);
    assert_eq!(TaskAllocation::assigned_count(&result), 2);
}

#[test]
fn test_task_allocation_bid_prefers_nearby() {
    let alloc = TaskAllocation::new(1.0);
    let agent_near = SwarmAgent::new(0, [1.0, 0.0, 0.0], 10.0);
    let agent_far = SwarmAgent::new(1, [100.0, 0.0, 0.0], 10.0);
    let task = Task {
        id: 0,
        position: [0.0, 0.0, 0.0],
        utility: 10.0,
    };
    let bid_near = alloc.bid(&agent_near, &task);
    let bid_far = alloc.bid(&agent_far, &task);
    assert!(
        bid_near > bid_far,
        "near={:.6} far={:.6}",
        bid_near,
        bid_far
    );
}

#[test]
fn test_task_allocation_partial_assign() {
    let alloc = TaskAllocation::new(1.0);
    let agents = vec![SwarmAgent::new(0, [0.0, 0.0, 0.0], 10.0)];
    let tasks = vec![
        Task {
            id: 0,
            position: [0.0, 0.0, 0.0],
            utility: 10.0,
        },
        Task {
            id: 1,
            position: [1.0, 0.0, 0.0],
            utility: 10.0,
        },
    ];
    let result = alloc.allocate(&agents, &tasks);
    // Only 1 agent, so at most 1 task can be assigned.
    let assigned = TaskAllocation::assigned_count(&result);
    assert!(assigned <= 1, "assigned={}", assigned);
}

// ── SwarmSearch (PSO) ───────────────────────────────────────────────────

#[test]
fn test_pso_converges_to_optimum() {
    // Minimize f(x,y,z) = x^2 + y^2 + z^2 (optimum at origin)
    let fitness = |p: [f64; 3]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    let particles = vec![
        Particle::new([3.0, 3.0, 3.0]),
        Particle::new([-3.0, 2.0, 1.0]),
        Particle::new([1.0, -4.0, 2.0]),
        Particle::new([-2.0, -2.0, -2.0]),
    ];
    let mut pso = SwarmSearch::new(particles, 0.5, 1.5, 1.5, 2.0);
    for _ in 0..200 {
        pso.evaluate(&fitness);
        pso.step_deterministic();
    }
    pso.evaluate(&fitness);
    let best = pso.best_fitness();
    assert!(best < 1.0, "PSO best fitness={:.6}", best);
}

#[test]
fn test_pso_best_fitness_decreases() {
    let fitness = |p: [f64; 3]| (p[0] - 2.0).powi(2) + p[1].powi(2) + p[2].powi(2);
    let particles = vec![
        Particle::new([5.0, 5.0, 5.0]),
        Particle::new([-5.0, -5.0, -5.0]),
    ];
    let mut pso = SwarmSearch::new(particles, 0.7, 1.4, 1.4, 3.0);
    pso.evaluate(&fitness);
    let f0 = pso.best_fitness();
    for _ in 0..50 {
        pso.step_deterministic();
        pso.evaluate(&fitness);
    }
    let f1 = pso.best_fitness();
    assert!(
        f1 <= f0,
        "fitness should not increase: {:.6} → {:.6}",
        f0,
        f1
    );
}

#[test]
fn test_pso_step_with_random() {
    let fitness = |p: [f64; 3]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    let particles = vec![
        Particle::new([2.0, 2.0, 2.0]),
        Particle::new([-2.0, -2.0, -2.0]),
    ];
    let mut pso = SwarmSearch::new(particles, 0.5, 1.5, 1.5, 3.0);
    pso.evaluate(&fitness);
    let f0 = pso.best_fitness();
    for i in 0..100 {
        let r = (i as f64 * 0.137).sin().abs() * 0.9 + 0.05;
        pso.step_with_random(r, 1.0 - r);
        pso.evaluate(&fitness);
    }
    let f1 = pso.best_fitness();
    assert!(
        f1 <= f0 + 1e-6,
        "pso step_with_random: {:.6} → {:.6}",
        f0,
        f1
    );
}

#[test]
fn test_pso_particles_count_unchanged() {
    let particles: Vec<Particle> = (0..10)
        .map(|i| Particle::new([i as f64, 0.0, 0.0]))
        .collect();
    let mut pso = SwarmSearch::new(particles, 0.5, 1.5, 1.5, 5.0);
    pso.evaluate(&|p: [f64; 3]| p[0].powi(2));
    pso.step_deterministic();
    assert_eq!(pso.particles.len(), 10);
}

// ── EmergentBehavior ────────────────────────────────────────────────────

#[test]
fn test_pheromone_deposit_increases_concentration() {
    let mut cell = PheromoneCell::new(0.0, 0.1);
    cell.deposit(5.0);
    assert!((cell.concentration - 5.0).abs() < 1e-10);
}

#[test]
fn test_pheromone_decay() {
    let mut cell = PheromoneCell::new(10.0, 1.0);
    cell.decay(1.0);
    assert!(cell.concentration < 10.0);
    assert!(cell.concentration > 0.0);
}

#[test]
fn test_stigmergy_deposit_and_read() {
    let mut env = EmergentBehavior::new([[0.0, 10.0], [0.0, 10.0]], 10, 0.1, 0.01);
    env.deposit_at([5.0, 5.0, 0.0], 3.0);
    let c = env.concentration_at([5.0, 5.0, 0.0]);
    assert!(c > 0.0, "should have pheromone: {:.6}", c);
}

#[test]
fn test_stigmergy_total_decays_over_time() {
    let mut env = EmergentBehavior::new([[0.0, 10.0], [0.0, 10.0]], 10, 0.5, 0.0);
    env.deposit_at([5.0, 5.0, 0.0], 10.0);
    let t0 = env.total_pheromone();
    for _ in 0..10 {
        env.step(0.1);
    }
    let t1 = env.total_pheromone();
    assert!(t1 < t0, "pheromone should decay: {:.6} → {:.6}", t0, t1);
}

#[test]
fn test_stigmergy_multiple_deposits() {
    let mut env = EmergentBehavior::new([[0.0, 10.0], [0.0, 10.0]], 10, 0.1, 0.0);
    env.deposit_at([1.0, 1.0, 0.0], 1.0);
    env.deposit_at([9.0, 9.0, 0.0], 2.0);
    let t = env.total_pheromone();
    assert!(t > 2.9 && t < 3.1, "total={:.6}", t);
}

#[test]
fn test_stigmergy_world_to_grid_bounds() {
    let env = EmergentBehavior::new([[0.0, 10.0], [0.0, 10.0]], 5, 0.1, 0.0);
    let (ix, iy) = env.world_to_grid([0.0, 0.0, 0.0]);
    assert_eq!(ix, 0);
    assert_eq!(iy, 0);
    let (ix2, iy2) = env.world_to_grid([9.99, 9.99, 0.0]);
    assert!(ix2 < 5);
    assert!(iy2 < 5);
}

// ── Additional edge-case tests ──────────────────────────────────────────

#[test]
fn test_flocking_empty_neighbors_separation() {
    let flock = FlockingBehavior::new(2.0);
    let agent = SwarmAgent::new(0, [0.0; 3], 5.0);
    let sep = flock.separation(&agent, &[]);
    for c in sep {
        assert!(c.abs() < 1e-10);
    }
}

#[test]
fn test_flocking_empty_neighbors_alignment() {
    let flock = FlockingBehavior::new(2.0);
    let agent = SwarmAgent::new(0, [0.0; 3], 5.0);
    let ali = flock.alignment(&agent, &[]);
    for c in ali {
        assert!(c.abs() < 1e-10);
    }
}

#[test]
fn test_flocking_empty_neighbors_cohesion() {
    let flock = FlockingBehavior::new(2.0);
    let agent = SwarmAgent::new(0, [0.0; 3], 5.0);
    let coh = flock.cohesion(&agent, &[]);
    for c in coh {
        assert!(c.abs() < 1e-10);
    }
}

#[test]
fn test_consensus_single_agent_unchanged() {
    let mut proto = ConsensusProtocol::new_fully_connected(vec![42.0], 0.5);
    proto.step(0.1);
    assert!((proto.states[0] - 42.0).abs() < 1e-10);
}

#[test]
fn test_formation_zero_error_at_desired() {
    let offsets = vec![[1.0, 0.0, 0.0]];
    let fc = FormationControl::new(FormationType::VirtualStructure, offsets, 1.0);
    let agents = vec![SwarmAgent::new(0, [1.0, 0.0, 0.0], 5.0)];
    let err = fc.tracking_error(&agents);
    assert!(err.abs() < 1e-10, "err={:.6}", err);
}

#[test]
fn test_pso_initial_global_best_set() {
    let particles = vec![Particle::new([3.0, 0.0, 0.0])];
    let pso = SwarmSearch::new(particles, 0.5, 1.5, 1.5, 2.0);
    // Global best fitness starts at infinity (not evaluated yet).
    assert!(pso.global_best_fitness.is_infinite());
}

#[test]
fn test_coverage_voronoi_centroid_two_agents() {
    let bounds = [[0.0, 10.0], [0.0, 10.0]];
    let ctrl = CoverageControl::new(bounds, 400, 1.0);
    let agents = vec![
        SwarmAgent::new(0, [2.0, 5.0, 0.0], 20.0),
        SwarmAgent::new(1, [8.0, 5.0, 0.0], 20.0),
    ];
    let c0 = ctrl
        .voronoi_centroid(&agents, 0)
        .expect("agent 0 is in the slice");
    let c1 = ctrl
        .voronoi_centroid(&agents, 1)
        .expect("agent 1 is in the slice");
    // Agent 0 is on the left, its centroid should have x < 5.
    assert!(c0[0] < 5.0, "centroid0_x={:.6}", c0[0]);
    // Agent 1 is on the right, its centroid should have x > 5.
    assert!(c1[0] > 5.0, "centroid1_x={:.6}", c1[0]);
}

// ── Helpers for the regression tests below ──────────────────────────────

fn agent_with_velocity(id: usize, position: [f64; 3], velocity: [f64; 3]) -> SwarmAgent {
    let mut agent = SwarmAgent::new(id, position, 1.0e3);
    agent.velocity = velocity;
    agent
}

fn triangle_offsets() -> Vec<[f64; 3]> {
    vec![[2.0, 0.0, 0.0], [-1.0, 1.7, 0.0], [-1.0, -1.7, 0.5]]
}

fn square_offsets() -> Vec<[f64; 3]> {
    vec![
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
}

fn scattered_agents() -> Vec<SwarmAgent> {
    vec![
        agent_with_velocity(0, [3.0, -2.0, 0.0], [0.5, 0.2, 0.0]),
        agent_with_velocity(1, [-4.0, 5.0, 1.0], [-0.3, 0.0, 0.1]),
        agent_with_velocity(2, [0.5, 0.5, -1.0], [0.0, 0.4, 0.0]),
        agent_with_velocity(3, [7.0, 1.0, 0.0], [0.1, -0.2, -0.3]),
    ]
}

// ── FormationControl: damping, feed-forward, formation types (F0483) ────

#[test]
fn test_formation_default_damping_is_critical() {
    let fc = FormationControl::new(FormationType::VirtualStructure, vec![], 9.0);
    assert!((fc.damping - 6.0).abs() < 1e-12, "damping={}", fc.damping);
    let no_gain = FormationControl::new(FormationType::VirtualStructure, vec![], -4.0);
    assert_eq!(no_gain.damping, 0.0);
    let custom = FormationControl::new(FormationType::Consensus, vec![], 9.0).with_damping(1.5);
    assert_eq!(custom.damping, 1.5);
}

#[test]
fn test_virtual_structure_tracking_error_vanishes() {
    let mut fc = FormationControl::new(FormationType::VirtualStructure, triangle_offsets(), 4.0);
    let mut agents = vec![
        agent_with_velocity(0, [5.0, 3.0, 1.0], [1.0, -2.0, 0.0]),
        agent_with_velocity(1, [-6.0, 2.0, 0.0], [0.0, 3.0, 1.0]),
        agent_with_velocity(2, [0.0, -7.0, 2.0], [-2.0, 0.0, 0.0]),
    ];
    assert!(fc.tracking_error(&agents) > 3.0);
    // 20 s at dt = 0.01.
    for _ in 0..2000 {
        fc.step(&mut agents, 0.01);
    }
    let error = fc.tracking_error(&agents);
    assert!(error < 1e-3, "tracking error after 20 s: {error}");
    for agent in &agents {
        assert!(
            vec3_len(agent.velocity) < 1e-3,
            "agent {} is still moving at {:?}",
            agent.id,
            agent.velocity
        );
    }
}

#[test]
fn test_virtual_structure_follows_a_moving_center_without_lag() {
    let mut fc = FormationControl::new(FormationType::VirtualStructure, triangle_offsets(), 4.0);
    fc.virtual_velocity = [0.5, -0.25, 0.1];
    let mut agents = vec![
        agent_with_velocity(0, [5.0, 3.0, 1.0], [0.0; 3]),
        agent_with_velocity(1, [-6.0, 2.0, 0.0], [0.0; 3]),
        agent_with_velocity(2, [0.0, -7.0, 2.0], [0.0; 3]),
    ];
    for _ in 0..2000 {
        fc.step(&mut agents, 0.01);
    }
    // The velocity of the structure is fed forward, so there is no steady-state
    // lag of `damping * |V| / gain` (about 0.56 m here).
    let error = fc.tracking_error(&agents);
    assert!(error < 1e-3, "tracking error with a moving centre: {error}");
    for agent in &agents {
        let relative = vec3_sub(agent.velocity, fc.virtual_velocity);
        assert!(
            vec3_len(relative) < 1e-3,
            "agent {} does not move with the structure: {:?}",
            agent.id,
            agent.velocity
        );
    }
}

#[test]
fn test_virtual_structure_converges_without_overshoot() {
    let mut fc = FormationControl::new(FormationType::VirtualStructure, vec![[0.0; 3]], 4.0);
    let mut agents = vec![SwarmAgent::new(0, [3.0, 0.0, 0.0], 10.0)];
    let mut lowest = f64::INFINITY;
    for _ in 0..4000 {
        fc.step(&mut agents, 0.005);
        lowest = lowest.min(agents[0].position[0]);
    }
    // Critical damping: the agent approaches the slot from one side only (up to
    // the discretisation of the time integration).
    assert!(lowest > -0.03, "overshoot to x = {lowest}");
    assert!(agents[0].position[0].abs() < 1e-3);
}

#[test]
fn test_undamped_formation_oscillates_forever() {
    let mut fc = FormationControl::new(FormationType::VirtualStructure, vec![[0.0; 3]], 4.0)
        .with_damping(0.0);
    let mut agents = vec![SwarmAgent::new(0, [3.0, 0.0, 0.0], 10.0)];
    let mut late_amplitude = 0.0_f64;
    for step in 0..4000 {
        fc.step(&mut agents, 0.005);
        if step >= 3000 {
            late_amplitude = late_amplitude.max(agents[0].position[0].abs());
        }
    }
    // Without damping the 3 m error never decays (the old behaviour of the
    // controller): after 15 s the amplitude is still close to its start.
    assert!(
        late_amplitude > 2.9,
        "amplitude after 15 s: {late_amplitude}"
    );
}

#[test]
fn test_consensus_formation_reaches_the_designed_offsets() {
    let mut fc = FormationControl::new(FormationType::Consensus, square_offsets(), 1.0);
    // A consensus formation has no leader: the virtual centre must be ignored.
    fc.virtual_center = [100.0, 100.0, 0.0];
    fc.virtual_velocity = [3.0, 0.0, 0.0];
    let mut agents = scattered_agents();
    let origin = |agents: &[SwarmAgent]| {
        let sum = agents
            .iter()
            .zip(square_offsets())
            .fold([0.0; 3], |acc, (a, off)| {
                vec3_add(acc, vec3_sub(a.position, off))
            });
        vec3_scale(sum, 1.0 / agents.len() as f64)
    };
    let origin_before = origin(&agents);
    let mean_velocity = vec3_scale(
        agents
            .iter()
            .fold([0.0; 3], |acc, a| vec3_add(acc, a.velocity)),
        0.25,
    );
    assert!(fc.relative_offset_error(&agents) > 2.0);

    let (dt, steps) = (0.01, 4000);
    for _ in 0..steps {
        fc.step(&mut agents, dt);
    }

    let error = fc.relative_offset_error(&agents);
    assert!(error < 1e-3, "relative offset error after 40 s: {error}");
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let separation = vec3_sub(agents[i].position, agents[j].position);
            let designed = vec3_sub(square_offsets()[i], square_offsets()[j]);
            assert!(
                vec3_dist(separation, designed) < 3e-3,
                "agents {i} and {j} are {separation:?} apart, designed {designed:?}"
            );
        }
    }
    // With every agent in range of every other one the accelerations cancel,
    // so the group keeps the mean velocity it started with and drifts with it
    // instead of heading for the virtual centre.
    let expected = vec3_add(
        origin_before,
        vec3_scale(mean_velocity, dt * f64::from(steps)),
    );
    assert!(
        vec3_dist(origin(&agents), expected) < 1e-8,
        "formation origin {:?}, expected {:?}",
        origin(&agents),
        expected
    );
}

#[test]
fn test_formation_type_selects_the_control_law() {
    // Both agents are exactly in formation (their offset-corrected positions
    // agree and they move together), but far from the virtual centre.
    let offsets = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    let agents = vec![
        agent_with_velocity(0, [11.0, 5.0, 0.0], [0.5, 0.0, 0.0]),
        agent_with_velocity(1, [9.0, 5.0, 0.0], [0.5, 0.0, 0.0]),
    ];
    let virtual_structure =
        FormationControl::new(FormationType::VirtualStructure, offsets.clone(), 2.0);
    let consensus = FormationControl::new(FormationType::Consensus, offsets, 2.0);

    let pull = virtual_structure.formation_steering(&agents, 0);
    assert!(
        pull[0] < -10.0,
        "the virtual structure pulls towards its slot: {pull:?}"
    );
    assert_eq!(consensus.formation_steering(&agents, 0), [0.0; 3]);
    assert_eq!(consensus.formation_steering(&agents, 1), [0.0; 3]);

    // Consensus steering reacts to a formation error between the agents.
    let mut disturbed = agents.clone();
    disturbed[1].position[1] += 1.0;
    let correction = consensus.formation_steering(&disturbed, 0);
    // a_0 = gain * ((p_1 - off_1) - (p_0 - off_0)) = 2 * (0, 1, 0)
    assert!(
        vec3_dist(correction, [0.0, 2.0, 0.0]) < 1e-12,
        "{correction:?}"
    );
}

#[test]
fn test_consensus_uses_the_velocities_of_its_neighbours() {
    let consensus =
        FormationControl::new(FormationType::Consensus, vec![[0.0; 3]; 3], 1.0).with_damping(3.0);
    let agents = vec![
        agent_with_velocity(0, [0.0; 3], [1.0, 0.0, 0.0]),
        agent_with_velocity(1, [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]),
        agent_with_velocity(2, [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
    ];
    // Agent 0: position term 1 * ((1 - 0) + (2 - 0)) = 3, velocity term
    // 3 * (mean(3, 5) - 1) = 9.
    let steering = consensus.formation_steering(&agents, 0);
    assert!(
        vec3_dist(steering, [12.0, 0.0, 0.0]) < 1e-12,
        "{steering:?}"
    );
}

#[test]
fn test_consensus_only_listens_to_agents_inside_the_sensor_radius() {
    let consensus = FormationControl::new(FormationType::Consensus, vec![[0.0; 3]; 3], 1.0);
    let mut agents = vec![
        SwarmAgent::new(0, [0.0, 0.0, 0.0], 5.0),
        SwarmAgent::new(1, [3.0, 0.0, 0.0], 5.0),
        SwarmAgent::new(2, [100.0, 0.0, 0.0], 5.0),
    ];
    let near_only = consensus.formation_steering(&agents, 0);
    // Only agent 1 is in range: a_0 = gain * (3, 0, 0) + damping * (0 - 0).
    assert!(
        vec3_dist(near_only, [3.0, 0.0, 0.0]) < 1e-12,
        "{near_only:?}"
    );
    // Agent 2 is alone in its range, so nothing steers it.
    assert_eq!(consensus.formation_steering(&agents, 2), [0.0; 3]);
    // Moving the distant agent does not change what agent 0 does.
    agents[2].position = [-50.0, 20.0, 7.0];
    assert_eq!(consensus.formation_steering(&agents, 0), near_only);
    // A NaN sensor radius sees nobody.
    agents[0].sensor_radius = f64::NAN;
    assert_eq!(consensus.formation_steering(&agents, 0), [0.0; 3]);
}

#[test]
fn test_agents_without_an_offset_are_neither_steered_nor_seen() {
    let two_offsets = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    let members = vec![
        agent_with_velocity(0, [2.0, 0.0, 0.0], [0.0; 3]),
        agent_with_velocity(1, [-3.0, 1.0, 0.0], [0.0; 3]),
    ];
    let mut with_bystander = members.clone();
    with_bystander.push(agent_with_velocity(2, [0.0, 0.5, 0.0], [9.0, 9.0, 9.0]));
    for formation_type in [FormationType::VirtualStructure, FormationType::Consensus] {
        let fc = FormationControl::new(formation_type, two_offsets.clone(), 2.0);
        assert_eq!(fc.formation_steering(&with_bystander, 2), [0.0; 3]);
        for member in 0..2 {
            assert_eq!(
                fc.formation_steering(&with_bystander, member),
                fc.formation_steering(&members, member),
                "the bystander must not influence member {member}"
            );
        }
    }
}

#[test]
fn test_formation_step_does_not_depend_on_the_agent_order() {
    for formation_type in [FormationType::VirtualStructure, FormationType::Consensus] {
        let mut forward = FormationControl::new(formation_type.clone(), square_offsets(), 2.0);
        let mut reversed_offsets = square_offsets();
        reversed_offsets.reverse();
        let mut backward = FormationControl::new(formation_type, reversed_offsets, 2.0);
        let mut agents_forward = scattered_agents();
        let mut agents_backward = scattered_agents();
        agents_backward.reverse();
        for _ in 0..300 {
            forward.step(&mut agents_forward, 0.01);
            backward.step(&mut agents_backward, 0.01);
        }
        for agent in &agents_forward {
            let twin = agents_backward
                .iter()
                .find(|a| a.id == agent.id)
                .expect("the same agents");
            assert!(
                vec3_dist(agent.position, twin.position) < 1e-9,
                "agent {} differs between the two orders",
                agent.id
            );
            assert!(vec3_dist(agent.velocity, twin.velocity) < 1e-9);
        }
    }
}

#[test]
fn test_formation_error_matches_the_formation_type() {
    let offsets = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    // In formation, translated by (10, 0, 0) from the virtual centre.
    let agents = vec![
        SwarmAgent::new(0, [11.0, 0.0, 0.0], 5.0),
        SwarmAgent::new(1, [9.0, 0.0, 0.0], 5.0),
    ];
    let virtual_structure =
        FormationControl::new(FormationType::VirtualStructure, offsets.clone(), 1.0);
    let consensus = FormationControl::new(FormationType::Consensus, offsets, 1.0);
    assert!((virtual_structure.formation_error(&agents) - 10.0).abs() < 1e-12);
    assert!(consensus.formation_error(&agents) < 1e-12);
    assert!((consensus.relative_offset_error(&agents)).abs() < 1e-12);
    // Distorting the formation shows up in the relative offset error.
    let mut bent = agents.clone();
    bent[1].position[1] = 2.0;
    assert!((consensus.formation_error(&bent) - 1.0).abs() < 1e-12);
}

// ── ConsensusProtocol: the Fiedler value (F0484) ─────────────────────────

fn graph(n: usize, edges: &[(usize, usize)]) -> ConsensusProtocol {
    let mut neighbors = vec![Vec::new(); n];
    for &(a, b) in edges {
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    ConsensusProtocol::new(vec![0.0; n], neighbors, 1.0)
}

fn path_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n - 1).map(|i| (i, i + 1)).collect()
}

#[test]
fn test_fiedler_value_of_complete_graphs_is_the_number_of_agents() {
    for n in 2..=8usize {
        let lambda_2 =
            ConsensusProtocol::new_fully_connected(vec![0.0; n], 1.0).laplacian_second_eigenvalue();
        assert!((lambda_2 - n as f64).abs() < 1e-9, "K_{n}: {lambda_2}");
        // The old `min_degree / n` formula would have returned (n - 1) / n.
        assert!((lambda_2 - (n as f64 - 1.0) / n as f64).abs() > 0.5);
    }
}

#[test]
fn test_fiedler_value_of_paths() {
    for n in 2..=10usize {
        let expected = 2.0 * (1.0 - (std::f64::consts::PI / n as f64).cos());
        let lambda_2 = graph(n, &path_edges(n)).laplacian_second_eigenvalue();
        assert!(
            (lambda_2 - expected).abs() < 1e-9,
            "P_{n}: {lambda_2} vs {expected}"
        );
    }
}

#[test]
fn test_fiedler_value_of_cycles_and_stars() {
    for n in 3..=9usize {
        let mut edges = path_edges(n);
        edges.push((n - 1, 0));
        let expected = 2.0 - 2.0 * (2.0 * std::f64::consts::PI / n as f64).cos();
        let lambda_2 = graph(n, &edges).laplacian_second_eigenvalue();
        assert!(
            (lambda_2 - expected).abs() < 1e-9,
            "C_{n}: {lambda_2} vs {expected}"
        );

        let star: Vec<(usize, usize)> = (1..n).map(|leaf| (0, leaf)).collect();
        let lambda_2 = graph(n, &star).laplacian_second_eigenvalue();
        assert!(
            (lambda_2 - 1.0).abs() < 1e-9,
            "star on {n} agents: {lambda_2}"
        );
    }
}

#[test]
fn test_fiedler_value_of_a_disconnected_graph_is_zero() {
    let two_pairs = graph(4, &[(0, 1), (2, 3)]);
    assert!(two_pairs.laplacian_second_eigenvalue().abs() < 1e-9);
    let isolated_agent = graph(5, &path_edges(4));
    assert!(isolated_agent.laplacian_second_eigenvalue().abs() < 1e-9);
    let no_edges = graph(3, &[]);
    assert!(no_edges.laplacian_second_eigenvalue().abs() < 1e-9);
}

#[test]
fn test_fiedler_value_reads_the_neighbour_lists_as_an_undirected_graph() {
    let expected = graph(5, &path_edges(5)).laplacian_second_eigenvalue();
    // Every agent only lists its successor: a directed chain.
    let directed = ConsensusProtocol::new(
        vec![0.0; 5],
        vec![vec![1], vec![2], vec![3], vec![4], vec![]],
        1.0,
    );
    assert!((directed.laplacian_second_eigenvalue() - expected).abs() < 1e-9);
    // Self links, repeated entries and indices beyond the agents are ignored.
    let untidy = ConsensusProtocol::new(
        vec![0.0; 5],
        vec![
            vec![0, 1, 1],
            vec![0, 2, 99],
            vec![1, 3, 3],
            vec![2, 4, 7],
            vec![3, 4],
        ],
        1.0,
    );
    assert!((untidy.laplacian_second_eigenvalue() - expected).abs() < 1e-9);
}

#[test]
fn test_fiedler_value_with_missing_neighbour_rows_and_tiny_graphs() {
    // Two neighbour rows for four agents: agents 2 and 3 are isolated.
    let short = ConsensusProtocol::new(vec![0.0; 4], vec![vec![1], vec![0]], 1.0);
    assert!(short.laplacian_second_eigenvalue().abs() < 1e-9);
    // More neighbour rows than agents: the extra rows are ignored.
    let long = ConsensusProtocol::new(vec![0.0; 2], vec![vec![1], vec![0], vec![0, 1]], 1.0);
    assert!((long.laplacian_second_eigenvalue() - 2.0).abs() < 1e-9);
    assert_eq!(
        ConsensusProtocol::new(vec![], vec![], 1.0).laplacian_second_eigenvalue(),
        0.0
    );
    assert_eq!(
        ConsensusProtocol::new(vec![1.0], vec![vec![]], 1.0).laplacian_second_eigenvalue(),
        0.0
    );
}

#[test]
fn test_fiedler_value_is_the_consensus_convergence_rate() {
    // x' = -gain * L * x: the disagreement of the slowest mode decays like
    // exp(-gain * lambda_2 * t).
    let gain = 1.0;
    let mut protocol = graph(5, &path_edges(5));
    protocol.gain = gain;
    protocol.states = vec![4.0, -1.0, 2.5, 0.0, -3.0];
    let lambda_2 = protocol.laplacian_second_eigenvalue();
    let disagreement = |p: &ConsensusProtocol| {
        let mean = p.mean_state();
        p.states
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let dt = 1e-3;
    // Let the faster modes die out (they decay at least 1.0 faster).
    for _ in 0..20_000 {
        protocol.step(dt);
    }
    let before = disagreement(&protocol);
    for _ in 0..1000 {
        protocol.step(dt);
    }
    let after = disagreement(&protocol);
    let measured_rate = -(after / before).ln();
    assert!(
        (measured_rate - gain * lambda_2).abs() < 1e-3,
        "decay rate {measured_rate}, lambda_2 = {lambda_2}"
    );
}

// ── CoverageControl: batched Voronoi centroids (F0485) ───────────────────

/// The per-agent algorithm of the original implementation: scan every agent
/// for every sample point and keep the first nearest one.
fn reference_centroid(ctrl: &CoverageControl, agents: &[SwarmAgent], idx: usize) -> [f64; 3] {
    let grid_side = (ctrl.n_samples as f64).sqrt() as usize + 1;
    let dx = (ctrl.bounds[0][1] - ctrl.bounds[0][0]) / grid_side as f64;
    let dy = (ctrl.bounds[1][1] - ctrl.bounds[1][0]) / grid_side as f64;
    let mut sum = [0.0_f64; 3];
    let mut count = 0usize;
    for ix in 0..grid_side {
        for iy in 0..grid_side {
            let pt = [
                ctrl.bounds[0][0] + (ix as f64 + 0.5) * dx,
                ctrl.bounds[1][0] + (iy as f64 + 0.5) * dy,
                0.0,
            ];
            let nearest = agents
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    vec3_dist(a.position, pt)
                        .partial_cmp(&vec3_dist(b.position, pt))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if nearest == Some(idx) {
                sum = vec3_add(sum, pt);
                count += 1;
            }
        }
    }
    if count > 0 {
        vec3_scale(sum, 1.0 / count as f64)
    } else {
        agents[idx].position
    }
}

fn irregular_agents() -> Vec<SwarmAgent> {
    // No two sample points are equally far from two of these agents.
    [
        [1.3, 2.9, 0.0],
        [7.1, 3.3, 0.4],
        [4.4, 8.2, 0.0],
        [8.8, 8.9, -0.2],
        [2.2, 6.1, 0.0],
        [5.5, 0.7, 0.0],
        [9.4, 5.0, 0.3],
    ]
    .iter()
    .enumerate()
    .map(|(id, &p)| SwarmAgent::new(id, p, 20.0))
    .collect()
}

#[test]
fn test_batched_centroids_equal_the_per_agent_algorithm() {
    let agents = irregular_agents();
    for (n_samples, bounds) in [
        (100, [[0.0, 10.0], [0.0, 10.0]]),
        (900, [[0.0, 10.0], [0.0, 10.0]]),
        (500, [[-2.0, 12.0], [1.0, 9.0]]),
    ] {
        let ctrl = CoverageControl::new(bounds, n_samples, 1.0);
        let batched = ctrl.voronoi_centroids(&agents);
        assert_eq!(batched.len(), agents.len());
        for (idx, centroid) in batched.iter().enumerate() {
            let expected = reference_centroid(&ctrl, &agents, idx);
            assert!(
                vec3_dist(*centroid, expected) < 1e-12,
                "agent {idx}, {n_samples} samples: {centroid:?} vs {expected:?}"
            );
            assert_eq!(ctrl.voronoi_centroid(&agents, idx), Some(*centroid));
        }
    }
}

#[test]
fn test_voronoi_centroid_of_an_unknown_agent_is_none() {
    let ctrl = CoverageControl::new([[0.0, 10.0], [0.0, 10.0]], 100, 1.0);
    let agents = irregular_agents();
    assert_eq!(ctrl.voronoi_centroid(&agents, agents.len()), None);
    assert_eq!(ctrl.voronoi_centroid(&agents, usize::MAX), None);
    assert_eq!(ctrl.voronoi_centroid(&[], 0), None);
    assert!(ctrl.voronoi_centroids(&[]).is_empty());
}

#[test]
fn test_agent_with_an_empty_cell_keeps_its_position() {
    let ctrl = CoverageControl::new([[0.0, 10.0], [0.0, 10.0]], 100, 1.0);
    // The second agent is so far away that no sample point is nearest to it,
    // and the third one sits on top of the first one: the first of two equally
    // near agents owns the points.
    let agents = vec![
        SwarmAgent::new(0, [5.0, 5.0, 0.0], 20.0),
        SwarmAgent::new(1, [1000.0, 1000.0, 0.0], 20.0),
        SwarmAgent::new(2, [5.0, 5.0, 0.0], 20.0),
    ];
    let centroids = ctrl.voronoi_centroids(&agents);
    assert_eq!(centroids[1], agents[1].position);
    assert_eq!(centroids[2], agents[2].position);
    // Agent 0 owns the whole region: its centroid is the middle of the grid.
    assert!(
        vec3_dist(centroids[0], [5.0, 5.0, 0.0]) < 1e-9,
        "{:?}",
        centroids[0]
    );
}

#[test]
fn test_coverage_step_moves_every_agent_towards_its_centroid() {
    let ctrl = CoverageControl::new([[0.0, 10.0], [0.0, 10.0]], 400, 0.5);
    let mut agents = irregular_agents();
    let centroids = ctrl.voronoi_centroids(&agents);
    let before = agents.clone();
    ctrl.step(&mut agents, 0.2);
    for ((agent, old), centroid) in agents.iter().zip(&before).zip(&centroids) {
        let expected_velocity = vec3_scale(vec3_sub(*centroid, old.position), 0.5);
        assert!(vec3_dist(agent.velocity, expected_velocity) < 1e-12);
        let expected_position = vec3_add(old.position, vec3_scale(expected_velocity, 0.2));
        assert!(vec3_dist(agent.position, expected_position) < 1e-12);
    }
}

#[test]
fn test_coverage_iterations_spread_the_agents_out() {
    // Lloyd's algorithm: repeatedly moving to the centroid improves coverage.
    let ctrl = CoverageControl::new([[0.0, 10.0], [0.0, 10.0]], 400, 1.0);
    let mut agents: Vec<SwarmAgent> = [
        [4.0, 4.0, 0.0],
        [4.5, 5.0, 0.0],
        [5.0, 4.2, 0.0],
        [5.2, 5.1, 0.0],
    ]
    .iter()
    .enumerate()
    .map(|(id, &p)| SwarmAgent::new(id, p, 20.0))
    .collect();
    let before = ctrl.coverage_quality(&agents);
    for _ in 0..40 {
        ctrl.step(&mut agents, 1.0);
    }
    let after = ctrl.coverage_quality(&agents);
    assert!(after < 0.6 * before, "coverage {before} -> {after}");
}

// ── EmergentBehavior: diffusion (F0487) ──────────────────────────────────

fn concentrations(env: &EmergentBehavior) -> Vec<f64> {
    env.grid
        .iter()
        .flat_map(|row| row.iter().map(|cell| cell.concentration))
        .collect()
}

#[test]
fn test_diffusion_conserves_the_total_for_a_strong_diffusion() {
    // D * dt = 5 is twenty times beyond the limit (D * dt <= 0.25 cells^2) of
    // the old unscaled scheme, which diverged and hid it behind `.max(0.0)`.
    let mut env = EmergentBehavior::new([[0.0, 10.0], [0.0, 10.0]], 10, 0.0, 50.0);
    env.deposit_at([2.5, 7.5, 0.0], 100.0);
    env.deposit_at([9.9, 0.1, 0.0], 25.0);
    let before = env.total_pheromone();
    assert!((before - 125.0).abs() < 1e-12);
    for _ in 0..100 {
        env.step(0.1);
    }
    let after = env.total_pheromone();
    assert!(
        (after - before).abs() / before < 1e-9,
        "total pheromone {before} -> {after}"
    );
    let field = concentrations(&env);
    assert!(field.iter().all(|c| c.is_finite() && *c >= 0.0));
    // After 10 s with D = 50 the 10 x 10 domain is fully mixed.
    let mean = before / 100.0;
    for c in &field {
        assert!((c - mean).abs() < 1e-6 * mean, "not mixed: {c} vs {mean}");
    }
}

#[test]
fn test_diffusion_spreads_with_world_units_on_anisotropic_cells() {
    // 30 x 30 cells over 60 x 30 world units: cells are 2 wide and 1 high.
    let mut env = EmergentBehavior::new([[0.0, 60.0], [0.0, 30.0]], 30, 0.0, 1.0);
    env.deposit_at([31.0, 15.5, 0.0], 8.0);
    let (diffusion, time, dt) = (1.0, 2.0, 0.1);
    for _ in 0..20 {
        env.step(dt);
    }
    // The variance of a diffusing point source grows by 2 D per unit time along
    // each axis, in world units, whatever the cell size.
    let (mut mass, mut mean_x, mut mean_y) = (0.0, 0.0, 0.0);
    for (iy, row) in env.grid.iter().enumerate() {
        for (ix, cell) in row.iter().enumerate() {
            let (x, y) = ((ix as f64 + 0.5) * 2.0, (iy as f64 + 0.5) * 1.0);
            mass += cell.concentration;
            mean_x += cell.concentration * x;
            mean_y += cell.concentration * y;
        }
    }
    mean_x /= mass;
    mean_y /= mass;
    let (mut var_x, mut var_y) = (0.0, 0.0);
    for (iy, row) in env.grid.iter().enumerate() {
        for (ix, cell) in row.iter().enumerate() {
            let (x, y) = ((ix as f64 + 0.5) * 2.0, (iy as f64 + 0.5) * 1.0);
            var_x += cell.concentration * (x - mean_x).powi(2);
            var_y += cell.concentration * (y - mean_y).powi(2);
        }
    }
    var_x /= mass;
    var_y /= mass;
    let expected = 2.0 * diffusion * time;
    assert!((mass - 8.0).abs() < 1e-9, "mass {mass}");
    assert!((mean_x - 31.0).abs() < 1e-9 && (mean_y - 15.5).abs() < 1e-9);
    assert!(
        (var_x - expected).abs() < 1e-9 * expected,
        "var_x = {var_x}, expected {expected}"
    );
    assert!(
        (var_y - expected).abs() < 1e-9 * expected,
        "var_y = {var_y}, expected {expected}"
    );
}

#[test]
fn test_diffusion_substeps_keep_a_stiff_step_stable_and_non_negative() {
    let mut env = EmergentBehavior::new([[0.0, 4.0], [0.0, 4.0]], 8, 0.0, 400.0);
    env.deposit_at([0.2, 0.2, 0.0], 1.0);
    env.step(1.0);
    let field = concentrations(&env);
    assert!(field.iter().all(|c| c.is_finite() && *c >= 0.0));
    assert!((env.total_pheromone() - 1.0).abs() < 1e-12);
    // The domain is mixed long before one second has passed.
    for c in &field {
        assert!((c - 1.0 / 64.0).abs() < 1e-9, "{c}");
    }
}

#[test]
fn test_diffusion_that_needs_too_many_substeps_stays_bounded() {
    // The sub-step cap must neither hang nor lose stability.
    let mut env = EmergentBehavior::new([[0.0, 4.0], [0.0, 4.0]], 8, 0.0, 1.0e12);
    env.deposit_at([0.2, 0.2, 0.0], 1.0);
    env.step(1.0);
    let field = concentrations(&env);
    assert!(field.iter().all(|c| c.is_finite() && *c >= 0.0));
    assert!((env.total_pheromone() - 1.0).abs() < 1e-9);
}

#[test]
fn test_diffusion_ignores_unusable_parameters() {
    let reference = {
        let mut env = EmergentBehavior::new([[0.0, 4.0], [0.0, 4.0]], 4, 0.0, 1.0);
        env.deposit_at([0.5, 0.5, 0.0], 2.0);
        concentrations(&env)
    };
    let unchanged = |env: &EmergentBehavior| concentrations(env) == reference;
    for diffusion in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let mut env = EmergentBehavior::new([[0.0, 4.0], [0.0, 4.0]], 4, 0.0, diffusion);
        env.deposit_at([0.5, 0.5, 0.0], 2.0);
        env.step(0.1);
        assert!(unchanged(&env), "diffusion {diffusion}");
    }
    for bounds in [
        [[0.0, 0.0], [0.0, 4.0]],
        [[0.0, 4.0], [3.0, 3.0]],
        [[4.0, 0.0], [0.0, 4.0]],
    ] {
        let mut env = EmergentBehavior::new(bounds, 4, 0.0, 1.0);
        env.deposit_at([0.5, 0.5, 0.0], 2.0);
        let before = concentrations(&env);
        env.step(0.1);
        assert_eq!(concentrations(&env), before, "bounds {bounds:?}");
    }
    for dt in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let mut env = EmergentBehavior::new([[0.0, 4.0], [0.0, 4.0]], 4, 5.0, 1.0);
        env.deposit_at([0.5, 0.5, 0.0], 2.0);
        env.step(dt);
        assert!(unchanged(&env), "dt {dt}");
    }
}

#[test]
fn test_zero_decay_rate_loses_nothing_even_below_the_cutoff() {
    let mut cell = PheromoneCell::new(1e-15, 0.0);
    cell.decay(10.0);
    assert_eq!(cell.concentration, 1e-15);
    // A decaying cell still rounds a vanishing trace to zero.
    let mut fading = PheromoneCell::new(2e-12, 1.0);
    fading.decay(1.0);
    assert_eq!(fading.concentration, 0.0);
    let mut cell = PheromoneCell::new(1.0, 2.0);
    cell.decay(0.5);
    assert!((cell.concentration - (-1.0_f64).exp()).abs() < 1e-15);
}

#[test]
fn test_environment_always_has_a_cell() {
    let mut env = EmergentBehavior::new([[0.0, 1.0], [0.0, 1.0]], 0, 0.1, 1.0);
    assert_eq!(env.resolution, 1);
    env.deposit_at([0.5, 0.5, 0.0], 1.0);
    env.step(0.1);
    assert!(env.concentration_at([0.5, 0.5, 0.0]) > 0.0);
}
