// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0
//! Swarm robotics constraints: formation control, consensus, coverage.
//!
//! Implements Reynolds flocking, formation control, consensus protocols,
//! Voronoi-based coverage, auction-based task allocation, PSO search,
//! and stigmergy-based emergent behavior.

// ── Helper math ───────────────────────────────────────────────────────────────

fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_len(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn vec3_normalize(v: [f64; 3]) -> [f64; 3] {
    let len = vec3_len(v);
    if len < 1e-15 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn vec3_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    vec3_len(vec3_sub(a, b))
}

// ── SwarmAgent ────────────────────────────────────────────────────────────────

/// A single agent in a swarm with position, velocity, heading, and sensor radius.
#[derive(Clone, Debug)]
pub struct SwarmAgent {
    /// World-space position \[x, y, z\].
    pub position: [f64; 3],
    /// Linear velocity \[vx, vy, vz\].
    pub velocity: [f64; 3],
    /// Heading angle in radians (yaw, XY-plane).
    pub heading: f64,
    /// Sensor radius for neighbor detection.
    pub sensor_radius: f64,
    /// Unique agent identifier.
    pub id: usize,
}

impl SwarmAgent {
    /// Create a new swarm agent.
    pub fn new(id: usize, position: [f64; 3], sensor_radius: f64) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            heading: 0.0,
            sensor_radius,
            id,
        }
    }

    /// Return agents within `self.sensor_radius` from `agents` slice.
    pub fn neighbors<'a>(&self, agents: &'a [SwarmAgent]) -> Vec<&'a SwarmAgent> {
        agents
            .iter()
            .filter(|a| {
                a.id != self.id && vec3_dist(self.position, a.position) <= self.sensor_radius
            })
            .collect()
    }

    /// Integrate position by one timestep `dt` using current velocity.
    pub fn integrate(&mut self, dt: f64) {
        self.position = vec3_add(self.position, vec3_scale(self.velocity, dt));
        // Update heading from velocity projection onto XY plane.
        if self.velocity[0].abs() > 1e-12 || self.velocity[1].abs() > 1e-12 {
            self.heading = self.velocity[1].atan2(self.velocity[0]);
        }
    }

    /// Apply a steering acceleration, clamped to `max_speed`.
    pub fn apply_steering(&mut self, steering: [f64; 3], max_speed: f64, dt: f64) {
        let new_vel = vec3_add(self.velocity, vec3_scale(steering, dt));
        let speed = vec3_len(new_vel);
        self.velocity = if speed > max_speed {
            vec3_scale(vec3_normalize(new_vel), max_speed)
        } else {
            new_vel
        };
    }
}

// ── FlockingBehavior ──────────────────────────────────────────────────────────

/// Parameters for Reynolds flocking (separation, alignment, cohesion).
#[derive(Clone, Debug)]
pub struct FlockingBehavior {
    /// Separation radius (repulsion below this distance).
    pub r_separation: f64,
    /// Weight for separation force.
    pub w_separation: f64,
    /// Weight for alignment force.
    pub w_alignment: f64,
    /// Weight for cohesion force.
    pub w_cohesion: f64,
    /// Maximum steering force magnitude.
    pub max_force: f64,
    /// Maximum agent speed.
    pub max_speed: f64,
}

impl FlockingBehavior {
    /// Create a new FlockingBehavior with default weights.
    pub fn new(r_separation: f64) -> Self {
        Self {
            r_separation,
            w_separation: 1.5,
            w_alignment: 1.0,
            w_cohesion: 1.0,
            max_force: 2.0,
            max_speed: 5.0,
        }
    }

    /// Compute separation steering: steer away from too-close neighbors.
    ///
    /// Returns a steering vector away from neighbors closer than `r_separation`.
    pub fn separation(&self, agent: &SwarmAgent, neighbors: &[&SwarmAgent]) -> [f64; 3] {
        let mut steer = [0.0_f64; 3];
        let mut count = 0usize;
        for nb in neighbors {
            let d = vec3_dist(agent.position, nb.position);
            if d < self.r_separation && d > 1e-12 {
                // Weight by inverse distance
                let away = vec3_normalize(vec3_sub(agent.position, nb.position));
                let weighted = vec3_scale(away, 1.0 / d);
                steer = vec3_add(steer, weighted);
                count += 1;
            }
        }
        if count > 0 {
            steer = vec3_scale(steer, 1.0 / count as f64);
            let speed = vec3_len(steer);
            if speed > 1e-12 {
                steer = vec3_scale(vec3_normalize(steer), self.max_speed);
                steer = vec3_sub(steer, agent.velocity);
                steer = clamp_vec3_magnitude(steer, self.max_force);
            }
        }
        steer
    }

    /// Compute alignment steering: match average velocity of neighbors.
    pub fn alignment(&self, agent: &SwarmAgent, neighbors: &[&SwarmAgent]) -> [f64; 3] {
        if neighbors.is_empty() {
            return [0.0; 3];
        }
        let mut avg_vel = [0.0_f64; 3];
        for nb in neighbors {
            avg_vel = vec3_add(avg_vel, nb.velocity);
        }
        avg_vel = vec3_scale(avg_vel, 1.0 / neighbors.len() as f64);
        // Steer toward average velocity
        let steer = vec3_sub(avg_vel, agent.velocity);
        clamp_vec3_magnitude(steer, self.max_force)
    }

    /// Compute cohesion steering: steer toward average position of neighbors.
    pub fn cohesion(&self, agent: &SwarmAgent, neighbors: &[&SwarmAgent]) -> [f64; 3] {
        if neighbors.is_empty() {
            return [0.0; 3];
        }
        let mut center = [0.0_f64; 3];
        for nb in neighbors {
            center = vec3_add(center, nb.position);
        }
        center = vec3_scale(center, 1.0 / neighbors.len() as f64);
        let desired_dir = vec3_sub(center, agent.position);
        let desired = if vec3_len(desired_dir) > 1e-12 {
            vec3_scale(vec3_normalize(desired_dir), self.max_speed)
        } else {
            [0.0; 3]
        };
        clamp_vec3_magnitude(vec3_sub(desired, agent.velocity), self.max_force)
    }

    /// Compute total flocking steering force for one agent.
    pub fn compute_steering(&self, agent: &SwarmAgent, neighbors: &[&SwarmAgent]) -> [f64; 3] {
        let sep = vec3_scale(self.separation(agent, neighbors), self.w_separation);
        let ali = vec3_scale(self.alignment(agent, neighbors), self.w_alignment);
        let coh = vec3_scale(self.cohesion(agent, neighbors), self.w_cohesion);
        let total = vec3_add(vec3_add(sep, ali), coh);
        clamp_vec3_magnitude(total, self.max_force)
    }

    /// Step an entire swarm by `dt` applying flocking rules.
    pub fn step_swarm(&self, agents: &mut [SwarmAgent], dt: f64) {
        // Gather all positions and velocities first (avoid borrow issues).
        let snapshots: Vec<SwarmAgent> = agents.to_vec();
        for (i, agent) in agents.iter_mut().enumerate() {
            let nbs: Vec<&SwarmAgent> = snapshots
                .iter()
                .filter(|a| {
                    a.id != agent.id && vec3_dist(a.position, agent.position) <= agent.sensor_radius
                })
                .collect();
            let steering = self.compute_steering(agent, &nbs);
            agent.apply_steering(steering, self.max_speed, dt);
            agent.integrate(dt);
            let _ = i;
        }
    }
}

/// Clamp a vector's magnitude to `max`.
fn clamp_vec3_magnitude(v: [f64; 3], max: f64) -> [f64; 3] {
    let mag = vec3_len(v);
    if mag > max && mag > 1e-15 {
        vec3_scale(vec3_normalize(v), max)
    } else {
        v
    }
}

// ── FormationControl ──────────────────────────────────────────────────────────

/// Formation type for rigid structure control.
#[derive(Clone, Debug, PartialEq)]
pub enum FormationType {
    /// Virtual structure: every agent tracks its slot (an offset from the
    /// virtual center) of a rigid structure that moves with the virtual
    /// velocity.
    VirtualStructure,
    /// Consensus-based: the agents agree on a common formation origin through
    /// their neighbours, without a leader and without a virtual center.
    Consensus,
}

/// Rigid formation maintenance controller.
///
/// Every agent is steered by an acceleration command that
/// [`step`](FormationControl::step) integrates with semi-implicit Euler; the
/// law depends on the [`FormationType`]:
///
/// * [`VirtualStructure`](FormationType::VirtualStructure): the agent tracks
///   its slot `virtual_center + offsets[i]` of a rigid structure that moves
///   with `virtual_velocity`:
///   `a_i = gain·(slot_i − p_i) + damping·(virtual_velocity − v_i)`.
///   The velocity of the structure is fed forward, so with the default
///   critical damping the tracking error of a structure moving at a constant
///   velocity decays without steady-state lag and, up to the discretisation of
///   the time integration, without overshoot.
/// * [`Consensus`](FormationType::Consensus): there is no leader; the agents
///   agree on a common formation origin through the offset-corrected positions
///   `q_i = p_i − offsets[i]`:
///   `a_i = gain·Σ_{j∈N_i}(q_j − q_i) + damping·(mean_{j∈N_i} v_j − v_i)`,
///   where `N_i` are the other formation members within the `sensor_radius` of
///   agent `i`.  The formation keeps its shape wherever it ends up;
///   `virtual_center` and `virtual_velocity` play no role.
///
/// Only the first `offsets.len()` agents are members of the formation; agents
/// beyond that are not steered (and are invisible to a consensus).
#[derive(Clone, Debug)]
pub struct FormationControl {
    /// Type of formation control.
    pub formation_type: FormationType,
    /// Desired offsets from virtual center for each agent (indexed by agent position in team).
    pub offsets: Vec<[f64; 3]>,
    /// Position of the virtual center.
    pub virtual_center: [f64; 3],
    /// Virtual center velocity.
    pub virtual_velocity: [f64; 3],
    /// Control gain for position error (a stiffness in 1/s²).
    pub gain: f64,
    /// Control gain for velocity error (a damping rate in 1/s).  Without it a
    /// formation oscillates forever.  [`new`](FormationControl::new) chooses
    /// the critical value `2·√gain`; replace it with
    /// [`with_damping`](FormationControl::with_damping).
    pub damping: f64,
}

/// The critical damping `2·√gain` of the second-order law `a = gain·e − c·ė`
/// (zero for a gain that is not positive).
fn critical_damping(gain: f64) -> f64 {
    2.0 * gain.max(0.0).sqrt()
}

impl FormationControl {
    /// Create a new formation controller.  The velocity damping is critical,
    /// `2·√gain`.
    pub fn new(formation_type: FormationType, offsets: Vec<[f64; 3]>, gain: f64) -> Self {
        Self {
            formation_type,
            offsets,
            virtual_center: [0.0; 3],
            virtual_velocity: [0.0; 3],
            gain,
            damping: critical_damping(gain),
        }
    }

    /// Replace the velocity damping gain.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Compute desired positions for all agents.
    pub fn desired_positions(&self) -> Vec<[f64; 3]> {
        self.offsets
            .iter()
            .map(|&off| vec3_add(self.virtual_center, off))
            .collect()
    }

    /// Compute the steering acceleration of agent `agent_index` of `agents`
    /// for the formation law of [`formation_type`](FormationControl::formation_type)
    /// (see [`FormationControl`]).
    ///
    /// The consensus law needs the other agents, hence the whole slice.
    /// Returns zero for an agent without an offset (`agent_index` beyond
    /// `agents` or `offsets`).
    pub fn formation_steering(&self, agents: &[SwarmAgent], agent_index: usize) -> [f64; 3] {
        let (Some(agent), Some(offset)) = (agents.get(agent_index), self.offsets.get(agent_index))
        else {
            return [0.0; 3];
        };
        match self.formation_type {
            FormationType::VirtualStructure => {
                let slot = vec3_add(self.virtual_center, *offset);
                let position_term = vec3_scale(vec3_sub(slot, agent.position), self.gain);
                let velocity_term = vec3_scale(
                    vec3_sub(self.virtual_velocity, agent.velocity),
                    self.damping,
                );
                vec3_add(position_term, velocity_term)
            }
            FormationType::Consensus => self.consensus_steering(agents, agent_index),
        }
    }

    /// The consensus law for the member `agent_index` (which must have an
    /// offset): the neighbours are the other members within its sensor radius.
    fn consensus_steering(&self, agents: &[SwarmAgent], agent_index: usize) -> [f64; 3] {
        let (Some(agent), Some(offset)) = (agents.get(agent_index), self.offsets.get(agent_index))
        else {
            return [0.0; 3];
        };
        let own_origin = vec3_sub(agent.position, *offset);
        let mut position_sum = [0.0_f64; 3];
        let mut velocity_sum = [0.0_f64; 3];
        let mut count = 0usize;
        for (j, (other, other_offset)) in agents.iter().zip(self.offsets.iter()).enumerate() {
            // A NaN sensor radius sees nobody (the comparison is false).
            let in_range = vec3_dist(agent.position, other.position) <= agent.sensor_radius;
            if j == agent_index || !in_range {
                continue;
            }
            let other_origin = vec3_sub(other.position, *other_offset);
            position_sum = vec3_add(position_sum, vec3_sub(other_origin, own_origin));
            velocity_sum = vec3_add(velocity_sum, other.velocity);
            count += 1;
        }
        if count == 0 {
            return [0.0; 3];
        }
        let mean_velocity = vec3_scale(velocity_sum, 1.0 / count as f64);
        let position_term = vec3_scale(position_sum, self.gain);
        let velocity_term = vec3_scale(vec3_sub(mean_velocity, agent.velocity), self.damping);
        vec3_add(position_term, velocity_term)
    }

    /// Formation tracking error: mean distance of agents to desired positions.
    pub fn tracking_error(&self, agents: &[SwarmAgent]) -> f64 {
        let n = agents.len().min(self.offsets.len());
        if n == 0 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .map(|i| {
                let desired = vec3_add(self.virtual_center, self.offsets[i]);
                vec3_dist(agents[i].position, desired)
            })
            .sum();
        total / n as f64
    }

    /// Relative offset error: mean distance of the offset-corrected positions
    /// `p_i − offsets[i]` of the formation members from their centroid.
    ///
    /// It is zero exactly when every separation `p_i − p_j` equals the
    /// designed `offsets[i] − offsets[j]`, wherever the formation as a whole
    /// is; this is the error a [`Consensus`](FormationType::Consensus)
    /// formation drives to zero.
    pub fn relative_offset_error(&self, agents: &[SwarmAgent]) -> f64 {
        let n = agents.len().min(self.offsets.len());
        if n == 0 {
            return 0.0;
        }
        let origins: Vec<[f64; 3]> = agents
            .iter()
            .zip(self.offsets.iter())
            .map(|(agent, offset)| vec3_sub(agent.position, *offset))
            .collect();
        let sum = origins
            .iter()
            .fold([0.0_f64; 3], |acc, q| vec3_add(acc, *q));
        let centroid = vec3_scale(sum, 1.0 / n as f64);
        origins.iter().map(|q| vec3_dist(*q, centroid)).sum::<f64>() / n as f64
    }

    /// The error that measures the formation type: the
    /// [`tracking_error`](FormationControl::tracking_error) for a virtual
    /// structure, the
    /// [`relative_offset_error`](FormationControl::relative_offset_error) for a
    /// consensus formation.
    pub fn formation_error(&self, agents: &[SwarmAgent]) -> f64 {
        match self.formation_type {
            FormationType::VirtualStructure => self.tracking_error(agents),
            FormationType::Consensus => self.relative_offset_error(agents),
        }
    }

    /// Step the virtual center forward by `dt`.
    pub fn advance_center(&mut self, dt: f64) {
        self.virtual_center = vec3_add(self.virtual_center, vec3_scale(self.virtual_velocity, dt));
    }

    /// Step the agents of the formation by `dt`: give every agent its
    /// [`formation_steering`](FormationControl::formation_steering)
    /// acceleration, advance the virtual center and integrate the agents
    /// (velocity first, then position).
    ///
    /// All accelerations are computed from the same snapshot of the agents and
    /// the virtual center at the start of the step, so the result does not
    /// depend on the order of `agents`, and an agent that moves with the
    /// virtual center has no steady-state lag.
    pub fn step(&mut self, agents: &mut [SwarmAgent], dt: f64) {
        let accelerations: Vec<[f64; 3]> = (0..agents.len())
            .map(|i| self.formation_steering(agents, i))
            .collect();
        self.advance_center(dt);
        for (agent, acceleration) in agents.iter_mut().zip(accelerations) {
            agent.velocity = vec3_add(agent.velocity, vec3_scale(acceleration, dt));
            agent.integrate(dt);
        }
    }
}

// ── ConsensusProtocol ─────────────────────────────────────────────────────────

/// Multi-agent consensus (distributed averaging) protocol.
///
/// Each agent updates its state as:
/// `x_dot_i = Σ_{j ∈ N_i} (x_j - x_i)`
#[derive(Clone, Debug)]
pub struct ConsensusProtocol {
    /// Current scalar state values for each agent.
    pub states: Vec<f64>,
    /// Adjacency list: `neighbors[i]` contains indices of agent `i`'s neighbors.
    pub neighbors: Vec<Vec<usize>>,
    /// Convergence gain (scales the update step).
    pub gain: f64,
}

impl ConsensusProtocol {
    /// Create a fully-connected consensus protocol.
    pub fn new_fully_connected(initial_states: Vec<f64>, gain: f64) -> Self {
        let n = initial_states.len();
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect();
        Self {
            states: initial_states,
            neighbors,
            gain,
        }
    }

    /// Create a consensus protocol with explicit neighbor graph.
    pub fn new(initial_states: Vec<f64>, neighbors: Vec<Vec<usize>>, gain: f64) -> Self {
        Self {
            states: initial_states,
            neighbors,
            gain,
        }
    }

    /// Perform one consensus update step with timestep `dt`.
    ///
    /// `x_i += dt * gain * Σ_{j ∈ N_i} (x_j - x_i)`
    pub fn step(&mut self, dt: f64) {
        let old = self.states.clone();
        for i in 0..self.states.len() {
            let update: f64 = self.neighbors[i]
                .iter()
                .map(|&j| old[j] - old[i])
                .sum::<f64>();
            self.states[i] += dt * self.gain * update;
        }
    }

    /// Check convergence: max deviation from mean < `tol`.
    pub fn is_converged(&self, tol: f64) -> bool {
        let mean = self.states.iter().sum::<f64>() / self.states.len() as f64;
        self.states.iter().all(|&x| (x - mean).abs() < tol)
    }

    /// The second-smallest eigenvalue of the graph Laplacian `L = D − A` (the
    /// Fiedler value or algebraic connectivity).
    ///
    /// It is the convergence rate of the consensus: the disagreement between
    /// the agents decays like `exp(−gain·λ₂·t)` under
    /// [`step`](ConsensusProtocol::step).
    ///
    /// The graph is the undirected one in which agents `i` and `j` are linked
    /// if either of them lists the other as a neighbour; self-links and
    /// indices beyond the number of agents are ignored.  `L` is therefore
    /// symmetric, and all its eigenvalues are computed to rounding accuracy
    /// with cyclic Jacobi rotations.  The result is `0` for a graph that is
    /// not connected and for fewer than two agents, `n` for the complete graph
    /// on `n` agents and `2·(1 − cos(π/n))` for a path of `n` agents.
    pub fn laplacian_second_eigenvalue(&self) -> f64 {
        let n = self.states.len();
        if n < 2 {
            return 0.0;
        }
        let mut laplacian = vec![0.0_f64; n * n];
        for (i, links) in self.neighbors.iter().enumerate().take(n) {
            for &j in links {
                // `-1` marks an edge that was already counted (listed twice, or
                // listed by both of its ends).
                if j >= n || j == i || laplacian[i * n + j] != 0.0 {
                    continue;
                }
                laplacian[i * n + j] = -1.0;
                laplacian[j * n + i] = -1.0;
                laplacian[i * n + i] += 1.0;
                laplacian[j * n + j] += 1.0;
            }
        }
        let mut eigenvalues = symmetric_eigenvalues(laplacian, n);
        eigenvalues.sort_by(f64::total_cmp);
        // The smallest eigenvalue of a Laplacian is 0 and the next one is the
        // Fiedler value; rounding may leave it a hair below zero.
        eigenvalues.get(1).copied().unwrap_or(0.0).max(0.0)
    }

    /// Former name of [`laplacian_second_eigenvalue`](ConsensusProtocol::laplacian_second_eigenvalue).
    ///
    /// It used to return `min_degree / n`, which is not an eigenvalue of the
    /// Laplacian (the complete graph on `n` agents gave `(n − 1) / n` instead
    /// of `n`); it now returns the exact Fiedler value.
    #[deprecated(
        since = "0.1.3",
        note = "this is no longer an approximation: use `laplacian_second_eigenvalue`"
    )]
    pub fn laplacian_second_eigenvalue_approx(&self) -> f64 {
        self.laplacian_second_eigenvalue()
    }

    /// Return the mean of all agent states.
    pub fn mean_state(&self) -> f64 {
        self.states.iter().sum::<f64>() / self.states.len() as f64
    }
}

/// Most sweeps [`symmetric_eigenvalues`] performs; the iteration converges
/// quadratically, so a handful of sweeps is all that is ever needed.
const MAX_JACOBI_SWEEPS: usize = 64;

/// The off-diagonal norm, relative to the norm of the whole matrix, at which
/// [`symmetric_eigenvalues`] stops.
const JACOBI_TOLERANCE: f64 = 1e-12;

/// All eigenvalues (in no particular order) of the symmetric `n × n` matrix
/// `a` (row-major), found with cyclic Jacobi rotations.
///
/// Every rotation annihilates one off-diagonal pair and keeps the matrix
/// symmetric and similar to the original, so the diagonal converges to the
/// eigenvalues.
fn symmetric_eigenvalues(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    let threshold = JACOBI_TOLERANCE * a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..MAX_JACOBI_SWEEPS {
        let mut off_diagonal_sq = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diagonal_sq += a[p * n + q] * a[p * n + q];
            }
        }
        // The matrix is symmetric, so its off-diagonal part counts every pair twice.
        if (2.0 * off_diagonal_sq).sqrt() <= threshold {
            break;
        }
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                // Rotation angle with tan(φ) = t that zeroes a[p][q].
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                a[p * n + p] -= t * apq;
                a[q * n + q] += t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_rp = c * arp - s * arq;
                    let new_rq = s * arp + c * arq;
                    a[r * n + p] = new_rp;
                    a[p * n + r] = new_rp;
                    a[r * n + q] = new_rq;
                    a[q * n + r] = new_rq;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

// ── CoverageControl ───────────────────────────────────────────────────────────

/// Index of the agent nearest to `point` (by squared distance), `None` if
/// there is no agent.  Of several equally near agents the first one wins, so
/// every Voronoi query of [`CoverageControl`] partitions the plane the same way.
fn nearest_agent(agents: &[SwarmAgent], point: [f64; 3]) -> Option<usize> {
    let mut nearest = None;
    let mut best = f64::INFINITY;
    for (i, agent) in agents.iter().enumerate() {
        let d = vec3_sub(agent.position, point);
        let distance_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if distance_sq < best {
            best = distance_sq;
            nearest = Some(i);
        }
    }
    nearest
}

/// Voronoi-based multi-agent coverage control.
///
/// Each agent moves toward the centroid of its Voronoi cell within
/// a bounded region.
#[derive(Clone, Debug)]
pub struct CoverageControl {
    /// Region bounds: \[\[x_min, x_max\], \[y_min, y_max\]\].
    pub bounds: [[f64; 2]; 2],
    /// Number of Monte-Carlo samples for Voronoi centroid estimation.
    pub n_samples: usize,
    /// Control gain.
    pub gain: f64,
}

impl CoverageControl {
    /// Create a new coverage controller.
    pub fn new(bounds: [[f64; 2]; 2], n_samples: usize, gain: f64) -> Self {
        Self {
            bounds,
            n_samples,
            gain,
        }
    }

    /// Compute the Voronoi centroids of all agents using grid sampling.
    ///
    /// The region is sampled on a grid of about `n_samples` cell centres.  One
    /// pass over the grid assigns every sample point to its nearest agent (the
    /// first of the nearest ones if several are equally near) and accumulates
    /// the sum and the count of the points per agent, so the cost is
    /// `O(samples · agents)` for all centroids together.  The centroid of an
    /// agent is the mean of its points; an agent whose cell contains no sample
    /// point keeps its own position.
    ///
    /// The result has one entry per agent, in the order of `agents`.
    pub fn voronoi_centroids(&self, agents: &[SwarmAgent]) -> Vec<[f64; 3]> {
        let grid_side = (self.n_samples as f64).sqrt() as usize + 1;
        let dx = (self.bounds[0][1] - self.bounds[0][0]) / grid_side as f64;
        let dy = (self.bounds[1][1] - self.bounds[1][0]) / grid_side as f64;
        let mut sums = vec![[0.0_f64; 3]; agents.len()];
        let mut counts = vec![0usize; agents.len()];
        for ix in 0..grid_side {
            let x = self.bounds[0][0] + (ix as f64 + 0.5) * dx;
            for iy in 0..grid_side {
                let y = self.bounds[1][0] + (iy as f64 + 0.5) * dy;
                let pt = [x, y, 0.0];
                if let Some(nearest) = nearest_agent(agents, pt) {
                    sums[nearest] = vec3_add(sums[nearest], pt);
                    counts[nearest] += 1;
                }
            }
        }
        agents
            .iter()
            .zip(sums.into_iter().zip(counts))
            .map(|(agent, (sum, count))| {
                if count > 0 {
                    vec3_scale(sum, 1.0 / count as f64)
                } else {
                    agent.position
                }
            })
            .collect()
    }

    /// Compute the Voronoi centroid for agent `idx` using grid sampling.
    ///
    /// The centroid is the average of all grid points closest to agent `idx`
    /// (see [`voronoi_centroids`](CoverageControl::voronoi_centroids), which is
    /// cheaper when the centroids of several agents are needed).  An agent
    /// whose cell contains no grid point keeps its own position.  Returns
    /// `None` if `idx` is not the index of an agent.
    pub fn voronoi_centroid(&self, agents: &[SwarmAgent], idx: usize) -> Option<[f64; 3]> {
        if idx >= agents.len() {
            return None;
        }
        self.voronoi_centroids(agents).get(idx).copied()
    }

    /// Compute coverage quality: max distance from any point to nearest agent.
    ///
    /// Sampled over a coarse grid.
    pub fn coverage_quality(&self, agents: &[SwarmAgent]) -> f64 {
        if agents.is_empty() {
            return f64::INFINITY;
        }
        let grid_side = 10usize;
        let dx = (self.bounds[0][1] - self.bounds[0][0]) / grid_side as f64;
        let dy = (self.bounds[1][1] - self.bounds[1][0]) / grid_side as f64;
        let mut max_dist = 0.0_f64;
        for ix in 0..grid_side {
            for iy in 0..grid_side {
                let x = self.bounds[0][0] + (ix as f64 + 0.5) * dx;
                let y = self.bounds[1][0] + (iy as f64 + 0.5) * dy;
                let pt = [x, y, 0.0];
                let min_d = agents
                    .iter()
                    .map(|a| vec3_dist(a.position, pt))
                    .fold(f64::INFINITY, f64::min);
                if min_d > max_dist {
                    max_dist = min_d;
                }
            }
        }
        max_dist
    }

    /// Move each agent toward its Voronoi centroid by one step.
    ///
    /// The centroids of all agents are computed once, in a single pass over
    /// the sample grid, before any agent moves.
    pub fn step(&self, agents: &mut [SwarmAgent], dt: f64) {
        let centroids = self.voronoi_centroids(agents);
        for (agent, centroid) in agents.iter_mut().zip(centroids) {
            let err = vec3_sub(centroid, agent.position);
            let steer = vec3_scale(err, self.gain);
            agent.velocity = steer;
            agent.integrate(dt);
        }
    }
}

// ── TaskAllocation ────────────────────────────────────────────────────────────

/// A task that can be assigned to an agent.
#[derive(Clone, Debug)]
pub struct Task {
    /// Task identifier.
    pub id: usize,
    /// Task location.
    pub position: [f64; 3],
    /// Task utility value.
    pub utility: f64,
}

/// Result of a task allocation.
#[derive(Clone, Debug)]
pub struct AllocationResult {
    /// For each task, the assigned agent index (if any).
    pub task_to_agent: Vec<Option<usize>>,
    /// For each agent, the assigned task index (if any).
    pub agent_to_task: Vec<Option<usize>>,
}

/// Auction-based greedy task allocator.
///
/// Bid = utility - cost, where cost = distance from agent to task.
#[derive(Clone, Debug)]
pub struct TaskAllocation {
    /// Cost weight for distance.
    pub cost_weight: f64,
}

impl TaskAllocation {
    /// Create a new task allocator.
    pub fn new(cost_weight: f64) -> Self {
        Self { cost_weight }
    }

    /// Compute bid of agent `i` for task `t`.
    ///
    /// `bid = utility - cost_weight * distance(agent, task)`
    pub fn bid(&self, agent: &SwarmAgent, task: &Task) -> f64 {
        let cost = self.cost_weight * vec3_dist(agent.position, task.position);
        task.utility - cost
    }

    /// Greedy auction: assign each task to the highest-bidding unassigned agent.
    ///
    /// Each agent may receive at most one task.
    pub fn allocate(&self, agents: &[SwarmAgent], tasks: &[Task]) -> AllocationResult {
        let mut task_to_agent: Vec<Option<usize>> = vec![None; tasks.len()];
        let mut agent_to_task: Vec<Option<usize>> = vec![None; agents.len()];
        // For each task, find the best unassigned agent.
        for (ti, task) in tasks.iter().enumerate() {
            let mut best_agent = None;
            let mut best_bid = f64::NEG_INFINITY;
            for (ai, agent) in agents.iter().enumerate() {
                if agent_to_task[ai].is_none() {
                    let b = self.bid(agent, task);
                    if b > best_bid {
                        best_bid = b;
                        best_agent = Some(ai);
                    }
                }
            }
            if let Some(ai) = best_agent {
                task_to_agent[ti] = Some(ai);
                agent_to_task[ai] = Some(ti);
            }
        }
        AllocationResult {
            task_to_agent,
            agent_to_task,
        }
    }

    /// Count the number of assigned tasks.
    pub fn assigned_count(result: &AllocationResult) -> usize {
        result.task_to_agent.iter().filter(|a| a.is_some()).count()
    }
}

// ── SwarmSearch (PSO) ─────────────────────────────────────────────────────────

/// A particle in PSO (Particle Swarm Optimization).
#[derive(Clone, Debug)]
pub struct Particle {
    /// Current position in search space.
    pub position: [f64; 3],
    /// Current velocity.
    pub velocity: [f64; 3],
    /// Personal best position.
    pub personal_best: [f64; 3],
    /// Personal best fitness value.
    pub personal_best_fitness: f64,
}

impl Particle {
    /// Create a new particle at the given position.
    pub fn new(position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            personal_best: position,
            personal_best_fitness: f64::INFINITY,
        }
    }
}

/// Particle Swarm Optimization for swarm search.
///
/// Update rule:
/// `v_i = w*v_i + c1*r1*(p_best - x_i) + c2*r2*(g_best - x_i)`
/// `x_i += v_i`
#[derive(Clone, Debug)]
pub struct SwarmSearch {
    /// PSO particles.
    pub particles: Vec<Particle>,
    /// Global best position found.
    pub global_best: [f64; 3],
    /// Global best fitness value.
    pub global_best_fitness: f64,
    /// Inertia weight `w`.
    pub inertia: f64,
    /// Cognitive coefficient `c1`.
    pub c1: f64,
    /// Social coefficient `c2`.
    pub c2: f64,
    /// Maximum velocity magnitude.
    pub max_velocity: f64,
}

impl SwarmSearch {
    /// Create a new PSO with given particles.
    pub fn new(
        particles: Vec<Particle>,
        inertia: f64,
        c1: f64,
        c2: f64,
        max_velocity: f64,
    ) -> Self {
        let global_best = if particles.is_empty() {
            [0.0; 3]
        } else {
            particles[0].position
        };
        Self {
            particles,
            global_best,
            global_best_fitness: f64::INFINITY,
            inertia,
            c1,
            c2,
            max_velocity,
        }
    }

    /// Evaluate a fitness function for all particles and update bests.
    ///
    /// `fitness_fn` maps position to scalar fitness (lower = better).
    pub fn evaluate<F: Fn([f64; 3]) -> f64>(&mut self, fitness_fn: &F) {
        for particle in &mut self.particles {
            let f = fitness_fn(particle.position);
            if f < particle.personal_best_fitness {
                particle.personal_best_fitness = f;
                particle.personal_best = particle.position;
            }
            if f < self.global_best_fitness {
                self.global_best_fitness = f;
                self.global_best = particle.position;
            }
        }
    }

    /// Perform one PSO update step using deterministic r1=r2=0.5.
    ///
    /// For reproducibility in tests, uses fixed r1=r2=0.5.
    pub fn step_deterministic(&mut self) {
        let r1 = 0.5;
        let r2 = 0.5;
        for particle in &mut self.particles {
            for d in 0..3 {
                let cognitive = self.c1 * r1 * (particle.personal_best[d] - particle.position[d]);
                let social = self.c2 * r2 * (self.global_best[d] - particle.position[d]);
                particle.velocity[d] = self.inertia * particle.velocity[d] + cognitive + social;
            }
            // Clamp velocity
            let speed = vec3_len(particle.velocity);
            if speed > self.max_velocity {
                particle.velocity =
                    vec3_scale(vec3_normalize(particle.velocity), self.max_velocity);
            }
            // Update position
            particle.position = vec3_add(particle.position, particle.velocity);
        }
    }

    /// Perform one PSO update step using provided random values r1, r2.
    pub fn step_with_random(&mut self, r1: f64, r2: f64) {
        for particle in &mut self.particles {
            for d in 0..3 {
                let cognitive = self.c1 * r1 * (particle.personal_best[d] - particle.position[d]);
                let social = self.c2 * r2 * (self.global_best[d] - particle.position[d]);
                particle.velocity[d] = self.inertia * particle.velocity[d] + cognitive + social;
            }
            let speed = vec3_len(particle.velocity);
            if speed > self.max_velocity {
                particle.velocity =
                    vec3_scale(vec3_normalize(particle.velocity), self.max_velocity);
            }
            particle.position = vec3_add(particle.position, particle.velocity);
        }
    }

    /// Return the current global best position.
    pub fn best_position(&self) -> [f64; 3] {
        self.global_best
    }

    /// Return the current global best fitness.
    pub fn best_fitness(&self) -> f64 {
        self.global_best_fitness
    }
}

// ── EmergentBehavior (Stigmergy) ──────────────────────────────────────────────

/// A pheromone/trace cell in the stigmergy environment grid.
#[derive(Clone, Debug)]
pub struct PheromoneCell {
    /// Pheromone concentration.
    pub concentration: f64,
    /// Decay rate in 1/time: the concentration decays like `exp(−decay_rate·t)`.
    pub decay_rate: f64,
}

impl PheromoneCell {
    /// Create a new pheromone cell.
    pub fn new(concentration: f64, decay_rate: f64) -> Self {
        Self {
            concentration,
            decay_rate,
        }
    }

    /// Decay the pheromone by one timestep `dt`.
    ///
    /// A cell with a decay rate of zero (or a zero `dt`) does not lose anything,
    /// not even the traces below the cut-off of `1e-12` that a decaying cell
    /// rounds to zero.
    pub fn decay(&mut self, dt: f64) {
        let exponent = -self.decay_rate * dt;
        if exponent == 0.0 {
            return;
        }
        self.concentration *= exponent.exp();
        if self.concentration < 1e-12 {
            self.concentration = 0.0;
        }
    }

    /// Deposit additional pheromone.
    pub fn deposit(&mut self, amount: f64) {
        self.concentration += amount;
    }
}

/// Fraction of the explicit-Euler stability limit that one diffusion sub-step
/// of [`EmergentBehavior::step`] may use.
const DIFFUSION_STABILITY_FRACTION: f64 = 0.9;

/// Most diffusion sub-steps [`EmergentBehavior::step`] takes per call; beyond
/// that the diffusion rate is reduced to what the stability limit allows.
const MAX_DIFFUSION_SUBSTEPS: usize = 10_000;

/// Stigmergy environment: indirect agent communication via environment.
///
/// Agents deposit pheromone traces at their positions; the environment
/// diffuses and decays traces over time.
#[derive(Clone, Debug)]
pub struct EmergentBehavior {
    /// 2D grid of pheromone cells (row-major, \[y\]\[x\]).
    pub grid: Vec<Vec<PheromoneCell>>,
    /// Grid resolution: the number of cells along each side.  The world
    /// bounds are divided into `resolution × resolution` cells, so a cell is
    /// `(x_max − x_min) / resolution` wide and `(y_max − y_min) / resolution`
    /// high.
    pub resolution: usize,
    /// World bounds `[[x_min, x_max], [y_min, y_max]]`.
    pub bounds: [[f64; 2]; 2],
    /// Diffusion coefficient in world length² per unit time.
    pub diffusion: f64,
}

impl EmergentBehavior {
    /// Create a new stigmergy environment with `resolution × resolution` cells
    /// (at least one) over `bounds`.
    pub fn new(bounds: [[f64; 2]; 2], resolution: usize, decay_rate: f64, diffusion: f64) -> Self {
        let resolution = resolution.max(1);
        let grid = (0..resolution)
            .map(|_| {
                (0..resolution)
                    .map(|_| PheromoneCell::new(0.0, decay_rate))
                    .collect()
            })
            .collect();
        Self {
            grid,
            resolution,
            bounds,
            diffusion,
        }
    }

    /// Convert world position to grid indices `(ix, iy)`, clamped to grid.
    pub fn world_to_grid(&self, pos: [f64; 3]) -> (usize, usize) {
        let nx = self.grid[0].len();
        let ny = self.grid.len();
        let fx = (pos[0] - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]);
        let fy = (pos[1] - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]);
        let ix = ((fx * nx as f64) as isize).max(0).min(nx as isize - 1) as usize;
        let iy = ((fy * ny as f64) as isize).max(0).min(ny as isize - 1) as usize;
        (ix, iy)
    }

    /// Deposit pheromone at agent's position.
    pub fn deposit_at(&mut self, pos: [f64; 3], amount: f64) {
        let (ix, iy) = self.world_to_grid(pos);
        self.grid[iy][ix].deposit(amount);
    }

    /// Read pheromone concentration at position.
    pub fn concentration_at(&self, pos: [f64; 3]) -> f64 {
        let (ix, iy) = self.world_to_grid(pos);
        self.grid[iy][ix].concentration
    }

    /// Decay and diffuse all cells by `dt`.
    ///
    /// Diffusion solves `∂c/∂t = D·∇²c` on the grid with zero flux through the
    /// boundary: the Laplacian is `(W + E − 2C)/hx² + (S + N − 2C)/hy²` for the
    /// cell size `hx × hy`, so the result does not depend on how finely the
    /// bounds are divided.  The explicit scheme is stable only while
    /// `D·dt·(2/hx² + 2/hy²) ≤ 1`, so `dt` is split into sub-steps that stay at
    /// 90 % of that limit; the field then stays non-negative without any
    /// clamping, and the total amount of pheromone is conserved (decay aside).
    /// At most 10 000 sub-steps are taken per call; a diffusion so strong that
    /// it would need more is reduced to what the stability limit allows for
    /// them.
    ///
    /// A `dt` that is not positive and finite does nothing.
    pub fn step(&mut self, dt: f64) {
        if !(dt > 0.0 && dt.is_finite()) {
            return;
        }
        for row in &mut self.grid {
            for cell in row {
                cell.decay(dt);
            }
        }
        self.diffuse(dt);
    }

    /// Diffuse the pheromone over `dt` (see [`step`](EmergentBehavior::step)).
    fn diffuse(&mut self, dt: f64) {
        let ny = self.grid.len();
        let nx = self.grid.first().map_or(0, Vec::len);
        if nx == 0 || ny == 0 || self.grid.iter().any(|row| row.len() != nx) {
            return;
        }
        let hx = (self.bounds[0][1] - self.bounds[0][0]) / nx as f64;
        let hy = (self.bounds[1][1] - self.bounds[1][0]) / ny as f64;
        let usable = |x: f64| x > 0.0 && x.is_finite();
        if !(usable(self.diffusion) && usable(hx) && usable(hy)) {
            return;
        }
        let inv_hx2 = 1.0 / (hx * hx);
        let inv_hy2 = 1.0 / (hy * hy);
        let rate_sum = 2.0 * (inv_hx2 + inv_hy2);
        let wanted = (self.diffusion * dt * rate_sum / DIFFUSION_STABILITY_FRACTION).ceil();
        let substeps = if wanted.is_finite() {
            (wanted as usize).clamp(1, MAX_DIFFUSION_SUBSTEPS)
        } else {
            MAX_DIFFUSION_SUBSTEPS
        };
        let sub_dt = dt / substeps as f64;
        let diffusion = if wanted.is_finite() && wanted <= MAX_DIFFUSION_SUBSTEPS as f64 {
            self.diffusion
        } else {
            DIFFUSION_STABILITY_FRACTION / (sub_dt * rate_sum)
        };
        let (cx, cy) = (diffusion * sub_dt * inv_hx2, diffusion * sub_dt * inv_hy2);

        let mut field: Vec<f64> = self
            .grid
            .iter()
            .flat_map(|row| row.iter().map(|cell| cell.concentration))
            .collect();
        let mut next = field.clone();
        for _ in 0..substeps {
            for iy in 0..ny {
                for ix in 0..nx {
                    let i = iy * nx + ix;
                    let c = field[i];
                    // A missing neighbour is the cell itself: zero flux.
                    let west = if ix > 0 { field[i - 1] } else { c };
                    let east = if ix + 1 < nx { field[i + 1] } else { c };
                    let south = if iy > 0 { field[i - nx] } else { c };
                    let north = if iy + 1 < ny { field[i + nx] } else { c };
                    next[i] = c + cx * (west + east - 2.0 * c) + cy * (south + north - 2.0 * c);
                }
            }
            std::mem::swap(&mut field, &mut next);
        }
        for (cell, value) in self.grid.iter_mut().flatten().zip(field) {
            cell.concentration = value;
        }
    }

    /// Total pheromone in the grid.
    pub fn total_pheromone(&self) -> f64 {
        self.grid
            .iter()
            .flat_map(|r| r.iter())
            .map(|c| c.concentration)
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
