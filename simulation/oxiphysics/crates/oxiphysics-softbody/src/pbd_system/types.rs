//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
/// Constraint graph coloring result.
///
/// Groups constraints into independent sets (colors) so that all constraints
/// in the same color group can be solved in parallel (no shared particles).
#[derive(Debug, Clone)]
pub struct ConstraintColoring {
    /// Groups of constraint indices. Constraints in the same group share no particles.
    pub groups: Vec<Vec<usize>>,
}
impl ConstraintColoring {
    /// Build a greedy graph coloring of constraints.
    ///
    /// Two constraints are adjacent in the graph if they share any particle.
    /// This greedy algorithm assigns the first available color to each constraint.
    pub fn build(constraints: &[PbdConstraint], n_particles: usize) -> Self {
        let n = constraints.len();
        let mut colors = vec![usize::MAX; n];
        let mut particle_colors: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n_particles];
        for ci in 0..n {
            let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &pi in &constraints[ci].particles {
                if pi < n_particles {
                    for &c in &particle_colors[pi] {
                        used.insert(c);
                    }
                }
            }
            let mut color = 0;
            while used.contains(&color) {
                color += 1;
            }
            colors[ci] = color;
            for &pi in &constraints[ci].particles {
                if pi < n_particles {
                    particle_colors[pi].insert(color);
                }
            }
        }
        let max_color = colors
            .iter()
            .filter(|&&c| c != usize::MAX)
            .copied()
            .max()
            .unwrap_or(0);
        let mut groups = vec![Vec::new(); max_color + 1];
        for (ci, &c) in colors.iter().enumerate() {
            if c != usize::MAX {
                groups[c].push(ci);
            }
        }
        ConstraintColoring { groups }
    }
    /// Total number of colors (groups).
    pub fn num_colors(&self) -> usize {
        self.groups.len()
    }
}
/// A rigid body proxy: a single fixed anchor point with orientation.
///
/// Used to couple PBD cloth particles to a rigid object.
#[derive(Debug, Clone)]
pub struct RigidBodyProxy {
    /// Position of the rigid body centre.
    pub center: [f64; 3],
    /// Optional velocity (for coupling).
    pub velocity: [f64; 3],
    /// Radius for surface contact.
    pub radius: f64,
}
impl RigidBodyProxy {
    /// Create a new rigid body proxy.
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self {
            center,
            velocity: [0.0; 3],
            radius,
        }
    }
    /// Move the rigid body by `velocity * dt`.
    pub fn step(&mut self, dt: f64) {
        self.center = add3(self.center, scale3(self.velocity, dt));
    }
    /// Apply surface collision to a PBD particle.
    pub fn apply_collision(&self, particle: &mut PbdParticle, restitution: f64) {
        solve_sphere_collision(particle, self.center, self.radius, restitution);
    }
}
/// Warm-start cache: stores Lagrange multipliers from the previous frame
/// so they can be reused as an initial guess.
#[derive(Debug, Clone)]
pub struct WarmStartCache {
    /// Cached Lagrange multipliers per constraint.
    pub lambdas: Vec<f64>,
}
impl WarmStartCache {
    /// Create a new cache for `n` constraints, all zero.
    pub fn new(n: usize) -> Self {
        Self {
            lambdas: vec![0.0; n],
        }
    }
    /// Apply warm-start corrections to predicted positions.
    ///
    /// For each distance constraint, pre-applies the cached lambda correction
    /// scaled by `warm_start_factor` (typically 0.0–1.0).
    pub fn apply_warm_start(
        &self,
        particles: &mut [PbdParticle],
        constraints: &[PbdConstraint],
        warm_start_factor: f64,
        dt: f64,
    ) {
        for (ci, c) in constraints.iter().enumerate() {
            if ci >= self.lambdas.len() {
                break;
            }
            if let PbdConstraintType::Distance { .. } = c.constraint_type {
                if c.particles.len() < 2 {
                    continue;
                }
                let (i, j) = (c.particles[0], c.particles[1]);
                let pi = particles[i].predicted;
                let pj = particles[j].predicted;
                let diff = sub3(pi, pj);
                let cur_len = len3(diff);
                if cur_len < 1e-12 {
                    continue;
                }
                let n_hat = scale3(diff, 1.0 / cur_len);
                let lambda_ws = self.lambdas[ci] * warm_start_factor;
                let wi = particles[i].inv_mass;
                let wj = particles[j].inv_mass;
                let correction = lambda_ws / (dt * dt).max(1e-30);
                particles[i].predicted = add3(pi, scale3(n_hat, correction * wi));
                particles[j].predicted = add3(pj, scale3(n_hat, -correction * wj));
            }
        }
    }
    /// Update cache after a solve pass.
    pub fn update(&mut self, constraints: &[PbdConstraint], particles: &[PbdParticle]) {
        for (ci, c) in constraints.iter().enumerate() {
            if ci >= self.lambdas.len() {
                break;
            }
            if let PbdConstraintType::Distance { rest_length, .. } = c.constraint_type {
                if c.particles.len() < 2 {
                    continue;
                }
                let (i, j) = (c.particles[0], c.particles[1]);
                let d = dist3(particles[i].position, particles[j].position);
                self.lambdas[ci] = d - rest_length;
            }
        }
    }
}
/// The kind of PBD constraint together with its parameters.
#[derive(Clone, Debug)]
pub enum PbdConstraintType {
    /// Keep two particles at a fixed distance.
    Distance {
        /// Rest (target) length.
        rest_length: f64,
        /// XPBD compliance (α). Zero = rigid.
        compliance: f64,
    },
    /// Dihedral bending constraint between four particles `[p0, p1, p2, p3]`:
    /// `p0`-`p1` is the edge shared by the triangles `(p0, p1, p2)` and
    /// `(p0, p1, p3)`.
    BendingDihedral {
        /// Rest interior dihedral angle (radians): `PI` for a flat hinge, below
        /// `PI` when the wings `p2`, `p3` are folded towards the side that the
        /// normal `(p1 - p0) x (p2 - p0)` points to, above `PI` when they are
        /// folded to the other side (see [`compute_dihedral_angle`]).  The
        /// difference to the current angle is taken modulo `2 PI`.
        rest_angle: f64,
        /// XPBD compliance.
        compliance: f64,
    },
    /// Volume conservation over a tetrahedron (four particles).
    VolumeConservation {
        /// Rest volume, signed like [`compute_tetrahedron_volume`]: negative for
        /// a tetrahedron whose vertices are wound the other way round, so that a
        /// tetrahedron at rest satisfies the constraint whatever its winding.
        rest_volume: f64,
        /// XPBD compliance.
        compliance: f64,
    },
    /// Unilateral floor collision for a single particle.
    CollisionFloor {
        /// Y coordinate of the floor plane.
        y_level: f64,
        /// Coefficient of restitution (0 = fully inelastic).
        restitution: f64,
    },
    /// Unilateral collision of a single particle with the outside of a sphere.
    CollisionSphere {
        /// Centre of the sphere.
        center: [f64; 3],
        /// Radius of the sphere.
        radius: f64,
        /// Coefficient of restitution (0 = fully inelastic), forwarded to
        /// [`solve_sphere_collision`], which applies it to the particle velocity
        /// during the projection.  Like for [`PbdConstraintType::CollisionFloor`],
        /// [`PbdSystem::step`] recomputes the velocities from the position
        /// change after the projections, which overrides that velocity.
        restitution: f64,
    },
}
/// A PBD constraint with an associated list of particle indices.
#[derive(Clone, Debug)]
pub struct PbdConstraint {
    /// What kind of constraint this is.
    pub constraint_type: PbdConstraintType,
    /// Indices into the parent system's particle array.
    pub particles: Vec<usize>,
}
/// PBD system with adaptive substep selection.
pub struct AdaptivePbdSystem {
    /// The inner PBD system (substeps updated each step).
    pub system: PbdSystem,
    /// Minimum substeps per step.
    pub min_substeps: u32,
    /// Maximum substeps per step.
    pub max_substeps: u32,
    /// Target CFL number for adaptive control.
    pub target_cfl: f64,
}
impl AdaptivePbdSystem {
    /// Create a new adaptive PBD system.
    pub fn new(gravity: [f64; 3], min_substeps: u32, max_substeps: u32) -> Self {
        Self {
            system: PbdSystem::new(gravity, min_substeps),
            min_substeps,
            max_substeps,
            target_cfl: 0.5,
        }
    }
    /// Step with adaptive substep count.
    pub fn step(&mut self, dt: f64) {
        let n = adaptive_substep_count(
            &self.system.particles,
            dt,
            self.target_cfl,
            self.min_substeps,
            self.max_substeps,
        );
        self.system.substeps = n;
        self.system.step(dt);
    }
}
/// Wake/sleep threshold controller for PBD particles.
///
/// Particles with kinetic energy below `sleep_threshold` for `sleep_frames`
/// consecutive frames are put to sleep (treated as fixed).
#[derive(Debug, Clone)]
pub struct SleepController {
    /// Energy threshold below which a particle is considered for sleeping.
    pub sleep_threshold: f64,
    /// Number of consecutive frames below threshold before sleeping.
    pub sleep_frames: u32,
    /// Per-particle frame counter (frames spent below threshold).
    pub counters: Vec<u32>,
    /// Per-particle sleep state.
    pub asleep: Vec<bool>,
}
impl SleepController {
    /// Create a new sleep controller for `n` particles.
    pub fn new(n: usize, sleep_threshold: f64, sleep_frames: u32) -> Self {
        Self {
            sleep_threshold,
            sleep_frames,
            counters: vec![0; n],
            asleep: vec![false; n],
        }
    }
    /// Update sleep state for all particles after a step.
    ///
    /// Returns the number of newly sleeping particles.
    pub fn update(&mut self, particles: &[PbdParticle]) -> usize {
        let mut newly_sleeping = 0;
        for (i, p) in particles.iter().enumerate() {
            if i >= self.counters.len() {
                break;
            }
            if p.fixed || p.inv_mass < 1e-30 {
                continue;
            }
            let m = 1.0 / p.inv_mass;
            let ke = 0.5 * m * dot3(p.velocity, p.velocity);
            if ke < self.sleep_threshold {
                self.counters[i] += 1;
                if self.counters[i] >= self.sleep_frames && !self.asleep[i] {
                    self.asleep[i] = true;
                    newly_sleeping += 1;
                }
            } else {
                self.counters[i] = 0;
                self.asleep[i] = false;
            }
        }
        newly_sleeping
    }
    /// Wake all sleeping particles.
    pub fn wake_all(&mut self) {
        for (c, s) in self.counters.iter_mut().zip(self.asleep.iter_mut()) {
            *c = 0;
            *s = false;
        }
    }
    /// Number of sleeping particles.
    pub fn sleeping_count(&self) -> usize {
        self.asleep.iter().filter(|&&s| s).count()
    }
}
/// Timing statistics for a single PBD step.
#[derive(Debug, Clone, Default)]
pub struct PbdStepProfile {
    /// Number of substeps executed.
    pub substeps: u32,
    /// Number of constraint solve iterations.
    pub total_constraint_solves: usize,
    /// Estimated constraint residual (sum of |C| across all constraints).
    pub residual: f64,
}
/// A complete PBD / XPBD simulation system.
pub struct PbdSystem {
    /// All particles in the simulation.
    pub particles: Vec<PbdParticle>,
    /// All constraints.
    pub constraints: Vec<PbdConstraint>,
    /// Gravitational acceleration.
    pub gravity: [f64; 3],
    /// Number of substeps per `step()` call.
    pub substeps: u32,
}
impl PbdSystem {
    /// Create an empty system.
    pub fn new(gravity: [f64; 3], substeps: u32) -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            gravity,
            substeps: substeps.max(1),
        }
    }
    /// Add a dynamic particle and return its index.
    pub fn add_particle(&mut self, pos: [f64; 3], mass: f64) -> usize {
        let idx = self.particles.len();
        self.particles.push(PbdParticle::new(pos, mass));
        idx
    }
    /// Add a fixed (pinned) particle and return its index.
    pub fn add_fixed_particle(&mut self, pos: [f64; 3]) -> usize {
        let idx = self.particles.len();
        self.particles.push(PbdParticle::new_fixed(pos));
        idx
    }
    /// Add a distance constraint between particles `i` and `j`.
    /// The rest length is automatically measured from their current positions.
    pub fn add_distance_constraint(&mut self, i: usize, j: usize, compliance: f64) {
        let rest_length = dist3(self.particles[i].position, self.particles[j].position);
        self.constraints.push(PbdConstraint {
            constraint_type: PbdConstraintType::Distance {
                rest_length,
                compliance,
            },
            particles: vec![i, j],
        });
    }
    /// Add a floor collision constraint for particle `i`.
    pub fn add_floor_constraint(&mut self, i: usize, y: f64, restitution: f64) {
        self.constraints.push(PbdConstraint {
            constraint_type: PbdConstraintType::CollisionFloor {
                y_level: y,
                restitution,
            },
            particles: vec![i],
        });
    }
    /// Number of particles in the system.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
    /// Total kinetic energy: ½ Σ mᵢ |vᵢ|².
    pub fn total_kinetic_energy(&self) -> f64 {
        self.particles
            .iter()
            .map(|p| {
                if p.inv_mass < 1e-30 {
                    return 0.0;
                }
                let m = 1.0 / p.inv_mass;
                let v2 = dot3(p.velocity, p.velocity);
                0.5 * m * v2
            })
            .sum()
    }
    /// Advance the simulation by time `dt` using XPBD substeps.
    ///
    /// Every substep predicts the positions, projects each constraint once, in
    /// the order of [`PbdSystem::constraints`], with [`project_constraint`] and
    /// derives the velocities from the position change.
    pub fn step(&mut self, dt: f64) {
        let sub_dt = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            for p in self.particles.iter_mut() {
                if p.fixed {
                    p.predicted = p.position;
                    continue;
                }
                let v = add3(p.velocity, scale3(self.gravity, sub_dt));
                p.predicted = add3(p.position, scale3(v, sub_dt));
            }
            for constraint in &self.constraints {
                project_constraint(&mut self.particles, constraint, sub_dt);
            }
            for p in self.particles.iter_mut() {
                if p.fixed {
                    continue;
                }
                p.velocity = scale3(sub3(p.predicted, p.position), 1.0 / sub_dt);
                p.position = p.predicted;
            }
        }
    }
}
impl PbdSystem {
    /// Add a sphere collision constraint for particle `i`: the particle is kept
    /// outside the sphere with the given `center` and `radius` (see
    /// [`PbdConstraintType::CollisionSphere`] for the role of `restitution`).
    pub fn add_sphere_constraint(
        &mut self,
        i: usize,
        center: [f64; 3],
        radius: f64,
        restitution: f64,
    ) {
        self.constraints.push(PbdConstraint {
            constraint_type: PbdConstraintType::CollisionSphere {
                center,
                radius,
                restitution,
            },
            particles: vec![i],
        });
    }
    /// Add a volume conservation constraint for four particles.
    ///
    /// The rest volume is the *signed* volume of the tetrahedron at its current
    /// positions, so a tetrahedron with reversed winding (negative volume) is at
    /// rest, not inverted, when the constraint is added.
    pub fn add_volume_constraint(
        &mut self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        compliance: f64,
    ) {
        let p0 = self.particles[i0].position;
        let p1 = self.particles[i1].position;
        let p2 = self.particles[i2].position;
        let p3 = self.particles[i3].position;
        let rest_volume = compute_tetrahedron_volume(p0, p1, p2, p3);
        self.constraints.push(PbdConstraint {
            constraint_type: PbdConstraintType::VolumeConservation {
                rest_volume,
                compliance,
            },
            particles: vec![i0, i1, i2, i3],
        });
    }
    /// Add a bending constraint between four particles.
    pub fn add_bending_constraint(
        &mut self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        rest_angle: f64,
        compliance: f64,
    ) {
        self.constraints.push(PbdConstraint {
            constraint_type: PbdConstraintType::BendingDihedral {
                rest_angle,
                compliance,
            },
            particles: vec![i0, i1, i2, i3],
        });
    }
    /// Apply velocity damping to all particles.
    pub fn apply_velocity_damping(&mut self, coeff: f64) {
        apply_velocity_damping(&mut self.particles, coeff);
    }
    /// Apply position damping (COM-based) to all particles.
    pub fn apply_position_damping(&mut self, alpha: f64) {
        apply_position_damping(&mut self.particles, alpha);
    }
    /// Compute the total potential energy (gravitational).
    pub fn total_potential_energy(&self) -> f64 {
        let g_mag = len3(self.gravity);
        self.particles
            .iter()
            .filter(|p| !p.fixed && p.inv_mass > 1e-30)
            .map(|p| {
                let m = 1.0 / p.inv_mass;
                -m * g_mag * p.position[1]
            })
            .sum()
    }
    /// Compute the total number of constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}
/// Profiled PBD system that records performance statistics each step.
pub struct ProfiledPbdSystem {
    /// Inner system.
    pub system: PbdSystem,
    /// Profile of the last step.
    pub last_profile: PbdStepProfile,
}
impl ProfiledPbdSystem {
    /// Create a new profiled system.
    pub fn new(gravity: [f64; 3], substeps: u32) -> Self {
        Self {
            system: PbdSystem::new(gravity, substeps),
            last_profile: PbdStepProfile::default(),
        }
    }
    /// Step the system and record profiling info.
    pub fn step(&mut self, dt: f64) {
        let n_constraints = self.system.constraints.len();
        let substeps = self.system.substeps;
        self.system.step(dt);
        self.last_profile = PbdStepProfile {
            substeps,
            total_constraint_solves: (substeps as usize) * n_constraints,
            residual: self.compute_residual(),
        };
    }
    fn compute_residual(&self) -> f64 {
        self.system
            .constraints
            .iter()
            .map(|c| constraint_violation(&self.system.particles, c))
            .sum()
    }
}
/// A single PBD particle.
#[derive(Clone, Debug)]
pub struct PbdParticle {
    /// Current position.
    pub position: [f64; 3],
    /// Predicted position (scratch space during solve).
    pub predicted: [f64; 3],
    /// Current velocity.
    pub velocity: [f64; 3],
    /// Inverse mass (0 for fixed/static particles).
    pub inv_mass: f64,
    /// Whether this particle is pinned.
    pub fixed: bool,
}
impl PbdParticle {
    /// Create a dynamic particle.
    pub fn new(pos: [f64; 3], mass: f64) -> Self {
        let inv_mass = if mass > 1e-30 { 1.0 / mass } else { 0.0 };
        Self {
            position: pos,
            predicted: pos,
            velocity: [0.0; 3],
            inv_mass,
            fixed: false,
        }
    }
    /// Create a fixed (pinned) particle.
    pub fn new_fixed(pos: [f64; 3]) -> Self {
        Self {
            position: pos,
            predicted: pos,
            velocity: [0.0; 3],
            inv_mass: 0.0,
            fixed: true,
        }
    }
}
