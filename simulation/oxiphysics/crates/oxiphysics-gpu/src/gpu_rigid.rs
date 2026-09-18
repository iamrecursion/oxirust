// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated rigid body batch simulation (CPU mock).
//!
//! Provides structs and functions for simulating many rigid bodies in batch,
//! broadphase collision detection (SAP), and constraint solving via sequential
//! impulse. All computation is done on the CPU as a reference implementation.
//!
//! A body whose mass is (almost) zero is *static*: it has neither inverse mass
//! nor inverse inertia, so gravity, impulses and the contact solver leave its
//! velocities alone (it may still be moved kinematically through its own
//! velocity).

// ── Quaternion helpers ────────────────────────────────────────────────────────

/// Rotate vector `v` by unit quaternion `q` (format: \[x, y, z, w\]).
///
/// Uses the sandwich product `q * v * q⁻¹` via the Rodrigues formula.
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    // t = 2 * cross(q.xyz, v)
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    // result = v + qw * t + cross(q.xyz, t)
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

/// Multiply two unit quaternions `a` and `b` (Hamilton product).
///
/// Format for both operands and the result: \[x, y, z, w\].
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Normalise a quaternion to unit length.
///
/// Returns the identity quaternion `[0,0,0,1]` if the norm is nearly zero.
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-9 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

/// Integrate a quaternion orientation `q` by angular velocity `omega` (rad/s)
/// over time step `dt` (seconds).
///
/// Uses the first-order approximation `q' = normalize(q + 0.5 * dt * Ω⊗q)`.
pub fn integrate_orientation(q: [f32; 4], omega: [f32; 3], dt: f32) -> [f32; 4] {
    // omega as a pure quaternion [ox, oy, oz, 0]
    let omega_q = [omega[0], omega[1], omega[2], 0.0_f32];
    let dq = quat_mul(omega_q, q);
    let q_new = [
        q[0] + 0.5 * dt * dq[0],
        q[1] + 0.5 * dt * dq[1],
        q[2] + 0.5 * dt * dq[2],
        q[3] + 0.5 * dt * dq[3],
    ];
    quat_normalize(q_new)
}

// ── Core rigid body ───────────────────────────────────────────────────────────

/// A single rigid body stored in GPU-friendly packed arrays of `f32`.
///
/// Orientation is stored as a unit quaternion `[x, y, z, w]`.
/// The inverse inertia tensor is stored in row-major order as a 3×3 matrix.
#[derive(Debug, Clone)]
pub struct GpuRigidBody {
    /// World-space position (metres).
    pub position: [f32; 3],
    /// Linear velocity (m/s).
    pub velocity: [f32; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub orientation: [f32; 4],
    /// Angular velocity in world space (rad/s).
    pub angular_velocity: [f32; 3],
    /// Mass in kilograms.  A mass of at most `1e-12` (or a non-finite mass)
    /// makes the body static.
    pub mass: f32,
    /// Inverse of the body-space inertia tensor, row-major (3×3).
    ///
    /// It is expressed in the body frame, so it must be rotated into the world
    /// frame (`R · I⁻¹ · Rᵀ`) before it is applied to a world-space torque.
    pub inv_inertia: [f32; 9],
}

impl GpuRigidBody {
    /// Create a new rigid body at rest with the given mass and diagonal inertia.
    ///
    /// `ixx`, `iyy`, `izz` are the principal moments of inertia.
    pub fn new(mass: f32, ixx: f32, iyy: f32, izz: f32) -> Self {
        let safe_inv = |v: f32| if v.abs() > 1e-12 { 1.0 / v } else { 0.0 };
        let inv_inertia = [
            safe_inv(ixx),
            0.0,
            0.0,
            0.0,
            safe_inv(iyy),
            0.0,
            0.0,
            0.0,
            safe_inv(izz),
        ];
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0; 3],
            mass,
            inv_inertia,
        }
    }
}

// ── Batch integration ─────────────────────────────────────────────────────────

/// A batch of GPU rigid bodies supporting bulk integration.
#[derive(Debug, Clone, Default)]
pub struct GpuRigidBodyBatch {
    /// The rigid bodies in this batch.
    pub bodies: Vec<GpuRigidBody>,
}

impl GpuRigidBodyBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a body to the batch and return its index.
    pub fn add(&mut self, body: GpuRigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Integrate all bodies by `dt` seconds under a uniform `gravity` (m/s²).
    ///
    /// Uses explicit Euler integration for linear dynamics and a first-order
    /// quaternion update for rotational dynamics.  Static bodies are not
    /// accelerated by gravity; they still move with whatever velocity they were
    /// given.
    pub fn integrate_all(&mut self, dt: f32, gravity: [f32; 3]) {
        for body in &mut self.bodies {
            // linear: v += g*dt (dynamic bodies only),  p += v*dt
            if inverse_mass(body) > 0.0 {
                body.velocity[0] += gravity[0] * dt;
                body.velocity[1] += gravity[1] * dt;
                body.velocity[2] += gravity[2] * dt;
            }
            body.position[0] += body.velocity[0] * dt;
            body.position[1] += body.velocity[1] * dt;
            body.position[2] += body.velocity[2] * dt;
            // rotational
            let new_q = integrate_orientation(body.orientation, body.angular_velocity, dt);
            body.orientation = new_q;
        }
    }

    /// Apply a linear impulse `impulse` (N·s) at world-space `point` to body
    /// at `body_idx`.
    ///
    /// Updates both linear and angular velocity: `Δv = J / m` and
    /// `Δω = I_world⁻¹ (r × J)` with `r = point − position`, where
    /// `I_world⁻¹ = R · I_body⁻¹ · Rᵀ` follows the body's orientation.  A static
    /// body (see the module documentation) ignores the impulse.
    ///
    /// Returns `false`, changing nothing, when `body_idx` is out of range, and
    /// `true` otherwise.
    pub fn apply_impulse(&mut self, body_idx: usize, impulse: [f32; 3], point: [f32; 3]) -> bool {
        let Some(body) = self.bodies.get_mut(body_idx) else {
            return false;
        };
        let inv_mass = inverse_mass(body);
        // linear impulse
        body.velocity[0] += impulse[0] * inv_mass;
        body.velocity[1] += impulse[1] * inv_mass;
        body.velocity[2] += impulse[2] * inv_mass;
        // torque arm: r = point - position
        let r = [
            point[0] - body.position[0],
            point[1] - body.position[1],
            point[2] - body.position[2],
        ];
        // angular impulse: omega += I_world⁻¹ * (r × impulse)
        let torque_impulse = cross3f(r, impulse);
        let delta_omega = apply_mat3(world_inv_inertia(body), torque_impulse);
        body.angular_velocity[0] += delta_omega[0];
        body.angular_velocity[1] += delta_omega[1];
        body.angular_velocity[2] += delta_omega[2];
        true
    }
}

// ── Broadphase SAP ────────────────────────────────────────────────────────────

/// A candidate collision pair from the broadphase, with AABB information.
#[derive(Debug, Clone)]
pub struct BroadphasePairGpu {
    /// Index of the first body.
    pub body_a: usize,
    /// Index of the second body.
    pub body_b: usize,
    /// AABB centre of body A.
    pub aabb_a_center: [f32; 3],
    /// AABB half-extents of body A.
    pub aabb_a_half: [f32; 3],
    /// AABB centre of body B.
    pub aabb_b_center: [f32; 3],
    /// AABB half-extents of body B.
    pub aabb_b_half: [f32; 3],
}

/// Broadphase collision detection for GPU rigid bodies using a sweep-and-prune
/// (SAP) approach.
///
/// Each body is approximated by a sphere; overlapping sphere AABBs are reported
/// as candidate pairs.
#[derive(Debug, Clone, Default)]
pub struct GpuBroadphase {
    /// Bodies to test.
    pub bodies: Vec<GpuRigidBody>,
    /// Bounding sphere radii, one per body.
    pub radii: Vec<f32>,
}

impl GpuBroadphase {
    /// Create an empty broadphase structure.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a body with its bounding sphere radius.
    pub fn add_body(&mut self, body: GpuRigidBody, radius: f32) {
        self.bodies.push(body);
        self.radii.push(radius);
    }

    /// Run a sort-and-sweep on the X axis and return overlapping pairs.
    ///
    /// Only pairs whose AABBs overlap on all three axes are returned.
    pub fn compute_pairs_sap(&self) -> Vec<BroadphasePairGpu> {
        let n = self.bodies.len();
        let mut pairs = Vec::new();

        // Sort by AABB min-x
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let ax = self.bodies[a].position[0] - self.radii[a];
            let bx = self.bodies[b].position[0] - self.radii[b];
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, &ai) in order.iter().enumerate() {
            let a_max_x = self.bodies[ai].position[0] + self.radii[ai];
            for &bi in order.iter().skip(i + 1) {
                let b_min_x = self.bodies[bi].position[0] - self.radii[bi];
                if b_min_x > a_max_x {
                    break; // sorted: no further pair can overlap on X
                }
                // Check all three axes
                if self.aabb_overlap(ai, bi) {
                    let ra = self.radii[ai];
                    let rb = self.radii[bi];
                    pairs.push(BroadphasePairGpu {
                        body_a: ai,
                        body_b: bi,
                        aabb_a_center: self.bodies[ai].position,
                        aabb_a_half: [ra, ra, ra],
                        aabb_b_center: self.bodies[bi].position,
                        aabb_b_half: [rb, rb, rb],
                    });
                }
            }
        }
        pairs
    }

    /// Test whether two bodies' sphere AABBs overlap on all three axes.
    fn aabb_overlap(&self, a: usize, b: usize) -> bool {
        for k in 0..3 {
            let a_min = self.bodies[a].position[k] - self.radii[a];
            let a_max = self.bodies[a].position[k] + self.radii[a];
            let b_min = self.bodies[b].position[k] - self.radii[b];
            let b_max = self.bodies[b].position[k] + self.radii[b];
            if a_max < b_min || b_max < a_min {
                return false;
            }
        }
        true
    }
}

// ── Contact manifold ──────────────────────────────────────────────────────────

/// Contact manifold between two bodies produced by the narrowphase.
#[derive(Debug, Clone)]
pub struct ContactManifoldGpu {
    /// Index of the first body.
    pub body_a: usize,
    /// Index of the second body.
    pub body_b: usize,
    /// Contact point positions in world space.
    pub contact_points: Vec<[f32; 3]>,
    /// Outward contact normals (pointing from B to A), one per contact point.
    pub normals: Vec<[f32; 3]>,
    /// Penetration depths (positive means overlapping), one per contact point.
    pub penetrations: Vec<f32>,
}

impl ContactManifoldGpu {
    /// Create a new, empty contact manifold between two bodies.
    pub fn new(body_a: usize, body_b: usize) -> Self {
        Self {
            body_a,
            body_b,
            contact_points: Vec::new(),
            normals: Vec::new(),
            penetrations: Vec::new(),
        }
    }

    /// Add a single contact point to the manifold.
    pub fn add_contact(&mut self, point: [f32; 3], normal: [f32; 3], penetration: f32) {
        self.contact_points.push(point);
        self.normals.push(normal);
        self.penetrations.push(penetration);
    }

    /// Number of contact points in the manifold.
    pub fn contact_count(&self) -> usize {
        self.contact_points.len()
    }
}

// ── Sequential impulse solver ─────────────────────────────────────────────────

/// Iterative sequential-impulse constraint solver for rigid body contacts.
///
/// Every contact point carries three accumulated impulses: one along the
/// contact normal (never negative: contacts push but do not pull) and two along
/// the tangent plane (Coulomb friction, limited to `friction ×` the normal
/// impulse accumulated so far).  Linear and angular velocities are both
/// updated, using each body's world-space inverse inertia.
///
/// Normals point from body B to body A (see [`ContactManifoldGpu`]).
#[derive(Debug, Clone)]
pub struct GpuConstraintSolver {
    /// Broadphase candidate pairs.
    pub pairs: Vec<BroadphasePairGpu>,
    /// Contact manifolds, aligned with `pairs`.
    pub manifolds: Vec<ContactManifoldGpu>,
    /// Coefficient of restitution in `[0, 1]`.  The post-solve relative normal
    /// velocity of a closing contact is `restitution` times its pre-solve
    /// approach speed (default `0.3`).
    pub restitution: f32,
    /// Coulomb friction coefficient (≥ 0): the tangential impulse of a contact
    /// never exceeds `friction` times its normal impulse (default `0.5`).
    pub friction: f32,
    /// Baumgarte factor `β` (≥ 0): the fraction of the penetration beyond
    /// [`slop`](Self::slop) that is removed per time step by adding the bias
    /// speed `β/dt · max(penetration − slop, 0)` along the normal (default
    /// `0.2`).  Needs a positive finite `dt`; otherwise no bias is applied.
    pub baumgarte: f32,
    /// Penetration depth (metres) tolerated without correction (default `0.01`).
    pub slop: f32,
}

impl Default for GpuConstraintSolver {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            manifolds: Vec::new(),
            restitution: 0.3,
            friction: 0.5,
            baumgarte: 0.2,
            slop: 0.01,
        }
    }
}

/// Material and stabilisation constants of one solver call, sanitised: every
/// field is finite and non-negative, `restitution` is at most one.
#[derive(Debug, Clone, Copy)]
struct ContactParams {
    restitution: f32,
    friction: f32,
    /// Bias speed per metre of penetration beyond the slop: `β / dt`.
    bias_gain: f32,
    slop: f32,
}

/// Finite positive values pass through; zero, negative and non-finite values
/// become zero.
fn non_negative_or_zero(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

impl ContactParams {
    fn new(solver: &GpuConstraintSolver, dt: f32) -> Self {
        let baumgarte = non_negative_or_zero(solver.baumgarte);
        let bias_gain = if dt.is_finite() && dt > 0.0 {
            non_negative_or_zero(baumgarte / dt)
        } else {
            0.0
        };
        Self {
            restitution: non_negative_or_zero(solver.restitution).min(1.0),
            friction: non_negative_or_zero(solver.friction),
            bias_gain,
            slop: non_negative_or_zero(solver.slop),
        }
    }
}

/// Velocity state and inverse mass properties of one body during a solve.
///
/// The orientation does not change while velocities are solved, so the
/// world-space inverse inertia is computed once per call.
#[derive(Debug, Clone, Copy)]
struct SolverBody {
    velocity: [f32; 3],
    angular_velocity: [f32; 3],
    inv_mass: f32,
    /// World-space inverse inertia tensor (row-major); zero for a static body.
    inv_inertia: [f32; 9],
}

impl SolverBody {
    fn new(body: &GpuRigidBody) -> Self {
        Self {
            velocity: body.velocity,
            angular_velocity: body.angular_velocity,
            inv_mass: inverse_mass(body),
            inv_inertia: world_inv_inertia(body),
        }
    }

    /// Velocity of the material point at offset `r` from the centre of mass.
    fn point_velocity(&self, r: [f32; 3]) -> [f32; 3] {
        add3f(self.velocity, cross3f(self.angular_velocity, r))
    }

    /// Apply the impulse `impulse` at offset `r` from the centre of mass.
    fn apply(&mut self, impulse: [f32; 3], r: [f32; 3]) {
        self.velocity = add3f(self.velocity, scale3f(impulse, self.inv_mass));
        let angular = apply_mat3(self.inv_inertia, cross3f(r, impulse));
        self.angular_velocity = add3f(self.angular_velocity, angular);
    }
}

/// Velocity of contact point A relative to contact point B.
fn relative_contact_velocity(
    a: &SolverBody,
    b: &SolverBody,
    r_a: [f32; 3],
    r_b: [f32; 3],
) -> [f32; 3] {
    sub3f(a.point_velocity(r_a), b.point_velocity(r_b))
}

/// Change of the relative contact velocity per impulse `p` applied to body A
/// (and `−p` to body B) at the contact:
/// `Δv_rel = (1/m_a + 1/m_b) p + (I_a⁻¹ (r_a × p)) × r_a + (I_b⁻¹ (r_b × p)) × r_b`.
fn impulse_response(
    a: &SolverBody,
    b: &SolverBody,
    r_a: [f32; 3],
    r_b: [f32; 3],
    p: [f32; 3],
) -> [f32; 3] {
    let linear = scale3f(p, a.inv_mass + b.inv_mass);
    let angular_a = cross3f(apply_mat3(a.inv_inertia, cross3f(r_a, p)), r_a);
    let angular_b = cross3f(apply_mat3(b.inv_inertia, cross3f(r_b, p)), r_b);
    add3f(linear, add3f(angular_a, angular_b))
}

/// One contact point prepared for the solver iterations.
#[derive(Debug, Clone, Copy)]
struct ContactRow {
    a: usize,
    b: usize,
    /// Contact offsets from the centres of mass of A and B.
    r_a: [f32; 3],
    r_b: [f32; 3],
    /// Unit contact normal, pointing from B to A.
    normal: [f32; 3],
    /// Orthonormal tangent basis of the contact plane.
    tangents: [[f32; 3]; 2],
    /// Effective inverse mass along the normal: `n · Δv_rel(n)`.
    normal_k: f32,
    /// Inverse of the 2×2 tangential effective-mass matrix, or `None` when the
    /// tangent plane cannot be driven (friction is skipped).
    friction_k_inv: Option<[[f32; 2]; 2]>,
    /// Desired post-solve relative normal velocity: the larger of the
    /// restitution bounce and the Baumgarte bias.
    target_speed: f32,
    /// Accumulated normal impulse (≥ 0).
    normal_impulse: f32,
    /// Accumulated impulse along the two tangents.
    tangent_impulse: [f32; 2],
}

/// Contact data read from a manifold (`normal` not yet normalised).
#[derive(Debug, Clone, Copy)]
struct ContactInput {
    a: usize,
    b: usize,
    point: [f32; 3],
    normal: [f32; 3],
    penetration: f32,
}

impl ContactRow {
    /// Prepare one contact, or return `None` when it cannot be solved: a
    /// degenerate or non-finite normal or offset, or two bodies that cannot
    /// respond (both static: `k < 1e-12`).
    fn prepare(
        input: &ContactInput,
        bodies: &[GpuRigidBody],
        state: &[SolverBody],
        params: &ContactParams,
    ) -> Option<Self> {
        let (body_a, body_b) = (bodies.get(input.a)?, bodies.get(input.b)?);
        let (sa, sb) = (state.get(input.a)?, state.get(input.b)?);
        let normal = normalize3f(input.normal);
        let r_a = sub3f(input.point, body_a.position);
        let r_b = sub3f(input.point, body_b.position);
        let finite = normal
            .iter()
            .chain(r_a.iter())
            .chain(r_b.iter())
            .all(|v| v.is_finite());
        if !finite || length3f(normal) < 0.5 {
            return None;
        }

        let normal_k = dot3f(normal, impulse_response(sa, sb, r_a, r_b, normal));
        if !normal_k.is_finite() || normal_k <= 1e-12 {
            return None;
        }

        let tangent_1 = tangent_to(normal);
        let tangents = [tangent_1, cross3f(normal, tangent_1)];
        let mut friction_k = [[0.0_f32; 2]; 2];
        for (row, ti) in friction_k.iter_mut().zip(tangents.iter()) {
            for (entry, tj) in row.iter_mut().zip(tangents.iter()) {
                *entry = dot3f(*ti, impulse_response(sa, sb, r_a, r_b, *tj));
            }
        }
        let friction_k_inv = invert_2x2(friction_k);

        // Restitution is taken from the approach speed before any impulse.
        let approach = dot3f(relative_contact_velocity(sa, sb, r_a, r_b), normal);
        let bounce = if approach < 0.0 {
            -params.restitution * approach
        } else {
            0.0
        };
        // `f32::max` ignores a NaN operand, so a NaN penetration gives no bias.
        let bias = params.bias_gain * (input.penetration - params.slop).max(0.0);

        Some(Self {
            a: input.a,
            b: input.b,
            r_a,
            r_b,
            normal,
            tangents,
            normal_k,
            friction_k_inv,
            target_speed: bounce.max(bias),
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        })
    }

    /// One Gauss-Seidel update of this contact: the non-penetration impulse,
    /// then Coulomb friction limited by the normal impulse accumulated so far.
    fn solve(&mut self, state: &mut [SolverBody], friction: f32) {
        let (Some(&start_a), Some(&start_b)) = (state.get(self.a), state.get(self.b)) else {
            return;
        };
        let (mut sa, mut sb) = (start_a, start_b);

        // Normal: drive the relative normal velocity to `target_speed` with a
        // push (never a pull).
        let approach = dot3f(
            relative_contact_velocity(&sa, &sb, self.r_a, self.r_b),
            self.normal,
        );
        let wanted = self.normal_impulse + (self.target_speed - approach) / self.normal_k;
        let new_normal_impulse = wanted.max(0.0);
        let delta = new_normal_impulse - self.normal_impulse;
        self.normal_impulse = new_normal_impulse;
        let push = scale3f(self.normal, delta);
        sa.apply(push, self.r_a);
        sb.apply(scale3f(push, -1.0), self.r_b);

        // Friction: cancel the tangential sliding velocity, staying inside the
        // friction cone |λ_t| ≤ μ λ_n.
        if let Some(k_inv) = self.friction_k_inv.filter(|_| friction > 0.0) {
            let sliding = relative_contact_velocity(&sa, &sb, self.r_a, self.r_b);
            let rhs = [
                -dot3f(sliding, self.tangents[0]),
                -dot3f(sliding, self.tangents[1]),
            ];
            let step = [
                k_inv[0][0] * rhs[0] + k_inv[0][1] * rhs[1],
                k_inv[1][0] * rhs[0] + k_inv[1][1] * rhs[1],
            ];
            let mut wanted = [
                self.tangent_impulse[0] + step[0],
                self.tangent_impulse[1] + step[1],
            ];
            let limit = friction * self.normal_impulse;
            let magnitude = wanted[0].hypot(wanted[1]);
            if magnitude > limit {
                let shrink = if magnitude > 0.0 {
                    limit / magnitude
                } else {
                    0.0
                };
                wanted = [wanted[0] * shrink, wanted[1] * shrink];
            }
            let change = [
                wanted[0] - self.tangent_impulse[0],
                wanted[1] - self.tangent_impulse[1],
            ];
            self.tangent_impulse = wanted;
            let drag = add3f(
                scale3f(self.tangents[0], change[0]),
                scale3f(self.tangents[1], change[1]),
            );
            sa.apply(drag, self.r_a);
            sb.apply(scale3f(drag, -1.0), self.r_b);
        }

        if let Some(slot) = state.get_mut(self.a) {
            *slot = sa;
        }
        if let Some(slot) = state.get_mut(self.b) {
            *slot = sb;
        }
    }
}

impl GpuConstraintSolver {
    /// Create a new solver with no constraints and the default material
    /// constants (see the field documentation).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a manifold (and its corresponding broadphase pair) to the solver.
    pub fn add_manifold(&mut self, pair: BroadphasePairGpu, manifold: ContactManifoldGpu) {
        self.pairs.push(pair);
        self.manifolds.push(manifold);
    }

    /// Solve all contact constraints via sequential impulses.
    ///
    /// Runs `iterations` Gauss-Seidel sweeps over every contact point of every
    /// manifold.  Each contact is solved for
    ///
    /// * a non-negative normal impulse that brings the relative normal velocity
    ///   to the larger of the restitution bounce (`restitution ×` the approach
    ///   speed measured before the solve) and the Baumgarte bias
    ///   `baumgarte/dt · max(penetration − slop, 0)`, and
    /// * a friction impulse in the tangent plane, clamped to the cone
    ///   `|λ_t| ≤ friction × λ_n`.
    ///
    /// Impulses change both linear and angular velocities, using the effective
    /// mass `1/m_a + 1/m_b + n · ((I_a⁻¹ (r_a × n)) × r_a + (I_b⁻¹ (r_b × n)) × r_b)`
    /// built from the world-space inverse inertias.  Static bodies (mass of at
    /// most `1e-12`) are never changed.  Manifolds that name a body index out of
    /// range (or the same body twice) are skipped, as are contacts whose normal
    /// is degenerate or whose two bodies are both static.
    pub fn solve_sequential_impulse(
        &self,
        bodies: &mut [GpuRigidBody],
        dt: f32,
        iterations: usize,
    ) {
        let params = ContactParams::new(self, dt);
        let mut state: Vec<SolverBody> = bodies.iter().map(SolverBody::new).collect();

        let mut rows: Vec<ContactRow> = Vec::new();
        for manifold in &self.manifolds {
            let (a, b) = (manifold.body_a, manifold.body_b);
            if a == b || a >= bodies.len() || b >= bodies.len() {
                continue;
            }
            let contacts = manifold
                .contact_points
                .iter()
                .zip(manifold.normals.iter())
                .zip(manifold.penetrations.iter());
            for ((&point, &normal), &penetration) in contacts {
                let input = ContactInput {
                    a,
                    b,
                    point,
                    normal,
                    penetration,
                };
                rows.extend(ContactRow::prepare(&input, bodies, &state, &params));
            }
        }

        for _ in 0..iterations {
            for row in rows.iter_mut() {
                row.solve(&mut state, params.friction);
            }
        }

        for (body, solved) in bodies.iter_mut().zip(state.iter()) {
            if solved.inv_mass > 0.0 {
                body.velocity = solved.velocity;
                body.angular_velocity = solved.angular_velocity;
            }
        }
    }
}

// ── Body helpers ──────────────────────────────────────────────────────────────

/// Inverse mass `1/m`, or zero for a static body (mass at most `1e-12`, or not
/// finite).
fn inverse_mass(body: &GpuRigidBody) -> f32 {
    if body.mass.is_finite() && body.mass > 1e-12 {
        1.0 / body.mass
    } else {
        0.0
    }
}

/// World-space inverse inertia tensor `R · I_body⁻¹ · Rᵀ` (row-major), with `R`
/// the rotation of the body's orientation quaternion.  Zero for a static body.
fn world_inv_inertia(body: &GpuRigidBody) -> [f32; 9] {
    if inverse_mass(body) == 0.0 {
        return [0.0; 9];
    }
    let rotation = rotation_matrix(body.orientation);
    mat3_mul(
        mat3_mul(rotation, body.inv_inertia),
        mat3_transpose(rotation),
    )
}

/// Row-major rotation matrix of the quaternion `q` (\[x, y, z, w\]); `q` is
/// normalised first, and a zero quaternion gives the identity.
fn rotation_matrix(q: [f32; 4]) -> [f32; 9] {
    let [x, y, z, w] = quat_normalize(q);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

// ── Internal math helpers ─────────────────────────────────────────────────────

/// Cross product of two `[f32; 3]` vectors.
fn cross3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two `[f32; 3]` vectors.
fn dot3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3f(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn length3f(a: [f32; 3]) -> f32 {
    dot3f(a, a).sqrt()
}

/// `a / |a|`, or the zero vector when `a` is (nearly) zero or not finite.
fn normalize3f(a: [f32; 3]) -> [f32; 3] {
    let len = length3f(a);
    if len.is_finite() && len > 1e-9 {
        scale3f(a, 1.0 / len)
    } else {
        [0.0; 3]
    }
}

/// A unit vector perpendicular to the unit vector `n`.
fn tangent_to(n: [f32; 3]) -> [f32; 3] {
    let helper = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize3f(cross3f(n, helper))
}

/// Inverse of a symmetric positive semi-definite 2×2 matrix, or `None` when it
/// is singular (or not finite) to working precision.
fn invert_2x2(m: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let scale = m[0][0].abs() * m[1][1].abs();
    if !(det.is_finite() && det > 1e-9 * scale && det > 1e-24) {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ])
}

/// Apply a row-major 3×3 matrix to a vector.
fn apply_mat3(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Product `a · b` of two row-major 3×3 matrices.
fn mat3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for (index, entry) in out.iter_mut().enumerate() {
        let (row, col) = (index / 3, index % 3);
        *entry = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
    }
    out
}

/// Transpose of a row-major 3×3 matrix.
fn mat3_transpose(m: [f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
