// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU-style particle system simulated on the CPU.
//!
//! Provides a complete particle simulation pipeline including emission,
//! integration, forces (gravity, turbulence), collision response, pooling,
//! and render-data collection for billboard quads.

// ---------------------------------------------------------------------------
// Internal LCG random number generator (no external crate needed)
// ---------------------------------------------------------------------------

/// Simple linear congruential random number generator for particle systems.
pub struct Lcg(u64);

impl Lcg {
    /// Create a new LCG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// Return the next raw 64-bit pseudo-random value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Return a uniform f32 in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Uniform float in `[lo, hi)`.
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// GpuParticle
// ---------------------------------------------------------------------------

/// A single GPU-style particle with position, velocity, life, color and size.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuParticle {
    /// World-space position `[x, y, z]`.
    pub position: [f32; 3],
    /// World-space velocity `[vx, vy, vz]`.
    pub velocity: [f32; 3],
    /// Remaining lifetime in seconds. `<= 0` means dead.
    pub life: f32,
    /// RGBA color `[r, g, b, a]` each in `[0, 1]`.
    pub color: [f32; 4],
    /// Billboard size in world units.
    pub size: f32,
}

impl GpuParticle {
    /// Create a new particle with the given attributes.
    pub fn new(
        position: [f32; 3],
        velocity: [f32; 3],
        life: f32,
        color: [f32; 4],
        size: f32,
    ) -> Self {
        Self {
            position,
            velocity,
            life,
            color,
            size,
        }
    }

    /// Returns `true` if this particle is still alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.life > 0.0
    }
}

impl Default for GpuParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            life: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleEmitter
// ---------------------------------------------------------------------------

/// Emits particles from a point with configurable velocity distribution
/// and lifetime.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    /// World-space emission origin.
    pub position: [f32; 3],
    /// Number of particles emitted per second.
    pub emission_rate: f32,
    /// Base initial velocity `[vx, vy, vz]`.
    pub initial_velocity: [f32; 3],
    /// Random spread (half-range) added to each velocity component.
    pub velocity_spread: f32,
    /// Initial particle lifetime in seconds.
    pub lifetime: f32,
    /// Initial particle color.
    pub color: [f32; 4],
    /// Initial particle size.
    pub size: f32,
    /// Accumulated fractional particles not yet emitted.
    pub accumulator: f32,
}

impl ParticleEmitter {
    /// Create a new emitter.
    pub fn new(
        position: [f32; 3],
        emission_rate: f32,
        initial_velocity: [f32; 3],
        velocity_spread: f32,
        lifetime: f32,
    ) -> Self {
        Self {
            position,
            emission_rate,
            initial_velocity,
            velocity_spread,
            lifetime,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
            accumulator: 0.0,
        }
    }

    /// Advance the accumulator and return how many particles to emit this step.
    pub fn particles_to_emit(&mut self, dt: f32) -> usize {
        self.accumulator += self.emission_rate * dt;
        let n = self.accumulator as usize;
        self.accumulator -= n as f32;
        n
    }

    /// Build a fresh particle, adding random velocity spread via the given RNG.
    pub fn spawn(&self, rng: &mut Lcg) -> GpuParticle {
        let spread = self.velocity_spread;
        let vx = self.initial_velocity[0] + rng.range_f32(-spread, spread);
        let vy = self.initial_velocity[1] + rng.range_f32(-spread, spread);
        let vz = self.initial_velocity[2] + rng.range_f32(-spread, spread);
        GpuParticle {
            position: self.position,
            velocity: [vx, vy, vz],
            life: self.lifetime,
            color: self.color,
            size: self.size,
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleIntegrator
// ---------------------------------------------------------------------------

/// Explicit Euler integrator for particles: `pos += vel * dt`, `life -= dt`.
#[derive(Debug, Clone, Copy)]
pub struct ParticleIntegrator;

impl ParticleIntegrator {
    /// Integrate a single particle in place.
    #[inline]
    pub fn step(particle: &mut GpuParticle, dt: f32) {
        particle.position[0] += particle.velocity[0] * dt;
        particle.position[1] += particle.velocity[1] * dt;
        particle.position[2] += particle.velocity[2] * dt;
        particle.life -= dt;
    }

    /// Integrate all alive particles in a slice.
    pub fn step_all(particles: &mut [GpuParticle], dt: f32) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                Self::step(p, dt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GravityForce
// ---------------------------------------------------------------------------

/// Constant downward gravitational acceleration applied to particles.
#[derive(Debug, Clone, Copy)]
pub struct GravityForce {
    /// Gravitational acceleration vector `[gx, gy, gz]` (m/s²).
    pub acceleration: [f32; 3],
}

impl GravityForce {
    /// Create a standard Earth gravity force (downward along -Y).
    pub fn earth() -> Self {
        Self {
            acceleration: [0.0, -9.81, 0.0],
        }
    }

    /// Create a gravity force with custom acceleration vector.
    pub fn new(acceleration: [f32; 3]) -> Self {
        Self { acceleration }
    }

    /// Apply gravity to a single particle for one timestep.
    #[inline]
    pub fn apply(&self, particle: &mut GpuParticle, dt: f32) {
        particle.velocity[0] += self.acceleration[0] * dt;
        particle.velocity[1] += self.acceleration[1] * dt;
        particle.velocity[2] += self.acceleration[2] * dt;
    }

    /// Apply gravity to all alive particles.
    pub fn apply_all(&self, particles: &mut [GpuParticle], dt: f32) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                self.apply(p, dt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Smooth 3-D gradient (Perlin) noise
// ---------------------------------------------------------------------------

/// `1/sqrt(2)`, the non-zero component of a unit edge direction of a cube.
const INV_SQRT_2: f32 = 0.707_106_77;

/// The 16 lattice gradient directions of improved Perlin noise, normalised to
/// unit length: the 12 cube-edge directions, four of them repeated to fill a
/// power-of-two table.
const NOISE_GRADIENTS: [[f32; 3]; 16] = [
    [INV_SQRT_2, INV_SQRT_2, 0.0],
    [-INV_SQRT_2, INV_SQRT_2, 0.0],
    [INV_SQRT_2, -INV_SQRT_2, 0.0],
    [-INV_SQRT_2, -INV_SQRT_2, 0.0],
    [INV_SQRT_2, 0.0, INV_SQRT_2],
    [-INV_SQRT_2, 0.0, INV_SQRT_2],
    [INV_SQRT_2, 0.0, -INV_SQRT_2],
    [-INV_SQRT_2, 0.0, -INV_SQRT_2],
    [0.0, INV_SQRT_2, INV_SQRT_2],
    [0.0, -INV_SQRT_2, INV_SQRT_2],
    [0.0, INV_SQRT_2, -INV_SQRT_2],
    [0.0, -INV_SQRT_2, -INV_SQRT_2],
    [INV_SQRT_2, INV_SQRT_2, 0.0],
    [-INV_SQRT_2, INV_SQRT_2, 0.0],
    [0.0, -INV_SQRT_2, INV_SQRT_2],
    [0.0, -INV_SQRT_2, -INV_SQRT_2],
];

/// Hash of an integer lattice point and a noise `channel` (an independent
/// noise field per channel): multiplicative mixing of the coordinates followed
/// by the MurmurHash3 finaliser.
fn lattice_hash(ix: i32, iy: i32, iz: i32, channel: u32) -> u32 {
    let mut h = (ix as u32).wrapping_mul(0x8da6_b343)
        ^ (iy as u32).wrapping_mul(0xd816_3841)
        ^ (iz as u32).wrapping_mul(0xcb1a_b31f)
        ^ channel.wrapping_mul(0x9e37_79b1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Quintic fade curve `6t⁵ − 15t⁴ + 10t³`: its first and second derivatives
/// vanish at `t = 0` and `t = 1`, so the noise is C² across lattice cells.
fn noise_fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Derivative of [`noise_fade`]: `30 t² (t − 1)²`.
fn noise_fade_slope(t: f32) -> f32 {
    30.0 * t * t * (t - 1.0) * (t - 1.0)
}

/// Value and analytic gradient of 3-D gradient noise at `p` for one `channel`.
///
/// Each of the eight lattice corners around `p` carries a unit gradient vector
/// chosen by hashing its integer coordinates; the corner contributes the dot
/// product of that gradient with the offset from the corner, and the eight
/// contributions are blended trilinearly with the quintic fade of the position
/// inside the cell.  The value is zero at every lattice point and stays within
/// about ±0.65, the gradient has magnitude up to about 2.2 (RMS 0.86), and both
/// are continuous everywhere.
fn gradient_noise(p: [f32; 3], channel: u32) -> (f32, [f32; 3]) {
    let base = [p[0].floor(), p[1].floor(), p[2].floor()];
    let cell = [base[0] as i32, base[1] as i32, base[2] as i32];
    let local = [p[0] - base[0], p[1] - base[1], p[2] - base[2]];
    let fade = local.map(noise_fade);
    let slope = local.map(noise_fade_slope);
    // Blend weight of one axis (and its derivative) for the lower (`bit = 0`)
    // or upper (`bit = 1`) lattice plane.
    let axis_weight = |bit: i32, axis: usize| {
        if bit == 1 {
            (fade[axis], slope[axis])
        } else {
            (1.0 - fade[axis], -slope[axis])
        }
    };

    let mut value = 0.0_f32;
    let mut gradient = [0.0_f32; 3];
    for corner in 0..8_i32 {
        let bits = [corner & 1, (corner >> 1) & 1, corner >> 2];
        let hash = lattice_hash(
            cell[0].wrapping_add(bits[0]),
            cell[1].wrapping_add(bits[1]),
            cell[2].wrapping_add(bits[2]),
            channel,
        );
        let direction = NOISE_GRADIENTS[(hash & 15) as usize];
        let offset = [
            local[0] - bits[0] as f32,
            local[1] - bits[1] as f32,
            local[2] - bits[2] as f32,
        ];
        let contribution =
            direction[0] * offset[0] + direction[1] * offset[1] + direction[2] * offset[2];
        let (wx, dwx) = axis_weight(bits[0], 0);
        let (wy, dwy) = axis_weight(bits[1], 1);
        let (wz, dwz) = axis_weight(bits[2], 2);
        let weight = wx * wy * wz;
        value += weight * contribution;
        // Product rule: the blend weights vary with `p`, and so does the dot
        // product (its derivative along axis `a` is `direction[a]`).
        gradient[0] += dwx * wy * wz * contribution + weight * direction[0];
        gradient[1] += wx * dwy * wz * contribution + weight * direction[1];
        gradient[2] += wx * wy * dwz * contribution + weight * direction[2];
    }
    (value, gradient)
}

// ---------------------------------------------------------------------------
// TurbulenceForce — curl-noise perturbation
// ---------------------------------------------------------------------------

/// Curl-noise turbulence force.
///
/// The perturbation is the curl of a vector potential whose three components
/// are independent fields of smooth 3-D gradient (Perlin) noise.  The curl of
/// any smooth field is divergence-free, so particles swirl without bunching up
/// or thinning out; the curl is formed from the *analytic* gradients of the
/// noise, so divergence-freeness holds to rounding error and the field is
/// continuous (C¹) in space and time.
///
/// The noise is sampled at `frequency × position`: features are about
/// `1 / frequency` world units across.  The curl is taken in noise space, so
/// its magnitude does not depend on `frequency`: it is of order `strength`
/// (mean about `1.1 × strength`, peaks about `3.5 × strength`).
#[derive(Debug, Clone, Copy)]
pub struct TurbulenceForce {
    /// Scale of the swirl acceleration (velocity perturbation per second).
    pub strength: f32,
    /// Spatial frequency of the noise field.
    pub frequency: f32,
    /// Time offset (varies the noise over time).
    pub time_offset: f32,
}

impl TurbulenceForce {
    /// Create a new turbulence force.
    pub fn new(strength: f32, frequency: f32) -> Self {
        Self {
            strength,
            frequency,
            time_offset: 0.0,
        }
    }

    /// Advance the internal time offset.
    pub fn advance(&mut self, dt: f32) {
        self.time_offset += dt;
    }

    /// Curl-noise perturbation (an acceleration) at a world position.
    ///
    /// With the potential `ψ = (ψx, ψy, ψz)` sampled at `q = frequency · pos`
    /// this is `strength · (∂ψz/∂y − ∂ψy/∂z, ∂ψx/∂z − ∂ψz/∂x, ∂ψy/∂x − ∂ψx/∂y)`
    /// with the derivatives taken with respect to `q`.  Each potential component
    /// slides through noise space along its own axis as `time_offset` grows
    /// (a drift of `0.1` noise units per unit time), so the field evolves
    /// smoothly rather than just translating.
    pub fn curl_at(&self, pos: [f32; 3]) -> [f32; 3] {
        let f = self.frequency;
        let drift = self.time_offset * 0.1;
        let q = [pos[0] * f, pos[1] * f, pos[2] * f];
        let (_, grad_x) = gradient_noise([q[0] + drift, q[1], q[2]], 0);
        let (_, grad_y) = gradient_noise([q[0], q[1] + drift, q[2]], 1);
        let (_, grad_z) = gradient_noise([q[0], q[1], q[2] + drift], 2);
        [
            (grad_z[1] - grad_y[2]) * self.strength,
            (grad_x[2] - grad_z[0]) * self.strength,
            (grad_y[0] - grad_x[1]) * self.strength,
        ]
    }

    /// Apply turbulence to a single particle.
    pub fn apply(&self, particle: &mut GpuParticle, dt: f32) {
        let curl = self.curl_at(particle.position);
        particle.velocity[0] += curl[0] * dt;
        particle.velocity[1] += curl[1] * dt;
        particle.velocity[2] += curl[2] * dt;
    }

    /// Apply turbulence to all alive particles.
    pub fn apply_all(&self, particles: &mut [GpuParticle], dt: f32) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                self.apply(p, dt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleCollider — axis-aligned plane
// ---------------------------------------------------------------------------

/// Particle-plane collision with restitution coefficient.
///
/// The plane is defined by a point and a normal (pointing into the "safe" side).
#[derive(Debug, Clone, Copy)]
pub struct ParticleCollider {
    /// A point on the plane.
    pub plane_point: [f32; 3],
    /// Outward unit normal of the plane.
    pub plane_normal: [f32; 3],
    /// Coefficient of restitution `[0, 1]`. 0 = inelastic, 1 = elastic.
    pub restitution: f32,
    /// Friction coefficient applied to tangential velocity on collision.
    pub friction: f32,
}

impl ParticleCollider {
    /// Create a horizontal floor at height `y` with given restitution.
    pub fn floor(y: f32, restitution: f32) -> Self {
        Self {
            plane_point: [0.0, y, 0.0],
            plane_normal: [0.0, 1.0, 0.0],
            restitution,
            friction: 0.0,
        }
    }

    /// Create a collider for a general plane.
    pub fn new(plane_point: [f32; 3], plane_normal: [f32; 3], restitution: f32) -> Self {
        // Normalise the normal
        let n = plane_normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
        Self {
            plane_point,
            plane_normal: [n[0] / len, n[1] / len, n[2] / len],
            restitution,
            friction: 0.0,
        }
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Resolve collision for one particle (modifies position and velocity).
    pub fn resolve(&self, particle: &mut GpuParticle) {
        let n = self.plane_normal;
        let p = self.plane_point;
        // Signed distance from plane
        let diff = [
            particle.position[0] - p[0],
            particle.position[1] - p[1],
            particle.position[2] - p[2],
        ];
        let dist = Self::dot(diff, n);
        if dist < 0.0 {
            // Push particle back onto the plane
            particle.position[0] -= dist * n[0];
            particle.position[1] -= dist * n[1];
            particle.position[2] -= dist * n[2];

            // Reflect normal component of velocity with restitution
            let vn = Self::dot(particle.velocity, n);
            if vn < 0.0 {
                // Normal impulse
                let impulse = -(1.0 + self.restitution) * vn;
                // Tangential velocity
                let vt = [
                    particle.velocity[0] - vn * n[0],
                    particle.velocity[1] - vn * n[1],
                    particle.velocity[2] - vn * n[2],
                ];
                particle.velocity[0] = vt[0] * (1.0 - self.friction) + impulse * n[0] + vn * n[0];
                particle.velocity[1] = vt[1] * (1.0 - self.friction) + impulse * n[1] + vn * n[1];
                particle.velocity[2] = vt[2] * (1.0 - self.friction) + impulse * n[2] + vn * n[2];
            }
        }
    }

    /// Resolve collisions for all alive particles.
    pub fn resolve_all(&self, particles: &mut [GpuParticle]) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                self.resolve(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParticlePool — fixed-size free-list pool
// ---------------------------------------------------------------------------

/// Fixed-capacity particle pool with a free list.
///
/// [`emit`](Self::emit) is O(1).  [`recycle_dead`](Self::recycle_dead) is one
/// O(capacity) sweep over the slots: a per-slot `in_free` flag makes the
/// "is this slot already on the free list?" test O(1), so the cost does not
/// grow with the number of free slots.
pub struct ParticlePool {
    /// All particle slots.
    pub slots: Vec<GpuParticle>,
    /// Indices of currently free (dead) slots.
    free_list: Vec<usize>,
    /// `in_free[i]` is `true` exactly when slot `i` is on `free_list`.
    in_free: Vec<bool>,
    /// Capacity of the pool.
    capacity: usize,
}

impl ParticlePool {
    /// Create a pool with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let slots = vec![GpuParticle::default(); capacity];
        let free_list: Vec<usize> = (0..capacity).collect();
        Self {
            slots,
            free_list,
            in_free: vec![true; capacity],
            capacity,
        }
    }

    /// Total capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently alive particles.
    pub fn alive_count(&self) -> usize {
        self.capacity - self.free_list.len()
    }

    /// Number of free slots.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Emit a particle from the pool.  Returns the slot index, or `None` if
    /// the pool is full.
    pub fn emit(&mut self, particle: GpuParticle) -> Option<usize> {
        let idx = self.free_list.pop()?;
        self.slots[idx] = particle;
        if let Some(flag) = self.in_free.get_mut(idx) {
            *flag = false;
        }
        Some(idx)
    }

    /// Recycle all dead particles back to the free list.
    ///
    /// A single O(capacity) pass; slots that are already free are skipped via
    /// the `in_free` flags, so repeated calls never create duplicates.  A
    /// particle emitted with `life <= 0` occupies its slot until the next call
    /// reclaims it.
    pub fn recycle_dead(&mut self) {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_alive() {
                continue;
            }
            if let Some(flag) = self.in_free.get_mut(i)
                && !*flag
            {
                *flag = true;
                self.free_list.push(i);
            }
        }
    }

    /// Iterate over all alive particles (immutable).
    pub fn alive_iter(&self) -> impl Iterator<Item = &GpuParticle> {
        self.slots.iter().filter(|p| p.is_alive())
    }

    /// Iterate over all alive particles (mutable).
    pub fn alive_iter_mut(&mut self) -> impl Iterator<Item = &mut GpuParticle> {
        self.slots.iter_mut().filter(|p| p.is_alive())
    }
}

// ---------------------------------------------------------------------------
// ColorOverLife
// ---------------------------------------------------------------------------

/// Lerps a particle's color from `birth_color` to `death_color` over its life.
#[derive(Debug, Clone, Copy)]
pub struct ColorOverLife {
    /// Color when the particle is born (life == `max_life`).
    pub birth_color: [f32; 4],
    /// Color when the particle is about to die (life → 0).
    pub death_color: [f32; 4],
    /// The initial (maximum) lifetime used to compute normalised age.
    pub max_life: f32,
}

impl ColorOverLife {
    /// Create a new color-over-life modifier.
    pub fn new(birth_color: [f32; 4], death_color: [f32; 4], max_life: f32) -> Self {
        Self {
            birth_color,
            death_color,
            max_life: max_life.max(1e-9),
        }
    }

    /// Update the color of a single particle based on remaining life.
    pub fn apply(&self, particle: &mut GpuParticle) {
        // t=1 at birth, t=0 at death
        let t = (particle.life / self.max_life).clamp(0.0, 1.0);
        for i in 0..4 {
            particle.color[i] =
                self.death_color[i] + t * (self.birth_color[i] - self.death_color[i]);
        }
    }

    /// Update colors for all alive particles.
    pub fn apply_all(&self, particles: &mut [GpuParticle]) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                self.apply(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SizeOverLife
// ---------------------------------------------------------------------------

/// Modulates a particle's size along its lifetime via a simple quadratic curve.
#[derive(Debug, Clone, Copy)]
pub struct SizeOverLife {
    /// Size at birth.
    pub birth_size: f32,
    /// Size at death.
    pub death_size: f32,
    /// Maximum lifetime (normalisation factor).
    pub max_life: f32,
}

impl SizeOverLife {
    /// Create a new size-over-life modifier.
    pub fn new(birth_size: f32, death_size: f32, max_life: f32) -> Self {
        Self {
            birth_size,
            death_size,
            max_life: max_life.max(1e-9),
        }
    }

    /// Update the size of one particle.
    pub fn apply(&self, particle: &mut GpuParticle) {
        let t = (particle.life / self.max_life).clamp(0.0, 1.0);
        particle.size = self.death_size + t * (self.birth_size - self.death_size);
    }

    /// Update sizes for all alive particles.
    pub fn apply_all(&self, particles: &mut [GpuParticle]) {
        for p in particles.iter_mut() {
            if p.is_alive() {
                self.apply(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleRenderer — depth-sorted billboard quads
// ---------------------------------------------------------------------------

/// Render-ready vertex data for a single billboard quad corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BillboardVertex {
    /// World-space corner position.
    pub position: [f32; 3],
    /// UV coordinates for the quad corner.
    pub uv: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
}

/// Output of the particle renderer: sorted vertices and indices.
#[derive(Debug, Clone)]
pub struct RenderBatch {
    /// Interleaved vertex data (4 vertices per particle, in depth order).
    pub vertices: Vec<BillboardVertex>,
    /// Index buffer (6 indices per particle — two triangles per quad).
    pub indices: Vec<u32>,
}

impl RenderBatch {
    /// Number of particles represented in this batch.
    pub fn particle_count(&self) -> usize {
        self.vertices.len() / 4
    }
}

/// Collects alive particles, depth-sorts them back-to-front, and generates
/// billboard quad geometry aligned to the view plane.
#[derive(Debug, Clone)]
pub struct ParticleRenderer {
    /// Camera position used for depth sorting and billboard orientation.
    pub camera_pos: [f32; 3],
    /// Camera right vector (normalised).
    pub camera_right: [f32; 3],
    /// Camera up vector (normalised).
    pub camera_up: [f32; 3],
}

impl ParticleRenderer {
    /// Create a renderer with a default front-facing camera.
    pub fn new() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 10.0],
            camera_right: [1.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
        }
    }

    /// Update camera orientation.
    pub fn set_camera(&mut self, pos: [f32; 3], right: [f32; 3], up: [f32; 3]) {
        self.camera_pos = pos;
        self.camera_right = right;
        self.camera_up = up;
    }

    fn depth_sq(&self, pos: [f32; 3]) -> f32 {
        let dx = pos[0] - self.camera_pos[0];
        let dy = pos[1] - self.camera_pos[1];
        let dz = pos[2] - self.camera_pos[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Build a `RenderBatch` from alive particles, sorted back-to-front.
    pub fn render(&self, particles: &[GpuParticle]) -> RenderBatch {
        // Collect alive indices with depth
        let mut alive: Vec<(usize, f32)> = particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_alive())
            .map(|(i, p)| (i, self.depth_sq(p.position)))
            .collect();

        // Sort back-to-front (farthest first)
        alive.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut vertices = Vec::with_capacity(alive.len() * 4);
        let mut indices = Vec::with_capacity(alive.len() * 6);

        for (quad_idx, (particle_idx, _)) in alive.iter().enumerate() {
            let p = &particles[*particle_idx];
            let half = p.size * 0.5;

            let r = self.camera_right;
            let u = self.camera_up;

            // Four corners: bottom-left, bottom-right, top-right, top-left
            let corners = [
                ([-1.0_f32, -1.0_f32], [0.0_f32, 0.0_f32]),
                ([1.0, -1.0], [1.0, 0.0]),
                ([1.0, 1.0], [1.0, 1.0]),
                ([-1.0, 1.0], [0.0, 1.0]),
            ];

            for (dir, uv) in &corners {
                let corner_pos = [
                    p.position[0] + (r[0] * dir[0] + u[0] * dir[1]) * half,
                    p.position[1] + (r[1] * dir[0] + u[1] * dir[1]) * half,
                    p.position[2] + (r[2] * dir[0] + u[2] * dir[1]) * half,
                ];
                vertices.push(BillboardVertex {
                    position: corner_pos,
                    uv: *uv,
                    color: p.color,
                });
            }

            let base = (quad_idx * 4) as u32;
            // Two triangles: (0,1,2) and (0,2,3)
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        RenderBatch { vertices, indices }
    }
}

impl Default for ParticleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// High-level simulation tick helper
// ---------------------------------------------------------------------------

/// Convenience function: run one full simulation tick on a pool.
///
/// Applies gravity, turbulence, integrates, resolves collisions, recycles dead
/// particles, then emits new ones from the emitter.
pub fn tick(
    pool: &mut ParticlePool,
    emitter: &mut ParticleEmitter,
    gravity: &GravityForce,
    turbulence: &mut TurbulenceForce,
    collider: &ParticleCollider,
    color_over_life: &ColorOverLife,
    size_over_life: &SizeOverLife,
    dt: f32,
    rng: &mut Lcg,
) {
    gravity.apply_all(&mut pool.slots, dt);
    turbulence.apply_all(&mut pool.slots, dt);
    turbulence.advance(dt);
    ParticleIntegrator::step_all(&mut pool.slots, dt);
    collider.resolve_all(&mut pool.slots);
    color_over_life.apply_all(&mut pool.slots);
    size_over_life.apply_all(&mut pool.slots);
    pool.recycle_dead();

    let n = emitter.particles_to_emit(dt);
    for _ in 0..n {
        let particle = emitter.spawn(rng);
        pool.emit(particle);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GpuParticle --

    #[test]
    fn test_particle_alive_and_dead() {
        let alive = GpuParticle::new([0.0; 3], [0.0; 3], 1.0, [1.0; 4], 1.0);
        let dead = GpuParticle::new([0.0; 3], [0.0; 3], 0.0, [1.0; 4], 1.0);
        assert!(alive.is_alive());
        assert!(!dead.is_alive());
    }

    #[test]
    fn test_particle_default_is_dead() {
        let p = GpuParticle::default();
        assert!(!p.is_alive());
    }

    #[test]
    fn test_particle_color_stored() {
        let c = [0.1, 0.2, 0.3, 0.4];
        let p = GpuParticle::new([0.0; 3], [0.0; 3], 1.0, c, 1.0);
        assert_eq!(p.color, c);
    }

    // -- ParticleIntegrator --

    #[test]
    fn test_integrator_moves_position() {
        let mut p = GpuParticle::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 5.0, [1.0; 4], 1.0);
        ParticleIntegrator::step(&mut p, 1.0);
        assert!((p.position[0] - 1.0).abs() < 1e-6);
        assert!((p.position[1] - 2.0).abs() < 1e-6);
        assert!((p.position[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_integrator_decrements_life() {
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 1.0, [1.0; 4], 1.0);
        ParticleIntegrator::step(&mut p, 0.1);
        assert!((p.life - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_integrator_skips_dead_particle() {
        let p = GpuParticle::default(); // life == 0
        let pos_before = p.position;
        ParticleIntegrator::step_all(&mut [p.clone()], 1.0);
        // step_all on a dead particle should not move it
        let mut particles = vec![GpuParticle::default()];
        ParticleIntegrator::step_all(&mut particles, 1.0);
        assert_eq!(particles[0].position, pos_before);
    }

    #[test]
    fn test_integrator_step_all() {
        let mut particles = vec![
            GpuParticle::new([0.0; 3], [1.0, 0.0, 0.0], 2.0, [1.0; 4], 1.0),
            GpuParticle::new([0.0; 3], [0.0, 1.0, 0.0], 2.0, [1.0; 4], 1.0),
        ];
        ParticleIntegrator::step_all(&mut particles, 0.5);
        assert!((particles[0].position[0] - 0.5).abs() < 1e-6);
        assert!((particles[1].position[1] - 0.5).abs() < 1e-6);
    }

    // -- GravityForce --

    #[test]
    fn test_gravity_accelerates_down() {
        let g = GravityForce::earth();
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 5.0, [1.0; 4], 1.0);
        g.apply(&mut p, 1.0);
        assert!((p.velocity[1] - (-9.81)).abs() < 1e-4);
    }

    #[test]
    fn test_gravity_custom() {
        let g = GravityForce::new([0.0, -1.0, 0.0]);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 5.0, [1.0; 4], 1.0);
        g.apply(&mut p, 2.0);
        assert!((p.velocity[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gravity_apply_all_skips_dead() {
        let g = GravityForce::earth();
        let mut particles = vec![
            GpuParticle::new([0.0; 3], [0.0; 3], 2.0, [1.0; 4], 1.0),
            GpuParticle::default(), // dead
        ];
        g.apply_all(&mut particles, 1.0);
        assert!((particles[0].velocity[1] - (-9.81)).abs() < 1e-4);
        assert!((particles[1].velocity[1]).abs() < 1e-9); // unchanged
    }

    // -- Gradient noise --

    /// Deterministic sample points in the cube `[-extent, extent]³`.
    fn sample_points(count: usize, extent: f32, seed: u64) -> Vec<[f32; 3]> {
        let mut rng = Lcg::new(seed);
        (0..count)
            .map(|_| {
                [
                    rng.range_f32(-extent, extent),
                    rng.range_f32(-extent, extent),
                    rng.range_f32(-extent, extent),
                ]
            })
            .collect()
    }

    fn norm3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn test_noise_gradient_table_is_unit_length() {
        for (index, g) in NOISE_GRADIENTS.iter().enumerate() {
            assert!((norm3(*g) - 1.0).abs() < 1e-6, "gradient {index}: {g:?}");
        }
    }

    #[test]
    fn test_gradient_noise_vanishes_at_lattice_points() {
        for p in [[0.0, 0.0, 0.0], [3.0, -2.0, 5.0], [-7.0, 11.0, 1.0]] {
            for channel in 0..3 {
                let (value, gradient) = gradient_noise(p, channel);
                assert_eq!(value, 0.0, "lattice point {p:?}");
                // At a lattice point the gradient is the corner's own vector.
                assert!((norm3(gradient) - 1.0).abs() < 1e-6, "{gradient:?}");
            }
        }
    }

    #[test]
    fn test_gradient_noise_is_bounded_and_not_degenerate() {
        let points = sample_points(20_000, 30.0, 3);
        let mut sum_sq = 0.0_f64;
        let mut peak = 0.0_f32;
        for p in &points {
            let (value, gradient) = gradient_noise(*p, 0);
            assert!(value.is_finite() && norm3(gradient) < 3.0);
            peak = peak.max(value.abs());
            sum_sq += f64::from(value * value);
        }
        let rms = (sum_sq / points.len() as f64).sqrt();
        assert!(peak < 1.0, "noise peak {peak}");
        assert!(rms > 0.1 && rms < 0.4, "noise RMS {rms}");
    }

    #[test]
    fn test_gradient_noise_analytic_gradient_matches_finite_differences() {
        let h = 4.0e-3_f32;
        for channel in 0..3 {
            for p in sample_points(400, 6.0, 10 + u64::from(channel)) {
                let (_, analytic) = gradient_noise(p, channel);
                for (axis, &expected) in analytic.iter().enumerate() {
                    let mut up = p;
                    up[axis] += h;
                    let mut down = p;
                    down[axis] -= h;
                    let numeric = (gradient_noise(up, channel).0 - gradient_noise(down, channel).0)
                        / (2.0 * h);
                    assert!(
                        (numeric - expected).abs() < 5e-3,
                        "channel {channel} axis {axis} at {p:?}: {numeric} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gradient_noise_is_smooth_across_cell_boundaries() {
        // Value and gradient are continuous on the lattice planes (C2 noise).
        let delta = 1.0e-4_f32;
        for base in sample_points(40, 8.0, 17) {
            for axis in 0..3 {
                for plane in [-3.0_f32, 0.0, 2.0, 5.0] {
                    let mut below = base;
                    below[axis] = plane - delta;
                    let mut above = base;
                    above[axis] = plane + delta;
                    let (v0, g0) = gradient_noise(below, 1);
                    let (v1, g1) = gradient_noise(above, 1);
                    assert!((v1 - v0).abs() < 1e-3, "value jumps at {below:?}");
                    let jump = norm3([g1[0] - g0[0], g1[1] - g0[1], g1[2] - g0[2]]);
                    assert!(jump < 2e-2, "gradient jumps by {jump} at {below:?}");
                }
            }
        }
    }

    #[test]
    fn test_gradient_noise_channels_are_independent_fields() {
        let points = sample_points(500, 10.0, 21);
        let differing = points
            .iter()
            .filter(|p| (gradient_noise(**p, 0).0 - gradient_noise(**p, 1).0).abs() > 1e-3)
            .count();
        assert!(differing > 400, "channels 0 and 1 agree at too many points");
    }

    // -- TurbulenceForce --

    #[test]
    fn test_turbulence_produces_perturbation() {
        let turb = TurbulenceForce::new(5.0, 1.0);
        let points = sample_points(2_000, 10.0, 1);
        let mean =
            points.iter().map(|p| norm3(turb.curl_at(*p))).sum::<f32>() / points.len() as f32;
        assert!(mean > 2.0, "mean curl magnitude {mean} for strength 5");
        assert!(turb.curl_at([1.0, 2.0, 3.0]).iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_turbulence_advance_changes_field() {
        let mut turb = TurbulenceForce::new(1.0, 1.0);
        let before = turb.curl_at([1.3, 1.1, 0.7]);
        turb.advance(100.0);
        assert_eq!(turb.time_offset, 100.0);
        let after = turb.curl_at([1.3, 1.1, 0.7]);
        let change = norm3([
            after[0] - before[0],
            after[1] - before[1],
            after[2] - before[2],
        ]);
        assert!(
            change > 0.05,
            "advancing time left the field alone: {change}"
        );
    }

    #[test]
    fn test_turbulence_is_continuous_in_time() {
        let mut turb = TurbulenceForce::new(2.0, 1.5);
        turb.advance(3.0);
        let points = sample_points(500, 8.0, 5);
        let reference: Vec<[f32; 3]> = points.iter().map(|p| turb.curl_at(*p)).collect();
        turb.advance(1.0e-3);
        let mut moved = 0.0_f32;
        let mut scale = 0.0_f32;
        for (p, before) in points.iter().zip(reference.iter()) {
            let after = turb.curl_at(*p);
            moved += norm3([
                after[0] - before[0],
                after[1] - before[1],
                after[2] - before[2],
            ]);
            scale += norm3(*before);
        }
        assert!(
            moved < 0.01 * scale,
            "field jumps in time: {moved} vs {scale}"
        );
        // Advancing is additive.
        let mut a = TurbulenceForce::new(1.0, 1.0);
        a.advance(0.25);
        a.advance(0.5);
        let mut b = TurbulenceForce::new(1.0, 1.0);
        b.advance(0.75);
        assert_eq!(a.curl_at([0.3, 0.2, 0.1]), b.curl_at([0.3, 0.2, 0.1]));
    }

    /// The old hash "noise" gave independent values 0.001 apart, so its finite
    /// differences were uncorrelated: the field must now be spatially coherent.
    #[test]
    fn test_turbulence_is_spatially_coherent() {
        for frequency in [0.5_f32, 1.0, 3.0] {
            let turb = TurbulenceForce::new(1.0, frequency);
            let step = 1.0e-3 / frequency;
            let points = sample_points(10_000, 20.0 / frequency, 7);
            let mut change = 0.0_f64;
            let mut magnitude = 0.0_f64;
            let mut worst = 0.0_f32;
            for p in &points {
                let here = turb.curl_at(*p);
                let there = turb.curl_at([p[0] + step, p[1] + step, p[2] + step]);
                let delta = norm3([there[0] - here[0], there[1] - here[1], there[2] - here[2]]);
                change += f64::from(delta);
                magnitude += f64::from(norm3(here));
                worst = worst.max(delta);
            }
            assert!(
                change < 0.05 * magnitude,
                "f = {frequency}: mean |dC| {} vs mean |C| {}",
                change / points.len() as f64,
                magnitude / points.len() as f64
            );
            assert!(worst < 0.1, "f = {frequency}: a point jumps by {worst}");
        }
    }

    /// The magnitude is of order `strength` and independent of the frequency
    /// (the old finite differences scaled as 1/eps of the white noise).
    #[test]
    fn test_turbulence_magnitude_is_of_order_strength() {
        for frequency in [0.25_f32, 1.0, 4.0] {
            let turb = TurbulenceForce::new(3.0, frequency);
            let points = sample_points(10_000, 20.0 / frequency, 9);
            let mut sum = 0.0_f32;
            let mut peak = 0.0_f32;
            for p in &points {
                let m = norm3(turb.curl_at(*p));
                sum += m;
                peak = peak.max(m);
            }
            let mean = sum / points.len() as f32;
            assert!(
                mean > 0.9 * 3.0 && mean < 1.5 * 3.0,
                "f = {frequency}: mean |curl| {mean} for strength 3"
            );
            assert!(peak < 5.0 * 3.0, "f = {frequency}: peak {peak}");
        }
    }

    #[test]
    fn test_turbulence_scales_linearly_with_strength() {
        let weak = TurbulenceForce::new(1.0, 1.0);
        let strong = TurbulenceForce::new(4.0, 1.0);
        let none = TurbulenceForce::new(0.0, 1.0);
        for p in sample_points(50, 5.0, 2) {
            let a = weak.curl_at(p);
            let b = strong.curl_at(p);
            for axis in 0..3 {
                assert!((b[axis] - 4.0 * a[axis]).abs() < 1e-5);
            }
            assert_eq!(none.curl_at(p), [0.0; 3]);
        }
    }

    /// `div(curl ψ) = 0`: central differences of `curl_at` give a divergence
    /// that vanishes against the size of the Jacobian (a sign error in any
    /// component of the curl would leave about a third of it).
    #[test]
    fn test_turbulence_field_is_divergence_free() {
        for frequency in [1.0_f32, 2.5] {
            let turb = TurbulenceForce::new(1.0, frequency);
            let h = 4.0e-3 / frequency;
            let mut jacobian_total = 0.0_f32;
            let mut samples = 0.0_f32;
            for p in sample_points(1_500, 8.0 / frequency, 13) {
                let mut divergence = 0.0_f32;
                let mut jacobian_norm = 0.0_f32;
                for axis in 0..3 {
                    let mut up = p;
                    up[axis] += h;
                    let mut down = p;
                    down[axis] -= h;
                    let (cu, cd) = (turb.curl_at(up), turb.curl_at(down));
                    for component in 0..3 {
                        let partial = (cu[component] - cd[component]) / (2.0 * h);
                        jacobian_norm += partial.abs();
                        if component == axis {
                            divergence += partial;
                        }
                    }
                }
                assert!(
                    divergence.abs() < 0.01 * jacobian_norm.max(1.0),
                    "f = {frequency}: div {divergence} at {p:?} (|J|_1 = {jacobian_norm})"
                );
                jacobian_total += jacobian_norm;
                samples += 1.0;
            }
            // The test is only meaningful if the field actually varies.
            assert!(jacobian_total / samples > 1.0 * frequency);
        }
    }

    #[test]
    fn test_turbulence_apply_changes_velocity_by_curl_times_dt() {
        let turb = TurbulenceForce::new(100.0, 0.5);
        let mut p = GpuParticle::new([1.23, 4.56, 7.89], [1.0, 2.0, 3.0], 3.0, [1.0; 4], 1.0);
        let curl = turb.curl_at(p.position);
        assert!(norm3(curl) > 1.0, "the field is non-zero at this position");
        turb.apply(&mut p, 0.1);
        let expected = [
            1.0 + curl[0] * 0.1,
            2.0 + curl[1] * 0.1,
            3.0 + curl[2] * 0.1,
        ];
        for (actual, want) in p.velocity.iter().zip(expected.iter()) {
            assert!((actual - want).abs() < 1e-5);
        }
        // Position and life are not touched by the force itself.
        assert_eq!(p.position, [1.23, 4.56, 7.89]);
        assert_eq!(p.life, 3.0);
    }

    #[test]
    fn test_turbulence_apply_all_skips_dead_particles() {
        let turb = TurbulenceForce::new(10.0, 1.0);
        let mut particles = vec![
            GpuParticle::new([0.4, 0.5, 0.6], [0.0; 3], 1.0, [1.0; 4], 1.0),
            GpuParticle::default(),
        ];
        particles[1].position = [0.4, 0.5, 0.6];
        turb.apply_all(&mut particles, 0.5);
        assert!(norm3(particles[0].velocity) > 0.1);
        assert_eq!(particles[1].velocity, [0.0; 3]);
    }

    #[test]
    fn test_turbulence_is_deterministic() {
        let turb = TurbulenceForce::new(2.0, 1.3);
        for p in sample_points(20, 4.0, 4) {
            assert_eq!(turb.curl_at(p), turb.curl_at(p));
        }
    }

    // -- ParticleCollider --

    #[test]
    fn test_collider_floor_resolves_below() {
        let floor = ParticleCollider::floor(0.0, 0.8);
        let mut p = GpuParticle::new([0.0, -0.5, 0.0], [0.0, -1.0, 0.0], 5.0, [1.0; 4], 1.0);
        floor.resolve(&mut p);
        assert!(p.position[1] >= 0.0);
        assert!(p.velocity[1] >= 0.0);
    }

    #[test]
    fn test_collider_restitution_elastic() {
        let floor = ParticleCollider::floor(0.0, 1.0);
        let mut p = GpuParticle::new([0.0, -0.1, 0.0], [0.0, -2.0, 0.0], 5.0, [1.0; 4], 1.0);
        floor.resolve(&mut p);
        assert!(
            (p.velocity[1] - 2.0).abs() < 1e-5,
            "elastic: vy={}",
            p.velocity[1]
        );
    }

    #[test]
    fn test_collider_restitution_inelastic() {
        let floor = ParticleCollider::floor(0.0, 0.0);
        let mut p = GpuParticle::new([0.0, -0.1, 0.0], [0.0, -3.0, 0.0], 5.0, [1.0; 4], 1.0);
        floor.resolve(&mut p);
        assert!(
            (p.velocity[1]).abs() < 1e-5,
            "inelastic: vy={}",
            p.velocity[1]
        );
    }

    #[test]
    fn test_collider_no_collision_above() {
        let floor = ParticleCollider::floor(0.0, 0.8);
        let mut p = GpuParticle::new([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], 5.0, [1.0; 4], 1.0);
        let pos_before = p.position;
        let vel_before = p.velocity;
        floor.resolve(&mut p);
        assert_eq!(p.position, pos_before);
        assert_eq!(p.velocity, vel_before);
    }

    #[test]
    fn test_collider_resolve_all() {
        let floor = ParticleCollider::floor(0.0, 0.5);
        let mut particles = vec![
            GpuParticle::new([0.0, -1.0, 0.0], [0.0, -2.0, 0.0], 5.0, [1.0; 4], 1.0),
            GpuParticle::new([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], 5.0, [1.0; 4], 1.0),
        ];
        floor.resolve_all(&mut particles);
        assert!(particles[0].position[1] >= 0.0);
        assert!((particles[1].position[1] - 1.0).abs() < 1e-6);
    }

    // -- ParticlePool --

    #[test]
    fn test_pool_capacity_and_free_count() {
        let pool = ParticlePool::new(100);
        assert_eq!(pool.capacity(), 100);
        assert_eq!(pool.free_count(), 100);
        assert_eq!(pool.alive_count(), 0);
    }

    #[test]
    fn test_pool_emit_and_alive_count() {
        let mut pool = ParticlePool::new(10);
        let p = GpuParticle::new([0.0; 3], [0.0; 3], 5.0, [1.0; 4], 1.0);
        pool.emit(p).unwrap();
        assert_eq!(pool.alive_count(), 1);
        assert_eq!(pool.free_count(), 9);
    }

    #[test]
    fn test_pool_full_returns_none() {
        let mut pool = ParticlePool::new(2);
        let p = || GpuParticle::new([0.0; 3], [0.0; 3], 5.0, [1.0; 4], 1.0);
        pool.emit(p()).unwrap();
        pool.emit(p()).unwrap();
        assert!(pool.emit(p()).is_none());
    }

    #[test]
    fn test_pool_recycle_dead() {
        let mut pool = ParticlePool::new(5);
        let live = GpuParticle::new([0.0; 3], [0.0; 3], 10.0, [1.0; 4], 1.0);
        pool.emit(live).unwrap();
        // Kill all slots manually
        for slot in pool.slots.iter_mut() {
            slot.life = -1.0;
        }
        pool.recycle_dead();
        assert_eq!(pool.free_count(), 5);
        assert_eq!(pool.alive_count(), 0);
    }

    #[test]
    fn test_pool_alive_iter() {
        let mut pool = ParticlePool::new(5);
        let p = GpuParticle::new([1.0, 2.0, 3.0], [0.0; 3], 3.0, [1.0; 4], 1.0);
        pool.emit(p).unwrap();
        let alive: Vec<_> = pool.alive_iter().collect();
        assert_eq!(alive.len(), 1);
        assert_eq!(alive[0].position, [1.0, 2.0, 3.0]);
    }

    /// Structural invariants of the pool: the alive and free slots partition
    /// the capacity, the free list has no duplicates, and the `in_free` flags
    /// mirror the free list exactly.
    fn assert_pool_consistent(pool: &ParticlePool) {
        assert_eq!(pool.alive_count() + pool.free_count(), pool.capacity());
        let mut sorted = pool.free_list.clone();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), before, "free list contains duplicate slots");
        assert!(sorted.iter().all(|&i| i < pool.capacity()));
        assert_eq!(pool.in_free.len(), pool.capacity());
        for (i, &flag) in pool.in_free.iter().enumerate() {
            assert_eq!(
                flag,
                sorted.binary_search(&i).is_ok(),
                "in_free[{i}] out of sync with the free list"
            );
        }
    }

    fn live_particle() -> GpuParticle {
        GpuParticle::new([0.0; 3], [0.0; 3], 1.0, [1.0; 4], 1.0)
    }

    /// Regression for the O(capacity x free) recycle sweep: a 200k pool with
    /// 1000 dead particles must recycle in a single linear pass (the former
    /// `Vec::contains` scan needed ~4e10 comparisons here).
    #[test]
    fn test_pool_recycle_dead_large_capacity_consistent() {
        let capacity = 200_000;
        let mut pool = ParticlePool::new(capacity);
        let slots: Vec<usize> = (0..1000)
            .map(|_| pool.emit(live_particle()).unwrap())
            .collect();
        assert_eq!(pool.alive_count(), 1000);
        assert_pool_consistent(&pool);

        for &s in &slots {
            pool.slots[s].life = 0.0;
        }
        pool.recycle_dead();
        assert_eq!(pool.alive_count(), 0);
        assert_eq!(pool.free_count(), capacity);
        assert_pool_consistent(&pool);
    }

    /// Same scenario through the high-level `tick` entry point: every particle
    /// expires within the step and is recycled by the sweep inside `tick`.
    #[test]
    fn test_tick_recycles_expired_particles_in_large_pool() {
        let capacity = 200_000;
        let mut pool = ParticlePool::new(capacity);
        for _ in 0..1000 {
            let p = GpuParticle::new([0.0, 5.0, 0.0], [0.0; 3], 0.05, [1.0; 4], 1.0);
            pool.emit(p).unwrap();
        }
        // Zero emission rate: nothing is spawned back.
        let mut emitter = ParticleEmitter::new([0.0; 3], 0.0, [0.0; 3], 0.0, 1.0);
        let gravity = GravityForce::new([0.0; 3]);
        let mut turbulence = TurbulenceForce::new(0.0, 1.0);
        let collider = ParticleCollider::floor(-100.0, 0.5);
        let col = ColorOverLife::new([1.0; 4], [0.0; 4], 1.0);
        let sizer = SizeOverLife::new(1.0, 0.0, 1.0);
        let mut rng = Lcg::new(11);

        tick(
            &mut pool,
            &mut emitter,
            &gravity,
            &mut turbulence,
            &collider,
            &col,
            &sizer,
            0.1,
            &mut rng,
        );
        assert_eq!(pool.alive_count(), 0);
        assert_eq!(pool.free_count(), capacity);
        assert_pool_consistent(&pool);
    }

    #[test]
    fn test_pool_recycle_dead_is_idempotent() {
        let mut pool = ParticlePool::new(64);
        let slots: Vec<usize> = (0..10)
            .map(|_| pool.emit(live_particle()).unwrap())
            .collect();
        for &s in slots.iter().take(6) {
            pool.slots[s].life = -1.0;
        }
        pool.recycle_dead();
        let free_after_first = pool.free_count();
        assert_eq!(free_after_first, 64 - 4);
        pool.recycle_dead();
        pool.recycle_dead();
        assert_eq!(pool.free_count(), free_after_first);
        assert_eq!(pool.alive_count(), 4);
        assert_pool_consistent(&pool);
    }

    /// Recycled slots are reusable and the pool never hands out a slot that is
    /// still occupied by a live particle.
    #[test]
    fn test_pool_reuses_recycled_slots_without_aliasing() {
        let mut pool = ParticlePool::new(8);
        let slots: Vec<usize> = (0..8)
            .map(|_| pool.emit(live_particle()).unwrap())
            .collect();
        assert!(pool.emit(live_particle()).is_none());

        // Kill the even slots only.
        let killed: Vec<usize> = slots.iter().copied().filter(|s| s % 2 == 0).collect();
        for &s in &killed {
            pool.slots[s].life = 0.0;
        }
        pool.recycle_dead();
        assert_eq!(pool.free_count(), killed.len());

        let mut reused: Vec<usize> = (0..killed.len())
            .map(|_| pool.emit(live_particle()).unwrap())
            .collect();
        assert!(pool.emit(live_particle()).is_none());
        reused.sort_unstable();
        let mut expected = killed.clone();
        expected.sort_unstable();
        assert_eq!(reused, expected, "only the killed slots may be reused");
        assert_pool_consistent(&pool);
    }

    /// Random emit / kill / recycle cycles never corrupt the pool: after every
    /// phase the free list has no duplicates, `free_count() + alive == capacity`
    /// and the `in_free` flags mirror the free list.
    #[test]
    fn test_pool_random_emit_kill_recycle_cycles_stay_consistent() {
        let capacity = 97;
        let mut pool = ParticlePool::new(capacity);
        let mut rng = Lcg::new(0xB00B5);
        for cycle in 0..300 {
            // Emit a random burst (it may exceed the free slots).
            let burst = (rng.next_f32() * 40.0) as usize;
            for _ in 0..burst {
                let emitted = pool.emit(live_particle());
                if let Some(idx) = emitted {
                    assert!(pool.slots[idx].is_alive(), "cycle {cycle}: slot {idx}");
                    assert!(!pool.in_free[idx]);
                }
            }
            assert_pool_consistent(&pool);
            // Kill a random subset of the slots (including free ones).
            for slot in pool.slots.iter_mut() {
                if rng.next_f32() < 0.35 {
                    slot.life = 0.0;
                }
            }
            // Recycle once or several times: the result must be identical.
            pool.recycle_dead();
            assert_pool_consistent(&pool);
            let free_after_first = pool.free_count();
            let repeats = (rng.next_f32() * 3.0) as usize;
            for _ in 0..repeats {
                pool.recycle_dead();
            }
            assert_eq!(pool.free_count(), free_after_first);
            let alive_slots = pool.slots.iter().filter(|s| s.is_alive()).count();
            assert_eq!(pool.alive_count(), alive_slots);
            assert_eq!(pool.free_count() + alive_slots, capacity);
        }
    }

    /// A particle emitted already dead occupies its slot until the next sweep
    /// reclaims it; it must not leak the slot permanently.
    #[test]
    fn test_pool_emit_dead_particle_is_reclaimed() {
        let mut pool = ParticlePool::new(3);
        let dead = GpuParticle::new([0.0; 3], [0.0; 3], 0.0, [1.0; 4], 1.0);
        let idx = pool.emit(dead).unwrap();
        assert_eq!(pool.free_count(), 2);
        assert!(!pool.in_free[idx]);
        pool.recycle_dead();
        assert_eq!(pool.free_count(), 3);
        assert!(pool.in_free[idx]);
        assert_pool_consistent(&pool);
    }

    // -- ParticleEmitter --

    #[test]
    fn test_emitter_particles_to_emit_accumulates() {
        let mut emitter = ParticleEmitter::new([0.0; 3], 10.0, [0.0, 1.0, 0.0], 0.0, 2.0);
        let n = emitter.particles_to_emit(0.5); // 10 * 0.5 = 5
        assert_eq!(n, 5);
    }

    #[test]
    fn test_emitter_spawn_sets_life() {
        let emitter = ParticleEmitter::new([0.0; 3], 1.0, [0.0; 3], 0.0, 3.5);
        let mut rng = Lcg::new(42);
        let p = emitter.spawn(&mut rng);
        assert!((p.life - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_emitter_spawn_uses_position() {
        let emitter = ParticleEmitter::new([1.0, 2.0, 3.0], 1.0, [0.0; 3], 0.0, 1.0);
        let mut rng = Lcg::new(7);
        let p = emitter.spawn(&mut rng);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_emitter_spawn_with_spread() {
        let emitter = ParticleEmitter::new([0.0; 3], 1.0, [0.0, 1.0, 0.0], 0.5, 1.0);
        let mut rng = Lcg::new(99);
        for _ in 0..20 {
            let p = emitter.spawn(&mut rng);
            assert!(p.velocity[1] >= 0.5 && p.velocity[1] <= 1.5);
        }
    }

    // -- ColorOverLife --

    #[test]
    fn test_color_over_life_at_birth() {
        let col = ColorOverLife::new([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], 2.0);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 2.0, [0.5; 4], 1.0);
        col.apply(&mut p);
        // t=1 → birth color
        assert!((p.color[0] - 1.0).abs() < 1e-5);
        assert!((p.color[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_color_over_life_at_death() {
        let col = ColorOverLife::new([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], 2.0);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 0.0, [0.5; 4], 1.0);
        col.apply(&mut p);
        // t=0 → death color
        assert!((p.color[0] - 0.0).abs() < 1e-5);
        assert!((p.color[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_color_over_life_midpoint() {
        let col = ColorOverLife::new([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], 2.0);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 1.0, [0.5; 4], 1.0);
        col.apply(&mut p);
        // t=0.5 → midpoint
        assert!((p.color[0] - 0.5).abs() < 1e-5);
        assert!((p.color[1] - 0.5).abs() < 1e-5);
    }

    // -- SizeOverLife --

    #[test]
    fn test_size_over_life_at_birth() {
        let sizer = SizeOverLife::new(2.0, 0.1, 3.0);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 3.0, [1.0; 4], 1.0);
        sizer.apply(&mut p);
        assert!((p.size - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_size_over_life_at_death() {
        let sizer = SizeOverLife::new(2.0, 0.1, 3.0);
        let mut p = GpuParticle::new([0.0; 3], [0.0; 3], 0.0, [1.0; 4], 1.0);
        sizer.apply(&mut p);
        assert!((p.size - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_size_over_life_apply_all() {
        let sizer = SizeOverLife::new(4.0, 0.0, 2.0);
        let mut particles = vec![
            GpuParticle::new([0.0; 3], [0.0; 3], 2.0, [1.0; 4], 0.0),
            GpuParticle::default(), // dead
        ];
        sizer.apply_all(&mut particles);
        assert!((particles[0].size - 4.0).abs() < 1e-5);
        assert!((particles[1].size - 1.0).abs() < 1e-5); // unchanged default
    }

    // -- ParticleRenderer --

    #[test]
    fn test_renderer_empty_scene() {
        let renderer = ParticleRenderer::new();
        let batch = renderer.render(&[]);
        assert_eq!(batch.particle_count(), 0);
        assert!(batch.vertices.is_empty());
        assert!(batch.indices.is_empty());
    }

    #[test]
    fn test_renderer_single_particle_quad() {
        let renderer = ParticleRenderer::new();
        let p = GpuParticle::new([0.0, 0.0, 0.0], [0.0; 3], 1.0, [1.0; 4], 0.2);
        let batch = renderer.render(&[p]);
        assert_eq!(batch.particle_count(), 1);
        assert_eq!(batch.vertices.len(), 4);
        assert_eq!(batch.indices.len(), 6);
    }

    #[test]
    fn test_renderer_dead_particle_excluded() {
        let renderer = ParticleRenderer::new();
        let dead = GpuParticle::default();
        let batch = renderer.render(&[dead]);
        assert_eq!(batch.particle_count(), 0);
    }

    #[test]
    fn test_renderer_index_buffer_valid() {
        let renderer = ParticleRenderer::new();
        let particles: Vec<GpuParticle> = (0..3)
            .map(|i| GpuParticle::new([i as f32, 0.0, 0.0], [0.0; 3], 1.0, [1.0; 4], 0.1))
            .collect();
        let batch = renderer.render(&particles);
        assert_eq!(batch.vertices.len(), 12);
        assert_eq!(batch.indices.len(), 18);
        // All indices must be in bounds
        for &idx in &batch.indices {
            assert!((idx as usize) < batch.vertices.len());
        }
    }

    #[test]
    fn test_renderer_depth_sort_back_to_front() {
        let renderer = ParticleRenderer::new(); // camera at z=10
        // Particle closer to camera (z=8) should come second (front)
        // Particle farther from camera (z=0) should come first (back)
        let p_far = GpuParticle::new([0.0, 0.0, 0.0], [0.0; 3], 1.0, [1.0, 0.0, 0.0, 1.0], 0.1);
        let p_near = GpuParticle::new([0.0, 0.0, 8.0], [0.0; 3], 1.0, [0.0, 1.0, 0.0, 1.0], 0.1);
        let batch = renderer.render(&[p_far, p_near]);
        // Far particle's quad (red) should be first in the batch
        assert_eq!(batch.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    // -- Lcg --

    #[test]
    fn test_lcg_different_seeds() {
        let mut rng1 = Lcg::new(1);
        let mut rng2 = Lcg::new(2);
        let v1 = rng1.next_f32();
        let v2 = rng2.next_f32();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_lcg_range_f32_bounds() {
        let mut rng = Lcg::new(123);
        for _ in 0..1000 {
            let v = rng.range_f32(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&v) || (v - 1.0).abs() < 1e-6);
        }
    }

    // -- Integration tick test --

    #[test]
    fn test_tick_emits_particles() {
        let mut pool = ParticlePool::new(200);
        let mut emitter = ParticleEmitter::new([0.0, 1.0, 0.0], 50.0, [0.0, 2.0, 0.0], 0.1, 2.0);
        let gravity = GravityForce::earth();
        let mut turbulence = TurbulenceForce::new(0.1, 1.0);
        let collider = ParticleCollider::floor(0.0, 0.5);
        let col = ColorOverLife::new([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], 2.0);
        let sizer = SizeOverLife::new(0.2, 0.01, 2.0);
        let mut rng = Lcg::new(42);

        tick(
            &mut pool,
            &mut emitter,
            &gravity,
            &mut turbulence,
            &collider,
            &col,
            &sizer,
            0.1,
            &mut rng,
        );
        assert!(pool.alive_count() > 0);
    }

    #[test]
    fn test_tick_particles_age() {
        let mut pool = ParticlePool::new(100);
        let mut emitter = ParticleEmitter::new([0.0; 3], 100.0, [0.0, 1.0, 0.0], 0.0, 0.5);
        let gravity = GravityForce::earth();
        let mut turbulence = TurbulenceForce::new(0.0, 1.0);
        let collider = ParticleCollider::floor(-100.0, 0.5); // far below
        let col = ColorOverLife::new([1.0; 4], [0.0; 4], 0.5);
        let sizer = SizeOverLife::new(1.0, 0.0, 0.5);
        let mut rng = Lcg::new(7);

        // Emit first
        tick(
            &mut pool,
            &mut emitter,
            &gravity,
            &mut turbulence,
            &collider,
            &col,
            &sizer,
            0.1,
            &mut rng,
        );
        let alive_after_emit = pool.alive_count();

        // Let them expire
        for _ in 0..10 {
            tick(
                &mut pool,
                &mut emitter,
                &gravity,
                &mut turbulence,
                &collider,
                &col,
                &sizer,
                0.1,
                &mut rng,
            );
        }
        // Some particles will die and be replaced; pool should still function
        assert!(pool.alive_count() <= pool.capacity());
        let _ = alive_after_emit;
    }
}
