//! Particle simulation system for `OxiMedia` VFX.
//!
//! Provides a flexible particle system with emitters and per-particle physics.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A single particle with position, velocity and lifetime state.
#[derive(Debug, Clone)]
pub struct Particle {
    /// X position in normalized screen space.
    pub x: f32,
    /// Y position in normalized screen space.
    pub y: f32,
    /// X velocity per second.
    pub vx: f32,
    /// Y velocity per second.
    pub vy: f32,
    /// Total lifetime in seconds.
    pub lifetime: f32,
    /// Age of the particle in seconds.
    pub age: f32,
    /// Size of the particle in pixels.
    pub size: f32,
    /// Alpha opacity (0.0–1.0).
    pub alpha: f32,
}

impl Particle {
    /// Create a new particle.
    #[must_use]
    pub fn new(x: f32, y: f32, vx: f32, vy: f32, lifetime: f32, size: f32) -> Self {
        Self {
            x,
            y,
            vx,
            vy,
            lifetime: lifetime.max(0.001),
            age: 0.0,
            size: size.max(0.1),
            alpha: 1.0,
        }
    }

    /// Returns `true` while the particle is alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.age < self.lifetime
    }

    /// Advance the particle by `dt` seconds applying gravity.
    pub fn update_position(&mut self, dt: f32, gravity: f32) {
        if !self.is_alive() {
            return;
        }
        self.vy += gravity * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.age += dt;

        // Fade out linearly over lifetime
        let t = (self.age / self.lifetime).clamp(0.0, 1.0);
        self.alpha = 1.0 - t;
    }

    /// Normalised age in [0, 1].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn normalized_age(&self) -> f32 {
        (self.age / self.lifetime).clamp(0.0, 1.0)
    }
}

/// Configuration for a [`ParticleEmitter`].
#[derive(Debug, Clone)]
pub struct EmitterConfig {
    /// Particles spawned per second.
    pub emit_rate: f32,
    /// Minimum particle lifetime in seconds.
    pub lifetime_min: f32,
    /// Maximum particle lifetime in seconds.
    pub lifetime_max: f32,
    /// Initial speed range minimum.
    pub speed_min: f32,
    /// Initial speed range maximum.
    pub speed_max: f32,
    /// Emission angle in radians (0 = upward).
    pub angle: f32,
    /// Half-angle spread in radians.
    pub spread: f32,
    /// Gravity acceleration applied each second.
    pub gravity: f32,
    /// Maximum particles alive at once.
    pub max_particles: usize,
    /// Particle size in pixels.
    pub particle_size: f32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            emit_rate: 60.0,
            lifetime_min: 1.0,
            lifetime_max: 3.0,
            speed_min: 0.1,
            speed_max: 0.3,
            angle: -PI / 2.0, // straight up
            spread: PI / 6.0,
            gravity: 0.2,
            max_particles: 512,
            particle_size: 4.0,
        }
    }
}

/// Emitter that spawns and manages a pool of particles.
#[derive(Debug)]
pub struct ParticleEmitter {
    /// Spawn position X.
    pub x: f32,
    /// Spawn position Y.
    pub y: f32,
    config: EmitterConfig,
    particles: Vec<Particle>,
    /// Fractional spawn accumulator.
    emit_accum: f32,
    /// Running time in seconds.
    time: f32,
}

impl ParticleEmitter {
    /// Create a new emitter at (`x`, `y`) with the given config.
    #[must_use]
    pub fn new(x: f32, y: f32, config: EmitterConfig) -> Self {
        Self {
            x,
            y,
            config,
            particles: Vec::new(),
            emit_accum: 0.0,
            time: 0.0,
        }
    }

    /// Spawn a single particle using a deterministic pseudo-random seed.
    pub fn emit(&mut self) {
        if self.particles.len() >= self.config.max_particles {
            // Recycle dead particles first — build the replacement before borrowing
            // the particle list mutably to satisfy the borrow checker.
            let replacement = self.make_particle();
            if let Some(dead) = self.particles.iter_mut().find(|p| !p.is_alive()) {
                *dead = replacement;
            }
            return;
        }
        let p = self.make_particle();
        self.particles.push(p);
    }

    fn make_particle(&self) -> Particle {
        // Use a simple deterministic LCG based on current time + count
        let seed = (self.time * 1_000_003.0) as u64 ^ (self.particles.len() as u64 * 6_364_136);
        let r1 = lcg_f32(seed);
        let r2 = lcg_f32(seed.wrapping_add(1));
        let r3 = lcg_f32(seed.wrapping_add(2));

        let speed = self.config.speed_min + r1 * (self.config.speed_max - self.config.speed_min);
        let angle_offset = (r2 - 0.5) * 2.0 * self.config.spread;
        let angle = self.config.angle + angle_offset;
        let vx = angle.cos() * speed;
        let vy = angle.sin() * speed;

        let lifetime =
            self.config.lifetime_min + r3 * (self.config.lifetime_max - self.config.lifetime_min);

        Particle::new(self.x, self.y, vx, vy, lifetime, self.config.particle_size)
    }

    /// Advance the emitter simulation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;

        // Spawn new particles
        self.emit_accum += self.config.emit_rate * dt;
        while self.emit_accum >= 1.0 {
            self.emit();
            self.emit_accum -= 1.0;
        }

        // Update alive particles
        let gravity = self.config.gravity;
        for p in &mut self.particles {
            p.update_position(dt, gravity);
        }
    }

    /// Number of currently alive particles.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.particles.iter().filter(|p| p.is_alive()).count()
    }

    /// Borrow the full particle slice (includes dead particles).
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Total particles ever allocated (alive + dead).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.particles.len()
    }
}

/// Minimal LCG pseudo-random number → [0, 1).
#[allow(clippy::cast_precision_loss)]
fn lcg_f32(seed: u64) -> f32 {
    let s = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((s >> 33) as f32) / (1u64 << 31) as f32
}

/// High-level particle system managing multiple emitters.
#[derive(Debug, Default)]
pub struct ParticleSystem {
    emitters: Vec<ParticleEmitter>,
}

impl ParticleSystem {
    /// Create an empty particle system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an emitter to the system and return its index.
    pub fn add_emitter(&mut self, emitter: ParticleEmitter) -> usize {
        let idx = self.emitters.len();
        self.emitters.push(emitter);
        idx
    }

    /// Step all emitters forward by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        for emitter in &mut self.emitters {
            emitter.update(dt);
        }
    }

    /// Total alive particles across all emitters.
    #[must_use]
    pub fn total_active(&self) -> usize {
        self.emitters
            .iter()
            .map(ParticleEmitter::active_count)
            .sum()
    }

    /// Number of emitters registered.
    #[must_use]
    pub fn emitter_count(&self) -> usize {
        self.emitters.len()
    }

    /// Borrow emitter by index.
    #[must_use]
    pub fn emitter(&self, idx: usize) -> Option<&ParticleEmitter> {
        self.emitters.get(idx)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_is_alive_initially() {
        let p = Particle::new(0.5, 0.5, 0.0, 0.0, 2.0, 4.0);
        assert!(p.is_alive());
    }

    #[test]
    fn test_particle_dies_after_lifetime() {
        let mut p = Particle::new(0.5, 0.5, 0.0, 0.0, 1.0, 4.0);
        p.update_position(1.1, 0.0);
        assert!(!p.is_alive());
    }

    #[test]
    fn test_particle_position_changes() {
        let mut p = Particle::new(0.0, 0.0, 1.0, 0.0, 5.0, 4.0);
        p.update_position(1.0, 0.0);
        assert!((p.x - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_particle_gravity() {
        let mut p = Particle::new(0.0, 0.0, 0.0, 0.0, 5.0, 4.0);
        p.update_position(1.0, 1.0);
        assert!(p.vy > 0.0);
        assert!(p.y > 0.0);
    }

    #[test]
    fn test_particle_alpha_fades() {
        let mut p = Particle::new(0.0, 0.0, 0.0, 0.0, 2.0, 4.0);
        p.update_position(1.0, 0.0); // half lifetime
        assert!(p.alpha < 1.0 && p.alpha > 0.0);
    }

    #[test]
    fn test_particle_normalized_age() {
        let mut p = Particle::new(0.0, 0.0, 0.0, 0.0, 4.0, 4.0);
        p.update_position(1.0, 0.0);
        assert!((p.normalized_age() - 0.25).abs() < 1e-3);
    }

    #[test]
    fn test_emitter_spawns_particles() {
        let mut e = ParticleEmitter::new(0.5, 0.5, EmitterConfig::default());
        e.update(0.1); // 6 particles expected at rate 60
        assert!(e.active_count() > 0);
    }

    #[test]
    fn test_emitter_active_count_increases() {
        let mut e = ParticleEmitter::new(0.5, 0.5, EmitterConfig::default());
        e.update(0.5);
        let count_a = e.active_count();
        e.update(0.5);
        let count_b = e.total_count();
        assert!(count_b >= count_a);
    }

    #[test]
    fn test_emitter_respects_max_particles() {
        let cfg = EmitterConfig {
            max_particles: 10,
            emit_rate: 1000.0,
            ..Default::default()
        };
        let mut e = ParticleEmitter::new(0.0, 0.0, cfg);
        e.update(1.0);
        assert!(e.total_count() <= 10);
    }

    #[test]
    fn test_particle_system_add_emitter() {
        let mut sys = ParticleSystem::new();
        let idx = sys.add_emitter(ParticleEmitter::new(0.0, 0.0, EmitterConfig::default()));
        assert_eq!(idx, 0);
        assert_eq!(sys.emitter_count(), 1);
    }

    #[test]
    fn test_particle_system_step() {
        let mut sys = ParticleSystem::new();
        sys.add_emitter(ParticleEmitter::new(0.5, 0.5, EmitterConfig::default()));
        sys.step(0.2);
        assert!(sys.total_active() > 0);
    }

    #[test]
    fn test_particle_system_multiple_emitters() {
        let mut sys = ParticleSystem::new();
        sys.add_emitter(ParticleEmitter::new(0.2, 0.5, EmitterConfig::default()));
        sys.add_emitter(ParticleEmitter::new(0.8, 0.5, EmitterConfig::default()));
        sys.step(0.5);
        assert_eq!(sys.emitter_count(), 2);
        assert!(sys.total_active() > 0);
    }

    #[test]
    fn test_particle_size_clamped() {
        let p = Particle::new(0.0, 0.0, 0.0, 0.0, 1.0, -5.0);
        assert!(p.size >= 0.1);
    }
}
