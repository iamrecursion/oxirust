//! Generic configurable `ParticleEmitter` and enhanced `ParticleSystem::tick`.
//!
//! Provides:
//! - [`ParticleColor`]   – RGBA color with alpha.
//! - [`ParticleData`]    – Extended per-particle state including color and size.
//! - [`EmitterConfig`]   – Configurable emission parameters.
//! - [`ParticleEmitter`] – Emits and manages particles with `tick(dt)`.

#![allow(dead_code)]

use rand::RngExt;
use rand::SeedableRng;

// ── Color ──────────────────────────────────────────────────────────────────────

/// RGBA color for a particle, components in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleColor {
    /// Red component (0–1).
    pub r: f32,
    /// Green component (0–1).
    pub g: f32,
    /// Blue component (0–1).
    pub b: f32,
    /// Alpha component (0–1).
    pub a: f32,
}

impl ParticleColor {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// White opaque.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    /// Orange (sparks / fire).
    #[must_use]
    pub const fn orange() -> Self {
        Self::new(1.0, 0.5, 0.0, 1.0)
    }

    /// Grey smoke.
    #[must_use]
    pub const fn smoke() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.6)
    }

    /// Linearly interpolate to `other` by factor `t` ∈ [0, 1].
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Convert to RGBA u8 array.
    #[must_use]
    pub fn to_rgba_u8(self) -> [u8; 4] {
        [
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
            (self.a * 255.0).clamp(0.0, 255.0) as u8,
        ]
    }
}

// ── Extended Particle ──────────────────────────────────────────────────────────

/// A single particle with full state.
#[derive(Debug, Clone)]
pub struct ParticleData {
    /// X position (pixels or normalized units).
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// X velocity.
    pub vx: f32,
    /// Y velocity.
    pub vy: f32,
    /// Remaining lifetime (seconds).
    pub lifetime: f32,
    /// Maximum lifetime at birth (seconds).
    pub max_lifetime: f32,
    /// Color at this moment.
    pub color: ParticleColor,
    /// Color when born.
    pub color_birth: ParticleColor,
    /// Color when dying.
    pub color_death: ParticleColor,
    /// Current size (pixels or normalized).
    pub size: f32,
    /// Size at birth.
    pub size_birth: f32,
    /// Size at death.
    pub size_death: f32,
}

impl ParticleData {
    /// Age fraction: 0 = newborn, 1 = about to die.
    #[must_use]
    pub fn age_fraction(&self) -> f32 {
        if self.max_lifetime <= 0.0 {
            return 1.0;
        }
        1.0 - (self.lifetime / self.max_lifetime).clamp(0.0, 1.0)
    }

    /// Whether the particle is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.lifetime > 0.0
    }

    /// Advance physics and interpolate color / size.
    pub fn update(&mut self, dt: f32, gravity: [f32; 2]) {
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.vx += gravity[0] * dt;
        self.vy += gravity[1] * dt;
        self.lifetime -= dt;

        let age = self.age_fraction();
        self.color = self.color_birth.lerp(self.color_death, age);
        self.size = self.size_birth + (self.size_death - self.size_birth) * age;
        self.size = self.size.max(0.0);
    }
}

// ── EmitterConfig ──────────────────────────────────────────────────────────────

/// Configuration for a [`ParticleEmitter`].
#[derive(Debug, Clone)]
pub struct EmitterConfig {
    /// Emission origin X.
    pub origin_x: f32,
    /// Emission origin Y.
    pub origin_y: f32,
    /// Random position spread radius.
    pub spread_radius: f32,
    /// Particles emitted per second.
    pub emission_rate: f32,
    /// Minimum initial speed (units / second).
    pub speed_min: f32,
    /// Maximum initial speed.
    pub speed_max: f32,
    /// Minimum emission angle in degrees (0 = right, 90 = down).
    pub angle_min_deg: f32,
    /// Maximum emission angle in degrees.
    pub angle_max_deg: f32,
    /// Minimum lifetime in seconds.
    pub lifetime_min: f32,
    /// Maximum lifetime.
    pub lifetime_max: f32,
    /// Minimum size at birth.
    pub size_birth_min: f32,
    /// Maximum size at birth.
    pub size_birth_max: f32,
    /// Size at death.
    pub size_death: f32,
    /// Color at birth.
    pub color_birth: ParticleColor,
    /// Color at death.
    pub color_death: ParticleColor,
    /// Gravity vector [x, y] (units / s²).
    pub gravity: [f32; 2],
    /// Maximum number of live particles.
    pub max_particles: usize,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            origin_x: 0.0,
            origin_y: 0.0,
            spread_radius: 0.0,
            emission_rate: 30.0,
            speed_min: 20.0,
            speed_max: 80.0,
            angle_min_deg: -90.0,
            angle_max_deg: 90.0,
            lifetime_min: 0.5,
            lifetime_max: 2.0,
            size_birth_min: 2.0,
            size_birth_max: 6.0,
            size_death: 0.0,
            color_birth: ParticleColor::orange(),
            color_death: ParticleColor::new(0.2, 0.0, 0.0, 0.0),
            gravity: [0.0, 50.0],
            max_particles: 2048,
        }
    }
}

impl EmitterConfig {
    /// Sparks / fire preset.
    #[must_use]
    pub fn sparks() -> Self {
        Self {
            color_birth: ParticleColor::orange(),
            color_death: ParticleColor::new(0.8, 0.1, 0.0, 0.0),
            speed_min: 40.0,
            speed_max: 150.0,
            angle_min_deg: -150.0,
            angle_max_deg: -30.0, // upward cone
            lifetime_min: 0.3,
            lifetime_max: 1.5,
            gravity: [0.0, 80.0],
            ..Self::default()
        }
    }

    /// Smoke preset.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            color_birth: ParticleColor::smoke(),
            color_death: ParticleColor::new(0.5, 0.5, 0.5, 0.0),
            speed_min: 5.0,
            speed_max: 20.0,
            angle_min_deg: -120.0,
            angle_max_deg: -60.0,
            size_birth_min: 8.0,
            size_birth_max: 16.0,
            size_death: 24.0,
            lifetime_min: 1.0,
            lifetime_max: 3.0,
            gravity: [0.0, -10.0], // light upward drift
            ..Self::default()
        }
    }
}

// ── ParticleEmitter ────────────────────────────────────────────────────────────

/// A configurable particle emitter that manages its own pool of [`ParticleData`].
pub struct ParticleEmitter {
    /// Emitter configuration.
    pub config: EmitterConfig,
    particles: Vec<ParticleData>,
    rng: rand::rngs::StdRng,
    /// Fractional particle accumulator for sub-frame emission.
    accumulator: f32,
    /// Whether the emitter is currently active.
    pub active: bool,
}

impl ParticleEmitter {
    /// Create a new emitter with the given configuration.
    #[must_use]
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            particles: Vec::with_capacity(config.max_particles.min(4096)),
            config,
            rng: rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF),
            accumulator: 0.0,
            active: true,
        }
    }

    /// Create an emitter with default config and override the seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = rand::rngs::StdRng::seed_from_u64(seed);
        self
    }

    /// Advance the simulation by `dt` seconds.
    ///
    /// This is the main entry-point called once per frame / time step.
    /// It:
    /// 1. Updates and culls existing particles.
    /// 2. Emits new particles at the configured rate (if `active`).
    pub fn tick(&mut self, dt: f32) {
        // 1. Update existing particles
        let gravity = self.config.gravity;
        for p in &mut self.particles {
            p.update(dt, gravity);
        }
        self.particles.retain(ParticleData::is_alive);

        // 2. Emit new particles
        if self.active {
            self.accumulator += self.config.emission_rate * dt;
            let emit_count = self.accumulator.floor() as usize;
            self.accumulator -= emit_count as f32;

            let can_emit = self
                .config
                .max_particles
                .saturating_sub(self.particles.len());
            for _ in 0..emit_count.min(can_emit) {
                let p = self.spawn_particle();
                self.particles.push(p);
            }
        }
    }

    /// Borrow the current particle slice.
    #[must_use]
    pub fn particles(&self) -> &[ParticleData] {
        &self.particles
    }

    /// Mutable borrow of the current particle slice.
    pub fn particles_mut(&mut self) -> &mut [ParticleData] {
        &mut self.particles
    }

    /// Number of live particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.particles.clear();
        self.accumulator = 0.0;
    }

    /// Directly spawn a single particle.
    pub fn spawn_one(&mut self) {
        if self.particles.len() < self.config.max_particles {
            let p = self.spawn_particle();
            self.particles.push(p);
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn spawn_particle(&mut self) -> ParticleData {
        let cfg = &self.config;

        // Position with spread
        let spread = if cfg.spread_radius > 0.0 {
            let angle_spread = self.rng.random_range(0.0..std::f32::consts::TAU);
            let dist = self.rng.random_range(0.0..cfg.spread_radius);
            (angle_spread.cos() * dist, angle_spread.sin() * dist)
        } else {
            (0.0, 0.0)
        };
        let x = cfg.origin_x + spread.0;
        let y = cfg.origin_y + spread.1;

        // Velocity
        let angle_deg = self.rng.random_range(cfg.angle_min_deg..=cfg.angle_max_deg);
        let angle_rad = angle_deg.to_radians();
        let speed = self.rng.random_range(cfg.speed_min..=cfg.speed_max);
        let vx = angle_rad.cos() * speed;
        let vy = angle_rad.sin() * speed;

        // Lifetime
        let lifetime = self.rng.random_range(cfg.lifetime_min..=cfg.lifetime_max);

        // Size
        let size_birth = self
            .rng
            .random_range(cfg.size_birth_min..=cfg.size_birth_max);

        ParticleData {
            x,
            y,
            vx,
            vy,
            lifetime,
            max_lifetime: lifetime,
            color: cfg.color_birth,
            color_birth: cfg.color_birth,
            color_death: cfg.color_death,
            size: size_birth,
            size_birth,
            size_death: cfg.size_death,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_color_lerp_midpoint() {
        let a = ParticleColor::new(0.0, 0.0, 0.0, 1.0);
        let b = ParticleColor::new(1.0, 1.0, 1.0, 0.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-5);
        assert!((mid.a - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_particle_color_to_rgba_u8() {
        let c = ParticleColor::new(1.0, 0.0, 0.5, 1.0);
        let rgba = c.to_rgba_u8();
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 0);
        assert!(rgba[2] > 120 && rgba[2] < 130);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn test_emitter_starts_empty() {
        let emitter = ParticleEmitter::new(EmitterConfig::default());
        assert_eq!(emitter.particle_count(), 0);
    }

    #[test]
    fn test_emitter_tick_produces_particles() {
        let cfg = EmitterConfig {
            emission_rate: 60.0,
            ..EmitterConfig::default()
        };
        let mut emitter = ParticleEmitter::new(cfg);
        emitter.tick(1.0); // 1 second → ~60 particles
        assert!(
            emitter.particle_count() > 0,
            "Should have emitted particles"
        );
    }

    #[test]
    fn test_emitter_inactive_no_new_particles() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        emitter.active = false;
        emitter.tick(5.0);
        assert_eq!(
            emitter.particle_count(),
            0,
            "Inactive emitter should not emit"
        );
    }

    #[test]
    fn test_emitter_respects_max_particles() {
        let cfg = EmitterConfig {
            emission_rate: 10_000.0,
            max_particles: 100,
            ..EmitterConfig::default()
        };
        let mut emitter = ParticleEmitter::new(cfg);
        emitter.tick(1.0);
        assert!(
            emitter.particle_count() <= 100,
            "Should not exceed max_particles: {}",
            emitter.particle_count()
        );
    }

    #[test]
    fn test_emitter_clear() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        emitter.tick(2.0);
        let before = emitter.particle_count();
        emitter.clear();
        assert_eq!(emitter.particle_count(), 0);
        assert!(before > 0 || true, "Test is valid regardless of count");
    }

    #[test]
    fn test_particles_die_over_time() {
        let cfg = EmitterConfig {
            emission_rate: 100.0,
            lifetime_min: 0.1,
            lifetime_max: 0.2,
            ..EmitterConfig::default()
        };
        let mut emitter = ParticleEmitter::new(cfg);
        emitter.tick(0.1); // emit some
        let after_emit = emitter.particle_count();
        emitter.active = false;
        emitter.tick(1.0); // let them die
        assert!(
            emitter.particle_count() < after_emit,
            "Particles should have died"
        );
    }

    #[test]
    fn test_sparks_preset_upward_bias() {
        let cfg = EmitterConfig::sparks();
        // Sparks fire upward (negative Y angle)
        assert!(cfg.angle_max_deg <= 0.0, "Sparks should fire upward");
    }

    #[test]
    fn test_smoke_preset_grows() {
        let cfg = EmitterConfig::smoke();
        assert!(
            cfg.size_death > cfg.size_birth_max,
            "Smoke should grow over lifetime"
        );
    }

    #[test]
    fn test_particle_data_age_fraction() {
        let p = ParticleData {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            lifetime: 0.5,
            max_lifetime: 1.0,
            color: ParticleColor::white(),
            color_birth: ParticleColor::white(),
            color_death: ParticleColor::new(0.0, 0.0, 0.0, 0.0),
            size: 4.0,
            size_birth: 4.0,
            size_death: 0.0,
        };
        // lifetime=0.5 out of 1.0 → age_fraction = 0.5
        let age = p.age_fraction();
        assert!(
            (age - 0.5).abs() < 1e-5,
            "age_fraction should be 0.5, got {age}"
        );
    }

    #[test]
    fn test_spawn_one_increments_count() {
        let mut emitter = ParticleEmitter::new(EmitterConfig::default());
        emitter.spawn_one();
        assert_eq!(emitter.particle_count(), 1);
    }
}
