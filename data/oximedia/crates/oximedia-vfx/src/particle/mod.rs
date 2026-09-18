//! Particle system effects.

pub mod dust;
pub mod emitter;
pub mod rain;
pub mod snow;
pub mod sparks;
/// Uniform spatial hash grid for O(1) amortized particle-neighbourhood queries.
pub mod spatial_grid;
pub mod system;

pub use dust::{Dust, DustMode};
pub use emitter::{EmitterConfig, ParticleColor, ParticleData, ParticleEmitter};
pub use rain::{Rain, RainIntensity};
pub use snow::{Snow, SnowStyle};
pub use sparks::{SparkType, Sparks};
pub use system::{
    Color as ParticleColorF32, EmitterConfig as ParticleEmitterConfig, EmitterShape,
    Particle as ParticleAdvanced, ParticleEmitter as ParticleEmitterAdvanced, Vec2,
};

use rand::SeedableRng;

/// Generic particle.
#[derive(Debug, Clone)]
pub struct Particle {
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// X velocity.
    pub vx: f32,
    /// Y velocity.
    pub vy: f32,
    /// Particle size.
    pub size: f32,
    /// Current life remaining.
    pub life: f32,
    /// Maximum life span.
    pub max_life: f32,
}

impl Particle {
    /// Create a new particle.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            size: 1.0,
            life: 1.0,
            max_life: 1.0,
        }
    }

    /// Update particle physics.
    pub fn update(&mut self, dt: f32, gravity: f32) {
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.vy += gravity * dt;
        self.life -= dt;
    }

    /// Check if particle is alive.
    #[must_use]
    pub const fn is_alive(&self) -> bool {
        self.life > 0.0
    }

    /// Get particle opacity based on life.
    #[must_use]
    pub fn get_opacity(&self) -> f32 {
        (self.life / self.max_life).clamp(0.0, 1.0)
    }
}

/// Particle system base.
pub struct ParticleSystem {
    particles: Vec<Particle>,
    rng: rand::rngs::StdRng,
    spawn_rate: f32,
    accumulator: f32,
}

impl ParticleSystem {
    /// Create a new particle system.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            particles: Vec::new(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            spawn_rate: 10.0,
            accumulator: 0.0,
        }
    }

    /// Set spawn rate (particles per second).
    pub fn set_spawn_rate(&mut self, rate: f32) {
        self.spawn_rate = rate.max(0.0);
    }

    /// Spawn a particle.
    pub fn spawn(&mut self, particle: Particle) {
        self.particles.push(particle);
    }

    /// Update all particles.
    pub fn update(&mut self, dt: f32, gravity: f32) {
        // Update existing particles
        for particle in &mut self.particles {
            particle.update(dt, gravity);
        }

        // Remove dead particles
        self.particles.retain(Particle::is_alive);
    }

    /// Get all particles.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Get RNG.
    pub fn rng(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.particles.clear();
    }
}
