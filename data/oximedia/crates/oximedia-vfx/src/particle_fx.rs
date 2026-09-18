//! Extended particle system for VFX with full lifecycle management.
//!
//! Provides [`ParticleFx`], [`ParticleEmitterFx`], and [`ParticleSystemFx`]
//! for producing spark, smoke, rain, and custom particle effects.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── Particle ──────────────────────────────────────────────────────────────────

/// A single particle with position, velocity, lifecycle, and visual properties.
#[derive(Debug, Clone)]
pub struct ParticleFx {
    /// Unique particle identifier.
    pub id: u64,
    /// (x, y) world-space position.
    pub position: (f32, f32),
    /// (vx, vy) velocity in pixels per millisecond.
    pub velocity: (f32, f32),
    /// (ax, ay) acceleration in pixels per ms².
    pub acceleration: (f32, f32),
    /// Total lifetime in milliseconds.
    pub lifetime_ms: u32,
    /// Current age in milliseconds.
    pub age_ms: u32,
    /// Visual size in pixels.
    pub size: f32,
    /// RGBA colour.
    pub color: [u8; 4],
    /// Whether this particle is still alive.
    pub active: bool,
}

impl ParticleFx {
    /// Returns `true` while `age_ms < lifetime_ms` and `active` is set.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.active && self.age_ms < self.lifetime_ms
    }

    /// Normalised age in [0.0, 1.0] (0 = just born, 1 = expired).
    #[must_use]
    pub fn normalized_age(&self) -> f32 {
        if self.lifetime_ms == 0 {
            return 1.0;
        }
        (self.age_ms as f32 / self.lifetime_ms as f32).clamp(0.0, 1.0)
    }
}

// ── ParticleEmitterFx ─────────────────────────────────────────────────────────

/// Emitter configuration that spawns [`ParticleFx`] particles.
#[derive(Debug, Clone)]
pub struct ParticleEmitterFx {
    /// World-space emission origin.
    pub position: (f32, f32),
    /// Particles emitted per second.
    pub emission_rate: f32,
    /// Initial speed magnitude (pixels / ms).
    pub initial_speed: f32,
    /// Half-angle of the emission cone in degrees.
    pub spread_degrees: f32,
    /// Lifetime assigned to each spawned particle (ms).
    pub lifetime_ms: u32,
}

impl ParticleEmitterFx {
    /// Preset for hot sparks (narrow cone, fast, short-lived).
    #[must_use]
    pub fn spark() -> Self {
        Self {
            position: (0.0, 0.0),
            emission_rate: 120.0,
            initial_speed: 0.4,
            spread_degrees: 30.0,
            lifetime_ms: 600,
        }
    }

    /// Preset for rising smoke (wide cone, slow, long-lived).
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            position: (0.0, 0.0),
            emission_rate: 20.0,
            initial_speed: 0.05,
            spread_degrees: 60.0,
            lifetime_ms: 3000,
        }
    }

    /// Preset for falling rain (vertical, medium speed).
    #[must_use]
    pub fn rain() -> Self {
        Self {
            position: (0.0, 0.0),
            emission_rate: 200.0,
            initial_speed: 0.3,
            spread_degrees: 5.0,
            lifetime_ms: 1200,
        }
    }
}

// ── ParticleSystemFx ──────────────────────────────────────────────────────────

/// Manages a collection of emitters and the particles they produce.
#[derive(Debug, Default)]
pub struct ParticleSystemFx {
    /// Registered emitters.
    pub emitters: Vec<ParticleEmitterFx>,
    /// Live and recently-dead particles.
    pub particles: Vec<ParticleFx>,
    next_id: u64,
}

impl ParticleSystemFx {
    /// Create an empty particle system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new emitter and return its index.
    pub fn add_emitter(&mut self, emitter: ParticleEmitterFx) -> usize {
        let idx = self.emitters.len();
        self.emitters.push(emitter);
        idx
    }

    /// Spawn `count` particles from emitter at `emitter_idx`.
    ///
    /// Does nothing if the index is out of range.
    pub fn emit(&mut self, emitter_idx: usize, count: usize) {
        let Some(emitter) = self.emitters.get(emitter_idx) else {
            return;
        };
        let (ex, ey) = emitter.position;
        let speed = emitter.initial_speed;
        let lifetime_ms = emitter.lifetime_ms;

        for k in 0..count {
            // Distribute evenly across the spread cone so tests are deterministic.
            let spread_rad = emitter.spread_degrees.to_radians();
            let angle_step = if count == 1 {
                0.0_f32
            } else {
                spread_rad / (count - 1) as f32
            };
            let base_angle: f32 = -spread_rad / 2.0;
            let angle = base_angle + angle_step * k as f32;
            let vx = speed * angle.sin();
            let vy = -speed * angle.cos(); // upward by default

            let id = self.next_id;
            self.next_id += 1;

            self.particles.push(ParticleFx {
                id,
                position: (ex, ey),
                velocity: (vx, vy),
                acceleration: (0.0, 0.001), // gravity
                lifetime_ms,
                age_ms: 0,
                size: 2.0,
                color: [255, 200, 80, 255],
                active: true,
            });
        }
    }

    /// Advance simulation by `dt_ms` milliseconds.
    ///
    /// Updates positions and ages all particles; deactivates those that have
    /// exceeded their lifetime.
    pub fn update(&mut self, dt_ms: u32) {
        let dt = dt_ms as f32;
        for p in &mut self.particles {
            if !p.active {
                continue;
            }
            // Symplectic Euler integration
            p.velocity.0 += p.acceleration.0 * dt;
            p.velocity.1 += p.acceleration.1 * dt;
            p.position.0 += p.velocity.0 * dt;
            p.position.1 += p.velocity.1 * dt;
            p.age_ms = p.age_ms.saturating_add(dt_ms);
            if p.age_ms >= p.lifetime_ms {
                p.active = false;
            }
        }
        // Prune dead particles to keep memory bounded.
        self.particles.retain(|p| p.active);
    }

    /// Count of currently alive particles.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.particles.iter().filter(|p| p.is_alive()).count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_emitter() -> ParticleEmitterFx {
        ParticleEmitterFx {
            position: (100.0, 200.0),
            emission_rate: 60.0,
            initial_speed: 0.1,
            spread_degrees: 10.0,
            lifetime_ms: 500,
        }
    }

    #[test]
    fn test_particle_is_alive_new() {
        let p = ParticleFx {
            id: 0,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 1000,
            age_ms: 0,
            size: 1.0,
            color: [255; 4],
            active: true,
        };
        assert!(p.is_alive());
    }

    #[test]
    fn test_particle_is_alive_expired() {
        let p = ParticleFx {
            id: 1,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 500,
            age_ms: 500,
            size: 1.0,
            color: [255; 4],
            active: true,
        };
        assert!(!p.is_alive());
    }

    #[test]
    fn test_particle_is_alive_inactive() {
        let p = ParticleFx {
            id: 2,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 1000,
            age_ms: 100,
            size: 1.0,
            color: [255; 4],
            active: false,
        };
        assert!(!p.is_alive());
    }

    #[test]
    fn test_normalized_age_zero() {
        let p = ParticleFx {
            id: 3,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 1000,
            age_ms: 0,
            size: 1.0,
            color: [255; 4],
            active: true,
        };
        assert_eq!(p.normalized_age(), 0.0);
    }

    #[test]
    fn test_normalized_age_half() {
        let p = ParticleFx {
            id: 4,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 1000,
            age_ms: 500,
            size: 1.0,
            color: [255; 4],
            active: true,
        };
        assert!((p.normalized_age() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_age_zero_lifetime() {
        let p = ParticleFx {
            id: 5,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
            lifetime_ms: 0,
            age_ms: 0,
            size: 1.0,
            color: [255; 4],
            active: true,
        };
        assert_eq!(p.normalized_age(), 1.0);
    }

    #[test]
    fn test_spark_preset_emission_rate() {
        let e = ParticleEmitterFx::spark();
        assert!(e.emission_rate > 60.0);
    }

    #[test]
    fn test_smoke_preset_lifetime() {
        let e = ParticleEmitterFx::smoke();
        assert!(e.lifetime_ms > 1000);
    }

    #[test]
    fn test_rain_preset_spread_narrow() {
        let e = ParticleEmitterFx::rain();
        assert!(e.spread_degrees < 15.0);
    }

    #[test]
    fn test_add_emitter_returns_index() {
        let mut sys = ParticleSystemFx::new();
        let idx0 = sys.add_emitter(default_emitter());
        let idx1 = sys.add_emitter(default_emitter());
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
    }

    #[test]
    fn test_emit_count() {
        let mut sys = ParticleSystemFx::new();
        sys.add_emitter(default_emitter());
        sys.emit(0, 5);
        assert_eq!(sys.alive_count(), 5);
    }

    #[test]
    fn test_emit_invalid_emitter_noop() {
        let mut sys = ParticleSystemFx::new();
        sys.emit(99, 10); // no emitters registered
        assert_eq!(sys.alive_count(), 0);
    }

    #[test]
    fn test_update_ages_particles() {
        let mut sys = ParticleSystemFx::new();
        sys.add_emitter(default_emitter());
        sys.emit(0, 3);
        sys.update(100);
        // All particles should still be alive (lifetime 500 ms, only 100 ms passed)
        assert_eq!(sys.alive_count(), 3);
    }

    #[test]
    fn test_update_removes_expired_particles() {
        let mut sys = ParticleSystemFx::new();
        sys.add_emitter(ParticleEmitterFx {
            lifetime_ms: 50,
            ..default_emitter()
        });
        sys.emit(0, 4);
        sys.update(100); // advance past lifetime
        assert_eq!(sys.alive_count(), 0);
    }

    #[test]
    fn test_particle_ids_unique() {
        let mut sys = ParticleSystemFx::new();
        sys.add_emitter(default_emitter());
        sys.emit(0, 5);
        let ids: Vec<u64> = sys.particles.iter().map(|p| p.id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique.len());
    }
}
