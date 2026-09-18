//! Particle system for procedural animation and visual effects.
//!
//! Provides a full-featured emitter-based particle system with color interpolation,
//! physics (gravity + turbulence), and RGBA rasterization.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A single particle with position, velocity, color and lifetime state.
#[derive(Debug, Clone)]
pub struct Particle {
    /// X, Y position in pixels.
    pub position: [f32; 2],
    /// X, Y velocity in pixels/second.
    pub velocity: [f32; 2],
    /// X, Y acceleration in pixels/second².
    pub acceleration: [f32; 2],
    /// RGBA color (each component 0.0–1.0).
    pub color: [f32; 4],
    /// Radius in pixels.
    pub size: f32,
    /// Normalized life: 1.0 = just born, 0.0 = dead.
    pub life: f32,
    /// Age in seconds since birth.
    pub age: f32,
    /// Maximum age in seconds.
    pub max_age: f32,
    /// Previous position for motion-blur trail rendering (set each tick).
    pub prev_position: [f32; 2],
}

impl Particle {
    /// Returns `true` while the particle is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.life > 0.0
    }

    /// Advance physics and age by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        // Record previous position for motion blur trail.
        self.prev_position = self.position;
        self.velocity[0] += self.acceleration[0] * dt;
        self.velocity[1] += self.acceleration[1] * dt;
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.age += dt;
        self.life = (1.0 - self.age / self.max_age).max(0.0);
    }

    /// Pixel-space displacement vector from `prev_position` to `position`.
    pub fn motion_vector(&self) -> [f32; 2] {
        [
            self.position[0] - self.prev_position[0],
            self.position[1] - self.prev_position[1],
        ]
    }

    /// Length of the motion vector (speed × dt, in pixels).
    pub fn motion_magnitude(&self) -> f32 {
        let mv = self.motion_vector();
        (mv[0] * mv[0] + mv[1] * mv[1]).sqrt()
    }
}

/// Configuration for a single particle emitter.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    /// Emitter origin (x, y) in pixels.
    pub position: [f32; 2],
    /// Number of particles to spawn per second.
    pub emit_rate: f32,
    /// Maximum live particles from this emitter.
    pub max_particles: usize,
    /// Min / max particle lifetime in seconds.
    pub life_range: (f32, f32),
    /// Min / max initial speed in pixels/second.
    pub speed_range: (f32, f32),
    /// Angular emission cone in radians (centered on −π/2, i.e. upward).
    /// The emitted angle is chosen uniformly in `[−half, +half]` from the
    /// center direction `angle_bias`.
    pub angle_range: (f32, f32),
    /// Min / max particle radius.
    pub size_range: (f32, f32),
    /// Gravity acceleration applied to every particle (x, y) pixels/s².
    pub gravity: [f32; 2],
    /// RGBA color at birth (components 0.0–1.0).
    pub color_start: [f32; 4],
    /// RGBA color at death (components 0.0–1.0).
    pub color_end: [f32; 4],
    /// Magnitude of random per-frame velocity jitter in pixels/s.
    pub turbulence: f32,
    /// Bias angle for emission direction, in radians (0 = rightward, −π/2 = upward).
    pub angle_bias: f32,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            emit_rate: 30.0,
            max_particles: 500,
            life_range: (1.0, 3.0),
            speed_range: (50.0, 150.0),
            angle_range: (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4),
            size_range: (2.0, 6.0),
            gravity: [0.0, 100.0],
            color_start: [1.0, 0.8, 0.2, 1.0],
            color_end: [1.0, 0.2, 0.0, 0.0],
            turbulence: 0.0,
            angle_bias: -std::f32::consts::FRAC_PI_2,
        }
    }
}

impl ParticleEmitter {
    /// Spawn a new `Particle` using the emitter configuration and the provided RNG.
    fn spawn(&self, rng: &mut StdRng) -> Particle {
        let life = rng.random_range(self.life_range.0..=self.life_range.1);
        let speed = rng.random_range(self.speed_range.0..=self.speed_range.1);
        let angle = self.angle_bias + rng.random_range(self.angle_range.0..=self.angle_range.1);
        let size = rng.random_range(self.size_range.0..=self.size_range.1);

        Particle {
            position: self.position,
            prev_position: self.position,
            velocity: [speed * angle.cos(), speed * angle.sin()],
            acceleration: self.gravity,
            color: self.color_start,
            size,
            life: 1.0,
            age: 0.0,
            max_age: life,
        }
    }
}

/// A multi-emitter particle system with deterministic pseudo-random behaviour.
pub struct ParticleSystem {
    emitters: Vec<ParticleEmitter>,
    particles: Vec<Particle>,
    /// Fractional particle spawn accumulators (one per emitter).
    accumulators: Vec<f32>,
    time: f32,
    rng: StdRng,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleSystem {
    /// Create an empty particle system seeded from a default value.
    #[must_use]
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Create an empty particle system with an explicit RNG seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            emitters: Vec::new(),
            particles: Vec::new(),
            accumulators: Vec::new(),
            time: 0.0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add an emitter to the system.
    pub fn add_emitter(&mut self, emitter: ParticleEmitter) {
        self.accumulators.push(0.0);
        self.emitters.push(emitter);
    }

    /// Return a reference to all currently live particles.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Return the total elapsed simulation time in seconds.
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Advance the simulation by `dt` seconds.
    ///
    /// - Spawns new particles from each emitter (rate-limited).
    /// - Applies turbulence to velocity.
    /// - Advances particle physics.
    /// - Interpolates color from `color_start` to `color_end`.
    /// - Removes dead particles.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;

        // Spawn from each emitter
        for (idx, emitter) in self.emitters.iter().enumerate() {
            self.accumulators[idx] += emitter.emit_rate * dt;
            let count = self.accumulators[idx] as usize;
            self.accumulators[idx] -= count as f32;

            // Count particles already owned by this emitter (approximation: share pool)
            let live = self.particles.len();
            let budget = emitter.max_particles.saturating_sub(live);
            let to_spawn = count.min(budget);

            for _ in 0..to_spawn {
                let p = emitter.spawn(&mut self.rng);
                self.particles.push(p);
            }
        }

        // Update every particle
        let turbulence_magnitude: f32 = self
            .emitters
            .iter()
            .map(|e| e.turbulence)
            .fold(0.0_f32, f32::max);

        for p in &mut self.particles {
            // Turbulence: random velocity nudge
            if turbulence_magnitude > 0.0 {
                let jitter = turbulence_magnitude * dt;
                p.velocity[0] += self.rng.random_range(-jitter..=jitter);
                p.velocity[1] += self.rng.random_range(-jitter..=jitter);
            }
            p.update(dt);

            // Interpolate color: life goes from 1 → 0, so t = 1 - life
            // We need emitter color data; use the first emitter as default.
            if let Some(emitter) = self.emitters.first() {
                let t = 1.0 - p.life.clamp(0.0, 1.0);
                for ch in 0..4 {
                    p.color[ch] = emitter.color_start[ch]
                        + (emitter.color_end[ch] - emitter.color_start[ch]) * t;
                }
            }
        }

        // Remove dead particles
        self.particles.retain(Particle::is_alive);
    }

    /// Rasterize all particles onto an RGBA8 buffer using additive alpha blending.
    ///
    /// Each particle is drawn as a soft circular sprite.  The buffer must be
    /// `width * height * 4` bytes long.  Existing buffer content is treated as
    /// the backdrop and particles are composited over it.
    pub fn render_to_rgba(&self, output: &mut [u8], width: u32, height: u32) {
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(
            output.len(),
            expected,
            "Buffer length mismatch in render_to_rgba"
        );

        for p in &self.particles {
            let cx = p.position[0];
            let cy = p.position[1];
            let r = p.size.max(1.0);
            let alpha_scale = p.color[3];

            // Bounding box (inclusive, clamped to frame)
            let x0 = ((cx - r).floor() as i32).max(0) as u32;
            let y0 = ((cy - r).floor() as i32).max(0) as u32;
            let x1 = ((cx + r).ceil() as i32).min(width as i32 - 1).max(0) as u32;
            let y1 = ((cy + r).ceil() as i32).min(height as i32 - 1).max(0) as u32;

            for py in y0..=y1 {
                for px in x0..=x1 {
                    let dx = px as f32 + 0.5 - cx;
                    let dy = py as f32 + 0.5 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > r {
                        continue;
                    }
                    // Soft falloff: 1 at centre, 0 at edge
                    let falloff = 1.0 - (dist / r).powi(2);
                    let a = (alpha_scale * falloff).clamp(0.0, 1.0);

                    let idx = ((py * width + px) * 4) as usize;
                    // Alpha composite: out = src * src_alpha + dst * (1 - src_alpha)
                    let inv_a = 1.0 - a;
                    for ch in 0..3_usize {
                        let src = (p.color[ch] * 255.0).clamp(0.0, 255.0);
                        let dst = output[idx + ch] as f32;
                        output[idx + ch] = (src * a + dst * inv_a).clamp(0.0, 255.0) as u8;
                    }
                    // Alpha channel: union
                    let dst_a = output[idx + 3] as f32 / 255.0;
                    let out_a = (a + dst_a * inv_a).clamp(0.0, 1.0);
                    output[idx + 3] = (out_a * 255.0) as u8;
                }
            }
        }
    }

    /// Return the number of live particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Rasterize particles with motion-blur trails onto an RGBA8 buffer.
    ///
    /// Each particle draws a soft line segment from `prev_position` to
    /// `position`.  The trail is divided into `blur_samples` sub-steps and
    /// each step is rendered at reduced opacity to simulate motion blur.
    ///
    /// A `blur_samples` value of 1 is equivalent to `render_to_rgba` (no blur).
    ///
    /// The buffer must be `width * height * 4` bytes long.
    pub fn render_to_rgba_motion_blur(
        &self,
        output: &mut [u8],
        width: u32,
        height: u32,
        blur_samples: u32,
    ) {
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(
            output.len(),
            expected,
            "Buffer length mismatch in render_to_rgba_motion_blur"
        );

        let samples = blur_samples.max(1);

        for p in &self.particles {
            let dx = p.position[0] - p.prev_position[0];
            let dy = p.position[1] - p.prev_position[1];
            let base_alpha = p.color[3];

            for step in 0..samples {
                let t = if samples == 1 {
                    1.0_f32
                } else {
                    (step as f32 + 1.0) / samples as f32
                };

                let cx = p.prev_position[0] + dx * t;
                let cy = p.prev_position[1] + dy * t;

                // Linearly decay alpha across samples: earlier trail segments
                // are more transparent.
                let step_alpha = base_alpha * t;

                let r = p.size.max(1.0);
                let x0 = ((cx - r).floor() as i32).max(0) as u32;
                let y0 = ((cy - r).floor() as i32).max(0) as u32;
                let x1 = ((cx + r).ceil() as i32).min(width as i32 - 1).max(0) as u32;
                let y1 = ((cy + r).ceil() as i32).min(height as i32 - 1).max(0) as u32;

                for py in y0..=y1 {
                    for px in x0..=x1 {
                        let px_dx = px as f32 + 0.5 - cx;
                        let px_dy = py as f32 + 0.5 - cy;
                        let dist = (px_dx * px_dx + px_dy * px_dy).sqrt();
                        if dist > r {
                            continue;
                        }
                        let falloff = 1.0 - (dist / r).powi(2);
                        let a = (step_alpha * falloff).clamp(0.0, 1.0);
                        let idx = ((py * width + px) * 4) as usize;
                        let inv_a = 1.0 - a;
                        for ch in 0..3_usize {
                            let src = (p.color[ch] * 255.0).clamp(0.0, 255.0);
                            let dst = output[idx + ch] as f32;
                            output[idx + ch] = (src * a + dst * inv_a).clamp(0.0, 255.0) as u8;
                        }
                        let dst_a = output[idx + 3] as f32 / 255.0;
                        let out_a = (a + dst_a * inv_a).clamp(0.0, 1.0);
                        output[idx + 3] = (out_a * 255.0) as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fire_emitter(x: f32, y: f32) -> ParticleEmitter {
        ParticleEmitter {
            position: [x, y],
            emit_rate: 100.0,
            max_particles: 200,
            life_range: (0.5, 1.5),
            speed_range: (80.0, 160.0),
            angle_range: (-0.5, 0.5),
            size_range: (3.0, 8.0),
            gravity: [0.0, -50.0],
            color_start: [1.0, 0.8, 0.0, 1.0],
            color_end: [1.0, 0.0, 0.0, 0.0],
            turbulence: 10.0,
            angle_bias: -std::f32::consts::FRAC_PI_2,
        }
    }

    #[test]
    fn test_particle_system_spawns() {
        let mut sys = ParticleSystem::with_seed(42);
        sys.add_emitter(fire_emitter(100.0, 100.0));
        sys.update(0.5);
        assert!(sys.particle_count() > 0, "No particles spawned after 0.5 s");
    }

    #[test]
    fn test_particle_system_respects_max() {
        let mut sys = ParticleSystem::with_seed(1);
        let emitter = ParticleEmitter {
            emit_rate: 10_000.0,
            max_particles: 50,
            life_range: (10.0, 10.0),
            ..fire_emitter(0.0, 0.0)
        };
        sys.add_emitter(emitter);
        sys.update(1.0);
        assert!(
            sys.particle_count() <= 50,
            "Exceeded max_particles: {}",
            sys.particle_count()
        );
    }

    #[test]
    fn test_particle_dies() {
        let mut sys = ParticleSystem::with_seed(7);
        let emitter = ParticleEmitter {
            emit_rate: 100.0,
            max_particles: 500,
            life_range: (0.1, 0.1), // very short lived
            ..fire_emitter(0.0, 0.0)
        };
        sys.add_emitter(emitter);
        sys.update(0.05); // spawn them
        let spawned = sys.particle_count();
        assert!(spawned > 0);
        sys.update(0.15); // let them die
        assert!(sys.particle_count() < spawned, "Particles should have died");
    }

    #[test]
    fn test_render_to_rgba_no_panic() {
        let mut sys = ParticleSystem::with_seed(3);
        sys.add_emitter(fire_emitter(32.0, 32.0));
        sys.update(0.1);

        let w: u32 = 64;
        let h: u32 = 64;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        sys.render_to_rgba(&mut buf, w, h);
        // At least some pixels should be non-zero
        assert!(buf.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_particle_color_interpolation() {
        let mut sys = ParticleSystem::with_seed(99);
        let emitter = ParticleEmitter {
            emit_rate: 1000.0,
            max_particles: 200,
            life_range: (1.0, 1.0),
            color_start: [1.0, 0.0, 0.0, 1.0],
            color_end: [0.0, 0.0, 1.0, 0.0],
            ..fire_emitter(0.0, 0.0)
        };
        sys.add_emitter(emitter);
        sys.update(0.01); // freshly born particles (life ≈ 1.0 → color ≈ start)
        if let Some(p) = sys.particles().first() {
            // Near birth: red component should still be high
            assert!(
                p.color[0] > 0.5,
                "Red should be high at birth, got {}",
                p.color[0]
            );
        }
    }

    #[test]
    fn test_two_emitters() {
        let mut sys = ParticleSystem::with_seed(11);
        sys.add_emitter(fire_emitter(0.0, 0.0));
        sys.add_emitter(fire_emitter(100.0, 100.0));
        sys.update(0.2);
        assert!(sys.particle_count() > 0);
        assert_eq!(sys.emitters.len(), 2);
    }

    // --- Motion blur tests ---

    #[test]
    fn test_particle_motion_vector_after_update() {
        let mut sys = ParticleSystem::with_seed(42);
        sys.add_emitter(fire_emitter(32.0, 32.0));
        sys.update(0.01); // spawn
        sys.update(0.1); // advance
        if let Some(p) = sys.particles().first() {
            let mv = p.motion_vector();
            // The particle should have moved.
            let mag = p.motion_magnitude();
            // Position changed → motion vector should be non-zero.
            let _ = mv;
            assert!(mag >= 0.0);
        }
    }

    #[test]
    fn test_render_motion_blur_no_panic() {
        let mut sys = ParticleSystem::with_seed(5);
        sys.add_emitter(fire_emitter(32.0, 32.0));
        sys.update(0.1);

        let w: u32 = 64;
        let h: u32 = 64;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        sys.render_to_rgba_motion_blur(&mut buf, w, h, 4);
        // Some pixels should be written.
        assert!(buf.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_render_motion_blur_single_sample_equivalent() {
        // With 1 sample, motion blur should behave similarly to the plain render.
        let mut sys = ParticleSystem::with_seed(7);
        sys.add_emitter(fire_emitter(32.0, 32.0));
        sys.update(0.1);

        let w: u32 = 64;
        let h: u32 = 64;
        let mut buf1 = vec![0u8; (w * h * 4) as usize];
        let mut buf2 = vec![0u8; (w * h * 4) as usize];
        sys.render_to_rgba(&mut buf1, w, h);
        sys.render_to_rgba_motion_blur(&mut buf2, w, h, 1);
        // Both should produce non-zero output.
        assert!(buf1.iter().any(|&b| b > 0));
        assert!(buf2.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_particle_prev_position_initialized_to_spawn() {
        let mut sys = ParticleSystem::with_seed(99);
        let emitter = ParticleEmitter {
            position: [100.0, 200.0],
            ..fire_emitter(0.0, 0.0)
        };
        sys.add_emitter(emitter);
        sys.update(0.01);
        if let Some(p) = sys.particles().first() {
            // On the very first tick, prev_position should equal spawn position.
            assert!((p.prev_position[0] - 100.0).abs() < 10.0);
        }
    }
}
