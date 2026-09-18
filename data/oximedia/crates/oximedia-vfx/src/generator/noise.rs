//! Noise generator.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// Noise type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// White noise (uniform random).
    White,
    /// Pink noise (1/f).
    Pink,
    /// Perlin noise (smooth).
    Perlin,
    /// Simplex noise.
    Simplex,
}

/// Noise generator.
pub struct Noise {
    noise_type: NoiseType,
    amplitude: f32,
    seed: u64,
    scale: f32,
    rng: rand::rngs::StdRng,
}

impl Noise {
    /// Create a new noise generator.
    #[must_use]
    pub fn new(noise_type: NoiseType) -> Self {
        Self {
            noise_type,
            amplitude: 1.0,
            seed: 0,
            scale: 1.0,
            rng: rand::rngs::StdRng::seed_from_u64(0),
        }
    }

    /// Set noise amplitude (0.0 - 1.0).
    #[must_use]
    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude.clamp(0.0, 1.0);
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = rand::rngs::StdRng::seed_from_u64(seed);
        self
    }

    /// Set noise scale.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale.max(0.01);
        self
    }

    fn perlin_fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn perlin_grad(hash: i32, x: f32, y: f32) -> f32 {
        let h = hash & 15;
        let u = if h < 8 { x } else { y };
        let v = if h < 4 {
            y
        } else if h == 12 || h == 14 {
            x
        } else {
            0.0
        };

        (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
    }

    fn perlin_noise(&mut self, x: f32, y: f32) -> f32 {
        let xi = (x.floor() as i32) & 255;
        let yi = (y.floor() as i32) & 255;

        let xf = x - x.floor();
        let yf = y - y.floor();

        let u = Self::perlin_fade(xf);
        let v = Self::perlin_fade(yf);

        // Simple permutation table using seed
        let p =
            |i: i32| (((i as u32).wrapping_mul(2654435761_u32) ^ self.seed as u32) & 255) as i32;

        let aa = p(p(xi) + yi);
        let ab = p(p(xi) + yi + 1);
        let ba = p(p(xi + 1) + yi);
        let bb = p(p(xi + 1) + yi + 1);

        let x1 = Self::perlin_grad(aa, xf, yf);
        let x2 = Self::perlin_grad(ba, xf - 1.0, yf);
        let y1 = x1 + u * (x2 - x1);

        let x1 = Self::perlin_grad(ab, xf, yf - 1.0);
        let x2 = Self::perlin_grad(bb, xf - 1.0, yf - 1.0);
        let y2 = x1 + u * (x2 - x1);

        (y1 + v * (y2 - y1) + 1.0) * 0.5
    }
}

impl VideoEffect for Noise {
    fn name(&self) -> &'static str {
        "Noise"
    }

    fn description(&self) -> &'static str {
        "Generate various types of noise"
    }

    fn apply(
        &mut self,
        _input: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        // Re-seed for consistency if needed
        if params.time == 0.0 {
            self.rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        }

        for y in 0..output.height {
            for x in 0..output.width {
                let value = match self.noise_type {
                    NoiseType::White => self.rng.random_range(0.0..1.0),
                    NoiseType::Pink => {
                        // Approximate pink noise with multiple octaves
                        let mut sum = 0.0_f32;
                        let mut amp = 1.0_f32;
                        for _ in 0..4 {
                            sum += self.rng.random_range(0.0..1.0) * amp;
                            amp *= 0.5;
                        }
                        (sum / 2.0).clamp(0.0, 1.0)
                    }
                    NoiseType::Perlin => {
                        let nx = x as f32 * self.scale / 100.0;
                        let ny = y as f32 * self.scale / 100.0;
                        self.perlin_noise(nx, ny)
                    }
                    NoiseType::Simplex => {
                        // Simplified simplex noise (similar to Perlin for now)
                        let nx = x as f32 * self.scale / 100.0;
                        let ny = y as f32 * self.scale / 100.0;
                        self.perlin_noise(nx * 0.866, ny * 0.866)
                    }
                };

                let scaled = (value * self.amplitude * 255.0) as u8;
                output.set_pixel(x, y, [scaled, scaled, scaled, 255]);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_types() {
        let types = [NoiseType::White, NoiseType::Pink, NoiseType::Perlin];

        for noise_type in types {
            let mut noise = Noise::new(noise_type);
            let input = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");
            let params = EffectParams::new();
            noise
                .apply(&input, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_noise_customization() {
        let noise = Noise::new(NoiseType::Perlin)
            .with_amplitude(0.5)
            .with_scale(2.0)
            .with_seed(42);

        assert_eq!(noise.amplitude, 0.5);
        assert_eq!(noise.scale, 2.0);
        assert_eq!(noise.seed, 42);
    }
}
