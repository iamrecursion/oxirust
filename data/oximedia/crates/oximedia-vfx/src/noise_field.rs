//! Procedural noise field generation for VFX applications.
//!
//! Provides value-noise, gradient-noise, and turbulence generators that can be
//! sampled in 2-D to drive displacement maps, particle forces, or texture
//! synthesis.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Type of procedural noise algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Simple value noise (random grid + bilinear interpolation).
    Value,
    /// Gradient noise (Perlin-style dot-product interpolation).
    Gradient,
    /// Sum-of-octaves turbulence built on top of gradient noise.
    Turbulence,
    /// Voronoi / Worley cell noise based on closest-point distance.
    Worley,
}

impl NoiseType {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Value => "Value",
            Self::Gradient => "Gradient",
            Self::Turbulence => "Turbulence",
            Self::Worley => "Worley",
        }
    }
}

/// A single sample result from the noise field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSample {
    /// Primary noise value in [-1, 1] (or [0, 1] depending on type).
    pub value: f64,
    /// Approximate gradient in X direction.
    pub grad_x: f64,
    /// Approximate gradient in Y direction.
    pub grad_y: f64,
}

impl NoiseSample {
    /// Create a new sample.
    #[must_use]
    pub fn new(value: f64, grad_x: f64, grad_y: f64) -> Self {
        Self {
            value,
            grad_x,
            grad_y,
        }
    }

    /// Magnitude of the gradient vector.
    #[must_use]
    pub fn gradient_magnitude(&self) -> f64 {
        (self.grad_x * self.grad_x + self.grad_y * self.grad_y).sqrt()
    }
}

/// A configurable 2-D noise field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseField {
    /// Noise algorithm to use.
    pub noise_type: NoiseType,
    /// Base frequency (cells per unit length).
    pub frequency: f64,
    /// Number of octaves for turbulence.
    pub octaves: u32,
    /// Amplitude multiplier between successive octaves.
    pub persistence: f64,
    /// Frequency multiplier between successive octaves (lacunarity).
    pub lacunarity: f64,
    /// Seed used to offset the noise field.
    pub seed: u64,
    /// Global amplitude scaling.
    pub amplitude: f64,
}

impl Default for NoiseField {
    fn default() -> Self {
        Self {
            noise_type: NoiseType::Gradient,
            frequency: 4.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            seed: 0,
            amplitude: 1.0,
        }
    }
}

impl NoiseField {
    /// Create a noise field with default settings.
    #[must_use]
    pub fn new(noise_type: NoiseType) -> Self {
        Self {
            noise_type,
            ..Self::default()
        }
    }

    /// Set frequency.
    #[must_use]
    pub fn with_frequency(mut self, f: f64) -> Self {
        self.frequency = f.max(0.001);
        self
    }

    /// Set number of octaves.
    #[must_use]
    pub fn with_octaves(mut self, n: u32) -> Self {
        self.octaves = n.max(1).min(16);
        self
    }

    /// Set persistence.
    #[must_use]
    pub fn with_persistence(mut self, p: f64) -> Self {
        self.persistence = p.clamp(0.0, 1.0);
        self
    }

    /// Set seed.
    #[must_use]
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Internal hash for integer grid coordinates.
    fn hash2d(&self, ix: i64, iy: i64) -> f64 {
        // Simple integer hash (good enough for procedural use).
        let mut h = (ix.wrapping_mul(374_761_393))
            .wrapping_add(iy.wrapping_mul(668_265_263))
            .wrapping_add(self.seed as i64);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        h = h ^ (h >> 16);
        // Map to [-1, 1].
        #[allow(clippy::cast_precision_loss)]
        let v = (h & 0x7FFF_FFFF) as f64 / 0x7FFF_FFFF_u32 as f64;
        v * 2.0 - 1.0
    }

    /// Smooth interpolation weight (quintic).
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Sample value noise at `(x, y)`.
    fn sample_value(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let u = Self::fade(fx);
        let v = Self::fade(fy);

        let c00 = self.hash2d(ix, iy);
        let c10 = self.hash2d(ix + 1, iy);
        let c01 = self.hash2d(ix, iy + 1);
        let c11 = self.hash2d(ix + 1, iy + 1);

        let a = c00 + (c10 - c00) * u;
        let b = c01 + (c11 - c01) * u;
        a + (b - a) * v
    }

    /// Sample gradient noise at `(x, y)`.
    fn sample_gradient(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let u = Self::fade(fx);
        let v = Self::fade(fy);

        // Pseudo-gradient via hash differences.
        let dot = |gx: i64, gy: i64, dx: f64, dy: f64| -> f64 {
            let angle = self.hash2d(gx, gy) * std::f64::consts::PI;
            let gx_val = angle.cos();
            let gy_val = angle.sin();
            gx_val * dx + gy_val * dy
        };

        let n00 = dot(ix, iy, fx, fy);
        let n10 = dot(ix + 1, iy, fx - 1.0, fy);
        let n01 = dot(ix, iy + 1, fx, fy - 1.0);
        let n11 = dot(ix + 1, iy + 1, fx - 1.0, fy - 1.0);

        let a = n00 + (n10 - n00) * u;
        let b = n01 + (n11 - n01) * u;
        a + (b - a) * v
    }

    /// Sample Worley / cellular noise at `(x, y)`.
    fn sample_worley(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();

        let mut min_dist = f64::MAX;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = ix + dx;
                let ny = iy + dy;
                // Feature point inside cell.
                let px = (self.hash2d(nx, ny) + 1.0) * 0.5 + dx as f64;
                let py = (self.hash2d(ny, nx) + 1.0) * 0.5 + dy as f64;
                let ddx = px - fx;
                let ddy = py - fy;
                let d = ddx * ddx + ddy * ddy;
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        min_dist.sqrt().min(1.0)
    }
}

/// A generator that wraps a [`NoiseField`] and provides convenience sampling.
#[derive(Debug, Clone)]
pub struct NoiseFieldGenerator {
    /// The underlying noise field configuration.
    pub field: NoiseField,
}

impl NoiseFieldGenerator {
    /// Create a generator from a noise field.
    #[must_use]
    pub fn new(field: NoiseField) -> Self {
        Self { field }
    }

    /// Sample the noise field at `(x, y)` and return a [`NoiseSample`].
    #[must_use]
    pub fn sample(&self, x: f64, y: f64) -> NoiseSample {
        let sx = x * self.field.frequency;
        let sy = y * self.field.frequency;

        let value = match self.field.noise_type {
            NoiseType::Value => self.field.sample_value(sx, sy) * self.field.amplitude,
            NoiseType::Gradient => self.field.sample_gradient(sx, sy) * self.field.amplitude,
            NoiseType::Turbulence => {
                let mut total = 0.0;
                let mut amp = self.field.amplitude;
                let mut freq = 1.0;
                for _ in 0..self.field.octaves {
                    total += self.field.sample_gradient(sx * freq, sy * freq).abs() * amp;
                    amp *= self.field.persistence;
                    freq *= self.field.lacunarity;
                }
                total
            }
            NoiseType::Worley => self.field.sample_worley(sx, sy) * self.field.amplitude,
        };

        // Numerical gradient via central differences.
        let eps = 0.001;
        let vx_plus = self.sample_raw(x + eps, y);
        let vx_minus = self.sample_raw(x - eps, y);
        let vy_plus = self.sample_raw(x, y + eps);
        let vy_minus = self.sample_raw(x, y - eps);

        NoiseSample::new(
            value,
            (vx_plus - vx_minus) / (2.0 * eps),
            (vy_plus - vy_minus) / (2.0 * eps),
        )
    }

    /// Raw scalar sample (no gradient computation).
    #[must_use]
    pub fn sample_raw(&self, x: f64, y: f64) -> f64 {
        let sx = x * self.field.frequency;
        let sy = y * self.field.frequency;
        match self.field.noise_type {
            NoiseType::Value => self.field.sample_value(sx, sy) * self.field.amplitude,
            NoiseType::Gradient => self.field.sample_gradient(sx, sy) * self.field.amplitude,
            NoiseType::Turbulence => {
                let mut total = 0.0;
                let mut amp = self.field.amplitude;
                let mut freq = 1.0;
                for _ in 0..self.field.octaves {
                    total += self.field.sample_gradient(sx * freq, sy * freq).abs() * amp;
                    amp *= self.field.persistence;
                    freq *= self.field.lacunarity;
                }
                total
            }
            NoiseType::Worley => self.field.sample_worley(sx, sy) * self.field.amplitude,
        }
    }

    /// Generate a 2-D grid of raw noise values.
    ///
    /// Returns a row-major `Vec<f64>` with `width * height` entries.
    #[must_use]
    pub fn generate_grid(&self, width: usize, height: usize) -> Vec<f64> {
        let mut buf = Vec::with_capacity(width * height);
        for row in 0..height {
            #[allow(clippy::cast_precision_loss)]
            let y = row as f64 / height.max(1) as f64;
            for col in 0..width {
                #[allow(clippy::cast_precision_loss)]
                let x = col as f64 / width.max(1) as f64;
                buf.push(self.sample_raw(x, y));
            }
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_type_labels() {
        assert_eq!(NoiseType::Value.label(), "Value");
        assert_eq!(NoiseType::Gradient.label(), "Gradient");
        assert_eq!(NoiseType::Turbulence.label(), "Turbulence");
        assert_eq!(NoiseType::Worley.label(), "Worley");
    }

    #[test]
    fn test_noise_sample_gradient_magnitude() {
        let s = NoiseSample::new(0.5, 3.0, 4.0);
        assert!((s.gradient_magnitude() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_noise_field_default() {
        let nf = NoiseField::default();
        assert_eq!(nf.noise_type, NoiseType::Gradient);
        assert!(nf.frequency > 0.0);
        assert_eq!(nf.octaves, 4);
    }

    #[test]
    fn test_noise_field_builder() {
        let nf = NoiseField::new(NoiseType::Value)
            .with_frequency(8.0)
            .with_octaves(6)
            .with_persistence(0.3)
            .with_seed(42);
        assert_eq!(nf.noise_type, NoiseType::Value);
        assert!((nf.frequency - 8.0).abs() < 1e-12);
        assert_eq!(nf.octaves, 6);
        assert!((nf.persistence - 0.3).abs() < 1e-12);
        assert_eq!(nf.seed, 42);
    }

    #[test]
    fn test_value_noise_bounded() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Value));
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = gen.sample_raw(i as f64 * 0.07, i as f64 * 0.03);
            assert!(
                v >= -1.5 && v <= 1.5,
                "Value noise out of expected range: {v}"
            );
        }
    }

    #[test]
    fn test_gradient_noise_bounded() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Gradient));
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = gen.sample_raw(i as f64 * 0.05, i as f64 * 0.11);
            assert!(
                v >= -2.0 && v <= 2.0,
                "Gradient noise out of expected range: {v}"
            );
        }
    }

    #[test]
    fn test_turbulence_non_negative() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Turbulence));
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = gen.sample_raw(i as f64 * 0.03, i as f64 * 0.07);
            assert!(v >= 0.0, "Turbulence should be non-negative: {v}");
        }
    }

    #[test]
    fn test_worley_noise_bounded() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Worley));
        for i in 0..50 {
            #[allow(clippy::cast_precision_loss)]
            let v = gen.sample_raw(i as f64 * 0.06, i as f64 * 0.04);
            assert!(v >= 0.0 && v <= 1.5, "Worley noise out of range: {v}");
        }
    }

    #[test]
    fn test_different_seeds_different_output() {
        let gen_a = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Value).with_seed(1));
        let gen_b = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Value).with_seed(9999));
        let va = gen_a.sample_raw(0.5, 0.5);
        let vb = gen_b.sample_raw(0.5, 0.5);
        // With different seeds, extremely unlikely to be equal.
        assert!(
            (va - vb).abs() > 1e-9,
            "Different seeds should produce different values"
        );
    }

    #[test]
    fn test_sample_returns_gradient() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Gradient));
        let s = gen.sample(0.5, 0.5);
        // Gradient should have finite values.
        assert!(s.grad_x.is_finite());
        assert!(s.grad_y.is_finite());
    }

    #[test]
    fn test_generate_grid_dimensions() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Value));
        let grid = gen.generate_grid(16, 8);
        assert_eq!(grid.len(), 128);
    }

    #[test]
    fn test_generate_grid_values_finite() {
        let gen = NoiseFieldGenerator::new(NoiseField::new(NoiseType::Gradient).with_seed(7));
        let grid = gen.generate_grid(10, 10);
        for v in &grid {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_octave_clamp() {
        let nf = NoiseField::new(NoiseType::Turbulence).with_octaves(0);
        assert_eq!(nf.octaves, 1);
        let nf2 = NoiseField::new(NoiseType::Turbulence).with_octaves(100);
        assert_eq!(nf2.octaves, 16);
    }

    #[test]
    fn test_frequency_clamp() {
        let nf = NoiseField::new(NoiseType::Value).with_frequency(-5.0);
        assert!(nf.frequency > 0.0);
    }
}
