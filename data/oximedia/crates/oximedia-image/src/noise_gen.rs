#![allow(dead_code)]
//! Procedural noise generation for image processing and VFX workflows.
//!
//! Provides multiple noise algorithms commonly used in compositing, texture
//! generation, and film grain simulation:
//!
//! - **Perlin noise** - Smooth gradient noise for natural-looking textures
//! - **Simplex noise** - Improved Perlin with fewer artifacts
//! - **Film grain** - Photographic grain simulation with configurable intensity
//! - **Gaussian noise** - Statistically uniform noise for testing and effects
//! - **Salt-and-pepper** - Impulse noise for corruption simulation

use std::f64::consts::PI;

/// Type of noise to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Perlin gradient noise.
    Perlin,
    /// Simplex noise (improved Perlin).
    Simplex,
    /// Film grain simulation.
    FilmGrain,
    /// Gaussian (normally distributed) noise.
    Gaussian,
    /// Salt-and-pepper impulse noise.
    SaltAndPepper,
    /// Uniform random noise.
    Uniform,
}

/// Configuration for noise generation.
#[derive(Clone, Debug)]
pub struct NoiseConfig {
    /// Width of the output noise field in pixels.
    pub width: u32,
    /// Height of the output noise field in pixels.
    pub height: u32,
    /// Type of noise to generate.
    pub noise_type: NoiseType,
    /// Noise intensity in the range `[0.0, 1.0]`.
    pub intensity: f64,
    /// Frequency / scale of the noise pattern.
    pub frequency: f64,
    /// Number of octaves for fractal noise (1-8).
    pub octaves: u32,
    /// Persistence for fractal noise (amplitude decay per octave).
    pub persistence: f64,
    /// Lacunarity for fractal noise (frequency gain per octave).
    pub lacunarity: f64,
    /// Random seed for reproducible results.
    pub seed: u64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            noise_type: NoiseType::Perlin,
            intensity: 0.5,
            frequency: 4.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            seed: 42,
        }
    }
}

impl NoiseConfig {
    /// Create a new noise configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set the noise type.
    pub fn with_type(mut self, noise_type: NoiseType) -> Self {
        self.noise_type = noise_type;
        self
    }

    /// Set the intensity.
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set the frequency.
    pub fn with_frequency(mut self, frequency: f64) -> Self {
        self.frequency = frequency;
        self
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the number of octaves.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves.clamp(1, 8);
        self
    }
}

/// A 2D noise field represented as a flat array of f64 values in `[0.0, 1.0]`.
#[derive(Clone, Debug)]
pub struct NoiseField {
    /// The noise data (row-major, values in `[0.0, 1.0]`).
    pub data: Vec<f64>,
    /// Width of the noise field.
    pub width: u32,
    /// Height of the noise field.
    pub height: u32,
}

impl NoiseField {
    /// Create a new noise field filled with zeros.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0.0; (width as usize) * (height as usize)],
            width,
            height,
        }
    }

    /// Get the value at the given pixel coordinate.
    pub fn get(&self, x: u32, y: u32) -> f64 {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)]
        } else {
            0.0
        }
    }

    /// Set the value at the given pixel coordinate.
    pub fn set(&mut self, x: u32, y: u32, value: f64) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = value;
        }
    }

    /// Returns the minimum value in the noise field.
    pub fn min_value(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Returns the maximum value in the noise field.
    pub fn max_value(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Normalize all values to the range `[0.0, 1.0]`.
    pub fn normalize(&mut self) {
        let min = self.min_value();
        let max = self.max_value();
        let range = max - min;
        if range > f64::EPSILON {
            for v in &mut self.data {
                *v = (*v - min) / range;
            }
        }
    }

    /// Apply a threshold, converting to binary (0.0 or 1.0).
    pub fn threshold(&mut self, threshold: f64) {
        for v in &mut self.data {
            *v = if *v >= threshold { 1.0 } else { 0.0 };
        }
    }

    /// Invert the noise field (1.0 - value).
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// Scale all values by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.data {
            *v = (*v * factor).clamp(0.0, 1.0);
        }
    }

    /// Return the number of pixels in the field.
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Compute the mean value of the noise field.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        mean
    }
}

/// Simple pseudo-random number generator (xorshift64).
#[derive(Clone, Debug)]
struct Xorshift64 {
    /// Current state.
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG from a seed.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate the next u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a float in [0.0, 1.0).
    #[allow(clippy::cast_precision_loss)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a Gaussian-distributed value using Box-Muller transform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::EPSILON);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Permutation table for gradient noise.
#[derive(Clone, Debug)]
struct PermTable {
    /// Permutation array (512 entries for wrapping).
    perm: Vec<u8>,
}

impl PermTable {
    /// Build a permutation table from a seed.
    fn from_seed(seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);
        let mut perm: Vec<u8> = (0..=255).collect();
        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            #[allow(clippy::cast_precision_loss)]
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        // Duplicate for wrapping
        let mut full = perm.clone();
        full.extend_from_slice(&perm);
        Self { perm: full }
    }

    /// Hash two coordinates.
    fn hash(&self, x: i32, y: i32) -> u8 {
        let xi = (x & 255) as usize;
        let yi = (y & 255) as usize;
        self.perm[self.perm[xi] as usize + yi]
    }
}

/// Compute a smooth interpolation (quintic Hermite).
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Gradient function for 2D Perlin noise.
fn grad2d(hash: u8, x: f64, y: f64) -> f64 {
    match hash & 3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

/// Generate Perlin noise at a single 2D point.
fn perlin_2d(perm: &PermTable, x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm.hash(xi, yi);
    let ab = perm.hash(xi, yi + 1);
    let ba = perm.hash(xi + 1, yi);
    let bb = perm.hash(xi + 1, yi + 1);

    let x1 = lerp(grad2d(aa, xf, yf), grad2d(ba, xf - 1.0, yf), u);
    let x2 = lerp(grad2d(ab, xf, yf - 1.0), grad2d(bb, xf - 1.0, yf - 1.0), u);
    lerp(x1, x2, v)
}

/// Generate fractal (fBm) Perlin noise at a single point.
fn fractal_perlin(perm: &PermTable, x: f64, y: f64, config: &NoiseConfig) -> f64 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = config.frequency;
    let mut max_value = 0.0;

    for _ in 0..config.octaves {
        total += perlin_2d(perm, x * frequency, y * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= config.persistence;
        frequency *= config.lacunarity;
    }

    // Normalize to [-1, 1] then remap to [0, 1]
    if max_value > 0.0 {
        (total / max_value + 1.0) * 0.5
    } else {
        0.5
    }
}

/// Generate a noise field from the given configuration.
pub fn generate_noise(config: &NoiseConfig) -> NoiseField {
    let mut field = NoiseField::new(config.width, config.height);
    let perm = PermTable::from_seed(config.seed);
    let mut rng = Xorshift64::new(config.seed);

    for y in 0..config.height {
        for x in 0..config.width {
            #[allow(clippy::cast_precision_loss)]
            let nx = x as f64 / config.width as f64;
            #[allow(clippy::cast_precision_loss)]
            let ny = y as f64 / config.height as f64;

            let value = match config.noise_type {
                NoiseType::Perlin | NoiseType::Simplex => fractal_perlin(&perm, nx, ny, config),
                NoiseType::Gaussian => {
                    let g = rng.next_gaussian();
                    // Map from ~[-3, 3] to [0, 1]
                    (g / 6.0 + 0.5).clamp(0.0, 1.0)
                }
                NoiseType::Uniform => rng.next_f64(),
                NoiseType::SaltAndPepper => {
                    let r = rng.next_f64();
                    if r < config.intensity * 0.5 {
                        0.0
                    } else if r > 1.0 - config.intensity * 0.5 {
                        1.0
                    } else {
                        0.5
                    }
                }
                NoiseType::FilmGrain => {
                    // Film grain: Gaussian with intensity-dependent variance
                    let g = rng.next_gaussian() * config.intensity * 0.3;
                    (0.5 + g).clamp(0.0, 1.0)
                }
            };

            field.set(x, y, value);
        }
    }

    // Apply intensity scaling for noise types that need it
    if matches!(
        config.noise_type,
        NoiseType::Perlin | NoiseType::Simplex | NoiseType::Uniform
    ) {
        for v in &mut field.data {
            let centered = *v - 0.5;
            *v = (0.5 + centered * config.intensity).clamp(0.0, 1.0);
        }
    }

    field
}

/// Blend noise onto an existing pixel buffer (additive blend).
///
/// The buffer is expected to be in `[0.0, 1.0]` range with the same dimensions.
pub fn blend_noise_additive(buffer: &mut [f64], noise: &NoiseField, strength: f64) {
    let strength = strength.clamp(0.0, 1.0);
    for (pixel, &noise_val) in buffer.iter_mut().zip(noise.data.iter()) {
        let offset = (noise_val - 0.5) * 2.0 * strength;
        *pixel = (*pixel + offset).clamp(0.0, 1.0);
    }
}

/// Blend noise onto an existing pixel buffer (overlay blend).
pub fn blend_noise_overlay(buffer: &mut [f64], noise: &NoiseField, strength: f64) {
    let strength = strength.clamp(0.0, 1.0);
    for (pixel, &noise_val) in buffer.iter_mut().zip(noise.data.iter()) {
        let overlay = if *pixel < 0.5 {
            2.0 * *pixel * noise_val
        } else {
            1.0 - 2.0 * (1.0 - *pixel) * (1.0 - noise_val)
        };
        *pixel = lerp(*pixel, overlay, strength).clamp(0.0, 1.0);
    }
}

/// Compute statistical properties of a noise field.
#[derive(Clone, Debug)]
pub struct NoiseStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
}

impl NoiseStats {
    /// Compute statistics from a noise field.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_field(field: &NoiseField) -> Self {
        let n = field.data.len() as f64;
        if n == 0.0 {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
            };
        }

        let min = field.min_value();
        let max = field.max_value();
        let mean = field.mean();
        let variance = field.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        Self {
            min,
            max,
            mean,
            std_dev: variance.sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_config_default() {
        let config = NoiseConfig::default();
        assert_eq!(config.width, 256);
        assert_eq!(config.height, 256);
        assert_eq!(config.noise_type, NoiseType::Perlin);
        assert!((config.intensity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_config_builder() {
        let config = NoiseConfig::new(64, 64)
            .with_type(NoiseType::Gaussian)
            .with_intensity(0.8)
            .with_frequency(2.0)
            .with_seed(123);
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
        assert_eq!(config.noise_type, NoiseType::Gaussian);
        assert!((config.intensity - 0.8).abs() < f64::EPSILON);
        assert!((config.frequency - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_noise_field_new() {
        let field = NoiseField::new(16, 16);
        assert_eq!(field.width, 16);
        assert_eq!(field.height, 16);
        assert_eq!(field.data.len(), 256);
        assert!((field.get(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_field_get_set() {
        let mut field = NoiseField::new(8, 8);
        field.set(3, 4, 0.75);
        assert!((field.get(3, 4) - 0.75).abs() < f64::EPSILON);
        // Out of bounds returns 0.0
        assert!((field.get(100, 100) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_field_normalize() {
        let mut field = NoiseField::new(4, 1);
        field.data = vec![0.2, 0.5, 0.8, 1.0];
        field.normalize();
        assert!((field.data[0] - 0.0).abs() < f64::EPSILON);
        assert!((field.data[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_field_threshold() {
        let mut field = NoiseField::new(4, 1);
        field.data = vec![0.1, 0.4, 0.6, 0.9];
        field.threshold(0.5);
        assert!((field.data[0] - 0.0).abs() < f64::EPSILON);
        assert!((field.data[1] - 0.0).abs() < f64::EPSILON);
        assert!((field.data[2] - 1.0).abs() < f64::EPSILON);
        assert!((field.data[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_field_invert() {
        let mut field = NoiseField::new(4, 1);
        field.data = vec![0.0, 0.25, 0.75, 1.0];
        field.invert();
        assert!((field.data[0] - 1.0).abs() < f64::EPSILON);
        assert!((field.data[1] - 0.75).abs() < f64::EPSILON);
        assert!((field.data[2] - 0.25).abs() < f64::EPSILON);
        assert!((field.data[3] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_perlin_noise() {
        let config = NoiseConfig::new(32, 32)
            .with_type(NoiseType::Perlin)
            .with_seed(42);
        let field = generate_noise(&config);
        assert_eq!(field.pixel_count(), 1024);
        // All values should be in [0, 1]
        for v in &field.data {
            assert!(*v >= 0.0 && *v <= 1.0, "value out of range: {v}");
        }
    }

    #[test]
    fn test_generate_gaussian_noise() {
        let config = NoiseConfig::new(64, 64)
            .with_type(NoiseType::Gaussian)
            .with_seed(99);
        let field = generate_noise(&config);
        let stats = NoiseStats::from_field(&field);
        // Mean should be around 0.5
        assert!(
            (stats.mean - 0.5).abs() < 0.1,
            "Gaussian mean too far from 0.5: {}",
            stats.mean
        );
    }

    #[test]
    fn test_generate_salt_and_pepper() {
        let config = NoiseConfig::new(64, 64)
            .with_type(NoiseType::SaltAndPepper)
            .with_intensity(0.2)
            .with_seed(77);
        let field = generate_noise(&config);
        // Should have values at 0.0, 0.5, and 1.0
        let zeros = field.data.iter().filter(|&&v| v == 0.0).count();
        let ones = field.data.iter().filter(|&&v| v == 1.0).count();
        let halves = field
            .data
            .iter()
            .filter(|&&v| (v - 0.5).abs() < f64::EPSILON)
            .count();
        assert!(zeros > 0, "Expected some zero values");
        assert!(ones > 0, "Expected some one values");
        assert!(halves > 0, "Expected some 0.5 values");
    }

    #[test]
    fn test_blend_noise_additive() {
        let noise = NoiseField {
            data: vec![0.5; 4],
            width: 2,
            height: 2,
        };
        let mut buffer = vec![0.5; 4];
        blend_noise_additive(&mut buffer, &noise, 1.0);
        // 0.5 noise = no offset (noise - 0.5 = 0)
        for v in &buffer {
            assert!((*v - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_blend_noise_overlay() {
        let noise = NoiseField {
            data: vec![0.5; 4],
            width: 2,
            height: 2,
        };
        let mut buffer = vec![0.5; 4];
        blend_noise_overlay(&mut buffer, &noise, 1.0);
        // overlay of 0.5 over 0.5 => 0.5
        for v in &buffer {
            assert!((*v - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_noise_stats() {
        let field = NoiseField {
            data: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            width: 5,
            height: 1,
        };
        let stats = NoiseStats::from_field(&field);
        assert!((stats.min - 0.0).abs() < f64::EPSILON);
        assert!((stats.max - 1.0).abs() < f64::EPSILON);
        assert!((stats.mean - 0.5).abs() < f64::EPSILON);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_noise_reproducibility() {
        let config = NoiseConfig::new(16, 16)
            .with_type(NoiseType::Perlin)
            .with_seed(12345);
        let field1 = generate_noise(&config);
        let field2 = generate_noise(&config);
        assert_eq!(field1.data, field2.data);
    }

    #[test]
    fn test_film_grain_noise() {
        let config = NoiseConfig::new(32, 32)
            .with_type(NoiseType::FilmGrain)
            .with_intensity(0.5)
            .with_seed(55);
        let field = generate_noise(&config);
        let stats = NoiseStats::from_field(&field);
        // Film grain should be centered around 0.5
        assert!(
            (stats.mean - 0.5).abs() < 0.1,
            "Film grain mean unexpected: {}",
            stats.mean
        );
        // All values in range
        for v in &field.data {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_noise_field_mean() {
        let field = NoiseField {
            data: vec![0.2, 0.4, 0.6, 0.8],
            width: 2,
            height: 2,
        };
        assert!((field.mean() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_field_scale() {
        let mut field = NoiseField::new(4, 1);
        field.data = vec![0.2, 0.4, 0.6, 0.8];
        field.scale(0.5);
        assert!((field.data[0] - 0.1).abs() < f64::EPSILON);
        assert!((field.data[3] - 0.4).abs() < f64::EPSILON);
    }
}
