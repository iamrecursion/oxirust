//! Film grain synthesis for restored video.
//!
//! When a video clip has been digitally cleaned or upscaled, it can look
//! unnaturally smooth compared to source footage. This module simulates
//! analogue film grain to restore a natural, organic look.

#![allow(dead_code)]

/// Type of grain to synthesise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainType {
    /// Gaussian white-noise grain (simplest model).
    Gaussian,
    /// Luma-dependent grain: less grain in shadows and highlights.
    LumaDependent,
    /// Frequency-shaped grain matching typical film stock.
    FilmStock,
    /// Chroma-only grain affecting colour planes only.
    ChromaOnly,
}

/// Configuration for grain synthesis.
#[derive(Debug, Clone)]
pub struct GrainConfig {
    /// Type of grain to apply.
    pub grain_type: GrainType,
    /// Overall strength of grain in the range `[0.0, 1.0]`.
    pub strength: f32,
    /// Grain size (higher = coarser).
    pub size: f32,
    /// Grain frequency in cycles per pixel (for `FilmStock`).
    pub frequency: f32,
    /// Seed for the pseudo-random number generator.
    pub seed: u64,
    /// Whether to apply the same grain pattern to both colour channels.
    pub link_channels: bool,
}

impl Default for GrainConfig {
    fn default() -> Self {
        Self {
            grain_type: GrainType::Gaussian,
            strength: 0.05,
            size: 1.0,
            frequency: 0.5,
            seed: 42,
            link_channels: false,
        }
    }
}

/// Lightweight LCG PRNG used internally (no external dependency).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x1234_ABCD_EF01_2345,
        }
    }

    /// Return a sample in `[-1.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    fn next_f32(&mut self) -> f32 {
        // Knuth multiplicative hash
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Applies synthetic grain to a luma plane represented as a flat pixel buffer.
#[derive(Debug)]
pub struct GrainAdder {
    config: GrainConfig,
}

impl GrainAdder {
    /// Create a new `GrainAdder` with default configuration.
    pub fn new() -> Self {
        Self {
            config: GrainConfig::default(),
        }
    }

    /// Create a new `GrainAdder` with custom configuration.
    pub fn with_config(config: GrainConfig) -> Self {
        Self { config }
    }

    /// Apply grain to a frame represented as a `width × height` slice of
    /// `f32` luma values in the range `[0.0, 1.0]`.
    ///
    /// `frame_index` is used to vary the grain pattern per frame.
    pub fn apply_frame(&self, luma: &mut [f32], width: u32, height: u32, frame_index: u64) {
        let seed = self
            .config
            .seed
            .wrapping_add(frame_index.wrapping_mul(0xDEAD_BEEF));
        let mut rng = Lcg::new(seed);
        let strength = self.config.strength.clamp(0.0, 1.0);
        let expected = (width as usize).saturating_mul(height as usize);
        let len = luma.len().min(expected);

        match self.config.grain_type {
            GrainType::Gaussian => {
                for px in luma[..len].iter_mut() {
                    *px = (*px + rng.next_f32() * strength).clamp(0.0, 1.0);
                }
            }
            GrainType::LumaDependent => {
                for px in luma[..len].iter_mut() {
                    // Least grain at 0.0 and 1.0, most at 0.5
                    let mask = 1.0 - (2.0 * *px - 1.0).powi(2);
                    *px = (*px + rng.next_f32() * strength * mask).clamp(0.0, 1.0);
                }
            }
            GrainType::FilmStock | GrainType::ChromaOnly => {
                // Simple approximation: apply plain Gaussian grain
                for px in luma[..len].iter_mut() {
                    *px = (*px + rng.next_f32() * strength * 0.7).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Return the current configuration.
    pub fn config(&self) -> &GrainConfig {
        &self.config
    }

    /// Update the grain strength.
    pub fn set_strength(&mut self, strength: f32) {
        self.config.strength = strength.clamp(0.0, 1.0);
    }

    /// Update the grain type.
    pub fn set_grain_type(&mut self, grain_type: GrainType) {
        self.config.grain_type = grain_type;
    }
}

impl Default for GrainAdder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Return the RMS of differences between two slices.
    #[allow(clippy::cast_precision_loss)]
    fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        (sum / a.len() as f32).sqrt()
    }

    #[test]
    fn test_grain_changes_pixels() {
        let adder = GrainAdder::new();
        let original = vec![0.5_f32; 100];
        let mut frame = original.clone();
        adder.apply_frame(&mut frame, 10, 10, 0);
        // At least some pixels should have changed
        assert!(rms_diff(&original, &frame) > 0.0);
    }

    #[test]
    fn test_grain_stays_in_range() {
        let adder = GrainAdder::with_config(GrainConfig {
            strength: 1.0,
            ..Default::default()
        });
        let mut frame = vec![0.5_f32; 64];
        adder.apply_frame(&mut frame, 8, 8, 0);
        for &px in &frame {
            assert!((0.0..=1.0).contains(&px), "pixel out of range: {px}");
        }
    }

    #[test]
    fn test_zero_strength_no_change() {
        let adder = GrainAdder::with_config(GrainConfig {
            strength: 0.0,
            ..Default::default()
        });
        let original = vec![0.5_f32; 100];
        let mut frame = original.clone();
        adder.apply_frame(&mut frame, 10, 10, 0);
        for (a, b) in original.iter().zip(frame.iter()) {
            assert!((a - b).abs() < 1e-7, "expected no change with strength=0");
        }
    }

    #[test]
    fn test_different_frames_differ() {
        let adder = GrainAdder::new();
        let mut frame0 = vec![0.5_f32; 100];
        let mut frame1 = vec![0.5_f32; 100];
        adder.apply_frame(&mut frame0, 10, 10, 0);
        adder.apply_frame(&mut frame1, 10, 10, 1);
        assert!(
            rms_diff(&frame0, &frame1) > 0.0,
            "frames 0 and 1 should differ"
        );
    }

    #[test]
    fn test_same_seed_same_result() {
        let adder = GrainAdder::new();
        let mut frame_a = vec![0.5_f32; 100];
        let mut frame_b = vec![0.5_f32; 100];
        adder.apply_frame(&mut frame_a, 10, 10, 7);
        adder.apply_frame(&mut frame_b, 10, 10, 7);
        assert_eq!(frame_a, frame_b);
    }

    #[test]
    fn test_luma_dependent_grain() {
        let adder = GrainAdder::with_config(GrainConfig {
            grain_type: GrainType::LumaDependent,
            strength: 0.3,
            ..Default::default()
        });
        let mut frame = vec![0.5_f32; 64];
        adder.apply_frame(&mut frame, 8, 8, 0);
        for &px in &frame {
            assert!((0.0..=1.0).contains(&px));
        }
    }

    #[test]
    fn test_film_stock_grain() {
        let adder = GrainAdder::with_config(GrainConfig {
            grain_type: GrainType::FilmStock,
            strength: 0.1,
            ..Default::default()
        });
        let mut frame = vec![0.5_f32; 64];
        adder.apply_frame(&mut frame, 8, 8, 0);
        for &px in &frame {
            assert!((0.0..=1.0).contains(&px));
        }
    }

    #[test]
    fn test_chroma_only_grain() {
        let adder = GrainAdder::with_config(GrainConfig {
            grain_type: GrainType::ChromaOnly,
            strength: 0.2,
            ..Default::default()
        });
        let mut frame = vec![0.5_f32; 64];
        adder.apply_frame(&mut frame, 8, 8, 0);
        for &px in &frame {
            assert!((0.0..=1.0).contains(&px));
        }
    }

    #[test]
    fn test_set_strength() {
        let mut adder = GrainAdder::new();
        adder.set_strength(0.8);
        assert!((adder.config().strength - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_set_strength_clamps() {
        let mut adder = GrainAdder::new();
        adder.set_strength(5.0);
        assert!((adder.config().strength - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_grain_type() {
        let mut adder = GrainAdder::new();
        adder.set_grain_type(GrainType::FilmStock);
        assert_eq!(adder.config().grain_type, GrainType::FilmStock);
    }

    #[test]
    fn test_empty_frame_no_panic() {
        let adder = GrainAdder::new();
        let mut frame: Vec<f32> = Vec::new();
        adder.apply_frame(&mut frame, 0, 0, 0); // Should not panic
    }

    #[test]
    fn test_default_config_values() {
        let cfg = GrainConfig::default();
        assert_eq!(cfg.grain_type, GrainType::Gaussian);
        assert!((cfg.strength - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_black_frame_stays_near_black() {
        let adder = GrainAdder::with_config(GrainConfig {
            strength: 0.01,
            grain_type: GrainType::LumaDependent,
            ..Default::default()
        });
        let mut frame = vec![0.0_f32; 64];
        adder.apply_frame(&mut frame, 8, 8, 0);
        // LumaDependent has near-zero mask at luma=0, so grain is minimal.
        for &px in &frame {
            assert!(px.abs() < 0.05, "black frame grain too strong: {px}");
        }
    }
}
