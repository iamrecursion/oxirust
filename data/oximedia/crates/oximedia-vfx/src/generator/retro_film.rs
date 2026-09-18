//! Retro / vintage film look generator.
//!
//! Combines film grain, desaturation, vignette darkening, and gate weave
//! (horizontal/vertical jitter) to produce a convincing aged film aesthetic.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// Configuration for the retro film look.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetroFilmConfig {
    /// Film grain intensity (0.0 = none, 1.0 = heavy grain).
    pub grain_intensity: f32,
    /// Desaturation level (0.0 = full colour, 1.0 = monochrome).
    pub desaturation: f32,
    /// Vignette strength (0.0 = none, 1.0 = dark corners).
    pub vignette_strength: f32,
    /// Vignette inner radius (normalised).
    pub vignette_inner: f32,
    /// Gate weave horizontal jitter amplitude in pixels.
    pub weave_x: f32,
    /// Gate weave vertical jitter amplitude in pixels.
    pub weave_y: f32,
    /// Lift (raise black level) for a faded look (0.0-0.3).
    pub lift: f32,
    /// Colour temperature shift: positive = warm (sepia), negative = cool.
    pub warmth: f32,
    /// Contrast reduction (0.0 = no change, 1.0 = flat).
    pub contrast_reduction: f32,
    /// Random seed for grain and weave.
    pub seed: u64,
}

impl Default for RetroFilmConfig {
    fn default() -> Self {
        Self {
            grain_intensity: 0.3,
            desaturation: 0.5,
            vignette_strength: 0.6,
            vignette_inner: 0.4,
            weave_x: 1.5,
            weave_y: 1.0,
            lift: 0.05,
            warmth: 0.15,
            contrast_reduction: 0.1,
            seed: 42,
        }
    }
}

/// Simple deterministic hash for reproducible pseudo-random values.
fn hash_mix(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x517c_c1b7_2722_0a95);
    x ^= x >> 32;
    x = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
    x ^= x >> 32;
    x
}

/// Generate a pseudo-random f32 in [-1, 1] from a seed and index.
fn pseudo_random(seed: u64, index: u64) -> f32 {
    let h = hash_mix(seed.wrapping_add(index));
    // Map to [-1, 1]
    (h as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
}

/// Retro film look generator.
///
/// Applies grain, desaturation, vignette, gate weave jitter, faded blacks,
/// and warmth shift to produce an aged film aesthetic.
#[derive(Debug, Clone)]
pub struct RetroFilmGenerator {
    config: RetroFilmConfig,
    frame_counter: u64,
}

impl RetroFilmGenerator {
    /// Create a new retro film generator.
    #[must_use]
    pub fn new(config: RetroFilmConfig) -> Self {
        Self {
            config,
            frame_counter: 0,
        }
    }

    /// Create with default "classic 8mm" look.
    #[must_use]
    pub fn classic_8mm() -> Self {
        Self::new(RetroFilmConfig {
            grain_intensity: 0.5,
            desaturation: 0.6,
            vignette_strength: 0.8,
            vignette_inner: 0.3,
            weave_x: 2.0,
            weave_y: 1.5,
            lift: 0.08,
            warmth: 0.2,
            contrast_reduction: 0.15,
            seed: 42,
        })
    }

    /// Create with "faded VHS" look.
    #[must_use]
    pub fn faded_vhs() -> Self {
        Self::new(RetroFilmConfig {
            grain_intensity: 0.2,
            desaturation: 0.3,
            vignette_strength: 0.3,
            vignette_inner: 0.5,
            weave_x: 0.5,
            weave_y: 3.0,
            lift: 0.1,
            warmth: -0.05,
            contrast_reduction: 0.2,
            seed: 7,
        })
    }

    /// Advance frame counter (for temporal grain variation).
    pub fn next_frame(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    /// Get the current gate weave offset for this frame.
    #[must_use]
    pub fn gate_weave_offset(&self) -> (f32, f32) {
        let base = hash_mix(self.config.seed.wrapping_add(self.frame_counter * 1000));
        let wx = pseudo_random(base, 0) * self.config.weave_x;
        let wy = pseudo_random(base, 1) * self.config.weave_y;
        (wx, wy)
    }

    /// Apply the retro film look to a frame, writing into `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if the output frame dimensions are zero.
    pub fn apply(&self, input: &Frame, output: &mut Frame) -> VfxResult<()> {
        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let (weave_dx, weave_dy) = self.gate_weave_offset();
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt().max(1.0);

        for y in 0..output.height.min(h) {
            for x in 0..output.width.min(w) {
                // Gate weave: sample from offset position
                let sx = (x as f32 + weave_dx).round().clamp(0.0, (w - 1) as f32) as u32;
                let sy = (y as f32 + weave_dy).round().clamp(0.0, (h - 1) as f32) as u32;
                let pixel = input.get_pixel(sx, sy).unwrap_or([0, 0, 0, 0]);

                // Convert to float
                let mut rf = pixel[0] as f32 / 255.0;
                let mut gf = pixel[1] as f32 / 255.0;
                let mut bf = pixel[2] as f32 / 255.0;

                // 1. Contrast reduction: compress toward 0.5
                let cr = self.config.contrast_reduction;
                rf = rf * (1.0 - cr) + 0.5 * cr;
                gf = gf * (1.0 - cr) + 0.5 * cr;
                bf = bf * (1.0 - cr) + 0.5 * cr;

                // 2. Lift (raise black level)
                rf = rf + self.config.lift * (1.0 - rf);
                gf = gf + self.config.lift * (1.0 - gf);
                bf = bf + self.config.lift * (1.0 - bf);

                // 3. Desaturation
                let luma = 0.299 * rf + 0.587 * gf + 0.114 * bf;
                let desat = self.config.desaturation;
                rf = rf * (1.0 - desat) + luma * desat;
                gf = gf * (1.0 - desat) + luma * desat;
                bf = bf * (1.0 - desat) + luma * desat;

                // 4. Warmth shift
                let warmth = self.config.warmth;
                rf = (rf + warmth * 0.1).clamp(0.0, 1.0);
                gf = (gf + warmth * 0.02).clamp(0.0, 1.0);
                bf = (bf - warmth * 0.08).clamp(0.0, 1.0);

                // 5. Grain
                let grain_seed = self
                    .config
                    .seed
                    .wrapping_add(self.frame_counter * 100_000)
                    .wrapping_add((y as u64) * w as u64 + x as u64);
                let noise = pseudo_random(grain_seed, 2);
                // Grain stronger in midtones
                let midtone_weight = 1.0 - (luma * 2.0 - 1.0).abs();
                let grain = noise * self.config.grain_intensity * midtone_weight * 0.3;
                rf = (rf + grain).clamp(0.0, 1.0);
                gf = (gf + grain).clamp(0.0, 1.0);
                bf = (bf + grain).clamp(0.0, 1.0);

                // 6. Vignette
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt() / max_r;
                let inner = self.config.vignette_inner;
                let vig = if r <= inner {
                    1.0
                } else {
                    let t = ((r - inner) / (1.0 - inner).max(0.001)).min(1.0);
                    let falloff = t * t * (3.0 - 2.0 * t); // smoothstep
                    1.0 - self.config.vignette_strength * falloff
                };
                rf *= vig;
                gf *= vig;
                bf *= vig;

                output.set_pixel(
                    x,
                    y,
                    [
                        (rf * 255.0).clamp(0.0, 255.0) as u8,
                        (gf * 255.0).clamp(0.0, 255.0) as u8,
                        (bf * 255.0).clamp(0.0, 255.0) as u8,
                        pixel[3],
                    ],
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear([255, 255, 255, 255]);
        f
    }

    fn grey_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear([128, 128, 128, 255]);
        f
    }

    #[test]
    fn test_retro_film_config_default() {
        let cfg = RetroFilmConfig::default();
        assert!(cfg.grain_intensity > 0.0);
        assert!(cfg.desaturation > 0.0);
        assert!(cfg.vignette_strength > 0.0);
    }

    #[test]
    fn test_retro_film_apply_basic() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig::default());
        let input = white_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        gen.apply(&input, &mut output).expect("apply");
        // Output should not be pure white anymore
        let center = output.get_pixel(16, 16).expect("pixel");
        assert!(
            center[0] < 255 || center[1] < 255 || center[2] < 255,
            "retro effect should modify pixels"
        );
    }

    #[test]
    fn test_retro_film_desaturation() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig {
            grain_intensity: 0.0,
            desaturation: 1.0,
            vignette_strength: 0.0,
            weave_x: 0.0,
            weave_y: 0.0,
            lift: 0.0,
            warmth: 0.0,
            contrast_reduction: 0.0,
            ..Default::default()
        });
        let mut input = Frame::new(8, 8).expect("frame");
        input.clear([200, 100, 50, 255]);
        let mut output = Frame::new(8, 8).expect("frame");
        gen.apply(&input, &mut output).expect("apply");

        let p = output.get_pixel(4, 4).expect("pixel");
        // Fully desaturated: R ~= G ~= B
        let max_diff = (p[0] as i32 - p[1] as i32)
            .abs()
            .max((p[1] as i32 - p[2] as i32).abs());
        assert!(max_diff < 5, "should be nearly monochrome, diff={max_diff}");
    }

    #[test]
    fn test_retro_film_vignette() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig {
            grain_intensity: 0.0,
            desaturation: 0.0,
            vignette_strength: 1.0,
            vignette_inner: 0.2,
            weave_x: 0.0,
            weave_y: 0.0,
            lift: 0.0,
            warmth: 0.0,
            contrast_reduction: 0.0,
            ..Default::default()
        });
        let input = white_frame(64, 64);
        let mut output = Frame::new(64, 64).expect("frame");
        gen.apply(&input, &mut output).expect("apply");

        let center = output.get_pixel(32, 32).expect("center");
        let corner = output.get_pixel(0, 0).expect("corner");
        assert!(
            center[0] > corner[0],
            "center={} should be brighter than corner={}",
            center[0],
            corner[0]
        );
    }

    #[test]
    fn test_gate_weave_offset() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig {
            weave_x: 3.0,
            weave_y: 2.0,
            ..Default::default()
        });
        let (wx, wy) = gen.gate_weave_offset();
        assert!(wx.abs() <= 3.0);
        assert!(wy.abs() <= 2.0);
    }

    #[test]
    fn test_gate_weave_varies_per_frame() {
        let mut gen = RetroFilmGenerator::new(RetroFilmConfig::default());
        let offset1 = gen.gate_weave_offset();
        gen.next_frame();
        let offset2 = gen.gate_weave_offset();
        // Offsets should differ between frames (extremely unlikely to be identical)
        assert!(
            (offset1.0 - offset2.0).abs() > 0.001 || (offset1.1 - offset2.1).abs() > 0.001,
            "weave should vary per frame"
        );
    }

    #[test]
    fn test_retro_film_warmth_shift() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig {
            grain_intensity: 0.0,
            desaturation: 0.0,
            vignette_strength: 0.0,
            weave_x: 0.0,
            weave_y: 0.0,
            lift: 0.0,
            warmth: 1.0,
            contrast_reduction: 0.0,
            ..Default::default()
        });
        let input = grey_frame(8, 8);
        let mut output = Frame::new(8, 8).expect("frame");
        gen.apply(&input, &mut output).expect("apply");

        let p = output.get_pixel(4, 4).expect("pixel");
        // Warm shift: red > blue
        assert!(p[0] > p[2], "warm: red={} should be > blue={}", p[0], p[2]);
    }

    #[test]
    fn test_retro_film_lift() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig {
            grain_intensity: 0.0,
            desaturation: 0.0,
            vignette_strength: 0.0,
            weave_x: 0.0,
            weave_y: 0.0,
            lift: 0.2,
            warmth: 0.0,
            contrast_reduction: 0.0,
            ..Default::default()
        });
        let mut input = Frame::new(8, 8).expect("frame");
        input.clear([0, 0, 0, 255]);
        let mut output = Frame::new(8, 8).expect("frame");
        gen.apply(&input, &mut output).expect("apply");

        let p = output.get_pixel(4, 4).expect("pixel");
        // Black pixels should be lifted above 0
        assert!(p[0] > 0, "lift should raise black, got {}", p[0]);
    }

    #[test]
    fn test_classic_8mm_preset() {
        let gen = RetroFilmGenerator::classic_8mm();
        assert!((gen.config.grain_intensity - 0.5).abs() < 0.01);
        assert!((gen.config.desaturation - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_faded_vhs_preset() {
        let gen = RetroFilmGenerator::faded_vhs();
        assert!(gen.config.warmth < 0.0); // VHS is slightly cool
        assert!((gen.config.lift - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_pseudo_random_range() {
        for i in 0..100 {
            let v = pseudo_random(42, i);
            assert!(v >= -1.0 && v <= 1.0, "pseudo_random out of range: {v}");
        }
    }

    #[test]
    fn test_retro_film_alpha_preserved() {
        let gen = RetroFilmGenerator::new(RetroFilmConfig::default());
        let mut input = Frame::new(16, 16).expect("frame");
        input.clear([128, 128, 128, 200]);
        let mut output = Frame::new(16, 16).expect("frame");
        gen.apply(&input, &mut output).expect("apply");

        for y in 0..16 {
            for x in 0..16 {
                let p = output.get_pixel(x, y).expect("pixel");
                assert_eq!(p[3], 200, "alpha should be preserved");
            }
        }
    }
}
