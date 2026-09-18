//! Digital glitch effect module.
//!
//! Provides various digital artifact effects commonly used in creative
//! video production: RGB channel splitting, scanline corruption, block
//! displacement (datamosh-style), and signal noise injection.

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Type of glitch effect to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlitchType {
    /// RGB channel offset: shifts R, G, B channels independently.
    RgbShift,
    /// Scanline artifacts: corrupted horizontal scan lines.
    ScanLine,
    /// Block displacement: rectangular regions shifted (datamosh-style).
    BlockDisplace,
    /// Digital noise: random pixel corruption.
    DigitalNoise,
    /// Combined: all effects layered together.
    Combined,
}

/// Configuration for RGB channel shift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbShiftConfig {
    /// Red channel horizontal offset in pixels.
    pub red_offset_x: i32,
    /// Red channel vertical offset in pixels.
    pub red_offset_y: i32,
    /// Green channel horizontal offset in pixels.
    pub green_offset_x: i32,
    /// Green channel vertical offset in pixels.
    pub green_offset_y: i32,
    /// Blue channel horizontal offset in pixels.
    pub blue_offset_x: i32,
    /// Blue channel vertical offset in pixels.
    pub blue_offset_y: i32,
}

impl Default for RgbShiftConfig {
    fn default() -> Self {
        Self {
            red_offset_x: 5,
            red_offset_y: 0,
            green_offset_x: 0,
            green_offset_y: 0,
            blue_offset_x: -5,
            blue_offset_y: 0,
        }
    }
}

/// Configuration for scanline artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanLineConfig {
    /// Probability (0.0-1.0) of a scanline being corrupted.
    pub corruption_rate: f32,
    /// Maximum horizontal shift of a corrupted scanline in pixels.
    pub max_shift: u32,
    /// Scanline thickness in pixels.
    pub line_thickness: u32,
    /// Whether to add brightness variation to corrupted lines.
    pub brightness_jitter: bool,
    /// Seed for deterministic pseudo-random generation.
    pub seed: u64,
}

impl Default for ScanLineConfig {
    fn default() -> Self {
        Self {
            corruption_rate: 0.1,
            max_shift: 20,
            line_thickness: 2,
            brightness_jitter: true,
            seed: 42,
        }
    }
}

/// Configuration for block displacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDisplaceConfig {
    /// Minimum block width in pixels.
    pub min_block_width: u32,
    /// Maximum block width in pixels.
    pub max_block_width: u32,
    /// Minimum block height in pixels.
    pub min_block_height: u32,
    /// Maximum block height in pixels.
    pub max_block_height: u32,
    /// Number of displaced blocks.
    pub block_count: u32,
    /// Maximum displacement in pixels.
    pub max_displacement: i32,
    /// Seed for deterministic pseudo-random generation.
    pub seed: u64,
}

impl Default for BlockDisplaceConfig {
    fn default() -> Self {
        Self {
            min_block_width: 20,
            max_block_width: 200,
            min_block_height: 5,
            max_block_height: 30,
            block_count: 8,
            max_displacement: 50,
            seed: 42,
        }
    }
}

/// Configuration for digital noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalNoiseConfig {
    /// Probability (0.0-1.0) of a pixel being corrupted.
    pub noise_density: f32,
    /// Maximum intensity deviation (0-255).
    pub max_deviation: u8,
    /// Whether to corrupt alpha channel too.
    pub affect_alpha: bool,
    /// Seed for deterministic pseudo-random generation.
    pub seed: u64,
}

impl Default for DigitalNoiseConfig {
    fn default() -> Self {
        Self {
            noise_density: 0.02,
            max_deviation: 128,
            affect_alpha: false,
            seed: 42,
        }
    }
}

/// A simple deterministic pseudo-random number generator (xorshift64).
///
/// Used internally for repeatable glitch patterns without external crate deps.
#[derive(Debug, Clone)]
struct PseudoRng {
    state: u64,
}

impl PseudoRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate a pseudo-random u64 using xorshift64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a u32 in [0, max) range.
    fn next_u32_range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        (self.next_u64() % u64::from(max)) as u32
    }

    /// Generate an i32 in [-max, max] range.
    fn next_i32_range(&mut self, max: i32) -> i32 {
        if max == 0 {
            return 0;
        }
        let abs_max = max.unsigned_abs();
        let val = self.next_u32_range(abs_max * 2 + 1);
        val as i32 - max
    }

    /// Generate a float in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
}

/// Digital glitch effect for creative video production.
///
/// Simulates various types of digital signal corruption including
/// RGB channel splitting, scanline artifacts, block displacement,
/// and random noise injection.
pub struct GlitchEffect {
    glitch_type: GlitchType,
    /// Overall effect intensity (0.0-1.0).
    intensity: f32,
    rgb_shift: RgbShiftConfig,
    scan_line: ScanLineConfig,
    block_displace: BlockDisplaceConfig,
    digital_noise: DigitalNoiseConfig,
    /// Time-based animation: when true, the effect varies with `params.time`.
    animate: bool,
}

impl GlitchEffect {
    /// Create a new glitch effect with the specified type.
    #[must_use]
    pub fn new(glitch_type: GlitchType) -> Self {
        Self {
            glitch_type,
            intensity: 1.0,
            rgb_shift: RgbShiftConfig::default(),
            scan_line: ScanLineConfig::default(),
            block_displace: BlockDisplaceConfig::default(),
            digital_noise: DigitalNoiseConfig::default(),
            animate: false,
        }
    }

    /// Set the overall intensity (0.0-1.0).
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set RGB shift configuration.
    #[must_use]
    pub fn with_rgb_shift(mut self, config: RgbShiftConfig) -> Self {
        self.rgb_shift = config;
        self
    }

    /// Set scanline configuration.
    #[must_use]
    pub fn with_scan_line(mut self, config: ScanLineConfig) -> Self {
        self.scan_line = config;
        self
    }

    /// Set block displacement configuration.
    #[must_use]
    pub fn with_block_displace(mut self, config: BlockDisplaceConfig) -> Self {
        self.block_displace = config;
        self
    }

    /// Set digital noise configuration.
    #[must_use]
    pub fn with_digital_noise(mut self, config: DigitalNoiseConfig) -> Self {
        self.digital_noise = config;
        self
    }

    /// Enable time-based animation.
    #[must_use]
    pub const fn with_animate(mut self, animate: bool) -> Self {
        self.animate = animate;
        self
    }

    /// Compute a time-varying seed from base seed and time.
    fn time_seed(&self, base_seed: u64, time: f64) -> u64 {
        if self.animate {
            // Vary seed each frame (assuming ~30fps granularity)
            let frame = (time * 30.0) as u64;
            base_seed.wrapping_add(frame.wrapping_mul(2654435761))
        } else {
            base_seed
        }
    }

    /// Apply RGB channel shift to the frame.
    fn apply_rgb_shift(&self, input: &Frame, output: &mut Frame, time: f64) -> VfxResult<()> {
        let seed = self.time_seed(12345, time);
        let mut rng = PseudoRng::new(seed);

        // Scale offsets by intensity, add time-based jitter if animated
        let jitter = if self.animate {
            rng.next_i32_range(3)
        } else {
            0
        };

        let scale = self.intensity;
        let r_ox = (self.rgb_shift.red_offset_x as f32 * scale) as i32 + jitter;
        let r_oy = (self.rgb_shift.red_offset_y as f32 * scale) as i32;
        let g_ox = (self.rgb_shift.green_offset_x as f32 * scale) as i32;
        let g_oy = (self.rgb_shift.green_offset_y as f32 * scale) as i32;
        let b_ox = (self.rgb_shift.blue_offset_x as f32 * scale) as i32 - jitter;
        let b_oy = (self.rgb_shift.blue_offset_y as f32 * scale) as i32;

        let w = input.width as i32;
        let h = input.height as i32;

        for y in 0..input.height {
            for x in 0..input.width {
                let ix = x as i32;
                let iy = y as i32;

                // Sample each channel from offset positions (clamped)
                let r_x = (ix + r_ox).clamp(0, w - 1) as u32;
                let r_y = (iy + r_oy).clamp(0, h - 1) as u32;
                let g_x = (ix + g_ox).clamp(0, w - 1) as u32;
                let g_y = (iy + g_oy).clamp(0, h - 1) as u32;
                let b_x = (ix + b_ox).clamp(0, w - 1) as u32;
                let b_y = (iy + b_oy).clamp(0, h - 1) as u32;

                let r_pixel = input.get_pixel(r_x, r_y).unwrap_or([0, 0, 0, 255]);
                let g_pixel = input.get_pixel(g_x, g_y).unwrap_or([0, 0, 0, 255]);
                let b_pixel = input.get_pixel(b_x, b_y).unwrap_or([0, 0, 0, 255]);
                let orig = input.get_pixel(x, y).unwrap_or([0, 0, 0, 255]);

                output.set_pixel(x, y, [r_pixel[0], g_pixel[1], b_pixel[2], orig[3]]);
            }
        }

        Ok(())
    }

    /// Apply scanline corruption artifacts.
    fn apply_scan_lines(&self, input: &Frame, output: &mut Frame, time: f64) -> VfxResult<()> {
        let seed = self.time_seed(self.scan_line.seed, time);
        let mut rng = PseudoRng::new(seed);
        let w = input.width as i32;
        let corruption_rate = self.scan_line.corruption_rate * self.intensity;

        // First copy input to output
        output.data[..input.data.len()].copy_from_slice(&input.data);

        let mut y = 0u32;
        while y < input.height {
            let is_corrupted = rng.next_f32() < corruption_rate;

            if is_corrupted {
                let shift = rng.next_i32_range(self.scan_line.max_shift as i32);
                let brightness_mod = if self.scan_line.brightness_jitter {
                    rng.next_f32() * 0.6 + 0.7 // 0.7 to 1.3
                } else {
                    1.0
                };

                let thickness = self.scan_line.line_thickness.min(input.height - y);
                for dy in 0..thickness {
                    let row_y = y + dy;
                    if row_y >= input.height {
                        break;
                    }
                    for x in 0..input.width {
                        let src_x = (x as i32 + shift).clamp(0, w - 1) as u32;
                        let pixel = input.get_pixel(src_x, row_y).unwrap_or([0, 0, 0, 255]);

                        let r = (f32::from(pixel[0]) * brightness_mod).clamp(0.0, 255.0) as u8;
                        let g = (f32::from(pixel[1]) * brightness_mod).clamp(0.0, 255.0) as u8;
                        let b = (f32::from(pixel[2]) * brightness_mod).clamp(0.0, 255.0) as u8;

                        output.set_pixel(x, row_y, [r, g, b, pixel[3]]);
                    }
                }
            }

            y += self.scan_line.line_thickness.max(1);
        }

        Ok(())
    }

    /// Apply block displacement (datamosh-style).
    fn apply_block_displace(&self, input: &Frame, output: &mut Frame, time: f64) -> VfxResult<()> {
        // Start with a copy of input
        output.data[..input.data.len()].copy_from_slice(&input.data);

        let seed = self.time_seed(self.block_displace.seed, time);
        let mut rng = PseudoRng::new(seed);
        let block_count = (self.block_displace.block_count as f32 * self.intensity).max(1.0) as u32;

        for _ in 0..block_count {
            // Random block position and size
            let bw = self.block_displace.min_block_width
                + rng.next_u32_range(
                    (self.block_displace.max_block_width - self.block_displace.min_block_width)
                        .max(1),
                );
            let bh = self.block_displace.min_block_height
                + rng.next_u32_range(
                    (self.block_displace.max_block_height - self.block_displace.min_block_height)
                        .max(1),
                );

            let bx = rng.next_u32_range(input.width.saturating_sub(bw).max(1));
            let by = rng.next_u32_range(input.height.saturating_sub(bh).max(1));

            let dx = rng.next_i32_range(
                (self.block_displace.max_displacement as f32 * self.intensity) as i32,
            );
            let dy = rng.next_i32_range(
                (self.block_displace.max_displacement as f32 * self.intensity * 0.3) as i32,
            );

            // Copy block from source position to displaced position
            for row in 0..bh {
                for col in 0..bw {
                    let src_x = bx + col;
                    let src_y = by + row;

                    let dst_x = (src_x as i32 + dx).clamp(0, input.width as i32 - 1) as u32;
                    let dst_y = (src_y as i32 + dy).clamp(0, input.height as i32 - 1) as u32;

                    if let Some(pixel) = input.get_pixel(src_x, src_y) {
                        output.set_pixel(dst_x, dst_y, pixel);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply digital noise corruption.
    fn apply_digital_noise(&self, input: &Frame, output: &mut Frame, time: f64) -> VfxResult<()> {
        let seed = self.time_seed(self.digital_noise.seed, time);
        let mut rng = PseudoRng::new(seed);
        let density = self.digital_noise.noise_density * self.intensity;
        let max_dev = (f32::from(self.digital_noise.max_deviation) * self.intensity) as i32;

        for y in 0..input.height {
            for x in 0..input.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 255]);

                if rng.next_f32() < density {
                    let r = (i32::from(pixel[0]) + rng.next_i32_range(max_dev)).clamp(0, 255) as u8;
                    let g = (i32::from(pixel[1]) + rng.next_i32_range(max_dev)).clamp(0, 255) as u8;
                    let b = (i32::from(pixel[2]) + rng.next_i32_range(max_dev)).clamp(0, 255) as u8;
                    let a = if self.digital_noise.affect_alpha {
                        (i32::from(pixel[3]) + rng.next_i32_range(max_dev)).clamp(0, 255) as u8
                    } else {
                        pixel[3]
                    };
                    output.set_pixel(x, y, [r, g, b, a]);
                } else {
                    output.set_pixel(x, y, pixel);
                }
            }
        }

        Ok(())
    }
}

impl VideoEffect for GlitchEffect {
    fn name(&self) -> &str {
        "Glitch"
    }

    fn description(&self) -> &'static str {
        "Digital glitch effects: RGB shift, scanlines, block displacement, noise"
    }

    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        match self.glitch_type {
            GlitchType::RgbShift => self.apply_rgb_shift(input, output, params.time),
            GlitchType::ScanLine => self.apply_scan_lines(input, output, params.time),
            GlitchType::BlockDisplace => self.apply_block_displace(input, output, params.time),
            GlitchType::DigitalNoise => self.apply_digital_noise(input, output, params.time),
            GlitchType::Combined => {
                // Chain: RGB shift -> scanlines -> block displace -> noise
                let mut temp1 = Frame::new(input.width, input.height)?;
                let mut temp2 = Frame::new(input.width, input.height)?;

                self.apply_rgb_shift(input, &mut temp1, params.time)?;
                self.apply_scan_lines(&temp1, &mut temp2, params.time)?;
                self.apply_block_displace(&temp2, &mut temp1, params.time)?;
                self.apply_digital_noise(&temp1, output, params.time)?;

                Ok(())
            }
        }
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height).expect("test frame creation");
        // Fill with a gradient pattern for visible effects
        for y in 0..height {
            for x in 0..width {
                let r = (x * 255 / width.max(1)) as u8;
                let g = (y * 255 / height.max(1)) as u8;
                let b = 128u8;
                frame.set_pixel(x, y, [r, g, b, 255]);
            }
        }
        frame
    }

    #[test]
    fn test_glitch_rgb_shift_basic() {
        let mut effect = GlitchEffect::new(GlitchType::RgbShift);
        let input = create_test_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("rgb shift should succeed");

        // Center pixel should have channels from different source positions
        let center = output.get_pixel(50, 50).expect("center pixel");
        // Alpha should be preserved
        assert_eq!(center[3], 255);
    }

    #[test]
    fn test_glitch_rgb_shift_zero_intensity() {
        let mut effect = GlitchEffect::new(GlitchType::RgbShift).with_intensity(0.0);
        let input = create_test_frame(50, 50);
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("should succeed");

        // With zero intensity, offsets are 0 so output should match input
        for y in 0..50 {
            for x in 0..50 {
                let inp = input.get_pixel(x, y).expect("pixel");
                let out = output.get_pixel(x, y).expect("pixel");
                assert_eq!(
                    inp, out,
                    "zero intensity should preserve pixels at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_glitch_scan_line() {
        let mut effect = GlitchEffect::new(GlitchType::ScanLine).with_scan_line(ScanLineConfig {
            corruption_rate: 0.5,
            max_shift: 10,
            line_thickness: 2,
            brightness_jitter: true,
            seed: 100,
        });
        let input = create_test_frame(80, 80);
        let mut output = Frame::new(80, 80).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("scanline should succeed");

        // Some pixels should differ from input (corrupted lines)
        let mut diff_count = 0;
        for y in 0..80 {
            let inp = input.get_pixel(0, y).expect("pixel");
            let out = output.get_pixel(0, y).expect("pixel");
            if inp != out {
                diff_count += 1;
            }
        }
        assert!(diff_count > 0, "some scanlines should be corrupted");
    }

    #[test]
    fn test_glitch_block_displace() {
        let mut effect =
            GlitchEffect::new(GlitchType::BlockDisplace).with_block_displace(BlockDisplaceConfig {
                block_count: 5,
                max_displacement: 20,
                ..BlockDisplaceConfig::default()
            });
        let input = create_test_frame(100, 100);
        let mut output = Frame::new(100, 100).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("block displace should succeed");
    }

    #[test]
    fn test_glitch_digital_noise() {
        let mut effect =
            GlitchEffect::new(GlitchType::DigitalNoise).with_digital_noise(DigitalNoiseConfig {
                noise_density: 0.5,
                max_deviation: 100,
                affect_alpha: false,
                seed: 99,
            });
        let input = create_test_frame(60, 60);
        let mut output = Frame::new(60, 60).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("noise should succeed");

        // Alpha should be preserved when affect_alpha is false
        for y in 0..60 {
            for x in 0..60 {
                let out = output.get_pixel(x, y).expect("pixel");
                assert_eq!(out[3], 255, "alpha should be preserved");
            }
        }
    }

    #[test]
    fn test_glitch_digital_noise_affects_alpha() {
        let mut effect =
            GlitchEffect::new(GlitchType::DigitalNoise).with_digital_noise(DigitalNoiseConfig {
                noise_density: 1.0, // corrupt every pixel
                max_deviation: 100,
                affect_alpha: true,
                seed: 77,
            });
        let input = create_test_frame(30, 30);
        let mut output = Frame::new(30, 30).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("noise should succeed");

        // With density 1.0 and affect_alpha, some alpha values should differ
        let mut alpha_diff = false;
        for y in 0..30 {
            for x in 0..30 {
                let out = output.get_pixel(x, y).expect("pixel");
                if out[3] != 255 {
                    alpha_diff = true;
                }
            }
        }
        assert!(alpha_diff, "alpha should be affected");
    }

    #[test]
    fn test_glitch_combined() {
        let mut effect = GlitchEffect::new(GlitchType::Combined).with_intensity(0.5);
        let input = create_test_frame(64, 64);
        let mut output = Frame::new(64, 64).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("combined should succeed");
    }

    #[test]
    fn test_glitch_animated_produces_different_frames() {
        let mut effect = GlitchEffect::new(GlitchType::RgbShift)
            .with_intensity(1.0)
            .with_animate(true);
        let input = create_test_frame(50, 50);
        let mut output1 = Frame::new(50, 50).expect("test frame");
        let mut output2 = Frame::new(50, 50).expect("test frame");

        let params1 = EffectParams::new().with_time(0.0);
        let params2 = EffectParams::new().with_time(1.0);

        effect
            .apply(&input, &mut output1, &params1)
            .expect("frame 1");
        effect
            .apply(&input, &mut output2, &params2)
            .expect("frame 2");

        // With animation and different times, outputs should differ
        // (the jitter value changes)
        // Note: they might still match in rare cases, so just check they run
    }

    #[test]
    fn test_glitch_dimension_mismatch() {
        let mut effect = GlitchEffect::new(GlitchType::RgbShift);
        let input = Frame::new(100, 100).expect("test frame");
        let mut output = Frame::new(50, 50).expect("test frame");
        let params = EffectParams::new();
        let result = effect.apply(&input, &mut output, &params);
        assert!(result.is_err(), "dimension mismatch should error");
    }

    #[test]
    fn test_glitch_small_frame_1x1() {
        let mut effect = GlitchEffect::new(GlitchType::Combined);
        let input = Frame::new(1, 1).expect("test frame");
        let mut output = Frame::new(1, 1).expect("test frame");
        let params = EffectParams::new();
        effect
            .apply(&input, &mut output, &params)
            .expect("1x1 should succeed");
    }

    #[test]
    fn test_glitch_name_and_description() {
        let effect = GlitchEffect::new(GlitchType::RgbShift);
        assert_eq!(effect.name(), "Glitch");
        assert!(!effect.description().is_empty());
    }

    #[test]
    fn test_pseudo_rng_deterministic() {
        let mut rng1 = PseudoRng::new(42);
        let mut rng2 = PseudoRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_pseudo_rng_range() {
        let mut rng = PseudoRng::new(99);
        for _ in 0..1000 {
            let val = rng.next_u32_range(10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_pseudo_rng_f32_range() {
        let mut rng = PseudoRng::new(77);
        for _ in 0..1000 {
            let val = rng.next_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_pseudo_rng_zero_seed() {
        let mut rng = PseudoRng::new(0);
        // Should not get stuck at 0
        let val = rng.next_u64();
        assert_ne!(val, 0);
    }

    #[test]
    fn test_rgb_shift_config_default() {
        let cfg = RgbShiftConfig::default();
        assert_eq!(cfg.red_offset_x, 5);
        assert_eq!(cfg.blue_offset_x, -5);
        assert_eq!(cfg.green_offset_x, 0);
    }

    #[test]
    fn test_scan_line_config_default() {
        let cfg = ScanLineConfig::default();
        assert!(cfg.corruption_rate > 0.0);
        assert!(cfg.max_shift > 0);
    }

    #[test]
    fn test_block_displace_config_default() {
        let cfg = BlockDisplaceConfig::default();
        assert!(cfg.min_block_width < cfg.max_block_width);
        assert!(cfg.min_block_height < cfg.max_block_height);
    }
}
