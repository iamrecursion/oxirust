#![allow(dead_code)]
//! Chroma key (green/blue screen) compositing for broadcast graphics.
//!
//! Provides production-quality chroma keying with:
//! - Color-distance keying in YCbCr space for perceptually accurate results
//! - Spill suppression to neutralize color contamination on foreground subjects
//! - Soft-edge alpha ramp for smooth transitions (inner/outer tolerance)
//! - Despill modes: average, desaturate, and complement replacement
//! - Per-pixel alpha matte generation for downstream compositing

use crate::error::{GraphicsError, Result};

/// Color space used for key distance computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyColorSpace {
    /// Compute distance in RGB space (simple but perceptually uneven).
    Rgb,
    /// Compute distance in YCbCr space (better perceptual uniformity).
    YCbCr,
    /// Compute distance in CbCr plane only (ignores luminance, best for green/blue screens).
    CbCr,
}

impl Default for KeyColorSpace {
    fn default() -> Self {
        Self::CbCr
    }
}

/// Strategy for suppressing color spill on foreground subjects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DespillMode {
    /// No spill suppression applied.
    None,
    /// Replace the key channel with the average of the other two channels.
    Average,
    /// Desaturate pixels proportional to their proximity to the key color.
    Desaturate,
    /// Replace spill with the complement color (e.g. green spill -> magenta tint).
    Complement,
}

impl Default for DespillMode {
    fn default() -> Self {
        Self::Average
    }
}

/// Configuration for chroma key operation.
#[derive(Clone, Debug)]
pub struct ChromaKeyConfig {
    /// Key color in RGB [0..255].
    pub key_color: [u8; 3],
    /// Inner tolerance: pixels closer than this are fully transparent.
    /// Range: 0.0..=1.0 (normalized distance).
    pub inner_tolerance: f32,
    /// Outer tolerance: pixels farther than this are fully opaque.
    /// Range: 0.0..=1.0 (must be >= inner_tolerance).
    pub outer_tolerance: f32,
    /// Color space for distance computation.
    pub color_space: KeyColorSpace,
    /// Despill mode.
    pub despill_mode: DespillMode,
    /// Despill strength (0.0 = no despill, 1.0 = full despill).
    pub despill_strength: f32,
    /// Edge softness: additional blur passes on the alpha matte.
    pub edge_softness: u32,
}

impl Default for ChromaKeyConfig {
    fn default() -> Self {
        Self {
            key_color: [0, 255, 0], // Default green screen
            inner_tolerance: 0.15,
            outer_tolerance: 0.40,
            color_space: KeyColorSpace::default(),
            despill_mode: DespillMode::default(),
            despill_strength: 0.8,
            edge_softness: 0,
        }
    }
}

impl ChromaKeyConfig {
    /// Create a config for green screen keying.
    pub fn green_screen() -> Self {
        Self::default()
    }

    /// Create a config for blue screen keying.
    pub fn blue_screen() -> Self {
        Self {
            key_color: [0, 0, 255],
            ..Self::default()
        }
    }

    /// Set the key color.
    pub fn with_key_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.key_color = [r, g, b];
        self
    }

    /// Set the inner tolerance (fully transparent threshold).
    pub fn with_inner_tolerance(mut self, t: f32) -> Self {
        self.inner_tolerance = t.clamp(0.0, 1.0);
        self
    }

    /// Set the outer tolerance (fully opaque threshold).
    pub fn with_outer_tolerance(mut self, t: f32) -> Self {
        self.outer_tolerance = t.clamp(0.0, 1.0);
        self
    }

    /// Set the color space for distance computation.
    pub fn with_color_space(mut self, cs: KeyColorSpace) -> Self {
        self.color_space = cs;
        self
    }

    /// Set the despill mode.
    pub fn with_despill_mode(mut self, mode: DespillMode) -> Self {
        self.despill_mode = mode;
        self
    }

    /// Set the despill strength.
    pub fn with_despill_strength(mut self, s: f32) -> Self {
        self.despill_strength = s.clamp(0.0, 1.0);
        self
    }

    /// Set the edge softness (number of blur passes).
    pub fn with_edge_softness(mut self, passes: u32) -> Self {
        self.edge_softness = passes;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.inner_tolerance > self.outer_tolerance {
            return Err(GraphicsError::InvalidParameter(
                "inner_tolerance must be <= outer_tolerance".to_string(),
            ));
        }
        if self.outer_tolerance <= 0.0 {
            return Err(GraphicsError::InvalidParameter(
                "outer_tolerance must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Chroma key processor that generates alpha mattes and applies spill suppression.
pub struct ChromaKeyer {
    config: ChromaKeyConfig,
    /// Pre-computed key color in YCbCr space.
    key_ycbcr: (f32, f32, f32),
}

impl ChromaKeyer {
    /// Create a new chroma keyer with the given configuration.
    pub fn new(config: ChromaKeyConfig) -> Result<Self> {
        config.validate()?;
        let key_ycbcr = rgb_to_ycbcr(
            config.key_color[0],
            config.key_color[1],
            config.key_color[2],
        );
        Ok(Self { config, key_ycbcr })
    }

    /// Generate an alpha matte for the given RGBA frame.
    ///
    /// Returns a `Vec<f32>` of length `width * height`, each value in [0.0, 1.0].
    /// 0.0 = fully keyed (transparent), 1.0 = fully opaque (foreground).
    pub fn generate_matte(&self, frame: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        let expected = (width as usize) * (height as usize) * 4;
        if frame.len() != expected {
            return Err(GraphicsError::InvalidParameter(format!(
                "Frame size mismatch: expected {expected}, got {}",
                frame.len()
            )));
        }

        let pixel_count = (width as usize) * (height as usize);
        let mut matte = Vec::with_capacity(pixel_count);

        for pixel in frame.chunks_exact(4) {
            let dist = self.color_distance(pixel[0], pixel[1], pixel[2]);
            let alpha = self.distance_to_alpha(dist);
            matte.push(alpha);
        }

        // Apply edge softness via box blur passes on the matte
        let mut result = matte;
        for _ in 0..self.config.edge_softness {
            result = box_blur_matte(&result, width, height);
        }

        Ok(result)
    }

    /// Apply chroma key in-place on an RGBA frame.
    ///
    /// Modifies alpha channel based on key distance, and applies spill suppression
    /// to the RGB channels.
    pub fn apply(&self, frame: &mut [u8], width: u32, height: u32) -> Result<()> {
        let matte = self.generate_matte(frame, width, height)?;

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let alpha = matte[i];

            // Set alpha channel
            let original_alpha = f32::from(pixel[3]) / 255.0;
            pixel[3] = (alpha * original_alpha * 255.0).clamp(0.0, 255.0) as u8;

            // Apply spill suppression on foreground pixels
            if alpha > 0.01 {
                self.apply_despill(pixel, alpha);
            }
        }

        Ok(())
    }

    /// Apply chroma key and composite foreground over a background frame.
    ///
    /// Both frames must be RGBA and the same dimensions.
    pub fn composite(
        &self,
        foreground: &mut [u8],
        background: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let expected = (width as usize) * (height as usize) * 4;
        if foreground.len() != expected || background.len() != expected {
            return Err(GraphicsError::InvalidParameter(
                "Frame size mismatch".to_string(),
            ));
        }

        let matte = self.generate_matte(foreground, width, height)?;

        for (i, fg_pixel) in foreground.chunks_exact_mut(4).enumerate() {
            let alpha = matte[i];
            let bg_idx = i * 4;

            // Despill the foreground
            if alpha > 0.01 {
                self.apply_despill(fg_pixel, alpha);
            }

            // Alpha-over compositing
            let inv_alpha = 1.0 - alpha;
            fg_pixel[0] = (f32::from(fg_pixel[0]) * alpha
                + f32::from(background[bg_idx]) * inv_alpha)
                .clamp(0.0, 255.0) as u8;
            fg_pixel[1] = (f32::from(fg_pixel[1]) * alpha
                + f32::from(background[bg_idx + 1]) * inv_alpha)
                .clamp(0.0, 255.0) as u8;
            fg_pixel[2] = (f32::from(fg_pixel[2]) * alpha
                + f32::from(background[bg_idx + 2]) * inv_alpha)
                .clamp(0.0, 255.0) as u8;
            fg_pixel[3] = (alpha * 255.0 + f32::from(background[bg_idx + 3]) * inv_alpha)
                .clamp(0.0, 255.0) as u8;
        }

        Ok(())
    }

    /// Compute color distance between a pixel and the key color.
    /// Returns a normalized distance in [0.0, 1.0].
    fn color_distance(&self, r: u8, g: u8, b: u8) -> f32 {
        match self.config.color_space {
            KeyColorSpace::Rgb => {
                let dr = f32::from(r) - f32::from(self.config.key_color[0]);
                let dg = f32::from(g) - f32::from(self.config.key_color[1]);
                let db = f32::from(b) - f32::from(self.config.key_color[2]);
                let dist = (dr * dr + dg * dg + db * db).sqrt();
                // Max possible distance in RGB space is sqrt(3 * 255^2) ~ 441.67
                dist / 441.67
            }
            KeyColorSpace::YCbCr => {
                let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                let dy = y - self.key_ycbcr.0;
                let dcb = cb - self.key_ycbcr.1;
                let dcr = cr - self.key_ycbcr.2;
                let dist = (dy * dy + dcb * dcb + dcr * dcr).sqrt();
                // Normalize: max YCbCr distance ~ 362
                (dist / 362.0).min(1.0)
            }
            KeyColorSpace::CbCr => {
                let (_y, cb, cr) = rgb_to_ycbcr(r, g, b);
                let dcb = cb - self.key_ycbcr.1;
                let dcr = cr - self.key_ycbcr.2;
                let dist = (dcb * dcb + dcr * dcr).sqrt();
                // Normalize: max CbCr distance ~ 255
                (dist / 255.0).min(1.0)
            }
        }
    }

    /// Convert distance to alpha using inner/outer tolerance ramp.
    fn distance_to_alpha(&self, distance: f32) -> f32 {
        if distance <= self.config.inner_tolerance {
            0.0 // Fully keyed
        } else if distance >= self.config.outer_tolerance {
            1.0 // Fully opaque
        } else {
            // Smooth ramp between inner and outer
            let range = self.config.outer_tolerance - self.config.inner_tolerance;
            if range <= f32::EPSILON {
                return 1.0;
            }
            let t = (distance - self.config.inner_tolerance) / range;
            // Smoothstep for natural-looking edges
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Apply spill suppression to a pixel.
    fn apply_despill(&self, pixel: &mut [u8], alpha: f32) {
        if self.config.despill_strength <= 0.0 || self.config.despill_mode == DespillMode::None {
            return;
        }

        let r = f32::from(pixel[0]);
        let g = f32::from(pixel[1]);
        let b = f32::from(pixel[2]);

        // Determine the dominant key channel
        let key = &self.config.key_color;
        let key_channel = if key[1] >= key[0] && key[1] >= key[2] {
            1 // Green dominant
        } else if key[2] >= key[0] && key[2] >= key[1] {
            2 // Blue dominant
        } else {
            0 // Red dominant
        };

        let channels = [r, g, b];
        let key_val = channels[key_channel];

        // How much of the non-key channels
        let other1 = channels[(key_channel + 1) % 3];
        let other2 = channels[(key_channel + 2) % 3];

        let (new_r, new_g, new_b) = match self.config.despill_mode {
            DespillMode::None => return,
            DespillMode::Average => {
                let avg = (other1 + other2) / 2.0;
                let clamped = key_val.min(avg);
                let mut result = [r, g, b];
                let spill_amount = (key_val - clamped).max(0.0) * self.config.despill_strength;
                // Reduce spill proportional to how keyed the pixel is (1.0 - alpha means near key)
                let proximity = 1.0 - alpha;
                result[key_channel] = (key_val - spill_amount * proximity).max(0.0);
                (result[0], result[1], result[2])
            }
            DespillMode::Desaturate => {
                let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let proximity = 1.0 - alpha;
                let factor = self.config.despill_strength * proximity;
                let nr = r + (luminance - r) * factor;
                let ng = g + (luminance - g) * factor;
                let nb = b + (luminance - b) * factor;
                (nr, ng, nb)
            }
            DespillMode::Complement => {
                let avg = (other1 + other2) / 2.0;
                let spill = (key_val - avg).max(0.0);
                let proximity = 1.0 - alpha;
                let correction = spill * self.config.despill_strength * proximity;
                let mut result = [r, g, b];
                result[key_channel] = (key_val - correction).max(0.0);
                // Add complement tint to other channels
                let tint = correction * 0.3;
                result[(key_channel + 1) % 3] = (result[(key_channel + 1) % 3] + tint).min(255.0);
                result[(key_channel + 2) % 3] = (result[(key_channel + 2) % 3] + tint).min(255.0);
                (result[0], result[1], result[2])
            }
        };

        pixel[0] = new_r.clamp(0.0, 255.0) as u8;
        pixel[1] = new_g.clamp(0.0, 255.0) as u8;
        pixel[2] = new_b.clamp(0.0, 255.0) as u8;
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &ChromaKeyConfig {
        &self.config
    }

    /// Update the configuration (re-validates and re-computes internals).
    pub fn set_config(&mut self, config: ChromaKeyConfig) -> Result<()> {
        config.validate()?;
        self.key_ycbcr = rgb_to_ycbcr(
            config.key_color[0],
            config.key_color[1],
            config.key_color[2],
        );
        self.config = config;
        Ok(())
    }
}

/// Convert RGB to YCbCr (BT.709).
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = f32::from(r);
    let gf = f32::from(g);
    let bf = f32::from(b);
    let y = 16.0 + 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
    let cb = 128.0 + (-0.1146 * rf - 0.3854 * gf + 0.5000 * bf);
    let cr = 128.0 + (0.5000 * rf - 0.4542 * gf - 0.0458 * bf);
    (y, cb, cr)
}

/// Box blur a single-channel matte (f32 values).
fn box_blur_matte(matte: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let len = w * h;
    let mut temp = vec![0.0_f32; len];
    let mut result = vec![0.0_f32; len];

    // Horizontal pass (3-tap box)
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let left = if x > 0 { matte[idx - 1] } else { matte[idx] };
            let center = matte[idx];
            let right = if x + 1 < w {
                matte[idx + 1]
            } else {
                matte[idx]
            };
            temp[idx] = (left + center + right) / 3.0;
        }
    }

    // Vertical pass (3-tap box)
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let top = if y > 0 { temp[idx - w] } else { temp[idx] };
            let center = temp[idx];
            let bottom = if y + 1 < h { temp[idx + w] } else { temp[idx] };
            result[idx] = (top + center + bottom) / 3.0;
        }
    }

    result
}

/// Analyze a frame to find the dominant key color (useful for auto-detection).
///
/// Examines the border pixels to determine the most common color,
/// which is likely the backdrop.
pub fn detect_key_color(frame: &[u8], width: u32, height: u32) -> Result<[u8; 3]> {
    let expected = (width as usize) * (height as usize) * 4;
    if frame.len() != expected {
        return Err(GraphicsError::InvalidParameter(
            "Frame size mismatch".to_string(),
        ));
    }

    // Sample border pixels (top, bottom, left, right edges)
    let mut r_sum: u64 = 0;
    let mut g_sum: u64 = 0;
    let mut b_sum: u64 = 0;
    let mut count: u64 = 0;

    let w = width as usize;
    let h = height as usize;

    for x in 0..w {
        // Top row
        let idx = x * 4;
        r_sum += u64::from(frame[idx]);
        g_sum += u64::from(frame[idx + 1]);
        b_sum += u64::from(frame[idx + 2]);
        count += 1;

        // Bottom row
        let idx = ((h - 1) * w + x) * 4;
        r_sum += u64::from(frame[idx]);
        g_sum += u64::from(frame[idx + 1]);
        b_sum += u64::from(frame[idx + 2]);
        count += 1;
    }

    for y in 1..h.saturating_sub(1) {
        // Left column
        let idx = (y * w) * 4;
        r_sum += u64::from(frame[idx]);
        g_sum += u64::from(frame[idx + 1]);
        b_sum += u64::from(frame[idx + 2]);
        count += 1;

        // Right column
        let idx = (y * w + w - 1) * 4;
        r_sum += u64::from(frame[idx]);
        g_sum += u64::from(frame[idx + 1]);
        b_sum += u64::from(frame[idx + 2]);
        count += 1;
    }

    if count == 0 {
        return Err(GraphicsError::InvalidParameter(
            "Frame too small to detect key color".to_string(),
        ));
    }

    Ok([
        (r_sum / count) as u8,
        (g_sum / count) as u8,
        (b_sum / count) as u8,
    ])
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_frame(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let count = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(count * 4);
        for _ in 0..count {
            data.extend_from_slice(&[r, g, b, a]);
        }
        data
    }

    #[test]
    fn test_config_defaults() {
        let cfg = ChromaKeyConfig::default();
        assert_eq!(cfg.key_color, [0, 255, 0]);
        assert!(cfg.inner_tolerance < cfg.outer_tolerance);
    }

    #[test]
    fn test_config_green_screen() {
        let cfg = ChromaKeyConfig::green_screen();
        assert_eq!(cfg.key_color, [0, 255, 0]);
    }

    #[test]
    fn test_config_blue_screen() {
        let cfg = ChromaKeyConfig::blue_screen();
        assert_eq!(cfg.key_color, [0, 0, 255]);
    }

    #[test]
    fn test_config_validation_ok() {
        let cfg = ChromaKeyConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validation_inner_gt_outer() {
        let cfg = ChromaKeyConfig::default()
            .with_inner_tolerance(0.5)
            .with_outer_tolerance(0.3);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = ChromaKeyConfig::default()
            .with_key_color(10, 200, 30)
            .with_inner_tolerance(0.1)
            .with_outer_tolerance(0.5)
            .with_color_space(KeyColorSpace::Rgb)
            .with_despill_mode(DespillMode::Complement)
            .with_despill_strength(0.5)
            .with_edge_softness(2);
        assert_eq!(cfg.key_color, [10, 200, 30]);
        assert!((cfg.inner_tolerance - 0.1).abs() < f32::EPSILON);
        assert_eq!(cfg.color_space, KeyColorSpace::Rgb);
        assert_eq!(cfg.despill_mode, DespillMode::Complement);
        assert_eq!(cfg.edge_softness, 2);
    }

    #[test]
    fn test_keyer_creation() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::default());
        assert!(keyer.is_ok());
    }

    #[test]
    fn test_keyer_green_pixels_become_transparent() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let mut frame = make_solid_frame(10, 10, 0, 255, 0, 255);
        keyer
            .apply(&mut frame, 10, 10)
            .expect("apply should succeed");
        // All green pixels should be transparent (alpha = 0)
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 0, "Green pixel should be keyed out");
        }
    }

    #[test]
    fn test_keyer_non_key_pixels_remain_opaque() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let mut frame = make_solid_frame(10, 10, 255, 0, 0, 255); // Red
        keyer
            .apply(&mut frame, 10, 10)
            .expect("apply should succeed");
        // Red pixels should remain opaque
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "Red pixel should remain opaque");
        }
    }

    #[test]
    fn test_keyer_blue_screen() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::blue_screen()).expect("valid config");
        let mut frame = make_solid_frame(10, 10, 0, 0, 255, 255);
        keyer
            .apply(&mut frame, 10, 10)
            .expect("apply should succeed");
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 0, "Blue pixel should be keyed out");
        }
    }

    #[test]
    fn test_matte_generation() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let frame = make_solid_frame(5, 5, 0, 255, 0, 255);
        let matte = keyer
            .generate_matte(&frame, 5, 5)
            .expect("matte should succeed");
        assert_eq!(matte.len(), 25);
        for val in &matte {
            assert!(
                *val < 0.01,
                "Green pixels should have near-zero alpha in matte"
            );
        }
    }

    #[test]
    fn test_matte_mixed_frame() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let mut frame = Vec::with_capacity(8 * 4);
        // 4 green pixels, 4 red pixels
        for _ in 0..4 {
            frame.extend_from_slice(&[0, 255, 0, 255]);
        }
        for _ in 0..4 {
            frame.extend_from_slice(&[255, 0, 0, 255]);
        }
        let matte = keyer
            .generate_matte(&frame, 4, 2)
            .expect("matte should succeed");
        // Green pixels: low alpha
        for val in &matte[0..4] {
            assert!(*val < 0.01);
        }
        // Red pixels: high alpha
        for val in &matte[4..8] {
            assert!(*val > 0.99);
        }
    }

    #[test]
    fn test_composite() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let mut fg = make_solid_frame(4, 4, 0, 255, 0, 255); // Green foreground
        let bg = make_solid_frame(4, 4, 255, 0, 0, 255); // Red background
        keyer
            .composite(&mut fg, &bg, 4, 4)
            .expect("composite should succeed");
        // Green keyed out, should be mostly red background
        for pixel in fg.chunks_exact(4) {
            assert!(pixel[0] > 200, "Should be mostly red from background");
            assert!(pixel[1] < 50, "Green channel should be suppressed");
        }
    }

    #[test]
    fn test_frame_size_mismatch() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::default()).expect("valid config");
        let frame = vec![0u8; 10]; // Wrong size
        assert!(keyer.generate_matte(&frame, 10, 10).is_err());
    }

    #[test]
    fn test_rgb_color_space() {
        let cfg = ChromaKeyConfig::green_screen().with_color_space(KeyColorSpace::Rgb);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 0, 255, 0, 255);
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 0, "Green should be keyed in RGB space too");
        }
    }

    #[test]
    fn test_ycbcr_color_space() {
        let cfg = ChromaKeyConfig::green_screen().with_color_space(KeyColorSpace::YCbCr);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 0, 255, 0, 255);
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 0);
        }
    }

    #[test]
    fn test_edge_softness() {
        let cfg = ChromaKeyConfig::green_screen().with_edge_softness(3);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let frame = make_solid_frame(10, 10, 0, 255, 0, 255);
        let matte = keyer
            .generate_matte(&frame, 10, 10)
            .expect("matte should succeed");
        // After blur, uniform green matte should still be near zero
        for val in &matte {
            assert!(*val < 0.01);
        }
    }

    #[test]
    fn test_despill_none() {
        let cfg = ChromaKeyConfig::green_screen().with_despill_mode(DespillMode::None);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 100, 200, 50, 255); // Greenish
        let original = frame.clone();
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        // With DespillMode::None, RGB should not be modified for opaque pixels
        // (only alpha changes)
        for (orig, result) in original.chunks_exact(4).zip(frame.chunks_exact(4)) {
            if result[3] == 255 {
                assert_eq!(orig[0], result[0]);
                assert_eq!(orig[1], result[1]);
                assert_eq!(orig[2], result[2]);
            }
        }
    }

    #[test]
    fn test_despill_desaturate() {
        let cfg = ChromaKeyConfig::green_screen()
            .with_despill_mode(DespillMode::Desaturate)
            .with_inner_tolerance(0.05)
            .with_outer_tolerance(0.8);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 100, 180, 80, 255);
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        // Should reduce saturation on greenish pixels
        // The result channels should be closer together than the input
        for pixel in frame.chunks_exact(4) {
            if pixel[3] > 0 {
                let diff = (i16::from(pixel[1]) - i16::from(pixel[0])).unsigned_abs();
                // After desaturation, green channel should be closer to red
                assert!(
                    diff < 80,
                    "Channels should be more uniform after desaturation, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_despill_complement() {
        let cfg = ChromaKeyConfig::green_screen()
            .with_despill_mode(DespillMode::Complement)
            .with_inner_tolerance(0.05)
            .with_outer_tolerance(0.8);
        let keyer = ChromaKeyer::new(cfg).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 100, 200, 80, 255);
        let original_green = frame[1];
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        // Green channel should be reduced for spill
        for pixel in frame.chunks_exact(4) {
            if pixel[3] > 0 {
                assert!(
                    pixel[1] <= original_green,
                    "Green channel should be reduced"
                );
            }
        }
    }

    #[test]
    fn test_detect_key_color_green_frame() {
        let frame = make_solid_frame(20, 20, 0, 255, 0, 255);
        let key = detect_key_color(&frame, 20, 20).expect("detection should succeed");
        assert_eq!(key, [0, 255, 0]);
    }

    #[test]
    fn test_detect_key_color_blue_frame() {
        let frame = make_solid_frame(20, 20, 0, 0, 200, 255);
        let key = detect_key_color(&frame, 20, 20).expect("detection should succeed");
        assert_eq!(key, [0, 0, 200]);
    }

    #[test]
    fn test_detect_key_color_size_mismatch() {
        assert!(detect_key_color(&[0u8; 10], 10, 10).is_err());
    }

    #[test]
    fn test_set_config() {
        let mut keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let new_cfg = ChromaKeyConfig::blue_screen();
        assert!(keyer.set_config(new_cfg).is_ok());
        assert_eq!(keyer.config().key_color, [0, 0, 255]);
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert!((y - 16.0).abs() < 1.0);
        assert!((cb - 128.0).abs() < 1.0);
        assert!((cr - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, _cb, _cr) = rgb_to_ycbcr(255, 255, 255);
        assert!(y > 200.0); // Should be high luminance
    }

    #[test]
    fn test_box_blur_uniform() {
        let matte = vec![1.0_f32; 25];
        let result = box_blur_matte(&matte, 5, 5);
        for val in &result {
            assert!(
                (*val - 1.0).abs() < 0.01,
                "Uniform matte blur should stay ~1.0"
            );
        }
    }

    #[test]
    fn test_smoothstep_alpha_ramp() {
        let keyer = ChromaKeyer::new(
            ChromaKeyConfig::default()
                .with_inner_tolerance(0.2)
                .with_outer_tolerance(0.6),
        )
        .expect("valid config");

        let a0 = keyer.distance_to_alpha(0.1); // Below inner
        let a1 = keyer.distance_to_alpha(0.4); // Mid ramp
        let a2 = keyer.distance_to_alpha(0.7); // Above outer
        assert!(a0 < 0.01);
        assert!(a1 > 0.0 && a1 < 1.0);
        assert!(a2 > 0.99);
    }

    #[test]
    fn test_composite_size_mismatch() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::default()).expect("valid config");
        let mut fg = vec![0u8; 16];
        let bg = vec![0u8; 32];
        assert!(keyer.composite(&mut fg, &bg, 2, 2).is_err());
    }

    #[test]
    fn test_zero_alpha_foreground() {
        let keyer = ChromaKeyer::new(ChromaKeyConfig::green_screen()).expect("valid config");
        let mut frame = make_solid_frame(4, 4, 0, 255, 0, 0); // Green but alpha=0
        keyer.apply(&mut frame, 4, 4).expect("apply should succeed");
        for pixel in frame.chunks_exact(4) {
            assert_eq!(pixel[3], 0);
        }
    }
}
