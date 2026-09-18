//! Chroma key (green screen) removal with edge refinement.
//!
//! Implements a multi-pass chroma keying pipeline:
//! 1. **Core key**: YCbCr-space per-pixel distance from the key colour
//! 2. **Spill suppression**: reduces green tinge on foreground edges
//! 3. **Edge refinement**: morphological dilation/erosion based matte cleanup
//!    to produce soft, accurate alpha transitions at subject boundaries
//!
//! All operations work on raw RGBA pixel buffers (`[R, G, B, A]` byte layout).

use crate::{GamingError, GamingResult};

/// Chroma key configuration.
#[derive(Debug, Clone)]
pub struct ChromaKeyConfig {
    /// Key color (RGB) — the colour to make transparent.
    pub key_color: (u8, u8, u8),
    /// Similarity threshold (0.0–1.0).
    ///
    /// Pixels whose normalised colour distance from `key_color` falls below
    /// this value are considered fully keyed (alpha = 0).
    pub similarity: f32,
    /// Smoothness (0.0–1.0).
    ///
    /// The width of the soft transition zone around `similarity`. Larger values
    /// produce softer, less precise edges; smaller values yield sharper cuts.
    pub smoothness: f32,
    /// Spill reduction strength (0.0–1.0).
    ///
    /// After the core key, a fraction of the dominant key channel (green for a
    /// green screen) is subtracted from foreground pixels to remove colour
    /// bleed-through.
    pub spill_reduction: f32,
    /// Edge refinement passes (0 = disabled).
    ///
    /// Each pass runs a 3×3 erosion on bright regions of the alpha matte
    /// followed by a dilation to close small holes, improving boundary quality.
    pub edge_refinement_passes: u32,
}

impl Default for ChromaKeyConfig {
    fn default() -> Self {
        Self {
            key_color: (0, 255, 0), // Green screen
            similarity: 0.4,
            smoothness: 0.08,
            spill_reduction: 0.1,
            edge_refinement_passes: 1,
        }
    }
}

/// A processed frame output from the chroma key pipeline.
#[derive(Debug, Clone)]
pub struct KeyedFrame {
    /// RGBA pixel data with alpha channel set by the key operation.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Fraction of pixels that were fully keyed out (0.0–1.0).
    pub keyed_fraction: f32,
}

/// Chroma key processor with configurable pipeline stages.
pub struct ChromaKey {
    config: ChromaKeyConfig,
}

impl ChromaKey {
    /// Create a new chroma key processor.
    ///
    /// # Errors
    ///
    /// Returns error if configuration values are out of range.
    pub fn new(config: ChromaKeyConfig) -> GamingResult<Self> {
        if config.edge_refinement_passes > 8 {
            return Err(GamingError::InvalidConfig(
                "edge_refinement_passes must be 0-8".into(),
            ));
        }
        Ok(Self { config })
    }

    /// Create from config without validation (for ergonomic default usage).
    #[must_use]
    pub fn new_unchecked(config: ChromaKeyConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ChromaKeyConfig {
        &self.config
    }

    /// Apply chroma key to an RGBA frame buffer.
    ///
    /// `width` × `height` × 4 bytes are expected in `rgba`.  The function
    /// creates a new buffer; the input is not modified.
    ///
    /// # Errors
    ///
    /// Returns error if `rgba.len() != width * height * 4`.
    pub fn apply(&self, rgba: &[u8], width: u32, height: u32) -> GamingResult<KeyedFrame> {
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            return Err(GamingError::CaptureFailed(format!(
                "RGBA buffer length {} does not match {}×{}×4 = {}",
                rgba.len(),
                width,
                height,
                expected
            )));
        }

        let mut out = rgba.to_vec();

        // Stage 1: compute alpha for each pixel via YCbCr distance
        let (_ky, key_cb, key_cr) = rgb_to_ycbcr_float(
            self.config.key_color.0,
            self.config.key_color.1,
            self.config.key_color.2,
        );

        let mut keyed_count = 0usize;
        let pixel_count = (width as usize) * (height as usize);

        let similarity = self.config.similarity.clamp(0.0, 1.0);
        let smoothness = self.config.smoothness.clamp(0.0, 1.0);
        let hard_threshold = similarity;
        let soft_threshold = (similarity + smoothness).min(1.0);

        for i in 0..pixel_count {
            let base = i * 4;
            let (_yp, cbp, crp) = rgb_to_ycbcr_float(out[base], out[base + 1], out[base + 2]);

            // Chroma-only distance (ignore luma for better key performance)
            let dcb = cbp - key_cb;
            let dcr = crp - key_cr;

            // Normalised distance: 0 = identical chroma, 1 = far away
            let dist = ((dcb * dcb + dcr * dcr) * 0.5).sqrt().min(1.0);

            let alpha = if dist <= hard_threshold {
                keyed_count += 1;
                0u8
            } else if dist <= soft_threshold {
                // Soft transition zone
                let t = (dist - hard_threshold) / (soft_threshold - hard_threshold + f32::EPSILON);
                (t * 255.0) as u8
            } else {
                255u8
            };

            out[base + 3] = alpha;
        }

        // Stage 2: spill suppression on partially-foreground pixels
        if self.config.spill_reduction > 0.0 {
            self.apply_spill_suppression(&mut out, pixel_count);
        }

        // Stage 3: edge refinement (erode then dilate the matte)
        if self.config.edge_refinement_passes > 0 {
            let mut alpha_plane: Vec<f32> = (0..pixel_count)
                .map(|i| out[i * 4 + 3] as f32 / 255.0)
                .collect();

            for _ in 0..self.config.edge_refinement_passes {
                alpha_plane = erode_alpha(&alpha_plane, width, height);
                alpha_plane = dilate_alpha(&alpha_plane, width, height);
            }

            for i in 0..pixel_count {
                out[i * 4 + 3] = (alpha_plane[i] * 255.0) as u8;
            }
        }

        let keyed_fraction = keyed_count as f32 / pixel_count.max(1) as f32;

        Ok(KeyedFrame {
            data: out,
            width,
            height,
            keyed_fraction,
        })
    }

    /// Apply spill suppression: reduce the dominant key channel on pixels
    /// that are partially or fully keyed.
    #[allow(clippy::cast_possible_truncation)]
    fn apply_spill_suppression(&self, out: &mut [u8], pixel_count: usize) {
        // Determine dominant channel of the key colour
        let (kr, kg, kb) = (
            self.config.key_color.0,
            self.config.key_color.1,
            self.config.key_color.2,
        );
        let dominant = if kr >= kg && kr >= kb {
            0usize
        } else if kg >= kb {
            1
        } else {
            2
        };

        let strength = self.config.spill_reduction;
        for i in 0..pixel_count {
            let base = i * 4;
            let alpha = out[base + 3] as f32 / 255.0;
            // Only suppress on pixels with partial key (transition zone)
            if alpha < 1.0 {
                let ch = out[base + dominant] as f32;
                let suppressed = (ch * (1.0 - strength * (1.0 - alpha))).min(255.0) as u8;
                out[base + dominant] = suppressed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Colour space helpers
// ---------------------------------------------------------------------------

/// Convert an sRGB (0..255) pixel to a (Y, Cb, Cr) tuple in the range 0..1.
#[allow(clippy::cast_lossless)]
fn rgb_to_ycbcr_float(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = -0.16874 * rf - 0.33126 * gf + 0.5 * bf + 0.5;
    let cr = 0.5 * rf - 0.41869 * gf - 0.08131 * bf + 0.5;
    (y, cb, cr)
}

// ---------------------------------------------------------------------------
// Morphological operations on an alpha plane (values 0.0–1.0)
// ---------------------------------------------------------------------------

/// 3×3 morphological erosion on a single-channel (alpha) plane.
///
/// Each output pixel is the **minimum** of the 3×3 neighbourhood — this
/// shrinks bright regions and removes small specks in the foreground matte.
fn erode_alpha(alpha: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0.0_f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut min_val = 1.0_f32;
            for dy in 0..3 {
                let ny = (y + dy).wrapping_sub(1);
                if ny >= h {
                    continue;
                }
                for dx in 0..3 {
                    let nx = (x + dx).wrapping_sub(1);
                    if nx >= w {
                        continue;
                    }
                    let v = alpha[ny * w + nx];
                    if v < min_val {
                        min_val = v;
                    }
                }
            }
            out[y * w + x] = min_val;
        }
    }
    out
}

/// 3×3 morphological dilation on a single-channel (alpha) plane.
///
/// Each output pixel is the **maximum** of the 3×3 neighbourhood — this
/// expands bright regions and fills small holes in the foreground matte.
fn dilate_alpha(alpha: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0.0_f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut max_val = 0.0_f32;
            for dy in 0..3 {
                let ny = (y + dy).wrapping_sub(1);
                if ny >= h {
                    continue;
                }
                for dx in 0..3 {
                    let nx = (x + dx).wrapping_sub(1);
                    if nx >= w {
                        continue;
                    }
                    let v = alpha[ny * w + nx];
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            out[y * w + x] = max_val;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut buf = vec![0u8; n * 4];
        for i in 0..n {
            buf[i * 4] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = 255;
        }
        buf
    }

    #[test]
    fn test_default_config() {
        let cfg = ChromaKeyConfig::default();
        assert_eq!(cfg.key_color, (0, 255, 0));
        assert!(cfg.similarity > 0.0);
    }

    #[test]
    fn test_chroma_key_creation() {
        let key = ChromaKey::new(ChromaKeyConfig::default()).expect("valid config");
        assert_eq!(key.config().key_color, (0, 255, 0));
    }

    #[test]
    fn test_invalid_refinement_passes() {
        let mut cfg = ChromaKeyConfig::default();
        cfg.edge_refinement_passes = 9;
        assert!(ChromaKey::new(cfg).is_err());
    }

    #[test]
    fn test_pure_green_fully_keyed() {
        // A frame of pure green (0, 255, 0) should be almost fully keyed out.
        let buf = solid_rgba(0, 255, 0, 8, 8);
        let key = ChromaKey::new_unchecked(ChromaKeyConfig::default());
        let result = key.apply(&buf, 8, 8).expect("apply should succeed");

        // Most pixels should have low or zero alpha
        let avg_alpha: f32 = result.data.chunks(4).map(|p| p[3] as f32).sum::<f32>() / 64.0;
        assert!(avg_alpha < 50.0, "avg alpha {avg_alpha} should be < 50");
        assert!(result.keyed_fraction > 0.5);
    }

    #[test]
    fn test_pure_red_not_keyed() {
        // A frame of pure red should be retained (high alpha).
        let buf = solid_rgba(255, 0, 0, 8, 8);
        let key = ChromaKey::new_unchecked(ChromaKeyConfig::default());
        let result = key.apply(&buf, 8, 8).expect("apply should succeed");

        let avg_alpha: f32 = result.data.chunks(4).map(|p| p[3] as f32).sum::<f32>() / 64.0;
        assert!(
            avg_alpha > 200.0,
            "avg alpha {avg_alpha} should be > 200 for red pixels"
        );
    }

    #[test]
    fn test_wrong_buffer_size_returns_error() {
        let key = ChromaKey::new_unchecked(ChromaKeyConfig::default());
        let bad_buf = vec![0u8; 10]; // too small
        assert!(key.apply(&bad_buf, 4, 4).is_err());
    }

    #[test]
    fn test_output_size_matches_input() {
        let buf = solid_rgba(128, 128, 128, 16, 16);
        let key = ChromaKey::new_unchecked(ChromaKeyConfig::default());
        let result = key.apply(&buf, 16, 16).expect("apply should succeed");
        assert_eq!(result.data.len(), 16 * 16 * 4);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_edge_refinement_disabled() {
        let mut cfg = ChromaKeyConfig::default();
        cfg.edge_refinement_passes = 0;
        let buf = solid_rgba(0, 255, 0, 8, 8);
        let key = ChromaKey::new_unchecked(cfg);
        let result = key.apply(&buf, 8, 8).expect("apply");
        // Should still key green pixels without refinement pass
        assert!(result.keyed_fraction > 0.0);
    }

    #[test]
    fn test_spill_suppression_disabled() {
        let mut cfg = ChromaKeyConfig::default();
        cfg.spill_reduction = 0.0;
        let buf = solid_rgba(50, 200, 50, 8, 8); // greenish foreground
        let key = ChromaKey::new_unchecked(cfg);
        let result = key.apply(&buf, 8, 8).expect("apply");
        assert_eq!(result.data.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_blue_screen_config() {
        let cfg = ChromaKeyConfig {
            key_color: (0, 0, 255), // blue screen
            similarity: 0.4,
            smoothness: 0.08,
            spill_reduction: 0.1,
            edge_refinement_passes: 1,
        };
        let buf = solid_rgba(0, 0, 255, 8, 8);
        let key = ChromaKey::new_unchecked(cfg);
        let result = key.apply(&buf, 8, 8).expect("apply");
        let avg_alpha: f32 = result.data.chunks(4).map(|p| p[3] as f32).sum::<f32>() / 64.0;
        assert!(
            avg_alpha < 100.0,
            "blue screen: avg alpha {avg_alpha} should be low"
        );
    }

    #[test]
    fn test_multiple_refinement_passes() {
        let mut cfg = ChromaKeyConfig::default();
        cfg.edge_refinement_passes = 3;
        let buf = solid_rgba(128, 200, 128, 16, 16);
        let key = ChromaKey::new(cfg).expect("valid");
        let result = key.apply(&buf, 16, 16).expect("apply");
        assert_eq!(result.data.len(), 16 * 16 * 4);
    }

    #[test]
    fn test_high_similarity_keys_more() {
        let low_cfg = ChromaKeyConfig {
            similarity: 0.1,
            ..ChromaKeyConfig::default()
        };
        let high_cfg = ChromaKeyConfig {
            similarity: 0.8,
            ..ChromaKeyConfig::default()
        };
        // Slightly-off-green that might not be keyed with low similarity
        let buf = solid_rgba(30, 200, 30, 8, 8);
        let key_low = ChromaKey::new_unchecked(low_cfg);
        let key_high = ChromaKey::new_unchecked(high_cfg);
        let r_low = key_low.apply(&buf, 8, 8).expect("apply low");
        let r_high = key_high.apply(&buf, 8, 8).expect("apply high");
        // Higher similarity should key MORE pixels (lower average alpha)
        let avg_low = r_low.data.chunks(4).map(|p| p[3] as f32).sum::<f32>() / 64.0;
        let avg_high = r_high.data.chunks(4).map(|p| p[3] as f32).sum::<f32>() / 64.0;
        assert!(
            avg_high <= avg_low + 1.0,
            "high similarity ({avg_high}) should key at least as much as low ({avg_low})"
        );
    }

    #[test]
    fn test_rgb_to_ycbcr_float_black() {
        let (y, cb, cr) = rgb_to_ycbcr_float(0, 0, 0);
        assert!(y.abs() < 0.01);
        // Cb and Cr should be near 0.5 for black
        assert!((cb - 0.5).abs() < 0.01);
        assert!((cr - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_erode_dilate_preserves_size() {
        let alpha = vec![0.5_f32; 16];
        let eroded = erode_alpha(&alpha, 4, 4);
        assert_eq!(eroded.len(), 16);
        let dilated = dilate_alpha(&alpha, 4, 4);
        assert_eq!(dilated.len(), 16);
    }

    #[test]
    fn test_erode_uniform_unchanged() {
        let alpha = vec![0.8_f32; 25];
        let eroded = erode_alpha(&alpha, 5, 5);
        for &v in &eroded {
            assert!((v - 0.8).abs() < 0.01);
        }
    }

    #[test]
    fn test_dilate_uniform_unchanged() {
        let alpha = vec![0.3_f32; 25];
        let dilated = dilate_alpha(&alpha, 5, 5);
        for &v in &dilated {
            assert!((v - 0.3).abs() < 0.01);
        }
    }
}
