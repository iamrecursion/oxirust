//! Fragile watermarking for tamper detection.
//!
//! A fragile watermark is destroyed by any modification of the content —
//! unlike a robust watermark which survives transformations.  This module
//! uses a PRNG-driven pattern to set the LSB of selected luma samples.
//! Any block that has been modified will show mismatched LSBs.

/// A fragile watermark descriptor.
///
/// `seed` initialises the PRNG used to generate the embedding pattern so
/// the same pattern can be reproduced at detection time.
#[derive(Debug, Clone)]
pub struct FragileWatermark {
    /// Width and height of each block in pixels.
    pub block_size: usize,
    /// PRNG seed for pattern generation.
    pub seed: u32,
}

impl FragileWatermark {
    /// Create a new fragile watermark with the given block size and seed.
    #[must_use]
    pub fn new(block_size: usize, seed: u32) -> Self {
        Self { block_size, seed }
    }

    /// Generate the embedding pattern for an image of the given dimensions.
    ///
    /// Returns a `Vec<bool>` with `width * height` entries; `true` means the
    /// corresponding pixel carries a watermark bit of 1, `false` means 0.
    #[must_use]
    pub fn generate_pattern(&self, width: usize, height: usize) -> Vec<bool> {
        let total = width * height;
        let mut pattern = Vec::with_capacity(total);
        // Simple linear-congruential PRNG seeded from `self.seed`
        let mut state: u32 = self.seed.wrapping_add(1);
        for _ in 0..total {
            state = lcg_next(state);
            pattern.push((state >> 31) == 1);
        }
        pattern
    }
}

/// One step of a 32-bit LCG (Numerical Recipes constants).
#[inline]
fn lcg_next(state: u32) -> u32 {
    state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Embed a fragile watermark into a flat pixel buffer.
///
/// The pixel buffer is assumed to be 8-bit luma (or the luma channel of a
/// packed format).  The LSB of each pixel is set to the corresponding
/// watermark pattern bit.
///
/// # Arguments
/// * `pixels`     – mutable flat pixel buffer (`width * height` bytes).
/// * `width`      – image width in pixels.
/// * `height`     – image height in pixels.
/// * `watermark`  – the `FragileWatermark` descriptor.
///
/// # Panics
///
/// Panics if `pixels.len() < width * height`.
pub fn embed_fragile(pixels: &mut [u8], width: usize, height: usize, watermark: &FragileWatermark) {
    let pattern = watermark.generate_pattern(width, height);
    for (px, &bit) in pixels.iter_mut().zip(pattern.iter()) {
        // Clear LSB then set to watermark bit
        *px = (*px & 0xFE) | (u8::from(bit));
    }
}

/// Result of verifying a fragile watermark.
#[derive(Debug, Clone)]
pub struct FragileVerification {
    /// List of (`block_x`, `block_y`) coordinates of tampered blocks.
    pub tampered_blocks: Vec<(usize, usize)>,
    /// Percentage of blocks that appear intact (0.0 – 100.0).
    pub integrity_pct: f32,
}

impl FragileVerification {
    /// Return `true` if `integrity_pct ≥ threshold_pct`.
    #[must_use]
    pub fn is_intact(&self, threshold_pct: f32) -> bool {
        self.integrity_pct >= threshold_pct
    }

    /// Number of tampered blocks detected.
    #[must_use]
    pub fn tampered_block_count(&self) -> usize {
        self.tampered_blocks.len()
    }
}

/// Verify whether a pixel buffer still contains a valid fragile watermark.
///
/// The image is divided into non-overlapping blocks of size
/// `watermark.block_size × watermark.block_size`.  For each block the
/// fraction of pixels whose LSB matches the expected pattern is computed;
/// blocks where fewer than 90 % of LSBs match are flagged as tampered.
///
/// # Arguments
/// * `pixels`     – flat pixel buffer (`width * height` bytes).
/// * `width`      – image width in pixels.
/// * `height`     – image height in pixels.
/// * `watermark`  – the `FragileWatermark` descriptor (same as used for embedding).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn verify_fragile(
    pixels: &[u8],
    width: usize,
    height: usize,
    watermark: &FragileWatermark,
) -> FragileVerification {
    let pattern = watermark.generate_pattern(width, height);
    let bs = watermark.block_size.max(1);

    let blocks_x = width.div_ceil(bs);
    let blocks_y = height.div_ceil(bs);
    let total_blocks = blocks_x * blocks_y;

    let mut tampered_blocks = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut correct = 0_usize;
            let mut total = 0_usize;

            for dy in 0..bs {
                let y = by * bs + dy;
                if y >= height {
                    break;
                }
                for dx in 0..bs {
                    let x = bx * bs + dx;
                    if x >= width {
                        break;
                    }
                    let idx = y * width + x;
                    if idx >= pixels.len() || idx >= pattern.len() {
                        continue;
                    }
                    let expected_lsb: u8 = u8::from(pattern[idx]);
                    let actual_lsb: u8 = pixels[idx] & 1;
                    if actual_lsb == expected_lsb {
                        correct += 1;
                    }
                    total += 1;
                }
            }

            let match_frac = if total == 0 {
                1.0_f32
            } else {
                correct as f32 / total as f32
            };

            if match_frac < 0.9 {
                tampered_blocks.push((bx, by));
            }
        }
    }

    let intact_blocks = total_blocks - tampered_blocks.len();
    let integrity_pct = if total_blocks == 0 {
        100.0_f32
    } else {
        100.0 * intact_blocks as f32 / total_blocks as f32
    };

    FragileVerification {
        tampered_blocks,
        integrity_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixels(w: usize, h: usize) -> Vec<u8> {
        // Predictable non-zero pixels
        (0..(w * h)).map(|i| (i % 256) as u8).collect()
    }

    // ── FragileWatermark ──────────────────────────────────────────────────────

    #[test]
    fn test_generate_pattern_length() {
        let wm = FragileWatermark::new(8, 42);
        let pattern = wm.generate_pattern(16, 8);
        assert_eq!(pattern.len(), 128);
    }

    #[test]
    fn test_generate_pattern_not_all_false() {
        let wm = FragileWatermark::new(8, 1);
        let pattern = wm.generate_pattern(10, 10);
        assert!(pattern.iter().any(|&b| b));
    }

    #[test]
    fn test_generate_pattern_not_all_true() {
        let wm = FragileWatermark::new(8, 1);
        let pattern = wm.generate_pattern(10, 10);
        assert!(pattern.iter().any(|&b| !b));
    }

    #[test]
    fn test_generate_pattern_deterministic() {
        let wm = FragileWatermark::new(4, 99);
        let p1 = wm.generate_pattern(8, 8);
        let p2 = wm.generate_pattern(8, 8);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_generate_pattern_different_seeds_differ() {
        let wm1 = FragileWatermark::new(4, 1);
        let wm2 = FragileWatermark::new(4, 2);
        let p1 = wm1.generate_pattern(8, 8);
        let p2 = wm2.generate_pattern(8, 8);
        assert_ne!(p1, p2);
    }

    // ── embed_fragile ─────────────────────────────────────────────────────────

    #[test]
    fn test_embed_sets_lsbs() {
        let wm = FragileWatermark::new(4, 7);
        let mut pixels = make_pixels(4, 4);
        embed_fragile(&mut pixels, 4, 4, &wm);
        let pattern = wm.generate_pattern(4, 4);
        for (i, (&px, &bit)) in pixels.iter().zip(pattern.iter()).enumerate() {
            assert_eq!((px & 1) as u8, u8::from(bit), "LSB mismatch at pixel {i}");
        }
    }

    #[test]
    fn test_embed_preserves_upper_bits() {
        let wm = FragileWatermark::new(4, 7);
        let original = make_pixels(4, 4);
        let mut pixels = original.clone();
        embed_fragile(&mut pixels, 4, 4, &wm);
        for (orig, new) in original.iter().zip(pixels.iter()) {
            // Upper 7 bits must be the same
            assert_eq!(orig >> 1, new >> 1);
        }
    }

    // ── FragileVerification ───────────────────────────────────────────────────

    #[test]
    fn test_is_intact_above_threshold() {
        let v = FragileVerification {
            tampered_blocks: vec![],
            integrity_pct: 98.0,
        };
        assert!(v.is_intact(95.0));
    }

    #[test]
    fn test_is_intact_below_threshold() {
        let v = FragileVerification {
            tampered_blocks: vec![(0, 0), (1, 1)],
            integrity_pct: 50.0,
        };
        assert!(!v.is_intact(80.0));
    }

    #[test]
    fn test_tampered_block_count() {
        let v = FragileVerification {
            tampered_blocks: vec![(0, 0), (2, 3)],
            integrity_pct: 75.0,
        };
        assert_eq!(v.tampered_block_count(), 2);
    }

    // ── verify_fragile (round-trip) ───────────────────────────────────────────

    #[test]
    fn test_verify_intact_after_embed() {
        let wm = FragileWatermark::new(4, 17);
        let mut pixels = make_pixels(16, 16);
        embed_fragile(&mut pixels, 16, 16, &wm);
        let result = verify_fragile(&pixels, 16, 16, &wm);
        assert!(
            result.is_intact(95.0),
            "integrity: {}",
            result.integrity_pct
        );
    }

    #[test]
    fn test_verify_tampered_after_flip() {
        let wm = FragileWatermark::new(4, 17);
        let mut pixels = make_pixels(16, 16);
        embed_fragile(&mut pixels, 16, 16, &wm);
        // Corrupt a full block
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 16 + j;
                pixels[idx] ^= 0xFF; // flip all bits
            }
        }
        let result = verify_fragile(&pixels, 16, 16, &wm);
        assert!(result.tampered_block_count() > 0);
    }

    #[test]
    fn test_verify_empty_image_100_pct() {
        let wm = FragileWatermark::new(4, 1);
        let pixels: Vec<u8> = vec![];
        let result = verify_fragile(&pixels, 0, 0, &wm);
        assert!((result.integrity_pct - 100.0).abs() < 1e-3);
    }
}
