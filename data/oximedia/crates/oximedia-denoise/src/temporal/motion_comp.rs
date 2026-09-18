//! Motion-compensated temporal denoising.
//!
//! Reduces temporal noise by finding the best-matching block in a reference
//! frame and blending the two frames according to temporal strength.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

// ---------------------------------------------------------------------------
// Motion vector
// ---------------------------------------------------------------------------

/// A 2-D integer motion vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// Horizontal displacement in pixels (positive = right).
    pub dx: i16,
    /// Vertical displacement in pixels (positive = down).
    pub dy: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(dx: i16, dy: i16) -> Self {
        Self { dx, dy }
    }

    /// Euclidean magnitude of the vector.
    pub fn magnitude(&self) -> f32 {
        let dx = self.dx as f32;
        let dy = self.dy as f32;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns `true` if both components are zero.
    pub fn is_still(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }
}

// ---------------------------------------------------------------------------
// Block matching helpers
// ---------------------------------------------------------------------------

/// Compute the **sum of absolute differences** (SAD) between two same-sized
/// blocks.
///
/// Both slices must have `block_size * block_size` elements.
pub fn block_sad(current: &[u8], reference: &[u8], block_size: usize) -> u32 {
    let n = block_size * block_size;
    current
        .iter()
        .take(n)
        .zip(reference.iter().take(n))
        .map(|(&c, &r)| (c as i32 - r as i32).unsigned_abs())
        .sum()
}

/// Extract a `block_size × block_size` block from `frame` with top-left at
/// (`bx`, `by`).
///
/// Returns `None` if the block goes out-of-bounds.
fn extract_block(
    frame: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
) -> Option<Vec<u8>> {
    if bx + block_size > width || by + block_size > height {
        return None;
    }
    let mut block = Vec::with_capacity(block_size * block_size);
    for row in 0..block_size {
        let start = (by + row) * width + bx;
        block.extend_from_slice(&frame[start..start + block_size]);
    }
    Some(block)
}

/// Full exhaustive block-matching motion search.
///
/// Returns the motion vector (from `reference` to `current`) that minimises
/// the SAD between the block at (`bx`, `by`) in `current` and the
/// displaced block in `reference`.
pub fn full_search_mv(
    current: &[u8],
    reference: &[u8],
    bx: usize,
    by: usize,
    block_size: usize,
    search_range: usize,
    stride: usize,
) -> MotionVector {
    let height = current.len().checked_div(stride).unwrap_or(0);
    let Some(cur_block) = extract_block(current, stride, height, bx, by, block_size) else {
        return MotionVector::default();
    };

    let mut best_sad = u32::MAX;
    let mut best_mv = MotionVector::default();

    let sr = search_range as isize;
    let bx_i = bx as isize;
    let by_i = by as isize;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let rx = bx_i + dx;
            let ry = by_i + dy;
            if rx < 0 || ry < 0 {
                continue;
            }
            let ref_height = reference.len().checked_div(stride).unwrap_or(0);
            let Some(ref_block) = extract_block(
                reference,
                stride,
                ref_height,
                rx as usize,
                ry as usize,
                block_size,
            ) else {
                continue;
            };
            let sad = block_sad(&cur_block, &ref_block, block_size);
            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector::new(dx as i16, dy as i16);
            }
        }
    }

    best_mv
}

// ---------------------------------------------------------------------------
// Motion-compensated denoiser
// ---------------------------------------------------------------------------

/// Motion-compensated temporal denoiser.
///
/// For each block in the current frame the denoiser searches the reference
/// frame for the best-matching block and blends them.  A high `temporal_strength`
/// means more reference blending (more denoising, potential ghosting if MV is
/// wrong); 0.0 means no blending.
pub struct McTemporalDenoiser {
    /// Block size for motion search (e.g. 8 or 16).
    pub block_size: usize,
    /// Search radius for full-search motion estimation.
    pub search_range: usize,
    /// Blending weight for the reference frame (0.0 = no blend, 1.0 = full).
    pub temporal_strength: f32,
}

impl McTemporalDenoiser {
    /// Create a new motion-compensated temporal denoiser.
    pub fn new(block_size: usize, search_range: usize, temporal_strength: f32) -> Self {
        Self {
            block_size,
            search_range,
            temporal_strength: temporal_strength.clamp(0.0, 1.0),
        }
    }

    /// Blend `current` with a motion-compensated version of `reference`.
    ///
    /// Both frames must be `width × height` luma (8-bit) images.
    /// Returns a denoised frame of the same dimensions.
    pub fn blend_frames(
        &self,
        current: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let mut out = current.to_vec();
        let bs = self.block_size;

        let by_max = height.saturating_sub(bs);
        let bx_max = width.saturating_sub(bs);

        let mut by = 0;
        while by <= by_max {
            let mut bx = 0;
            while bx <= bx_max {
                let mv = full_search_mv(current, reference, bx, by, bs, self.search_range, width);

                // Compensated position in reference.
                let rx = (bx as isize + mv.dx as isize).clamp(0, (width - bs) as isize) as usize;
                let ry = (by as isize + mv.dy as isize).clamp(0, (height - bs) as isize) as usize;

                // Blend block pixels.
                for row in 0..bs {
                    for col in 0..bs {
                        let cur_idx = (by + row) * width + bx + col;
                        let ref_idx = (ry + row) * width + rx + col;

                        if cur_idx < current.len() && ref_idx < reference.len() {
                            let c = current[cur_idx] as f32;
                            let r = reference[ref_idx] as f32;
                            let blended =
                                c * (1.0 - self.temporal_strength) + r * self.temporal_strength;
                            out[cur_idx] = blended.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }

                bx += bs;
            }
            by += bs;
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- MotionVector ----------

    #[test]
    fn test_motion_vector_default_is_still() {
        let mv = MotionVector::default();
        assert!(mv.is_still());
    }

    #[test]
    fn test_motion_vector_magnitude_zero() {
        let mv = MotionVector::new(0, 0);
        assert!((mv.magnitude()).abs() < 1e-6);
    }

    #[test]
    fn test_motion_vector_magnitude_3_4_is_5() {
        let mv = MotionVector::new(3, 4);
        assert!((mv.magnitude() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_motion_vector_not_still() {
        let mv = MotionVector::new(1, 0);
        assert!(!mv.is_still());
    }

    #[test]
    fn test_motion_vector_negative_components() {
        let mv = MotionVector::new(-3, -4);
        assert!((mv.magnitude() - 5.0).abs() < 1e-4);
    }

    // ---------- block_sad ----------

    #[test]
    fn test_block_sad_identical() {
        let block = vec![100u8; 16];
        assert_eq!(block_sad(&block, &block, 4), 0);
    }

    #[test]
    fn test_block_sad_known_value() {
        let a = vec![10u8; 4]; // 2×2 block
        let b = vec![20u8; 4];
        // SAD = 4 × 10 = 40
        assert_eq!(block_sad(&a, &b, 2), 40);
    }

    #[test]
    fn test_block_sad_asymmetric() {
        // block_size=1 → n=1, only compares first element: |0 - 255| = 255
        let a = vec![0u8, 255];
        let b = vec![255u8, 0];
        assert_eq!(block_sad(&a, &b, 1), 255);
    }

    // ---------- full_search_mv ----------

    #[test]
    fn test_full_search_mv_no_motion() {
        // Same frame → zero MV.
        let frame = vec![128u8; 32 * 32];
        let mv = full_search_mv(&frame, &frame, 0, 0, 8, 4, 32);
        assert!(mv.is_still());
    }

    #[test]
    fn test_full_search_mv_returns_mv() {
        // Shifted frame.
        let mut current = vec![0u8; 32 * 32];
        let mut reference = vec![0u8; 32 * 32];
        // Put a bright 8×8 block at (4,4) in current, (8,4) in reference.
        for r in 0..8_usize {
            for c in 0..8_usize {
                current[(4 + r) * 32 + (4 + c)] = 200;
                reference[(4 + r) * 32 + (8 + c)] = 200;
            }
        }
        let mv = full_search_mv(&current, &reference, 4, 4, 8, 8, 32);
        // Expected: dx ≈ 4, dy ≈ 0
        assert_eq!(mv.dy, 0);
        assert_eq!(mv.dx, 4);
    }

    // ---------- McTemporalDenoiser ----------

    #[test]
    fn test_mc_denoiser_output_length() {
        let frame = vec![128u8; 16 * 16];
        let d = McTemporalDenoiser::new(8, 4, 0.5);
        let out = d.blend_frames(&frame, &frame, 16, 16);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_mc_denoiser_zero_strength_unchanged() {
        let current: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let reference = vec![0u8; 64];
        let d = McTemporalDenoiser::new(4, 2, 0.0);
        let out = d.blend_frames(&current, &reference, 8, 8);
        assert_eq!(out, current);
    }

    #[test]
    fn test_mc_denoiser_full_strength_copies_reference() {
        let current = vec![0u8; 64];
        let reference = vec![200u8; 64];
        let d = McTemporalDenoiser::new(8, 2, 1.0);
        let out = d.blend_frames(&current, &reference, 8, 8);
        // With strength=1.0, output should equal reference values
        for &v in &out {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_mc_denoiser_temporal_strength_clamped() {
        let d = McTemporalDenoiser::new(8, 4, 2.5);
        assert!((d.temporal_strength - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mc_denoiser_uniform_frames() {
        let current = vec![128u8; 32 * 32];
        let reference = vec![128u8; 32 * 32];
        let d = McTemporalDenoiser::new(8, 4, 0.5);
        let out = d.blend_frames(&current, &reference, 32, 32);
        for &v in &out {
            assert_eq!(v, 128);
        }
    }
}
