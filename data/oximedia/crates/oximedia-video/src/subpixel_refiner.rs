//! Standalone sub-pixel motion vector refinement for `oximedia-video`.
//!
//! [`SubpixelRefiner`] takes an integer motion vector produced by a coarse
//! block-matching search and refines it to half-pixel or quarter-pixel
//! accuracy using bilinear (half-pel) and bicubic (quarter-pel) interpolation.
//!
//! # Algorithm overview
//!
//! ## Half-pel refinement
//!
//! Nine candidate positions are evaluated around the integer best match:
//! `dx ∈ {-0.5, 0.0, +0.5}` × `dy ∈ {-0.5, 0.0, +0.5}`.  For each
//! candidate the reference block is constructed via bilinear interpolation and
//! the SAD (sum of absolute differences) against the current block is
//! computed.  The candidate with the lowest SAD is selected.
//!
//! ## Quarter-pel refinement
//!
//! The best half-pel position is first determined, then nine additional
//! quarter-pel offsets `∈ {-0.25, 0.0, +0.25}` around it are evaluated using
//! bicubic interpolation.  This provides sub-pixel accuracy at the cost of
//! additional computation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::subpixel_refiner::{SubpixelRefiner, SubpixelMode};
//!
//! let width = 16u32;
//! let height = 16u32;
//! let frame: Vec<u8> = (0..256).map(|i| (i % 200) as u8).collect();
//!
//! let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
//! let (dx, dy) = refiner.refine(&frame, &frame, width, height, (0, 0), 0, 0, 8);
//! // Identical frames → zero displacement expected.
//! assert!((dx).abs() < 1.0);
//! assert!((dy).abs() < 1.0);
//! ```

// ---------------------------------------------------------------------------
// SubpixelMode
// ---------------------------------------------------------------------------

/// Sub-pixel refinement precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpixelMode {
    /// Refine to ±0.5-pixel precision using bilinear interpolation.
    HalfPel,
    /// Refine to ±0.25-pixel precision: bilinear half-pel then bicubic quarter-pel.
    QuarterPel,
}

// ---------------------------------------------------------------------------
// SubpixelRefiner
// ---------------------------------------------------------------------------

/// Refines integer motion vectors to sub-pixel accuracy.
#[derive(Debug, Clone)]
pub struct SubpixelRefiner {
    /// Refinement precision mode.
    pub mode: SubpixelMode,
}

impl SubpixelRefiner {
    /// Creates a new refiner with the specified precision `mode`.
    #[must_use]
    pub fn new(mode: SubpixelMode) -> Self {
        Self { mode }
    }

    // ------------------------------------------------------------------
    // Public interface
    // ------------------------------------------------------------------

    /// Refines `initial_mv` to sub-pixel accuracy.
    ///
    /// # Parameters
    ///
    /// * `reference` – luma plane of the reference (anchor) frame, row-major,
    ///   one byte per pixel.
    /// * `current` – luma plane of the current frame.
    /// * `width` / `height` – frame dimensions in pixels.
    /// * `initial_mv` – integer `(dx, dy)` motion vector to refine (pixels).
    /// * `block_x` / `block_y` – top-left corner of the block in `current`.
    /// * `block_size` – block edge length in pixels (square block assumed).
    ///
    /// # Returns
    ///
    /// Fractional `(dx, dy)` motion vector in pixels.
    #[must_use]
    pub fn refine(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        initial_mv: (i32, i32),
        block_x: u32,
        block_y: u32,
        block_size: u32,
    ) -> (f32, f32) {
        match self.mode {
            SubpixelMode::HalfPel => self.refine_half_pel(
                reference, current, width, height, initial_mv, block_x, block_y, block_size,
            ),
            SubpixelMode::QuarterPel => self.refine_quarter_pel(
                reference, current, width, height, initial_mv, block_x, block_y, block_size,
            ),
        }
    }

    // ------------------------------------------------------------------
    // Half-pel refinement
    // ------------------------------------------------------------------

    fn refine_half_pel(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        initial_mv: (i32, i32),
        block_x: u32,
        block_y: u32,
        block_size: u32,
    ) -> (f32, f32) {
        let base_dx = initial_mv.0 as f32;
        let base_dy = initial_mv.1 as f32;

        let mut best_sad = f32::MAX;
        let mut best_dx = base_dx;
        let mut best_dy = base_dy;

        // Evaluate 3×3 grid of half-pixel offsets.
        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                let dx = base_dx + di as f32 * 0.5;
                let dy = base_dy + dj as f32 * 0.5;
                let sad = Self::half_pel_sad(
                    reference, current, width, height, block_x, block_y, block_size, dx, dy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        (best_dx, best_dy)
    }

    // ------------------------------------------------------------------
    // Quarter-pel refinement
    // ------------------------------------------------------------------

    fn refine_quarter_pel(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        initial_mv: (i32, i32),
        block_x: u32,
        block_y: u32,
        block_size: u32,
    ) -> (f32, f32) {
        // First do half-pel search.
        let (half_dx, half_dy) = self.refine_half_pel(
            reference, current, width, height, initial_mv, block_x, block_y, block_size,
        );

        // Then refine with quarter-pel (bicubic) around the half-pel best.
        let mut best_sad = f32::MAX;
        let mut best_dx = half_dx;
        let mut best_dy = half_dy;

        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                let dx = half_dx + di as f32 * 0.25;
                let dy = half_dy + dj as f32 * 0.25;
                let sad = Self::bicubic_sad(
                    reference, current, width, height, block_x, block_y, block_size, dx, dy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        (best_dx, best_dy)
    }

    // ------------------------------------------------------------------
    // Interpolation helpers
    // ------------------------------------------------------------------

    /// Bilinearly samples a luma `frame` at fractional position `(x, y)`.
    ///
    /// Coordinates are clamped to `[0, width-1] × [0, height-1]`.
    #[must_use]
    pub fn bilinear_sample(frame: &[u8], width: u32, height: u32, x: f32, y: f32) -> f32 {
        let w = width as f32;
        let h = height as f32;

        // Clamp to valid range.
        let x = x.clamp(0.0, w - 1.0);
        let y = y.clamp(0.0, h - 1.0);

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let w = width as usize;
        let p00 = frame[y0 as usize * w + x0 as usize] as f32;
        let p10 = frame[y0 as usize * w + x1 as usize] as f32;
        let p01 = frame[y1 as usize * w + x0 as usize] as f32;
        let p11 = frame[y1 as usize * w + x1 as usize] as f32;

        p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy
    }

    /// Bicubic kernel weight for position `t` (Mitchell–Netravali with B=0, C=0.5).
    ///
    /// This is also known as the "Catmull-Rom" spline kernel.
    #[inline]
    fn bicubic_weight(t: f32) -> f32 {
        let t = t.abs();
        if t < 1.0 {
            // (a+2)|t|^3 - (a+3)|t|^2 + 1  with a = -0.5
            1.5 * t * t * t - 2.5 * t * t + 1.0
        } else if t < 2.0 {
            // a|t|^3 - 5a|t|^2 + 8a|t| - 4a  with a = -0.5
            -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
        } else {
            0.0
        }
    }

    /// Bicubic (Catmull-Rom) samples `frame` at fractional position `(x, y)`.
    #[must_use]
    pub fn bicubic_sample(frame: &[u8], width: u32, height: u32, x: f32, y: f32) -> f32 {
        let w = width as f32;
        let h = height as f32;

        let x = x.clamp(0.0, w - 1.0);
        let y = y.clamp(0.0, h - 1.0);

        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let fx = x - xi as f32;
        let fy = y - yi as f32;

        let stride = width as i32;
        let max_x = width as i32 - 1;
        let max_y = height as i32 - 1;

        let mut result = 0.0f32;
        for m in -1i32..=2 {
            let wy = Self::bicubic_weight(fy - m as f32);
            let ry = (yi + m).clamp(0, max_y) as usize;
            for n in -1i32..=2 {
                let wx = Self::bicubic_weight(fx - n as f32);
                let rx = (xi + n).clamp(0, max_x) as usize;
                result += frame[ry * stride as usize + rx] as f32 * wx * wy;
            }
        }
        result
    }

    // ------------------------------------------------------------------
    // SAD computation
    // ------------------------------------------------------------------

    /// Computes the SAD between:
    /// - the reference block at fractional position `(block_x + dx, block_y + dy)`
    ///   reconstructed via **bilinear** interpolation, and
    /// - the current block at integer position `(block_x, block_y)`.
    #[must_use]
    pub fn half_pel_sad(
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        bx: u32,
        by: u32,
        bs: u32,
        dx: f32,
        dy: f32,
    ) -> f32 {
        let mut sad = 0.0f32;
        let stride = width as usize;

        for row in 0..bs {
            for col in 0..bs {
                let cur_x = bx + col;
                let cur_y = by + row;

                // Current pixel (integer position — direct array access).
                let cur_val = if cur_y < height && cur_x < width {
                    current[cur_y as usize * stride + cur_x as usize] as f32
                } else {
                    0.0
                };

                // Reference pixel at fractional position.
                let ref_x = cur_x as f32 + dx;
                let ref_y = cur_y as f32 + dy;
                let ref_val = Self::bilinear_sample(reference, width, height, ref_x, ref_y);

                sad += (ref_val - cur_val).abs();
            }
        }

        sad
    }

    /// Computes the SAD between:
    /// - the reference block at fractional position `(block_x + dx, block_y + dy)`
    ///   reconstructed via **bicubic** interpolation, and
    /// - the current block at integer position `(block_x, block_y)`.
    #[must_use]
    fn bicubic_sad(
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        bx: u32,
        by: u32,
        bs: u32,
        dx: f32,
        dy: f32,
    ) -> f32 {
        let mut sad = 0.0f32;
        let stride = width as usize;

        for row in 0..bs {
            for col in 0..bs {
                let cur_x = bx + col;
                let cur_y = by + row;

                let cur_val = if cur_y < height && cur_x < width {
                    current[cur_y as usize * stride + cur_x as usize] as f32
                } else {
                    0.0
                };

                let ref_x = cur_x as f32 + dx;
                let ref_y = cur_y as f32 + dy;
                let ref_val = Self::bicubic_sample(reference, width, height, ref_x, ref_y);

                sad += (ref_val - cur_val).abs();
            }
        }

        sad
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: build test frames
    // ------------------------------------------------------------------

    fn solid_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            for col in 0..width {
                frame.push(((row * width + col) % 256) as u8);
            }
        }
        frame
    }

    /// Frame shifted by integer `(shift_x, shift_y)` to simulate motion.
    fn shifted_frame(width: u32, height: u32, shift_x: i32, shift_y: i32) -> Vec<u8> {
        let mut frame = vec![128u8; (width * height) as usize];
        let w = width as i32;
        let h = height as i32;
        for row in 0..h {
            for col in 0..w {
                let src_x = col - shift_x;
                let src_y = row - shift_y;
                if src_x >= 0 && src_x < w && src_y >= 0 && src_y < h {
                    let dst = (row * w + col) as usize;
                    let src = (src_y * w + src_x) as usize;
                    frame[dst] = (src % 256) as u8;
                }
            }
        }
        frame
    }

    // ------------------------------------------------------------------
    // bilinear_sample
    // ------------------------------------------------------------------

    #[test]
    fn test_bilinear_sample_exact_pixel() {
        let frame = gradient_frame(8, 8);
        // Sampling at exact integer coordinates should equal the pixel value.
        let val = SubpixelRefiner::bilinear_sample(&frame, 8, 8, 3.0, 2.0);
        let expected = frame[2 * 8 + 3] as f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "val={val} expected={expected}"
        );
    }

    #[test]
    fn test_bilinear_sample_clamping() {
        let frame = solid_frame(4, 4, 200);
        // Out-of-bounds coordinates should be clamped, not panic.
        let val = SubpixelRefiner::bilinear_sample(&frame, 4, 4, -1.0, -1.0);
        assert!((val - 200.0).abs() < 1e-4);
        let val2 = SubpixelRefiner::bilinear_sample(&frame, 4, 4, 100.0, 100.0);
        assert!((val2 - 200.0).abs() < 1e-4);
    }

    #[test]
    fn test_bilinear_sample_mid_point() {
        // A 2×2 solid frame: midpoint should equal the constant value.
        let frame = solid_frame(2, 2, 100);
        let val = SubpixelRefiner::bilinear_sample(&frame, 2, 2, 0.5, 0.5);
        assert!((val - 100.0).abs() < 1e-3, "val={val}");
    }

    #[test]
    fn test_bilinear_sample_interpolation() {
        // Frame:  0  255
        //        255  0
        // At (0.5, 0.0) the bilinear interpolated value should be ~127.5
        let frame: Vec<u8> = vec![0, 255, 255, 0];
        let val = SubpixelRefiner::bilinear_sample(&frame, 2, 2, 0.5, 0.0);
        assert!((val - 127.5).abs() < 1.0, "Expected ~127.5, got {val}");
    }

    // ------------------------------------------------------------------
    // bicubic_sample
    // ------------------------------------------------------------------

    #[test]
    fn test_bicubic_sample_exact_pixel() {
        let frame = gradient_frame(16, 16);
        let val = SubpixelRefiner::bicubic_sample(&frame, 16, 16, 5.0, 3.0);
        let expected = frame[3 * 16 + 5] as f32;
        // Bicubic at an exact integer coordinate should equal the pixel value.
        assert!(
            (val - expected).abs() < 1.0,
            "val={val} expected={expected}"
        );
    }

    #[test]
    fn test_bicubic_sample_solid_frame() {
        // Any sample from a constant frame should equal the constant.
        let frame = solid_frame(16, 16, 128);
        let val = SubpixelRefiner::bicubic_sample(&frame, 16, 16, 4.3, 7.8);
        assert!(
            (val - 128.0).abs() < 2.0,
            "bicubic on solid frame: val={val}"
        );
    }

    // ------------------------------------------------------------------
    // half_pel_sad
    // ------------------------------------------------------------------

    #[test]
    fn test_half_pel_sad_zero_displacement_identical_frames() {
        let frame = gradient_frame(16, 16);
        let sad = SubpixelRefiner::half_pel_sad(&frame, &frame, 16, 16, 0, 0, 8, 0.0, 0.0);
        assert!(
            sad < 1e-4,
            "SAD with zero displacement on identical frames should be ~0, got {sad}"
        );
    }

    #[test]
    fn test_half_pel_sad_non_zero_displacement() {
        let frame = gradient_frame(32, 32);
        let sad_zero = SubpixelRefiner::half_pel_sad(&frame, &frame, 32, 32, 4, 4, 8, 0.0, 0.0);
        let sad_shift = SubpixelRefiner::half_pel_sad(&frame, &frame, 32, 32, 4, 4, 8, 2.0, 2.0);
        // Non-zero displacement on a non-constant frame should produce SAD > 0.
        assert!(
            sad_shift >= sad_zero,
            "Shifted SAD ({sad_shift}) should be >= zero-displacement SAD ({sad_zero})"
        );
    }

    // ------------------------------------------------------------------
    // SubpixelRefiner::refine — half-pel
    // ------------------------------------------------------------------

    #[test]
    fn test_refine_half_pel_identical_frames_zero_mv() {
        let frame = gradient_frame(32, 32);
        let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        let (dx, dy) = refiner.refine(&frame, &frame, 32, 32, (0, 0), 4, 4, 8);
        // Identical frame, zero initial MV → should stay at (0, 0) or very close.
        assert!(dx.abs() <= 0.5, "dx={dx}");
        assert!(dy.abs() <= 0.5, "dy={dy}");
    }

    #[test]
    fn test_refine_half_pel_solid_frame() {
        // Solid frame: any displacement gives equal SAD; refiner should not panic.
        let frame = solid_frame(16, 16, 128);
        let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        let (dx, dy) = refiner.refine(&frame, &frame, 16, 16, (0, 0), 0, 0, 8);
        // All candidates equal — refiner picks first (0, 0) or any; just no panic.
        let _ = (dx, dy);
    }

    #[test]
    fn test_refine_half_pel_returns_fractional_mv() {
        let frame = gradient_frame(64, 64);
        let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        let (dx, dy) = refiner.refine(&frame, &frame, 64, 64, (2, 3), 8, 8, 8);
        // Result should be within half-pel range of the initial MV.
        assert!((dx - 2.0).abs() <= 0.5 + 1e-4, "dx={dx} initial_dx=2");
        assert!((dy - 3.0).abs() <= 0.5 + 1e-4, "dy={dy} initial_dy=3");
    }

    #[test]
    fn test_refine_half_pel_is_better_or_equal_to_integer() {
        let width = 32u32;
        let height = 32u32;
        let reference = gradient_frame(width, height);
        let current = shifted_frame(width, height, 1, 1);

        let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        let (dx, dy) = refiner.refine(&reference, &current, width, height, (1, 1), 4, 4, 8);

        // Compute integer-pel SAD at (1, 1).
        let int_sad =
            SubpixelRefiner::half_pel_sad(&reference, &current, width, height, 4, 4, 8, 1.0, 1.0);
        // Refined SAD should be <= integer SAD.
        let refined_sad =
            SubpixelRefiner::half_pel_sad(&reference, &current, width, height, 4, 4, 8, dx, dy);
        assert!(
            refined_sad <= int_sad + 1e-4,
            "refined_sad={refined_sad} int_sad={int_sad}"
        );
    }

    // ------------------------------------------------------------------
    // SubpixelRefiner::refine — quarter-pel
    // ------------------------------------------------------------------

    #[test]
    fn test_refine_quarter_pel_identical_frames() {
        let frame = gradient_frame(32, 32);
        let refiner = SubpixelRefiner::new(SubpixelMode::QuarterPel);
        let (dx, dy) = refiner.refine(&frame, &frame, 32, 32, (0, 0), 4, 4, 8);
        assert!(dx.abs() <= 0.5, "dx={dx}");
        assert!(dy.abs() <= 0.5, "dy={dy}");
    }

    #[test]
    fn test_refine_quarter_pel_returns_fractional_mv() {
        let frame = gradient_frame(64, 64);
        let refiner = SubpixelRefiner::new(SubpixelMode::QuarterPel);
        let (dx, dy) = refiner.refine(&frame, &frame, 64, 64, (1, 2), 8, 8, 8);
        // Result should be within quarter-pel range of the half-pel best.
        assert!((dx - 1.0).abs() <= 0.75 + 1e-4, "dx={dx}");
        assert!((dy - 2.0).abs() <= 0.75 + 1e-4, "dy={dy}");
    }

    #[test]
    fn test_refine_quarter_pel_leq_half_pel_sad() {
        let width = 32u32;
        let height = 32u32;
        let reference = gradient_frame(width, height);
        let current = shifted_frame(width, height, 2, 1);

        let half_refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        let qpel_refiner = SubpixelRefiner::new(SubpixelMode::QuarterPel);

        let (hdx, hdy) = half_refiner.refine(&reference, &current, width, height, (2, 1), 4, 4, 8);
        let (qdx, qdy) = qpel_refiner.refine(&reference, &current, width, height, (2, 1), 4, 4, 8);

        let half_sad =
            SubpixelRefiner::half_pel_sad(&reference, &current, width, height, 4, 4, 8, hdx, hdy);
        let qpel_sad =
            SubpixelRefiner::half_pel_sad(&reference, &current, width, height, 4, 4, 8, qdx, qdy);

        // Quarter-pel should be at least as good as half-pel.
        assert!(
            qpel_sad <= half_sad + 1.0,
            "qpel_sad={qpel_sad} half_sad={half_sad}"
        );
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_refine_block_at_frame_boundary() {
        let width = 16u32;
        let height = 16u32;
        let frame = gradient_frame(width, height);
        let refiner = SubpixelRefiner::new(SubpixelMode::HalfPel);
        // Block at top-left corner.
        let (dx, dy) = refiner.refine(&frame, &frame, width, height, (0, 0), 0, 0, 4);
        let _ = (dx, dy); // Just verify no panic.
    }

    #[test]
    fn test_refine_block_at_right_bottom_boundary() {
        let width = 16u32;
        let height = 16u32;
        let frame = gradient_frame(width, height);
        let refiner = SubpixelRefiner::new(SubpixelMode::QuarterPel);
        // Block near bottom-right corner.
        let (dx, dy) = refiner.refine(&frame, &frame, width, height, (0, 0), 12, 12, 4);
        let _ = (dx, dy);
    }

    #[test]
    fn test_subpixel_mode_equality() {
        assert_eq!(SubpixelMode::HalfPel, SubpixelMode::HalfPel);
        assert_ne!(SubpixelMode::HalfPel, SubpixelMode::QuarterPel);
    }

    #[test]
    fn test_refiner_debug_clone() {
        let r = SubpixelRefiner::new(SubpixelMode::QuarterPel);
        let r2 = r.clone();
        let _ = format!("{:?}", r2);
    }
}
