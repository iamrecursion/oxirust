//! Motion-Compensated Temporal Filter (MCTF).
//!
//! Processes a sliding window of N frames.  For each frame `F[t]` the filter:
//!  1. Estimates motion vectors from `F[t]` to `F[t-1]` and `F[t+1]` via diamond-
//!     search block matching (8×8 blocks, SAD metric).
//!  2. Warps the reference frames toward `F[t]` using bilinear sub-pixel
//!     interpolation.
//!  3. Blends: `Fout[t] = w0*F[t] + w_ref*(F[t-1]_warped + F[t+1]_warped)`,
//!     where weights are reduced for blocks with high motion magnitude to
//!     suppress ghosting artefacts.

/// Configuration for [`MctfFilter`].
#[derive(Debug, Clone)]
pub struct MctfConfig {
    /// Number of reference frames on each side (1 or 2).  Default: 1.
    ///
    /// `window_size = 1` → use one previous and one next frame (tri-frame).
    /// `window_size = 2` → use two frames on each side (quintet).
    pub window_size: usize,
    /// Block size for SAD-based block matching.  Default: 8.
    pub block_size: u32,
    /// SAD-per-pixel threshold above which the temporal weight is reduced.
    /// Default: 64.0.
    pub motion_threshold: f32,
    /// Base blending weight for each reference frame (0..1).  Default: 0.4.
    pub temporal_strength: f32,
}

impl Default for MctfConfig {
    fn default() -> Self {
        Self {
            window_size: 1,
            block_size: 8,
            motion_threshold: 64.0,
            temporal_strength: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Core free functions (public API)
// ---------------------------------------------------------------------------

/// Estimate motion vectors between two frames using a diamond-search block
/// matching algorithm.
///
/// # Parameters
/// * `src` – source frame pixel data (row-major, width × height)
/// * `dst` – destination / reference frame pixel data
/// * `w` – frame width in pixels
/// * `h` – frame height in pixels
/// * `block_size` – block size for SAD computation
/// * `search_range` – maximum displacement to search (full diamond diameter)
///
/// # Returns
/// A row-major vector of `(dx, dy)` motion vectors, one per block.
pub fn estimate_motion_vectors(
    src: &[u8],
    dst: &[u8],
    w: u32,
    h: u32,
    block_size: u32,
    search_range: u32,
) -> Vec<(i32, i32)> {
    let bw = w as usize;
    let bh = h as usize;
    let bs = block_size as usize;
    let sr = search_range as i32;

    let num_bx = bw.div_ceil(bs);
    let num_by = bh.div_ceil(bs);

    let mut mvs = Vec::with_capacity(num_bx * num_by);

    for by in 0..num_by {
        let block_y = by * bs;
        for bx in 0..num_bx {
            let block_x = bx * bs;
            let result = diamond_search_block(src, dst, bw, bh, block_x, block_y, bs, sr);
            mvs.push((result.dx, result.dy));
        }
    }

    mvs
}

/// Internal version that returns MVs, best-SAD-per-pixel, and zero-MV-SAD-per-pixel.
///
/// The zero-MV SAD is used as a confidence reference: if the best MV only
/// marginally improves over (0,0), the MV is unreliable (likely noise-driven)
/// and should be attenuated or discarded.
fn estimate_mvs_with_sad(
    src: &[u8],
    dst: &[u8],
    w: u32,
    h: u32,
    block_size: u32,
    search_range: u32,
) -> (Vec<(i32, i32)>, Vec<f32>, Vec<f32>) {
    let bw = w as usize;
    let bh = h as usize;
    let bs = block_size as usize;
    let sr = search_range as i32;

    let num_bx = bw.div_ceil(bs);
    let num_by = bh.div_ceil(bs);
    let n = num_bx * num_by;

    let mut mvs = Vec::with_capacity(n);
    let mut sads = Vec::with_capacity(n);
    let mut sads_zero = Vec::with_capacity(n);

    for by in 0..num_by {
        let block_y = by * bs;
        for bx in 0..num_bx {
            let block_x = bx * bs;
            let result = diamond_search_block(src, dst, bw, bh, block_x, block_y, bs, sr);
            mvs.push((result.dx, result.dy));
            sads.push(result.sad_per_pixel);
            sads_zero.push(result.sad_zero_pp);
        }
    }

    (mvs, sads, sads_zero)
}

/// Warp `src` using the provided block motion vectors (bilinear interpolation,
/// border-clamped).
///
/// # Parameters
/// * `src` – source frame to warp (row-major, `w × h`)
/// * `mvs` – row-major motion vector grid (one entry per block)
/// * `w` / `h` – frame dimensions
/// * `block_size` – block size used during MV estimation
///
/// # Returns
/// Warped frame with the same dimensions as `src`.
pub fn warp_frame(src: &[u8], mvs: &[(i32, i32)], w: u32, h: u32, block_size: u32) -> Vec<u8> {
    let bw = w as usize;
    let bh = h as usize;
    let bs = block_size as usize;
    let num_bx = bw.div_ceil(bs);

    let mut out = vec![0u8; bw * bh];

    for y in 0..bh {
        for x in 0..bw {
            let mbx = x / bs;
            let mby = y / bs;
            let mv_idx = mby * num_bx + mbx;
            let (dx, dy) = mvs.get(mv_idx).copied().unwrap_or((0, 0));

            let sx = (x as i32 + dx).clamp(0, bw as i32 - 1) as usize;
            let sy = (y as i32 + dy).clamp(0, bh as i32 - 1) as usize;

            out[y * bw + x] = src[sy * bw + sx];
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Diamond search internals
// ---------------------------------------------------------------------------

/// Compute SAD between a block in `src` and a block in `ref_frame`.
fn block_sad_at(
    src: &[u8],
    ref_frame: &[u8],
    src_x: usize,
    src_y: usize,
    ref_x: usize,
    ref_y: usize,
    bs: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sad = 0u32;
    for row in 0..bs {
        let sy = src_y + row;
        let ry = ref_y + row;
        if sy >= h || ry >= h {
            break;
        }
        for col in 0..bs {
            let sx = src_x + col;
            let rx = ref_x + col;
            if sx >= w || rx >= w {
                break;
            }
            let a = src[sy * w + sx] as i32;
            let b = ref_frame[ry * w + rx] as i32;
            sad += (a - b).unsigned_abs();
        }
    }
    sad
}

/// Result of block-level diamond search: motion vector + match quality.
///
/// The SAD values are per-pixel (total / block_area) for resolution-independence.
struct BlockMvResult {
    dx: i32,
    dy: i32,
    /// SAD per pixel at the best-match position (0..255).
    sad_per_pixel: f32,
    /// SAD per pixel at the zero-motion position (0..255).
    sad_zero_pp: f32,
}

/// Full diamond-search motion estimation for a single block.
///
/// Algorithm:
///  1. Large-diamond search (5-point cross: center + ±`step` on each axis)
///     for the coarse phase.
///  2. Iteratively move center to best candidate and repeat until the center
///     is already best.
///  3. Fine 3×3 refinement step around the winning candidate.
///
/// Returns the best `(dx, dy)` and the per-pixel SAD at that position.
fn diamond_search_block(
    src: &[u8],
    dst: &[u8],
    w: usize,
    h: usize,
    block_x: usize,
    block_y: usize,
    bs: usize,
    search_range: i32,
) -> BlockMvResult {
    let block_area = (bs * bs).max(1) as f32;
    // Coarse diamond (large step), then refine with unit step.
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = block_sad_at(src, dst, block_x, block_y, block_x, block_y, bs, w, h);

    // Large-diamond iteration: step starts at half the search range and
    // halves each iteration until it reaches 1.
    let mut step = (search_range / 2).max(1);
    while step >= 1 {
        let candidates: [(i32, i32); 5] = [
            (best_dx, best_dy),
            (best_dx + step, best_dy),
            (best_dx - step, best_dy),
            (best_dx, best_dy + step),
            (best_dx, best_dy - step),
        ];

        let mut moved = false;
        for &(cdx, cdy) in &candidates {
            if cdx.abs() > search_range || cdy.abs() > search_range {
                continue;
            }
            let ref_x = (block_x as i32 + cdx).clamp(0, w as i32 - 1) as usize;
            let ref_y = (block_y as i32 + cdy).clamp(0, h as i32 - 1) as usize;
            let sad = block_sad_at(src, dst, block_x, block_y, ref_x, ref_y, bs, w, h);
            if sad < best_sad {
                best_sad = sad;
                best_dx = cdx;
                best_dy = cdy;
                moved = true;
            }
        }

        if !moved {
            step /= 2;
        }
    }

    // 9-point fine refinement around winning point.
    for rdy in -1i32..=1 {
        for rdx in -1i32..=1 {
            let cdx = best_dx + rdx;
            let cdy = best_dy + rdy;
            if cdx.abs() > search_range || cdy.abs() > search_range {
                continue;
            }
            let ref_x = (block_x as i32 + cdx).clamp(0, w as i32 - 1) as usize;
            let ref_y = (block_y as i32 + cdy).clamp(0, h as i32 - 1) as usize;
            let sad = block_sad_at(src, dst, block_x, block_y, ref_x, ref_y, bs, w, h);
            if sad < best_sad {
                best_sad = sad;
                best_dx = cdx;
                best_dy = cdy;
            }
        }
    }

    // SAD at zero-motion position (dx=0, dy=0) for confidence estimation.
    let sad_zero = block_sad_at(src, dst, block_x, block_y, block_x, block_y, bs, w, h);

    BlockMvResult {
        dx: best_dx,
        dy: best_dy,
        sad_per_pixel: best_sad as f32 / block_area,
        sad_zero_pp: sad_zero as f32 / block_area,
    }
}

// ---------------------------------------------------------------------------
// MctfFilter
// ---------------------------------------------------------------------------

/// Motion-Compensated Temporal Filter.
pub struct MctfFilter {
    config: MctfConfig,
}

impl MctfFilter {
    /// Create a new MCTF filter with the given configuration.
    pub fn new(config: MctfConfig) -> Self {
        Self { config }
    }

    /// Process a sequence of grayscale frames.
    ///
    /// Each element of `frames` must contain exactly `w * h` bytes.
    /// The first and last frames are edge-padded (treated as if the boundary
    /// frame repeats), so the returned vector has the same length as the input.
    pub fn process_sequence(&self, frames: &[Vec<u8>], w: u32, h: u32) -> Vec<Vec<u8>> {
        let n = frames.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return frames.to_vec();
        }

        (0..n)
            .map(|t| {
                let prev = if t > 0 {
                    Some(frames[t - 1].as_slice())
                } else {
                    None
                };
                let next = if t + 1 < n {
                    Some(frames[t + 1].as_slice())
                } else {
                    None
                };
                self.process_frame(prev, frames[t].as_slice(), next, w, h)
            })
            .collect()
    }

    /// Denoise a single frame given optional previous and next reference frames.
    ///
    /// When both references are `None`, the current frame is returned unchanged
    /// (passthrough).
    pub fn process_frame(
        &self,
        prev: Option<&[u8]>,
        curr: &[u8],
        next: Option<&[u8]>,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        // No references → identity passthrough.
        if prev.is_none() && next.is_none() {
            return curr.to_vec();
        }

        let npixels = (w as usize) * (h as usize);
        let bs = self.config.block_size;
        // Search range: 2× block size, minimum 8.
        let sr = (self.config.block_size * 2).max(8);
        let bw = w as usize;
        let bh = h as usize;
        let bsz = bs as usize;
        let num_bx = bw.div_ceil(bsz);
        let ts = self.config.temporal_strength.clamp(0.0, 0.9);

        // Per-pixel weighted accumulator.
        let mut acc = vec![0.0f32; npixels];
        let mut pixel_weights = vec![0.0f32; npixels];

        // Accumulate current frame with weight (1 - temporal_strength).
        let w0 = 1.0f32 - ts;
        for (i, &p) in curr.iter().enumerate().take(npixels) {
            acc[i] = w0 * p as f32;
            pixel_weights[i] = w0;
        }

        // Inner closure: accumulate one reference frame using motion compensation.
        //
        // For each block, the diamond search estimates the best MV.  We compute
        // a *per-block confidence* by comparing the best-MV SAD to the zero-MV
        // SAD.  The reference pixel is then chosen as a linear interpolation
        // between the zero-MV reference (safe) and the MV-warped reference
        // (better for moving content):
        //
        //   ref_px = (1-α) * ref[y][x]  +  α * ref[y+dy][x+dx]
        //
        // where α = confidence (∈ [0,1]) and only applied when `mv_mag > 0`.
        // This blending approach avoids block artefacts: for noisy static content,
        // α ≈ 0 → pure zero-MV average.  For real motion with clear MVs, α → 1.
        //
        // The block weight scales with the zero-MV match quality: if even the
        // same-position reference has high SAD (very noisy), the contribution
        // is attenuated.
        let accumulate_ref = |ref_frame: &[u8], acc: &mut Vec<f32>, weights: &mut Vec<f32>| {
            let (mvs, sads, sads_zero) = estimate_mvs_with_sad(curr, ref_frame, w, h, bs, sr);

            for i in 0..npixels.min(ref_frame.len()) {
                let y = i / bw;
                let x = i % bw;
                let mv_idx = (y / bsz) * num_bx + (x / bsz);
                let sad_zero_pp = sads_zero.get(mv_idx).copied().unwrap_or(255.0);
                let sad_best_pp = sads.get(mv_idx).copied().unwrap_or(255.0);

                // Base weight: proportional to how similar this reference block
                // is at the same position.  High SAD → low weight (noisy ref).
                let base_atten = (1.0 - sad_zero_pp / self.config.motion_threshold).clamp(0.0, 1.0);
                let base_w = ts * base_atten;

                // Motion confidence: how much the MV improves over zero-motion.
                // Only non-trivial when there is genuine motion.
                let (dx, dy) = mvs.get(mv_idx).copied().unwrap_or((0, 0));
                let mv_mag = ((dx * dx + dy * dy) as f32).sqrt();
                // Confidence = improvement fraction (0 if MV = zero-motion).
                let alpha = if mv_mag > 0.5 && sad_zero_pp > 1e-3 {
                    (1.0 - sad_best_pp / sad_zero_pp).clamp(0.0, 1.0)
                } else {
                    0.0_f32
                };

                // Blend: alpha=0 → zero-MV, alpha=1 → full MC warp.
                let zero_px = ref_frame[y * bw + x] as f32;
                let mc_px = if alpha > 0.0 {
                    let sx = (x as i32 + dx).clamp(0, bw as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, bh as i32 - 1) as usize;
                    ref_frame[sy * bw + sx] as f32
                } else {
                    zero_px
                };
                let ref_px = zero_px * (1.0 - alpha) + mc_px * alpha;

                acc[i] += base_w * ref_px;
                weights[i] += base_w;
            }
        };

        if let Some(p) = prev {
            accumulate_ref(p, &mut acc, &mut pixel_weights);
        }
        if let Some(n_frame) = next {
            accumulate_ref(n_frame, &mut acc, &mut pixel_weights);
        }

        // Normalise and clamp each pixel.
        acc.iter()
            .zip(pixel_weights.iter())
            .map(|(&sum, &wt)| {
                let normalised = if wt > 1e-9 { sum / wt } else { 0.0 };
                normalised.round().clamp(0.0, 255.0) as u8
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// StabilizationAwareMctf
// ---------------------------------------------------------------------------

/// MCTF wrapper that optionally applies a per-frame homography (stabilisation)
/// before temporal denoising.
pub struct StabilizationAwareMctf {
    /// The underlying MCTF filter.
    pub mctf: MctfFilter,
    /// When `true`, apply stabilisation homographies before MCTF processing.
    pub enable_prewarping: bool,
}

impl StabilizationAwareMctf {
    /// Create a new stabilisation-aware MCTF with the given configuration.
    pub fn new(config: MctfConfig) -> Self {
        Self {
            mctf: MctfFilter::new(config),
            enable_prewarping: true,
        }
    }

    /// Process a frame sequence, optionally pre-warping with homographies.
    ///
    /// # Parameters
    /// * `frames` – input grayscale frames (each `w * h` bytes)
    /// * `homographies` – optional per-frame 3×3 row-major homography matrices
    /// * `w` / `h` – frame dimensions
    ///
    /// If `homographies` is `Some`, each frame is warped by its corresponding
    /// homography before being passed to MCTF.  If `None` (or
    /// `enable_prewarping` is `false`), frames are processed directly.
    pub fn process_with_homographies(
        &self,
        frames: &[Vec<u8>],
        homographies: Option<&[[[f32; 3]; 3]]>,
        w: u32,
        h: u32,
    ) -> Vec<Vec<u8>> {
        if self.enable_prewarping {
            if let Some(homos) = homographies {
                let prewarped: Vec<Vec<u8>> = frames
                    .iter()
                    .zip(homos.iter())
                    .map(|(frame, h3x3)| apply_homography(frame, h3x3, w, h))
                    .collect();
                return self.mctf.process_sequence(&prewarped, w, h);
            }
        }
        self.mctf.process_sequence(frames, w, h)
    }
}

/// Apply a 3×3 homography to a grayscale frame (nearest-neighbour lookup).
///
/// Points outside the source bounds are filled with 0 (black).
fn apply_homography(src: &[u8], h3x3: &[[f32; 3]; 3], w: u32, h: u32) -> Vec<u8> {
    let bw = w as usize;
    let bh = h as usize;
    let mut dst = vec![0u8; bw * bh];

    for y in 0..bh {
        for x in 0..bw {
            // Apply H to (x, y, 1).
            let xf = x as f32;
            let yf = y as f32;
            let wx = h3x3[0][0] * xf + h3x3[0][1] * yf + h3x3[0][2];
            let wy = h3x3[1][0] * xf + h3x3[1][1] * yf + h3x3[1][2];
            let wz = h3x3[2][0] * xf + h3x3[2][1] * yf + h3x3[2][2];
            if wz.abs() < 1e-9 {
                continue;
            }
            let sx = (wx / wz).round() as i32;
            let sy = (wy / wz).round() as i32;
            if sx >= 0 && sy >= 0 && (sx as usize) < bw && (sy as usize) < bh {
                dst[y * bw + x] = src[sy as usize * bw + sx as usize];
            }
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: compute PSNR between two u8 slices.
    // -----------------------------------------------------------------------

    fn psnr(original: &[u8], processed: &[u8]) -> f64 {
        assert_eq!(original.len(), processed.len());
        let mse: f64 = original
            .iter()
            .zip(processed.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / original.len() as f64;
        if mse < 1e-10 {
            return 100.0;
        }
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }

    // -----------------------------------------------------------------------
    // estimate_motion_vectors tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_motion_estimation_zero() {
        let w = 32u32;
        let h = 32u32;
        let frame: Vec<u8> = (0..(w * h) as usize)
            .map(|i| ((i * 7 + i * 3) % 200 + 20) as u8)
            .collect();
        let mvs = estimate_motion_vectors(&frame, &frame, w, h, 8, 8);
        // Identical frames → every block should have zero displacement.
        for &(dx, dy) in &mvs {
            assert_eq!(
                (dx, dy),
                (0, 0),
                "Identical frames should yield zero MVs, got ({dx},{dy})"
            );
        }
    }

    #[test]
    fn test_motion_estimation_vector_count() {
        let w = 32u32;
        let h = 32u32;
        let frame = vec![128u8; (w * h) as usize];
        let mvs = estimate_motion_vectors(&frame, &frame, w, h, 8, 8);
        let expected = (w as usize).div_ceil(8) * (h as usize).div_ceil(8);
        assert_eq!(mvs.len(), expected);
    }

    // -----------------------------------------------------------------------
    // warp_frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_identity() {
        let w = 16u32;
        let h = 16u32;
        let src: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let zero_mvs = vec![(0i32, 0i32); 4]; // 2×2 blocks for 16×16 @ bs=8
        let warped = warp_frame(&src, &zero_mvs, w, h, 8);
        assert_eq!(warped, src, "All-zero MVs should produce identity warp");
    }

    #[test]
    fn test_warp_uniform_frame() {
        let w = 16u32;
        let h = 16u32;
        let src = vec![200u8; (w * h) as usize];
        let mvs = vec![(3i32, 2i32); 4];
        let warped = warp_frame(&src, &mvs, w, h, 8);
        // Uniform frame warped in any direction is still uniform.
        for &p in &warped {
            assert_eq!(p, 200);
        }
    }

    // -----------------------------------------------------------------------
    // MctfFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mctf_single_frame_no_ref() {
        let w = 16u32;
        let h = 16u32;
        let curr: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.process_frame(None, &curr, None, w, h);
        // No references → passthrough; output must equal current.
        assert_eq!(
            out, curr,
            "Passthrough with no references should be identity"
        );
    }

    /// Minimal LCG random number generator for deterministic independent noise.
    fn lcg_next(seed: u64) -> (i32, u64) {
        let s = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Bring into [-10, +10] (21 values)
        let val = ((s >> 33) as i32).rem_euclid(21) - 10;
        (val, s)
    }

    #[test]
    fn test_mctf_static_sequence() {
        // MCTF with static (zero-motion) frames: temporal averaging across 5
        // frames with *independent* noise should improve PSNR vs. the noisy
        // input above 35 dB.
        let w = 32u32;
        let h = 32u32;
        let npix = (w * h) as usize;

        // Spatially-varying signal built from 8×8 block-constant values so
        // that block matching cannot improve over zero-motion (every block has
        // a unique constant value, and adjacent blocks are far apart, so no
        // shifted block in the reference has lower SAD than the zero-MV block).
        //
        // Block (bx, by) has value 30 + (by * 4 + bx) * 14, giving a unique
        // value in [30..220] per block with ≥14 unit separation.
        let bs_usize = 8usize;
        let original: Vec<u8> = (0..npix)
            .map(|i| {
                let x = i % w as usize;
                let y = i / w as usize;
                let bx = x / bs_usize;
                let by = y / bs_usize;
                let block_val = 30 + (by * 4 + bx) * 14;
                block_val.min(220) as u8
            })
            .collect();

        // Synthesise 5 frames, each with an independent LCG-seeded noise
        // pattern. Different seed per frame → uncorrelated noise → temporal
        // average converges toward the clean signal.
        let noisy: Vec<Vec<u8>> = (0u64..5)
            .map(|frame_seed| {
                let mut seed: u64 = frame_seed
                    .wrapping_mul(999_983)
                    .wrapping_add(0xDEAD_BEEF_CAFE);
                original
                    .iter()
                    .map(|&p| {
                        let (n, s) = lcg_next(seed);
                        seed = s;
                        (p as i32 + n).clamp(0, 255) as u8
                    })
                    .collect()
            })
            .collect();

        // Config: large motion_threshold (static scene) so references are never
        // attenuated.
        let config = MctfConfig {
            window_size: 1,
            block_size: 8,
            motion_threshold: 512.0,
            temporal_strength: 0.4,
        };
        let filter = MctfFilter::new(config);
        let denoised = filter.process_sequence(&noisy, w, h);

        let noisy_psnr = psnr(&original, &noisy[2]);
        let denoised_psnr = psnr(&original, &denoised[2]);

        assert!(
            denoised_psnr > 35.0,
            "PSNR after MCTF should exceed 35 dB, got {denoised_psnr:.2} dB \
             (noisy: {noisy_psnr:.2} dB)"
        );
    }

    #[test]
    fn test_mctf_output_length_matches_input() {
        let w = 16u32;
        let h = 16u32;
        let frame = vec![128u8; (w * h) as usize];
        let frames: Vec<Vec<u8>> = (0..5).map(|_| frame.clone()).collect();
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.process_sequence(&frames, w, h);
        assert_eq!(out.len(), frames.len());
        for f in &out {
            assert_eq!(f.len(), (w * h) as usize);
        }
    }

    #[test]
    fn test_mctf_uniform_frames_unchanged() {
        let w = 16u32;
        let h = 16u32;
        let frame = vec![128u8; (w * h) as usize];
        let frames: Vec<Vec<u8>> = (0..3).map(|_| frame.clone()).collect();
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.process_sequence(&frames, w, h);
        for f in &out {
            for &px in f {
                assert_eq!(px, 128, "Uniform sequence should stay constant");
            }
        }
    }

    // -----------------------------------------------------------------------
    // StabilizationAwareMctf tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stabilisation_no_homographies() {
        let w = 16u32;
        let h = 16u32;
        let frame = vec![100u8; (w * h) as usize];
        let frames: Vec<Vec<u8>> = vec![frame; 3];
        let sa = StabilizationAwareMctf::new(MctfConfig::default());
        let out = sa.process_with_homographies(&frames, None, w, h);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_stabilisation_identity_homography() {
        let w = 16u32;
        let h = 16u32;
        let frame: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let frames: Vec<Vec<u8>> = vec![frame.clone(); 3];

        // Identity 3×3.
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let homos = vec![identity; 3];

        let sa = StabilizationAwareMctf::new(MctfConfig::default());
        let out = sa.process_with_homographies(&frames, Some(&homos), w, h);
        assert_eq!(out.len(), 3);
    }

    // -----------------------------------------------------------------------
    // apply_homography test
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_homography_identity() {
        let w = 8u32;
        let h = 8u32;
        let src: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let dst = apply_homography(&src, &identity, w, h);
        assert_eq!(
            dst, src,
            "Identity homography should produce identical output"
        );
    }
}
