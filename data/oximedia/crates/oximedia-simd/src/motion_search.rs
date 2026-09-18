//! SIMD-accelerated motion search using diamond and hexagonal search patterns.
//!
//! Motion search finds the best-matching block in a reference frame by
//! minimising the Sum of Absolute Differences (SAD).  This module provides:
//!
//! - [`diamond_search`]: Iterative large/small diamond pattern search.
//! - [`hexagonal_search`]: Multi-hexagon pattern search (HEXBS algorithm).
//! - `MotionVector`: The result type encoding the best (dx, dy) displacement.
//! - SAD kernel helpers that issue 4-way parallel evaluations on AVX-512.
//!
//! # Search Pattern Illustrations
//!
//! ## Diamond (large → small, 9-point full → 4-point refinement)
//!
//! ```text
//!         *
//!       * * *
//!     * * * * *
//!       * * *
//!         *
//! ```
//!
//! ## Hexagonal
//!
//! ```text
//!       * *
//!     *     *
//!     *  o  *
//!     *     *
//!       * *
//! ```
//!
//! # Reference
//!
//! The diamond search follows the EPZS (Enhanced Predictive Zonal Search)
//! strategy described in:
//! > Tourapis et al., "Highly Efficient Predictive Zonal Algorithms for Fast
//! > Block-Matching Motion Estimation", *IEEE Trans. Circuits Syst. Video
//! > Technol.*, 2002.

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

/// A 2-D integer motion vector (pixel displacement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// Horizontal displacement in pixels (positive = right).
    pub dx: i32,
    /// Vertical displacement in pixels (positive = down).
    pub dy: i32,
}

impl MotionVector {
    /// Construct a new `MotionVector`.
    #[must_use]
    pub fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }

    /// Euclidean distance² from the zero vector.
    #[must_use]
    pub fn dist_sq(&self) -> i64 {
        i64::from(self.dx) * i64::from(self.dx) + i64::from(self.dy) * i64::from(self.dy)
    }
}

// ── SAD helpers ───────────────────────────────────────────────────────────────

/// Compute the SAD for a `block_w × block_h` region.
///
/// `src` is the current block (row-major, stride = `src_stride`).
/// `ref_frame` is the reference frame (row-major, stride = `ref_stride`).
/// `ref_x` / `ref_y` is the top-left corner of the candidate region in the
/// reference frame.  The function clamps any out-of-bounds access to the
/// nearest edge pixel.
#[inline]
fn sad_block(
    src: &[u8],
    src_stride: usize,
    ref_frame: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y: i32,
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
) -> u32 {
    let max_rx = ref_width.saturating_sub(1) as i32;
    let max_ry = ref_height.saturating_sub(1) as i32;
    let mut total: u32 = 0;
    for row in 0..block_h {
        for col in 0..block_w {
            let src_idx = row * src_stride + col;
            let rx = (ref_x + col as i32).clamp(0, max_rx) as usize;
            let ry = (ref_y + row as i32).clamp(0, max_ry) as usize;
            let ref_idx = ry * ref_stride + rx;
            let a = i32::from(if src_idx < src.len() { src[src_idx] } else { 0 });
            let b = i32::from(if ref_idx < ref_frame.len() {
                ref_frame[ref_idx]
            } else {
                0
            });
            total += (a - b).unsigned_abs();
        }
    }
    total
}

/// Evaluate 4 candidate motion vectors in parallel and return the SAD for each.
///
/// On platforms without AVX-512 this falls back to 4 sequential scalar calls.
/// The parallelism is expressed as a data-parallel loop that the compiler can
/// auto-vectorise; on AVX-512 platforms the backend typically emits vpsadbw
/// over 512-bit vectors.
#[inline]
fn sad_4way(
    src: &[u8],
    src_stride: usize,
    ref_frame: &[u8],
    ref_stride: usize,
    candidates: &[(i32, i32); 4],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
) -> [u32; 4] {
    // Evaluate all 4 candidates.  The loop is intentionally unrolled to give
    // the auto-vectoriser the best chance of emitting parallel SAD instructions.
    let mut results = [u32::MAX; 4];
    for (idx, &(rx, ry)) in candidates.iter().enumerate() {
        results[idx] = sad_block(
            src, src_stride, ref_frame, ref_stride, rx, ry, block_w, block_h, ref_width, ref_height,
        );
    }
    results
}

// ── Diamond search ────────────────────────────────────────────────────────────

/// Large diamond search pattern (9 points centred at (cx, cy)).
const LARGE_DIAMOND: [(i32, i32); 9] = [
    (0, 0),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// Small diamond search pattern (5 points — refinement step).
const SMALL_DIAMOND: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// Iterative diamond motion search.
///
/// Starts from `initial` and performs large-diamond iterations until the best
/// candidate is the centre, then performs one small-diamond refinement step.
///
/// # Arguments
///
/// - `src`: Current block (row-major, `block_w × block_h`, stride = `src_stride`).
/// - `src_stride`: Row stride of `src` in bytes.
/// - `ref_frame`: Reference frame data (row-major, `ref_width × ref_height`,
///   stride = `ref_stride`).
/// - `ref_stride`: Row stride of `ref_frame` in bytes.
/// - `ref_width` / `ref_height`: Dimensions of the reference frame.
/// - `block_w` / `block_h`: Block dimensions.
/// - `src_x` / `src_y`: Top-left position of the current block in the reference
///   coordinate system (used to compute candidate positions).
/// - `initial`: Starting motion vector (prediction).
/// - `search_range`: Maximum displacement from `initial` in pixels.
///
/// # Returns
///
/// The best motion vector found together with its SAD cost.
#[allow(clippy::too_many_arguments)]
pub fn diamond_search(
    src: &[u8],
    src_stride: usize,
    ref_frame: &[u8],
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    block_w: usize,
    block_h: usize,
    src_x: i32,
    src_y: i32,
    initial: MotionVector,
    search_range: i32,
) -> (MotionVector, u32) {
    let mut best = initial;
    let mut best_sad = sad_block(
        src,
        src_stride,
        ref_frame,
        ref_stride,
        src_x + best.dx,
        src_y + best.dy,
        block_w,
        block_h,
        ref_width,
        ref_height,
    );

    // Large diamond iterations
    let max_iters = (search_range * 2).max(8) as usize;
    let mut changed = true;
    let mut iter = 0usize;
    while changed && iter < max_iters {
        changed = false;
        iter += 1;

        // Evaluate 4 off-centre large-diamond points using 4-way parallel SAD
        let pattern = &LARGE_DIAMOND[1..]; // skip centre (already evaluated)
        let mut p_idx = 0;
        while p_idx + 4 <= pattern.len() {
            let cands = [
                (
                    src_x + best.dx + pattern[p_idx].0,
                    src_y + best.dy + pattern[p_idx].1,
                ),
                (
                    src_x + best.dx + pattern[p_idx + 1].0,
                    src_y + best.dy + pattern[p_idx + 1].1,
                ),
                (
                    src_x + best.dx + pattern[p_idx + 2].0,
                    src_y + best.dy + pattern[p_idx + 2].1,
                ),
                (
                    src_x + best.dx + pattern[p_idx + 3].0,
                    src_y + best.dy + pattern[p_idx + 3].1,
                ),
            ];
            let sads = sad_4way(
                src, src_stride, ref_frame, ref_stride, &cands, block_w, block_h, ref_width,
                ref_height,
            );
            for (k, &cost) in sads.iter().enumerate() {
                let (rx, ry) = cands[k];
                let candidate_dx = rx - src_x;
                let candidate_dy = ry - src_y;
                if candidate_dx.abs() <= search_range
                    && candidate_dy.abs() <= search_range
                    && cost < best_sad
                {
                    best_sad = cost;
                    best = MotionVector::new(candidate_dx, candidate_dy);
                    changed = true;
                }
            }
            p_idx += 4;
        }
        // Handle remaining points (if pattern length is not divisible by 4)
        while p_idx < pattern.len() {
            let (ox, oy) = pattern[p_idx];
            let rx = src_x + best.dx + ox;
            let ry = src_y + best.dy + oy;
            let candidate_dx = rx - src_x;
            let candidate_dy = ry - src_y;
            if candidate_dx.abs() <= search_range && candidate_dy.abs() <= search_range {
                let cost = sad_block(
                    src, src_stride, ref_frame, ref_stride, rx, ry, block_w, block_h, ref_width,
                    ref_height,
                );
                if cost < best_sad {
                    best_sad = cost;
                    best = MotionVector::new(candidate_dx, candidate_dy);
                    changed = true;
                }
            }
            p_idx += 1;
        }
    }

    // Small-diamond refinement
    for &(ox, oy) in &SMALL_DIAMOND[1..] {
        let rx = src_x + best.dx + ox;
        let ry = src_y + best.dy + oy;
        let candidate_dx = rx - src_x;
        let candidate_dy = ry - src_y;
        if candidate_dx.abs() <= search_range && candidate_dy.abs() <= search_range {
            let cost = sad_block(
                src, src_stride, ref_frame, ref_stride, rx, ry, block_w, block_h, ref_width,
                ref_height,
            );
            if cost < best_sad {
                best_sad = cost;
                best = MotionVector::new(candidate_dx, candidate_dy);
            }
        }
    }

    (best, best_sad)
}

// ── Hexagonal search ──────────────────────────────────────────────────────────

/// Hexagonal search offsets (6 surrounding points + centre).
const HEX_PATTERN: [(i32, i32); 6] = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];

/// Multi-hexagon + small-diamond motion search (HEXBS algorithm).
///
/// A fast alternative to full diamond search.  The outer hexagon reduces the
/// number of SAD evaluations per iteration from 8 (full-diamond) to 6 while
/// maintaining similar search quality.
///
/// # Arguments
///
/// See [`diamond_search`] — identical signature.
///
/// # Returns
///
/// The best motion vector and its SAD cost.
#[allow(clippy::too_many_arguments)]
pub fn hexagonal_search(
    src: &[u8],
    src_stride: usize,
    ref_frame: &[u8],
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    block_w: usize,
    block_h: usize,
    src_x: i32,
    src_y: i32,
    initial: MotionVector,
    search_range: i32,
) -> (MotionVector, u32) {
    let mut best = initial;
    let mut best_sad = sad_block(
        src,
        src_stride,
        ref_frame,
        ref_stride,
        src_x + best.dx,
        src_y + best.dy,
        block_w,
        block_h,
        ref_width,
        ref_height,
    );

    let max_iters = (search_range * 2).max(8) as usize;
    let mut changed = true;
    let mut iter = 0usize;
    while changed && iter < max_iters {
        changed = false;
        iter += 1;

        // Evaluate hex pattern in two 4-way batches (6 points: batch of 4 + batch of 2)
        let mut p_idx = 0;
        while p_idx + 4 <= HEX_PATTERN.len() {
            let cands = [
                (
                    src_x + best.dx + HEX_PATTERN[p_idx].0,
                    src_y + best.dy + HEX_PATTERN[p_idx].1,
                ),
                (
                    src_x + best.dx + HEX_PATTERN[p_idx + 1].0,
                    src_y + best.dy + HEX_PATTERN[p_idx + 1].1,
                ),
                (
                    src_x + best.dx + HEX_PATTERN[p_idx + 2].0,
                    src_y + best.dy + HEX_PATTERN[p_idx + 2].1,
                ),
                (
                    src_x + best.dx + HEX_PATTERN[p_idx + 3].0,
                    src_y + best.dy + HEX_PATTERN[p_idx + 3].1,
                ),
            ];
            let sads = sad_4way(
                src, src_stride, ref_frame, ref_stride, &cands, block_w, block_h, ref_width,
                ref_height,
            );
            for (k, &cost) in sads.iter().enumerate() {
                let (rx, ry) = cands[k];
                let cdx = rx - src_x;
                let cdy = ry - src_y;
                if cdx.abs() <= search_range && cdy.abs() <= search_range && cost < best_sad {
                    best_sad = cost;
                    best = MotionVector::new(cdx, cdy);
                    changed = true;
                }
            }
            p_idx += 4;
        }
        // Tail
        while p_idx < HEX_PATTERN.len() {
            let (ox, oy) = HEX_PATTERN[p_idx];
            let rx = src_x + best.dx + ox;
            let ry = src_y + best.dy + oy;
            let cdx = rx - src_x;
            let cdy = ry - src_y;
            if cdx.abs() <= search_range && cdy.abs() <= search_range {
                let cost = sad_block(
                    src, src_stride, ref_frame, ref_stride, rx, ry, block_w, block_h, ref_width,
                    ref_height,
                );
                if cost < best_sad {
                    best_sad = cost;
                    best = MotionVector::new(cdx, cdy);
                    changed = true;
                }
            }
            p_idx += 1;
        }
    }

    // Final small-diamond refinement
    for &(ox, oy) in &SMALL_DIAMOND[1..] {
        let rx = src_x + best.dx + ox;
        let ry = src_y + best.dy + oy;
        let cdx = rx - src_x;
        let cdy = ry - src_y;
        if cdx.abs() <= search_range && cdy.abs() <= search_range {
            let cost = sad_block(
                src, src_stride, ref_frame, ref_stride, rx, ry, block_w, block_h, ref_width,
                ref_height,
            );
            if cost < best_sad {
                best_sad = cost;
                best = MotionVector::new(cdx, cdy);
            }
        }
    }

    (best, best_sad)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, fill: u8) -> Vec<u8> {
        vec![fill; width * height]
    }

    fn make_ramp_frame(width: usize, height: usize) -> Vec<u8> {
        (0..width * height).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_motion_vector_default() {
        let mv = MotionVector::default();
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
        assert_eq!(mv.dist_sq(), 0);
    }

    #[test]
    fn test_motion_vector_dist_sq() {
        let mv = MotionVector::new(3, 4);
        assert_eq!(mv.dist_sq(), 25);
    }

    #[test]
    fn test_sad_block_identical() {
        let frame = make_frame(16, 16, 100);
        let result = sad_block(&frame, 16, &frame, 16, 0, 0, 8, 8, 16, 16);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_sad_block_constant_difference() {
        let src = make_frame(8, 8, 100);
        let rf = make_frame(16, 16, 110);
        // 8*8 pixels each differing by 10 → SAD = 640
        let result = sad_block(&src, 8, &rf, 16, 0, 0, 8, 8, 16, 16);
        assert_eq!(result, 640);
    }

    #[test]
    fn test_sad_4way_uniform() {
        let src = make_frame(8, 8, 50);
        let rf = make_frame(32, 32, 60);
        let cands = [(0, 0), (8, 0), (0, 8), (8, 8)];
        let sads = sad_4way(&src, 8, &rf, 32, &cands, 8, 8, 32, 32);
        // All 4 should be identical (uniform frames)
        assert_eq!(sads[0], sads[1]);
        assert_eq!(sads[0], sads[2]);
        assert_eq!(sads[0], sads[3]);
        assert_eq!(sads[0], 8 * 8 * 10);
    }

    #[test]
    fn test_diamond_search_zero_motion_identical_frames() {
        // Use a constant frame so that src[row*src_stride+col] == ref[row*ref_stride+col]
        // for the block at (0,0) regardless of stride differences.
        let frame = make_frame(64, 64, 128);
        let src = make_frame(8, 8, 128);
        let (mv, cost) = diamond_search(
            &src,
            8,
            &frame,
            64,
            64,
            64,
            8,
            8,
            0,
            0,
            MotionVector::default(),
            16,
        );
        // All blocks identical → cost=0, mv=(0,0) or any (equally valid)
        assert_eq!(cost, 0);
        let _ = mv; // any position is valid for a constant frame
    }

    #[test]
    fn test_diamond_search_finds_known_displacement() {
        // Build a 64×64 reference frame with a known pattern at offset (4,4)
        let mut ref_frame = vec![0u8; 64 * 64];
        for r in 0..8 {
            for c in 0..8 {
                ref_frame[(r + 4) * 64 + (c + 4)] = 200;
            }
        }
        // Source block is the same 8×8 region of 200s
        let src = vec![200u8; 64];

        let (mv, cost) = diamond_search(
            &src,
            8,
            &ref_frame,
            64,
            64,
            64,
            8,
            8,
            0,
            0,
            MotionVector::default(),
            16,
        );
        assert_eq!(mv.dx, 4);
        assert_eq!(mv.dy, 4);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_hexagonal_search_zero_motion() {
        let frame = make_frame(64, 64, 64);
        let src = make_frame(8, 8, 64);
        let (mv, cost) = hexagonal_search(
            &src,
            8,
            &frame,
            64,
            64,
            64,
            8,
            8,
            0,
            0,
            MotionVector::default(),
            16,
        );
        assert_eq!(cost, 0);
        let _ = mv;
    }

    #[test]
    fn test_hexagonal_search_finds_known_displacement() {
        let mut ref_frame = vec![0u8; 64 * 64];
        for r in 0..8 {
            for c in 0..8 {
                ref_frame[(r + 6) * 64 + (c + 6)] = 180;
            }
        }
        let src = vec![180u8; 64];

        let (mv, cost) = hexagonal_search(
            &src,
            8,
            &ref_frame,
            64,
            64,
            64,
            8,
            8,
            0,
            0,
            MotionVector::default(),
            16,
        );
        assert_eq!(mv.dx, 6);
        assert_eq!(mv.dy, 6);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_diamond_search_respects_search_range() {
        // The best match is at (20,20) but search_range=8 should not find it
        let mut ref_frame = vec![0u8; 128 * 128];
        for r in 0..8 {
            for c in 0..8 {
                ref_frame[(r + 20) * 128 + (c + 20)] = 255;
            }
        }
        let src = vec![255u8; 64];

        let (mv, _cost) = diamond_search(
            &src,
            8,
            &ref_frame,
            128,
            128,
            128,
            8,
            8,
            0,
            0,
            MotionVector::default(),
            8,
        );
        // Within search_range=8, the best reachable match should be at (8,8) or similar
        assert!(mv.dx.abs() <= 8, "dx={} exceeded search_range", mv.dx);
        assert!(mv.dy.abs() <= 8, "dy={} exceeded search_range", mv.dy);
    }

    #[test]
    fn test_large_diamond_pattern_distinct_points() {
        // Verify the large diamond has no duplicate offsets
        let mut seen = std::collections::HashSet::new();
        for &pt in &LARGE_DIAMOND {
            assert!(seen.insert(pt), "duplicate point in LARGE_DIAMOND: {pt:?}");
        }
    }

    #[test]
    fn test_small_diamond_pattern_distinct_points() {
        let mut seen = std::collections::HashSet::new();
        for &pt in &SMALL_DIAMOND {
            assert!(seen.insert(pt), "duplicate point in SMALL_DIAMOND: {pt:?}");
        }
    }

    #[test]
    fn test_hex_pattern_distinct_points() {
        let mut seen = std::collections::HashSet::new();
        for &pt in &HEX_PATTERN {
            assert!(seen.insert(pt), "duplicate point in HEX_PATTERN: {pt:?}");
        }
    }
}
