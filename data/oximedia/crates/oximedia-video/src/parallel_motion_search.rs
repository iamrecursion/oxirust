//! Parallel block-matching motion search using rayon.
//!
//! Provides [`ParallelMotionSearch`] which searches all blocks in a frame
//! simultaneously using rayon thread parallelism.  Three cost metrics are
//! supported: SAD, SSD, and SATD (Sum of Absolute Transformed Differences via
//! the integer 4×4 Hadamard transform).

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Cost metric used when comparing reference and current blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMetric {
    /// Sum of Absolute Differences — fast, linear cost.
    Sad,
    /// Sum of Squared Differences — penalises large outliers more heavily.
    Ssd,
    /// Sum of Absolute Transformed Differences (integer 4×4 Hadamard).
    Satd,
}

/// A motion vector produced by [`ParallelMotionSearch`].
///
/// `dx` and `dy` are integer-pel displacements (positive = right/down).
/// `cost` is the raw metric value at the best matching candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelMotionVector {
    /// Top-left X coordinate of the block in the current frame (pixels).
    pub block_x: u32,
    /// Top-left Y coordinate of the block in the current frame (pixels).
    pub block_y: u32,
    /// Horizontal displacement from the current block to the best match in
    /// the reference frame (positive = right).
    pub dx: i32,
    /// Vertical displacement from the current block to the best match in
    /// the reference frame (positive = down).
    pub dy: i32,
    /// Cost (SAD / SSD / SATD) at the best-matching candidate.
    pub cost: u32,
}

/// Parallel block-matching motion search.
///
/// All blocks in the current frame are searched simultaneously using rayon.
/// The search is a full exhaustive scan over the integer-pel grid within
/// `[-search_range, search_range]` in both axes.
///
/// # Example
/// ```rust
/// use oximedia_video::parallel_motion_search::{ParallelMotionSearch, MatchMetric};
///
/// let w = 32u32;
/// let h = 32u32;
/// let frame = vec![128u8; (w * h) as usize];
/// let searcher = ParallelMotionSearch::new(8, 4);
/// let mvs = searcher.search_frame(&frame, &frame, w, h);
/// assert_eq!(mvs.len(), searcher.num_blocks(w, h));
/// ```
pub struct ParallelMotionSearch {
    /// Block dimension in pixels (both width and height).
    pub block_size: u32,
    /// Half-range of the search window (pixels in each axis direction).
    pub search_range: i32,
    /// Cost metric used for block comparisons.
    pub metric: MatchMetric,
}

impl ParallelMotionSearch {
    /// Create a new `ParallelMotionSearch` with the given block size and
    /// search range.  The default metric is [`MatchMetric::Sad`].
    pub fn new(block_size: u32, search_range: i32) -> Self {
        Self {
            block_size,
            search_range,
            metric: MatchMetric::Sad,
        }
    }

    /// Return the total number of blocks that cover the frame.
    ///
    /// Only complete blocks are counted; partial blocks at the right/bottom
    /// edges are excluded.
    pub fn num_blocks(&self, width: u32, height: u32) -> usize {
        let cols = (width / self.block_size) as usize;
        let rows = (height / self.block_size) as usize;
        cols * rows
    }

    /// Search every block in `current` for its best match in `reference`.
    ///
    /// Returns one [`ParallelMotionVector`] per block, in raster scan order
    /// (left-to-right, top-to-bottom).
    ///
    /// Both `reference` and `current` must be grayscale (luma-only) buffers
    /// of exactly `width × height` bytes.
    pub fn search_frame(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<ParallelMotionVector> {
        use rayon::prelude::*;

        let bs = self.block_size;

        // Collect all (block_x, block_y) grid positions.
        let mut positions: Vec<(u32, u32)> = Vec::new();
        let mut by = 0u32;
        while by + bs <= height {
            let mut bx = 0u32;
            while bx + bs <= width {
                positions.push((bx, by));
                bx += bs;
            }
            by += bs;
        }

        let metric = self.metric;
        let sr = self.search_range;

        let mut vectors: Vec<ParallelMotionVector> = positions
            .par_iter()
            .map(|&(bx, by)| {
                search_block(reference, current, width, height, bx, by, bs, sr, metric)
            })
            .collect();

        // Restore raster order (par_iter may reorder).
        vectors.sort_by_key(|mv| (mv.block_y, mv.block_x));
        vectors
    }
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

/// Search for the best-matching candidate for the block at (`bx`, `by`) in
/// `current` by scanning all integer-pel offsets in `[-sr, sr]² within the
/// reference frame bounds.
fn search_block(
    reference: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    bs: u32,
    sr: i32,
    metric: MatchMetric,
) -> ParallelMotionVector {
    let w = width as i32;
    let h = height as i32;
    let b = bs as i32;

    let mut best_cost = u32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let ref_x = (bx as i32 + dx).clamp(0, w - b);
            let ref_y = (by as i32 + dy).clamp(0, h - b);

            let cost = compute_block_cost(
                current,
                reference,
                width,
                bx,
                by,
                ref_x as u32,
                ref_y as u32,
                bs,
                metric,
            );

            if cost < best_cost {
                best_cost = cost;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    ParallelMotionVector {
        block_x: bx,
        block_y: by,
        dx: best_dx,
        dy: best_dy,
        cost: best_cost,
    }
}

/// Compute the cost between a block in `current` at (`cx`, `cy`) and a block
/// in `reference` at (`rx`, `ry`), both of size `bs × bs`, using `metric`.
fn compute_block_cost(
    current: &[u8],
    reference: &[u8],
    width: u32,
    cx: u32,
    cy: u32,
    rx: u32,
    ry: u32,
    bs: u32,
    metric: MatchMetric,
) -> u32 {
    let w = width as usize;
    let bs_usize = bs as usize;

    match metric {
        MatchMetric::Sad => {
            let mut acc = 0u32;
            for row in 0..bs_usize {
                let cur_row = (cy as usize + row) * w + cx as usize;
                let ref_row = (ry as usize + row) * w + rx as usize;
                for col in 0..bs_usize {
                    let c = current.get(cur_row + col).copied().unwrap_or(0) as i32;
                    let r = reference.get(ref_row + col).copied().unwrap_or(0) as i32;
                    acc += (c - r).unsigned_abs();
                }
            }
            acc
        }
        MatchMetric::Ssd => {
            let mut acc = 0u64;
            for row in 0..bs_usize {
                let cur_row = (cy as usize + row) * w + cx as usize;
                let ref_row = (ry as usize + row) * w + rx as usize;
                for col in 0..bs_usize {
                    let c = current.get(cur_row + col).copied().unwrap_or(0) as i64;
                    let r = reference.get(ref_row + col).copied().unwrap_or(0) as i64;
                    let d = c - r;
                    acc += (d * d) as u64;
                }
            }
            acc.min(u32::MAX as u64) as u32
        }
        MatchMetric::Satd => compute_satd(current, reference, width, cx, cy, rx, ry, bs),
    }
}

/// Compute SATD by tiling the `bs × bs` difference block into 4×4 sub-blocks,
/// applying the integer 4×4 Hadamard transform to each, and summing absolute
/// Hadamard coefficients.
///
/// For blocks smaller than 4×4 in any dimension, a direct SAD is returned.
fn compute_satd(
    current: &[u8],
    reference: &[u8],
    width: u32,
    cx: u32,
    cy: u32,
    rx: u32,
    ry: u32,
    bs: u32,
) -> u32 {
    let w = width as usize;
    let bs_usize = bs as usize;

    if bs_usize < 4 {
        // Fall back to SAD for sub-4×4 blocks.
        return compute_block_cost(
            current,
            reference,
            width,
            cx,
            cy,
            rx,
            ry,
            bs,
            MatchMetric::Sad,
        );
    }

    let mut total_satd = 0u32;

    // Tile into 4×4 sub-blocks.
    let tile_rows = bs_usize / 4;
    let tile_cols = bs_usize / 4;

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let mut diff = [0i32; 16];
            for r in 0..4usize {
                for c in 0..4usize {
                    let global_row = cy as usize + tr * 4 + r;
                    let global_col_cur = cx as usize + tc * 4 + c;
                    let global_col_ref = rx as usize + tc * 4 + c;
                    let ref_row_offset = ry as usize + tr * 4 + r;

                    let cur_px = current
                        .get(global_row * w + global_col_cur)
                        .copied()
                        .unwrap_or(0) as i32;
                    let ref_px = reference
                        .get(ref_row_offset * w + global_col_ref)
                        .copied()
                        .unwrap_or(0) as i32;

                    diff[r * 4 + c] = cur_px - ref_px;
                }
            }
            total_satd += hadamard4x4(&diff);
        }
    }

    total_satd
}

/// Apply the integer 4×4 Hadamard transform to `block` (row-major, 16
/// elements) and return the sum of absolute Hadamard coefficients.
///
/// The butterfly is the standard H.264 / JVT integer Hadamard:
/// two passes of the 1-D butterfly `[a+b, a-b, a-b, a+b]` applied
/// horizontally then vertically.
fn hadamard4x4(block: &[i32; 16]) -> u32 {
    let mut tmp = *block;

    // Horizontal pass (4 rows of 4).
    for row in 0..4usize {
        let o = row * 4;
        let a0 = tmp[o] + tmp[o + 1];
        let a1 = tmp[o] - tmp[o + 1];
        let a2 = tmp[o + 2] + tmp[o + 3];
        let a3 = tmp[o + 2] - tmp[o + 3];

        tmp[o] = a0 + a2;
        tmp[o + 1] = a1 + a3;
        tmp[o + 2] = a0 - a2;
        tmp[o + 3] = a1 - a3;
    }

    // Vertical pass (4 columns of 4).
    for col in 0..4usize {
        let a0 = tmp[col] + tmp[4 + col];
        let a1 = tmp[col] - tmp[4 + col];
        let a2 = tmp[8 + col] + tmp[12 + col];
        let a3 = tmp[8 + col] - tmp[12 + col];

        tmp[col] = a0 + a2;
        tmp[4 + col] = a1 + a3;
        tmp[8 + col] = a0 - a2;
        tmp[12 + col] = a1 - a3;
    }

    tmp.iter().map(|&v| v.unsigned_abs()).sum()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width * height) as usize]
    }

    fn ramp_frame(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    // 1. num_blocks: 32×32 / block_size=8 → 16 blocks
    #[test]
    fn test_num_blocks_basic() {
        let s = ParallelMotionSearch::new(8, 4);
        assert_eq!(s.num_blocks(32, 32), 16);
    }

    // 2. num_blocks: partial blocks excluded
    #[test]
    fn test_num_blocks_partial_excluded() {
        let s = ParallelMotionSearch::new(8, 4);
        // 33×33: only 4×4 = 16 complete 8×8 blocks
        assert_eq!(s.num_blocks(33, 33), 16);
    }

    // 3. search_frame: returns correct block count for 32×32 / bs=8
    #[test]
    fn test_search_frame_count() {
        let s = ParallelMotionSearch::new(8, 4);
        let frame = ramp_frame(32, 32);
        let mvs = s.search_frame(&frame, &frame, 32, 32);
        assert_eq!(mvs.len(), 16);
    }

    // 4. SAD: identical frames → all costs 0
    // Note: on an identical frame every candidate (dx, dy) yields cost=0, so
    // the best dx/dy may be anything in [-search_range, search_range]; only
    // the cost matters here.
    #[test]
    fn test_sad_identical_frames_zero_cost() {
        let mut s = ParallelMotionSearch::new(8, 4);
        s.metric = MatchMetric::Sad;
        let frame = ramp_frame(32, 32);
        let mvs = s.search_frame(&frame, &frame, 32, 32);
        for mv in &mvs {
            assert_eq!(mv.cost, 0, "SAD cost should be 0 on identical frames");
        }
    }

    // 5. SSD: identical frames → all costs 0
    #[test]
    fn test_ssd_identical_frames_zero_cost() {
        let mut s = ParallelMotionSearch::new(8, 4);
        s.metric = MatchMetric::Ssd;
        let frame = ramp_frame(32, 32);
        let mvs = s.search_frame(&frame, &frame, 32, 32);
        for mv in &mvs {
            assert_eq!(mv.cost, 0, "SSD cost should be 0 on identical frames");
        }
    }

    // 6. SATD: identical frames → all costs 0
    #[test]
    fn test_satd_identical_frames_zero_cost() {
        let mut s = ParallelMotionSearch::new(8, 4);
        s.metric = MatchMetric::Satd;
        let frame = ramp_frame(32, 32);
        let mvs = s.search_frame(&frame, &frame, 32, 32);
        for mv in &mvs {
            assert_eq!(mv.cost, 0, "SATD cost should be 0 on identical frames");
        }
    }

    // 7. search_frame: output is in raster order
    #[test]
    fn test_search_frame_raster_order() {
        let s = ParallelMotionSearch::new(8, 4);
        let frame = ramp_frame(32, 32);
        let mvs = s.search_frame(&frame, &frame, 32, 32);
        for w in mvs.windows(2) {
            let a = (w[0].block_y, w[0].block_x);
            let b = (w[1].block_y, w[1].block_x);
            assert!(a < b, "not in raster order: {:?} >= {:?}", a, b);
        }
    }

    // 8. hadamard4x4: all-zero → 0
    #[test]
    fn test_hadamard4x4_zero_block() {
        let block = [0i32; 16];
        assert_eq!(hadamard4x4(&block), 0);
    }

    // 9. hadamard4x4: DC-only (constant block) — only DC coefficient is non-zero
    #[test]
    fn test_hadamard4x4_dc_only() {
        // A constant block: all values = 1.  After Hadamard the DC coefficient
        // should be 16 and all other coefficients 0.
        let block = [1i32; 16];
        let satd = hadamard4x4(&block);
        // DC = sum = 16; AC = 0 → total absolute sum = 16
        assert_eq!(satd, 16);
    }

    // 10. SAD metric: known cost on different constant frames
    #[test]
    fn test_sad_known_cost() {
        let mut s = ParallelMotionSearch::new(8, 0); // search_range = 0, only (0,0)
        s.metric = MatchMetric::Sad;
        let cur = flat_frame(8, 8, 10);
        let ref_f = flat_frame(8, 8, 20);
        let mvs = s.search_frame(&ref_f, &cur, 8, 8);
        assert_eq!(mvs.len(), 1);
        // 64 pixels × |10 - 20| = 640
        assert_eq!(mvs[0].cost, 640);
        assert_eq!(mvs[0].dx, 0);
        assert_eq!(mvs[0].dy, 0);
    }

    // 11. SSD metric: known cost on different constant frames
    #[test]
    fn test_ssd_known_cost() {
        let mut s = ParallelMotionSearch::new(4, 0);
        s.metric = MatchMetric::Ssd;
        let cur = flat_frame(4, 4, 10);
        let ref_f = flat_frame(4, 4, 20);
        let mvs = s.search_frame(&ref_f, &cur, 4, 4);
        // 16 pixels × (10-20)² = 16 * 100 = 1600
        assert_eq!(mvs[0].cost, 1600);
    }

    // 12. search_frame: finds correct translation
    // Build a 32×32 frame where a bright 8×8 block is at position (8,0) in
    // the reference and at (0,0) in the current.  With search_range ≥ 8 the
    // searcher should find dx=8 (or cost=0) for the first block.
    #[test]
    fn test_search_frame_known_translation() {
        let width = 32u32;
        let height = 32u32;
        let bs = 8u32;
        let fill = 50u8;
        let bright = 200u8;

        let mut reference = flat_frame(width, height, fill);
        let mut current = flat_frame(width, height, fill);

        // Bright block in current at (0, 0).
        for row in 0..bs as usize {
            for col in 0..bs as usize {
                current[row * width as usize + col] = bright;
            }
        }
        // Matching block in reference at (8, 0).
        for row in 0..bs as usize {
            for col in 0..bs as usize {
                reference[row * width as usize + bs as usize + col] = bright;
            }
        }

        let mut s = ParallelMotionSearch::new(bs, 8);
        s.metric = MatchMetric::Sad;
        let mvs = s.search_frame(&reference, &current, width, height);

        // First block should have zero cost (exact match found).
        let first = &mvs[0];
        assert_eq!(
            first.cost, 0,
            "expected SAD=0 at best match, got {}",
            first.cost
        );
    }

    // 13. SATD on a 4×4 constant-difference block is consistent with SAD
    #[test]
    fn test_satd_vs_sad_on_4x4_block() {
        // For a block where every difference is the same constant D,
        // SATD DC coefficient = 16*D, all AC = 0 → SATD = 16*D = SAD.
        let cur = flat_frame(4, 4, 30);
        let ref_f = flat_frame(4, 4, 10);

        let mut s_sad = ParallelMotionSearch::new(4, 0);
        s_sad.metric = MatchMetric::Sad;
        let mut s_satd = ParallelMotionSearch::new(4, 0);
        s_satd.metric = MatchMetric::Satd;

        let sad_mvs = s_sad.search_frame(&ref_f, &cur, 4, 4);
        let satd_mvs = s_satd.search_frame(&ref_f, &cur, 4, 4);

        assert_eq!(
            sad_mvs[0].cost, satd_mvs[0].cost,
            "SAD and SATD must agree for constant-difference block"
        );
    }
}
