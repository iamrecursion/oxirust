//! Advanced motion search algorithms with thread-local scratch buffers.
//!
//! The parallel search path uses `MOTION_SCRATCH` (a `thread_local!` `RefCell`)
//! to avoid per-call heap allocation when rayon dispatches blocks in parallel.
//! Each thread owns exactly one `MotionScratch` instance; the buffers are grown
//! on demand via `resize` and then reused across calls within the same thread.
//!
//! The sequential (non-parallel) paths are left unchanged so that their
//! allocation behaviour and code paths remain easy to read and verify.

use std::cell::RefCell;

use crate::{OptimizationLevel, OptimizerConfig};
use oximedia_core::OxiResult;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Thread-local scratch storage
// ─────────────────────────────────────────────────────────────────────────────

/// Per-thread scratch buffers for the parallel motion search path.
///
/// The buffers start empty and are grown lazily via `Vec::resize`.  Growth
/// only ever appends capacity — it never shrinks — so steady-state allocation
/// is zero after the first large block has been processed by a thread.
struct MotionScratch {
    /// Scratch buffer for SAD computation (one entry per pixel in a block).
    sad_buffer: Vec<u32>,
    /// Per-candidate MV costs, length = number of search candidates.
    cost_buffer: Vec<f32>,
    /// Search candidate motion-vector list (integer-pel, pre-scale).
    candidate_mvs: Vec<(i32, i32)>,
}

thread_local! {
    static MOTION_SCRATCH: RefCell<MotionScratch> = const { RefCell::new(MotionScratch {
        sad_buffer: Vec::new(),
        cost_buffer: Vec::new(),
        candidate_mvs: Vec::new(),
    }) };
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Motion vector (quarter-pel units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (in quarter-pel units).
    pub x: i16,
    /// Vertical component (in quarter-pel units).
    pub y: i16,
}

impl MotionVector {
    /// Creates a new motion vector.
    #[must_use]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }
}

/// Motion search algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAlgorithm {
    /// Full search (exhaustive).
    Full,
    /// Diamond search.
    Diamond,
    /// Hexagon search.
    Hexagon,
    /// Test Zone Search.
    TzSearch,
    /// Enhanced Predictive Zonal Search.
    Epzs,
    /// Uneven Multi-Hexagon.
    Umh,
}

/// Motion optimizer for advanced motion estimation.
pub struct MotionOptimizer {
    algorithm: SearchAlgorithm,
    search_range: i16,
    #[allow(dead_code)]
    subpel_enabled: bool,
}

impl MotionOptimizer {
    /// Creates a new motion optimizer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let (algorithm, search_range) = match config.level {
            OptimizationLevel::Fast => (SearchAlgorithm::Diamond, 16),
            OptimizationLevel::Medium => (SearchAlgorithm::Hexagon, 32),
            OptimizationLevel::Slow => (SearchAlgorithm::TzSearch, 64),
            OptimizationLevel::Placebo => (SearchAlgorithm::Umh, 128),
        };

        Ok(Self {
            algorithm,
            search_range,
            subpel_enabled: config.level != OptimizationLevel::Fast,
        })
    }

    /// Performs motion search.
    #[allow(dead_code)]
    #[must_use]
    pub fn search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        match self.algorithm {
            SearchAlgorithm::Full => self.full_search(src, reference, width, height),
            SearchAlgorithm::Diamond => {
                self.diamond_search(src, reference, width, height, predictor)
            }
            SearchAlgorithm::Hexagon => {
                self.hexagon_search(src, reference, width, height, predictor)
            }
            SearchAlgorithm::TzSearch => self.tz_search(src, reference, width, height, predictor),
            SearchAlgorithm::Epzs => self.epzs_search(src, reference, width, height, predictor),
            SearchAlgorithm::Umh => self.umh_search(src, reference, width, height, predictor),
        }
    }

    // ── Full search ───────────────────────────────────────────────────────────

    fn full_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
    ) -> MotionSearchResult {
        let mut best_mv = MotionVector::zero();
        let mut best_cost = self.calculate_cost(src, reference, best_mv);

        for y in -self.search_range..=self.search_range {
            for x in -self.search_range..=self.search_range {
                let mv = MotionVector::new(x * 4, y * 4); // Convert to qpel
                let cost = self.calculate_cost(src, reference, mv);
                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                }
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations: ((2 * self.search_range + 1) * (2 * self.search_range + 1)) as usize,
        }
    }

    // ── Diamond search ───────────────────────────────────────────────────────

    fn diamond_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let mut best_mv = predictor;
        let mut best_cost = self.calculate_cost(src, reference, best_mv);
        let mut iterations = 0;

        let large_diamond = [
            (0_i16, -2_i16),
            (-1, -1),
            (1, -1),
            (-2, 0),
            (2, 0),
            (-1, 1),
            (1, 1),
            (0, 2),
        ];

        loop {
            let mut improved = false;

            for &(dx, dy) in &large_diamond {
                let mv = MotionVector::new(best_mv.x + dx * 4, best_mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                iterations += 1;

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations,
        }
    }

    // ── Hexagon search ───────────────────────────────────────────────────────

    fn hexagon_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let mut best_mv = predictor;
        let mut best_cost = self.calculate_cost(src, reference, best_mv);
        let mut iterations = 0;

        let hexagon = [(0_i16, -2_i16), (-2, -1), (2, -1), (-2, 1), (2, 1), (0, 2)];

        loop {
            let mut improved = false;

            for &(dx, dy) in &hexagon {
                let mv = MotionVector::new(best_mv.x + dx * 4, best_mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                iterations += 1;

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations,
        }
    }

    // ── TZ search ────────────────────────────────────────────────────────────

    fn tz_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        // Test Zone Search: diamond + small refinement
        let mut result = self.diamond_search(src, reference, width, height, predictor);

        let small_diamond = [(0_i16, -1_i16), (-1, 0), (1, 0), (0, 1)];
        let mut improved = true;

        while improved {
            improved = false;
            for &(dx, dy) in &small_diamond {
                let mv = MotionVector::new(result.mv.x + dx * 4, result.mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                result.iterations += 1;

                if cost < result.cost {
                    result.cost = cost;
                    result.mv = mv;
                    improved = true;
                }
            }
        }

        result
    }

    // ── EPZS search ──────────────────────────────────────────────────────────

    fn epzs_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let best_mv = predictor;
        let best_cost = self.calculate_cost(src, reference, best_mv);
        let iterations = 1;

        if best_cost < 100.0 {
            return MotionSearchResult {
                mv: best_mv,
                cost: best_cost,
                iterations,
            };
        }

        self.diamond_search(src, reference, width, height, predictor)
    }

    // ── UMH search ───────────────────────────────────────────────────────────

    fn umh_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let mut result = self.hexagon_search(src, reference, width, height, predictor);

        for _ in 0..2 {
            let small_hex = [(0_i16, -1_i16), (-1, 0), (1, 0), (0, 1)];
            for &(dx, dy) in &small_hex {
                let mv = MotionVector::new(result.mv.x + dx * 2, result.mv.y + dy * 2);
                let cost = self.calculate_cost(src, reference, mv);
                result.iterations += 1;

                if cost < result.cost {
                    result.cost = cost;
                    result.mv = mv;
                }
            }
        }

        result
    }

    // ── Cost function ─────────────────────────────────────────────────────────

    fn calculate_cost(&self, src: &[u8], _reference: &[u8], _mv: MotionVector) -> f64 {
        // Simplified cost proxy: uses src pixel mean.
        // In a production encoder this would compute SAD/SATD against the
        // motion-compensated reference region.
        src.iter().map(|&x| f64::from(x)).sum::<f64>() / src.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Parallel search with thread-local scratch
// ─────────────────────────────────────────────────────────────────────────────

/// SAD with separate reference dimensions (for padded reference frames).
pub(crate) fn sad_with_mv_ref(
    src: &[u8],
    reference: &[u8],
    src_w: usize,
    src_h: usize,
    ref_w: usize,
    ref_h: usize,
    mv_x: i32,
    mv_y: i32,
) -> u32 {
    let iref_w = ref_w as i32;
    let iref_h = ref_h as i32;
    let mut sum = 0u32;

    for row in 0..src_h {
        let ref_row = row as i32 + mv_y;
        if ref_row < 0 || ref_row >= iref_h {
            return u32::MAX;
        }
        for col in 0..src_w {
            let ref_col = col as i32 + mv_x;
            if ref_col < 0 || ref_col >= iref_w {
                return u32::MAX;
            }
            let src_pix = src[row * src_w + col];
            let ref_pix = reference[ref_row as usize * ref_w + ref_col as usize];
            sum = sum.saturating_add(u32::from(src_pix.abs_diff(ref_pix)));
        }
    }
    sum
}

/// Result of a parallel block-level motion search.
#[derive(Debug, Clone, Copy)]
pub struct ParallelSearchResult {
    /// Best motion vector (integer-pel).
    pub mv_x: i32,
    /// Best motion vector (integer-pel).
    pub mv_y: i32,
    /// SADcost for the best MV.
    pub sad: u32,
}

/// Performs motion search over a list of blocks in parallel using rayon.
///
/// Each rayon thread borrows its own `MOTION_SCRATCH` instead of allocating
/// new `Vec`s per block.  The SAD scratch buffer is sized to `block_size²`;
/// the candidate MV buffer is sized to the search pattern.
///
/// The output is a `Vec<ParallelSearchResult>` parallel to `blocks`.
///
/// # Parameters
/// * `blocks`       — Slice of `(src, reference)` pixel byte slices.
///                    `src` must be exactly `block_size × block_size` bytes.
///                    `reference` must be exactly `ref_size × ref_size` bytes,
///                    where `ref_size = block_size + 2 * search_range as usize`.
///                    The extra `search_range` border on all four sides allows any
///                    candidate MV in `[-search_range, +search_range]²` to be
///                    evaluated without hitting an out-of-bounds boundary.
/// * `block_size`   — Source block side length (pixels).
/// * `search_range` — Integer-pel half-range; MVs in `[-r, +r] × [-r, +r]` (diamond).
///
/// The SAD for candidate MV `(dx, dy)` is computed as:
/// `sum |src[r][c] − reference[(r + search_range + dy)][(c + search_range + dx)]|`
/// where the `search_range` offset centres `src` in the padded reference.
#[must_use]
pub fn parallel_motion_search(
    blocks: &[(&[u8], &[u8])],
    block_size: usize,
    search_range: i32,
) -> Vec<ParallelSearchResult> {
    use rayon::prelude::*;

    let range = search_range as usize;
    let ref_size = block_size + 2 * range;
    let centre_offset = search_range; // row/col offset of src within padded reference

    blocks
        .par_iter()
        .map(|&(src, reference)| {
            MOTION_SCRATCH.with(|scratch| {
                let mut s = scratch.borrow_mut();

                // Resize scratch to block_size² — no allocation if already large enough.
                let pixel_count = block_size * block_size;
                s.sad_buffer.resize(pixel_count, 0);

                // Build diamond candidate MVs into the candidate list.
                // Diamond: all (dx, dy) with |dx|+|dy| <= search_range
                s.candidate_mvs.clear();
                for dy in -search_range..=search_range {
                    for dx in -search_range..=search_range {
                        if dx.unsigned_abs() + dy.unsigned_abs() <= search_range as u32 {
                            s.candidate_mvs.push((dx, dy));
                        }
                    }
                }

                // Evaluate SAD for each candidate, store in cost_buffer.
                // Each candidate MV (dx, dy) is applied relative to the centre offset so
                // that src[r][c] is compared to reference[r + centre + dy][c + centre + dx].
                let n_candidates = s.candidate_mvs.len();
                s.cost_buffer.resize(n_candidates, 0.0);
                let mut best_idx = 0usize;
                let mut best_sad = u32::MAX;

                for ci in 0..n_candidates {
                    let (mv_x, mv_y) = s.candidate_mvs[ci];
                    // Translate MV: reference origin is at (centre_offset, centre_offset)
                    // so the effective absolute MV in the padded reference is
                    // (centre_offset + mv_x, centre_offset + mv_y).
                    let sad = sad_with_mv_ref(
                        src,
                        reference,
                        block_size,
                        block_size,
                        ref_size,
                        ref_size,
                        mv_x + centre_offset,
                        mv_y + centre_offset,
                    );
                    s.cost_buffer[ci] = sad as f32;
                    if sad < best_sad {
                        best_sad = sad;
                        best_idx = ci;
                    }
                }

                let (best_x, best_y) = s.candidate_mvs[best_idx];
                ParallelSearchResult {
                    mv_x: best_x,
                    mv_y: best_y,
                    sad: best_sad,
                }
            })
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Motion search result.
#[derive(Debug, Clone, Copy)]
pub struct MotionSearchResult {
    /// Best motion vector.
    pub mv: MotionVector,
    /// Search cost.
    pub cost: f64,
    /// Number of iterations.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience wrapper: SAD with same-sized src and reference.
    fn sad_with_mv(
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        mv_x: i32,
        mv_y: i32,
    ) -> u32 {
        sad_with_mv_ref(src, reference, width, height, width, height, mv_x, mv_y)
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(16, -8);
        assert_eq!(mv.x, 16);
        assert_eq!(mv.y, -8);

        let zero = MotionVector::zero();
        assert_eq!(zero.x, 0);
        assert_eq!(zero.y, 0);
    }

    #[test]
    fn test_motion_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(optimizer.algorithm, SearchAlgorithm::Hexagon);
    }

    #[test]
    fn test_search_algorithm_selection() {
        let mut config = OptimizerConfig::default();

        config.level = OptimizationLevel::Fast;
        let opt_fast =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(opt_fast.algorithm, SearchAlgorithm::Diamond);

        config.level = OptimizationLevel::Placebo;
        let opt_placebo =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(opt_placebo.algorithm, SearchAlgorithm::Umh);
    }

    // ── Thread-local scratch tests ───────────────────────────────────────────

    /// Creates `(src, padded_reference)` for testing `parallel_motion_search`.
    ///
    /// `padded_reference` is `ref_size × ref_size` where `ref_size = block_size + 2*range`.
    /// `src[row][col]` is taken from the centre of `padded_reference` displaced by
    /// `(dx, dy)`, so that the true MV is exactly `(dx, dy)`.
    ///
    /// With `dx, dy ∈ [-range, range]` the SAD at the true MV is 0 (no noise).
    fn make_shifted_pair(
        block_size: usize,
        range: i32,
        dx: i32,
        dy: i32,
        noise_amp: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        let ref_size = block_size + 2 * range as usize;

        // Padded reference: fill with a pseudo-random LCG pattern that is
        // unlikely to repeat within the padded frame, ensuring a unique SAD minimum.
        let mut padded_ref = vec![0u8; ref_size * ref_size];
        let mut state = 0x1357_2468u32;
        for pixel in padded_ref.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *pixel = (state >> 24) as u8;
        }

        // Source block: sample padded_reference at (range + row + dy, range + col + dx)
        // so that sad_with_mv_ref(src, padded_ref, bs, bs, ref_size, ref_size, dx+range, dy+range) == 0.
        let mut src = vec![0u8; block_size * block_size];
        let mut noise_state = 0x1234_5678u32;
        for row in 0..block_size {
            // Use signed arithmetic to avoid overflow when dy/dx are negative
            let ref_row = (range + row as i32 + dy) as usize;
            for col in 0..block_size {
                let ref_col = (range + col as i32 + dx) as usize;
                let base = padded_ref[ref_row * ref_size + ref_col];
                let noise = if noise_amp > 0 {
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let n = (noise_state >> 24) as u8;
                    (n % (noise_amp * 2)).saturating_sub(noise_amp)
                } else {
                    0
                };
                src[row * block_size + col] = base.saturating_add(noise);
            }
        }
        (src, padded_ref)
    }

    /// Full search oracle: returns the best integer MV for (src, reference).
    #[allow(dead_code)]
    fn full_search_oracle(
        src: &[u8],
        reference: &[u8],
        block_size: usize,
        range: i32,
    ) -> (i32, i32) {
        let mut best_sad = u32::MAX;
        let mut best_mv = (0i32, 0i32);
        for dy in -range..=range {
            for dx in -range..=range {
                let sad = sad_with_mv(src, reference, block_size, block_size, dx, dy);
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = (dx, dy);
                }
            }
        }
        best_mv
    }

    /// Motion search accuracy: exact integer shift must be found (no noise).
    ///
    /// The padded reference is designed so that SAD at the true MV is 0.
    #[test]
    fn test_parallel_full_search_exact_shift() {
        let (dx, dy) = (3i32, -2i32);
        let block_size = 8;
        let range = 8i32;
        let (src, padded_ref) = make_shifted_pair(block_size, range, dx, dy, 0);
        let blocks = vec![(src.as_slice(), padded_ref.as_slice())];
        let results = parallel_motion_search(&blocks, block_size, range);
        assert_eq!(results.len(), 1);
        let r = results[0];
        assert_eq!(r.sad, 0, "SAD at true MV must be 0 without noise");
        assert_eq!(
            (r.mv_x, r.mv_y),
            (dx, dy),
            "parallel full search must find exact shift ({dx}, {dy}), got ({}, {})",
            r.mv_x,
            r.mv_y
        );
    }

    /// Thread-local scratch reuse: multiple blocks in one call must all succeed.
    #[test]
    fn test_parallel_multiple_blocks_thread_local() {
        let block_size = 8;
        let range = 8i32;
        let shifts = [(2i32, 1i32), (-3i32, 2i32), (0i32, -4i32), (4i32, 0i32)];
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = shifts
            .iter()
            .map(|&(dx, dy)| make_shifted_pair(block_size, range, dx, dy, 0))
            .collect();

        let blocks: Vec<(&[u8], &[u8])> = pairs
            .iter()
            .map(|(s, r)| (s.as_slice(), r.as_slice()))
            .collect();

        let results = parallel_motion_search(&blocks, block_size, range);
        assert_eq!(results.len(), shifts.len());

        for (i, (&(exp_dx, exp_dy), r)) in shifts.iter().zip(results.iter()).enumerate() {
            assert_eq!(
                r.sad, 0,
                "block {i}: SAD at true MV must be 0 without noise"
            );
            assert_eq!(
                (r.mv_x, r.mv_y),
                (exp_dx, exp_dy),
                "block {i}: expected ({exp_dx}, {exp_dy}), got ({}, {})",
                r.mv_x,
                r.mv_y
            );
        }
    }

    /// With small additive noise the found MV must be within ±2 of the true shift.
    #[test]
    fn test_parallel_search_with_noise() {
        let block_size = 16;
        let range = 8i32;
        let (dx, dy) = (3i32, -2i32);
        let (src, padded_ref) = make_shifted_pair(block_size, range, dx, dy, 3);
        let blocks = vec![(src.as_slice(), padded_ref.as_slice())];
        let results = parallel_motion_search(&blocks, block_size, range);
        let r = results[0];
        assert!(
            (r.mv_x - dx).abs() <= 2 && (r.mv_y - dy).abs() <= 2,
            "noisy block: expected MV near ({dx}, {dy}), got ({}, {})",
            r.mv_x,
            r.mv_y
        );
    }

    /// SAD helper returns 0 for identical blocks with zero offset.
    #[test]
    fn test_sad_identical_blocks() {
        let block = vec![128u8; 64];
        let sad = sad_with_mv(&block, &block, 8, 8, 0, 0);
        assert_eq!(sad, 0);
    }

    /// SAD helper returns u32::MAX when MV is out of bounds.
    #[test]
    fn test_sad_out_of_bounds() {
        let block = vec![100u8; 64];
        let sad = sad_with_mv(&block, &block, 8, 8, 100, 100);
        assert_eq!(sad, u32::MAX);
    }
}
