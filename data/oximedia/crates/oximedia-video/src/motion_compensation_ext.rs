//! Motion estimation enhancements — L9, L10, L11, L29, L30, L31, L33.
//!
//! This module is re-exported from [`super::motion_compensation`] and provides
//! the following APIs:
//!
//! * **L9** [`refine_subpel`] — sub-pixel refinement (half-pel bilinear,
//!   quarter-pel Catmull-Rom bicubic).
//! * **L10** [`adaptive_block_size_search`] — SAD-per-pixel block-size selector.
//! * **L11** [`bidir_motion_estimate`] — B-frame–style bidirectional estimation.
//! * **L29** [`estimate_frame_motion_parallel`] — rayon-parallel frame scan.
//! * **L30** [`motion_search_with_pattern`] — diamond / hexagonal / full / three-step.
//! * **L31** [`sad_simd`] / [`sad_scalar`] — SIMD-accelerated SAD (AVX2 → SSE4.1 → scalar).
//! * **L33** [`hierarchical_motion_estimate`] — coarse-to-fine pyramid ME.

use crate::motion_compensation::{interpolate_bicubic, interpolate_bilinear};

// -----------------------------------------------------------------------
// L9 — Sub-pixel motion estimation (half-pel and quarter-pel refinement)
// -----------------------------------------------------------------------

/// Sub-pixel refinement precision levels for motion estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpelMode {
    /// Integer-pel only (no refinement).
    IntegerPel,
    /// Half-pixel refinement using bilinear interpolation.
    HalfPel,
    /// Quarter-pixel refinement (half-pel first, then quarter-pel bicubic).
    QuarterPel,
}

/// Refine an integer motion vector to sub-pixel accuracy.
///
/// Returns `(refined_mv_x_qpel, refined_mv_y_qpel, sad)` where displacements
/// are in quarter-pixel units (integer pels are multiplied by 4).
///
/// * Half-pel uses bilinear interpolation over a 3×3 half-pel grid.
/// * Quarter-pel further refines with bicubic interpolation over a 3×3
///   quarter-pel grid centred on the best half-pel position.
pub fn refine_subpel(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    mv_x: i32,
    mv_y: i32,
    mode: SubpelMode,
) -> (i32, i32, i32) {
    if mode == SubpelMode::IntegerPel {
        let sad = compute_sad_rect(
            current_block,
            reference,
            block_w,
            block_h,
            ref_width,
            ref_height,
            mv_x,
            mv_y,
        );
        return (mv_x * 4, mv_y * 4, sad as i32);
    }

    let rw = ref_width as u32;
    let rh = ref_height as u32;

    // Half-pel search: 3×3 grid in half-pel units centred on int MV
    // Seed with the centre position so ties prefer zero displacement.
    let base_hx = mv_x * 2;
    let base_hy = mv_y * 2;
    let mut best_hx = base_hx;
    let mut best_hy = base_hy;
    let mut best_sad = eval_subpel_sad_rect(
        reference,
        current_block,
        rw,
        rh,
        block_w,
        block_h,
        base_hx,
        base_hy,
        2,
        false,
    );

    for dhy in -1i32..=1 {
        for dhx in -1i32..=1 {
            if dhx == 0 && dhy == 0 {
                continue; // already seeded
            }
            let hx = base_hx + dhx;
            let hy = base_hy + dhy;
            let sad = eval_subpel_sad_rect(
                reference,
                current_block,
                rw,
                rh,
                block_w,
                block_h,
                hx,
                hy,
                2,
                false,
            );
            if sad < best_sad {
                best_sad = sad;
                best_hx = hx;
                best_hy = hy;
            }
        }
    }

    if mode == SubpelMode::HalfPel {
        return (best_hx * 2, best_hy * 2, best_sad as i32);
    }

    // Quarter-pel: 3×3 grid in quarter-pel units centred on best half-pel (bicubic)
    // Seed with the centre (= best half-pel position × 2) so ties prefer it.
    let base_qx = best_hx * 2;
    let base_qy = best_hy * 2;
    let mut best_qx = base_qx;
    let mut best_qy = base_qy;
    let mut best_qsad = eval_subpel_sad_rect(
        reference,
        current_block,
        rw,
        rh,
        block_w,
        block_h,
        base_qx,
        base_qy,
        4,
        true,
    );

    for dqy in -1i32..=1 {
        for dqx in -1i32..=1 {
            if dqx == 0 && dqy == 0 {
                continue; // already seeded
            }
            let qx = base_qx + dqx;
            let qy = base_qy + dqy;
            let sad = eval_subpel_sad_rect(
                reference,
                current_block,
                rw,
                rh,
                block_w,
                block_h,
                qx,
                qy,
                4,
                true,
            );
            if sad < best_qsad {
                best_qsad = sad;
                best_qx = qx;
                best_qy = qy;
            }
        }
    }

    (best_qx, best_qy, best_qsad as i32)
}

/// Compute integer-pel SAD for a block at a given MV offset.
fn compute_sad_rect(
    block: &[u8],
    reference: &[u8],
    block_w: usize,
    block_h: usize,
    ref_w: usize,
    ref_h: usize,
    mv_x: i32,
    mv_y: i32,
) -> u32 {
    let mut sad = 0u32;
    for row in 0..block_h {
        for col in 0..block_w {
            let cur = block.get(row * block_w + col).copied().unwrap_or(0);
            let rx = (mv_x + col as i32).clamp(0, ref_w as i32 - 1) as usize;
            let ry = (mv_y + row as i32).clamp(0, ref_h as i32 - 1) as usize;
            let r = reference.get(ry * ref_w + rx).copied().unwrap_or(0);
            sad += (cur as i32 - r as i32).unsigned_abs();
        }
    }
    sad
}

/// Fractional-pel SAD using bilinear or bicubic interpolation.
fn eval_subpel_sad_rect(
    reference: &[u8],
    block: &[u8],
    ref_w: u32,
    ref_h: u32,
    block_w: usize,
    block_h: usize,
    spx: i32,
    spy: i32,
    divisor: i32,
    bicubic: bool,
) -> f64 {
    let div_f = divisor as f64;
    let mut sad = 0.0f64;
    for row in 0..block_h {
        for col in 0..block_w {
            let cur = block.get(row * block_w + col).copied().unwrap_or(0) as f64;
            let rx_f = spx as f64 / div_f + col as f64;
            let ry_f = spy as f64 / div_f + row as f64;
            let ix = rx_f.floor() as i32;
            let iy = ry_f.floor() as i32;
            let fx = rx_f - ix as f64;
            let fy = ry_f - iy as f64;
            let ref_val = if bicubic {
                interpolate_bicubic(reference, ref_w, ref_h, ix, iy, fx, fy)
            } else {
                interpolate_bilinear(reference, ref_w, ref_h, ix, iy, fx, fy)
            };
            sad += (cur - ref_val).abs();
        }
    }
    sad
}

// -----------------------------------------------------------------------
// L10 — Adaptive block size selection
// -----------------------------------------------------------------------

/// Canonical block sizes for adaptive block-size motion estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockSize {
    /// 4×4 pixel block.
    B4x4,
    /// 8×8 pixel block.
    B8x8,
    /// 16×16 pixel block.
    B16x16,
    /// 32×32 pixel block.
    B32x32,
    /// 64×64 pixel block.
    B64x64,
}

impl BlockSize {
    /// Width and height in pixels.
    #[must_use]
    pub fn dims(self) -> (usize, usize) {
        match self {
            Self::B4x4 => (4, 4),
            Self::B8x8 => (8, 8),
            Self::B16x16 => (16, 16),
            Self::B32x32 => (32, 32),
            Self::B64x64 => (64, 64),
        }
    }

    /// Area in pixels.
    #[must_use]
    pub fn area(self) -> usize {
        let (w, h) = self.dims();
        w * h
    }
}

/// Choose the best block size for a macroblock by comparing SAD per pixel.
///
/// Returns `(block_size, motion_vector, sad)`.
pub fn adaptive_block_size_search(
    reference: &[u8],
    current: &[u8],
    frame_width: usize,
    frame_height: usize,
    mb_x: usize,
    mb_y: usize,
    search_range: i32,
    sizes: &[BlockSize],
) -> (BlockSize, (i32, i32), u32) {
    let mut best_size = BlockSize::B16x16;
    let mut best_mv = (0i32, 0i32);
    let mut best_sad = u32::MAX;
    let mut best_sad_per_pixel = f64::MAX;

    for &size in sizes {
        let (bw, bh) = size.dims();
        if mb_x + bw > frame_width || mb_y + bh > frame_height {
            continue;
        }

        // Extract current block
        let mut cur_block = vec![0u8; bw * bh];
        for row in 0..bh {
            for col in 0..bw {
                let idx = (mb_y + row) * frame_width + (mb_x + col);
                cur_block[row * bw + col] = current.get(idx).copied().unwrap_or(0);
            }
        }

        let mut mv = (0i32, 0i32);
        let mut mv_sad = u32::MAX;

        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let rx = mb_x as i32 + dx;
                let ry = mb_y as i32 + dy;
                if rx < 0
                    || ry < 0
                    || rx as usize + bw > frame_width
                    || ry as usize + bh > frame_height
                {
                    continue;
                }
                let mut sad = 0u32;
                for row in 0..bh {
                    for col in 0..bw {
                        let ci = row * bw + col;
                        let ri = (ry as usize + row) * frame_width + (rx as usize + col);
                        let c = cur_block.get(ci).copied().unwrap_or(0);
                        let r = reference.get(ri).copied().unwrap_or(0);
                        sad += (c as i32 - r as i32).unsigned_abs();
                    }
                }
                if sad < mv_sad {
                    mv_sad = sad;
                    mv = (dx, dy);
                }
            }
        }

        let sad_per_pixel = mv_sad as f64 / size.area() as f64;
        if sad_per_pixel < best_sad_per_pixel {
            best_sad_per_pixel = sad_per_pixel;
            best_size = size;
            best_mv = mv;
            best_sad = mv_sad;
        }
    }

    (best_size, best_mv, best_sad)
}

// -----------------------------------------------------------------------
// L11 — Bidirectional motion estimation (B-frame style)
// -----------------------------------------------------------------------

/// Bidirectional motion estimation result.
#[derive(Debug, Clone)]
pub struct BidirMotionVector {
    /// Forward MV (current → reference_fwd) in pixel units.
    pub mv_fwd: (i32, i32),
    /// Backward MV (current → reference_bwd) in pixel units.
    pub mv_bwd: (i32, i32),
    /// Weighted average SAD (0.5 × fwd_sad + 0.5 × bwd_sad).
    pub sad: u32,
}

/// Estimate bidirectional motion for a block.
///
/// Independently searches the forward (`ref_fwd`) and backward (`ref_bwd`)
/// references with full-range scans, then combines with a 50/50 weighted
/// average predictor to produce a [`BidirMotionVector`].
pub fn bidir_motion_estimate(
    ref_fwd: &[u8],
    ref_bwd: &[u8],
    current: &[u8],
    frame_width: usize,
    frame_height: usize,
    mb_x: usize,
    mb_y: usize,
    block_size: usize,
    search_range: i32,
) -> BidirMotionVector {
    let mut cur_block = vec![0u8; block_size * block_size];
    for row in 0..block_size {
        for col in 0..block_size {
            let idx = (mb_y + row) * frame_width + (mb_x + col);
            cur_block[row * block_size + col] = current.get(idx).copied().unwrap_or(0);
        }
    }

    let search_one = |reference: &[u8]| -> ((i32, i32), u32) {
        let mut best_mv = (0i32, 0i32);
        let mut best_sad = u32::MAX;
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let rx = mb_x as i32 + dx;
                let ry = mb_y as i32 + dy;
                if rx < 0
                    || ry < 0
                    || rx as usize + block_size > frame_width
                    || ry as usize + block_size > frame_height
                {
                    continue;
                }
                let mut sad = 0u32;
                for row in 0..block_size {
                    for col in 0..block_size {
                        let ci = row * block_size + col;
                        let ri = (ry as usize + row) * frame_width + (rx as usize + col);
                        let c = cur_block.get(ci).copied().unwrap_or(0);
                        let r = reference.get(ri).copied().unwrap_or(0);
                        sad += (c as i32 - r as i32).unsigned_abs();
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = (dx, dy);
                }
            }
        }
        (best_mv, best_sad)
    };

    let (mv_fwd, fwd_sad) = search_one(ref_fwd);
    let (mv_bwd, bwd_sad) = search_one(ref_bwd);
    let blended_sad = (fwd_sad as f64 * 0.5 + bwd_sad as f64 * 0.5).round() as u32;

    BidirMotionVector {
        mv_fwd,
        mv_bwd,
        sad: blended_sad,
    }
}

// -----------------------------------------------------------------------
// L29 — Rayon parallelism for motion search
// -----------------------------------------------------------------------

/// Estimate motion vectors for every macroblock in a frame in parallel.
///
/// Uses [`rayon`] to distribute block-level full-search motion estimation
/// across all available CPU cores.  Macroblocks are independent.
///
/// Returns motion vectors in raster order (one `(mv_x, mv_y)` per macroblock).
pub fn estimate_frame_motion_parallel(
    reference: &[u8],
    current: &[u8],
    frame_width: usize,
    frame_height: usize,
    block_size: usize,
    search_range: i32,
) -> Vec<(i32, i32)> {
    use rayon::prelude::*;

    let mut positions: Vec<(usize, usize)> = Vec::new();
    let mut by = 0usize;
    while by + block_size <= frame_height {
        let mut bx = 0usize;
        while bx + block_size <= frame_width {
            positions.push((bx, by));
            bx += block_size;
        }
        by += block_size;
    }

    positions
        .par_iter()
        .map(|&(mb_x, mb_y)| {
            let mut cur_block = vec![0u8; block_size * block_size];
            for row in 0..block_size {
                for col in 0..block_size {
                    let idx = (mb_y + row) * frame_width + (mb_x + col);
                    cur_block[row * block_size + col] = current.get(idx).copied().unwrap_or(0);
                }
            }
            let mut best_mv = (0i32, 0i32);
            let mut best_sad = u32::MAX;
            for dy in -search_range..=search_range {
                for dx in -search_range..=search_range {
                    let rx = mb_x as i32 + dx;
                    let ry = mb_y as i32 + dy;
                    if rx < 0
                        || ry < 0
                        || rx as usize + block_size > frame_width
                        || ry as usize + block_size > frame_height
                    {
                        continue;
                    }
                    let mut sad = 0u32;
                    for row in 0..block_size {
                        for col in 0..block_size {
                            let ci = row * block_size + col;
                            let ri = (ry as usize + row) * frame_width + (rx as usize + col);
                            let c = cur_block.get(ci).copied().unwrap_or(0);
                            let r = reference.get(ri).copied().unwrap_or(0);
                            sad += (c as i32 - r as i32).unsigned_abs();
                        }
                    }
                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = (dx, dy);
                    }
                }
            }
            best_mv
        })
        .collect()
}

// -----------------------------------------------------------------------
// L30 — Diamond and hexagonal search patterns
// -----------------------------------------------------------------------

/// Motion search pattern selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPattern {
    /// Exhaustive search over the full ±search_range window.
    Full,
    /// Large diamond → small diamond refinement (LDSP/SDSP style).
    Diamond,
    /// Hexagonal 6-point search + small-diamond refinement (JM/x264 style).
    Hexagonal,
    /// Classic three-step search: step sizes 4 → 2 → 1.
    ThreeStep,
}

/// Search for a motion vector using the given pattern.
///
/// Returns `(mv_x, mv_y, sad)` where the displacement is relative to
/// `(start_x, start_y)` in the reference frame.
pub fn motion_search_with_pattern(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    search_range: i32,
    pattern: SearchPattern,
) -> (i32, i32, u32) {
    match pattern {
        SearchPattern::Full => pattern_full(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            search_range,
        ),
        SearchPattern::Diamond => pattern_diamond(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            search_range,
        ),
        SearchPattern::Hexagonal => pattern_hexagonal(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            search_range,
        ),
        SearchPattern::ThreeStep => pattern_three_step(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            search_range,
        ),
    }
}

/// SAD for a candidate displacement relative to `(start_x, start_y)`.
fn candidate_sad(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    dx: i32,
    dy: i32,
) -> u32 {
    let rx = start_x as i32 + dx;
    let ry = start_y as i32 + dy;
    if rx < 0 || ry < 0 || rx as usize + block_w > ref_width || ry as usize + block_h > ref_height {
        return u32::MAX;
    }
    let mut sad = 0u32;
    for row in 0..block_h {
        for col in 0..block_w {
            let ci = row * block_w + col;
            let ri = (ry as usize + row) * ref_width + (rx as usize + col);
            let c = current_block.get(ci).copied().unwrap_or(0);
            let r = reference.get(ri).copied().unwrap_or(0);
            sad += (c as i32 - r as i32).unsigned_abs();
        }
    }
    sad
}

fn pattern_full(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    search_range: i32,
) -> (i32, i32, u32) {
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = candidate_sad(
        reference,
        current_block,
        block_w,
        block_h,
        ref_width,
        ref_height,
        start_x,
        start_y,
        0,
        0,
    );
    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let sad = candidate_sad(
                reference,
                current_block,
                block_w,
                block_h,
                ref_width,
                ref_height,
                start_x,
                start_y,
                dx,
                dy,
            );
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    (best_dx, best_dy, best_sad)
}

fn pattern_diamond(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    search_range: i32,
) -> (i32, i32, u32) {
    const LDSP: [(i32, i32); 8] = [
        (0, -2),
        (1, -1),
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 1),
        (-2, 0),
        (-1, -1),
    ];
    const SDSP: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    let mut cx = 0i32;
    let mut cy = 0i32;
    let mut best_sad = candidate_sad(
        reference,
        current_block,
        block_w,
        block_h,
        ref_width,
        ref_height,
        start_x,
        start_y,
        0,
        0,
    );
    let max_iters = (search_range * 2).max(1) as usize;

    for _ in 0..max_iters {
        let mut improved = false;
        for &(ddx, ddy) in &LDSP {
            let ndx = cx + ddx;
            let ndy = cy + ddy;
            if ndx.abs() > search_range || ndy.abs() > search_range {
                continue;
            }
            let sad = candidate_sad(
                reference,
                current_block,
                block_w,
                block_h,
                ref_width,
                ref_height,
                start_x,
                start_y,
                ndx,
                ndy,
            );
            if sad < best_sad {
                best_sad = sad;
                cx = ndx;
                cy = ndy;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    for &(ddx, ddy) in &SDSP {
        let ndx = cx + ddx;
        let ndy = cy + ddy;
        if ndx.abs() > search_range || ndy.abs() > search_range {
            continue;
        }
        let sad = candidate_sad(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            ndx,
            ndy,
        );
        if sad < best_sad {
            best_sad = sad;
            cx = ndx;
            cy = ndy;
        }
    }

    (cx, cy, best_sad)
}

fn pattern_hexagonal(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    search_range: i32,
) -> (i32, i32, u32) {
    const HEX: [(i32, i32); 6] = [(-2, 0), (-1, 2), (1, 2), (2, 0), (1, -2), (-1, -2)];
    const SDSP: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    let mut cx = 0i32;
    let mut cy = 0i32;
    let mut best_sad = candidate_sad(
        reference,
        current_block,
        block_w,
        block_h,
        ref_width,
        ref_height,
        start_x,
        start_y,
        0,
        0,
    );
    let max_iters = (search_range * 2).max(1) as usize;

    for _ in 0..max_iters {
        let mut improved = false;
        for &(ddx, ddy) in &HEX {
            let ndx = cx + ddx;
            let ndy = cy + ddy;
            if ndx.abs() > search_range || ndy.abs() > search_range {
                continue;
            }
            let sad = candidate_sad(
                reference,
                current_block,
                block_w,
                block_h,
                ref_width,
                ref_height,
                start_x,
                start_y,
                ndx,
                ndy,
            );
            if sad < best_sad {
                best_sad = sad;
                cx = ndx;
                cy = ndy;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    for &(ddx, ddy) in &SDSP {
        let ndx = cx + ddx;
        let ndy = cy + ddy;
        if ndx.abs() > search_range || ndy.abs() > search_range {
            continue;
        }
        let sad = candidate_sad(
            reference,
            current_block,
            block_w,
            block_h,
            ref_width,
            ref_height,
            start_x,
            start_y,
            ndx,
            ndy,
        );
        if sad < best_sad {
            best_sad = sad;
            cx = ndx;
            cy = ndy;
        }
    }

    (cx, cy, best_sad)
}

fn pattern_three_step(
    reference: &[u8],
    current_block: &[u8],
    block_w: usize,
    block_h: usize,
    ref_width: usize,
    ref_height: usize,
    start_x: usize,
    start_y: usize,
    search_range: i32,
) -> (i32, i32, u32) {
    let mut cx = 0i32;
    let mut cy = 0i32;
    let mut best_sad = candidate_sad(
        reference,
        current_block,
        block_w,
        block_h,
        ref_width,
        ref_height,
        start_x,
        start_y,
        0,
        0,
    );

    let mut step = 4i32.min(search_range);
    while step >= 1 {
        let mut moved = false;
        for dy in [-step, 0, step] {
            for dx in [-step, 0, step] {
                let ndx = cx + dx;
                let ndy = cy + dy;
                if ndx.abs() > search_range || ndy.abs() > search_range {
                    continue;
                }
                let sad = candidate_sad(
                    reference,
                    current_block,
                    block_w,
                    block_h,
                    ref_width,
                    ref_height,
                    start_x,
                    start_y,
                    ndx,
                    ndy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    cx = ndx;
                    cy = ndy;
                    moved = true;
                }
            }
        }
        if !moved || step == 1 {
            step /= 2;
        }
    }

    (cx, cy, best_sad)
}

// -----------------------------------------------------------------------
// L31 — SIMD-optimised SAD computation
// -----------------------------------------------------------------------

/// Compute SAD between two byte slices using SIMD when available.
///
/// Dispatch order on `x86_64`: AVX2 → SSE4.1 → scalar.
/// On all other architectures or when no SIMD is detected, falls back to
/// [`sad_scalar`].
#[allow(unsafe_code)]
pub fn sad_simd(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sad_avx2(&a[..len], &b[..len]) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { sad_sse41(&a[..len], &b[..len]) };
        }
    }
    sad_scalar(&a[..len], &b[..len])
}

/// Portable scalar SAD — `Σ |a[i] - b[i]|`.
#[inline]
pub fn sad_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y) as u32)
        .sum()
}

/// AVX2 path: 32 bytes per iteration via `_mm256_sad_epu8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let chunks = a.len() / 32;
    let mut sum = _mm256_setzero_si256();
    for i in 0..chunks {
        let va = _mm256_loadu_si256(a.as_ptr().add(i * 32).cast::<__m256i>());
        let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32).cast::<__m256i>());
        let s = _mm256_sad_epu8(va, vb);
        sum = _mm256_add_epi64(sum, s);
    }
    let lo = _mm256_extracti128_si256(sum, 0);
    let hi = _mm256_extracti128_si256(sum, 1);
    let combined = _mm_add_epi64(lo, hi);
    let result = _mm_extract_epi64(combined, 0) + _mm_extract_epi64(combined, 1);
    result as u32 + sad_scalar(&a[chunks * 32..], &b[chunks * 32..])
}

/// SSE4.1 path: 16 bytes per iteration via `_mm_sad_epu8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_sse41(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let chunks = a.len() / 16;
    let mut sum = _mm_setzero_si128();
    for i in 0..chunks {
        let va = _mm_loadu_si128(a.as_ptr().add(i * 16).cast::<__m128i>());
        let vb = _mm_loadu_si128(b.as_ptr().add(i * 16).cast::<__m128i>());
        let s = _mm_sad_epu8(va, vb);
        sum = _mm_add_epi64(sum, s);
    }
    let lo = _mm_cvtsi128_si64(sum) as u64;
    let hi = _mm_cvtsi128_si64(_mm_srli_si128(sum, 8)) as u64;
    (lo + hi) as u32 + sad_scalar(&a[chunks * 16..], &b[chunks * 16..])
}

// -----------------------------------------------------------------------
// L33 — Hierarchical (coarse-to-fine) motion estimation
// -----------------------------------------------------------------------

/// Hierarchical (coarse-to-fine) motion estimation using a Gaussian pyramid.
///
/// Builds a `levels`-deep box-average pyramid for both frames, estimates
/// motion at the coarsest level, then propagates the prediction (×2 each
/// level) with a small local refinement search.
///
/// Returns `(mv_x, mv_y, sad)` at full resolution.
pub fn hierarchical_motion_estimate(
    reference: &[u8],
    current: &[u8],
    frame_width: usize,
    frame_height: usize,
    mb_x: usize,
    mb_y: usize,
    block_size: usize,
    levels: usize,
    search_range: i32,
) -> (i32, i32, u32) {
    let levels = levels.max(1);
    let pyr_ref = build_luma_pyramid(reference, frame_width, frame_height, levels);
    let pyr_cur = build_luma_pyramid(current, frame_width, frame_height, levels);

    let coarsest = levels - 1;
    let scale = 1usize << coarsest;
    let c_w = (frame_width / scale).max(1);
    let c_h = (frame_height / scale).max(1);
    let c_bs = (block_size / scale).max(1);
    let c_bx = (mb_x / scale).min(c_w.saturating_sub(c_bs));
    let c_by = (mb_y / scale).min(c_h.saturating_sub(c_bs));

    let coarse_block: Vec<u8> = {
        let mut b = vec![0u8; c_bs * c_bs];
        for row in 0..c_bs {
            for col in 0..c_bs {
                let idx = (c_by + row) * c_w + (c_bx + col);
                b[row * c_bs + col] = pyr_cur[coarsest].get(idx).copied().unwrap_or(0);
            }
        }
        b
    };

    let (mut pred_dx, mut pred_dy, _) = pattern_full(
        &pyr_ref[coarsest],
        &coarse_block,
        c_bs,
        c_bs,
        c_w,
        c_h,
        c_bx,
        c_by,
        search_range,
    );

    for level in (0..coarsest).rev() {
        pred_dx *= 2;
        pred_dy *= 2;

        let l_scale = 1usize << level;
        let l_w = (frame_width / l_scale).max(1);
        let l_h = (frame_height / l_scale).max(1);
        let l_bs = (block_size / l_scale).max(1);
        let l_bx = (mb_x / l_scale).min(l_w.saturating_sub(l_bs));
        let l_by = (mb_y / l_scale).min(l_h.saturating_sub(l_bs));

        let level_block: Vec<u8> = {
            let mut b = vec![0u8; l_bs * l_bs];
            for row in 0..l_bs {
                for col in 0..l_bs {
                    let idx = (l_by + row) * l_w + (l_bx + col);
                    b[row * l_bs + col] = pyr_cur[level].get(idx).copied().unwrap_or(0);
                }
            }
            b
        };

        let local_range = 2i32;
        let mut best_sad = u32::MAX;
        let mut best_dx = pred_dx;
        let mut best_dy = pred_dy;

        for ddy in -local_range..=local_range {
            for ddx in -local_range..=local_range {
                let ndx = pred_dx + ddx;
                let ndy = pred_dy + ddy;
                let sad = candidate_sad(
                    &pyr_ref[level],
                    &level_block,
                    l_bs,
                    l_bs,
                    l_w,
                    l_h,
                    l_bx,
                    l_by,
                    ndx,
                    ndy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_dx = ndx;
                    best_dy = ndy;
                }
            }
        }
        pred_dx = best_dx;
        pred_dy = best_dy;
    }

    let bx = mb_x.min(frame_width.saturating_sub(block_size));
    let by = mb_y.min(frame_height.saturating_sub(block_size));
    let full_block: Vec<u8> = {
        let mut b = vec![0u8; block_size * block_size];
        for row in 0..block_size {
            for col in 0..block_size {
                let idx = (by + row) * frame_width + (bx + col);
                b[row * block_size + col] = current.get(idx).copied().unwrap_or(0);
            }
        }
        b
    };
    let final_sad = candidate_sad(
        reference,
        &full_block,
        block_size,
        block_size,
        frame_width,
        frame_height,
        bx,
        by,
        pred_dx,
        pred_dy,
    );

    (pred_dx, pred_dy, final_sad)
}

/// Box-average Gaussian pyramid (level 0 = full resolution).
fn build_luma_pyramid(frame: &[u8], width: usize, height: usize, levels: usize) -> Vec<Vec<u8>> {
    let mut pyramid: Vec<Vec<u8>> = Vec::with_capacity(levels);
    pyramid.push(frame.to_vec());

    for _ in 1..levels {
        let prev_w = width >> (pyramid.len() - 1);
        let prev_h = height >> (pyramid.len() - 1);
        let out_w = (prev_w / 2).max(1);
        let out_h = (prev_h / 2).max(1);
        let prev = pyramid
            .last()
            .expect("pyramid always has at least one level");

        let mut down = vec![0u8; out_w * out_h];
        for y in 0..out_h {
            for x in 0..out_w {
                let sy = y * 2;
                let sx = x * 2;
                let p00 = prev.get(sy * prev_w + sx).copied().unwrap_or(0) as u32;
                let p01 = prev
                    .get(sy * prev_w + (sx + 1).min(prev_w - 1))
                    .copied()
                    .unwrap_or(0) as u32;
                let p10 = prev
                    .get((sy + 1).min(prev_h - 1) * prev_w + sx)
                    .copied()
                    .unwrap_or(0) as u32;
                let p11 = prev
                    .get((sy + 1).min(prev_h - 1) * prev_w + (sx + 1).min(prev_w - 1))
                    .copied()
                    .unwrap_or(0) as u32;
                down[y * out_w + x] = ((p00 + p01 + p10 + p11 + 2) / 4) as u8;
            }
        }
        pyramid.push(down);
    }

    pyramid
}

// -----------------------------------------------------------------------
// Tests for L9, L10, L11, L29, L30, L31, L33
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h]
    }

    fn make_ramp(w: usize, h: usize) -> Vec<u8> {
        (0..(w * h)).map(|i| (i % 256) as u8).collect()
    }

    // ── L9 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_refine_subpel_integer_pel() {
        // Use uniform frame so any block matches everywhere
        let frame = make_frame(32, 32, 100);
        let block = make_frame(8, 8, 100);
        let (qx, qy, _sad) =
            refine_subpel(&frame, 32, 32, &block, 8, 8, 0, 0, SubpelMode::IntegerPel);
        assert_eq!(qx, 0);
        assert_eq!(qy, 0);
    }

    #[test]
    fn test_refine_subpel_halfpel_identical_frame() {
        // Uniform frame: zero displacement always has zero SAD
        let frame = make_frame(32, 32, 128);
        let block = make_frame(8, 8, 128);
        let (qx, qy, sad) = refine_subpel(&frame, 32, 32, &block, 8, 8, 0, 0, SubpelMode::HalfPel);
        assert_eq!(qx, 0);
        assert_eq!(qy, 0);
        assert!(sad >= 0);
    }

    #[test]
    fn test_refine_subpel_quarterpel_identical_frame() {
        let frame = make_frame(32, 32, 64);
        let block = make_frame(8, 8, 64);
        let (qx, qy, _sad) =
            refine_subpel(&frame, 32, 32, &block, 8, 8, 0, 0, SubpelMode::QuarterPel);
        assert_eq!(qx, 0);
        assert_eq!(qy, 0);
    }

    #[test]
    fn test_refine_subpel_integer_pel_units() {
        let frame = make_frame(32, 32, 128);
        let block = make_frame(8, 8, 128);
        let (qx, qy, _sad) =
            refine_subpel(&frame, 32, 32, &block, 8, 8, 2, 3, SubpelMode::IntegerPel);
        // Integer pels are returned as qpel × 4
        assert_eq!(qx, 8);
        assert_eq!(qy, 12);
    }

    #[test]
    fn test_refine_subpel_halfpel_returns_even_qpel() {
        let frame = make_frame(32, 32, 200);
        let block = make_frame(8, 8, 200);
        let (qx, qy, _sad) = refine_subpel(&frame, 32, 32, &block, 8, 8, 0, 0, SubpelMode::HalfPel);
        // Half-pel units ×2 = even qpel values
        assert_eq!(qx % 2, 0);
        assert_eq!(qy % 2, 0);
    }

    // ── L10 ────────────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_block_size_identical_frame() {
        let frame = make_ramp(64, 64);
        let sizes = [BlockSize::B8x8, BlockSize::B16x16, BlockSize::B32x32];
        let (size, mv, sad) = adaptive_block_size_search(&frame, &frame, 64, 64, 0, 0, 4, &sizes);
        assert_eq!(mv, (0, 0));
        assert_eq!(sad, 0);
        // Any size is valid when SAD=0
        let _ = size;
    }

    #[test]
    fn test_adaptive_block_size_skips_oversized() {
        let frame = make_ramp(16, 16);
        // Only B4x4 and B8x8 fit; B64x64 should be skipped
        let sizes = [BlockSize::B4x4, BlockSize::B8x8, BlockSize::B64x64];
        let (size, _mv, _sad) = adaptive_block_size_search(&frame, &frame, 16, 16, 0, 0, 2, &sizes);
        assert!(size == BlockSize::B4x4 || size == BlockSize::B8x8);
    }

    #[test]
    fn test_block_size_dims() {
        assert_eq!(BlockSize::B4x4.dims(), (4, 4));
        assert_eq!(BlockSize::B8x8.dims(), (8, 8));
        assert_eq!(BlockSize::B16x16.dims(), (16, 16));
        assert_eq!(BlockSize::B32x32.dims(), (32, 32));
        assert_eq!(BlockSize::B64x64.dims(), (64, 64));
    }

    #[test]
    fn test_block_size_area() {
        assert_eq!(BlockSize::B4x4.area(), 16);
        assert_eq!(BlockSize::B8x8.area(), 64);
        assert_eq!(BlockSize::B16x16.area(), 256);
    }

    // ── L11 ────────────────────────────────────────────────────────────

    #[test]
    fn test_bidir_identical_references() {
        let frame = make_ramp(32, 32);
        let result = bidir_motion_estimate(&frame, &frame, &frame, 32, 32, 0, 0, 8, 4);
        assert_eq!(result.sad, 0);
        assert_eq!(result.mv_fwd, (0, 0));
        assert_eq!(result.mv_bwd, (0, 0));
    }

    #[test]
    fn test_bidir_fields_accessible() {
        let frame = make_frame(16, 16, 100);
        let result = bidir_motion_estimate(&frame, &frame, &frame, 16, 16, 0, 0, 8, 2);
        let _ = result.mv_fwd;
        let _ = result.mv_bwd;
        let _ = result.sad;
    }

    #[test]
    fn test_bidir_sad_is_average() {
        // Both references identical → fwd_sad = bwd_sad = 0, blended = 0
        let frame = make_ramp(32, 32);
        let result = bidir_motion_estimate(&frame, &frame, &frame, 32, 32, 8, 8, 8, 2);
        assert_eq!(result.sad, 0);
    }

    // ── L29 ────────────────────────────────────────────────────────────

    #[test]
    fn test_parallel_me_identical_frames() {
        let frame = make_ramp(32, 32);
        let mvs = estimate_frame_motion_parallel(&frame, &frame, 32, 32, 8, 4);
        // 32/8 * 32/8 = 16 macroblocks
        assert_eq!(mvs.len(), 16);
        for mv in &mvs {
            assert_eq!(*mv, (0, 0));
        }
    }

    #[test]
    fn test_parallel_me_vector_count() {
        let frame = make_frame(64, 48, 128);
        let mvs = estimate_frame_motion_parallel(&frame, &frame, 64, 48, 16, 8);
        // 64/16 * 48/16 = 4 * 3 = 12
        assert_eq!(mvs.len(), 12);
    }

    // ── L30 ────────────────────────────────────────────────────────────

    #[test]
    fn test_pattern_full_identical() {
        // Use uniform frame so any candidate has SAD=0 → zero displacement wins
        let frame = make_frame(32, 32, 150);
        let block = make_frame(8, 8, 150);
        let (dx, dy, sad) =
            motion_search_with_pattern(&frame, &block, 8, 8, 32, 32, 0, 0, 4, SearchPattern::Full);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_pattern_diamond_identical() {
        let frame = make_frame(32, 32, 80);
        let block = make_frame(8, 8, 80);
        let (dx, dy, sad) = motion_search_with_pattern(
            &frame,
            &block,
            8,
            8,
            32,
            32,
            0,
            0,
            8,
            SearchPattern::Diamond,
        );
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_pattern_hexagonal_identical() {
        let frame = make_frame(32, 32, 60);
        let block = make_frame(8, 8, 60);
        let (dx, dy, sad) = motion_search_with_pattern(
            &frame,
            &block,
            8,
            8,
            32,
            32,
            0,
            0,
            8,
            SearchPattern::Hexagonal,
        );
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_pattern_three_step_identical() {
        let frame = make_frame(32, 32, 40);
        let block = make_frame(8, 8, 40);
        let (dx, dy, sad) = motion_search_with_pattern(
            &frame,
            &block,
            8,
            8,
            32,
            32,
            0,
            0,
            8,
            SearchPattern::ThreeStep,
        );
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_pattern_start_position() {
        // Block at start_x=8 from a uniform frame → SAD=0 with dx=0, dy=0
        let frame = make_frame(32, 32, 170);
        let block = make_frame(8, 8, 170);
        let (dx, dy, sad) =
            motion_search_with_pattern(&frame, &block, 8, 8, 32, 32, 8, 0, 4, SearchPattern::Full);
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_diamond_finds_zero_motion_on_uniform() {
        let frame = make_frame(32, 32, 200);
        let block = make_frame(8, 8, 200);
        let (dx, dy, sad) = motion_search_with_pattern(
            &frame,
            &block,
            8,
            8,
            32,
            32,
            0,
            0,
            8,
            SearchPattern::Diamond,
        );
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    // ── L31 ────────────────────────────────────────────────────────────

    #[test]
    fn test_sad_simd_identical() {
        let block = vec![128u8; 256];
        assert_eq!(sad_simd(&block, &block), 0);
    }

    #[test]
    fn test_sad_simd_known_diff() {
        let a = vec![0u8; 256];
        let b = vec![1u8; 256];
        assert_eq!(sad_simd(&a, &b), 256);
    }

    #[test]
    fn test_sad_simd_max_diff() {
        let a = vec![0u8; 256];
        let b = vec![255u8; 256];
        assert_eq!(sad_simd(&a, &b), 255 * 256);
    }

    #[test]
    fn test_sad_simd_matches_scalar() {
        let a: Vec<u8> = (0u8..=255).collect();
        let b: Vec<u8> = (0u8..=255).rev().collect();
        assert_eq!(sad_simd(&a, &b), sad_scalar(&a, &b));
    }

    #[test]
    fn test_sad_scalar_identical() {
        let data = vec![50u8; 64];
        assert_eq!(sad_scalar(&data, &data), 0);
    }

    #[test]
    fn test_sad_simd_unequal_lengths() {
        let a = vec![10u8; 10];
        let b = vec![20u8; 8];
        // Only first 8 bytes compared
        assert_eq!(sad_simd(&a, &b), 8 * 10);
    }

    // ── L33 ────────────────────────────────────────────────────────────

    #[test]
    fn test_hierarchical_me_identical_frames() {
        let frame = make_ramp(64, 64);
        let (dx, dy, sad) = hierarchical_motion_estimate(&frame, &frame, 64, 64, 0, 0, 8, 3, 8);
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_hierarchical_me_single_level() {
        // levels=1 degenerates to flat full search
        let frame = make_ramp(32, 32);
        let (dx, dy, sad) = hierarchical_motion_estimate(&frame, &frame, 32, 32, 0, 0, 8, 1, 4);
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }

    #[test]
    fn test_hierarchical_me_shifted_frame() {
        let w = 32usize;
        let h = 32usize;
        let mut reference = make_frame(w, h, 100);
        let mut current = make_frame(w, h, 100);
        // Place a 8×8 block of 200 in current at (0,0) and in reference at (2,0)
        for row in 0..8 {
            for col in 0..8 {
                current[row * w + col] = 200;
                reference[row * w + col + 2] = 200;
            }
        }
        let (dx, dy, sad) = hierarchical_motion_estimate(&reference, &current, w, h, 0, 0, 8, 2, 4);
        assert_eq!(sad, 0, "expected zero SAD after ME: dx={dx} dy={dy}");
    }

    #[test]
    fn test_hierarchical_me_levels_zero_treated_as_one() {
        let frame = make_ramp(32, 32);
        // levels=0 must not panic (clamped to 1)
        let (dx, dy, sad) = hierarchical_motion_estimate(&frame, &frame, 32, 32, 0, 0, 8, 0, 4);
        assert_eq!(sad, 0);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
    }
}
