//! Block-based motion estimation and compensation.
//!
//! Provides multiple motion estimation algorithms (full search, three-step,
//! diamond, hexagon) operating on luma planes, plus helpers to build
//! predicted frames and compute residuals.  Sub-pixel refinement (half-pel
//! and quarter-pel) is available via [`SubPixelMode`].

/// Sub-pixel refinement precision for motion estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubPixelMode {
    /// Integer-pel only (no sub-pixel refinement).
    None,
    /// Half-pixel refinement after integer search.
    HalfPel,
    /// Quarter-pixel refinement (half-pel then quarter-pel).
    QuarterPel,
}

/// A motion vector describing how a block in the current frame maps to the reference frame.
///
/// When sub-pixel refinement is used, `dx` and `dy` are stored in **quarter-pel units**.
/// For integer-pel vectors (no refinement), multiply by 4 is implicit — callers should
/// check `sub_pixel` to interpret the values correctly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal displacement in pixels (positive = right).
    pub dx: i16,
    /// Vertical displacement in pixels (positive = down).
    pub dy: i16,
    /// Sum of absolute differences at best match (lower = better).
    pub sad: u32,
    /// Block top-left X coordinate in the current frame.
    pub block_x: u32,
    /// Block top-left Y coordinate in the current frame.
    pub block_y: u32,
}

/// A sub-pixel motion vector with fractional displacement in quarter-pel units.
#[derive(Debug, Clone, PartialEq)]
pub struct SubPixelMotionVector {
    /// Horizontal displacement in quarter-pixel units (positive = right).
    /// e.g. `dx_qpel = 6` means 1.5 pixels to the right.
    pub dx_qpel: i32,
    /// Vertical displacement in quarter-pixel units (positive = down).
    pub dy_qpel: i32,
    /// Interpolated SAD at the sub-pixel position (lower = better).
    pub sad: f64,
    /// Block top-left X coordinate in the current frame.
    pub block_x: u32,
    /// Block top-left Y coordinate in the current frame.
    pub block_y: u32,
    /// The sub-pixel precision level that produced this vector.
    pub precision: SubPixelMode,
}

/// Motion estimation algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeAlgorithm {
    /// Exhaustive search over the full ±search_range window.
    FullSearch,
    /// Classic three-step search: step sizes 4 → 2 → 1.
    ThreeStep,
    /// EPZS diamond-pattern search (large then small diamond).
    DiamondSearch,
    /// Hexagon-pattern search with small-diamond refinement.
    HexagonSearch,
}

/// Configuration for a block-based motion estimator.
pub struct MotionEstimator {
    /// Side length of each square block in pixels (typically 16).
    pub block_size: u32,
    /// Search radius in pixels (±range).
    pub search_range: i32,
    /// Algorithm used to search for the best match.
    pub algorithm: MeAlgorithm,
}

impl MotionEstimator {
    /// Create a new `MotionEstimator`.
    pub fn new(block_size: u32, search_range: i32, algorithm: MeAlgorithm) -> Self {
        Self {
            block_size,
            search_range,
            algorithm,
        }
    }

    /// Estimate the motion vector for a single block.
    ///
    /// Both `ref_frame` and `cur_frame` are row-major luma planes of `width` columns.
    pub fn estimate_block(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        match self.algorithm {
            MeAlgorithm::FullSearch => {
                self.full_search(ref_frame, cur_frame, width, block_x, block_y)
            }
            MeAlgorithm::ThreeStep => {
                self.three_step(ref_frame, cur_frame, width, block_x, block_y)
            }
            MeAlgorithm::DiamondSearch => {
                self.diamond_search(ref_frame, cur_frame, width, block_x, block_y)
            }
            MeAlgorithm::HexagonSearch => {
                self.hexagon_search(ref_frame, cur_frame, width, block_x, block_y)
            }
        }
    }

    /// Estimate motion vectors for every block in a frame.
    pub fn estimate_frame(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<MotionVector> {
        let mut vectors = Vec::new();
        let mut by = 0u32;
        while by + self.block_size <= height {
            let mut bx = 0u32;
            while bx + self.block_size <= width {
                vectors.push(self.estimate_block(ref_frame, cur_frame, width, bx, by));
                bx += self.block_size;
            }
            by += self.block_size;
        }
        vectors
    }

    /// Estimate a sub-pixel motion vector for a single block by first performing
    /// integer-pel search and then refining to half-pel and/or quarter-pel.
    ///
    /// The returned [`SubPixelMotionVector`] stores displacements in quarter-pel units.
    pub fn estimate_block_subpel(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
        mode: SubPixelMode,
    ) -> SubPixelMotionVector {
        // Step 1: integer-pel search
        let int_mv = self.estimate_block(ref_frame, cur_frame, width, block_x, block_y);

        if mode == SubPixelMode::None {
            return SubPixelMotionVector {
                dx_qpel: int_mv.dx as i32 * 4,
                dy_qpel: int_mv.dy as i32 * 4,
                sad: int_mv.sad as f64,
                block_x,
                block_y,
                precision: SubPixelMode::None,
            };
        }

        // Step 2: half-pel refinement around integer best match
        let int_dx = int_mv.dx as i32;
        let int_dy = int_mv.dy as i32;
        let mut best_hx = int_dx * 2; // half-pel units
        let mut best_hy = int_dy * 2;
        let mut best_sad = int_mv.sad as f64;

        // Evaluate 8 half-pel neighbors + centre
        for &dhy in &[-1i32, 0, 1] {
            for &dhx in &[-1i32, 0, 1] {
                let hx = int_dx * 2 + dhx;
                let hy = int_dy * 2 + dhy;
                let sad = eval_subpel_sad(
                    ref_frame,
                    cur_frame,
                    width,
                    height,
                    self.block_size,
                    block_x,
                    block_y,
                    hx,
                    hy,
                    2,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_hx = hx;
                    best_hy = hy;
                }
            }
        }

        if mode == SubPixelMode::HalfPel {
            return SubPixelMotionVector {
                dx_qpel: best_hx * 2,
                dy_qpel: best_hy * 2,
                sad: best_sad,
                block_x,
                block_y,
                precision: SubPixelMode::HalfPel,
            };
        }

        // Step 3: quarter-pel refinement around half-pel best match
        let mut best_qx = best_hx * 2; // quarter-pel units
        let mut best_qy = best_hy * 2;
        let mut best_qsad = best_sad;

        for &dqy in &[-1i32, 0, 1] {
            for &dqx in &[-1i32, 0, 1] {
                let qx = best_hx * 2 + dqx;
                let qy = best_hy * 2 + dqy;
                let sad = eval_subpel_sad(
                    ref_frame,
                    cur_frame,
                    width,
                    height,
                    self.block_size,
                    block_x,
                    block_y,
                    qx,
                    qy,
                    4,
                );
                if sad < best_qsad {
                    best_qsad = sad;
                    best_qx = qx;
                    best_qy = qy;
                }
            }
        }

        SubPixelMotionVector {
            dx_qpel: best_qx,
            dy_qpel: best_qy,
            sad: best_qsad,
            block_x,
            block_y,
            precision: SubPixelMode::QuarterPel,
        }
    }

    /// Estimate sub-pixel motion vectors for every block in a frame.
    pub fn estimate_frame_subpel(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
        mode: SubPixelMode,
    ) -> Vec<SubPixelMotionVector> {
        let mut vectors = Vec::new();
        let mut by = 0u32;
        while by + self.block_size <= height {
            let mut bx = 0u32;
            while bx + self.block_size <= width {
                vectors.push(
                    self.estimate_block_subpel(ref_frame, cur_frame, width, height, bx, by, mode),
                );
                bx += self.block_size;
            }
            by += self.block_size;
        }
        vectors
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Evaluate SAD for the current block at candidate displacement (cdx, cdy).
    pub fn eval_candidate(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
        cdx: i32,
        cdy: i32,
    ) -> u32 {
        let size = self.block_size;
        let ref_x = block_x as i32 + cdx;
        let ref_y = block_y as i32 + cdy;

        // Clamp so the reference block stays inside the frame.
        let ref_x = ref_x.max(0).min(width as i32 - size as i32);
        let ref_y = ref_y.max(0).min(height as i32 - size as i32);

        let cur_offset = (block_y * width + block_x) as usize;
        let ref_offset = (ref_y as u32 * width + ref_x as u32) as usize;

        compute_sad(
            &cur_frame[cur_offset..],
            &ref_frame[ref_offset..],
            width as usize,
            width as usize,
            size,
        )
    }

    /// Infer `height` from frame length and `width`.  Returns 0 if ambiguous.
    fn frame_height(frame: &[u8], width: u32) -> u32 {
        if width == 0 {
            return 0;
        }
        (frame.len() / width as usize) as u32
    }

    fn full_search(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        let height = Self::frame_height(ref_frame, width);
        let range = self.search_range;
        // Seed with the zero-motion candidate so ties prefer (0,0).
        let mut best_sad =
            self.eval_candidate(ref_frame, cur_frame, width, height, block_x, block_y, 0, 0);
        let mut best_dx = 0i16;
        let mut best_dy = 0i16;

        for dy in -range..=range {
            for dx in -range..=range {
                let sad = self.eval_candidate(
                    ref_frame, cur_frame, width, height, block_x, block_y, dx, dy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    best_dx = dx as i16;
                    best_dy = dy as i16;
                }
            }
        }

        MotionVector {
            dx: best_dx,
            dy: best_dy,
            sad: best_sad,
            block_x,
            block_y,
        }
    }

    fn three_step(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        let height = Self::frame_height(ref_frame, width);
        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = u32::MAX;

        // Steps: 4, 2, 1
        for &step in &[4i32, 2, 1] {
            let mut local_best_dx = cx;
            let mut local_best_dy = cy;

            for dy in [-step, 0, step].iter() {
                for dx in [-step, 0, step].iter() {
                    let cdx = cx + dx;
                    let cdy = cy + dy;
                    let sad = self.eval_candidate(
                        ref_frame, cur_frame, width, height, block_x, block_y, cdx, cdy,
                    );
                    if sad < best_sad {
                        best_sad = sad;
                        local_best_dx = cdx;
                        local_best_dy = cdy;
                    }
                }
            }
            cx = local_best_dx;
            cy = local_best_dy;
        }

        MotionVector {
            dx: cx as i16,
            dy: cy as i16,
            sad: best_sad,
            block_x,
            block_y,
        }
    }

    fn diamond_search(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        let height = Self::frame_height(ref_frame, width);

        // Large diamond offsets (distance 2)
        const LARGE_DIAMOND: [(i32, i32); 8] = [
            (0, -2),
            (1, -1),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
        ];
        // Small diamond offsets (distance 1)
        const SMALL_DIAMOND: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = self.eval_candidate(
            ref_frame, cur_frame, width, height, block_x, block_y, cx, cy,
        );

        // Large diamond phase — iterate until no improvement
        loop {
            let mut improved = false;
            for &(ddx, ddy) in &LARGE_DIAMOND {
                let cdx = cx + ddx;
                let cdy = cy + ddy;
                let sad = self.eval_candidate(
                    ref_frame, cur_frame, width, height, block_x, block_y, cdx, cdy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    cx = cdx;
                    cy = cdy;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }

        // Small diamond refinement
        for &(ddx, ddy) in &SMALL_DIAMOND {
            let cdx = cx + ddx;
            let cdy = cy + ddy;
            let sad = self.eval_candidate(
                ref_frame, cur_frame, width, height, block_x, block_y, cdx, cdy,
            );
            if sad < best_sad {
                best_sad = sad;
                cx = cdx;
                cy = cdy;
            }
        }

        MotionVector {
            dx: cx as i16,
            dy: cy as i16,
            sad: best_sad,
            block_x,
            block_y,
        }
    }

    fn hexagon_search(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        let height = Self::frame_height(ref_frame, width);

        // 6 hexagonal neighbors
        const HEX: [(i32, i32); 6] = [(-2, 0), (-1, 2), (1, 2), (2, 0), (1, -2), (-1, -2)];
        const SMALL_DIAMOND: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = self.eval_candidate(
            ref_frame, cur_frame, width, height, block_x, block_y, cx, cy,
        );

        // Hexagon phase
        loop {
            let mut improved = false;
            for &(ddx, ddy) in &HEX {
                let cdx = cx + ddx;
                let cdy = cy + ddy;
                let sad = self.eval_candidate(
                    ref_frame, cur_frame, width, height, block_x, block_y, cdx, cdy,
                );
                if sad < best_sad {
                    best_sad = sad;
                    cx = cdx;
                    cy = cdy;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }

        // Small diamond refinement
        for &(ddx, ddy) in &SMALL_DIAMOND {
            let cdx = cx + ddx;
            let cdy = cy + ddy;
            let sad = self.eval_candidate(
                ref_frame, cur_frame, width, height, block_x, block_y, cdx, cdy,
            );
            if sad < best_sad {
                best_sad = sad;
                cx = cdx;
                cy = cdy;
            }
        }

        MotionVector {
            dx: cx as i16,
            dy: cy as i16,
            sad: best_sad,
            block_x,
            block_y,
        }
    }
}

/// Compute the sum of absolute differences between two blocks.
///
/// `block_a` and `block_b` are row-major planes; `stride_a` / `stride_b` are
/// the full row widths (in bytes) of the respective planes.  Only `size × size`
/// pixels starting at the pointer are compared.
pub fn compute_sad(
    block_a: &[u8],
    block_b: &[u8],
    stride_a: usize,
    stride_b: usize,
    size: u32,
) -> u32 {
    let size = size as usize;
    let mut sad = 0u32;
    for row in 0..size {
        let row_a = row * stride_a;
        let row_b = row * stride_b;
        for col in 0..size {
            let a = block_a.get(row_a + col).copied().unwrap_or(0);
            let b = block_b.get(row_b + col).copied().unwrap_or(0);
            sad += (a as i32 - b as i32).unsigned_abs();
        }
    }
    sad
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-accelerated SAD (Sum of Absolute Differences)
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar fallback SAD for a contiguous 16×16 byte block.
///
/// `block1` and `block2` must each be exactly **256 bytes** (16 rows × 16
/// columns stored contiguously with stride = 16).
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline]
fn sad_16x16_scalar(block1: &[u8], block2: &[u8]) -> u32 {
    debug_assert!(block1.len() >= 256, "block1 must be at least 256 bytes");
    debug_assert!(block2.len() >= 256, "block2 must be at least 256 bytes");
    let mut acc = 0u32;
    // Unroll 16 rows of 16 bytes for predictable codegen.
    for i in 0..256 {
        let a = block1.get(i).copied().unwrap_or(0) as i16;
        let b = block2.get(i).copied().unwrap_or(0) as i16;
        acc += (a - b).unsigned_abs() as u32;
    }
    acc
}

/// SIMD-accelerated Sum of Absolute Differences for contiguous 16×16 blocks
/// (256 bytes each, stride = 16).
///
/// Uses SSE2 `_mm_sad_epu8` when the SSE2 target feature is available at
/// *compile time*.  Each `_mm_sad_epu8` call computes the SAD of 8 pairs in
/// one instruction; two 128-bit loads per row let us process 16 bytes/row.
/// The results are accumulated in a 64-bit horizontal sum across the register.
///
/// # Safety
///
/// The caller must guarantee that both slices contain **at least 256 bytes**
/// and that `target_feature = "sse2"` is active (which is guaranteed by the
/// `#[target_feature(enable = "sse2")]` annotation).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn sad_16x16_sse2_impl(block1: &[u8], block2: &[u8]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Accumulate in a pair of 64-bit SSE lanes.
    let mut acc = _mm_setzero_si128();

    for row in 0..16usize {
        let base = row * 16;
        // SAFETY: slices are at least 256 bytes; base + 16 <= 256.
        let p1 = block1.as_ptr().add(base);
        let p2 = block2.as_ptr().add(base);
        let v1 = _mm_loadu_si128(p1.cast::<__m128i>());
        let v2 = _mm_loadu_si128(p2.cast::<__m128i>());
        // _mm_sad_epu8: computes |a-b| for 16 byte pairs, horizontally sums
        // them into two 16-bit values in the low 16 bits of each 64-bit lane.
        let sad_row = _mm_sad_epu8(v1, v2);
        acc = _mm_add_epi64(acc, sad_row);
    }

    // Extract the two 64-bit partial sums and add them together.
    let lo = _mm_cvtsi128_si64(acc) as u64;
    let hi = _mm_cvtsi128_si64(_mm_srli_si128(acc, 8)) as u64;
    (lo + hi) as u32
}

/// SIMD-accelerated Sum of Absolute Differences for contiguous **16×16** blocks.
///
/// `block1` and `block2` must each contain **at least 256 bytes**. Only the
/// first 256 bytes (16 rows × 16 columns, stride = 16) are examined.
///
/// On `x86`/`x86_64` targets compiled with SSE2 support (enabled at compile
/// time via `#[target_feature]`) this uses `_mm_sad_epu8` intrinsics for a
/// ~16× throughput improvement over scalar code.  On all other targets (or
/// when the feature is absent at runtime) the function falls back to the
/// portable scalar path.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub fn sad_16x16_simd(block1: &[u8], block2: &[u8]) -> u32 {
    #[cfg(target_feature = "sse2")]
    {
        if block1.len() >= 256 && block2.len() >= 256 {
            // SAFETY: we just verified slice bounds; SSE2 is guaranteed by
            // the compile-time `target_feature` check above.
            return unsafe { sad_16x16_sse2_impl(block1, block2) };
        }
    }
    sad_16x16_scalar(block1, block2)
}

/// Runtime-dispatched SAD that picks the best available SIMD path or falls
/// back to the portable scalar implementation.
///
/// Unlike the 16×16-specific `sad_16x16_simd`, this function accepts blocks
/// of arbitrary `width` × `height` stored **contiguously** (stride = width).
/// The caller must provide slices of at least `width * height` bytes.
///
/// Dispatch order on `x86`/`x86_64`:
/// 1. SSE2 path for 16×16 blocks when SSE2 is detected at runtime via
///    [`is_x86_feature_detected!`].
/// 2. Portable scalar loop for all other sizes or architectures.
#[allow(unsafe_code)]
pub fn sad_adaptive(block1: &[u8], block2: &[u8], width: u32, height: u32) -> u32 {
    let w = width as usize;
    let h = height as usize;
    let required = w.saturating_mul(h);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if w == 16 && h == 16 && block1.len() >= 256 && block2.len() >= 256 {
            if is_x86_feature_detected!("sse2") {
                // SAFETY: we verified slice lengths and confirmed SSE2 at runtime.
                return unsafe { sad_16x16_sse2_impl(block1, block2) };
            }
        }
    }

    // Portable scalar path — handles any width × height.
    let mut acc = 0u32;
    for row in 0..h {
        let row_off = row * w;
        for col in 0..w {
            let idx = row_off + col;
            if idx >= required {
                break;
            }
            let a = block1.get(idx).copied().unwrap_or(0) as i16;
            let b = block2.get(idx).copied().unwrap_or(0) as i16;
            acc += (a - b).unsigned_abs() as u32;
        }
    }
    acc
}

/// Reconstruct a predicted frame by applying motion vectors to a reference frame.
///
/// Both the reference frame and output are luma planes of `width × height` bytes.
pub fn compensate_frame(
    ref_frame: &[u8],
    vectors: &[MotionVector],
    width: u32,
    height: u32,
    block_size: u32,
) -> Vec<u8> {
    let total = (width * height) as usize;
    let mut out = vec![0u8; total];

    for mv in vectors {
        let ref_x = (mv.block_x as i32 + mv.dx as i32)
            .max(0)
            .min(width as i32 - block_size as i32) as u32;
        let ref_y = (mv.block_y as i32 + mv.dy as i32)
            .max(0)
            .min(height as i32 - block_size as i32) as u32;

        for row in 0..block_size {
            let src_row = ref_y + row;
            let dst_row = mv.block_y + row;
            if src_row >= height || dst_row >= height {
                continue;
            }
            for col in 0..block_size {
                let src_col = ref_x + col;
                let dst_col = mv.block_x + col;
                if src_col >= width || dst_col >= width {
                    continue;
                }
                let src_idx = (src_row * width + src_col) as usize;
                let dst_idx = (dst_row * width + dst_col) as usize;
                if let (Some(&pix), Some(dst)) = (ref_frame.get(src_idx), out.get_mut(dst_idx)) {
                    *dst = pix;
                }
            }
        }
    }

    out
}

/// Compute the signed per-pixel difference between `cur_frame` and `predicted`.
///
/// Returns a `Vec<i16>` of the same length as the inputs.
pub fn residual_frame(cur_frame: &[u8], predicted: &[u8]) -> Vec<i16> {
    cur_frame
        .iter()
        .zip(predicted.iter())
        .map(|(&c, &p)| c as i16 - p as i16)
        .collect()
}

/// Add a residual back to a predicted frame, clamping to [0, 255].
pub fn reconstruct_from_residual(predicted: &[u8], residual: &[i16]) -> Vec<u8> {
    predicted
        .iter()
        .zip(residual.iter())
        .map(|(&p, &r)| (p as i16 + r).clamp(0, 255) as u8)
        .collect()
}

// -----------------------------------------------------------------------
// Sub-pixel interpolation helpers
// -----------------------------------------------------------------------

/// Bilinear interpolation at a fractional position in a luma plane.
///
/// `x_frac` and `y_frac` are in [0.0, 1.0) representing the fractional
/// offset from the integer pixel at `(ix, iy)`.
pub fn interpolate_bilinear(
    plane: &[u8],
    width: u32,
    height: u32,
    ix: i32,
    iy: i32,
    x_frac: f64,
    y_frac: f64,
) -> f64 {
    let w = width as i32;
    let h = height as i32;

    let x0 = ix.clamp(0, w - 1);
    let y0 = iy.clamp(0, h - 1);
    let x1 = (ix + 1).clamp(0, w - 1);
    let y1 = (iy + 1).clamp(0, h - 1);

    let p00 = pixel_at(plane, width, x0, y0) as f64;
    let p10 = pixel_at(plane, width, x1, y0) as f64;
    let p01 = pixel_at(plane, width, x0, y1) as f64;
    let p11 = pixel_at(plane, width, x1, y1) as f64;

    let top = p00 * (1.0 - x_frac) + p10 * x_frac;
    let bot = p01 * (1.0 - x_frac) + p11 * x_frac;
    top * (1.0 - y_frac) + bot * y_frac
}

/// Bicubic (Catmull-Rom) interpolation at a fractional position in a luma plane.
///
/// `x_frac` and `y_frac` are in [0.0, 1.0) representing the fractional
/// offset from the integer pixel at `(ix, iy)`.  Uses a 4 × 4 neighbourhood
/// with the Catmull-Rom kernel:
///
/// ```text
/// w(t) = 1.5|t|³ − 2.5|t|² + 1          for |t| ≤ 1
/// w(t) = −0.5|t|³ + 2.5|t|² − 4|t| + 2  for 1 < |t| ≤ 2
/// w(t) = 0                                otherwise
/// ```
pub fn interpolate_bicubic(
    plane: &[u8],
    width: u32,
    height: u32,
    ix: i32,
    iy: i32,
    x_frac: f64,
    y_frac: f64,
) -> f64 {
    let w = width as i32;
    let h = height as i32;
    let mut result = 0.0f64;
    for ky in -1i32..=2 {
        let wy = bicubic_weight(y_frac - ky as f64);
        if wy.abs() < 1e-10 {
            continue;
        }
        for kx in -1i32..=2 {
            let wx = bicubic_weight(x_frac - kx as f64);
            if wx.abs() < 1e-10 {
                continue;
            }
            let px = (ix + kx).clamp(0, w - 1) as usize;
            let py = (iy + ky).clamp(0, h - 1) as usize;
            let pix = plane.get(py * width as usize + px).copied().unwrap_or(0) as f64;
            result += wx * wy * pix;
        }
    }
    result.clamp(0.0, 255.0)
}

/// Catmull-Rom basis weight for bicubic interpolation.
fn bicubic_weight(t: f64) -> f64 {
    let t = t.abs();
    if t <= 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t <= 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// 6-tap filter interpolation at half-pel positions (H.264-style).
///
/// Operates on a single row or column of the plane. The filter taps are:
/// `[1, -5, 20, 20, -5, 1] / 32`
pub fn interpolate_6tap(samples: &[f64; 6]) -> f64 {
    let val = (samples[0] - 5.0 * samples[1] + 20.0 * samples[2] + 20.0 * samples[3]
        - 5.0 * samples[4]
        + samples[5])
        / 32.0;
    val.clamp(0.0, 255.0)
}

/// Sample a pixel from a luma plane with bounds clamping.
fn pixel_at(plane: &[u8], width: u32, x: i32, y: i32) -> u8 {
    let w = width as i32;
    let h = (plane.len() / width as usize) as i32;
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    plane.get(cy * width as usize + cx).copied().unwrap_or(0)
}

/// Evaluate SAD at a sub-pixel position.
///
/// `spx`, `spy` are displacements in sub-pixel units (the denominator is
/// determined by `divisor`: 2 for half-pel, 4 for quarter-pel).
fn eval_subpel_sad(
    ref_frame: &[u8],
    cur_frame: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
    block_x: u32,
    block_y: u32,
    spx: i32,
    spy: i32,
    divisor: i32,
) -> f64 {
    let size = block_size as i32;
    let div_f = divisor as f64;
    let mut sad = 0.0f64;

    for row in 0..size {
        for col in 0..size {
            let cur_idx = (block_y as i32 + row) * width as i32 + (block_x as i32 + col);
            let cur_val = cur_frame.get(cur_idx as usize).copied().unwrap_or(0) as f64;

            // Reference position in fractional pixels
            let ref_x_f = (block_x as i32 + col) as f64 + spx as f64 / div_f;
            let ref_y_f = (block_y as i32 + row) as f64 + spy as f64 / div_f;

            let ix = ref_x_f.floor() as i32;
            let iy = ref_y_f.floor() as i32;
            let fx = ref_x_f - ix as f64;
            let fy = ref_y_f - iy as f64;

            let ref_val = interpolate_bilinear(ref_frame, width, height, ix, iy, fx, fy);
            sad += (cur_val - ref_val).abs();
        }
    }

    sad
}

/// Reconstruct a predicted frame using sub-pixel motion vectors.
///
/// Each vector's displacement is in quarter-pel units. The reference frame is
/// bilinearly interpolated at the fractional positions.
pub fn compensate_frame_subpel(
    ref_frame: &[u8],
    vectors: &[SubPixelMotionVector],
    width: u32,
    height: u32,
    block_size: u32,
) -> Vec<u8> {
    let total = (width * height) as usize;
    let mut out = vec![0u8; total];

    for mv in vectors {
        for row in 0..block_size {
            for col in 0..block_size {
                let dst_x = mv.block_x + col;
                let dst_y = mv.block_y + row;
                if dst_x >= width || dst_y >= height {
                    continue;
                }

                let ref_x_f = dst_x as f64 + mv.dx_qpel as f64 / 4.0;
                let ref_y_f = dst_y as f64 + mv.dy_qpel as f64 / 4.0;

                let ix = ref_x_f.floor() as i32;
                let iy = ref_y_f.floor() as i32;
                let fx = ref_x_f - ix as f64;
                let fy = ref_y_f - iy as f64;

                let val = interpolate_bilinear(ref_frame, width, height, ix, iy, fx, fy);
                let dst_idx = (dst_y * width + dst_x) as usize;
                if let Some(dst) = out.get_mut(dst_idx) {
                    *dst = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    out
}

// -----------------------------------------------------------------------
// Bidirectional motion estimation (B-frame style interpolation)
// -----------------------------------------------------------------------

/// A pair of motion vectors describing bidirectional correspondence for a
/// single block: one forward vector (current→reference0) and one backward
/// vector (current→reference1).
#[derive(Debug, Clone, PartialEq)]
pub struct BiMotionVector {
    /// Forward vector: displacement from current block position to `ref0`.
    pub forward: MotionVector,
    /// Backward vector: displacement from current block position to `ref1`.
    pub backward: MotionVector,
    /// Blended SAD combining forward and backward costs.
    pub blended_sad: u32,
}

/// Configuration for bidirectional motion estimation.
pub struct BiDirectionalEstimator {
    /// Underlying estimator (algorithm, block size, search range).
    pub estimator: MotionEstimator,
    /// Blend weight for the forward prediction in [0, 1] (0.5 = equal blend).
    pub forward_weight: f32,
}

impl BiDirectionalEstimator {
    /// Create a new `BiDirectionalEstimator`.
    pub fn new(estimator: MotionEstimator, forward_weight: f32) -> Self {
        Self {
            estimator,
            forward_weight: forward_weight.clamp(0.0, 1.0),
        }
    }

    /// Estimate bidirectional motion for a single block at `(block_x, block_y)`.
    ///
    /// `ref0` and `ref1` are the two anchor frames (e.g. the frames immediately
    /// before and after the B-frame).  `cur_frame` is the frame to estimate
    /// motion for.  All are luma planes of `width` columns.
    ///
    /// The blended SAD is `forward_weight * fwd_sad + (1 - forward_weight) * bwd_sad`.
    pub fn estimate_block(
        &self,
        ref0: &[u8],
        ref1: &[u8],
        cur_frame: &[u8],
        width: u32,
        block_x: u32,
        block_y: u32,
    ) -> BiMotionVector {
        let forward = self
            .estimator
            .estimate_block(ref0, cur_frame, width, block_x, block_y);
        let backward = self
            .estimator
            .estimate_block(ref1, cur_frame, width, block_x, block_y);

        let fw = self.forward_weight;
        let bw = 1.0 - fw;
        let blended_sad = (forward.sad as f32 * fw + backward.sad as f32 * bw).round() as u32;

        BiMotionVector {
            forward,
            backward,
            blended_sad,
        }
    }

    /// Estimate bidirectional motion for every block in the frame.
    ///
    /// Returns one `BiMotionVector` per block in raster order.
    pub fn estimate_frame(
        &self,
        ref0: &[u8],
        ref1: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<BiMotionVector> {
        let bs = self.estimator.block_size;
        let mut vectors = Vec::new();
        let mut by = 0u32;
        while by + bs <= height {
            let mut bx = 0u32;
            while bx + bs <= width {
                vectors.push(self.estimate_block(ref0, ref1, cur_frame, width, bx, by));
                bx += bs;
            }
            by += bs;
        }
        vectors
    }

    /// Reconstruct an interpolated frame by blending the forward and backward
    /// compensated predictions with `forward_weight`.
    pub fn reconstruct_bidirectional(
        ref0: &[u8],
        ref1: &[u8],
        vectors: &[BiMotionVector],
        width: u32,
        height: u32,
        block_size: u32,
        forward_weight: f32,
    ) -> Vec<u8> {
        // Build forward and backward compensated frames independently.
        let fwd_mvs: Vec<MotionVector> = vectors.iter().map(|bv| bv.forward.clone()).collect();
        let bwd_mvs: Vec<MotionVector> = vectors.iter().map(|bv| bv.backward.clone()).collect();

        let fwd_pred = compensate_frame(ref0, &fwd_mvs, width, height, block_size);
        let bwd_pred = compensate_frame(ref1, &bwd_mvs, width, height, block_size);

        let fw = forward_weight.clamp(0.0, 1.0);
        let bw = 1.0 - fw;
        fwd_pred
            .iter()
            .zip(bwd_pred.iter())
            .map(|(&f, &b)| (f as f32 * fw + b as f32 * bw).round().clamp(0.0, 255.0) as u8)
            .collect()
    }
}

// -----------------------------------------------------------------------
// Adaptive block size estimation
// -----------------------------------------------------------------------

/// A motion vector with the block size that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveMotionVector {
    /// Underlying motion vector.
    pub mv: MotionVector,
    /// The block size chosen by the adaptive selector.
    pub chosen_block_size: u32,
}

/// Block size selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSizePolicy {
    /// Always choose the smallest block (finest detail).
    Smallest,
    /// Always choose the largest block (fastest).
    Largest,
    /// Choose the block size that minimises SAD per pixel (rate–distortion proxy).
    BestSadPerPixel,
}

/// Adaptive motion estimator that tries multiple block sizes and selects the
/// one that best models the local motion (configurable policy).
///
/// Supported block sizes: 4, 8, 16, 32, 64.  Sizes exceeding the frame
/// dimensions are silently skipped.
pub struct AdaptiveBlockEstimator {
    /// Ordered list of block sizes to try (smallest first).
    pub block_sizes: Vec<u32>,
    /// Search range (in pixels, ±range).
    pub search_range: i32,
    /// Algorithm used for each candidate size.
    pub algorithm: MeAlgorithm,
    /// Block size selection policy.
    pub policy: BlockSizePolicy,
}

impl AdaptiveBlockEstimator {
    /// Create an estimator with the standard set of block sizes: `[4, 8, 16, 32, 64]`.
    pub fn new(search_range: i32, algorithm: MeAlgorithm, policy: BlockSizePolicy) -> Self {
        Self {
            block_sizes: vec![4, 8, 16, 32, 64],
            search_range,
            algorithm,
            policy,
        }
    }

    /// Estimate the best-fit motion vector for the block whose top-left corner
    /// is `(block_x, block_y)`.
    ///
    /// The returned `AdaptiveMotionVector` includes the chosen block size.
    /// Note: `block_x` and `block_y` must be aligned to the smallest block size.
    pub fn estimate_block(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
    ) -> AdaptiveMotionVector {
        let mut best_mv: Option<MotionVector> = None;
        let mut best_size = self.block_sizes.first().copied().unwrap_or(8);
        let mut best_score = f64::MAX;

        for &bs in &self.block_sizes {
            // Skip if block extends beyond frame.
            if block_x + bs > width || block_y + bs > height {
                continue;
            }

            let est = MotionEstimator::new(bs, self.search_range, self.algorithm);
            let mv = est.estimate_block(ref_frame, cur_frame, width, block_x, block_y);

            let score = match self.policy {
                BlockSizePolicy::Smallest => {
                    // Always pick the first (smallest) that fits.
                    if best_mv.is_none() {
                        best_mv = Some(mv);
                        best_size = bs;
                    }
                    break;
                }
                BlockSizePolicy::Largest => {
                    // Always replace with latest (largest that fits).
                    best_mv = Some(mv);
                    best_size = bs;
                    continue;
                }
                BlockSizePolicy::BestSadPerPixel => {
                    // SAD normalised by block area (favours larger blocks
                    // when they provide a proportionally lower cost).
                    let area = (bs * bs) as f64;
                    mv.sad as f64 / area
                }
            };

            if score < best_score {
                best_score = score;
                best_size = bs;
                best_mv = Some(mv);
            }
        }

        let mv = best_mv.unwrap_or(MotionVector {
            dx: 0,
            dy: 0,
            sad: 0,
            block_x,
            block_y,
        });

        AdaptiveMotionVector {
            mv,
            chosen_block_size: best_size,
        }
    }
}

// -----------------------------------------------------------------------
// Hierarchical (coarse-to-fine) motion estimation
// -----------------------------------------------------------------------

/// Hierarchical motion estimator that estimates motion on downsampled
/// versions of the frames first, then refines on successively finer scales.
///
/// # Algorithm
///
/// 1. Build a Gaussian pyramid of `levels` layers for both frames.
/// 2. At the coarsest level, perform a full-search or diamond search.
/// 3. Propagate the vector (scaled ×2) to the next finer level as a
///    warm-start centre; restrict the search range at each level.
/// 4. Return the fine-level motion vector.
pub struct HierarchicalEstimator {
    /// Number of pyramid levels (1 = no hierarchy, just flat search).
    pub levels: usize,
    /// Full-resolution block size.
    pub block_size: u32,
    /// Search range at the coarsest level.
    pub coarse_search_range: i32,
    /// Algorithm used at each level.
    pub algorithm: MeAlgorithm,
}

impl HierarchicalEstimator {
    /// Create a new `HierarchicalEstimator`.
    pub fn new(
        levels: usize,
        block_size: u32,
        coarse_search_range: i32,
        algorithm: MeAlgorithm,
    ) -> Self {
        Self {
            levels: levels.max(1),
            block_size,
            coarse_search_range,
            algorithm,
        }
    }

    /// Estimate the motion vector for a block at `(block_x, block_y)` in the
    /// full-resolution frame using hierarchical coarse-to-fine refinement.
    pub fn estimate_block(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
    ) -> MotionVector {
        // Build pyramids (simple 2×2 box downsample).
        let pyramid_ref = build_pyramid(ref_frame, width, height, self.levels);
        let pyramid_cur = build_pyramid(cur_frame, width, height, self.levels);

        // Start from the coarsest level.
        let coarsest = self.levels - 1;
        let scale_factor = 1u32 << coarsest; // 2^levels
        let coarse_bs = (self.block_size / scale_factor).max(1);
        let coarse_w = (width / scale_factor).max(1);

        let coarse_bx = block_x / scale_factor;
        let coarse_by = block_y / scale_factor;

        let coarse_est = MotionEstimator::new(coarse_bs, self.coarse_search_range, self.algorithm);
        let coarse_mv = coarse_est.estimate_block(
            &pyramid_ref[coarsest],
            &pyramid_cur[coarsest],
            coarse_w,
            coarse_bx,
            coarse_by,
        );

        let mut pred_dx = coarse_mv.dx as i32;
        let mut pred_dy = coarse_mv.dy as i32;

        // Refine through finer levels.
        for level in (0..coarsest).rev() {
            // Scale up the prediction.
            pred_dx *= 2;
            pred_dy *= 2;

            let level_scale = 1u32 << level;
            let level_w = (width / level_scale).max(1);
            let level_bs = (self.block_size / level_scale).max(1);
            let level_bx = block_x / level_scale;
            let level_by = block_y / level_scale;

            // Small local search around the predicted position.
            let local_range = 2i32;
            let level_est = MotionEstimator::new(level_bs, local_range, self.algorithm);

            let (ref_plane, cur_plane) = (&pyramid_ref[level], &pyramid_cur[level]);

            let mut best_sad = u32::MAX;
            let mut best_dx = pred_dx as i16;
            let mut best_dy = pred_dy as i16;

            let level_h = (height / level_scale).max(1);
            for dy in -local_range..=local_range {
                for dx in -local_range..=local_range {
                    let cdx = pred_dx + dx;
                    let cdy = pred_dy + dy;
                    let sad = level_est.eval_candidate(
                        ref_plane, cur_plane, level_w, level_h, level_bx, level_by, cdx, cdy,
                    );
                    if sad < best_sad {
                        best_sad = sad;
                        best_dx = cdx as i16;
                        best_dy = cdy as i16;
                    }
                }
            }

            pred_dx = best_dx as i32;
            pred_dy = best_dy as i32;
        }

        MotionVector {
            dx: pred_dx as i16,
            dy: pred_dy as i16,
            sad: {
                // Evaluate final SAD at full resolution.
                let full_est = MotionEstimator::new(self.block_size, 0, self.algorithm);
                full_est.eval_candidate(
                    ref_frame, cur_frame, width, height, block_x, block_y, pred_dx, pred_dy,
                )
            },
            block_x,
            block_y,
        }
    }

    /// Estimate hierarchical motion vectors for every block in the frame.
    pub fn estimate_frame(
        &self,
        ref_frame: &[u8],
        cur_frame: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<MotionVector> {
        let bs = self.block_size;
        let mut vectors = Vec::new();
        let mut by = 0u32;
        while by + bs <= height {
            let mut bx = 0u32;
            while bx + bs <= width {
                vectors.push(self.estimate_block(ref_frame, cur_frame, width, height, bx, by));
                bx += bs;
            }
            by += bs;
        }
        vectors
    }
}

/// Build a Gaussian pyramid of `levels` layers from `frame` (`width × height`).
///
/// Level 0 = full resolution, level `k` = `width/2^k × height/2^k`.
/// Downsampling uses simple 2×2 box averaging.
fn build_pyramid(frame: &[u8], width: u32, height: u32, levels: usize) -> Vec<Vec<u8>> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(frame.to_vec());

    for _ in 1..levels {
        let prev = pyramid.last().expect("at least one level");
        let prev_w = width as usize >> (pyramid.len() - 1);
        let prev_h = height as usize >> (pyramid.len() - 1);
        let out_w = (prev_w / 2).max(1);
        let out_h = (prev_h / 2).max(1);

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
// Parallel motion estimation (rayon)
// -----------------------------------------------------------------------

/// Estimate motion vectors for every block using rayon for parallelism.
///
/// Produces the same results as `MotionEstimator::estimate_frame` but
/// distributes block estimation across all available CPU cores.
pub fn estimate_frame_parallel(
    estimator: &MotionEstimator,
    ref_frame: &[u8],
    cur_frame: &[u8],
    width: u32,
    height: u32,
) -> Vec<MotionVector> {
    use rayon::prelude::*;

    let bs = estimator.block_size;

    // Collect block positions first.
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

    // Estimate in parallel, then sort back into raster order.
    let mut vectors: Vec<MotionVector> = positions
        .par_iter()
        .map(|&(bx, by)| estimator.estimate_block(ref_frame, cur_frame, width, bx, by))
        .collect();

    // Sort by (block_y, block_x) to restore raster order.
    vectors.sort_by_key(|mv| (mv.block_y, mv.block_x));
    vectors
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, fill: u8) -> Vec<u8> {
        vec![fill; width * height]
    }

    fn make_ramp_frame(width: usize, height: usize) -> Vec<u8> {
        (0..(width * height)).map(|i| (i % 256) as u8).collect()
    }

    // 1. compute_sad — identical blocks
    #[test]
    fn test_compute_sad_identical() {
        let block = vec![100u8; 64];
        let sad = compute_sad(&block, &block, 8, 8, 8);
        assert_eq!(sad, 0);
    }

    // 2. compute_sad — known difference
    #[test]
    fn test_compute_sad_known_difference() {
        let a = vec![10u8; 16]; // 4×4 block, each pixel = 10
        let b = vec![20u8; 16]; // each pixel = 20
        let sad = compute_sad(&a, &b, 4, 4, 4);
        assert_eq!(sad, 16 * 10); // 16 pixels × |10-20|
    }

    // 3. FullSearch — zero motion on identical frames
    #[test]
    fn test_full_search_zero_motion_identical() {
        let frame = make_ramp_frame(32, 32);
        let estimator = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let mv = estimator.estimate_block(&frame, &frame, 32, 0, 0);
        assert_eq!(mv.sad, 0);
    }

    // 4. FullSearch — correctly finds the minimum-SAD candidate
    #[test]
    fn test_full_search_nonzero_motion() {
        // Build frames where the cur block (0,0) is filled with value 200,
        // but the ref frame has that same value only at the block starting at
        // (block_size, 0) = (8, 0) — i.e. dx=8 in the reference — while
        // the ref block at (0,0) is filled with 50.
        // Expected: FullSearch finds dx=8, dy=0 with SAD=0.
        let width = 32usize;
        let height = 32usize;
        let block_size = 8usize;
        let search_range = 8i32;

        // Fill everything with 50.
        let mut ref_frame = vec![50u8; width * height];
        let mut cur_frame = vec![50u8; width * height];

        // Place a distinctive pattern (200) in cur at block (0,0).
        for row in 0..block_size {
            for col in 0..block_size {
                cur_frame[row * width + col] = 200;
            }
        }
        // Place matching pattern in ref at block (block_size, 0) = (8, 0).
        for row in 0..block_size {
            for col in 0..block_size {
                ref_frame[row * width + block_size + col] = 200;
            }
        }

        let estimator =
            MotionEstimator::new(block_size as u32, search_range, MeAlgorithm::FullSearch);
        let mv = estimator.estimate_block(&ref_frame, &cur_frame, width as u32, 0, 0);

        // The search must find a zero-SAD match somewhere.
        assert_eq!(
            mv.sad, 0,
            "expected SAD=0, got SAD={} at dx={} dy={}",
            mv.sad, mv.dx, mv.dy
        );
        // The matched reference block must actually contain the same content as the cur block:
        // this verifies that the returned (dx, dy) is valid, not just any candidate.
        let ref_x = (0i32 + mv.dx as i32)
            .max(0)
            .min(width as i32 - block_size as i32) as usize;
        let ref_y = (0i32 + mv.dy as i32)
            .max(0)
            .min(height as i32 - block_size as i32) as usize;
        let ref_block_val = ref_frame[ref_y * width + ref_x];
        let cur_block_val = cur_frame[0];
        assert_eq!(
            ref_block_val, cur_block_val,
            "matched block content mismatch"
        );
    }

    // 5. ThreeStep — zero motion on identical frames
    #[test]
    fn test_three_step_zero_motion_identical() {
        let frame = make_ramp_frame(32, 32);
        let estimator = MotionEstimator::new(8, 8, MeAlgorithm::ThreeStep);
        let mv = estimator.estimate_block(&frame, &frame, 32, 0, 0);
        assert_eq!(mv.sad, 0);
    }

    // 6. DiamondSearch — zero motion on identical frames
    #[test]
    fn test_diamond_search_zero_motion_identical() {
        let frame = make_ramp_frame(32, 32);
        let estimator = MotionEstimator::new(8, 8, MeAlgorithm::DiamondSearch);
        let mv = estimator.estimate_block(&frame, &frame, 32, 0, 0);
        assert_eq!(mv.sad, 0);
    }

    // 7. HexagonSearch — zero motion on identical frames
    #[test]
    fn test_hexagon_search_zero_motion_identical() {
        let frame = make_ramp_frame(32, 32);
        let estimator = MotionEstimator::new(8, 8, MeAlgorithm::HexagonSearch);
        let mv = estimator.estimate_block(&frame, &frame, 32, 0, 0);
        assert_eq!(mv.sad, 0);
    }

    // 8. estimate_frame — correct number of vectors
    #[test]
    fn test_estimate_frame_vector_count() {
        let frame = make_ramp_frame(32, 32);
        let estimator = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let vectors = estimator.estimate_frame(&frame, &frame, 32, 32);
        // 32/8 × 32/8 = 16 blocks
        assert_eq!(vectors.len(), 16);
    }

    // 9. compensate_frame — correct output for zero-motion vectors
    #[test]
    fn test_compensate_frame_zero_motion() {
        let width = 16u32;
        let height = 16u32;
        let ref_frame = make_ramp_frame(16, 16);
        let estimator = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let vectors = estimator.estimate_frame(&ref_frame, &ref_frame, width, height);
        let out = compensate_frame(&ref_frame, &vectors, width, height, 8);
        assert_eq!(out, ref_frame);
    }

    // 10. residual_frame — correct signed differences
    #[test]
    fn test_residual_frame_values() {
        let cur = vec![100u8, 50u8, 200u8];
        let pred = vec![80u8, 60u8, 180u8];
        let res = residual_frame(&cur, &pred);
        assert_eq!(res, vec![20i16, -10i16, 20i16]);
    }

    // 11. reconstruct_from_residual — correct addition
    #[test]
    fn test_reconstruct_from_residual_values() {
        let pred = vec![80u8, 60u8, 180u8];
        let res = vec![20i16, -10i16, 20i16];
        let out = reconstruct_from_residual(&pred, &res);
        assert_eq!(out, vec![100u8, 50u8, 200u8]);
    }

    // 12. reconstruct_from_residual — clamping (no overflow)
    #[test]
    fn test_reconstruct_from_residual_clamp() {
        let pred = vec![10u8, 250u8];
        let res = vec![-20i16, 20i16];
        let out = reconstruct_from_residual(&pred, &res);
        assert_eq!(out[0], 0u8); // 10 - 20 = -10, clamped to 0
        assert_eq!(out[1], 255u8); // 250 + 20 = 270, clamped to 255
    }

    // 13. Round-trip: residual + reconstruct = original
    #[test]
    fn test_roundtrip_residual_reconstruct() {
        let cur = make_ramp_frame(16, 16);
        let pred = make_frame(16, 16, 128);
        let res = residual_frame(&cur, &pred);
        let reconstructed = reconstruct_from_residual(&pred, &res);
        assert_eq!(reconstructed, cur);
    }

    // 14. MotionVector fields are accessible
    #[test]
    fn test_motion_vector_fields() {
        let mv = MotionVector {
            dx: 3,
            dy: -2,
            sad: 42,
            block_x: 16,
            block_y: 32,
        };
        assert_eq!(mv.dx, 3);
        assert_eq!(mv.dy, -2);
        assert_eq!(mv.sad, 42);
        assert_eq!(mv.block_x, 16);
        assert_eq!(mv.block_y, 32);
    }

    // 15. FullSearch and ThreeStep give SAD=0 on identical frames (same quality)
    #[test]
    fn test_full_vs_three_step_identical_frames_same_sad() {
        let frame = make_ramp_frame(32, 32);
        let est_full = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let est_three = MotionEstimator::new(8, 4, MeAlgorithm::ThreeStep);
        let mv_full = est_full.estimate_block(&frame, &frame, 32, 8, 8);
        let mv_three = est_three.estimate_block(&frame, &frame, 32, 8, 8);
        assert_eq!(mv_full.sad, 0);
        assert_eq!(mv_three.sad, 0);
    }

    // ---- Sub-pixel motion estimation tests ----

    // 16. SubPixelMode::None returns integer-pel vector in qpel units
    #[test]
    fn test_subpel_none_returns_integer() {
        let frame = make_ramp_frame(32, 32);
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let mv = est.estimate_block_subpel(&frame, &frame, 32, 32, 0, 0, SubPixelMode::None);
        assert_eq!(mv.precision, SubPixelMode::None);
        // For identical frames, displacement should be zero
        assert_eq!(mv.dx_qpel, 0);
        assert_eq!(mv.dy_qpel, 0);
        assert!(mv.sad < 1e-6);
    }

    // 17. HalfPel refinement on identical frames yields zero displacement
    #[test]
    fn test_subpel_halfpel_identical_frames() {
        let frame = make_ramp_frame(32, 32);
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let mv = est.estimate_block_subpel(&frame, &frame, 32, 32, 0, 0, SubPixelMode::HalfPel);
        assert_eq!(mv.precision, SubPixelMode::HalfPel);
        assert_eq!(mv.dx_qpel, 0);
        assert_eq!(mv.dy_qpel, 0);
    }

    // 18. QuarterPel refinement on identical frames yields zero displacement
    #[test]
    fn test_subpel_quarterpel_identical_frames() {
        let frame = make_ramp_frame(32, 32);
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let mv = est.estimate_block_subpel(&frame, &frame, 32, 32, 8, 8, SubPixelMode::QuarterPel);
        assert_eq!(mv.precision, SubPixelMode::QuarterPel);
        assert_eq!(mv.dx_qpel, 0);
        assert_eq!(mv.dy_qpel, 0);
    }

    // 19. Sub-pixel SAD is <= integer-pel SAD (refinement cannot worsen)
    #[test]
    fn test_subpel_sad_le_integer() {
        let width = 32usize;
        let height = 32usize;
        let mut ref_frame = vec![128u8; width * height];
        let mut cur_frame = vec![128u8; width * height];
        // Introduce a slight shift pattern
        for row in 0..8usize {
            for col in 0..8usize {
                cur_frame[row * width + col] = 200;
                ref_frame[row * width + col + 1] = 200;
            }
        }
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let int_mv = est.estimate_block(&ref_frame, &cur_frame, width as u32, 0, 0);
        let sub_mv = est.estimate_block_subpel(
            &ref_frame,
            &cur_frame,
            width as u32,
            height as u32,
            0,
            0,
            SubPixelMode::HalfPel,
        );
        assert!(
            sub_mv.sad <= int_mv.sad as f64 + 1e-6,
            "sub-pixel SAD {} should be <= integer SAD {}",
            sub_mv.sad,
            int_mv.sad
        );
    }

    // 20. estimate_frame_subpel returns correct number of vectors
    #[test]
    fn test_estimate_frame_subpel_count() {
        let frame = make_ramp_frame(32, 32);
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let vectors = est.estimate_frame_subpel(&frame, &frame, 32, 32, SubPixelMode::QuarterPel);
        assert_eq!(vectors.len(), 16); // 32/8 * 32/8
    }

    // 21. interpolate_bilinear at integer position returns exact pixel
    #[test]
    fn test_bilinear_integer_position() {
        let plane = vec![10u8, 20, 30, 40]; // 2x2
        let val = interpolate_bilinear(&plane, 2, 2, 0, 0, 0.0, 0.0);
        assert!((val - 10.0).abs() < 1e-6);
        let val2 = interpolate_bilinear(&plane, 2, 2, 1, 1, 0.0, 0.0);
        assert!((val2 - 40.0).abs() < 1e-6);
    }

    // 22. interpolate_bilinear at midpoint returns average
    #[test]
    fn test_bilinear_midpoint() {
        let plane = vec![0u8, 100, 0, 100]; // 2x2
        let val = interpolate_bilinear(&plane, 2, 2, 0, 0, 0.5, 0.0);
        assert!((val - 50.0).abs() < 1e-6);
    }

    // 23. 6-tap filter on uniform samples returns same value
    #[test]
    fn test_6tap_uniform() {
        let samples = [128.0f64; 6];
        let val = interpolate_6tap(&samples);
        assert!((val - 128.0).abs() < 1e-6);
    }

    // 24. compensate_frame_subpel with zero-motion returns reference
    #[test]
    fn test_compensate_frame_subpel_zero_motion() {
        let ref_frame = make_ramp_frame(16, 16);
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let vectors =
            est.estimate_frame_subpel(&ref_frame, &ref_frame, 16, 16, SubPixelMode::QuarterPel);
        let out = compensate_frame_subpel(&ref_frame, &vectors, 16, 16, 8);
        // Should match reference frame (zero motion = identity)
        for (i, (&a, &b)) in ref_frame.iter().zip(out.iter()).enumerate() {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 1,
                "pixel {} mismatch: ref={}, out={}",
                i,
                a,
                b,
            );
        }
    }

    // 25. QuarterPel SAD <= HalfPel SAD (finer refinement)
    #[test]
    fn test_quarterpel_sad_le_halfpel() {
        let width = 32usize;
        let height = 32usize;
        let mut ref_frame = vec![100u8; width * height];
        let mut cur_frame = vec![100u8; width * height];
        for row in 0..8usize {
            for col in 0..8usize {
                cur_frame[row * width + col] = 180;
                // Shift by 1 pixel in ref
                if col + 1 < width {
                    ref_frame[row * width + col + 1] = 180;
                }
            }
        }
        let est = MotionEstimator::new(8, 4, MeAlgorithm::FullSearch);
        let half_mv = est.estimate_block_subpel(
            &ref_frame,
            &cur_frame,
            width as u32,
            height as u32,
            0,
            0,
            SubPixelMode::HalfPel,
        );
        let qpel_mv = est.estimate_block_subpel(
            &ref_frame,
            &cur_frame,
            width as u32,
            height as u32,
            0,
            0,
            SubPixelMode::QuarterPel,
        );
        assert!(
            qpel_mv.sad <= half_mv.sad + 1e-6,
            "qpel SAD {} should be <= half-pel SAD {}",
            qpel_mv.sad,
            half_mv.sad
        );
    }

    // ── Tests for SIMD-accelerated SAD functions ──────────────────────────────

    /// Build a 256-byte (16×16) contiguous block filled with a constant value.
    fn make_block_16x16(value: u8) -> Vec<u8> {
        vec![value; 256]
    }

    // 26. sad_adaptive: identical blocks → SAD = 0
    #[test]
    fn test_sad_adaptive_identical_blocks() {
        let block = make_block_16x16(128);
        let result = sad_adaptive(&block, &block, 16, 16);
        assert_eq!(result, 0, "identical blocks must have SAD = 0");
    }

    // 27. sad_adaptive: all-zeros vs all-ones 16×16 → SAD = 256
    #[test]
    fn test_sad_adaptive_all_zeros_vs_all_ones() {
        let zeros = make_block_16x16(0);
        let ones = make_block_16x16(1);
        let result = sad_adaptive(&zeros, &ones, 16, 16);
        assert_eq!(result, 256, "SAD of 0s vs 1s over 256 pixels must be 256");
    }

    // 28. sad_adaptive: known difference, 4×4 block
    #[test]
    fn test_sad_adaptive_4x4_known_difference() {
        let a = vec![10u8; 16];
        let b = vec![20u8; 16];
        let result = sad_adaptive(&a, &b, 4, 4);
        assert_eq!(result, 10 * 16, "each of 16 pixels differs by 10");
    }

    // 29. sad_adaptive: asymmetric pixel values
    #[test]
    fn test_sad_adaptive_asymmetric() {
        let mut a = vec![0u8; 256];
        let mut b = vec![0u8; 256];
        // Only pixel 0 differs: |200 - 100| = 100
        a[0] = 200;
        b[0] = 100;
        let result = sad_adaptive(&a, &b, 16, 16);
        assert_eq!(result, 100);
    }

    // 30. sad_adaptive and scalar give the same result for random-ish data
    #[test]
    fn test_sad_adaptive_matches_scalar() {
        let a: Vec<u8> = (0u8..=255).collect();
        let b: Vec<u8> = (0u8..=255).rev().collect();
        let simd_result = sad_adaptive(&a, &b, 16, 16);
        let scalar_result = sad_16x16_scalar(&a, &b);
        assert_eq!(
            simd_result, scalar_result,
            "sad_adaptive and sad_16x16_scalar must agree: {} vs {}",
            simd_result, scalar_result
        );
    }

    // 31. sad_adaptive: max-difference 16×16 block (0 vs 255) → SAD = 255 × 256
    #[test]
    fn test_sad_adaptive_max_difference() {
        let zeros = make_block_16x16(0);
        let max_block = make_block_16x16(255);
        let result = sad_adaptive(&zeros, &max_block, 16, 16);
        assert_eq!(
            result,
            255 * 256,
            "max difference over 256 pixels must be 255*256"
        );
    }

    // 32. On x86/x86_64: sad_16x16_simd matches sad_16x16_scalar
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_sad_16x16_simd_matches_scalar() {
        let a: Vec<u8> = (0u8..=255).collect();
        let b: Vec<u8> = (128u8..=255).chain(0u8..128).collect();
        let simd = sad_16x16_simd(&a, &b);
        let scalar = sad_16x16_scalar(&a, &b);
        assert_eq!(
            simd, scalar,
            "sad_16x16_simd must match sad_16x16_scalar: {} vs {}",
            simd, scalar
        );
    }

    // 33. sad_16x16_simd: identical blocks → 0
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_sad_16x16_simd_identical() {
        let block = make_block_16x16(77);
        assert_eq!(sad_16x16_simd(&block, &block), 0);
    }

    // 34. sad_adaptive: non-square block (8×4 = 32 pixels, diff = 5 each)
    #[test]
    fn test_sad_adaptive_non_square() {
        let a = vec![50u8; 32];
        let b = vec![55u8; 32];
        let result = sad_adaptive(&a, &b, 8, 4);
        assert_eq!(result, 5 * 32, "32 pixels each differing by 5 = 160");
    }
}
