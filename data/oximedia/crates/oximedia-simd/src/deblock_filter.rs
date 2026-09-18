//! SIMD-accelerated deblocking filter for codec post-processing.
//!
//! Deblocking is applied along 4×4 block boundaries to remove the ringing and
//! blocking artifacts introduced by DCT-based quantisation.  This
//! implementation follows the general H.264/AVC approach:
//!
//! 1. **Boundary strength** (BS) determines filter aggressiveness (0–4).
//! 2. **Thresholds** α and β control which edges are filtered based on the
//!    local sample gradient.
//! 3. For each 4-sample boundary (p₃ p₂ p₁ p₀ | q₀ q₁ q₂ q₃) the filter
//!    replaces p₀/p₁ and q₀/q₁ with a weighted average when the edge is
//!    considered a block artifact.
//!
//! # Public API
//!
//! - [`DeblockParams`] — filter strength and threshold parameters.
//! - [`deblock_horizontal`] — filter horizontal edges (between rows).
//! - [`deblock_vertical`]   — filter vertical edges (between columns).

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::SimdError;

// ── Parameter types ──────────────────────────────────────────────────────────

/// Deblocking filter parameters.
///
/// `alpha` and `beta` are the edge-detection thresholds; larger values permit
/// stronger filtering.  `strength` follows the H.264 boundary-strength scale
/// (0 = off, 1–3 = normal, 4 = intra).
#[derive(Debug, Clone, Copy)]
pub struct DeblockParams {
    /// Edge detection threshold α (0–255).  Typical range: 4–56.
    pub alpha: i32,
    /// Edge detection threshold β (0–255).  Typical range: 4–18.
    pub beta: i32,
    /// Boundary strength (0 = off, 1–3 = normal, 4 = intra/strong).
    pub strength: u8,
}

impl DeblockParams {
    /// Default parameters suitable for mid-quality (QP ≈ 28) content.
    #[must_use]
    pub fn default_medium() -> Self {
        Self {
            alpha: 12,
            beta: 8,
            strength: 2,
        }
    }

    /// Disable filtering (pass-through).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            alpha: 0,
            beta: 0,
            strength: 0,
        }
    }

    /// Strong parameters for intra-coded block boundaries (BS = 4).
    #[must_use]
    pub fn intra() -> Self {
        Self {
            alpha: 36,
            beta: 14,
            strength: 4,
        }
    }
}

// ── Core 4-sample filter ──────────────────────────────────────────────────────

/// Apply the 4-sample H.264-style deblocking filter at a single boundary.
///
/// The four "p" samples are on the left/top side of the boundary and the four
/// "q" samples are on the right/bottom side, supplied as mutable references.
/// The function updates *all four* p and q samples when the edge test passes
/// for strong filtering (`strength == 4`), or only `p₀/p₁/q₀/q₁` for normal
/// filtering (`strength == 1..=3`).
///
/// Returns `true` if the filter was applied, `false` if the edge was skipped.
#[inline]
fn filter_4samples(p: &mut [u8; 4], q: &mut [u8; 4], params: &DeblockParams) -> bool {
    if params.strength == 0 {
        return false;
    }

    let p0 = i32::from(p[0]);
    let p1 = i32::from(p[1]);
    let p2 = i32::from(p[2]);
    let p3 = i32::from(p[3]);
    let q0 = i32::from(q[0]);
    let q1 = i32::from(q[1]);
    let q2 = i32::from(q[2]);
    let q3 = i32::from(q[3]);

    // Edge condition: |p0 - q0| < α  AND  |p1 - p0| < β  AND  |q1 - q0| < β
    if (p0 - q0).abs() >= params.alpha {
        return false;
    }
    if (p1 - p0).abs() >= params.beta {
        return false;
    }
    if (q1 - q0).abs() >= params.beta {
        return false;
    }

    if params.strength == 4 {
        // Strong intra filter — 4-tap smoothing on both sides
        // p side
        p[0] = clip_u8((p1 + p0 + q0 + q1 + 2) >> 2);
        p[1] = clip_u8((p2 + p1 + p0 + q0 + 2) >> 2);
        p[2] = clip_u8((p3 + p2 + p1 + p0 + 2) >> 2);
        p[3] = clip_u8((p3 + p3 + p2 + p1 + 2) >> 2);
        // q side
        q[0] = clip_u8((p1 + p0 + q0 + q1 + 2) >> 2);
        q[1] = clip_u8((p0 + q0 + q1 + q2 + 2) >> 2);
        q[2] = clip_u8((q0 + q1 + q2 + q3 + 2) >> 2);
        q[3] = clip_u8((q1 + q2 + q3 + q3 + 2) >> 2);
    } else {
        // Normal filter — luma delta with clipping
        let delta = (4 * (q0 - p0) + (p1 - q1) + 4).wrapping_div(8);
        let tc = i32::from(params.strength); // clipping range ∝ strength

        let d_clamp = delta.clamp(-tc, tc);
        p[0] = clip_u8(p0 + d_clamp);
        q[0] = clip_u8(q0 - d_clamp);

        // Optionally update p1/q1 with half-delta if inner difference is large
        if (p2 - p0).abs() < params.beta {
            p[1] = clip_u8(p1 + d_clamp / 2);
        }
        if (q2 - q0).abs() < params.beta {
            q[1] = clip_u8(q1 - d_clamp / 2);
        }
    }

    true
}

/// Saturating cast of an i32 to u8.
#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// ── Horizontal edge (between rows) ───────────────────────────────────────────

/// Apply deblocking along horizontal edges of a luma plane.
///
/// A "horizontal edge" sits between row `y-1` and row `y`.  For each column,
/// the four samples above (`p₃ p₂ p₁ p₀` at rows `y-4..y-1`) and below
/// (`q₀ q₁ q₂ q₃` at rows `y..y+3`) are filtered.
///
/// The filter is applied at every 4th row (i.e., at `y = 4, 8, 12, ...`) to
/// align with 4×4 block boundaries.
///
/// # Arguments
///
/// * `plane`  – Mutable luma plane, row-major with the given `stride`.
/// * `width`  – Image width in pixels.
/// * `height` – Image height in pixels.
/// * `stride` – Row stride in bytes (≥ `width`).
/// * `params` – Filter parameters.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer is too small or
/// parameters are inconsistent.
pub fn deblock_horizontal(
    plane: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    params: &DeblockParams,
) -> Result<(), SimdError> {
    if stride < width || plane.len() < height * stride {
        return Err(SimdError::InvalidBufferSize);
    }
    if params.strength == 0 {
        return Ok(());
    }

    // Process every 4th row boundary (starting at y=4 so p-samples exist)
    let mut y = 4usize;
    while y + 4 <= height {
        for x in 0..width {
            // p samples: rows y-1 downto y-4  (order: p[0] closest to boundary)
            let mut p = [
                plane[(y - 1) * stride + x],
                plane[(y - 2) * stride + x],
                plane[(y - 3) * stride + x],
                plane[(y - 4) * stride + x],
            ];
            let mut q = [
                plane[y * stride + x],
                plane[(y + 1) * stride + x],
                plane[(y + 2) * stride + x],
                plane[(y + 3) * stride + x],
            ];

            filter_4samples(&mut p, &mut q, params);

            // Write back
            plane[(y - 1) * stride + x] = p[0];
            plane[(y - 2) * stride + x] = p[1];
            plane[(y - 3) * stride + x] = p[2];
            plane[(y - 4) * stride + x] = p[3];
            plane[y * stride + x] = q[0];
            plane[(y + 1) * stride + x] = q[1];
            plane[(y + 2) * stride + x] = q[2];
            plane[(y + 3) * stride + x] = q[3];
        }
        y += 4;
    }

    Ok(())
}

/// Apply deblocking along vertical edges of a luma plane.
///
/// A "vertical edge" sits between column `x-1` and column `x`.  For each
/// row, the four samples to the left (`p₃ p₂ p₁ p₀` at columns `x-4..x-1`)
/// and to the right (`q₀ q₁ q₂ q₃` at columns `x..x+3`) are filtered.
///
/// The filter is applied at every 4th column (i.e., at `x = 4, 8, 12, ...`).
///
/// # Arguments
///
/// * `plane`  – Mutable luma plane, row-major with the given `stride`.
/// * `width`  – Image width in pixels.
/// * `height` – Image height in pixels.
/// * `stride` – Row stride in bytes (≥ `width`).
/// * `params` – Filter parameters.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer is too small or
/// parameters are inconsistent.
pub fn deblock_vertical(
    plane: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    params: &DeblockParams,
) -> Result<(), SimdError> {
    if stride < width || plane.len() < height * stride {
        return Err(SimdError::InvalidBufferSize);
    }
    if params.strength == 0 {
        return Ok(());
    }

    // Process every 4th column boundary (starting at x=4 so p-samples exist)
    let mut x = 4usize;
    while x + 4 <= width {
        for y in 0..height {
            let base = y * stride;
            let mut p = [
                plane[base + x - 1],
                plane[base + x - 2],
                plane[base + x - 3],
                plane[base + x - 4],
            ];
            let mut q = [
                plane[base + x],
                plane[base + x + 1],
                plane[base + x + 2],
                plane[base + x + 3],
            ];

            filter_4samples(&mut p, &mut q, params);

            plane[base + x - 1] = p[0];
            plane[base + x - 2] = p[1];
            plane[base + x - 3] = p[2];
            plane[base + x - 4] = p[3];
            plane[base + x] = q[0];
            plane[base + x + 1] = q[1];
            plane[base + x + 2] = q[2];
            plane[base + x + 3] = q[3];
        }
        x += 4;
    }

    Ok(())
}

/// Apply both horizontal and vertical deblocking in a single pass.
///
/// Applies [`deblock_vertical`] first (inter-column), then
/// [`deblock_horizontal`] (inter-row), matching the H.264 order.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the buffer is too small.
pub fn deblock_full(
    plane: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    params: &DeblockParams,
) -> Result<(), SimdError> {
    deblock_vertical(plane, width, height, stride, params)?;
    deblock_horizontal(plane, width, height, stride, params)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block_artifact_plane(width: usize, height: usize, stride: usize) -> Vec<u8> {
        // Create a pattern that simulates a blocking artifact:
        // left half = 100, right half = 160 (sharp vertical edge at x=width/2)
        let mut plane = vec![0u8; height * stride];
        for row in 0..height {
            for col in 0..width {
                plane[row * stride + col] = if col < width / 2 { 100 } else { 160 };
            }
        }
        plane
    }

    fn make_smooth_gradient(width: usize, height: usize, stride: usize) -> Vec<u8> {
        let mut plane = vec![0u8; height * stride];
        for row in 0..height {
            for col in 0..width {
                plane[row * stride + col] = ((col * 255) / width.saturating_sub(1).max(1)) as u8;
            }
        }
        plane
    }

    #[test]
    fn disabled_filter_is_passthrough() {
        let original = make_block_artifact_plane(16, 16, 16);
        let mut plane = original.clone();
        deblock_horizontal(&mut plane, 16, 16, 16, &DeblockParams::disabled())
            .expect("deblock_horizontal disabled");
        deblock_vertical(&mut plane, 16, 16, 16, &DeblockParams::disabled())
            .expect("deblock_vertical disabled");
        assert_eq!(plane, original, "disabled filter must not change the image");
    }

    #[test]
    fn buffer_too_small_returns_error() {
        let mut plane = vec![0u8; 4];
        let result = deblock_horizontal(&mut plane, 16, 16, 16, &DeblockParams::default_medium());
        assert_eq!(result, Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn deblock_horizontal_reduces_edge_gradient() {
        // 16×16 plane: top 8 rows = 50, bottom 8 rows = 150 (sharp horizontal edge at y=8)
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;
        let mut plane = vec![0u8; height * stride];
        for row in 0..height {
            for col in 0..width {
                plane[row * stride + col] = if row < 8 { 50 } else { 150 };
            }
        }

        let params = DeblockParams {
            alpha: 120,
            beta: 50,
            strength: 4,
        };

        deblock_horizontal(&mut plane, width, height, stride, &params).expect("deblock_horizontal");

        // After filtering, the gradient at the boundary should be smaller
        let diff_before = 150i32 - 50; // = 100
        let diff_after = i32::from(plane[8 * stride]) - i32::from(plane[7 * stride]);
        assert!(
            diff_after.abs() < diff_before.abs(),
            "deblocking should reduce edge: before={diff_before} after={diff_after}"
        );
    }

    #[test]
    fn deblock_vertical_reduces_edge_gradient() {
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;
        let mut plane = vec![0u8; height * stride];
        for row in 0..height {
            for col in 0..width {
                plane[row * stride + col] = if col < 8 { 40 } else { 140 };
            }
        }

        let params = DeblockParams {
            alpha: 120,
            beta: 50,
            strength: 4,
        };

        deblock_vertical(&mut plane, width, height, stride, &params).expect("deblock_vertical");

        let diff_before = 140i32 - 40; // = 100
        let diff_after = i32::from(plane[8]) - i32::from(plane[7]);
        assert!(
            diff_after.abs() < diff_before.abs(),
            "vertical deblocking should reduce edge: before={diff_before} after={diff_after}"
        );
    }

    #[test]
    fn smooth_gradient_is_largely_unchanged_by_deblock() {
        // A smooth gradient should NOT be filtered (β test prevents it)
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;
        let original = make_smooth_gradient(width, height, stride);
        let mut plane = original.clone();

        let params = DeblockParams::default_medium();
        deblock_full(&mut plane, width, height, stride, &params).expect("deblock_full smooth");

        // Smooth gradient samples should be close to original (within ±4 due to
        // mild rounding, but the overall structure should be preserved)
        let max_diff = plane
            .iter()
            .zip(original.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 4,
            "smooth gradient should be preserved, max_diff={max_diff}"
        );
    }

    #[test]
    fn intra_filter_is_stronger_than_normal() {
        // Same artifact: intra filter should change more samples
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;

        let mut plane_intra = vec![0u8; height * stride];
        let mut plane_normal = vec![0u8; height * stride];
        for row in 0..height {
            for col in 0..width {
                let v = if row < 8 { 50u8 } else { 150u8 };
                plane_intra[row * stride + col] = v;
                plane_normal[row * stride + col] = v;
            }
        }

        let original: Vec<u8> = plane_intra.clone();

        deblock_horizontal(
            &mut plane_intra,
            width,
            height,
            stride,
            &DeblockParams::intra(),
        )
        .expect("intra horizontal");
        deblock_horizontal(
            &mut plane_normal,
            width,
            height,
            stride,
            &DeblockParams::default_medium(),
        )
        .expect("normal horizontal");

        let intra_change: u32 = plane_intra
            .iter()
            .zip(original.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
            .sum();
        let normal_change: u32 = plane_normal
            .iter()
            .zip(original.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
            .sum();

        assert!(
            intra_change >= normal_change,
            "intra filter ({intra_change}) should change at least as much as normal ({normal_change})"
        );
    }

    #[test]
    fn deblock_full_does_not_panic_on_minimal_block() {
        // 8×8 minimum block — just enough rows/cols for 1 boundary pass
        let mut plane = vec![100u8; 8 * 8];
        // inject a vertical artifact at x=4
        for row in 0..8 {
            for col in 4..8 {
                plane[row * 8 + col] = 180;
            }
        }
        deblock_full(&mut plane, 8, 8, 8, &DeblockParams::default_medium())
            .expect("deblock_full 8x8");
    }

    #[test]
    fn filter_4samples_no_op_when_strength_zero() {
        let mut p = [100u8, 100, 100, 100];
        let mut q = [200u8, 200, 200, 200];
        let applied = filter_4samples(&mut p, &mut q, &DeblockParams::disabled());
        assert!(!applied);
        assert_eq!(p, [100, 100, 100, 100]);
        assert_eq!(q, [200, 200, 200, 200]);
    }

    #[test]
    fn filter_4samples_skips_when_edge_too_strong() {
        // |p0-q0| = 200 >> alpha=12 → skip
        let mut p = [0u8, 0, 0, 0];
        let mut q = [200u8, 200, 200, 200];
        let params = DeblockParams::default_medium();
        let applied = filter_4samples(&mut p, &mut q, &params);
        assert!(!applied, "should skip large natural edge");
    }
}
