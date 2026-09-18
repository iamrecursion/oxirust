//! Motion estimation using block matching.
//!
//! Block matching divides frames into blocks and searches for the best
//! matching block in a reference frame, producing motion vectors.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Motion estimator configuration and state.
pub struct MotionEstimator {
    /// Block size for motion estimation (typically 8 or 16).
    pub block_size: usize,
    /// Search range in pixels (typically 8-32).
    pub search_range: i32,
    /// Matching criterion threshold.
    pub threshold: u32,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            search_range: 16,
            threshold: 1000,
        }
    }

    /// Estimate motion between current and reference frames.
    ///
    /// Returns a vector of motion vectors (dx, dy) for each block.
    pub fn estimate(
        &self,
        current: &VideoFrame,
        reference: &VideoFrame,
    ) -> DenoiseResult<Vec<(i16, i16)>> {
        if current.planes.is_empty() || reference.planes.is_empty() {
            return Err(DenoiseError::MotionEstimationError(
                "Frame has no planes".to_string(),
            ));
        }

        // Use luma plane for motion estimation
        let current_plane = &current.planes[0];
        let reference_plane = &reference.planes[0];
        let (width, height) = current.plane_dimensions(0);

        let num_blocks_x = (width as usize).div_ceil(self.block_size);
        let num_blocks_y = (height as usize).div_ceil(self.block_size);
        let num_blocks = num_blocks_x * num_blocks_y;

        // Estimate motion for each block in parallel
        let motion_vectors: Vec<(i16, i16)> = (0..num_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let bx = (block_idx % num_blocks_x) * self.block_size;
                let by = (block_idx / num_blocks_x) * self.block_size;

                self.estimate_block_motion(
                    current_plane.data.as_ref(),
                    reference_plane.data.as_ref(),
                    width as usize,
                    height as usize,
                    current_plane.stride,
                    reference_plane.stride,
                    bx,
                    by,
                )
            })
            .collect();

        Ok(motion_vectors)
    }

    /// Estimate motion for a single block using full search.
    #[allow(clippy::too_many_arguments)]
    fn estimate_block_motion(
        &self,
        current: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        current_stride: usize,
        reference_stride: usize,
        block_x: usize,
        block_y: usize,
    ) -> (i16, i16) {
        let mut best_mv = (0i16, 0i16);
        let mut best_sad = u32::MAX;

        // Full search within search range
        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let ref_x = block_x as i32 + dx;
                let ref_y = block_y as i32 + dy;

                // Check bounds
                if ref_x < 0
                    || ref_y < 0
                    || ref_x + self.block_size as i32 > width as i32
                    || ref_y + self.block_size as i32 > height as i32
                {
                    continue;
                }

                // Compute SAD (Sum of Absolute Differences)
                let sad = self.compute_sad(
                    current,
                    reference,
                    current_stride,
                    reference_stride,
                    block_x,
                    block_y,
                    ref_x as usize,
                    ref_y as usize,
                );

                if sad < best_sad {
                    best_sad = sad;
                    best_mv = (dx as i16, dy as i16);

                    // Early termination if very good match
                    if sad < self.threshold {
                        break;
                    }
                }
            }

            if best_sad < self.threshold {
                break;
            }
        }

        best_mv
    }

    /// Compute Sum of Absolute Differences between two blocks.
    #[allow(clippy::too_many_arguments)]
    fn compute_sad(
        &self,
        current: &[u8],
        reference: &[u8],
        current_stride: usize,
        reference_stride: usize,
        current_x: usize,
        current_y: usize,
        reference_x: usize,
        reference_y: usize,
    ) -> u32 {
        let mut sad = 0u32;

        for y in 0..self.block_size {
            for x in 0..self.block_size {
                let curr_idx = (current_y + y) * current_stride + current_x + x;
                let ref_idx = (reference_y + y) * reference_stride + reference_x + x;

                let diff = (i32::from(current[curr_idx]) - i32::from(reference[ref_idx])).abs();
                sad += diff as u32;
            }
        }

        sad
    }
}

/// Fast motion estimation using diamond search.
pub fn diamond_search(
    current: &VideoFrame,
    reference: &VideoFrame,
    block_size: usize,
) -> DenoiseResult<Vec<(i16, i16)>> {
    let estimator = MotionEstimator {
        block_size,
        search_range: 16,
        threshold: 1000,
    };

    // Diamond search pattern (simplified - uses full search for now)
    estimator.estimate(current, reference)
}

/// Downsample a luma plane by 2x using box averaging.
///
/// Returns `(downsampled_data, new_width, new_height)`.
fn downsample_plane(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> (Vec<u8>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    if new_w == 0 || new_h == 0 {
        return (Vec::new(), 0, 0);
    }
    let mut out = vec![0u8; new_w * new_h];
    for y in 0..new_h {
        for x in 0..new_w {
            let sy = y * 2;
            let sx = x * 2;
            let a = u16::from(data[sy * stride + sx]);
            let b = u16::from(data[sy * stride + (sx + 1).min(width - 1)]);
            let c = u16::from(data[(sy + 1).min(height - 1) * stride + sx]);
            let d = u16::from(data[(sy + 1).min(height - 1) * stride + (sx + 1).min(width - 1)]);
            out[y * new_w + x] = ((a + b + c + d + 2) / 4) as u8;
        }
    }
    (out, new_w, new_h)
}

/// Hierarchical (multi-level pyramid) motion estimation.
///
/// Builds a Gaussian pyramid with up to `num_levels` levels, estimates
/// motion at the coarsest level, then propagates and refines vectors
/// at each finer level. This finds large displacements cheaply at coarse
/// scales, then refines to sub-block accuracy at full resolution.
///
/// # Arguments
/// * `current` - Current video frame
/// * `reference` - Reference video frame
/// * `block_size` - Block size at the finest (full-resolution) level
///
/// # Returns
/// Motion vectors `(dx, dy)` for each block at the finest level
pub fn hierarchical_motion_estimation(
    current: &VideoFrame,
    reference: &VideoFrame,
    block_size: usize,
) -> DenoiseResult<Vec<(i16, i16)>> {
    if current.planes.is_empty() || reference.planes.is_empty() {
        return Err(DenoiseError::MotionEstimationError(
            "Frame has no planes".to_string(),
        ));
    }

    let curr_plane = &current.planes[0];
    let ref_plane = &reference.planes[0];
    let (width, height) = current.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;

    // Build Gaussian pyramid (up to 3 levels)
    let num_levels: usize = 3;
    let mut curr_pyramid: Vec<(Vec<u8>, usize, usize)> = Vec::with_capacity(num_levels);
    let mut ref_pyramid: Vec<(Vec<u8>, usize, usize)> = Vec::with_capacity(num_levels);

    // Level 0 = full resolution (copy data with stride handling)
    let mut curr_l0 = vec![0u8; w * h];
    let mut ref_l0 = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            curr_l0[y * w + x] = curr_plane.data[y * curr_plane.stride + x];
            ref_l0[y * w + x] = ref_plane.data[y * ref_plane.stride + x];
        }
    }
    curr_pyramid.push((curr_l0, w, h));
    ref_pyramid.push((ref_l0, w, h));

    // Build coarser levels
    for _ in 1..num_levels {
        let (prev_c, pw, ph) = curr_pyramid.last().ok_or_else(|| {
            DenoiseError::MotionEstimationError("Pyramid build failed".to_string())
        })?;
        if *pw < block_size * 2 || *ph < block_size * 2 {
            break; // Too small to downsample further
        }
        let (dc, dw, dh) = downsample_plane(prev_c, *pw, *ph, *pw);
        let (prev_r, prw, prh) = ref_pyramid.last().ok_or_else(|| {
            DenoiseError::MotionEstimationError("Pyramid build failed".to_string())
        })?;
        let (dr, _, _) = downsample_plane(prev_r, *prw, *prh, *prw);
        if dw < block_size || dh < block_size {
            break;
        }
        curr_pyramid.push((dc, dw, dh));
        ref_pyramid.push((dr, dw, dh));
    }

    let actual_levels = curr_pyramid.len();

    // Start at coarsest level with large search range, small block search
    let coarse_idx = actual_levels - 1;
    let (ref coarse_curr, cw, ch) = &curr_pyramid[coarse_idx];
    let (ref coarse_ref, _, _) = &ref_pyramid[coarse_idx];

    let coarse_bs = block_size;
    let coarse_nbx = (*cw).div_ceil(coarse_bs);
    let coarse_nby = (*ch).div_ceil(coarse_bs);

    let coarse_estimator = MotionEstimator {
        block_size: coarse_bs,
        search_range: 16,
        threshold: 800,
    };

    // Estimate at coarsest level
    let mut prev_mvs: Vec<(i16, i16)> = (0..(coarse_nbx * coarse_nby))
        .into_par_iter()
        .map(|block_idx| {
            let bx = (block_idx % coarse_nbx) * coarse_bs;
            let by = (block_idx / coarse_nbx) * coarse_bs;
            coarse_estimator.estimate_block_motion(
                coarse_curr,
                coarse_ref,
                *cw,
                *ch,
                *cw,
                *cw,
                bx,
                by,
            )
        })
        .collect();

    // Propagate and refine at each finer level
    for level in (0..coarse_idx).rev() {
        let (ref level_curr, lw, lh) = &curr_pyramid[level];
        let (ref level_ref, _, _) = &ref_pyramid[level];
        let level_nbx = (*lw).div_ceil(block_size);
        let level_nby = (*lh).div_ceil(block_size);
        let prev_nbx = (lw / 2).div_ceil(block_size).max(1);

        let refinement_range: i32 = 4; // Small local search for refinement

        let refined: Vec<(i16, i16)> = (0..(level_nbx * level_nby))
            .into_par_iter()
            .map(|block_idx| {
                let bx = (block_idx % level_nbx) * block_size;
                let by = (block_idx / level_nbx) * block_size;

                // Get predicted MV from coarser level (scale by 2)
                let coarse_bx = bx / 2 / block_size;
                let coarse_by = by / 2 / block_size;
                let coarse_idx =
                    (coarse_by * prev_nbx + coarse_bx).min(prev_mvs.len().saturating_sub(1));
                let (pred_dx, pred_dy) = prev_mvs[coarse_idx];
                let pred_dx = pred_dx * 2; // Scale up
                let pred_dy = pred_dy * 2;

                // Build candidate set: refinement around prediction + (0,0)
                let mut candidates: Vec<(i32, i32)> = Vec::with_capacity(
                    ((2 * refinement_range + 1) * (2 * refinement_range + 1) + 1) as usize,
                );
                // Always include zero motion as a candidate
                candidates.push((0, 0));
                for rdy in -refinement_range..=refinement_range {
                    for rdx in -refinement_range..=refinement_range {
                        let dx = i32::from(pred_dx) + rdx;
                        let dy = i32::from(pred_dy) + rdy;
                        candidates.push((dx, dy));
                    }
                }

                // Refine around the predicted MV
                let mut best_mv = (pred_dx, pred_dy);
                let mut best_sad = u32::MAX;

                for &(dx, dy) in &candidates {
                    let ref_x = bx as i32 + dx;
                    let ref_y = by as i32 + dy;

                    if ref_x < 0
                        || ref_y < 0
                        || ref_x + block_size as i32 > *lw as i32
                        || ref_y + block_size as i32 > *lh as i32
                    {
                        continue;
                    }

                    // Check current block bounds too
                    if bx + block_size > *lw || by + block_size > *lh {
                        continue;
                    }

                    // Compute SAD
                    let mut sad = 0u32;
                    for sy in 0..block_size {
                        for sx in 0..block_size {
                            let ci = (by + sy) * *lw + bx + sx;
                            let ri = (ref_y as usize + sy) * *lw + ref_x as usize + sx;
                            if ci < level_curr.len() && ri < level_ref.len() {
                                let diff =
                                    (i32::from(level_curr[ci]) - i32::from(level_ref[ri])).abs();
                                sad += diff as u32;
                            }
                        }
                    }

                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = (dx as i16, dy as i16);
                    }
                }

                best_mv
            })
            .collect();

        prev_mvs = refined;
    }

    Ok(prev_mvs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_motion_estimator_creation() {
        let estimator = MotionEstimator::new(16);
        assert_eq!(estimator.block_size, 16);
        assert_eq!(estimator.search_range, 16);
    }

    #[test]
    fn test_motion_estimation() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let estimator = MotionEstimator::new(16);
        let result = estimator.estimate(&current, &reference);

        assert!(result.is_ok());
        let motion_vectors = result.expect("motion_vectors should be valid");
        assert!(!motion_vectors.is_empty());
    }

    #[test]
    fn test_diamond_search() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let result = diamond_search(&current, &reference, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hierarchical_motion_estimation() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let result = hierarchical_motion_estimation(&current, &reference, 16);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // Hierarchical block matching tests
    // -------------------------------------------------------------------

    #[test]
    fn test_downsample_plane() {
        let data = vec![128u8; 64 * 64];
        let (down, w, h) = downsample_plane(&data, 64, 64, 64);
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(down.len(), 32 * 32);
        // Uniform input should produce uniform output
        for &v in &down {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_downsample_plane_small() {
        let data = vec![100u8; 4 * 4];
        let (down, w, h) = downsample_plane(&data, 4, 4, 4);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(down.len(), 4);
    }

    #[test]
    fn test_downsample_plane_zero() {
        // 1x1 cannot be halved meaningfully
        let data = vec![200u8; 1];
        let (down, w, h) = downsample_plane(&data, 1, 1, 1);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
        assert!(down.is_empty());
    }

    #[test]
    fn test_hierarchical_identical_frames() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let reference = current.clone();

        let result = hierarchical_motion_estimation(&current, &reference, 8);
        assert!(result.is_ok());

        let mvs = result.expect("hierarchical estimation should succeed");
        // For identical frames, each MV should produce SAD=0
        // (any MV is valid for uniform data, so just check we got results)
        let expected_count = (64usize).div_ceil(8) * (64usize).div_ceil(8);
        assert_eq!(mvs.len(), expected_count);
    }

    #[test]
    fn test_hierarchical_unique_content_identical() {
        // Use unique per-block content so SAD is only zero for MV=(0,0)
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();
        {
            let stride = current.planes[0].stride;
            // Use a pattern that is locally unique (hash-like)
            for y in 0..64usize {
                for x in 0..64usize {
                    let val = ((x * 7 + y * 31 + x * y) % 200 + 20) as u8;
                    current.planes[0].data[y * stride + x] = val;
                }
            }
        }

        let reference = current.clone();

        let result = hierarchical_motion_estimation(&current, &reference, 8);
        assert!(result.is_ok());

        let mvs = result.expect("hierarchical estimation should succeed");
        // With unique content, identical frames should find (0,0) or very close
        let zero_count = mvs.iter().filter(|&&(dx, dy)| dx == 0 && dy == 0).count();
        assert!(
            zero_count > mvs.len() / 2,
            "Most MVs should be zero for identical unique-content frames, got {zero_count}/{}",
            mvs.len()
        );
    }

    #[test]
    fn test_hierarchical_produces_correct_count() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();
        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let block_size = 8;
        let result = hierarchical_motion_estimation(&current, &reference, block_size);
        assert!(result.is_ok());

        let mvs = result.expect("should produce MVs");
        let expected_blocks_x = 64usize.div_ceil(block_size);
        let expected_blocks_y = 64usize.div_ceil(block_size);
        assert_eq!(
            mvs.len(),
            expected_blocks_x * expected_blocks_y,
            "Should have one MV per block"
        );
    }

    #[test]
    fn test_hierarchical_larger_frame() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 128, 128);
        current.allocate();
        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 128, 128);
        reference.allocate();

        let result = hierarchical_motion_estimation(&current, &reference, 16);
        assert!(result.is_ok());

        let mvs = result.expect("should produce MVs for large frame");
        assert_eq!(mvs.len(), (128 / 16) * (128 / 16));
    }

    #[test]
    fn test_hierarchical_empty_planes() {
        let current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        let reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        // Not allocated — empty planes
        let result = hierarchical_motion_estimation(&current, &reference, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_block_size_8() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();
        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let result = hierarchical_motion_estimation(&current, &reference, 8);
        assert!(result.is_ok());
        let mvs = result.expect("block_size 8 should work");
        assert_eq!(mvs.len(), 8 * 8);
    }
}
