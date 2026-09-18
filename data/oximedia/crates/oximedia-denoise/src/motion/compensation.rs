//! Motion compensation for frame alignment.
//!
//! Motion compensation uses estimated motion vectors to align frames,
//! enabling better temporal filtering and prediction.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply motion compensation to align a frame using motion vectors.
///
/// # Arguments
/// * `reference` - Reference frame to warp
/// * `motion_vectors` - Motion vectors for each block
/// * `block_size` - Size of blocks used in motion estimation
///
/// # Returns
/// Motion-compensated frame
pub fn motion_compensate(
    reference: &VideoFrame,
    motion_vectors: &[(i16, i16)],
    block_size: usize,
) -> DenoiseResult<VideoFrame> {
    if reference.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = reference.clone();

    // Process each plane
    for (plane_idx, plane) in output.planes.iter_mut().enumerate() {
        let reference_plane = &reference.planes[plane_idx];
        let (width, height) = reference.plane_dimensions(plane_idx);

        let mut compensated = plane.data.clone();

        motion_compensate_plane(
            reference_plane.data.as_ref(),
            &mut compensated,
            motion_vectors,
            width as usize,
            height as usize,
            plane.stride,
            block_size,
        )?;

        // Update plane data
        plane.data = compensated;
    }

    Ok(output)
}

/// Apply motion compensation to a single plane.
#[allow(clippy::too_many_arguments)]
fn motion_compensate_plane(
    reference: &[u8],
    output: &mut [u8],
    motion_vectors: &[(i16, i16)],
    width: usize,
    height: usize,
    stride: usize,
    block_size: usize,
) -> DenoiseResult<()> {
    let num_blocks_x = width.div_ceil(block_size);

    for by in 0..(height / block_size) {
        for bx in 0..(width / block_size) {
            let block_idx = by * num_blocks_x + bx;

            if block_idx >= motion_vectors.len() {
                continue;
            }

            let (mv_x, mv_y) = motion_vectors[block_idx];

            // Copy block with motion compensation
            for y in 0..block_size {
                let py = by * block_size + y;
                if py >= height {
                    break;
                }

                for x in 0..block_size {
                    let px = bx * block_size + x;
                    if px >= width {
                        break;
                    }

                    // Calculate source position with motion vector
                    let src_x = (px as i32 + i32::from(mv_x)).clamp(0, (width - 1) as i32) as usize;
                    let src_y =
                        (py as i32 + i32::from(mv_y)).clamp(0, (height - 1) as i32) as usize;

                    output[py * stride + px] = reference[src_y * stride + src_x];
                }
            }
        }
    }

    Ok(())
}

/// Bidirectional motion compensation using two reference frames.
pub fn bidirectional_motion_compensate(
    reference1: &VideoFrame,
    reference2: &VideoFrame,
    motion_vectors1: &[(i16, i16)],
    motion_vectors2: &[(i16, i16)],
    block_size: usize,
) -> DenoiseResult<VideoFrame> {
    // Compensate both references
    let comp1 = motion_compensate(reference1, motion_vectors1, block_size)?;
    let comp2 = motion_compensate(reference2, motion_vectors2, block_size)?;

    // Average the two compensated frames
    let mut output = comp1.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .for_each(|(plane_idx, plane)| {
            let plane1 = &comp1.planes[plane_idx];
            let plane2 = &comp2.planes[plane_idx];
            let (width, height) = comp1.plane_dimensions(plane_idx);

            let mut data = plane.data.clone();

            for y in 0..(height as usize) {
                for x in 0..(width as usize) {
                    let idx = y * plane.stride + x;
                    let val1 = u16::from(plane1.data[idx]);
                    let val2 = u16::from(plane2.data[idx]);
                    data[idx] = ((val1 + val2) / 2) as u8;
                }
            }

            plane.data = data;
        });

    Ok(output)
}

/// Sub-pixel motion vector (in 1/4-pixel units).
///
/// Fractional components are stored as quarter-pixel offsets:
/// - `dx_qpel = 4 * integer_dx + sub_pixel_offset_x`
/// - Values are signed 16-bit to match integer motion vector conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubPixelMv {
    /// Horizontal displacement in quarter-pixel units.
    pub dx_qpel: i32,
    /// Vertical displacement in quarter-pixel units.
    pub dy_qpel: i32,
}

impl SubPixelMv {
    /// Create a sub-pixel MV from integer pixel displacements (no fractional part).
    #[must_use]
    pub fn from_integer(dx: i16, dy: i16) -> Self {
        Self {
            dx_qpel: i32::from(dx) * 4,
            dy_qpel: i32::from(dy) * 4,
        }
    }

    /// Integer (full-pixel) horizontal displacement.
    #[must_use]
    pub fn dx_int(&self) -> i32 {
        self.dx_qpel / 4
    }

    /// Integer (full-pixel) vertical displacement.
    #[must_use]
    pub fn dy_int(&self) -> i32 {
        self.dy_qpel / 4
    }

    /// Sub-pixel fractional horizontal offset in quarter-pixel units (0–3).
    #[must_use]
    pub fn dx_frac(&self) -> i32 {
        self.dx_qpel.rem_euclid(4)
    }

    /// Sub-pixel fractional vertical offset in quarter-pixel units (0–3).
    #[must_use]
    pub fn dy_frac(&self) -> i32 {
        self.dy_qpel.rem_euclid(4)
    }
}

/// Refine integer motion vectors to quarter-pixel accuracy.
///
/// For each integer-pixel MV, evaluates the four diagonal quarter-pixel
/// positions using bilinear interpolation on the reference plane and keeps
/// the one with the lowest interpolated SAD.
///
/// # Arguments
/// * `reference`      - Reference video frame
/// * `current`        - Current video frame (to compare against)
/// * `motion_vectors` - Integer-pixel motion vectors (one per block)
/// * `block_size`     - Block size in pixels
///
/// # Returns
/// A vector of [`SubPixelMv`] with the same length as `motion_vectors`.
pub fn refine_to_subpixel(
    reference: &VideoFrame,
    current: &VideoFrame,
    motion_vectors: &[(i16, i16)],
    block_size: usize,
) -> DenoiseResult<Vec<SubPixelMv>> {
    if reference.planes.is_empty() || current.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let ref_plane = &reference.planes[0];
    let cur_plane = &current.planes[0];
    let (width, height) = reference.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;
    let stride = ref_plane.stride;

    let num_blocks_x = w.div_ceil(block_size);

    let sub_mvs: Vec<SubPixelMv> = (0..motion_vectors.len())
        .into_par_iter()
        .map(|block_idx| {
            let (dx_int, dy_int) = motion_vectors[block_idx];
            let base_bx = (block_idx % num_blocks_x) * block_size;
            let base_by = (block_idx / num_blocks_x) * block_size;

            if base_bx + block_size > w || base_by + block_size > h {
                return SubPixelMv::from_integer(dx_int, dy_int);
            }

            // Quarter-pixel offsets to test: 0, 1, 2, 3 (in both x and y)
            // This covers all 4×4 = 16 sub-pixel positions in one integer block.
            let mut best_qx = 0i32;
            let mut best_qy = 0i32;
            let mut best_cost = u64::MAX;

            // Test each of the 16 half/quarter-pixel candidate positions
            for qdy in 0i32..4 {
                for qdx in 0i32..4 {
                    let ref_x0 = base_bx as i32 + i32::from(dx_int);
                    let ref_y0 = base_by as i32 + i32::from(dy_int);

                    // Compute interpolated SAD using bilinear weights
                    let cost = subpixel_sad(
                        ref_plane.data.as_ref(),
                        cur_plane.data.as_ref(),
                        w,
                        h,
                        stride,
                        cur_plane.stride,
                        base_bx,
                        base_by,
                        ref_x0,
                        ref_y0,
                        qdx,
                        qdy,
                        block_size,
                    );

                    if cost < best_cost {
                        best_cost = cost;
                        best_qx = qdx;
                        best_qy = qdy;
                    }
                }
            }

            SubPixelMv {
                dx_qpel: i32::from(dx_int) * 4 + best_qx,
                dy_qpel: i32::from(dy_int) * 4 + best_qy,
            }
        })
        .collect();

    Ok(sub_mvs)
}

/// Apply motion compensation using sub-pixel motion vectors.
///
/// Uses bilinear interpolation to reconstruct pixels at fractional positions.
///
/// # Arguments
/// * `reference`  - Reference video frame
/// * `sub_mvs`    - Sub-pixel motion vectors (in quarter-pixel units)
/// * `block_size` - Block size in pixels
///
/// # Returns
/// Sub-pixel motion-compensated frame
pub fn subpixel_motion_compensate(
    reference: &VideoFrame,
    sub_mvs: &[SubPixelMv],
    block_size: usize,
) -> DenoiseResult<VideoFrame> {
    if reference.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = reference.clone();

    for (plane_idx, plane) in output.planes.iter_mut().enumerate() {
        let ref_plane = &reference.planes[plane_idx];
        let (width, height) = reference.plane_dimensions(plane_idx);
        let w = width as usize;
        let h = height as usize;
        let stride = plane.stride;

        let num_blocks_x = w.div_ceil(block_size);

        for (block_idx, mv) in sub_mvs.iter().enumerate() {
            let bx = (block_idx % num_blocks_x) * block_size;
            let by = (block_idx / num_blocks_x) * block_size;

            for y in 0..block_size {
                let py = by + y;
                if py >= h {
                    break;
                }
                for x in 0..block_size {
                    let px = bx + x;
                    if px >= w {
                        break;
                    }

                    // Source position in quarter-pixel units
                    let src_qx = (px as i32) * 4 + mv.dx_qpel;
                    let src_qy = (py as i32) * 4 + mv.dy_qpel;

                    // Integer and fractional parts
                    let ix = src_qx / 4;
                    let iy = src_qy / 4;
                    let fx = src_qx.rem_euclid(4) as f32 / 4.0;
                    let fy = src_qy.rem_euclid(4) as f32 / 4.0;

                    // Clamp to frame
                    let ix0 = ix.clamp(0, (w - 1) as i32) as usize;
                    let ix1 = (ix + 1).clamp(0, (w - 1) as i32) as usize;
                    let iy0 = iy.clamp(0, (h - 1) as i32) as usize;
                    let iy1 = (iy + 1).clamp(0, (h - 1) as i32) as usize;

                    // Bilinear interpolation
                    let p00 = f32::from(ref_plane.data[iy0 * stride + ix0]);
                    let p10 = f32::from(ref_plane.data[iy0 * stride + ix1]);
                    let p01 = f32::from(ref_plane.data[iy1 * stride + ix0]);
                    let p11 = f32::from(ref_plane.data[iy1 * stride + ix1]);

                    let interpolated = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;

                    plane.data[py * stride + px] = interpolated.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(output)
}

/// Compute interpolated SAD for a sub-pixel position.
///
/// Uses bilinear interpolation on the reference block at the given
/// quarter-pixel offsets `(qdx, qdy)`.
#[allow(clippy::too_many_arguments)]
fn subpixel_sad(
    reference: &[u8],
    current: &[u8],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    cur_stride: usize,
    cur_bx: usize,
    cur_by: usize,
    ref_x0: i32,
    ref_y0: i32,
    qdx: i32,
    qdy: i32,
    block_size: usize,
) -> u64 {
    let fx = qdx as f32 / 4.0;
    let fy = qdy as f32 / 4.0;
    let mut sad = 0u64;

    for y in 0..block_size {
        let cur_y = cur_by + y;
        for x in 0..block_size {
            let cur_x = cur_bx + x;

            let ix0 = (ref_x0 + x as i32).clamp(0, (ref_w - 1) as i32) as usize;
            let ix1 = (ref_x0 + x as i32 + 1).clamp(0, (ref_w - 1) as i32) as usize;
            let iy0 = (ref_y0 + y as i32).clamp(0, (ref_h - 1) as i32) as usize;
            let iy1 = (ref_y0 + y as i32 + 1).clamp(0, (ref_h - 1) as i32) as usize;

            let p00 = f32::from(reference[iy0 * ref_stride + ix0]);
            let p10 = f32::from(reference[iy0 * ref_stride + ix1]);
            let p01 = f32::from(reference[iy1 * ref_stride + ix0]);
            let p11 = f32::from(reference[iy1 * ref_stride + ix1]);

            let interp = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            let cur_val = f32::from(current[cur_y * cur_stride + cur_x]);
            sad += (interp - cur_val).abs() as u64;
        }
    }

    sad
}

/// Weighted motion compensation with confidence values.
pub fn weighted_motion_compensate(
    reference: &VideoFrame,
    motion_vectors: &[(i16, i16)],
    weights: &[f32],
    block_size: usize,
) -> DenoiseResult<VideoFrame> {
    if reference.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = reference.clone();

    for (plane_idx, plane) in output.planes.iter_mut().enumerate() {
        let reference_plane = &reference.planes[plane_idx];
        let (width, height) = reference.plane_dimensions(plane_idx);

        let mut compensated = plane.data.clone();

        weighted_motion_compensate_plane(
            reference_plane.data.as_ref(),
            &mut compensated,
            motion_vectors,
            weights,
            width as usize,
            height as usize,
            plane.stride,
            block_size,
        )?;

        plane.data = compensated;
    }

    Ok(output)
}

/// Apply weighted motion compensation to a single plane.
#[allow(clippy::too_many_arguments)]
fn weighted_motion_compensate_plane(
    reference: &[u8],
    output: &mut [u8],
    motion_vectors: &[(i16, i16)],
    weights: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    block_size: usize,
) -> DenoiseResult<()> {
    let num_blocks_x = width.div_ceil(block_size);

    for by in 0..(height / block_size) {
        for bx in 0..(width / block_size) {
            let block_idx = by * num_blocks_x + bx;

            if block_idx >= motion_vectors.len() {
                continue;
            }

            let (mv_x, mv_y) = motion_vectors[block_idx];
            let weight = if block_idx < weights.len() {
                weights[block_idx].clamp(0.0, 1.0)
            } else {
                1.0
            };

            // Copy block with weighted motion compensation
            for y in 0..block_size {
                let py = by * block_size + y;
                if py >= height {
                    break;
                }

                for x in 0..block_size {
                    let px = bx * block_size + x;
                    if px >= width {
                        break;
                    }

                    let original = f32::from(reference[py * stride + px]);

                    let src_x = (px as i32 + i32::from(mv_x)).clamp(0, (width - 1) as i32) as usize;
                    let src_y =
                        (py as i32 + i32::from(mv_y)).clamp(0, (height - 1) as i32) as usize;
                    let compensated = f32::from(reference[src_y * stride + src_x]);

                    // Blend based on weight
                    let blended = (1.0 - weight) * original + weight * compensated;
                    output[py * stride + px] = blended.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_motion_compensate() {
        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let motion_vectors = vec![(0, 0); 16]; // 4x4 blocks for 64x64 with block_size=16

        let result = motion_compensate(&reference, &motion_vectors, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bidirectional_motion_compensate() {
        let mut ref1 = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        ref1.allocate();

        let mut ref2 = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        ref2.allocate();

        let mv1 = vec![(1, 0); 16];
        let mv2 = vec![(-1, 0); 16];

        let result = bidirectional_motion_compensate(&ref1, &ref2, &mv1, &mv2, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_weighted_motion_compensate() {
        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let motion_vectors = vec![(2, 2); 16];
        let weights = vec![0.5; 16];

        let result = weighted_motion_compensate(&reference, &motion_vectors, &weights, 16);
        assert!(result.is_ok());
    }
}
