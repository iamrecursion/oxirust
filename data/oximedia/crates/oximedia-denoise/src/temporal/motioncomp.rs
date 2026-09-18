//! Motion-compensated temporal denoising.
//!
//! Motion-compensated filtering aligns frames using motion vectors before
//! applying temporal filtering, allowing for better noise reduction in
//! scenes with motion.

use crate::motion::estimation::MotionEstimator;
use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Apply motion-compensated temporal filtering.
///
/// Estimates motion between frames, aligns them, and applies temporal
/// filtering to reduce noise while preserving motion detail.
///
/// # Arguments
/// * `current_frame` - The current frame to denoise
/// * `frame_buffer` - Buffer of recent frames
/// * `estimator` - Motion estimator
/// * `strength` - Denoising strength (0.0 - 1.0)
///
/// # Returns
/// Denoised frame
pub fn motion_compensated_denoise(
    current_frame: &VideoFrame,
    frame_buffer: &[VideoFrame],
    estimator: &mut MotionEstimator,
    strength: f32,
) -> DenoiseResult<VideoFrame> {
    if frame_buffer.len() < 2 {
        return Ok(current_frame.clone());
    }

    // Get reference frame (previous frame)
    let reference_idx = frame_buffer.len() - 2;
    let reference = &frame_buffer[reference_idx];

    // Estimate motion between current and reference
    let motion_vectors = estimator.estimate(current_frame, reference)?;

    // Apply motion-compensated filtering
    motion_compensated_filter(current_frame, reference, &motion_vectors, strength)
}

/// Apply motion-compensated filtering using estimated motion vectors.
fn motion_compensated_filter(
    current: &VideoFrame,
    reference: &VideoFrame,
    motion_vectors: &[(i16, i16)],
    strength: f32,
) -> DenoiseResult<VideoFrame> {
    if current.planes.is_empty() || reference.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = current.clone();
    let block_size = 16; // Motion estimation block size

    for (plane_idx, plane) in output.planes.iter_mut().enumerate() {
        let current_plane = &current.planes[plane_idx];
        let reference_plane = &reference.planes[plane_idx];
        let (width, height) = current.plane_dimensions(plane_idx);

        let mut output_data = plane.data.clone();

        for by in 0..(height as usize / block_size) {
            for bx in 0..(width as usize / block_size) {
                let block_idx = by * (width as usize / block_size) + bx;
                if block_idx >= motion_vectors.len() {
                    continue;
                }

                let (mv_x, mv_y) = motion_vectors[block_idx];

                // Apply motion compensation for this block
                for y in 0..block_size {
                    let py = by * block_size + y;
                    if py >= height as usize {
                        break;
                    }

                    for x in 0..block_size {
                        let px = bx * block_size + x;
                        if px >= width as usize {
                            break;
                        }

                        let current_val = f32::from(current_plane.data[py * plane.stride + px]);

                        // Get motion-compensated reference value
                        let ref_x =
                            (px as i32 + i32::from(mv_x)).clamp(0, (width - 1) as i32) as usize;
                        let ref_y =
                            (py as i32 + i32::from(mv_y)).clamp(0, (height - 1) as i32) as usize;
                        let ref_val =
                            f32::from(reference_plane.data[ref_y * reference_plane.stride + ref_x]);

                        // Blend based on strength
                        let blended = (1.0 - strength) * current_val + strength * ref_val;

                        output_data[py * plane.stride + px] =
                            blended.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        // Update plane data
        plane.data = output_data;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_motion_compensated_denoise() {
        let mut frames = Vec::new();
        for _ in 0..3 {
            let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
            frame.allocate();
            frames.push(frame);
        }

        let mut estimator = MotionEstimator::new(16);

        let result = motion_compensated_denoise(&frames[2], &frames, &mut estimator, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_motion_compensated_insufficient_frames() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let mut estimator = MotionEstimator::new(16);

        let result = motion_compensated_denoise(&frame, &[frame.clone()], &mut estimator, 0.5);
        assert!(result.is_ok());
    }
}
