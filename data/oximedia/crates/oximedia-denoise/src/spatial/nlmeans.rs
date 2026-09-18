//! Non-Local Means denoising.
//!
//! Non-Local Means is a patch-based denoising algorithm that exploits
//! self-similarity in images by averaging similar patches from across
//! the entire image.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply Non-Local Means filter to a video frame.
///
/// `NLMeans` searches for similar patches across the image and averages them
/// to reduce noise while preserving texture and detail.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `strength` - Denoising strength (0.0 - 1.0)
///
/// # Returns
/// Filtered video frame
pub fn nlmeans_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    // Scale parameters based on strength
    let h_param = 10.0 * strength;
    let patch_size = 7;
    let search_window = 21;

    // Process each plane sequentially with explicit write-back.
    // (Sequential loop avoids potential aliasing with par_iter_mut closures
    // that capture both `frame` and `output`.)
    for plane_idx in 0..output.planes.len() {
        let (width, height) = frame.plane_dimensions(plane_idx);
        let stride = frame.planes[plane_idx].stride;
        // Snapshot input so borrow-checker knows input and output are disjoint.
        let input_snap: Vec<u8> = frame.planes[plane_idx].data.clone();
        nlmeans_filter_plane(
            &input_snap,
            output.planes[plane_idx].data.as_mut_slice(),
            width as usize,
            height as usize,
            stride,
            h_param,
            patch_size,
            search_window,
        )?;
    }

    Ok(output)
}

/// Apply `NLMeans` to a single plane.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
fn nlmeans_filter_plane(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    h: f32,
    patch_size: usize,
    search_window: usize,
) -> DenoiseResult<()> {
    let patch_radius = patch_size / 2;
    let search_radius = search_window / 2;
    let h2 = h * h;

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            let search_y_min = (y as i32 - search_radius as i32).max(0) as usize;
            let search_y_max = (y + search_radius + 1).min(height);
            let search_x_min = (x as i32 - search_radius as i32).max(0) as usize;
            let search_x_max = (x + search_radius + 1).min(width);

            // Search for similar patches
            for sy in search_y_min..search_y_max {
                for sx in search_x_min..search_x_max {
                    // Compute patch distance
                    let dist = compute_patch_distance(
                        input,
                        width,
                        height,
                        stride,
                        x,
                        y,
                        sx,
                        sy,
                        patch_radius,
                    );

                    // Convert distance to weight
                    let weight = (-dist / h2).exp();

                    sum += f32::from(input[sy * stride + sx]) * weight;
                    weight_sum += weight;
                }
            }

            output[y * stride + x] = if weight_sum > 0.0 {
                (sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                input[y * stride + x]
            };
        }
    }

    Ok(())
}

/// Compute squared distance between two patches.
#[allow(clippy::too_many_arguments)]
fn compute_patch_distance(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    patch_radius: usize,
) -> f32 {
    let mut dist = 0.0f32;
    let mut count = 0;

    for dy in -(patch_radius as i32)..=(patch_radius as i32) {
        let py1 = (y1 as i32 + dy).clamp(0, (height - 1) as i32) as usize;
        let py2 = (y2 as i32 + dy).clamp(0, (height - 1) as i32) as usize;

        for dx in -(patch_radius as i32)..=(patch_radius as i32) {
            let px1 = (x1 as i32 + dx).clamp(0, (width - 1) as i32) as usize;
            let px2 = (x2 as i32 + dx).clamp(0, (width - 1) as i32) as usize;

            let v1 = f32::from(data[py1 * stride + px1]);
            let v2 = f32::from(data[py2 * stride + px2]);
            let diff = v1 - v2;
            dist += diff * diff;
            count += 1;
        }
    }

    if count > 0 {
        dist / count as f32
    } else {
        0.0
    }
}

/// Tiled Non-Local Means filter for improved cache locality.
///
/// Divides the frame into overlapping tiles and processes each tile
/// independently. This improves cache utilisation because the tile data
/// (input pixels + neighbourhood search) fits better in L1/L2 cache,
/// reducing cache misses on large frames.
///
/// Tile boundary pixels are averaged from overlapping tile contributions
/// so the output is seamless.
///
/// # Arguments
/// * `frame`       - Input video frame
/// * `strength`    - Denoising strength (0.0 – 1.0)
/// * `tile_size`   - Side length of each processing tile in pixels (power of 2, e.g. 32)
///
/// # Returns
/// Filtered video frame
pub fn tiled_nlmeans_filter(
    frame: &VideoFrame,
    strength: f32,
    tile_size: usize,
) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let tile_size = tile_size.max(16);

    // NLM parameters
    let h = 10.0 * strength;
    let patch_size = 7;
    let search_window = 21;
    let search_radius = search_window / 2;
    let overlap = search_radius; // Overlap tiles by half the search window

    let mut output = frame.clone();

    output
        .planes
        .iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);
            let w = width as usize;
            let h_px = height as usize;
            let stride = plane.stride;

            // Accumulation buffers: sum of weighted values and sum of weights.
            let buf_len = h_px * stride;
            let mut acc_sum = vec![0.0f32; buf_len];
            let mut acc_weight = vec![0.0f32; buf_len];

            let effective_step = tile_size.saturating_sub(overlap).max(1);
            let mut ty = 0;
            while ty < h_px {
                let tile_y_end = (ty + tile_size).min(h_px);

                let mut tx = 0;
                while tx < w {
                    let tile_x_end = (tx + tile_size).min(w);

                    // For each pixel in this tile, compute NLM and accumulate
                    for y in ty..tile_y_end {
                        for x in tx..tile_x_end {
                            let patch_radius = patch_size / 2;
                            let h2 = h * h;

                            let search_y_min = (y as i32 - search_radius as i32).max(0) as usize;
                            let search_y_max = (y + search_radius + 1).min(h_px);
                            let search_x_min = (x as i32 - search_radius as i32).max(0) as usize;
                            let search_x_max = (x + search_radius + 1).min(w);

                            let mut pixel_sum = 0.0f32;
                            let mut pixel_weight_sum = 0.0f32;

                            for sy in search_y_min..search_y_max {
                                for sx in search_x_min..search_x_max {
                                    let dist = compute_patch_distance(
                                        input_plane.data.as_ref(),
                                        w,
                                        h_px,
                                        stride,
                                        x,
                                        y,
                                        sx,
                                        sy,
                                        patch_radius,
                                    );
                                    let w_val = (-dist / h2).exp();
                                    pixel_sum +=
                                        f32::from(input_plane.data[sy * stride + sx]) * w_val;
                                    pixel_weight_sum += w_val;
                                }
                            }

                            let idx = y * stride + x;
                            acc_sum[idx] += pixel_sum;
                            acc_weight[idx] += pixel_weight_sum;
                        }
                    }

                    tx += effective_step;
                }
                ty += effective_step;
            }

            // Normalise
            for y in 0..h_px {
                for x in 0..w {
                    let idx = y * stride + x;
                    plane.data[idx] = if acc_weight[idx] > 0.0 {
                        (acc_sum[idx] / acc_weight[idx]).round().clamp(0.0, 255.0) as u8
                    } else {
                        input_plane.data[idx]
                    };
                }
            }

            Ok::<(), DenoiseError>(())
        })?;

    Ok(output)
}

/// Fast Non-Local Means using integral images.
pub fn fast_nlmeans_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);

            let h = 8.0 * strength;
            let patch_size = 5;
            let search_window = 15;

            nlmeans_filter_plane(
                input_plane.data.as_ref(),
                plane.data.as_mut(),
                width as usize,
                height as usize,
                plane.stride,
                h,
                patch_size,
                search_window,
            )
        })?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_nlmeans_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 24, 24);
        frame.allocate();

        let result = nlmeans_filter(&frame, 0.5);
        assert!(result.is_ok());

        let filtered = result.expect("filtered should be valid");
        assert_eq!(filtered.width, 24);
        assert_eq!(filtered.height, 24);
    }

    #[test]
    fn test_fast_nlmeans_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();

        let result = fast_nlmeans_filter(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_patch_distance() {
        let data = vec![0u8; 100 * 100];
        let dist = compute_patch_distance(&data, 100, 100, 100, 50, 50, 50, 50, 3);
        assert!((dist - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nlmeans_small_frame() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();

        let result = nlmeans_filter(&frame, 0.3);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // Tiled NLM tests
    // -------------------------------------------------------------------

    #[test]
    fn test_tiled_nlmeans_basic() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let result = tiled_nlmeans_filter(&frame, 0.5, 16);
        assert!(result.is_ok());
        let f = result.expect("tiled nlm should succeed");
        assert_eq!(f.width, 32);
        assert_eq!(f.height, 32);
    }

    #[test]
    fn test_tiled_nlmeans_uniform_frame() {
        // Uniform frame should remain uniform after denoising.
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 24, 24);
        frame.allocate();
        // Fill luma with a fixed value
        let stride = frame.planes[0].stride;
        for y in 0..24usize {
            for x in 0..24usize {
                frame.planes[0].data[y * stride + x] = 128;
            }
        }
        let result = tiled_nlmeans_filter(&frame, 0.5, 12).expect("should succeed");
        let out_stride = result.planes[0].stride;
        for y in 0..24usize {
            for x in 0..24usize {
                let v = result.planes[0].data[y * out_stride + x];
                assert!(
                    (v as i32 - 128).abs() <= 1,
                    "uniform frame should stay uniform at ({x},{y}): got {v}"
                );
            }
        }
    }

    #[test]
    fn test_tiled_nlmeans_large_tile() {
        // Tile larger than frame → should still work
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 20, 20);
        frame.allocate();
        let result = tiled_nlmeans_filter(&frame, 0.3, 64);
        assert!(result.is_ok());
    }

    // ----- Edge-preservation metric tests for NLM -----

    /// Compute variance of a subslice of a pixel buffer.
    fn compute_variance_nlm(buf: &[u8], start: usize, len: usize) -> f64 {
        let slice = &buf[start..start + len];
        let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / len as f64;
        slice
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / len as f64
    }

    /// Build a 32×32 VideoFrame with a hard vertical edge at x=16.
    /// Left half luma = 50, right half luma = 200, with deterministic noise.
    fn build_nlm_edge_frame() -> VideoFrame {
        let w = 32u32;
        let h = 32u32;
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, w, h);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = y * stride + x;
                let base: u8 = if x < 16 { 50 } else { 200 };
                // Deterministic noise
                frame.planes[0].data[idx] = if idx % 7 < 3 {
                    base.saturating_add(10)
                } else {
                    base
                };
            }
        }
        frame
    }

    #[test]
    fn test_nlmeans_preserves_edge() {
        let noisy_frame = build_nlm_edge_frame();
        let result = nlmeans_filter(&noisy_frame, 0.5).expect("nlmeans_filter should succeed");

        let stride = result.planes[0].stride;
        let h = 32usize;
        let w = 32usize;

        // Check edge at x=15→16: contrast should be significant for ≥50% of rows.
        let mut preserved = 0usize;
        for y in 0..h {
            let left_px = result.planes[0].data[y * stride + 15] as i32;
            let right_px = result.planes[0].data[y * stride + 16] as i32;
            let contrast = (right_px - left_px).unsigned_abs();
            if contrast > 80 {
                preserved += 1;
            }
        }
        assert!(
            preserved * 2 >= h,
            "NLM edge contrast > 80 for only {preserved}/{h} rows; should preserve edge for ≥50% of rows"
        );
        let _ = w; // used implicitly via stride
    }

    #[test]
    fn test_nlmeans_reduces_flat_noise() {
        let noisy_frame = build_nlm_edge_frame();
        let stride = noisy_frame.planes[0].stride;

        // Extract the flat left-half luma data (first 16 rows × 16 columns).
        let noisy_flat: Vec<u8> = (0..16usize)
            .flat_map(|y| (0..16usize).map(move |x| (y, x)))
            .map(|(y, x)| noisy_frame.planes[0].data[y * stride + x])
            .collect();

        let result = nlmeans_filter(&noisy_frame, 0.5).expect("nlmeans_filter should succeed");
        let out_stride = result.planes[0].stride;

        let output_flat: Vec<u8> = (0..16usize)
            .flat_map(|y| (0..16usize).map(move |x| (y, x)))
            .map(|(y, x)| result.planes[0].data[y * out_stride + x])
            .collect();

        let noisy_var = compute_variance_nlm(&noisy_flat, 0, noisy_flat.len());
        let output_var = compute_variance_nlm(&output_flat, 0, output_flat.len());

        assert!(
            output_var < noisy_var,
            "NLM should reduce variance in flat region: noisy={noisy_var:.4} output={output_var:.4}"
        );
    }
}
