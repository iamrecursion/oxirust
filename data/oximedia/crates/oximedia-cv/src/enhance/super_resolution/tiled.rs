//! Tiled super-resolution processing for large images.
//!
//! Splits the input into overlapping tiles, runs SR on each tile, then
//! reassembles with linear-feather blending across overlap regions.
//! This bounds peak memory usage to `O(tile_size²)` instead of `O(image_size²)`.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::enhance::super_resolution::tiled::{TiledSrConfig, process_tiled};
//!
//! let input = vec![128u8; 64 * 64 * 3];
//! let config = TiledSrConfig { tile_size: 32, overlap: 4, scale: 2 };
//! let result = process_tiled(&input, 64, 64, 3, &config, |tile, w, h, c| {
//!     Ok(vec![128u8; w * 2 * h * 2 * c])
//! }).unwrap();
//! assert_eq!(result.len(), 128 * 128 * 3);
//! ```

use crate::error::{CvError, CvResult};

/// Configuration for tiled super-resolution.
///
/// Distinct from [`super::TileConfig`] (which describes tile geometry for the
/// ESRGAN/model pipeline); this struct is the simpler config for the
/// functional [`process_tiled`] API.
#[derive(Debug, Clone)]
pub struct TiledSrConfig {
    /// Tile size in pixels (width and height are equal, in the *source* image).
    pub tile_size: usize,
    /// Overlap halo in pixels on each side (in the *source* image).
    pub overlap: usize,
    /// Scale factor applied by the SR function (e.g. `2` for 2×).
    pub scale: u32,
}

impl Default for TiledSrConfig {
    fn default() -> Self {
        Self {
            tile_size: 128,
            overlap: 16,
            scale: 2,
        }
    }
}

impl TiledSrConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CvError::InvalidParameter`] if any field is out of range.
    pub fn validate(&self) -> CvResult<()> {
        if self.tile_size == 0 {
            return Err(CvError::invalid_parameter("tile_size", "must be > 0"));
        }
        if self.scale == 0 {
            return Err(CvError::invalid_parameter("scale", "must be > 0"));
        }
        if self.overlap >= self.tile_size {
            return Err(CvError::invalid_parameter(
                "overlap",
                format!(
                    "{} must be < tile_size ({})",
                    self.overlap, self.tile_size
                ),
            ));
        }
        Ok(())
    }
}

/// Process a large image with tiled SR to bound peak memory.
///
/// Each tile (with overlap halo) is processed independently; overlap zones are
/// blended with linear feathering to hide tile boundaries.
///
/// # Arguments
///
/// * `input`    — Raw pixel bytes, row-major, interleaved channels.
/// * `width`    — Image width in pixels.
/// * `height`   — Image height in pixels.
/// * `channels` — Number of channels per pixel (e.g. `3` for RGB).
/// * `config`   — Tile geometry and scale factor.
/// * `sr_fn`    — SR upscale function.  Receives `(tile_bytes, tile_w, tile_h, channels)`
///               and must return upscaled bytes at `(tile_w * scale, tile_h * scale)`.
///
/// # Errors
///
/// Returns an error if the config is invalid or if `sr_fn` returns an error.
pub fn process_tiled<F>(
    input: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    config: &TiledSrConfig,
    sr_fn: F,
) -> CvResult<Vec<u8>>
where
    F: Fn(&[u8], usize, usize, usize) -> CvResult<Vec<u8>>,
{
    config.validate()?;

    if channels == 0 {
        return Err(CvError::invalid_parameter("channels", "must be > 0"));
    }
    let expected_len = width * height * channels;
    if input.len() < expected_len {
        return Err(CvError::insufficient_data(expected_len, input.len()));
    }

    let out_w = width * config.scale as usize;
    let out_h = height * config.scale as usize;
    let mut accum: Vec<f32> = vec![0.0_f32; out_w * out_h * channels];
    let mut weight: Vec<f32> = vec![0.0_f32; out_w * out_h * channels];

    let step = config.tile_size;
    let overlap = config.overlap;
    let scale = config.scale as usize;

    let mut ty = 0;
    while ty < height {
        let mut tx = 0;
        while tx < width {
            // Source tile bounds (with halo, clamped to image edges).
            // Track actual halo sizes so feathering only applies where there
            // is a real halo (adjacent tile to blend with).
            let src_x0 = tx.saturating_sub(overlap);
            let src_y0 = ty.saturating_sub(overlap);
            let src_x1 = (tx + step + overlap).min(width);
            let src_y1 = (ty + step + overlap).min(height);

            // How many pixels of halo were actually added on each side.
            // At image boundaries the halo is clipped, so the effective halo
            // is zero on that edge — no blending needed there.
            let halo_left = tx - src_x0; // 0 at image left boundary
            let halo_top = ty - src_y0;  // 0 at image top boundary
            let halo_right = if src_x1 < (tx + step + overlap) {
                0 // right boundary clipped
            } else {
                src_x1.saturating_sub(tx + step)
            };
            let halo_bottom = if src_y1 < (ty + step + overlap) {
                0 // bottom boundary clipped
            } else {
                src_y1.saturating_sub(ty + step)
            };

            let tw = src_x1 - src_x0;
            let th = src_y1 - src_y0;

            // Extract tile from input (row-major copy).
            let mut tile = vec![0u8; tw * th * channels];
            for row in 0..th {
                let src_row = src_y0 + row;
                let src_start = (src_row * width + src_x0) * channels;
                let dst_start = row * tw * channels;
                tile[dst_start..dst_start + tw * channels]
                    .copy_from_slice(&input[src_start..src_start + tw * channels]);
            }

            // Run SR on the tile.
            let upscaled = sr_fn(&tile, tw, th, channels)?;

            let out_tw = tw * scale;
            let out_th = th * scale;
            let out_halo_left = halo_left * scale;
            let out_halo_top = halo_top * scale;
            let out_halo_right = halo_right * scale;
            let out_halo_bottom = halo_bottom * scale;

            // Verify SR function returned the expected size.
            let expected_upscaled = out_tw * out_th * channels;
            if upscaled.len() < expected_upscaled {
                return Err(CvError::insufficient_data(expected_upscaled, upscaled.len()));
            }

            let out_x0 = src_x0 * scale;
            let out_y0 = src_y0 * scale;

            // Accumulate into output buffer with feather weights.
            // Each pixel's weight is determined by its distance from the
            // *actual* halo edges (zero where no halo → max weight 1.0).
            for oy in 0..out_th {
                for ox in 0..out_tw {
                    let wx = feather_weight_asymmetric(
                        ox,
                        out_tw,
                        out_halo_left,
                        out_halo_right,
                    );
                    let wy = feather_weight_asymmetric(
                        oy,
                        out_th,
                        out_halo_top,
                        out_halo_bottom,
                    );
                    let w = wx * wy;

                    let src_idx = (oy * out_tw + ox) * channels;
                    let dst_x = out_x0 + ox;
                    let dst_y = out_y0 + oy;

                    if dst_x < out_w && dst_y < out_h {
                        let dst_idx = (dst_y * out_w + dst_x) * channels;
                        for c in 0..channels {
                            accum[dst_idx + c] += upscaled[src_idx + c] as f32 * w;
                            weight[dst_idx + c] += w;
                        }
                    }
                }
            }

            tx += step;
        }
        ty += step;
    }

    // Normalize accumulated values by their blend weights.
    let result: Vec<u8> = accum
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| {
            if *w > 0.0 {
                (v / w).clamp(0.0, 255.0) as u8
            } else {
                0
            }
        })
        .collect();

    Ok(result)
}

/// Linear feather weight for a pixel at `pos` within a buffer of `size` pixels.
///
/// `halo_start` and `halo_end` are the actual number of halo pixels on the
/// left/top and right/bottom edges respectively.  When a halo is zero (image
/// boundary), that edge ramps to `1.0` immediately (no blending needed).
///
/// Returns `1.0` in the tile interior; ramps from `0.0` → `1.0` across
/// each halo zone so adjacent tiles blend smoothly.
fn feather_weight_asymmetric(pos: usize, size: usize, halo_start: usize, halo_end: usize) -> f32 {
    let w_start = if halo_start == 0 {
        1.0_f32
    } else {
        (pos.min(halo_start) as f32) / (halo_start as f32)
    };
    let dist_from_end = size.saturating_sub(1).saturating_sub(pos);
    let w_end = if halo_end == 0 {
        1.0_f32
    } else {
        (dist_from_end.min(halo_end) as f32) / (halo_end as f32)
    };
    w_start.min(w_end).clamp(0.0, 1.0)
}

/// Symmetric feather weight (used only in tests).
#[cfg(test)]
fn feather_weight(pos: usize, size: usize, overlap: usize) -> f32 {
    feather_weight_asymmetric(pos, size, overlap, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_sr_dimensions() {
        let input = vec![128u8; 64 * 64 * 3];
        let config = TiledSrConfig {
            tile_size: 32,
            overlap: 4,
            scale: 2,
        };
        let result = process_tiled(&input, 64, 64, 3, &config, |_tile, w, h, c| {
            // Identity upscale: fill with 128
            Ok(vec![128u8; w * 2 * h * 2 * c])
        })
        .expect("process_tiled must succeed");
        assert_eq!(result.len(), 128 * 128 * 3);
    }

    #[test]
    fn test_tiled_sr_matches_whole_within_tolerance() {
        // Solid-color image: tiled SR should produce the same result as whole-image SR.
        let input = vec![200u8; 32 * 32 * 3];
        let config = TiledSrConfig {
            tile_size: 16,
            overlap: 2,
            scale: 2,
        };
        let tiled = process_tiled(&input, 32, 32, 3, &config, |_tile, w, h, c| {
            Ok(vec![200u8; w * 2 * h * 2 * c])
        })
        .expect("process_tiled must succeed");
        assert_eq!(tiled.len(), 64 * 64 * 3);
        for &px in &tiled {
            assert!(
                (px as i32 - 200).abs() <= 5,
                "pixel {px} too far from 200 (solid-color tolerance check)"
            );
        }
    }

    #[test]
    fn test_tiled_config_validate_zero_tile_size() {
        let cfg = TiledSrConfig {
            tile_size: 0,
            overlap: 0,
            scale: 2,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_tiled_config_validate_overlap_ge_tile() {
        let cfg = TiledSrConfig {
            tile_size: 16,
            overlap: 16,
            scale: 2,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_tiled_config_validate_zero_scale() {
        let cfg = TiledSrConfig {
            tile_size: 32,
            overlap: 4,
            scale: 0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_feather_weight_interior() {
        // Interior pixels should have weight 1.0.
        let w = feather_weight(8, 16, 2);
        assert!((w - 1.0).abs() < 1e-6, "interior weight should be 1.0, got {w}");
    }

    #[test]
    fn test_feather_weight_edge() {
        // At position 0 with overlap 4, weight is 0/4 = 0.0.
        let w = feather_weight(0, 16, 4);
        assert!(w < 0.01, "edge weight should be near 0.0, got {w}");
    }

    #[test]
    fn test_feather_weight_no_overlap() {
        // With overlap=0, always returns 1.0.
        for pos in 0..10 {
            assert_eq!(feather_weight(pos, 10, 0), 1.0);
        }
    }

    #[test]
    fn test_tiled_single_tile() {
        // When image fits in one tile, result should be the direct upscale.
        let input = vec![100u8; 8 * 8 * 3];
        let config = TiledSrConfig {
            tile_size: 16, // larger than image — single tile
            overlap: 2,
            scale: 2,
        };
        let result = process_tiled(&input, 8, 8, 3, &config, |_tile, w, h, c| {
            Ok(vec![100u8; w * 2 * h * 2 * c])
        })
        .expect("single-tile must succeed");
        assert_eq!(result.len(), 16 * 16 * 3);
        for &px in &result {
            assert_eq!(px, 100, "single-tile result must equal the SR output exactly");
        }
    }
}
