//! DCT block artifact detection.
//!
//! JPEG and other block-DCT codecs introduce visible discontinuities at 8-pixel
//! block boundaries.  This module measures those discontinuities relative to
//! interior pixel differences to quantify blocking severity and estimate the
//! original JPEG quality setting.

#![allow(dead_code)]

/// Overall metrics for the blocking artifact analysis.
#[derive(Debug, Clone)]
pub struct BlockingMetrics {
    /// Mean absolute difference at 8-pixel block boundaries minus mean interior
    /// difference, normalised to [0, 1].  Higher → more blocking.
    pub block_effect_strength: f32,
    /// Composite blockiness score [0, 1].
    pub blockiness_score: f32,
    /// Estimated original JPEG quality (1-100).
    pub compression_ratio_est: f32,
}

/// Per-block and region-level blocking report.
#[derive(Debug, Clone)]
pub struct BlockingReport {
    /// Blockiness score for each 8×8 block (row-major, one entry per block).
    pub per_block_scores: Vec<f32>,
    /// (block_col, block_row) coordinates of the N worst blocks.
    pub worst_regions: Vec<(usize, usize)>,
    /// Overall metrics.
    pub metrics: BlockingMetrics,
}

const BLOCK: usize = 8;

/// Detect blocking artifacts in a luma plane.
///
/// # Arguments
///
/// * `luma`   - Luma values in [0, 1], row-major, `width * height` elements.
/// * `width`  - Image width in pixels.
/// * `height` - Image height in pixels.
///
/// # Returns
///
/// [`BlockingMetrics`] summarising the severity and estimated quality.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_blocking(luma: &[f32], width: usize, height: usize) -> BlockingMetrics {
    if luma.len() < width * height || width < BLOCK || height < BLOCK {
        return BlockingMetrics {
            block_effect_strength: 0.0,
            blockiness_score: 0.0,
            compression_ratio_est: 95.0,
        };
    }

    // --- boundary differences (at multiples of BLOCK) ---
    let mut boundary_sum = 0.0f64;
    let mut boundary_count = 0usize;

    // Horizontal boundaries (pixel differences at y = k*BLOCK)
    for y in (BLOCK..height).step_by(BLOCK) {
        for x in 0..width {
            let diff = (luma[y * width + x] - luma[(y - 1) * width + x]).abs() as f64;
            boundary_sum += diff;
            boundary_count += 1;
        }
    }
    // Vertical boundaries (pixel differences at x = k*BLOCK)
    for y in 0..height {
        for x in (BLOCK..width).step_by(BLOCK) {
            let diff = (luma[y * width + x] - luma[y * width + x - 1]).abs() as f64;
            boundary_sum += diff;
            boundary_count += 1;
        }
    }
    let mean_boundary = if boundary_count > 0 {
        boundary_sum / boundary_count as f64
    } else {
        0.0
    };

    // --- interior differences (not at block boundaries) ---
    let mut interior_sum = 0.0f64;
    let mut interior_count = 0usize;

    for y in 1..height {
        for x in 1..width {
            // Skip block boundary pixels
            if y % BLOCK == 0 || x % BLOCK == 0 {
                continue;
            }
            let dh = (luma[y * width + x] - luma[y * width + x - 1]).abs() as f64;
            let dv = (luma[y * width + x] - luma[(y - 1) * width + x]).abs() as f64;
            interior_sum += dh + dv;
            interior_count += 2;
        }
    }
    let mean_interior = if interior_count > 0 {
        interior_sum / interior_count as f64
    } else {
        0.0
    };

    // block_effect_strength: how much larger boundary differences are vs interior
    let block_effect_strength = if mean_interior > 1e-9 {
        ((mean_boundary / mean_interior) - 1.0).max(0.0).min(10.0) as f32 / 10.0
    } else if mean_boundary > 0.0 {
        1.0
    } else {
        0.0
    };

    let blockiness_score = block_effect_strength;
    let compression_ratio_est = estimate_quality(blockiness_score);

    BlockingMetrics {
        block_effect_strength,
        blockiness_score,
        compression_ratio_est,
    }
}

/// Full blocking analysis: per-block scores and worst-region identification.
///
/// # Arguments
///
/// * `luma`   - Luma values in [0, 1], row-major.
/// * `width`  - Image width.
/// * `height` - Image height.
/// * `top_n`  - How many worst blocks to return in [`BlockingReport::worst_regions`].
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn analyze_blocking(luma: &[f32], width: usize, height: usize, top_n: usize) -> BlockingReport {
    let metrics = detect_blocking(luma, width, height);

    let blocks_x = width / BLOCK;
    let blocks_y = height / BLOCK;
    let num_blocks = blocks_x * blocks_y;

    let mut per_block_scores = Vec::with_capacity(num_blocks);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let score = block_score(luma, width, height, bx, by);
            per_block_scores.push(score);
        }
    }

    // Find top_n worst blocks
    let mut indexed: Vec<(usize, f32)> = per_block_scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let worst_regions: Vec<(usize, usize)> = indexed
        .iter()
        .take(top_n)
        .map(|(idx, _)| (idx % blocks_x, idx / blocks_x))
        .collect();

    BlockingReport {
        per_block_scores,
        worst_regions,
        metrics,
    }
}

/// Compute a blockiness score for a single 8×8 block at (bx, by).
#[allow(clippy::cast_precision_loss)]
fn block_score(luma: &[f32], width: usize, height: usize, bx: usize, by: usize) -> f32 {
    let x0 = bx * BLOCK;
    let y0 = by * BLOCK;

    let mut boundary_sum = 0.0f64;
    let mut boundary_count = 0usize;
    let mut interior_sum = 0.0f64;
    let mut interior_count = 0usize;

    // Within the block, compare horizontal neighbors
    for dy in 0..BLOCK {
        for dx in 0..BLOCK {
            let y = y0 + dy;
            let x = x0 + dx;
            if y >= height || x >= width {
                continue;
            }

            // Right neighbor
            if x + 1 < width {
                let diff = (luma[y * width + x] - luma[y * width + x + 1]).abs() as f64;
                // Boundary if dx == BLOCK - 1 (right edge of block)
                if dx == BLOCK - 1 {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
            // Bottom neighbor
            if y + 1 < height {
                let diff = (luma[y * width + x] - luma[(y + 1) * width + x]).abs() as f64;
                if dy == BLOCK - 1 {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
        }
    }

    let mean_b = if boundary_count > 0 {
        boundary_sum / boundary_count as f64
    } else {
        0.0
    };
    let mean_i = if interior_count > 0 {
        interior_sum / interior_count as f64
    } else {
        0.0
    };

    if mean_i > 1e-9 {
        ((mean_b / mean_i) - 1.0).max(0.0).min(10.0) as f32 / 10.0
    } else if mean_b > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Estimate JPEG quality from a blockiness score [0, 1].
///
/// A score of 0 suggests near-lossless (quality ~95); a score of 1 suggests
/// heavy compression (quality ~5).
fn estimate_quality(blockiness: f32) -> f32 {
    // Linear inverse mapping: quality = 95 - blockiness * 90
    (95.0 - blockiness * 90.0).clamp(5.0, 100.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_luma(w: usize, h: usize, val: f32) -> Vec<f32> {
        vec![val; w * h]
    }

    /// Create a luma plane with sharp 8-pixel block boundaries (simulated blocking).
    fn blocked_luma(w: usize, h: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // Alternate 0.3 / 0.7 per block
                let block_val = if (x / BLOCK + y / BLOCK) % 2 == 0 {
                    0.3
                } else {
                    0.7
                };
                v[y * w + x] = block_val;
            }
        }
        v
    }

    #[test]
    fn test_detect_blocking_uniform() {
        let luma = uniform_luma(64, 64, 0.5);
        let metrics = detect_blocking(&luma, 64, 64);
        // Uniform image has no edges at all → blockiness should be ~0
        assert!(metrics.block_effect_strength < 0.01);
    }

    #[test]
    fn test_detect_blocking_blocked_image() {
        let luma = blocked_luma(64, 64);
        let metrics = detect_blocking(&luma, 64, 64);
        // Block boundaries are prominent → expect some nonzero strength
        assert!(metrics.block_effect_strength > 0.0);
    }

    #[test]
    fn test_blockiness_score_range() {
        let luma = blocked_luma(64, 64);
        let metrics = detect_blocking(&luma, 64, 64);
        assert!(metrics.blockiness_score >= 0.0 && metrics.blockiness_score <= 1.0);
    }

    #[test]
    fn test_compression_ratio_range() {
        let luma = blocked_luma(64, 64);
        let metrics = detect_blocking(&luma, 64, 64);
        assert!(metrics.compression_ratio_est >= 5.0 && metrics.compression_ratio_est <= 100.0);
    }

    #[test]
    fn test_analyze_blocking_report_structure() {
        let luma = blocked_luma(64, 64);
        let report = analyze_blocking(&luma, 64, 64, 3);
        let expected_blocks = (64 / BLOCK) * (64 / BLOCK);
        assert_eq!(report.per_block_scores.len(), expected_blocks);
        assert!(report.worst_regions.len() <= 3);
    }

    #[test]
    fn test_analyze_blocking_per_block_scores_range() {
        let luma = blocked_luma(64, 64);
        let report = analyze_blocking(&luma, 64, 64, 5);
        for &s in &report.per_block_scores {
            assert!(s >= 0.0 && s <= 1.0, "Per-block score out of range: {s}");
        }
    }

    #[test]
    fn test_too_small_image() {
        let luma = uniform_luma(4, 4, 0.5);
        let metrics = detect_blocking(&luma, 4, 4);
        // Should return defaults without panicking
        assert_eq!(metrics.block_effect_strength, 0.0);
    }

    #[test]
    fn test_estimate_quality_extremes() {
        assert!((estimate_quality(0.0) - 95.0).abs() < 0.1);
        assert!((estimate_quality(1.0) - 5.0).abs() < 0.1);
    }
}
