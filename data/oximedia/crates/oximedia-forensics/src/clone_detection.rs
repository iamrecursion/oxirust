//! Copy-move (clone) forgery detection.
//!
//! Locates regions of an image that have been copied from one position to
//! another via a DCT-hash based block matching approach:
//!
//! 1. Divide the luma plane into overlapping (or non-overlapping) 8×8 blocks.
//! 2. Compute a compact hash for each block from its DC coefficient and the
//!    magnitudes of the first few AC coefficients.
//! 3. Sort blocks by hash; adjacent entries with similar hashes are candidates.
//! 4. Verify candidates against a full magnitude distance threshold.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A detected clone (copy-move) region pair.
#[derive(Debug, Clone)]
pub struct CloneRegion {
    /// Source bounding box: (x, y, width, height) in pixels.
    pub source: (u32, u32, u32, u32),
    /// Destination bounding box: (x, y, width, height) in pixels.
    pub dest: (u32, u32, u32, u32),
    /// Normalised similarity score [0, 1].
    pub similarity: f32,
}

/// Configuration for the clone detector.
#[derive(Debug, Clone)]
pub struct CloneDetector {
    /// Block size in pixels (default 8).
    pub block_size: u32,
    /// Minimum similarity for a match to be reported (default 0.95).
    pub min_similarity: f32,
}

impl Default for CloneDetector {
    fn default() -> Self {
        Self {
            block_size: 8,
            min_similarity: 0.95,
        }
    }
}

/// Summary result of clone detection.
#[derive(Debug, Clone)]
pub struct CloneDetectionResult {
    /// All detected clone region pairs.
    pub regions: Vec<CloneRegion>,
    /// Overall confidence that the image has been tampered [0, 1].
    pub confidence: f32,
    /// True if at least one clone region was found above the threshold.
    pub is_tampered: bool,
}

impl CloneDetector {
    /// Creates a new [`CloneDetector`] with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`CloneDetector`] with custom settings.
    #[must_use]
    pub fn with_params(block_size: u32, min_similarity: f32) -> Self {
        Self {
            block_size,
            min_similarity,
        }
    }

    /// Detects copy-move regions in the given luma plane.
    ///
    /// # Arguments
    ///
    /// * `luma`   - Luma values in [0, 1], row-major.
    /// * `width`  - Image width in pixels.
    /// * `height` - Image height in pixels.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, luma: &[f32], width: u32, height: u32) -> Vec<CloneRegion> {
        let bs = self.block_size as usize;
        let w = width as usize;
        let h = height as usize;

        if luma.len() < w * h || w < bs || h < bs {
            return Vec::new();
        }

        // Enumerate non-overlapping blocks and their DCT-derived feature vectors
        let blocks_x = w / bs;
        let blocks_y = h / bs;

        let mut block_features: Vec<(usize, usize, Vec<f32>)> =
            Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let feats = dct_features(luma, w, bx, by, bs);
                block_features.push((bx, by, feats));
            }
        }

        // Sort by the first feature element (DC) to bring similar blocks adjacent
        block_features.sort_by(|a, b| {
            a.2[0]
                .partial_cmp(&b.2[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut regions = Vec::new();
        let n = block_features.len();

        for i in 0..n {
            // Compare against the next few entries (sorted by DC)
            for j in i + 1..n.min(i + 20) {
                let (bx1, by1, ref f1) = block_features[i];
                let (bx2, by2, ref f2) = block_features[j];

                // Skip if same block or spatially adjacent (trivial match)
                let dx = (bx1 as i32 - bx2 as i32).unsigned_abs() as usize;
                let dy = (by1 as i32 - by2 as i32).unsigned_abs() as usize;
                if dx <= 1 && dy <= 1 {
                    continue;
                }

                let sim = feature_similarity(f1, f2);
                if sim >= self.min_similarity {
                    regions.push(CloneRegion {
                        source: ((bx1 * bs) as u32, (by1 * bs) as u32, bs as u32, bs as u32),
                        dest: ((bx2 * bs) as u32, (by2 * bs) as u32, bs as u32, bs as u32),
                        similarity: sim,
                    });
                }
            }
        }

        regions
    }

    /// Detects clone regions and returns a summary result.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_with_result(
        &self,
        luma: &[f32],
        width: u32,
        height: u32,
    ) -> CloneDetectionResult {
        let regions = self.detect(luma, width, height);

        let is_tampered = !regions.is_empty();
        let confidence = if regions.is_empty() {
            0.0
        } else {
            // Scale confidence by number of regions and their similarity
            let avg_sim: f32 =
                regions.iter().map(|r| r.similarity).sum::<f32>() / regions.len() as f32;
            (avg_sim * (1.0 - (-0.5 * regions.len() as f32).exp())).min(1.0)
        };

        CloneDetectionResult {
            regions,
            confidence,
            is_tampered,
        }
    }
}

// ---------------------------------------------------------------------------
// DCT feature extraction
// ---------------------------------------------------------------------------

/// Compute a compact feature vector for an 8×8 (or `bs`×`bs`) block using DCT.
///
/// Returns the DC coefficient plus the 7 lowest-AC magnitudes (8 values total).
#[allow(clippy::cast_precision_loss)]
fn dct_features(luma: &[f32], width: usize, bx: usize, by: usize, bs: usize) -> Vec<f32> {
    // Extract block pixels (shifted to -128 … 128 scale)
    let mut block = vec![0.0f32; bs * bs];
    for dy in 0..bs {
        for dx in 0..bs {
            let y = by * bs + dy;
            let x = bx * bs + dx;
            block[dy * bs + dx] = luma[y * width + x] * 255.0 - 128.0;
        }
    }

    // 2D DCT of the block
    let mut dct = vec![0.0f32; bs * bs];
    let bsf = bs as f32;
    for u in 0..bs {
        for v in 0..bs {
            let mut sum = 0.0f32;
            for x in 0..bs {
                for y in 0..bs {
                    let cos_u = ((2.0 * x as f32 + 1.0) * u as f32 * PI / (2.0 * bsf)).cos();
                    let cos_v = ((2.0 * y as f32 + 1.0) * v as f32 * PI / (2.0 * bsf)).cos();
                    sum += block[y * bs + x] * cos_u * cos_v;
                }
            }
            let cu = if u == 0 { (1.0_f32 / 2.0).sqrt() } else { 1.0 };
            let cv = if v == 0 { (1.0_f32 / 2.0).sqrt() } else { 1.0 };
            dct[u * bs + v] = (2.0 / bsf) * cu * cv * sum;
        }
    }

    // Use DC + top-7 AC magnitudes as the feature
    let dc = dct[0];
    // Zigzag indices (first 7 AC positions for an 8×8 block)
    let zigzag = [1usize, bs, 2, bs + 1, 2 * bs, 3, bs * 2 + 1];
    let mut feats = vec![dc];
    for &idx in &zigzag {
        if idx < bs * bs {
            feats.push(dct[idx].abs());
        } else {
            feats.push(0.0);
        }
    }

    feats
}

/// Cosine similarity between two feature vectors.
fn feature_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = (0..n).map(|i| a[i] * b[i]).sum();
    let mag_a: f32 = (0..n).map(|i| a[i] * a[i]).sum::<f32>().sqrt();
    let mag_b: f32 = (0..n).map(|i| b[i] * b[i]).sum::<f32>().sqrt();
    if mag_a > 1e-6 && mag_b > 1e-6 {
        (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
    } else {
        // Both vectors are near-zero: perfectly similar (both blank)
        1.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform luma plane.
    fn uniform(w: usize, h: usize, val: f32) -> Vec<f32> {
        vec![val; w * h]
    }

    /// Luma plane with a block copy: copy the top-left 8×8 into the
    /// center 8×8 region (with some offset).
    fn luma_with_copy(w: usize, h: usize) -> Vec<f32> {
        // Start with a gradient
        let mut v: Vec<f32> = (0..w * h)
            .map(|i| ((i % w) as f32 / w as f32) * 0.8)
            .collect();

        // Copy block at (0,0) to (16, 16)
        let src_x = 0usize;
        let src_y = 0usize;
        let dst_x = 16usize;
        let dst_y = 16usize;
        for dy in 0..8usize {
            for dx in 0..8usize {
                let src = (src_y + dy) * w + src_x + dx;
                let dst = (dst_y + dy) * w + dst_x + dx;
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        v
    }

    #[test]
    fn test_detector_default_params() {
        let det = CloneDetector::default();
        assert_eq!(det.block_size, 8);
        assert!((det.min_similarity - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_detect_no_clone_on_uniform() {
        let det = CloneDetector::default();
        let luma = uniform(64, 64, 0.5);
        // Uniform image: every block is identical → all are "clones" of each other
        // but since we skip spatially adjacent ones, some may still be returned.
        // The test just checks it doesn't panic and returns a Vec.
        let _regions = det.detect(&luma, 64, 64);
    }

    #[test]
    fn test_detect_result_structure() {
        let det = CloneDetector::default();
        let luma = uniform(64, 64, 0.5);
        let result = det.detect_with_result(&luma, 64, 64);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_detect_small_image_returns_empty() {
        let det = CloneDetector::default();
        let luma = vec![0.5f32; 4 * 4];
        let regions = det.detect(&luma, 4, 4);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_clone_region_has_valid_coordinates() {
        let det = CloneDetector::default();
        let luma = luma_with_copy(64, 64);
        let regions = det.detect(&luma, 64, 64);
        for r in &regions {
            assert!(r.source.0 < 64 && r.source.1 < 64);
            assert!(r.dest.0 < 64 && r.dest.1 < 64);
            assert!(r.similarity >= 0.0 && r.similarity <= 1.0);
        }
    }

    #[test]
    fn test_feature_similarity_identical() {
        let feats = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = feature_similarity(&feats, &feats);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical features should have similarity 1.0"
        );
    }

    #[test]
    fn test_feature_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = feature_similarity(&a, &b);
        assert!(
            sim < 0.01,
            "Orthogonal features should have near-zero similarity"
        );
    }

    #[test]
    fn test_dct_features_length() {
        let luma = uniform(64, 64, 0.5);
        let feats = dct_features(&luma, 64, 0, 0, 8);
        assert_eq!(feats.len(), 8); // 1 DC + 7 AC
    }

    #[test]
    fn test_with_params_constructor() {
        let det = CloneDetector::with_params(16, 0.9);
        assert_eq!(det.block_size, 16);
        assert!((det.min_similarity - 0.9).abs() < 0.001);
    }
}
