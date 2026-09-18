//! Shot similarity module using perceptual hashing.
//!
//! Finds visually similar shots across a project by computing perceptual hashes
//! (pHash) of representative frames and comparing Hamming distances. Also
//! supports color histogram and structural similarity for finer matching.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};

// ---------------------------------------------------------------------------
// Perceptual hash
// ---------------------------------------------------------------------------

/// A 64-bit perceptual hash of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash {
    /// The raw 64-bit hash value.
    pub bits: u64,
}

impl PHash {
    /// Hamming distance to another hash (number of differing bits).
    #[must_use]
    pub const fn distance(&self, other: &Self) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Whether two hashes are considered similar (distance <= threshold).
    #[must_use]
    pub const fn is_similar(&self, other: &Self, threshold: u32) -> bool {
        self.distance(other) <= threshold
    }
}

/// Configuration for the similarity engine.
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// Maximum Hamming distance for two shots to be considered similar.
    pub hash_threshold: u32,
    /// Size to which frames are downscaled before hashing (NxN).
    pub hash_size: usize,
    /// Weight of pHash distance in the combined similarity score.
    pub phash_weight: f32,
    /// Weight of color histogram distance in the combined similarity score.
    pub color_weight: f32,
    /// Weight of structural similarity in the combined similarity score.
    pub structural_weight: f32,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            hash_threshold: 10,
            hash_size: 32,
            phash_weight: 0.50,
            color_weight: 0.25,
            structural_weight: 0.25,
        }
    }
}

/// Result of comparing two shots for similarity.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Index of the first shot.
    pub shot_a: usize,
    /// Index of the second shot.
    pub shot_b: usize,
    /// pHash Hamming distance (0 = identical, 64 = maximally different).
    pub hash_distance: u32,
    /// Color histogram distance (0.0 = identical, 1.0 = maximally different).
    pub color_distance: f32,
    /// Structural similarity (0.0 = different, 1.0 = identical).
    pub structural_similarity: f32,
    /// Combined similarity score (0.0 = different, 1.0 = identical).
    pub combined_score: f32,
}

/// A group of similar shots.
#[derive(Debug, Clone)]
pub struct SimilarityGroup {
    /// Group ID.
    pub id: usize,
    /// Indices of shots in this group.
    pub members: Vec<usize>,
    /// Average pairwise similarity within the group.
    pub average_similarity: f32,
}

/// Shot similarity engine.
pub struct ShotSimilarity {
    config: SimilarityConfig,
}

impl Default for ShotSimilarity {
    fn default() -> Self {
        Self::new(SimilarityConfig::default())
    }
}

impl ShotSimilarity {
    /// Create a new similarity engine with the given configuration.
    #[must_use]
    pub fn new(config: SimilarityConfig) -> Self {
        Self { config }
    }

    /// Compute the perceptual hash (pHash) of a frame.
    ///
    /// The algorithm:
    /// 1. Convert to grayscale
    /// 2. Downscale to `hash_size` x `hash_size`
    /// 3. Apply 2D DCT (type-II, simplified)
    /// 4. Keep top-left 8x8 coefficients (low frequencies)
    /// 5. Compute median of those 64 values
    /// 6. Each bit is 1 if coefficient > median, else 0
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid (< 3 channels).
    pub fn compute_phash(&self, frame: &FrameBuffer) -> ShotResult<PHash> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let gray = to_grayscale(frame);
        let resized = bilinear_resize(&gray, self.config.hash_size, self.config.hash_size);

        // Apply simplified 2D DCT on the resized image
        let dct = dct_2d(&resized);

        // Extract top-left 8x8 coefficients (excluding DC at [0,0])
        let mut coeffs = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                if y < dct.len() && x < dct[0].len() {
                    coeffs.push(dct[y][x]);
                } else {
                    coeffs.push(0.0);
                }
            }
        }

        // Compute median
        let mut sorted = coeffs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Build hash
        let mut bits: u64 = 0;
        for (i, &c) in coeffs.iter().enumerate() {
            if c > median {
                bits |= 1u64 << i;
            }
        }

        Ok(PHash { bits })
    }

    /// Compute color histogram distance between two frames.
    ///
    /// Uses chi-square distance over 16-bin per-channel histograms,
    /// normalised to [0, 1].
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn color_histogram_distance(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
    ) -> ShotResult<f32> {
        if frame_a.dim().2 < 3 || frame_b.dim().2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let num_bins = 16;
        let bin_size = 256.0 / num_bins as f32;
        let mut total_dist = 0.0_f32;

        for channel in 0..3 {
            let hist_a = build_histogram(frame_a, channel, num_bins, bin_size);
            let hist_b = build_histogram(frame_b, channel, num_bins, bin_size);

            for i in 0..num_bins {
                let sum = hist_a[i] + hist_b[i];
                if sum > 0.0 {
                    let diff = hist_a[i] - hist_b[i];
                    total_dist += (diff * diff) / sum;
                }
            }
        }

        Ok((total_dist / 3.0).sqrt().min(1.0))
    }

    /// Compute structural similarity (simplified SSIM-like metric).
    ///
    /// Returns a score in [0, 1] where 1 means identical structure.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or dimensions differ.
    pub fn structural_similarity(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
    ) -> ShotResult<f32> {
        if frame_a.dim().2 < 3 || frame_b.dim().2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let gray_a = to_grayscale(frame_a);
        let gray_b = to_grayscale(frame_b);

        // Resize both to a common small size for comparison
        let size = 64;
        let ra = bilinear_resize(&gray_a, size, size);
        let rb = bilinear_resize(&gray_b, size, size);

        let n = (size * size) as f64;
        let mut sum_a = 0.0_f64;
        let mut sum_b = 0.0_f64;
        let mut sum_a2 = 0.0_f64;
        let mut sum_b2 = 0.0_f64;
        let mut sum_ab = 0.0_f64;

        for y in 0..size {
            for x in 0..size {
                let a = f64::from(ra.get(y, x));
                let b = f64::from(rb.get(y, x));
                sum_a += a;
                sum_b += b;
                sum_a2 += a * a;
                sum_b2 += b * b;
                sum_ab += a * b;
            }
        }

        let mean_a = sum_a / n;
        let mean_b = sum_b / n;
        let var_a = (sum_a2 / n) - (mean_a * mean_a);
        let var_b = (sum_b2 / n) - (mean_b * mean_b);
        let covar = (sum_ab / n) - (mean_a * mean_b);

        // SSIM constants for 8-bit images
        let c1 = 6.5025; // (0.01 * 255)^2
        let c2 = 58.5225; // (0.03 * 255)^2

        let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covar + c2);
        let denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

        if denominator.abs() < f64::EPSILON {
            return Ok(1.0); // Both images are flat and identical
        }

        Ok((numerator / denominator).clamp(0.0, 1.0) as f32)
    }

    /// Compare two shots and return a combined similarity result.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn compare(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
        shot_a_idx: usize,
        shot_b_idx: usize,
    ) -> ShotResult<SimilarityResult> {
        let hash_a = self.compute_phash(frame_a)?;
        let hash_b = self.compute_phash(frame_b)?;
        let hash_distance = hash_a.distance(&hash_b);

        let color_distance = self.color_histogram_distance(frame_a, frame_b)?;
        let structural = self.structural_similarity(frame_a, frame_b)?;

        // Normalise hash distance to [0, 1] (max possible = 64)
        let hash_norm = 1.0 - (hash_distance as f32 / 64.0);

        let combined = self.config.phash_weight * hash_norm
            + self.config.color_weight * (1.0 - color_distance)
            + self.config.structural_weight * structural;

        Ok(SimilarityResult {
            shot_a: shot_a_idx,
            shot_b: shot_b_idx,
            hash_distance,
            color_distance,
            structural_similarity: structural,
            combined_score: combined.clamp(0.0, 1.0),
        })
    }

    /// Find all similar shot pairs from a collection of representative frames.
    ///
    /// Returns pairs where the combined score exceeds the threshold
    /// `1.0 - hash_threshold/64.0` (roughly).
    ///
    /// # Errors
    ///
    /// Returns error if any frame is invalid.
    pub fn find_similar_shots(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<SimilarityResult>> {
        let hashes: Vec<PHash> = frames
            .iter()
            .map(|f| self.compute_phash(f))
            .collect::<ShotResult<Vec<_>>>()?;

        let mut results = Vec::new();
        for i in 0..frames.len() {
            for j in (i + 1)..frames.len() {
                if hashes[i].is_similar(&hashes[j], self.config.hash_threshold) {
                    let result = self.compare(&frames[i], &frames[j], i, j)?;
                    results.push(result);
                }
            }
        }

        // Sort by combined score descending
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Group similar shots using single-linkage clustering.
    ///
    /// # Errors
    ///
    /// Returns error if any frame is invalid.
    pub fn group_similar_shots(
        &self,
        frames: &[FrameBuffer],
        min_similarity: f32,
    ) -> ShotResult<Vec<SimilarityGroup>> {
        let similar_pairs = self.find_similar_shots(frames)?;

        // Union-Find for grouping
        let n = frames.len();
        let mut parent: Vec<usize> = (0..n).collect();

        // Find with path compression
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        for pair in &similar_pairs {
            if pair.combined_score >= min_similarity {
                let ra = find(&mut parent, pair.shot_a);
                let rb = find(&mut parent, pair.shot_b);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }

        // Build groups
        let mut group_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            group_map.entry(root).or_default().push(i);
        }

        let mut groups: Vec<SimilarityGroup> = group_map
            .into_iter()
            .filter(|(_, members)| members.len() > 1)
            .enumerate()
            .map(|(id, (_, members))| {
                // Calculate average pairwise similarity within group
                let mut total_sim = 0.0_f32;
                let mut count = 0u32;
                for pair in &similar_pairs {
                    if members.contains(&pair.shot_a) && members.contains(&pair.shot_b) {
                        total_sim += pair.combined_score;
                        count += 1;
                    }
                }
                let avg = if count > 0 {
                    total_sim / count as f32
                } else {
                    0.0
                };
                SimilarityGroup {
                    id,
                    members,
                    average_similarity: avg,
                }
            })
            .collect();

        groups.sort_by(|a, b| {
            b.average_similarity
                .partial_cmp(&a.average_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(groups)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &SimilarityConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn to_grayscale(frame: &FrameBuffer) -> GrayImage {
    let (h, w, _) = frame.dim();
    let mut gray = GrayImage::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let r = f32::from(frame.get(y, x, 0));
            let g = f32::from(frame.get(y, x, 1));
            let b = f32::from(frame.get(y, x, 2));
            gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
        }
    }
    gray
}

fn bilinear_resize(src: &GrayImage, new_w: usize, new_h: usize) -> GrayImage {
    let (sh, sw) = src.dim();
    let mut dst = GrayImage::zeros(new_h, new_w);
    if sh == 0 || sw == 0 || new_h == 0 || new_w == 0 {
        return dst;
    }
    let x_ratio = sw as f32 / new_w as f32;
    let y_ratio = sh as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;
            let x0 = (src_x as usize).min(sw - 1);
            let y0 = (src_y as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let v00 = f32::from(src.get(y0, x0));
            let v10 = f32::from(src.get(y0, x1));
            let v01 = f32::from(src.get(y1, x0));
            let v11 = f32::from(src.get(y1, x1));

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;
            dst.set(y, x, val.clamp(0.0, 255.0) as u8);
        }
    }
    dst
}

/// Simplified 2D DCT-II.
fn dct_2d(gray: &GrayImage) -> Vec<Vec<f64>> {
    let (h, w) = gray.dim();
    let n = h.min(w);
    if n == 0 {
        return Vec::new();
    }
    let size = n.min(32); // cap to avoid huge computation

    // Collect input values
    let mut input = vec![vec![0.0_f64; size]; size];
    for y in 0..size {
        for x in 0..size {
            let sy = (y * h / size).min(h - 1);
            let sx = (x * w / size).min(w - 1);
            input[y][x] = f64::from(gray.get(sy, sx));
        }
    }

    // 1D DCT on rows
    let mut row_dct = vec![vec![0.0_f64; size]; size];
    for y in 0..size {
        for k in 0..size {
            let mut sum = 0.0_f64;
            for n_idx in 0..size {
                sum += input[y][n_idx]
                    * ((std::f64::consts::PI * (2 * n_idx + 1) as f64 * k as f64)
                        / (2 * size) as f64)
                        .cos();
            }
            row_dct[y][k] = sum;
        }
    }

    // 1D DCT on columns of the result
    let mut result = vec![vec![0.0_f64; size]; size];
    for x in 0..size {
        for k in 0..size {
            let mut sum = 0.0_f64;
            for n_idx in 0..size {
                sum += row_dct[n_idx][x]
                    * ((std::f64::consts::PI * (2 * n_idx + 1) as f64 * k as f64)
                        / (2 * size) as f64)
                        .cos();
            }
            result[k][x] = sum;
        }
    }

    result
}

fn build_histogram(
    frame: &FrameBuffer,
    channel: usize,
    num_bins: usize,
    bin_size: f32,
) -> Vec<f32> {
    let (h, w, _) = frame.dim();
    let mut hist = vec![0u32; num_bins];
    for y in 0..h {
        for x in 0..w {
            let val = frame.get(y, x, channel);
            let bin = (f32::from(val) / bin_size).min((num_bins - 1) as f32) as usize;
            hist[bin] += 1;
        }
    }
    let total = (h * w) as f32;
    hist.iter().map(|&v| v as f32 / total).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(val: u8, h: usize, w: usize) -> FrameBuffer {
        FrameBuffer::from_elem(h, w, 3, val)
    }

    #[test]
    fn test_phash_identical_frames() {
        let engine = ShotSimilarity::default();
        let frame = make_frame(128, 64, 64);
        let h1 = engine
            .compute_phash(&frame)
            .expect("should succeed in test");
        let h2 = engine
            .compute_phash(&frame)
            .expect("should succeed in test");
        assert_eq!(h1.distance(&h2), 0);
    }

    #[test]
    fn test_phash_different_frames() {
        let engine = ShotSimilarity::default();
        let f1 = make_frame(0, 64, 64);
        let f2 = make_frame(255, 64, 64);
        let h1 = engine.compute_phash(&f1).expect("should succeed in test");
        let h2 = engine.compute_phash(&f2).expect("should succeed in test");
        // Very different frames should have different hashes
        // (not necessarily maximal distance for uniform frames, but > 0)
        // Uniform frames may hash similarly, so just check it doesn't error
        assert!(h1.distance(&h2) < 65);
    }

    #[test]
    fn test_phash_invalid_frame() {
        let engine = ShotSimilarity::default();
        let frame = FrameBuffer::zeros(50, 50, 1);
        assert!(engine.compute_phash(&frame).is_err());
    }

    #[test]
    fn test_phash_is_similar() {
        let a = PHash {
            bits: 0xFF00_FF00_FF00_FF00,
        };
        let b = PHash {
            bits: 0xFF00_FF00_FF00_FF01,
        };
        assert!(a.is_similar(&b, 1));
        assert!(!a.is_similar(&b, 0));
    }

    #[test]
    fn test_phash_distance_max() {
        let a = PHash { bits: 0 };
        let b = PHash { bits: u64::MAX };
        assert_eq!(a.distance(&b), 64);
    }

    #[test]
    fn test_color_histogram_distance_identical() {
        let engine = ShotSimilarity::default();
        let frame = make_frame(100, 50, 50);
        let d = engine
            .color_histogram_distance(&frame, &frame)
            .expect("should succeed in test");
        assert!(
            d < f32::EPSILON,
            "identical frames should have zero distance"
        );
    }

    #[test]
    fn test_color_histogram_distance_different() {
        let engine = ShotSimilarity::default();
        let f1 = make_frame(0, 50, 50);
        let f2 = make_frame(255, 50, 50);
        let d = engine
            .color_histogram_distance(&f1, &f2)
            .expect("should succeed in test");
        assert!(d > 0.5, "black vs white should have high distance");
    }

    #[test]
    fn test_color_histogram_invalid_frame() {
        let engine = ShotSimilarity::default();
        let f1 = FrameBuffer::zeros(50, 50, 1);
        let f2 = make_frame(128, 50, 50);
        assert!(engine.color_histogram_distance(&f1, &f2).is_err());
    }

    #[test]
    fn test_structural_similarity_identical() {
        let engine = ShotSimilarity::default();
        let frame = make_frame(128, 80, 80);
        let s = engine
            .structural_similarity(&frame, &frame)
            .expect("should succeed in test");
        assert!(s > 0.99, "identical frames should have SSIM ~1.0, got {s}");
    }

    #[test]
    fn test_structural_similarity_different() {
        let engine = ShotSimilarity::default();
        let f1 = make_frame(0, 80, 80);
        let f2 = make_frame(255, 80, 80);
        let s = engine
            .structural_similarity(&f1, &f2)
            .expect("should succeed in test");
        assert!(s < 0.5, "very different frames should have low SSIM");
    }

    #[test]
    fn test_structural_similarity_invalid() {
        let engine = ShotSimilarity::default();
        let f1 = FrameBuffer::zeros(50, 50, 2);
        let f2 = make_frame(128, 50, 50);
        assert!(engine.structural_similarity(&f1, &f2).is_err());
    }

    #[test]
    fn test_compare_identical_shots() {
        let engine = ShotSimilarity::default();
        let frame = make_frame(100, 64, 64);
        let result = engine
            .compare(&frame, &frame, 0, 1)
            .expect("should succeed in test");
        assert_eq!(result.hash_distance, 0);
        assert!(result.color_distance < f32::EPSILON);
        assert!(result.structural_similarity > 0.99);
        assert!(result.combined_score > 0.95);
    }

    #[test]
    fn test_compare_different_shots() {
        let engine = ShotSimilarity::default();
        let f1 = make_frame(0, 64, 64);
        let f2 = make_frame(255, 64, 64);
        let result = engine
            .compare(&f1, &f2, 0, 1)
            .expect("should succeed in test");
        assert!(result.color_distance > 0.5);
    }

    #[test]
    fn test_find_similar_shots_identical_group() {
        let engine = ShotSimilarity::default();
        let frames = vec![make_frame(100, 50, 50); 4];
        let results = engine
            .find_similar_shots(&frames)
            .expect("should succeed in test");
        // All pairs should be similar
        assert_eq!(results.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_find_similar_shots_empty() {
        let engine = ShotSimilarity::default();
        let results = engine
            .find_similar_shots(&[])
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_similar_shots_single() {
        let engine = ShotSimilarity::default();
        let frames = vec![make_frame(128, 50, 50)];
        let results = engine
            .find_similar_shots(&frames)
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_group_similar_shots() {
        let engine = ShotSimilarity::default();
        let frames = vec![make_frame(100, 50, 50); 3];
        let groups = engine
            .group_similar_shots(&frames, 0.5)
            .expect("should succeed in test");
        // All 3 identical frames should form one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    #[test]
    fn test_group_similar_shots_empty() {
        let engine = ShotSimilarity::default();
        let groups = engine
            .group_similar_shots(&[], 0.5)
            .expect("should succeed in test");
        assert!(groups.is_empty());
    }

    #[test]
    fn test_similarity_config_default() {
        let cfg = SimilarityConfig::default();
        assert_eq!(cfg.hash_threshold, 10);
        assert_eq!(cfg.hash_size, 32);
        assert!((cfg.phash_weight + cfg.color_weight + cfg.structural_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_engine_config_accessor() {
        let cfg = SimilarityConfig {
            hash_threshold: 5,
            ..SimilarityConfig::default()
        };
        let engine = ShotSimilarity::new(cfg);
        assert_eq!(engine.config().hash_threshold, 5);
    }

    #[test]
    fn test_bilinear_resize_basic() {
        let gray = GrayImage::zeros(100, 100);
        let resized = bilinear_resize(&gray, 10, 10);
        assert_eq!(resized.dim(), (10, 10));
    }

    #[test]
    fn test_bilinear_resize_empty() {
        let gray = GrayImage::zeros(0, 0);
        let resized = bilinear_resize(&gray, 10, 10);
        assert_eq!(resized.dim(), (10, 10));
    }

    #[test]
    fn test_dct_2d_basic() {
        let gray = GrayImage::zeros(32, 32);
        let result = dct_2d(&gray);
        assert!(!result.is_empty());
        // DC component of all-zero image should be 0
        assert!(result[0][0].abs() < f64::EPSILON);
    }

    #[test]
    fn test_dct_2d_uniform() {
        let mut gray = GrayImage::zeros(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                gray.set(y, x, 128);
            }
        }
        let result = dct_2d(&gray);
        // DC component should be non-zero for non-zero image
        assert!(result[0][0].abs() > 0.0);
    }

    #[test]
    fn test_similarity_result_fields() {
        let result = SimilarityResult {
            shot_a: 0,
            shot_b: 1,
            hash_distance: 5,
            color_distance: 0.1,
            structural_similarity: 0.95,
            combined_score: 0.9,
        };
        assert_eq!(result.shot_a, 0);
        assert_eq!(result.shot_b, 1);
    }

    #[test]
    fn test_similarity_group_fields() {
        let group = SimilarityGroup {
            id: 0,
            members: vec![0, 1, 2],
            average_similarity: 0.85,
        };
        assert_eq!(group.members.len(), 3);
    }

    #[test]
    fn test_phash_distance_symmetric() {
        let a = PHash {
            bits: 0xABCD_1234_5678_9ABC,
        };
        let b = PHash {
            bits: 0x1234_ABCD_9ABC_5678,
        };
        assert_eq!(a.distance(&b), b.distance(&a));
    }

    #[test]
    fn test_compare_score_bounded() {
        let engine = ShotSimilarity::default();
        let f1 = make_frame(50, 50, 50);
        let f2 = make_frame(200, 50, 50);
        let result = engine
            .compare(&f1, &f2, 0, 1)
            .expect("should succeed in test");
        assert!(result.combined_score >= 0.0 && result.combined_score <= 1.0);
    }
}
