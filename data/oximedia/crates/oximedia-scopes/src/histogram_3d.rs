//! 3D histogram (RGB cube) for colour distribution analysis.
//!
//! A 3D histogram discretises the RGB colour space into an N×N×N grid of bins
//! and counts how many pixels fall into each bin.  This enables efficient
//! analysis of colour distribution, dominant colour detection, neutrality
//! scoring, and 2-D projection views.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// 3D colour histogram over the RGB cube.
pub struct Histogram3D {
    /// Number of bins along each of the R, G, and B axes.
    pub bins_per_axis: usize,
    /// Flat storage: index = `r_bin + g_bin * B + b_bin * B * B`
    /// where `B = bins_per_axis`.
    pub data: Vec<u64>,
    /// Total number of pixels that have been accumulated.
    pub total_pixels: u64,
}

/// A dominant colour cluster derived from the histogram.
#[derive(Debug, Clone)]
pub struct ColorCluster {
    /// Red channel centre of the bin (0.0–255.0 scale).
    pub center_r: f32,
    /// Green channel centre of the bin.
    pub center_g: f32,
    /// Blue channel centre of the bin.
    pub center_b: f32,
    /// Number of pixels in this bin.
    pub pixel_count: u64,
    /// Fraction of the total pixel count (0.0–1.0).
    pub percentage: f32,
}

impl Histogram3D {
    /// Creates an empty 3D histogram with the given number of bins per axis.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `bins_per_axis == 0`.
    #[must_use]
    pub fn new(bins_per_axis: usize) -> Self {
        assert!(bins_per_axis > 0, "bins_per_axis must be > 0");
        let size = bins_per_axis * bins_per_axis * bins_per_axis;
        Self {
            bins_per_axis,
            data: vec![0u64; size],
            total_pixels: 0,
        }
    }

    /// Builds a 3D histogram from an interleaved RGB byte slice.
    ///
    /// `pixels` must contain at least `width * height * 3` bytes; extra bytes
    /// are ignored.  The default bin resolution is 32 bins per axis, which
    /// gives a manageable 32³ = 32 768 bucket cube.
    #[must_use]
    pub fn from_rgb_frame(pixels: &[u8], width: u32, height: u32) -> Self {
        let n_pixels = (width as usize) * (height as usize);
        let expected = n_pixels * 3;
        // Use 32 bins by default — coarse enough to be fast, fine enough to
        // distinguish colours well.
        let mut hist = Self::new(32);
        if pixels.len() < expected {
            return hist;
        }
        for i in 0..n_pixels {
            let r = pixels[i * 3];
            let g = pixels[i * 3 + 1];
            let b = pixels[i * 3 + 2];
            hist.accumulate(r, g, b);
        }
        hist
    }

    /// Accumulates a single RGB pixel into the histogram.
    pub fn accumulate(&mut self, r: u8, g: u8, b: u8) {
        let b_r = self.bin_index(r);
        let b_g = self.bin_index(g);
        let b_b = self.bin_index(b);
        let idx = b_r + b_g * self.bins_per_axis + b_b * self.bins_per_axis * self.bins_per_axis;
        self.data[idx] += 1;
        self.total_pixels += 1;
    }

    /// Returns the pixel count for the bin that contains the colour `(r, g, b)`.
    #[must_use]
    pub fn bin_at(&self, r: u8, g: u8, b: u8) -> u64 {
        let b_r = self.bin_index(r);
        let b_g = self.bin_index(g);
        let b_b = self.bin_index(b);
        let idx = b_r + b_g * self.bins_per_axis + b_b * self.bins_per_axis * self.bins_per_axis;
        self.data[idx]
    }

    /// Returns the top-`n` most populated bins as [`ColorCluster`] values.
    ///
    /// If `n` is larger than the number of non-empty bins, only non-empty bins
    /// are returned.
    #[must_use]
    pub fn dominant_colors(&self, n: usize) -> Vec<ColorCluster> {
        let b = self.bins_per_axis;
        let bin_width = 256.0_f32 / b as f32;
        let total = self.total_pixels.max(1);

        // Collect (count, flat_index) pairs, sort descending.
        let mut indexed: Vec<(u64, usize)> = self
            .data
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, c)| c > 0)
            .map(|(i, c)| (c, i))
            .collect();
        indexed.sort_unstable_by(|a, b_val| b_val.0.cmp(&a.0));

        indexed
            .into_iter()
            .take(n)
            .map(|(count, flat)| {
                let b_r = flat % b;
                let b_g = (flat / b) % b;
                let b_b = flat / (b * b);
                ColorCluster {
                    center_r: (b_r as f32 + 0.5) * bin_width,
                    center_g: (b_g as f32 + 0.5) * bin_width,
                    center_b: (b_b as f32 + 0.5) * bin_width,
                    pixel_count: count,
                    percentage: count as f32 / total as f32,
                }
            })
            .collect()
    }

    /// Fraction of pixels within `radius_bins` of the bin containing `(r,g,b)`
    /// in 3D bin-space (Euclidean distance on bin indices).
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn color_coverage(&self, r: u8, g: u8, b: u8, radius_bins: usize) -> f32 {
        let b_r = self.bin_index(r) as isize;
        let b_g = self.bin_index(g) as isize;
        let b_b = self.bin_index(b) as isize;
        let rad = radius_bins as isize;
        let bsize = self.bins_per_axis as isize;
        let total = self.total_pixels.max(1);

        let mut covered = 0u64;
        let r_min = (b_r - rad).max(0);
        let r_max = (b_r + rad).min(bsize - 1);
        let g_min = (b_g - rad).max(0);
        let g_max = (b_g + rad).min(bsize - 1);
        let b_min = (b_b - rad).max(0);
        let b_max = (b_b + rad).min(bsize - 1);

        for ri in r_min..=r_max {
            for gi in g_min..=g_max {
                for bi in b_min..=b_max {
                    let dr = ri - b_r;
                    let dg = gi - b_g;
                    let db = bi - b_b;
                    // Euclidean distance check in bin space
                    if (dr * dr + dg * dg + db * db) as usize <= radius_bins * radius_bins {
                        let idx = ri as usize
                            + gi as usize * self.bins_per_axis
                            + bi as usize * self.bins_per_axis * self.bins_per_axis;
                        covered += self.data[idx];
                    }
                }
            }
        }

        covered as f32 / total as f32
    }

    /// Computes a neutrality score in `[0.0, 1.0]`.
    ///
    /// The score is based on how close the centroid of the colour distribution
    /// is to neutral grey `(128, 128, 128)` in the 0–255 RGB cube.
    ///
    /// - `1.0` → perfectly neutral (centroid at exact grey).
    /// - `0.0` → maximally colourful (centroid at a cube corner).
    #[must_use]
    pub fn neutrality_score(&self) -> f32 {
        let b = self.bins_per_axis;
        let bin_width = 256.0_f32 / b as f32;
        let total = self.total_pixels;
        if total == 0 {
            return 1.0; // treat empty as neutral by convention
        }

        let mut sum_r = 0.0_f64;
        let mut sum_g = 0.0_f64;
        let mut sum_b = 0.0_f64;

        for (flat, &count) in self.data.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let b_r = flat % b;
            let b_g = (flat / b) % b;
            let b_bb = flat / (b * b);
            let cr = (b_r as f32 + 0.5) * bin_width;
            let cg = (b_g as f32 + 0.5) * bin_width;
            let cb = (b_bb as f32 + 0.5) * bin_width;
            sum_r += cr as f64 * count as f64;
            sum_g += cg as f64 * count as f64;
            sum_b += cb as f64 * count as f64;
        }

        let mean_r = (sum_r / total as f64) as f32;
        let mean_g = (sum_g / total as f64) as f32;
        let mean_b = (sum_b / total as f64) as f32;

        // Distance from (128, 128, 128); maximum possible distance is
        // sqrt(3) × 128 ≈ 221.7 (corner to centre of cube)
        let dr = mean_r - 128.0;
        let dg = mean_g - 128.0;
        let db = mean_b - 128.0;
        let dist = (dr * dr + dg * dg + db * db).sqrt();
        let max_dist = (3.0_f32).sqrt() * 128.0;

        1.0 - (dist / max_dist).min(1.0)
    }

    /// Collapses the 3D cube to a 2D histogram by summing along one axis.
    ///
    /// - `axis == 0` → sum over R, result is G × B grid
    /// - `axis == 1` → sum over G, result is R × B grid
    /// - `axis == 2` → sum over B, result is R × G grid
    ///
    /// Returns an outer `Vec` of size `bins_per_axis` (first remaining axis)
    /// each containing an inner `Vec` of size `bins_per_axis` (second axis).
    ///
    /// Returns an empty Vec if `axis >= 3`.
    #[must_use]
    pub fn to_2d_projection(&self, axis: usize) -> Vec<Vec<u64>> {
        if axis >= 3 {
            return Vec::new();
        }
        let b = self.bins_per_axis;
        let mut proj = vec![vec![0u64; b]; b];

        for flat in 0..self.data.len() {
            let count = self.data[flat];
            if count == 0 {
                continue;
            }
            let i_r = flat % b;
            let i_g = (flat / b) % b;
            let i_b = flat / (b * b);
            let (row, col) = match axis {
                0 => (i_g, i_b), // sum over R
                1 => (i_r, i_b), // sum over G
                _ => (i_r, i_g), // sum over B  (axis == 2)
            };
            proj[row][col] += count;
        }

        proj
    }

    // ── private helpers ──────────────────────────────────────────────────────

    #[inline]
    fn bin_index(&self, channel: u8) -> usize {
        let b = self.bins_per_axis;
        // Map 0–255 uniformly into 0..b
        ((channel as usize) * b / 256).min(b - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_creates_correct_size() {
        let h = Histogram3D::new(8);
        assert_eq!(h.data.len(), 8 * 8 * 8);
        assert_eq!(h.total_pixels, 0);
    }

    #[test]
    fn test_accumulate_increments_total() {
        let mut h = Histogram3D::new(16);
        h.accumulate(100, 150, 200);
        h.accumulate(10, 20, 30);
        assert_eq!(h.total_pixels, 2);
    }

    // ── from_rgb_frame ────────────────────────────────────────────────────

    #[test]
    fn test_from_rgb_frame_counts_pixels() {
        // 4×4 solid red frame
        let pixels: Vec<u8> = (0..(4 * 4)).flat_map(|_| [255u8, 0, 0]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 4, 4);
        assert_eq!(h.total_pixels, 16);
    }

    #[test]
    fn test_from_rgb_frame_short_input() {
        // Less data than required → zero histogram
        let h = Histogram3D::from_rgb_frame(&[1, 2, 3], 10, 10);
        assert_eq!(h.total_pixels, 0);
    }

    // ── bin_at ────────────────────────────────────────────────────────────

    #[test]
    fn test_bin_at_matches_accumulated() {
        let mut h = Histogram3D::new(32);
        h.accumulate(128, 128, 128);
        h.accumulate(128, 128, 128);
        assert_eq!(h.bin_at(128, 128, 128), 2);
    }

    #[test]
    fn test_bin_at_zero_for_unset() {
        let h = Histogram3D::new(32);
        assert_eq!(h.bin_at(0, 0, 0), 0);
    }

    // ── dominant_colors ───────────────────────────────────────────────────

    #[test]
    fn test_dominant_colors_count() {
        let pixels: Vec<u8> = (0..(4 * 4)).flat_map(|_| [200u8, 100, 50]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 4, 4);
        let dc = h.dominant_colors(3);
        assert!(!dc.is_empty());
        assert!(dc.len() <= 3);
    }

    #[test]
    fn test_dominant_colors_sorted_descending() {
        let mut pixels = Vec::new();
        // 10 red pixels
        for _ in 0..10 {
            pixels.extend_from_slice(&[255u8, 0, 0]);
        }
        // 5 blue pixels
        for _ in 0..5 {
            pixels.extend_from_slice(&[0u8, 0, 255]);
        }
        let h = Histogram3D::from_rgb_frame(&pixels, 15, 1);
        let dc = h.dominant_colors(2);
        assert_eq!(dc.len(), 2);
        assert!(dc[0].pixel_count >= dc[1].pixel_count);
    }

    // ── color_coverage ────────────────────────────────────────────────────

    #[test]
    fn test_color_coverage_all_same() {
        // All pixels are (128,128,128); coverage at radius 0 should be 1.0
        let pixels: Vec<u8> = (0..100).flat_map(|_| [128u8, 128, 128]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 10, 10);
        let cov = h.color_coverage(128, 128, 128, 0);
        assert!((cov - 1.0).abs() < 1e-5, "coverage={cov}");
    }

    #[test]
    fn test_color_coverage_zero_for_absent_color() {
        let pixels: Vec<u8> = (0..100).flat_map(|_| [255u8, 0, 0]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 10, 10);
        // Query far from red with radius=0
        let cov = h.color_coverage(0, 255, 0, 0);
        assert!(cov < 0.05, "coverage should be near zero, got {cov}");
    }

    // ── neutrality_score ─────────────────────────────────────────────────

    #[test]
    fn test_neutrality_score_grey() {
        let pixels: Vec<u8> = (0..100).flat_map(|_| [128u8, 128, 128]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 10, 10);
        let score = h.neutrality_score();
        assert!(score > 0.95, "pure grey should be near 1.0, got {score}");
    }

    #[test]
    fn test_neutrality_score_colored() {
        // Strong red bias → low neutrality
        let pixels: Vec<u8> = (0..100).flat_map(|_| [255u8, 0, 0]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 10, 10);
        let score = h.neutrality_score();
        assert!(
            score < 0.6,
            "red-biased frame should score < 0.6, got {score}"
        );
    }

    #[test]
    fn test_neutrality_score_empty() {
        let h = Histogram3D::new(32);
        // Convention: empty histogram → neutral (1.0)
        assert_eq!(h.neutrality_score(), 1.0);
    }

    // ── to_2d_projection ─────────────────────────────────────────────────

    #[test]
    fn test_2d_projection_size() {
        let h = Histogram3D::new(8);
        for axis in 0..3_usize {
            let proj = h.to_2d_projection(axis);
            assert_eq!(proj.len(), 8);
            assert!(proj.iter().all(|row| row.len() == 8));
        }
    }

    #[test]
    fn test_2d_projection_sum_preserved() {
        let pixels: Vec<u8> = (0..50).flat_map(|_| [100u8, 150, 200]).collect();
        let h = Histogram3D::from_rgb_frame(&pixels, 50, 1);
        let proj = h.to_2d_projection(0); // sum over R
        let proj_sum: u64 = proj.iter().flatten().sum();
        assert_eq!(proj_sum, h.total_pixels);
    }

    #[test]
    fn test_2d_projection_invalid_axis() {
        let h = Histogram3D::new(8);
        assert!(h.to_2d_projection(3).is_empty());
    }
}
