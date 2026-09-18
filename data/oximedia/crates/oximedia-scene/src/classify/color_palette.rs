//! Dominant color palette extraction using k-means clustering.
//!
//! Extracts the dominant colors from a video frame by clustering pixels in RGB space
//! using Lloyd's k-means algorithm implemented from scratch.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// A single color in the extracted palette.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaletteColor {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
    /// Fraction of pixels represented by this cluster (0.0–1.0).
    pub weight: f32,
}

impl PaletteColor {
    /// Create a new palette color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, weight: f32) -> Self {
        Self { r, g, b, weight }
    }

    /// Get the color as an RGB tuple of f32 in [0, 1].
    #[must_use]
    pub fn as_float(&self) -> [f32; 3] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        ]
    }

    /// Compute perceptual luminance (ITU-R BT.709).
    #[must_use]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r as f32 / 255.0
            + 0.7152 * self.g as f32 / 255.0
            + 0.0722 * self.b as f32 / 255.0
    }

    /// Convert to CSS hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Euclidean distance squared to another color in RGB space.
    #[must_use]
    pub fn distance_sq(&self, other: &Self) -> f32 {
        let dr = self.r as f32 - other.r as f32;
        let dg = self.g as f32 - other.g as f32;
        let db = self.b as f32 - other.b as f32;
        dr * dr + dg * dg + db * db
    }

    /// Euclidean distance to another color.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        self.distance_sq(other).sqrt()
    }
}

/// Extracted color palette.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Dominant colors sorted by weight descending.
    pub colors: Vec<PaletteColor>,
    /// Number of clusters used.
    pub k: usize,
    /// Number of pixels sampled for clustering.
    pub pixels_sampled: usize,
    /// Final inertia (within-cluster sum of squared distances).
    pub inertia: f32,
}

impl ColorPalette {
    /// Get the most dominant color.
    #[must_use]
    pub fn primary(&self) -> Option<&PaletteColor> {
        self.colors.first()
    }

    /// Get colors as CSS hex strings.
    #[must_use]
    pub fn to_hex_strings(&self) -> Vec<String> {
        self.colors.iter().map(PaletteColor::to_hex).collect()
    }

    /// Average warmth of palette (0=cool, 1=warm).
    #[must_use]
    pub fn warmth(&self) -> f32 {
        let total_weight: f32 = self.colors.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 {
            return 0.5;
        }
        self.colors
            .iter()
            .map(|c| {
                let warmth = (c.r as f32 - c.b as f32) / 255.0 / 2.0 + 0.5;
                warmth * c.weight
            })
            .sum::<f32>()
            / total_weight
    }
}

/// Configuration for palette extraction.
#[derive(Debug, Clone)]
pub struct PaletteConfig {
    /// Number of clusters (colors) to extract.
    pub k: usize,
    /// Maximum k-means iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (centroid movement).
    pub tolerance: f32,
    /// Maximum pixels to sample (for performance).
    pub max_pixels: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for PaletteConfig {
    fn default() -> Self {
        Self {
            k: 6,
            max_iterations: 50,
            tolerance: 1.0,
            max_pixels: 4096,
            seed: 42,
        }
    }
}

/// Extracts dominant color palette from video frames using k-means clustering.
pub struct ColorPaletteExtractor {
    config: PaletteConfig,
}

impl ColorPaletteExtractor {
    /// Create with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PaletteConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: PaletteConfig) -> Self {
        Self { config }
    }

    /// Extract dominant colors from a single RGB frame.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Raw RGB pixel data (3 bytes per pixel, row-major)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match or k >= pixel count.
    pub fn extract(&self, rgb: &[u8], width: usize, height: usize) -> SceneResult<ColorPalette> {
        crate::classify::validate_frame(rgb, width, height)?;

        let pixel_count = width * height;
        if self.config.k > pixel_count {
            return Err(SceneError::InvalidParameter(format!(
                "k ({}) must be less than pixel count ({})",
                self.config.k, pixel_count
            )));
        }

        // Sample pixels (stride-based, deterministic)
        let samples = self.sample_pixels(rgb, pixel_count);
        let n = samples.len();

        if n < self.config.k {
            return Err(SceneError::InsufficientData(format!(
                "Not enough sample pixels ({}) for k={}",
                n, self.config.k
            )));
        }

        // Initialize centroids using k-means++ variant
        let mut centroids = self.init_centroids_kmeans_plus_plus(&samples);

        // Lloyd's algorithm iterations
        let mut assignments = vec![0usize; n];
        let mut inertia = f32::MAX;

        for _ in 0..self.config.max_iterations {
            // Assignment step
            let mut new_inertia = 0.0f32;
            for (i, pixel) in samples.iter().enumerate() {
                let (best_c, dist_sq) = Self::nearest_centroid(pixel, &centroids);
                assignments[i] = best_c;
                new_inertia += dist_sq;
            }

            // Update step
            let new_centroids =
                Self::update_centroids(&samples, &assignments, &centroids, self.config.k);

            // Check convergence
            let movement: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| {
                    let dr = old[0] - new[0];
                    let dg = old[1] - new[1];
                    let db = old[2] - new[2];
                    (dr * dr + dg * dg + db * db).sqrt()
                })
                .sum();

            centroids = new_centroids;
            inertia = new_inertia;

            if movement < self.config.tolerance {
                break;
            }
        }

        // Count cluster sizes for weights
        let mut counts = vec![0u32; self.config.k];
        for &a in &assignments {
            counts[a] += 1;
        }

        let mut palette: Vec<PaletteColor> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let weight = counts[i] as f32 / n as f32;
                PaletteColor::new(c[0] as u8, c[1] as u8, c[2] as u8, weight)
            })
            .collect();

        // Sort by weight descending
        palette.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ColorPalette {
            colors: palette,
            k: self.config.k,
            pixels_sampled: n,
            inertia,
        })
    }

    /// Extract palette from multiple frames (temporal pooling).
    ///
    /// Samples pixels across all frames for a temporally averaged palette.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame has inconsistent dimensions or extraction fails.
    pub fn extract_from_frames(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<ColorPalette> {
        if frames.is_empty() {
            return Err(SceneError::InsufficientData(
                "No frames provided".to_string(),
            ));
        }

        // Pool all frames into a single flat buffer then extract
        let pixels_per_frame = (self.config.max_pixels / frames.len()).max(64);
        let step = ((width * height) / pixels_per_frame).max(1);

        let mut pooled: Vec<u8> = Vec::new();
        for frame in frames {
            crate::classify::validate_frame(frame, width, height)?;
            let pixel_count = width * height;
            let mut i = 0;
            while i < pixel_count {
                let idx = i * 3;
                pooled.push(frame[idx]);
                pooled.push(frame[idx + 1]);
                pooled.push(frame[idx + 2]);
                i += step;
            }
        }

        // Use a synthetic width/height matching the pooled size
        let pooled_pixels = pooled.len() / 3;
        self.extract(&pooled, pooled_pixels, 1)
    }

    fn sample_pixels(&self, rgb: &[u8], pixel_count: usize) -> Vec<[f32; 3]> {
        let step = (pixel_count / self.config.max_pixels).max(1);
        let mut samples = Vec::with_capacity(self.config.max_pixels.min(pixel_count));
        let mut i = 0;
        while i < pixel_count {
            let idx = i * 3;
            samples.push([rgb[idx] as f32, rgb[idx + 1] as f32, rgb[idx + 2] as f32]);
            i += step;
        }
        samples
    }

    /// K-means++ initialization for better convergence.
    fn init_centroids_kmeans_plus_plus(&self, samples: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let n = samples.len();
        let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(self.config.k);

        // First centroid: use a deterministic sample based on seed
        let first_idx = (self.config.seed as usize) % n;
        centroids.push(samples[first_idx]);

        // Subsequent centroids: proportional to squared distance
        for _ in 1..self.config.k {
            // Compute min-squared-distances
            let mut dist_sq: Vec<f32> = samples
                .iter()
                .map(|p| {
                    centroids
                        .iter()
                        .map(|c| {
                            let dr = p[0] - c[0];
                            let dg = p[1] - c[1];
                            let db = p[2] - c[2];
                            dr * dr + dg * dg + db * db
                        })
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Pick next centroid: the point with maximum distance
            // (deterministic approximation of k-means++ proportional sampling)
            let max_idx = dist_sq
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            // Perturb slightly to avoid duplicate centroids when frame is uniform
            let seed_offset = centroids.len() as f32 * 3.7;
            let candidate = samples[max_idx];
            if dist_sq[max_idx] < 1.0 {
                // Nearly identical to existing — use evenly spaced fallback
                let idx2 = (max_idx + n / (self.config.k + 1) + 1) % n;
                centroids.push([
                    samples[idx2][0] + seed_offset.sin() * 2.0,
                    samples[idx2][1] + seed_offset.cos() * 2.0,
                    samples[idx2][2] + seed_offset * 0.5,
                ]);
            } else {
                centroids.push(candidate);
            }

            // Reset dist_sq to avoid unused variable warning
            dist_sq.clear();
        }

        // Clamp all centroids to [0, 255]
        for c in &mut centroids {
            c[0] = c[0].clamp(0.0, 255.0);
            c[1] = c[1].clamp(0.0, 255.0);
            c[2] = c[2].clamp(0.0, 255.0);
        }

        centroids
    }

    fn nearest_centroid(pixel: &[f32; 3], centroids: &[[f32; 3]]) -> (usize, f32) {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (i, c) in centroids.iter().enumerate() {
            let dr = pixel[0] - c[0];
            let dg = pixel[1] - c[1];
            let db = pixel[2] - c[2];
            let d = dr * dr + dg * dg + db * db;
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        (best, best_dist)
    }

    fn update_centroids(
        samples: &[[f32; 3]],
        assignments: &[usize],
        old: &[[f32; 3]],
        k: usize,
    ) -> Vec<[f32; 3]> {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u32; k];

        for (i, &c) in assignments.iter().enumerate() {
            sums[c][0] += samples[i][0] as f64;
            sums[c][1] += samples[i][1] as f64;
            sums[c][2] += samples[i][2] as f64;
            counts[c] += 1;
        }

        sums.iter()
            .enumerate()
            .map(|(i, s)| {
                if counts[i] == 0 {
                    old[i] // Keep old centroid if no assignments
                } else {
                    let n = counts[i] as f64;
                    [
                        (s[0] / n).clamp(0.0, 255.0) as f32,
                        (s[1] / n).clamp(0.0, 255.0) as f32,
                        (s[2] / n).clamp(0.0, 255.0) as f32,
                    ]
                }
            })
            .collect()
    }
}

impl Default for ColorPaletteExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    fn two_color_frame(
        r1: u8,
        g1: u8,
        b1: u8,
        r2: u8,
        g2: u8,
        b2: u8,
        w: usize,
        h: usize,
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for i in 0..w * h {
            if i < w * h / 2 {
                v.push(r1);
                v.push(g1);
                v.push(b1);
            } else {
                v.push(r2);
                v.push(g2);
                v.push(b2);
            }
        }
        v
    }

    #[test]
    fn test_palette_color_hex() {
        let c = PaletteColor::new(255, 0, 0, 1.0);
        assert_eq!(c.to_hex(), "#FF0000");
        let c2 = PaletteColor::new(0, 255, 0, 1.0);
        assert_eq!(c2.to_hex(), "#00FF00");
    }

    #[test]
    fn test_palette_color_luminance() {
        let white = PaletteColor::new(255, 255, 255, 1.0);
        assert!((white.luminance() - 1.0).abs() < 0.01);
        let black = PaletteColor::new(0, 0, 0, 1.0);
        assert!(black.luminance() < 0.01);
    }

    #[test]
    fn test_palette_color_distance() {
        let red = PaletteColor::new(255, 0, 0, 1.0);
        let blue = PaletteColor::new(0, 0, 255, 1.0);
        let dist = red.distance(&blue);
        assert!((dist - (255.0_f32 * std::f32::consts::SQRT_2)).abs() < 1.0);
    }

    #[test]
    fn test_extract_solid_frame() {
        let extractor = ColorPaletteExtractor::new();
        let frame = solid_frame(200, 100, 50, 64, 64);
        let palette = extractor
            .extract(&frame, 64, 64)
            .expect("should succeed in test");
        assert_eq!(palette.colors.len(), 6);
        // All pixels the same color, dominant should be ~(200,100,50)
        let primary = palette.primary().expect("should succeed in test");
        assert!((primary.r as i32 - 200).abs() < 10);
        assert!((primary.g as i32 - 100).abs() < 10);
    }

    #[test]
    fn test_extract_k_colors() {
        let config = PaletteConfig {
            k: 3,
            max_pixels: 512,
            ..Default::default()
        };
        let extractor = ColorPaletteExtractor::with_config(config);
        let frame = solid_frame(128, 128, 128, 32, 32);
        let palette = extractor
            .extract(&frame, 32, 32)
            .expect("should succeed in test");
        assert_eq!(palette.colors.len(), 3);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let extractor = ColorPaletteExtractor::new();
        let frame = two_color_frame(255, 0, 0, 0, 0, 255, 64, 64);
        let palette = extractor
            .extract(&frame, 64, 64)
            .expect("should succeed in test");
        let sum: f32 = palette.colors.iter().map(|c| c.weight).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights sum = {sum}");
    }

    #[test]
    fn test_two_color_frame_finds_both() {
        let config = PaletteConfig {
            k: 2,
            max_pixels: 1024,
            ..Default::default()
        };
        let extractor = ColorPaletteExtractor::with_config(config);
        let frame = two_color_frame(255, 0, 0, 0, 0, 255, 64, 64);
        let palette = extractor
            .extract(&frame, 64, 64)
            .expect("should succeed in test");
        assert_eq!(palette.colors.len(), 2);
        // One should be reddish, one blueish
        let has_red = palette.colors.iter().any(|c| c.r > 150 && c.b < 100);
        let has_blue = palette.colors.iter().any(|c| c.b > 150 && c.r < 100);
        assert!(has_red, "Expected a reddish color");
        assert!(has_blue, "Expected a blueish color");
    }

    #[test]
    fn test_extract_invalid_frame() {
        let extractor = ColorPaletteExtractor::new();
        let result = extractor.extract(&[0u8; 5], 64, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_k_too_large() {
        let config = PaletteConfig {
            k: 1000,
            ..Default::default()
        };
        let extractor = ColorPaletteExtractor::with_config(config);
        // 4x4 image = 16 pixels < k=1000
        let frame = solid_frame(100, 100, 100, 4, 4);
        let result = extractor.extract(&frame, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_palette_warmth() {
        let extractor = ColorPaletteExtractor::new();
        // Warm frame: orange-red
        let warm_frame = solid_frame(220, 140, 30, 32, 32);
        let palette = extractor
            .extract(&warm_frame, 32, 32)
            .expect("should succeed in test");
        assert!(palette.warmth() > 0.5, "Expected warm palette");
        // Cool frame: blue
        let cool_frame = solid_frame(30, 80, 220, 32, 32);
        let palette2 = extractor
            .extract(&cool_frame, 32, 32)
            .expect("should succeed in test");
        assert!(palette2.warmth() < 0.5, "Expected cool palette");
    }

    #[test]
    fn test_hex_strings() {
        let extractor = ColorPaletteExtractor::new();
        let frame = solid_frame(100, 100, 100, 32, 32);
        let palette = extractor
            .extract(&frame, 32, 32)
            .expect("should succeed in test");
        let hexes = palette.to_hex_strings();
        assert_eq!(hexes.len(), palette.colors.len());
        for h in &hexes {
            assert!(h.starts_with('#'));
            assert_eq!(h.len(), 7);
        }
    }

    #[test]
    fn test_extract_from_frames() {
        let extractor = ColorPaletteExtractor::new();
        let f1 = solid_frame(200, 50, 50, 32, 32);
        let f2 = solid_frame(50, 50, 200, 32, 32);
        let frames: Vec<&[u8]> = vec![&f1, &f2];
        let palette = extractor
            .extract_from_frames(&frames, 32, 32)
            .expect("should succeed in test");
        assert!(palette.colors.len() > 0);
    }

    #[test]
    fn test_palette_color_as_float() {
        let c = PaletteColor::new(255, 128, 0, 0.5);
        let f = c.as_float();
        assert!((f[0] - 1.0).abs() < 0.01);
        assert!((f[1] - 0.502).abs() < 0.01);
        assert!(f[2] < 0.01);
    }

    #[test]
    fn test_default_extractor() {
        let _ext = ColorPaletteExtractor::default();
    }

    #[test]
    fn test_inertia_is_finite() {
        let extractor = ColorPaletteExtractor::new();
        let frame = solid_frame(128, 64, 32, 32, 32);
        let palette = extractor
            .extract(&frame, 32, 32)
            .expect("should succeed in test");
        assert!(palette.inertia.is_finite());
    }
}
