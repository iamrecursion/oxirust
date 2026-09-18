//! Superpixel segmentation using a simplified SLIC algorithm.
//!
//! Provides nearest-center superpixel label assignment and basic superpixel statistics.

/// A single superpixel region with associated pixels and color statistics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Superpixel {
    /// Unique identifier for this superpixel.
    pub id: u32,
    /// X coordinate of the superpixel center.
    pub center_x: f32,
    /// Y coordinate of the superpixel center.
    pub center_y: f32,
    /// List of `(x, y)` pixel coordinates belonging to this superpixel.
    pub pixels: Vec<(u32, u32)>,
    /// Mean color `[R, G, B]` of pixels in this superpixel.
    pub mean_color: [f32; 3],
}

impl Superpixel {
    /// Returns the number of pixels in this superpixel.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.pixels.len()
    }

    /// Returns the pixel area (same as `pixel_count`).
    #[must_use]
    pub fn area(&self) -> usize {
        self.pixels.len()
    }

    /// Returns the compactness: ratio of `pixel_count` to the bounding-box area.
    ///
    /// A value of 1.0 means the superpixel perfectly fills its bounding box.
    /// Returns 0.0 if the superpixel is empty.
    #[must_use]
    pub fn compactness(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let min_x = self.pixels.iter().map(|p| p.0).min().unwrap_or(0);
        let max_x = self.pixels.iter().map(|p| p.0).max().unwrap_or(0);
        let min_y = self.pixels.iter().map(|p| p.1).min().unwrap_or(0);
        let max_y = self.pixels.iter().map(|p| p.1).max().unwrap_or(0);
        let bbox_w = (max_x - min_x + 1) as f32;
        let bbox_h = (max_y - min_y + 1) as f32;
        let bbox_area = bbox_w * bbox_h;
        if bbox_area < 1e-6 {
            return 1.0;
        }
        (self.pixels.len() as f32 / bbox_area).min(1.0)
    }
}

/// Configuration for the SLIC superpixel algorithm.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SlicConfig {
    /// Approximate number of superpixels to generate.
    pub num_superpixels: u32,
    /// Compactness weight controlling the trade-off between color and spatial distance.
    pub compactness: f32,
    /// Maximum number of iterations for the SLIC loop.
    pub max_iterations: u32,
}

impl SlicConfig {
    /// Creates the default SLIC configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            num_superpixels: 100,
            compactness: 10.0,
            max_iterations: 10,
        }
    }
}

impl Default for SlicConfig {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Superpixel segmenter based on a simplified SLIC algorithm.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SuperpixelSegmenter {
    /// Configuration parameters.
    pub config: SlicConfig,
}

impl SuperpixelSegmenter {
    /// Creates a new segmenter with the given configuration.
    #[must_use]
    pub fn new(config: SlicConfig) -> Self {
        Self { config }
    }

    /// Estimates the grid step size S = sqrt(W * H / k).
    ///
    /// Returns at least 1.
    #[must_use]
    pub fn estimate_grid_step(&self, img_w: u32, img_h: u32) -> u32 {
        let k = self.config.num_superpixels.max(1);
        let area = img_w as f64 * img_h as f64;
        let step = (area / f64::from(k)).sqrt() as u32;
        step.max(1)
    }

    /// Assigns each pixel to its nearest center using Euclidean distance in image space.
    ///
    /// `pixels` is an interleaved RGB buffer of length `width * height * 3`.
    /// Returns a label buffer of length `width * height` where each value is the
    /// index of the nearest center.
    #[must_use]
    pub fn assign_pixels_to_centers(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        centers: &[(f32, f32)],
    ) -> Vec<u32> {
        let total = (width * height) as usize;
        let mut labels = vec![0u32; total];
        if centers.is_empty() || total == 0 {
            return labels;
        }
        for py in 0..height {
            for px in 0..width {
                let idx = (py * width + px) as usize;
                let _ = pixels; // color unused in spatial-only assignment for simplicity
                let best = centers
                    .iter()
                    .enumerate()
                    .map(|(ci, &(cx, cy))| {
                        let dx = px as f32 - cx;
                        let dy = py as f32 - cy;
                        (ci, dx * dx + dy * dy)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(ci, _)| ci as u32);
                labels[idx] = best;
            }
        }
        labels
    }

    /// Counts the number of unique superpixel labels in the label buffer.
    #[must_use]
    pub fn count_superpixels(labels: &[u32]) -> u32 {
        if labels.is_empty() {
            return 0;
        }
        let mut seen = std::collections::HashSet::new();
        for &l in labels {
            seen.insert(l);
        }
        seen.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_segmenter() -> SuperpixelSegmenter {
        SuperpixelSegmenter::new(SlicConfig::default())
    }

    // ---- Superpixel tests ----

    fn make_superpixel(pixels: Vec<(u32, u32)>) -> Superpixel {
        Superpixel {
            id: 0,
            center_x: 0.0,
            center_y: 0.0,
            pixels,
            mean_color: [128.0, 64.0, 32.0],
        }
    }

    #[test]
    fn test_superpixel_pixel_count() {
        let sp = make_superpixel(vec![(0, 0), (1, 0), (2, 0)]);
        assert_eq!(sp.pixel_count(), 3);
    }

    #[test]
    fn test_superpixel_area_equals_pixel_count() {
        let sp = make_superpixel(vec![(0, 0), (0, 1)]);
        assert_eq!(sp.area(), sp.pixel_count());
    }

    #[test]
    fn test_superpixel_compactness_empty() {
        let sp = make_superpixel(vec![]);
        assert_eq!(sp.compactness(), 0.0);
    }

    #[test]
    fn test_superpixel_compactness_full_rectangle() {
        // 2x2 square: 4 pixels in a 2x2 bounding box -> compactness = 1.0
        let sp = make_superpixel(vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        let c = sp.compactness();
        assert!((c - 1.0).abs() < 1e-5, "compactness={}", c);
    }

    #[test]
    fn test_superpixel_compactness_sparse() {
        // 2 pixels in corners of a 3x1 bounding box -> 2/3 ≈ 0.667
        let sp = make_superpixel(vec![(0, 0), (2, 0)]);
        let c = sp.compactness();
        // bbox = 3x1 = 3; pixels = 2; ratio = 2/3
        assert!((c - 2.0 / 3.0).abs() < 1e-5, "compactness={}", c);
    }

    #[test]
    fn test_superpixel_single_pixel_compactness() {
        let sp = make_superpixel(vec![(5, 7)]);
        // single pixel: bbox = 1x1 = 1, count = 1 -> 1.0
        assert!((sp.compactness() - 1.0).abs() < 1e-5);
    }

    // ---- SlicConfig tests ----

    #[test]
    fn test_slic_config_default_values() {
        let cfg = SlicConfig::default();
        assert_eq!(cfg.num_superpixels, 100);
        assert!((cfg.compactness - 10.0).abs() < 1e-5);
        assert_eq!(cfg.max_iterations, 10);
    }

    // ---- SuperpixelSegmenter tests ----

    #[test]
    fn test_estimate_grid_step_basic() {
        let seg = default_segmenter(); // k=100
                                       // 100x100 image / 100 = step 10
        let step = seg.estimate_grid_step(100, 100);
        assert_eq!(step, 10);
    }

    #[test]
    fn test_estimate_grid_step_minimum_one() {
        let mut cfg = SlicConfig::default();
        cfg.num_superpixels = 10000;
        let seg = SuperpixelSegmenter::new(cfg);
        // Very small image -> step should be at least 1
        let step = seg.estimate_grid_step(5, 5);
        assert!(step >= 1);
    }

    #[test]
    fn test_estimate_grid_step_large_image() {
        let seg = default_segmenter(); // k=100
                                       // 1000x1000 / 100 = 100
        let step = seg.estimate_grid_step(1000, 1000);
        assert_eq!(step, 100);
    }

    #[test]
    fn test_assign_pixels_no_centers_returns_zeros() {
        let seg = default_segmenter();
        let pixels = vec![0u8; 9]; // 3x3 RGB? irrelevant here
        let labels = seg.assign_pixels_to_centers(&pixels, 3, 3, &[]);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_assign_pixels_single_center() {
        let seg = default_segmenter();
        let pixels = vec![128u8; 12]; // 2x2, 3ch
        let centers = vec![(1.0_f32, 1.0_f32)];
        let labels = seg.assign_pixels_to_centers(&pixels, 2, 2, &centers);
        assert_eq!(labels.len(), 4);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_assign_pixels_two_centers_partition() {
        let seg = default_segmenter();
        let pixels = vec![0u8; 12]; // 4x1 image
                                    // Left center at x=0.5, right center at x=2.5
        let centers = vec![(0.5_f32, 0.0_f32), (2.5_f32, 0.0_f32)];
        let labels = seg.assign_pixels_to_centers(&pixels, 4, 1, &centers);
        // pixels 0,1 -> center 0; pixels 2,3 -> center 1
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], 1);
    }

    #[test]
    fn test_count_superpixels_empty() {
        assert_eq!(SuperpixelSegmenter::count_superpixels(&[]), 0);
    }

    #[test]
    fn test_count_superpixels_all_same() {
        let labels = vec![3u32; 10];
        assert_eq!(SuperpixelSegmenter::count_superpixels(&labels), 1);
    }

    #[test]
    fn test_count_superpixels_unique_labels() {
        let labels = vec![0u32, 1, 2, 1, 0, 3];
        assert_eq!(SuperpixelSegmenter::count_superpixels(&labels), 4);
    }
}
