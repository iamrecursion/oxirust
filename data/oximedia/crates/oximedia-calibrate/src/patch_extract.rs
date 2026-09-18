#![allow(dead_code)]
//! Patch extraction and sampling from calibration target images.
//!
//! This module extracts color values from detected calibration target patches,
//! with support for sub-pixel sampling, outlier rejection, and statistical
//! analysis of the extracted values.

/// Sampling method for extracting patch color values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMethod {
    /// Sample only the center pixel of each patch.
    CenterPixel,
    /// Average all pixels within the patch region.
    FullAverage,
    /// Average pixels in the central portion of the patch (e.g., inner 50%).
    CenterAverage,
    /// Median of all pixels within the patch.
    MedianSample,
    /// Trimmed mean with outlier rejection.
    TrimmedMean,
}

/// A rectangular region in pixel coordinates identifying a patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchRegion {
    /// Left edge (pixels, can be sub-pixel).
    pub x: f64,
    /// Top edge (pixels, can be sub-pixel).
    pub y: f64,
    /// Width of the patch region.
    pub width: f64,
    /// Height of the patch region.
    pub height: f64,
}

impl PatchRegion {
    /// Create a new patch region.
    #[must_use]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Compute the center of this region.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Compute the area.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Get the inner region at a given ratio (e.g., 0.5 for inner 50%).
    #[must_use]
    pub fn inner_region(&self, ratio: f64) -> Self {
        let ratio = ratio.clamp(0.01, 1.0);
        let new_w = self.width * ratio;
        let new_h = self.height * ratio;
        Self {
            x: self.x + (self.width - new_w) * 0.5,
            y: self.y + (self.height - new_h) * 0.5,
            width: new_w,
            height: new_h,
        }
    }

    /// Check if a point is inside this region.
    #[must_use]
    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

/// Extracted color values from a single patch.
#[derive(Debug, Clone)]
pub struct ExtractedPatch {
    /// Patch index in the target grid.
    pub index: usize,
    /// Mean RGB value (0.0–1.0 range).
    pub mean_rgb: [f64; 3],
    /// Standard deviation per channel.
    pub std_rgb: [f64; 3],
    /// Number of pixels sampled.
    pub sample_count: usize,
    /// Uniformity score (0.0–1.0, higher = more uniform).
    pub uniformity: f64,
    /// Whether the extraction is considered reliable.
    pub reliable: bool,
}

/// Configuration for patch extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Sampling method.
    pub method: SamplingMethod,
    /// Inner region ratio for `CenterAverage` (0.0–1.0).
    pub inner_ratio: f64,
    /// Trim percentage for `TrimmedMean` (0.0–0.5, fraction to trim each side).
    pub trim_fraction: f64,
    /// Maximum allowed standard deviation per channel for reliability.
    pub max_std_for_reliable: f64,
    /// Minimum sample count for reliability.
    pub min_samples: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::CenterAverage,
            inner_ratio: 0.5,
            trim_fraction: 0.1,
            max_std_for_reliable: 0.05,
            min_samples: 10,
        }
    }
}

/// Patch extractor that samples color values from image data.
#[derive(Debug)]
pub struct PatchExtractor {
    /// Extraction configuration.
    config: ExtractionConfig,
}

impl PatchExtractor {
    /// Create a new patch extractor with the given configuration.
    #[must_use]
    pub fn new(config: ExtractionConfig) -> Self {
        Self { config }
    }

    /// Create a patch extractor with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ExtractionConfig::default())
    }

    /// Extract patch color from image pixel data.
    ///
    /// `image_data` is a flat array of RGB f64 values (0.0–1.0) in row-major order.
    /// `image_width` and `image_height` are the dimensions of the image.
    #[must_use]
    pub fn extract_patch(
        &self,
        image_data: &[f64],
        image_width: usize,
        image_height: usize,
        region: &PatchRegion,
        index: usize,
    ) -> ExtractedPatch {
        let effective_region = match self.config.method {
            SamplingMethod::CenterAverage => region.inner_region(self.config.inner_ratio),
            SamplingMethod::CenterPixel => {
                let (cx, cy) = region.center();
                PatchRegion::new(cx, cy, 1.0, 1.0)
            }
            _ => *region,
        };

        let pixels = self.sample_pixels(image_data, image_width, image_height, &effective_region);

        if pixels.is_empty() {
            return ExtractedPatch {
                index,
                mean_rgb: [0.0; 3],
                std_rgb: [0.0; 3],
                sample_count: 0,
                uniformity: 0.0,
                reliable: false,
            };
        }

        let processed = match self.config.method {
            SamplingMethod::TrimmedMean => self.trim_outliers(&pixels),
            _ => pixels,
        };

        let (mean, std_dev) = compute_channel_stats(&processed);
        let uniformity = compute_uniformity(&std_dev);
        let reliable = processed.len() >= self.config.min_samples
            && std_dev
                .iter()
                .all(|&s| s <= self.config.max_std_for_reliable);

        ExtractedPatch {
            index,
            mean_rgb: mean,
            std_rgb: std_dev,
            sample_count: processed.len(),
            uniformity,
            reliable,
        }
    }

    /// Extract all patches from an image given a list of regions.
    #[must_use]
    pub fn extract_all(
        &self,
        image_data: &[f64],
        image_width: usize,
        image_height: usize,
        regions: &[PatchRegion],
    ) -> Vec<ExtractedPatch> {
        regions
            .iter()
            .enumerate()
            .map(|(i, region)| self.extract_patch(image_data, image_width, image_height, region, i))
            .collect()
    }

    /// Sample pixels from an image within a region.
    fn sample_pixels(
        &self,
        image_data: &[f64],
        image_width: usize,
        image_height: usize,
        region: &PatchRegion,
    ) -> Vec<[f64; 3]> {
        let mut pixels = Vec::new();
        let x_start = (region.x.floor() as usize).min(image_width.saturating_sub(1));
        let y_start = (region.y.floor() as usize).min(image_height.saturating_sub(1));
        let x_end = ((region.x + region.width).ceil() as usize).min(image_width);
        let y_end = ((region.y + region.height).ceil() as usize).min(image_height);

        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = (y * image_width + x) * 3;
                if idx + 2 < image_data.len() {
                    pixels.push([image_data[idx], image_data[idx + 1], image_data[idx + 2]]);
                }
            }
        }
        pixels
    }

    /// Trim outlier pixels (trimmed mean).
    fn trim_outliers(&self, pixels: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if pixels.len() < 4 {
            return pixels.to_vec();
        }
        let trim_count = ((pixels.len() as f64 * self.config.trim_fraction) as usize).max(1);
        if trim_count * 2 >= pixels.len() {
            return pixels.to_vec();
        }

        // Sort by luminance for trimming
        let mut indexed: Vec<(usize, f64)> = pixels
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed[trim_count..indexed.len() - trim_count]
            .iter()
            .map(|(i, _)| pixels[*i])
            .collect()
    }
}

/// Compute per-channel mean and standard deviation.
fn compute_channel_stats(pixels: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    if pixels.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let n = pixels.len() as f64;
    let mut sum = [0.0_f64; 3];
    for p in pixels {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

    let mut var = [0.0_f64; 3];
    for p in pixels {
        var[0] += (p[0] - mean[0]).powi(2);
        var[1] += (p[1] - mean[1]).powi(2);
        var[2] += (p[2] - mean[2]).powi(2);
    }
    let std_dev = [
        (var[0] / n).sqrt(),
        (var[1] / n).sqrt(),
        (var[2] / n).sqrt(),
    ];
    (mean, std_dev)
}

/// Compute a uniformity score from standard deviations (0.0–1.0, higher is better).
fn compute_uniformity(std_dev: &[f64; 3]) -> f64 {
    let avg_std = (std_dev[0] + std_dev[1] + std_dev[2]) / 3.0;
    // Map average std to a 0–1 score: 0.0 std -> 1.0 score, 0.1+ -> near 0
    (1.0 - avg_std * 10.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_region_center() {
        let r = PatchRegion::new(10.0, 20.0, 100.0, 50.0);
        let (cx, cy) = r.center();
        assert!((cx - 60.0).abs() < 1e-10);
        assert!((cy - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_patch_region_area() {
        let r = PatchRegion::new(0.0, 0.0, 10.0, 20.0);
        assert!((r.area() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_patch_region_inner() {
        let r = PatchRegion::new(0.0, 0.0, 100.0, 100.0);
        let inner = r.inner_region(0.5);
        assert!((inner.x - 25.0).abs() < 1e-10);
        assert!((inner.y - 25.0).abs() < 1e-10);
        assert!((inner.width - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_patch_region_contains() {
        let r = PatchRegion::new(10.0, 10.0, 50.0, 50.0);
        assert!(r.contains(35.0, 35.0));
        assert!(!r.contains(5.0, 5.0));
        assert!(!r.contains(65.0, 35.0));
    }

    #[test]
    fn test_compute_channel_stats_empty() {
        let (mean, std) = compute_channel_stats(&[]);
        assert!((mean[0] - 0.0).abs() < 1e-10);
        assert!((std[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_channel_stats_uniform() {
        let pixels = vec![[0.5, 0.5, 0.5]; 10];
        let (mean, std) = compute_channel_stats(&pixels);
        assert!((mean[0] - 0.5).abs() < 1e-10);
        assert!((std[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_channel_stats_varied() {
        let pixels = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let (mean, std) = compute_channel_stats(&pixels);
        assert!((mean[0] - 0.5).abs() < 1e-10);
        assert!((std[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_uniformity_perfect() {
        let std = [0.0, 0.0, 0.0];
        assert!((compute_uniformity(&std) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_uniformity_poor() {
        let std = [0.2, 0.2, 0.2];
        assert!((compute_uniformity(&std) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_patch_uniform() {
        let extractor = PatchExtractor::new(ExtractionConfig {
            method: SamplingMethod::FullAverage,
            ..ExtractionConfig::default()
        });
        // 4x4 image, all pixels at (0.5, 0.3, 0.7)
        let image: Vec<f64> = (0..4 * 4).flat_map(|_| vec![0.5, 0.3, 0.7]).collect();
        let region = PatchRegion::new(0.0, 0.0, 4.0, 4.0);
        let result = extractor.extract_patch(&image, 4, 4, &region, 0);
        assert!((result.mean_rgb[0] - 0.5).abs() < 1e-10);
        assert!((result.mean_rgb[1] - 0.3).abs() < 1e-10);
        assert!((result.mean_rgb[2] - 0.7).abs() < 1e-10);
        assert!(result.reliable);
    }

    #[test]
    fn test_extract_patch_center_average() {
        let extractor = PatchExtractor::with_defaults();
        // 10x10 image
        let image: Vec<f64> = (0..10 * 10).flat_map(|_| vec![0.4, 0.5, 0.6]).collect();
        let region = PatchRegion::new(0.0, 0.0, 10.0, 10.0);
        let result = extractor.extract_patch(&image, 10, 10, &region, 0);
        assert!(result.sample_count > 0);
    }

    #[test]
    fn test_extract_all() {
        let extractor = PatchExtractor::new(ExtractionConfig {
            method: SamplingMethod::FullAverage,
            min_samples: 1,
            ..ExtractionConfig::default()
        });
        let image: Vec<f64> = (0..8 * 8).flat_map(|_| vec![0.5, 0.5, 0.5]).collect();
        let regions = vec![
            PatchRegion::new(0.0, 0.0, 4.0, 4.0),
            PatchRegion::new(4.0, 0.0, 4.0, 4.0),
            PatchRegion::new(0.0, 4.0, 4.0, 4.0),
        ];
        let results = extractor.extract_all(&image, 8, 8, &regions);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.sample_count > 0);
        }
    }

    #[test]
    fn test_trimmed_mean() {
        let extractor = PatchExtractor::new(ExtractionConfig {
            method: SamplingMethod::TrimmedMean,
            trim_fraction: 0.2,
            min_samples: 1,
            max_std_for_reliable: 1.0,
            ..ExtractionConfig::default()
        });
        // 10x1 image with one outlier
        let mut image: Vec<f64> = (0..10).flat_map(|_| vec![0.5, 0.5, 0.5]).collect();
        // Make first pixel an outlier
        image[0] = 1.0;
        image[1] = 1.0;
        image[2] = 1.0;
        let region = PatchRegion::new(0.0, 0.0, 10.0, 1.0);
        let result = extractor.extract_patch(&image, 10, 1, &region, 0);
        // Trimmed mean should be closer to 0.5 than simple average
        assert!(result.mean_rgb[0] < 0.6);
    }

    #[test]
    fn test_extract_empty_region() {
        let extractor = PatchExtractor::with_defaults();
        let image: Vec<f64> = Vec::new();
        let region = PatchRegion::new(0.0, 0.0, 10.0, 10.0);
        let result = extractor.extract_patch(&image, 0, 0, &region, 0);
        assert_eq!(result.sample_count, 0);
        assert!(!result.reliable);
    }
}
