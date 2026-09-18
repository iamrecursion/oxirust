//! Text region detection using connected components and edge analysis.

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Detected text region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDetection {
    /// Bounding box of text region.
    pub bbox: Rect,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Text properties.
    pub properties: TextProperties,
}

/// Properties of detected text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProperties {
    /// Text orientation (horizontal, vertical).
    pub orientation: TextOrientation,
    /// Estimated text size category.
    pub size_category: TextSizeCategory,
    /// Estimated contrast (0.0-1.0).
    pub contrast: f32,
    /// Density of text in region.
    pub density: f32,
}

/// Text orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextOrientation {
    /// Horizontal text.
    Horizontal,
    /// Vertical text.
    Vertical,
    /// Unknown orientation.
    Unknown,
}

/// Text size categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextSizeCategory {
    /// Small text (< 20px).
    Small,
    /// Medium text (20-50px).
    Medium,
    /// Large text (> 50px).
    Large,
}

/// Configuration for text detection.
#[derive(Debug, Clone)]
pub struct TextDetectorConfig {
    /// Minimum confidence threshold.
    pub confidence_threshold: f32,
    /// Minimum text region size.
    pub min_region_size: usize,
    /// Edge threshold for text detection.
    pub edge_threshold: u8,
    /// Merge nearby regions.
    pub merge_regions: bool,
}

impl Default for TextDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_region_size: 10,
            edge_threshold: 30,
            merge_regions: true,
        }
    }
}

/// Text detector using edge analysis and connected components.
pub struct TextDetector {
    config: TextDetectorConfig,
}

impl TextDetector {
    /// Create a new text detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TextDetectorConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: TextDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect text regions in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - RGB image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<TextDetection>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Convert to grayscale
        let gray = self.rgb_to_gray(rgb_data, width, height);

        // Detect edges
        let edges = self.detect_edges(&gray, width, height);

        // Find text candidates using stroke width transform concept
        let candidates = self.find_text_candidates(&edges, width, height);

        // Filter and merge regions
        let mut regions = self.filter_candidates(&candidates, width, height);

        if self.config.merge_regions {
            regions = self.merge_nearby_regions(&regions);
        }

        // Create text detections with properties
        let mut detections = Vec::new();
        for bbox in regions {
            let properties = self.extract_properties(rgb_data, &gray, width, height, &bbox);
            let confidence = self.calculate_confidence(&properties);

            if confidence.value() >= self.config.confidence_threshold {
                detections.push(TextDetection {
                    bbox,
                    confidence,
                    properties,
                });
            }
        }

        Ok(detections)
    }

    /// Convert RGB to grayscale.
    fn rgb_to_gray(&self, rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut gray = Vec::with_capacity(width * height);
        for i in (0..rgb.len()).step_by(3) {
            let r = rgb[i] as f32;
            let g = rgb[i + 1] as f32;
            let b = rgb[i + 2] as f32;
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            gray.push(y as u8);
        }
        gray
    }

    /// Detect edges using Sobel operator.
    fn detect_edges(&self, gray: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut edges = vec![0u8; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;

                // Sobel kernels
                let gx = (gray[idx - width + 1] as i32
                    + 2 * gray[idx + 1] as i32
                    + gray[idx + width + 1] as i32
                    - gray[idx - width - 1] as i32
                    - 2 * gray[idx - 1] as i32
                    - gray[idx + width - 1] as i32)
                    .abs();

                let gy = (gray[idx + width - 1] as i32
                    + 2 * gray[idx + width] as i32
                    + gray[idx + width + 1] as i32
                    - gray[idx - width - 1] as i32
                    - 2 * gray[idx - width] as i32
                    - gray[idx - width + 1] as i32)
                    .abs();

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as i32;
                edges[idx] = magnitude.min(255) as u8;
            }
        }

        edges
    }

    /// Find text candidate regions.
    fn find_text_candidates(&self, edges: &[u8], width: usize, height: usize) -> Vec<Rect> {
        let mut candidates = Vec::new();
        let mut visited = vec![false; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;

                if visited[idx] || edges[idx] < self.config.edge_threshold {
                    continue;
                }

                // Flood fill to find connected component
                let bbox = self.flood_fill_component(edges, &mut visited, width, height, x, y);

                if bbox.width as usize >= self.config.min_region_size
                    && bbox.height as usize >= self.config.min_region_size
                {
                    candidates.push(bbox);
                }
            }
        }

        candidates
    }

    /// Flood fill to find connected component.
    fn flood_fill_component(
        &self,
        edges: &[u8],
        visited: &mut [bool],
        width: usize,
        height: usize,
        start_x: usize,
        start_y: usize,
    ) -> Rect {
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;

        let mut stack = vec![(start_x, start_y)];
        visited[start_y * width + start_x] = true;

        while let Some((x, y)) = stack.pop() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            // Check 4-connected neighbors
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nidx = ny * width + nx;

                    if !visited[nidx] && edges[nidx] >= self.config.edge_threshold {
                        visited[nidx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
        }

        Rect::new(
            min_x as f32,
            min_y as f32,
            (max_x - min_x + 1) as f32,
            (max_y - min_y + 1) as f32,
        )
    }

    /// Filter candidates based on text-like properties.
    fn filter_candidates(&self, candidates: &[Rect], width: usize, height: usize) -> Vec<Rect> {
        candidates
            .iter()
            .filter(|bbox| {
                // Text regions typically have certain aspect ratios
                let aspect_ratio = bbox.width / bbox.height;

                // Horizontal text: wider than tall
                let is_horizontal = aspect_ratio > 1.5 && aspect_ratio < 20.0;

                // Vertical text: taller than wide
                let is_vertical = aspect_ratio < 0.67 && aspect_ratio > 0.05;

                // Not too large (not the whole image)
                let size_ok = bbox.area() < (width * height) as f32 * 0.5;

                (is_horizontal || is_vertical) && size_ok
            })
            .copied()
            .collect()
    }

    /// Merge nearby text regions.
    fn merge_nearby_regions(&self, regions: &[Rect]) -> Vec<Rect> {
        if regions.is_empty() {
            return Vec::new();
        }

        let mut merged = Vec::new();
        let mut used = vec![false; regions.len()];

        for i in 0..regions.len() {
            if used[i] {
                continue;
            }

            let mut current = regions[i];
            used[i] = true;

            // Try to merge with nearby regions
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..regions.len() {
                    if used[j] {
                        continue;
                    }

                    // Check if regions are close (within 20 pixels)
                    if self.are_regions_close(&current, &regions[j], 20.0) {
                        current = self.merge_rects(&current, &regions[j]);
                        used[j] = true;
                        changed = true;
                    }
                }
            }

            merged.push(current);
        }

        merged
    }

    /// Check if two regions are close.
    fn are_regions_close(&self, r1: &Rect, r2: &Rect, threshold: f32) -> bool {
        let dx = if r1.x + r1.width < r2.x {
            r2.x - (r1.x + r1.width)
        } else if r2.x + r2.width < r1.x {
            r1.x - (r2.x + r2.width)
        } else {
            0.0
        };

        let dy = if r1.y + r1.height < r2.y {
            r2.y - (r1.y + r1.height)
        } else if r2.y + r2.height < r1.y {
            r1.y - (r2.y + r2.height)
        } else {
            0.0
        };

        dx <= threshold && dy <= threshold
    }

    /// Merge two rectangles.
    fn merge_rects(&self, r1: &Rect, r2: &Rect) -> Rect {
        let min_x = r1.x.min(r2.x);
        let min_y = r1.y.min(r2.y);
        let max_x = (r1.x + r1.width).max(r2.x + r2.width);
        let max_y = (r1.y + r1.height).max(r2.y + r2.height);

        Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }

    /// Extract text properties.
    fn extract_properties(
        &self,
        rgb_data: &[u8],
        gray: &[u8],
        width: usize,
        _height: usize,
        bbox: &Rect,
    ) -> TextProperties {
        let aspect_ratio = bbox.width / bbox.height;

        let orientation = if aspect_ratio > 1.5 {
            TextOrientation::Horizontal
        } else if aspect_ratio < 0.67 {
            TextOrientation::Vertical
        } else {
            TextOrientation::Unknown
        };

        let size_category = if bbox.height < 20.0 {
            TextSizeCategory::Small
        } else if bbox.height < 50.0 {
            TextSizeCategory::Medium
        } else {
            TextSizeCategory::Large
        };

        // Calculate contrast in region
        let contrast = self.calculate_region_contrast(gray, width, bbox);

        // Calculate edge density
        let density = self.calculate_edge_density(rgb_data, width, bbox);

        TextProperties {
            orientation,
            size_category,
            contrast,
            density,
        }
    }

    /// Calculate contrast in region.
    fn calculate_region_contrast(&self, gray: &[u8], width: usize, bbox: &Rect) -> f32 {
        let x_start = bbox.x as usize;
        let y_start = bbox.y as usize;
        let x_end = (bbox.x + bbox.width) as usize;
        let y_end = (bbox.y + bbox.height) as usize;

        let mut min_val = 255u8;
        let mut max_val = 0u8;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = y * width + x;
                if idx < gray.len() {
                    min_val = min_val.min(gray[idx]);
                    max_val = max_val.max(gray[idx]);
                }
            }
        }

        (max_val - min_val) as f32 / 255.0
    }

    /// Calculate edge density in region.
    fn calculate_edge_density(&self, rgb_data: &[u8], width: usize, bbox: &Rect) -> f32 {
        let x_start = bbox.x as usize;
        let y_start = bbox.y as usize;
        let x_end = (bbox.x + bbox.width) as usize;
        let y_end = (bbox.y + bbox.height) as usize;

        let mut edge_pixels = 0;
        let mut total_pixels = 0;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() && x + 1 < x_end {
                    let idx_next = (y * width + (x + 1)) * 3;
                    let diff = ((rgb_data[idx] as i32 - rgb_data[idx_next] as i32).abs()
                        + (rgb_data[idx + 1] as i32 - rgb_data[idx_next + 1] as i32).abs()
                        + (rgb_data[idx + 2] as i32 - rgb_data[idx_next + 2] as i32).abs())
                        as u32;

                    if diff > 50 {
                        edge_pixels += 1;
                    }
                    total_pixels += 1;
                }
            }
        }

        if total_pixels > 0 {
            edge_pixels as f32 / total_pixels as f32
        } else {
            0.0
        }
    }

    /// Calculate detection confidence.
    fn calculate_confidence(&self, properties: &TextProperties) -> Confidence {
        let mut score = 0.0;

        // High contrast is good for text
        score += properties.contrast * 0.4;

        // Moderate edge density is typical for text
        if properties.density > 0.1 && properties.density < 0.7 {
            score += 0.4;
        }

        // Known orientation increases confidence
        if properties.orientation != TextOrientation::Unknown {
            score += 0.2;
        }

        Confidence::new(score)
    }
}

impl Default for TextDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_detector() {
        let detector = TextDetector::new();
        let width = 320;
        let height = 240;
        let rgb_data = vec![128u8; width * height * 3];

        let result = detector.detect(&rgb_data, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_detection() {
        let detector = TextDetector::new();
        let gray = vec![128u8; 100 * 100];
        let edges = detector.detect_edges(&gray, 100, 100);
        assert_eq!(edges.len(), 10000);
    }

    #[test]
    fn test_merge_rects() {
        let detector = TextDetector::new();
        let r1 = Rect::new(10.0, 10.0, 20.0, 10.0);
        let r2 = Rect::new(15.0, 15.0, 20.0, 10.0);
        let merged = detector.merge_rects(&r1, &r2);

        assert_eq!(merged.x, 10.0);
        assert_eq!(merged.y, 10.0);
        assert_eq!(merged.width, 25.0);
        assert_eq!(merged.height, 15.0);
    }
}
