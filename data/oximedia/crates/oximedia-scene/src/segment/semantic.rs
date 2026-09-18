//! Semantic segmentation using graph-based methods.

use crate::common::Rect;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Semantic region type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionType {
    /// Sky region.
    Sky,
    /// Ground/road region.
    Ground,
    /// Vegetation.
    Vegetation,
    /// Building/structure.
    Building,
    /// Water.
    Water,
    /// Person.
    Person,
    /// Unknown.
    Unknown,
}

/// Semantic region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegion {
    /// Region type.
    pub region_type: RegionType,
    /// Bounding box.
    pub bbox: Rect,
    /// Region mask indices.
    pub mask_indices: Vec<usize>,
    /// Average color.
    pub avg_color: [u8; 3],
}

/// Semantic segmenter using color-based clustering.
pub struct SemanticSegmenter {
    num_regions: usize,
}

impl SemanticSegmenter {
    /// Create a new semantic segmenter.
    #[must_use]
    pub fn new() -> Self {
        Self { num_regions: 10 }
    }

    /// Segment image into semantic regions.
    ///
    /// # Errors
    ///
    /// Returns error if segmentation fails.
    pub fn segment(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<SemanticRegion>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Simple region growing based on color similarity
        let mut labels = vec![0usize; width * height];
        let mut current_label = 1;
        let mut regions = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if labels[idx] == 0 {
                    let region =
                        self.grow_region(rgb_data, &mut labels, width, height, x, y, current_label);
                    if !region.mask_indices.is_empty() {
                        regions.push(region);
                        current_label += 1;
                    }
                }
            }
        }

        Ok(regions)
    }

    fn grow_region(
        &self,
        rgb_data: &[u8],
        labels: &mut [usize],
        width: usize,
        height: usize,
        start_x: usize,
        start_y: usize,
        label: usize,
    ) -> SemanticRegion {
        let threshold = 30;
        let start_idx = (start_y * width + start_x) * 3;
        let seed_color = [
            rgb_data[start_idx],
            rgb_data[start_idx + 1],
            rgb_data[start_idx + 2],
        ];

        let mut stack = vec![(start_x, start_y)];
        let mut mask_indices = Vec::new();
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut color_sum = [0u64; 3];

        labels[start_y * width + start_x] = label;

        while let Some((x, y)) = stack.pop() {
            mask_indices.push(y * width + x);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            let idx = (y * width + x) * 3;
            for c in 0..3 {
                color_sum[c] += u64::from(rgb_data[idx + c]);
            }

            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nidx = ny * width + nx;

                    if labels[nidx] == 0 {
                        let pidx = (ny * width + nx) * 3;
                        let color_diff = (0..3)
                            .map(|c| {
                                (rgb_data[pidx + c] as i32 - seed_color[c] as i32).unsigned_abs()
                            })
                            .sum::<u32>();

                        if color_diff < threshold * 3 {
                            labels[nidx] = label;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            // Limit region size
            if mask_indices.len() > 10000 {
                break;
            }
        }

        let avg_color = if mask_indices.is_empty() {
            [0, 0, 0]
        } else {
            [
                (color_sum[0] / mask_indices.len() as u64) as u8,
                (color_sum[1] / mask_indices.len() as u64) as u8,
                (color_sum[2] / mask_indices.len() as u64) as u8,
            ]
        };

        let region_type = self.classify_region(&avg_color, min_y, max_y, height);

        SemanticRegion {
            region_type,
            bbox: Rect::new(
                min_x as f32,
                min_y as f32,
                (max_x - min_x + 1) as f32,
                (max_y - min_y + 1) as f32,
            ),
            mask_indices,
            avg_color,
        }
    }

    fn classify_region(
        &self,
        color: &[u8; 3],
        min_y: usize,
        max_y: usize,
        height: usize,
    ) -> RegionType {
        let r = color[0];
        let g = color[1];
        let b = color[2];

        // Sky: blue, top of image
        if b > 150 && b > r && b > g && min_y < height / 3 {
            return RegionType::Sky;
        }

        // Vegetation: green dominant
        if g > r && g > b && g > 80 {
            return RegionType::Vegetation;
        }

        // Ground: bottom of image
        if max_y > height * 2 / 3 {
            return RegionType::Ground;
        }

        // Building: gray tones
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        if max > 0 && (max - min) < 40 {
            return RegionType::Building;
        }

        RegionType::Unknown
    }
}

impl Default for SemanticSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_segmenter() {
        let segmenter = SemanticSegmenter::new();
        let width = 100;
        let height = 100;
        let rgb_data = vec![128u8; width * height * 3];

        let result = segmenter.segment(&rgb_data, width, height);
        assert!(result.is_ok());
    }
}
