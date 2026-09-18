//! Feature extraction using HOG and other patent-free methods.

use crate::common::Point;
use crate::error::SceneResult;
use serde::{Deserialize, Serialize};

/// HOG (Histogram of Oriented Gradients) features.
pub struct HogFeatures {
    cell_size: usize,
    block_size: usize,
    num_bins: usize,
}

impl HogFeatures {
    /// Create a new HOG feature extractor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cell_size: 8,
            block_size: 2,
            num_bins: 9,
        }
    }

    /// Compute HOG features for an image.
    #[must_use]
    pub fn compute(&self, gray: &[f32], width: usize, height: usize) -> Vec<f32> {
        // Compute gradients
        let (grad_mag, grad_ang) = self.compute_gradients(gray, width, height);

        // Compute cell histograms
        let cell_histograms = self.compute_cell_histograms(&grad_mag, &grad_ang, width, height);

        // Normalize blocks
        self.normalize_blocks(&cell_histograms, width, height)
    }

    fn compute_gradients(&self, gray: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let mut grad_mag = vec![0.0; width * height];
        let mut grad_ang = vec![0.0; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;

                // Compute gradients using Sobel
                let gx = gray[idx + 1] - gray[idx - 1];
                let gy = gray[idx + width] - gray[idx - width];

                grad_mag[idx] = (gx * gx + gy * gy).sqrt();
                grad_ang[idx] = gy.atan2(gx);
            }
        }

        (grad_mag, grad_ang)
    }

    fn compute_cell_histograms(
        &self,
        grad_mag: &[f32],
        grad_ang: &[f32],
        width: usize,
        height: usize,
    ) -> Vec<Vec<f32>> {
        let cells_x = width / self.cell_size;
        let cells_y = height / self.cell_size;
        let mut histograms = vec![vec![0.0; self.num_bins]; cells_x * cells_y];

        for cy in 0..cells_y {
            for cx in 0..cells_x {
                let cell_idx = cy * cells_x + cx;

                for y in 0..self.cell_size {
                    for x in 0..self.cell_size {
                        let px = cx * self.cell_size + x;
                        let py = cy * self.cell_size + y;

                        if px < width && py < height {
                            let idx = py * width + px;
                            let magnitude = grad_mag[idx];
                            let angle = grad_ang[idx];

                            // Convert angle to bin (0 to num_bins)
                            let angle_deg = angle.to_degrees() + 180.0;
                            let bin = ((angle_deg / 360.0 * self.num_bins as f32) as usize)
                                .min(self.num_bins - 1);

                            histograms[cell_idx][bin] += magnitude;
                        }
                    }
                }
            }
        }

        histograms
    }

    fn normalize_blocks(
        &self,
        cell_histograms: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        let cells_x = width / self.cell_size;
        let cells_y = height / self.cell_size;
        let mut features = Vec::new();

        for by in 0..cells_y.saturating_sub(self.block_size - 1) {
            for bx in 0..cells_x.saturating_sub(self.block_size - 1) {
                let mut block_hist = Vec::new();
                let mut norm_sq = 0.0;

                // Collect block histogram
                for dy in 0..self.block_size {
                    for dx in 0..self.block_size {
                        let cell_idx = (by + dy) * cells_x + (bx + dx);
                        for &val in &cell_histograms[cell_idx] {
                            block_hist.push(val);
                            norm_sq += val * val;
                        }
                    }
                }

                // L2 normalization
                let norm = (norm_sq + 1e-6).sqrt();
                for val in &mut block_hist {
                    *val /= norm;
                }

                features.extend(block_hist);
            }
        }

        features
    }
}

impl Default for HogFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Local features (keypoints and descriptors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalFeatures {
    /// Keypoint locations.
    pub keypoints: Vec<Point>,
    /// Feature descriptors (one per keypoint).
    pub descriptors: Vec<Vec<f32>>,
}

/// Local feature extractor using Harris corners.
pub struct LocalFeatureExtractor;

impl LocalFeatureExtractor {
    /// Create a new local feature extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract local features.
    ///
    /// # Errors
    ///
    /// Returns error if extraction fails.
    pub fn extract(&self, gray: &[f32], width: usize, height: usize) -> SceneResult<LocalFeatures> {
        // Detect Harris corners
        let keypoints = self.detect_harris_corners(gray, width, height);

        // Compute descriptors for each keypoint
        let descriptors = keypoints
            .iter()
            .map(|kp| self.compute_descriptor(gray, width, height, kp))
            .collect();

        Ok(LocalFeatures {
            keypoints,
            descriptors,
        })
    }

    fn detect_harris_corners(&self, gray: &[f32], width: usize, height: usize) -> Vec<Point> {
        let mut keypoints = Vec::new();
        let window_size = 3;
        let k = 0.04;
        let threshold = 0.01;

        for y in window_size..height - window_size {
            for x in window_size..width - window_size {
                let _idx = y * width + x;

                // Compute structure tensor
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut ixy = 0.0;

                for dy in -(window_size as i32)..=window_size as i32 {
                    for dx in -(window_size as i32)..=window_size as i32 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx > 0 && nx < width as i32 - 1 && ny > 0 && ny < height as i32 - 1 {
                            let nidx = ny as usize * width + nx as usize;
                            let ix = (gray[nidx + 1] - gray[nidx - 1]) / 2.0;
                            let iy = (gray[nidx + width] - gray[nidx - width]) / 2.0;

                            ixx += ix * ix;
                            iyy += iy * iy;
                            ixy += ix * iy;
                        }
                    }
                }

                // Harris response
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - k * trace * trace;

                if response > threshold {
                    keypoints.push(Point::new(x as f32, y as f32));
                }
            }

            // Limit number of keypoints
            if keypoints.len() > 500 {
                break;
            }
        }

        keypoints
    }

    fn compute_descriptor(
        &self,
        gray: &[f32],
        width: usize,
        height: usize,
        kp: &Point,
    ) -> Vec<f32> {
        let patch_size = 8;
        let mut descriptor = Vec::new();

        let cx = kp.x as usize;
        let cy = kp.y as usize;

        for y in cy.saturating_sub(patch_size)..=(cy + patch_size).min(height - 1) {
            for x in cx.saturating_sub(patch_size)..=(cx + patch_size).min(width - 1) {
                descriptor.push(gray[y * width + x]);
            }
        }

        // Normalize
        let sum: f32 = descriptor.iter().sum();
        if sum > 0.0 {
            for d in &mut descriptor {
                *d /= sum;
            }
        }

        descriptor
    }
}

impl Default for LocalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hog_features() {
        let hog = HogFeatures::new();
        let gray = vec![0.5; 64 * 64];
        let features = hog.compute(&gray, 64, 64);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_local_features() {
        let extractor = LocalFeatureExtractor::new();
        let gray = vec![0.5; 100 * 100];
        let result = extractor.extract(&gray, 100, 100);
        assert!(result.is_ok());
    }
}
