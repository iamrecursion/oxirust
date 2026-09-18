//! View stitching for panoramic multi-camera compositions.

use super::Transform2D;
use crate::{AngleId, Result};

/// View stitcher
#[derive(Debug)]
pub struct ViewStitcher {
    /// Transformations for each view
    transforms: Vec<Transform2D>,
    /// Overlap regions between adjacent views
    overlap_regions: Vec<OverlapRegion>,
    /// Blend mode
    blend_mode: BlendMode,
}

/// Overlap region between two views
#[derive(Debug, Clone, Copy)]
pub struct OverlapRegion {
    /// First angle
    pub angle_a: AngleId,
    /// Second angle
    pub angle_b: AngleId,
    /// Overlap start X (in stitched panorama coordinates)
    pub start_x: u32,
    /// Overlap end X
    pub end_x: u32,
    /// Overlap width
    pub width: u32,
}

/// Blend mode for overlapping regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Linear blend (cross-fade)
    Linear,
    /// Multiband blend (Laplacian pyramid)
    Multiband,
    /// Feather blend
    Feather,
    /// No blending (hard seam)
    None,
}

impl ViewStitcher {
    /// Create a new view stitcher
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            transforms: vec![Transform2D::identity(); angle_count],
            overlap_regions: Vec::new(),
            blend_mode: BlendMode::Linear,
        }
    }

    /// Set blend mode
    pub fn set_blend_mode(&mut self, mode: BlendMode) {
        self.blend_mode = mode;
    }

    /// Get blend mode
    #[must_use]
    pub fn blend_mode(&self) -> BlendMode {
        self.blend_mode
    }

    /// Set transformation for angle
    pub fn set_transform(&mut self, angle: AngleId, transform: Transform2D) {
        if angle < self.transforms.len() {
            self.transforms[angle] = transform;
        }
    }

    /// Get transformation for angle
    #[must_use]
    pub fn get_transform(&self, angle: AngleId) -> Option<&Transform2D> {
        self.transforms.get(angle)
    }

    /// Add overlap region
    pub fn add_overlap_region(&mut self, region: OverlapRegion) {
        self.overlap_regions.push(region);
    }

    /// Calculate panorama dimensions
    #[must_use]
    pub fn calculate_panorama_size(&self, view_width: u32, view_height: u32) -> (u32, u32) {
        if self.transforms.is_empty() {
            return (0, 0);
        }

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        // Transform corners of each view
        for transform in &self.transforms {
            let corners = [
                (0.0, 0.0),
                (view_width as f32, 0.0),
                (view_width as f32, view_height as f32),
                (0.0, view_height as f32),
            ];

            for (x, y) in corners {
                let (tx, ty) = transform.apply(x, y);
                min_x = min_x.min(tx);
                max_x = max_x.max(tx);
                min_y = min_y.min(ty);
                max_y = max_y.max(ty);
            }
        }

        let width = (max_x - min_x).ceil() as u32;
        let height = (max_y - min_y).ceil() as u32;
        (width, height)
    }

    /// Calculate blend weight at position
    #[must_use]
    pub fn calculate_blend_weight(&self, x: u32, region: &OverlapRegion) -> f32 {
        if x < region.start_x || x >= region.end_x {
            return 0.0;
        }

        let position = (x - region.start_x) as f32 / region.width as f32;

        match self.blend_mode {
            BlendMode::Linear => position,
            BlendMode::Feather => {
                // Smooth step function
                let t = position.clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            BlendMode::Multiband => {
                // Simplified multiband (would normally use pyramid)
                position
            }
            BlendMode::None => {
                if position < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    /// Find overlap region at position
    #[must_use]
    pub fn find_overlap_at(&self, x: u32) -> Option<&OverlapRegion> {
        self.overlap_regions
            .iter()
            .find(|r| x >= r.start_x && x < r.end_x)
    }

    /// Detect overlapping regions between views
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails
    pub fn detect_overlaps(&mut self, view_width: u32, overlap_threshold: u32) -> Result<()> {
        self.overlap_regions.clear();

        // Check adjacent views for overlap
        for i in 0..self.transforms.len().saturating_sub(1) {
            let transform_a = &self.transforms[i];
            let transform_b = &self.transforms[i + 1];

            // Transform right edge of view A
            let (right_a, _) = transform_a.apply(view_width as f32, 0.0);
            // Transform left edge of view B
            let (left_b, _) = transform_b.apply(0.0, 0.0);

            if right_a > left_b && (right_a - left_b) > overlap_threshold as f32 {
                let overlap_width = (right_a - left_b) as u32;
                let region = OverlapRegion {
                    angle_a: i,
                    angle_b: i + 1,
                    start_x: left_b as u32,
                    end_x: right_a as u32,
                    width: overlap_width,
                };
                self.overlap_regions.push(region);
            }
        }

        Ok(())
    }

    /// Get overlap regions
    #[must_use]
    pub fn overlap_regions(&self) -> &[OverlapRegion] {
        &self.overlap_regions
    }

    /// Clear overlap regions
    pub fn clear_overlaps(&mut self) {
        self.overlap_regions.clear();
    }
}

/// Cylindrical projection for panorama stitching
#[derive(Debug)]
pub struct CylindricalProjection {
    /// Focal length
    focal_length: f32,
}

impl CylindricalProjection {
    /// Create new cylindrical projection
    #[must_use]
    pub fn new(focal_length: f32) -> Self {
        Self { focal_length }
    }

    /// Project point to cylindrical coordinates
    #[must_use]
    pub fn project(&self, x: f32, y: f32, width: f32, height: f32) -> (f32, f32) {
        let cx = width / 2.0;
        let cy = height / 2.0;

        // Center the coordinates
        let xc = x - cx;
        let yc = y - cy;

        // Cylindrical projection
        let theta = xc / self.focal_length;
        let h = yc / (theta.cos() * self.focal_length);

        let x_proj = self.focal_length * theta;
        let y_proj = self.focal_length * h;

        (x_proj + cx, y_proj + cy)
    }

    /// Unproject from cylindrical coordinates
    #[must_use]
    pub fn unproject(&self, x: f32, y: f32, width: f32, height: f32) -> (f32, f32) {
        let cx = width / 2.0;
        let cy = height / 2.0;

        let xc = x - cx;
        let yc = y - cy;

        let theta = xc / self.focal_length;
        let h = yc / self.focal_length;

        let x_orig = self.focal_length * theta.tan();
        let y_orig = h * theta.cos() * self.focal_length;

        (x_orig + cx, y_orig + cy)
    }

    /// Set focal length
    pub fn set_focal_length(&mut self, focal_length: f32) {
        self.focal_length = focal_length;
    }

    /// Get focal length
    #[must_use]
    pub fn focal_length(&self) -> f32 {
        self.focal_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stitcher_creation() {
        let stitcher = ViewStitcher::new(3);
        assert_eq!(stitcher.transforms.len(), 3);
        assert_eq!(stitcher.blend_mode(), BlendMode::Linear);
    }

    #[test]
    fn test_set_blend_mode() {
        let mut stitcher = ViewStitcher::new(3);
        stitcher.set_blend_mode(BlendMode::Feather);
        assert_eq!(stitcher.blend_mode(), BlendMode::Feather);
    }

    #[test]
    fn test_add_overlap_region() {
        let mut stitcher = ViewStitcher::new(3);
        let region = OverlapRegion {
            angle_a: 0,
            angle_b: 1,
            start_x: 800,
            end_x: 1120,
            width: 320,
        };

        stitcher.add_overlap_region(region);
        assert_eq!(stitcher.overlap_regions().len(), 1);
    }

    #[test]
    fn test_calculate_blend_weight() {
        let stitcher = ViewStitcher::new(2);
        let region = OverlapRegion {
            angle_a: 0,
            angle_b: 1,
            start_x: 800,
            end_x: 1120,
            width: 320,
        };

        // At start of overlap
        let weight_start = stitcher.calculate_blend_weight(800, &region);
        assert_eq!(weight_start, 0.0);

        // At middle of overlap
        let weight_mid = stitcher.calculate_blend_weight(960, &region);
        assert!((weight_mid - 0.5).abs() < 0.01);

        // At end of overlap
        let weight_end = stitcher.calculate_blend_weight(1119, &region);
        assert!(weight_end > 0.9);
    }

    #[test]
    fn test_find_overlap_at() {
        let mut stitcher = ViewStitcher::new(2);
        let region = OverlapRegion {
            angle_a: 0,
            angle_b: 1,
            start_x: 800,
            end_x: 1120,
            width: 320,
        };
        stitcher.add_overlap_region(region);

        assert!(stitcher.find_overlap_at(900).is_some());
        assert!(stitcher.find_overlap_at(500).is_none());
    }

    #[test]
    fn test_calculate_panorama_size() {
        let mut stitcher = ViewStitcher::new(2);

        // First view at origin
        stitcher.set_transform(0, Transform2D::identity());

        // Second view shifted right
        stitcher.set_transform(
            1,
            Transform2D {
                tx: 800.0,
                ty: 0.0,
                ..Transform2D::identity()
            },
        );

        let (width, height) = stitcher.calculate_panorama_size(1920, 1080);
        assert!(width > 1920); // Should be wider than single view
        assert_eq!(height, 1080);
    }

    #[test]
    fn test_cylindrical_projection() {
        let projection = CylindricalProjection::new(1000.0);
        let (x_proj, y_proj) = projection.project(960.0, 540.0, 1920.0, 1080.0);

        // Center point should stay at center
        assert!((x_proj - 960.0).abs() < 1.0);
        assert!((y_proj - 540.0).abs() < 1.0);
    }

    #[test]
    fn test_projection_roundtrip() {
        let projection = CylindricalProjection::new(1000.0);
        let original = (1000.0, 500.0);

        let projected = projection.project(original.0, original.1, 1920.0, 1080.0);
        let unprojected = projection.unproject(projected.0, projected.1, 1920.0, 1080.0);

        assert!((unprojected.0 - original.0).abs() < 1.0);
        assert!((unprojected.1 - original.1).abs() < 1.0);
    }

    #[test]
    fn test_detect_overlaps() {
        let mut stitcher = ViewStitcher::new(2);

        stitcher.set_transform(0, Transform2D::identity());
        stitcher.set_transform(
            1,
            Transform2D {
                tx: 1600.0, // 320 pixel overlap with 1920 width
                ty: 0.0,
                ..Transform2D::identity()
            },
        );

        let result = stitcher.detect_overlaps(1920, 100);
        assert!(result.is_ok());
        assert!(!stitcher.overlap_regions().is_empty());
    }
}
