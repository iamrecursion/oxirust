//! Object protection masks for content-aware scaling.
//!
//! This module provides functionality to create protection masks that
//! prevent important regions (faces, objects, etc.) from being removed
//! during seam carving.

use super::saliency::{SaliencyMap, SaliencyMethod, SalientRegion};
use crate::error::{CvError, CvResult};

/// Protection mask for content-aware scaling.
///
/// Protected regions have high values (255) and will be avoided
/// during seam removal.
#[derive(Debug, Clone)]
pub struct ProtectionMask {
    /// Mask data (0 = unprotected, 255 = fully protected).
    pub data: Vec<u8>,
    /// Mask width.
    pub width: u32,
    /// Mask height.
    pub height: u32,
}

impl ProtectionMask {
    /// Create a new empty protection mask.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            data: vec![0u8; size],
            width,
            height,
        }
    }

    /// Create from raw data.
    pub fn from_data(data: Vec<u8>, width: u32, height: u32) -> CvResult<Self> {
        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(CvError::insufficient_data(expected, data.len()));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Get protection value at position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Set protection value at position.
    pub fn set(&mut self, x: u32, y: u32, value: u8) {
        if x < self.width && y < self.height {
            self.data[y as usize * self.width as usize + x as usize] = value;
        }
    }

    /// Protect a rectangular region.
    ///
    /// # Arguments
    ///
    /// * `x` - Region x coordinate
    /// * `y` - Region y coordinate
    /// * `width` - Region width
    /// * `height` - Region height
    /// * `value` - Protection value (0-255)
    pub fn protect_rect(&mut self, x: u32, y: u32, width: u32, height: u32, value: u8) {
        let x1 = x.min(self.width);
        let y1 = y.min(self.height);
        let x2 = (x + width).min(self.width);
        let y2 = (y + height).min(self.height);

        for py in y1..y2 {
            for px in x1..x2 {
                self.set(px, py, value);
            }
        }
    }

    /// Protect a circular region.
    ///
    /// # Arguments
    ///
    /// * `cx` - Circle center x
    /// * `cy` - Circle center y
    /// * `radius` - Circle radius
    /// * `value` - Protection value (0-255)
    pub fn protect_circle(&mut self, cx: u32, cy: u32, radius: u32, value: u8) {
        let r_sq = radius * radius;

        let min_x = cx.saturating_sub(radius);
        let max_x = (cx + radius).min(self.width);
        let min_y = cy.saturating_sub(radius);
        let max_y = (cy + radius).min(self.height);

        for y in min_y..max_y {
            for x in min_x..max_x {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let dist_sq = (dx * dx + dy * dy) as u32;

                if dist_sq <= r_sq {
                    self.set(x, y, value);
                }
            }
        }
    }

    /// Protect an elliptical region.
    ///
    /// # Arguments
    ///
    /// * `cx` - Ellipse center x
    /// * `cy` - Ellipse center y
    /// * `rx` - X-axis radius
    /// * `ry` - Y-axis radius
    /// * `value` - Protection value (0-255)
    pub fn protect_ellipse(&mut self, cx: u32, cy: u32, rx: u32, ry: u32, value: u8) {
        let min_x = cx.saturating_sub(rx);
        let max_x = (cx + rx).min(self.width);
        let min_y = cy.saturating_sub(ry);
        let max_y = (cy + ry).min(self.height);

        for y in min_y..max_y {
            for x in min_x..max_x {
                let dx = (x as i32 - cx as i32) as f64;
                let dy = (y as i32 - cy as i32) as f64;

                // Ellipse equation: (dx/rx)^2 + (dy/ry)^2 <= 1
                let norm = (dx * dx) / (rx * rx) as f64 + (dy * dy) / (ry * ry) as f64;

                if norm <= 1.0 {
                    self.set(x, y, value);
                }
            }
        }
    }

    /// Dilate the protection mask to expand protected regions.
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of dilation iterations
    pub fn dilate(&mut self, iterations: u32) {
        for _ in 0..iterations {
            self.dilate_once();
        }
    }

    /// Single dilation iteration.
    fn dilate_once(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut new_data = self.data.clone();

        for y in 0..h {
            for x in 0..w {
                let mut max_val = self.data[y * w + x];

                // Check 8-connected neighbors
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            max_val = max_val.max(self.data[ny as usize * w + nx as usize]);
                        }
                    }
                }

                new_data[y * w + x] = max_val;
            }
        }

        self.data = new_data;
    }

    /// Erode the protection mask to shrink protected regions.
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of erosion iterations
    pub fn erode(&mut self, iterations: u32) {
        for _ in 0..iterations {
            self.erode_once();
        }
    }

    /// Single erosion iteration.
    fn erode_once(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut new_data = self.data.clone();

        for y in 0..h {
            for x in 0..w {
                let mut min_val = self.data[y * w + x];

                // Check 8-connected neighbors
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            min_val = min_val.min(self.data[ny as usize * w + nx as usize]);
                        }
                    }
                }

                new_data[y * w + x] = min_val;
            }
        }

        self.data = new_data;
    }

    /// Combine with another protection mask using maximum.
    pub fn merge(&mut self, other: &Self) {
        let size = self.data.len().min(other.data.len());
        for i in 0..size {
            self.data[i] = self.data[i].max(other.data[i]);
        }
    }

    /// Apply Gaussian blur to smooth the protection mask.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Gaussian blur sigma
    pub fn blur(&mut self, sigma: f64) {
        let kernel_size = (sigma * 3.0).ceil() as usize * 2 + 1;
        let kernel = create_gaussian_kernel(sigma, kernel_size);

        let blurred = separable_blur(
            &self.data,
            self.width as usize,
            self.height as usize,
            &kernel,
        );
        self.data = blurred;
    }

    /// Create protection mask from face regions.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `faces` - Face bounding boxes: (x, y, width, height)
    /// * `padding` - Padding around face boxes
    ///
    /// # Returns
    ///
    /// Protection mask with faces protected.
    #[must_use]
    pub fn from_faces(
        width: u32,
        height: u32,
        faces: &[(u32, u32, u32, u32)],
        padding: u32,
    ) -> Self {
        let mut mask = Self::new(width, height);

        for &(fx, fy, fw, fh) in faces {
            // Add padding
            let x = fx.saturating_sub(padding);
            let y = fy.saturating_sub(padding);
            let w = fw + 2 * padding;
            let h = fh + 2 * padding;

            mask.protect_rect(x, y, w, h, 255);
        }

        mask
    }

    /// Create protection mask from saliency map.
    ///
    /// # Arguments
    ///
    /// * `saliency` - Saliency map
    /// * `threshold` - Saliency threshold (0-255)
    ///
    /// # Returns
    ///
    /// Protection mask based on salient regions.
    #[must_use]
    pub fn from_saliency(saliency: &SaliencyMap, threshold: u8) -> Self {
        let binary = saliency.threshold(threshold);
        Self {
            data: binary,
            width: saliency.width,
            height: saliency.height,
        }
    }

    /// Create protection mask from salient regions.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `regions` - Salient regions
    /// * `padding` - Padding around regions
    ///
    /// # Returns
    ///
    /// Protection mask with regions protected.
    #[must_use]
    pub fn from_regions(width: u32, height: u32, regions: &[SalientRegion], padding: u32) -> Self {
        let mut mask = Self::new(width, height);

        for region in regions {
            let (rx, ry, rw, rh) = region.bbox;
            let x = rx.saturating_sub(padding);
            let y = ry.saturating_sub(padding);
            let w = rw + 2 * padding;
            let h = rh + 2 * padding;

            // Protection value based on region saliency
            let value = (region.avg_saliency * 255.0).round() as u8;
            mask.protect_rect(x, y, w, h, value);
        }

        mask
    }

    /// Create protection mask from image gradient.
    ///
    /// High-gradient regions are protected.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `threshold` - Gradient threshold
    ///
    /// # Returns
    ///
    /// Protection mask based on gradients.
    pub fn from_gradient(image: &[u8], width: u32, height: u32, threshold: f64) -> CvResult<Self> {
        use crate::image::SobelEdge;

        let sobel = SobelEdge::new();
        let (magnitude, _) = sobel.gradient_with_direction(image, width, height)?;

        let mut mask_data = vec![0u8; magnitude.len()];
        for (i, &mag) in magnitude.iter().enumerate() {
            if mag > threshold {
                mask_data[i] = ((mag / threshold) * 255.0).min(255.0) as u8;
            }
        }

        Self::from_data(mask_data, width, height)
    }

    /// Invert the protection mask.
    pub fn invert(&mut self) {
        for value in &mut self.data {
            *value = 255 - *value;
        }
    }

    /// Scale protection values.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factor (1.0 = no change)
    pub fn scale_values(&mut self, scale: f64) {
        for value in &mut self.data {
            let scaled = (*value as f64 * scale).round().clamp(0.0, 255.0) as u8;
            *value = scaled;
        }
    }
}

/// Create a 1D Gaussian kernel.
fn create_gaussian_kernel(sigma: f64, size: usize) -> Vec<f64> {
    let half = size / 2;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..size {
        let x = i as f64 - half as f64;
        let value = (-x * x / two_sigma_sq).exp();
        kernel.push(value);
        sum += value;
    }

    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

/// Apply separable Gaussian blur.
fn separable_blur(image: &[u8], width: usize, height: usize, kernel: &[f64]) -> Vec<u8> {
    let temp = blur_horizontal(image, width, height, kernel);
    blur_vertical(&temp, width, height, kernel)
}

/// Horizontal blur pass.
fn blur_horizontal(image: &[u8], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as i32 + i as i32 - half as i32).clamp(0, width as i32 - 1) as usize;
                sum += image[y * width + xi] as f64 * k;
            }
            result[y * width + x] = sum;
        }
    }

    result
}

/// Vertical blur pass.
fn blur_vertical(image: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<u8> {
    let half = kernel.len() / 2;
    let mut result = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as i32 + i as i32 - half as i32).clamp(0, height as i32 - 1) as usize;
                sum += image[yi * width + x] * k;
            }
            result[y * width + x] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }

    result
}

/// Builder for creating complex protection masks.
#[derive(Debug)]
pub struct ProtectionMaskBuilder {
    width: u32,
    height: u32,
    faces: Vec<(u32, u32, u32, u32)>,
    regions: Vec<SalientRegion>,
    saliency_threshold: Option<u8>,
    gradient_threshold: Option<f64>,
    padding: u32,
    blur_sigma: Option<f64>,
    dilation: u32,
}

impl ProtectionMaskBuilder {
    /// Create a new protection mask builder.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            faces: Vec::new(),
            regions: Vec::new(),
            saliency_threshold: None,
            gradient_threshold: None,
            padding: 0,
            blur_sigma: None,
            dilation: 0,
        }
    }

    /// Add face regions to protect.
    #[must_use]
    pub fn with_faces(mut self, faces: Vec<(u32, u32, u32, u32)>) -> Self {
        self.faces = faces;
        self
    }

    /// Add salient regions to protect.
    #[must_use]
    pub fn with_regions(mut self, regions: Vec<SalientRegion>) -> Self {
        self.regions = regions;
        self
    }

    /// Set saliency threshold.
    #[must_use]
    pub const fn with_saliency_threshold(mut self, threshold: u8) -> Self {
        self.saliency_threshold = Some(threshold);
        self
    }

    /// Set gradient threshold.
    #[must_use]
    pub const fn with_gradient_threshold(mut self, threshold: f64) -> Self {
        self.gradient_threshold = Some(threshold);
        self
    }

    /// Set padding around protected regions.
    #[must_use]
    pub const fn with_padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    /// Set blur sigma for smoothing.
    #[must_use]
    pub const fn with_blur(mut self, sigma: f64) -> Self {
        self.blur_sigma = Some(sigma);
        self
    }

    /// Set dilation iterations.
    #[must_use]
    pub const fn with_dilation(mut self, iterations: u32) -> Self {
        self.dilation = iterations;
        self
    }

    /// Build the protection mask.
    #[must_use]
    pub fn build(self) -> ProtectionMask {
        let mut mask = ProtectionMask::new(self.width, self.height);

        // Add face protections
        if !self.faces.is_empty() {
            let face_mask =
                ProtectionMask::from_faces(self.width, self.height, &self.faces, self.padding);
            mask.merge(&face_mask);
        }

        // Add region protections
        if !self.regions.is_empty() {
            let region_mask =
                ProtectionMask::from_regions(self.width, self.height, &self.regions, self.padding);
            mask.merge(&region_mask);
        }

        // Apply dilation if specified
        if self.dilation > 0 {
            mask.dilate(self.dilation);
        }

        // Apply blur if specified
        if let Some(sigma) = self.blur_sigma {
            mask.blur(sigma);
        }

        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protection_mask_new() {
        let mask = ProtectionMask::new(10, 10);
        assert_eq!(mask.width, 10);
        assert_eq!(mask.height, 10);
        assert_eq!(mask.data.len(), 100);
    }

    #[test]
    fn test_protect_rect() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.protect_rect(2, 2, 4, 4, 255);
        assert_eq!(mask.get(3, 3), 255);
        assert_eq!(mask.get(0, 0), 0);
    }

    #[test]
    fn test_protect_circle() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.protect_circle(5, 5, 3, 255);
        assert_eq!(mask.get(5, 5), 255); // Center
        assert_eq!(mask.get(0, 0), 0); // Outside
    }

    #[test]
    fn test_protect_ellipse() {
        let mut mask = ProtectionMask::new(20, 20);
        mask.protect_ellipse(10, 10, 5, 3, 255);
        assert_eq!(mask.get(10, 10), 255); // Center
    }

    #[test]
    fn test_dilate() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.set(5, 5, 255);
        mask.dilate(1);
        // Check that neighbors are now protected
        assert!(mask.get(4, 5) > 0);
        assert!(mask.get(6, 5) > 0);
    }

    #[test]
    fn test_erode() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.protect_rect(3, 3, 4, 4, 255);
        mask.erode(1);
        // Edges should be eroded
        let center_val = mask.get(5, 5);
        assert!(center_val > 0);
    }

    #[test]
    fn test_merge() {
        let mut mask1 = ProtectionMask::new(10, 10);
        let mut mask2 = ProtectionMask::new(10, 10);
        mask1.set(0, 0, 100);
        mask2.set(0, 0, 200);
        mask1.merge(&mask2);
        assert_eq!(mask1.get(0, 0), 200);
    }

    #[test]
    fn test_from_faces() {
        let faces = vec![(10, 10, 20, 20)];
        let mask = ProtectionMask::from_faces(100, 100, &faces, 5);
        assert!(mask.get(15, 15) > 0);
    }

    #[test]
    fn test_invert() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.data.fill(100);
        mask.invert();
        assert_eq!(mask.get(0, 0), 155);
    }

    #[test]
    fn test_scale_values() {
        let mut mask = ProtectionMask::new(10, 10);
        mask.data.fill(100);
        mask.scale_values(2.0);
        assert_eq!(mask.get(0, 0), 200);
    }

    #[test]
    fn test_builder() {
        let mask = ProtectionMaskBuilder::new(100, 100)
            .with_faces(vec![(10, 10, 20, 20)])
            .with_padding(5)
            .with_dilation(1)
            .build();

        assert!(mask.get(15, 15) > 0);
    }
}
