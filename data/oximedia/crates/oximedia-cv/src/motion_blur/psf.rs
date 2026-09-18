//! Point Spread Function (PSF) estimation and generation.
//!
//! This module provides algorithms for estimating and generating motion blur PSFs.

use super::MotionVector;
use crate::error::{CvError, CvResult};

/// Point Spread Function (PSF) for motion blur.
///
/// Represents the blur kernel that describes how a point of light is spread
/// due to motion during exposure.
#[derive(Debug, Clone)]
pub struct MotionPSF {
    /// PSF kernel data.
    pub kernel: Vec<f32>,
    /// Kernel width.
    pub width: usize,
    /// Kernel height.
    pub height: usize,
    /// Center X coordinate.
    pub center_x: usize,
    /// Center Y coordinate.
    pub center_y: usize,
}

impl MotionPSF {
    /// Create a new PSF with given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            kernel: vec![0.0; width * height],
            width,
            height,
            center_x: width / 2,
            center_y: height / 2,
        }
    }

    /// Create a linear motion PSF from a motion vector.
    #[must_use]
    pub fn from_motion_vector(motion: MotionVector, kernel_size: usize) -> Self {
        let mut psf = Self::new(kernel_size, kernel_size);

        let length = motion.magnitude();
        if length < 0.5 {
            // Delta function for no motion
            psf.kernel[psf.center_y * kernel_size + psf.center_x] = 1.0;
            return psf;
        }

        let angle = motion.direction();
        let dx = angle.cos();
        let dy = angle.sin();

        let samples = (length * 2.0).ceil() as usize + 1;
        let weight = 1.0 / samples as f32;

        for i in 0..samples {
            let t = i as f32 / samples as f32 - 0.5;
            let x = psf.center_x as f32 + motion.dx * t;
            let y = psf.center_y as f32 + motion.dy * t;

            psf.add_point(x, y, weight);
        }

        psf.normalize();
        psf
    }

    /// Add a point to the PSF with bilinear distribution.
    fn add_point(&mut self, x: f32, y: f32, weight: f32) {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        for dy in 0..=1 {
            for dx in 0..=1 {
                let xi = x0 + dx;
                let yi = y0 + dy;

                if xi >= 0 && xi < self.width as i32 && yi >= 0 && yi < self.height as i32 {
                    let w =
                        if dx == 0 { 1.0 - fx } else { fx } * if dy == 0 { 1.0 - fy } else { fy };

                    let idx = yi as usize * self.width + xi as usize;
                    if idx < self.kernel.len() {
                        self.kernel[idx] += weight * w;
                    }
                }
            }
        }
    }

    /// Normalize the PSF so it sums to 1.
    pub fn normalize(&mut self) {
        let sum: f32 = self.kernel.iter().sum();
        if sum > f32::EPSILON {
            for val in &mut self.kernel {
                *val /= sum;
            }
        }
    }

    /// Get PSF value at a specific position.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y * self.width + x;
        if idx < self.kernel.len() {
            self.kernel[idx]
        } else {
            0.0
        }
    }

    /// Set PSF value at a specific position.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = y * self.width + x;
        if idx < self.kernel.len() {
            self.kernel[idx] = value;
        }
    }

    /// Create a Gaussian-shaped PSF for uniform blur.
    #[must_use]
    pub fn gaussian(size: usize, sigma: f32) -> Self {
        let mut psf = Self::new(size, size);
        // Use integer center to ensure maximum is exactly at center pixel
        let center = (size / 2) as f32;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist_sq = dx * dx + dy * dy;
                let value = (-dist_sq / (2.0 * sigma * sigma)).exp();
                psf.set(x, y, value);
            }
        }

        psf.normalize();
        psf
    }

    /// Apply the PSF to an image channel.
    pub fn apply_to_channel(&self, image: &[f32], width: u32, height: u32) -> Vec<f32> {
        let mut output = vec![0.0; image.len()];

        let half_w = self.width as i32 / 2;
        let half_h = self.height as i32 / 2;

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut sum = 0.0;

                for ky in 0..self.height as i32 {
                    for kx in 0..self.width as i32 {
                        let ix = x + kx - half_w;
                        let iy = y + ky - half_h;

                        if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                            let img_idx = (iy * width as i32 + ix) as usize;
                            let kernel_idx = ky as usize * self.width + kx as usize;

                            if img_idx < image.len() && kernel_idx < self.kernel.len() {
                                sum += image[img_idx] * self.kernel[kernel_idx];
                            }
                        }
                    }
                }

                let out_idx = (y * width as i32 + x) as usize;
                if out_idx < output.len() {
                    output[out_idx] = sum;
                }
            }
        }

        output
    }
}

/// PSF shape types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PSFShape {
    /// Linear motion blur.
    #[default]
    Linear,
    /// Gaussian blur.
    Gaussian,
    /// Box blur.
    Box,
    /// Circular/disk blur.
    Disk,
    /// Custom arbitrary shape.
    Custom,
}

/// PSF estimator for blind deconvolution.
pub struct PSFEstimator {
    /// Maximum PSF size to estimate.
    max_size: usize,
    /// Number of iterations for estimation.
    iterations: usize,
    /// PSF shape constraint.
    shape: PSFShape,
}

impl PSFEstimator {
    /// Create a new PSF estimator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_size: 31,
            iterations: 50,
            shape: PSFShape::Linear,
        }
    }

    /// Set maximum PSF size.
    #[must_use]
    pub const fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Set number of iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set PSF shape constraint.
    #[must_use]
    pub const fn with_shape(mut self, shape: PSFShape) -> Self {
        self.shape = shape;
        self
    }

    /// Estimate PSF from a blurred image.
    ///
    /// Uses blind deconvolution to estimate the blur kernel.
    pub fn estimate_from_image(
        &self,
        blurred: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<MotionPSF> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if blurred.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, blurred.len()));
        }

        // Convert to grayscale float
        let gray = to_grayscale_float(blurred, width, height);

        // Initialize PSF (start with delta function)
        let mut psf = MotionPSF::new(self.max_size, self.max_size);
        psf.kernel[psf.center_y * self.max_size + psf.center_x] = 1.0;

        // Iterative blind deconvolution
        for _iter in 0..self.iterations {
            // Estimate sharp image given current PSF
            let sharp = self.estimate_sharp_image(&gray, width, height, &psf)?;

            // Estimate PSF given sharp image
            psf = self.estimate_psf_from_sharp(&gray, &sharp, width, height)?;

            // Apply shape constraints
            self.apply_shape_constraint(&mut psf);

            psf.normalize();
        }

        Ok(psf)
    }

    /// Estimate PSF from a pair of blurred and sharp images.
    pub fn estimate_from_pair(
        &self,
        blurred: &[u8],
        sharp: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<MotionPSF> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if blurred.len() < expected_size || sharp.len() < expected_size {
            return Err(CvError::insufficient_data(
                expected_size,
                blurred.len().min(sharp.len()),
            ));
        }

        let blurred_gray = to_grayscale_float(blurred, width, height);
        let sharp_gray = to_grayscale_float(sharp, width, height);

        self.estimate_psf_from_sharp(&blurred_gray, &sharp_gray, width, height)
    }

    /// Estimate sharp image using current PSF.
    fn estimate_sharp_image(
        &self,
        blurred: &[f32],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<f32>> {
        // Simple Wiener deconvolution for estimation
        let noise_level = 0.01;
        let mut sharp = vec![0.0; blurred.len()];

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                let half_w = psf.width as i32 / 2;
                let half_h = psf.height as i32 / 2;

                for ky in 0..psf.height as i32 {
                    for kx in 0..psf.width as i32 {
                        let ix = x as i32 + kx - half_w;
                        let iy = y as i32 + ky - half_h;

                        if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                            let idx = (iy as u32 * width + ix as u32) as usize;
                            let kernel_idx = ky as usize * psf.width + kx as usize;

                            if idx < blurred.len() && kernel_idx < psf.kernel.len() {
                                let kernel_val = psf.kernel[kernel_idx];
                                let weight = kernel_val / (kernel_val * kernel_val + noise_level);
                                sum += blurred[idx] * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                }

                let out_idx = (y * width + x) as usize;
                if out_idx < sharp.len() && weight_sum > 0.0 {
                    sharp[out_idx] = sum / weight_sum;
                }
            }
        }

        Ok(sharp)
    }

    /// Estimate PSF from sharp and blurred images.
    fn estimate_psf_from_sharp(
        &self,
        blurred: &[f32],
        sharp: &[f32],
        width: u32,
        height: u32,
    ) -> CvResult<MotionPSF> {
        let mut psf = MotionPSF::new(self.max_size, self.max_size);

        // Accumulate PSF estimates from local patches
        for y in (psf.height / 2) as u32..(height - psf.height as u32 / 2) {
            for x in (psf.width / 2) as u32..(width - psf.width as u32 / 2) {
                self.accumulate_psf_patch(&mut psf, blurred, sharp, width, height, x, y);
            }
        }

        psf.normalize();
        Ok(psf)
    }

    /// Accumulate PSF estimate from a local patch.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_psf_patch(
        &self,
        psf: &mut MotionPSF,
        blurred: &[f32],
        sharp: &[f32],
        width: u32,
        height: u32,
        cx: u32,
        cy: u32,
    ) {
        let half_w = psf.width as i32 / 2;
        let half_h = psf.height as i32 / 2;

        let center_idx = (cy * width + cx) as usize;
        if center_idx >= sharp.len() || center_idx >= blurred.len() {
            return;
        }

        let sharp_val = sharp[center_idx];
        if sharp_val < 0.01 {
            return;
        }

        for ky in 0..psf.height as i32 {
            for kx in 0..psf.width as i32 {
                let ix = cx as i32 + kx - half_w;
                let iy = cy as i32 + ky - half_h;

                if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                    let idx = (iy as u32 * width + ix as u32) as usize;
                    if idx < blurred.len() {
                        let contribution = blurred[idx] / sharp_val;
                        let kernel_idx = ky as usize * psf.width + kx as usize;
                        if kernel_idx < psf.kernel.len() {
                            psf.kernel[kernel_idx] += contribution;
                        }
                    }
                }
            }
        }
    }

    /// Apply shape constraints to the PSF.
    fn apply_shape_constraint(&self, psf: &mut MotionPSF) {
        match self.shape {
            PSFShape::Linear => self.apply_linear_constraint(psf),
            PSFShape::Gaussian => self.apply_gaussian_constraint(psf),
            PSFShape::Box => self.apply_box_constraint(psf),
            PSFShape::Disk => self.apply_disk_constraint(psf),
            PSFShape::Custom => {}
        }
    }

    /// Apply linear motion constraint.
    fn apply_linear_constraint(&self, psf: &mut MotionPSF) {
        // Find dominant direction
        let (angle, length) = self.find_dominant_direction(psf);

        // Reconstruct PSF along that direction
        let mut new_kernel = vec![0.0; psf.kernel.len()];
        let dx = angle.cos();
        let dy = angle.sin();

        let samples = (length * 2.0).ceil() as usize + 1;
        let weight = 1.0 / samples as f32;

        for i in 0..samples {
            let t = i as f32 / samples as f32 - 0.5;
            let x = psf.center_x as f32 + length * dx * t;
            let y = psf.center_y as f32 + length * dy * t;

            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;
            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            for dy_i in 0..=1 {
                for dx_i in 0..=1 {
                    let xi = x0 + dx_i;
                    let yi = y0 + dy_i;

                    if xi >= 0 && xi < psf.width as i32 && yi >= 0 && yi < psf.height as i32 {
                        let w = if dx_i == 0 { 1.0 - fx } else { fx }
                            * if dy_i == 0 { 1.0 - fy } else { fy };

                        let idx = yi as usize * psf.width + xi as usize;
                        if idx < new_kernel.len() {
                            new_kernel[idx] += weight * w;
                        }
                    }
                }
            }
        }

        psf.kernel = new_kernel;
    }

    /// Find dominant direction in PSF.
    fn find_dominant_direction(&self, psf: &MotionPSF) -> (f32, f32) {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut total_weight = 0.0;

        for y in 0..psf.height {
            for x in 0..psf.width {
                let weight = psf.get(x, y);
                let dx = x as f32 - psf.center_x as f32;
                let dy = y as f32 - psf.center_y as f32;

                sum_x += dx * weight;
                sum_y += dy * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            sum_x /= total_weight;
            sum_y /= total_weight;
        }

        let length = (sum_x * sum_x + sum_y * sum_y).sqrt();
        let angle = sum_y.atan2(sum_x);

        (angle, length)
    }

    /// Apply Gaussian constraint.
    fn apply_gaussian_constraint(&self, psf: &mut MotionPSF) {
        // Estimate Gaussian parameters and refit
        let sigma = self.estimate_gaussian_sigma(psf);
        let gaussian = MotionPSF::gaussian(psf.width, sigma);
        psf.kernel = gaussian.kernel;
    }

    /// Estimate Gaussian sigma from PSF.
    fn estimate_gaussian_sigma(&self, psf: &MotionPSF) -> f32 {
        let mut variance = 0.0;
        let mut total_weight = 0.0;

        for y in 0..psf.height {
            for x in 0..psf.width {
                let dx = x as f32 - psf.center_x as f32;
                let dy = y as f32 - psf.center_y as f32;
                let dist_sq = dx * dx + dy * dy;
                let weight = psf.get(x, y);

                variance += dist_sq * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            variance /= total_weight;
        }

        variance.sqrt()
    }

    /// Apply box constraint.
    fn apply_box_constraint(&self, psf: &mut MotionPSF) {
        // Threshold and normalize
        let threshold = psf.kernel.iter().copied().fold(0.0f32, f32::max) * 0.1;
        for val in &mut psf.kernel {
            if *val < threshold {
                *val = 0.0;
            }
        }
    }

    /// Apply disk constraint.
    fn apply_disk_constraint(&self, psf: &mut MotionPSF) {
        // Estimate radius and create disk PSF
        let radius = self.estimate_disk_radius(psf);
        let center = psf.center_x as f32;

        for y in 0..psf.height {
            for x in 0..psf.width {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= radius {
                    psf.set(x, y, 1.0);
                } else {
                    psf.set(x, y, 0.0);
                }
            }
        }
    }

    /// Estimate disk radius from PSF.
    fn estimate_disk_radius(&self, psf: &MotionPSF) -> f32 {
        let mut max_dist: f32 = 0.0;
        let threshold = psf.kernel.iter().copied().fold(0.0f32, f32::max) * 0.1;

        for y in 0..psf.height {
            for x in 0..psf.width {
                if psf.get(x, y) > threshold {
                    let dx = x as f32 - psf.center_x as f32;
                    let dy = y as f32 - psf.center_y as f32;
                    let dist = (dx * dx + dy * dy).sqrt();
                    max_dist = max_dist.max(dist);
                }
            }
        }

        max_dist
    }
}

impl Default for PSFEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert RGB/grayscale image to float grayscale.
fn to_grayscale_float(image: &[u8], width: u32, height: u32) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut gray = vec![0.0; size];

    if image.len() >= size * 3 {
        // RGB input
        for i in 0..size {
            let r = image[i * 3] as f32;
            let g = image[i * 3 + 1] as f32;
            let b = image[i * 3 + 2] as f32;
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        }
    } else if image.len() >= size {
        // Grayscale input
        for i in 0..size {
            gray[i] = image[i] as f32 / 255.0;
        }
    }

    gray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_psf_new() {
        let psf = MotionPSF::new(15, 15);
        assert_eq!(psf.width, 15);
        assert_eq!(psf.height, 15);
        assert_eq!(psf.kernel.len(), 225);
    }

    #[test]
    fn test_motion_psf_from_vector() {
        let motion = MotionVector::new(10.0, 0.0);
        let psf = MotionPSF::from_motion_vector(motion, 21);
        assert_eq!(psf.width, 21);

        let sum: f32 = psf.kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_motion_psf_gaussian() {
        let psf = MotionPSF::gaussian(15, 2.0);
        let sum: f32 = psf.kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Center pixel (7,7) should have the highest value in the kernel
        let center_val = psf.get(7, 7);
        let max_val = psf.kernel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (center_val - max_val).abs() < 1e-6,
            "Center should be the maximum value"
        );
        // For sigma=2 over 15x15, center is ~1/sum ≈ 0.04; just verify it's positive
        assert!(center_val > 0.0, "Center value should be positive");
    }

    #[test]
    fn test_psf_estimator_new() {
        let estimator = PSFEstimator::new();
        assert_eq!(estimator.max_size, 31);
        assert_eq!(estimator.iterations, 50);
    }

    #[test]
    fn test_psf_normalize() {
        let mut psf = MotionPSF::new(5, 5);
        psf.kernel[12] = 2.0;
        psf.normalize();

        let sum: f32 = psf.kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}
