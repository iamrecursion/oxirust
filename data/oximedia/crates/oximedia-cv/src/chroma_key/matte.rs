//! Alpha matte generation and refinement.
//!
//! This module provides functionality for creating and refining alpha mattes,
//! which define the transparency of each pixel in chroma keying operations.

use crate::error::{CvError, CvResult};

/// Alpha matte representing pixel transparency.
///
/// Stores alpha values in the range [0.0, 1.0] where:
/// - 0.0 = fully transparent (keyed out)
/// - 1.0 = fully opaque (foreground)
#[derive(Debug, Clone)]
pub struct AlphaMatte {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl AlphaMatte {
    /// Create a new alpha matte.
    ///
    /// # Arguments
    ///
    /// * `width` - Matte width in pixels
    /// * `height` - Matte height in pixels
    /// * `data` - Alpha values (must be width * height in length)
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match width * height.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            (width * height) as usize,
            "Data length must match dimensions"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a solid matte (all pixels same alpha).
    #[must_use]
    pub fn solid(width: u32, height: u32, alpha: f32) -> Self {
        let data = vec![alpha.clamp(0.0, 1.0); (width * height) as usize];
        Self {
            width,
            height,
            data,
        }
    }

    /// Get matte width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get matte height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get alpha data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable alpha data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get alpha value at specific coordinates.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            Some(self.data[idx])
        } else {
            None
        }
    }

    /// Set alpha value at specific coordinates.
    pub fn set(&mut self, x: u32, y: u32, alpha: f32) -> bool {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] = alpha.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Invert the matte (swap foreground and background).
    #[must_use]
    pub fn invert(&self) -> Self {
        let inverted_data: Vec<f32> = self.data.iter().map(|&a| 1.0 - a).collect();
        Self {
            width: self.width,
            height: self.height,
            data: inverted_data,
        }
    }

    /// Multiply two mattes together.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn multiply(&self, other: &Self) -> CvResult<Self> {
        if self.width != other.width || self.height != other.height {
            return Err(CvError::invalid_parameter(
                "dimensions",
                format!(
                    "{}x{} != {}x{}",
                    self.width, self.height, other.width, other.height
                ),
            ));
        }

        let multiplied_data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Self {
            width: self.width,
            height: self.height,
            data: multiplied_data,
        })
    }

    /// Calculate statistics about the matte.
    #[must_use]
    pub fn statistics(&self) -> MatteStatistics {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;

        for &alpha in &self.data {
            min = min.min(alpha);
            max = max.max(alpha);
            sum += alpha;
        }

        let mean = sum / self.data.len() as f32;

        MatteStatistics {
            min,
            max,
            mean,
            total_pixels: self.data.len(),
            opaque_pixels: self.data.iter().filter(|&&a| a > 0.99).count(),
            transparent_pixels: self.data.iter().filter(|&&a| a < 0.01).count(),
        }
    }
}

/// Statistics about an alpha matte.
#[derive(Debug, Clone, Copy)]
pub struct MatteStatistics {
    /// Minimum alpha value.
    pub min: f32,
    /// Maximum alpha value.
    pub max: f32,
    /// Mean alpha value.
    pub mean: f32,
    /// Total number of pixels.
    pub total_pixels: usize,
    /// Number of fully opaque pixels.
    pub opaque_pixels: usize,
    /// Number of fully transparent pixels.
    pub transparent_pixels: usize,
}

/// Morphological operation for matte refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefineOperation {
    /// Erosion (shrink transparent regions).
    Erode,
    /// Dilation (expand transparent regions).
    Dilate,
    /// Blur (smooth edges).
    Blur,
    /// Edge feathering.
    Feather,
}

/// Matte refinement processor.
///
/// Provides operations for improving matte quality through morphological
/// operations, blurring, and edge feathering.
pub struct MatteRefiner {
    /// Kernel size for morphological operations.
    kernel_size: usize,
}

impl MatteRefiner {
    /// Create a new matte refiner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { kernel_size: 3 }
    }

    /// Set the kernel size for operations.
    pub fn set_kernel_size(&mut self, size: usize) {
        self.kernel_size = size.max(3) | 1; // Ensure odd and >= 3
    }

    /// Erode the matte (shrink foreground).
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn erode(&self, matte: &AlphaMatte, iterations: u32) -> CvResult<AlphaMatte> {
        let mut result = matte.clone();
        for _ in 0..iterations {
            result = self.erode_once(&result)?;
        }
        Ok(result)
    }

    /// Dilate the matte (expand foreground).
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn dilate(&self, matte: &AlphaMatte, iterations: u32) -> CvResult<AlphaMatte> {
        let mut result = matte.clone();
        for _ in 0..iterations {
            result = self.dilate_once(&result)?;
        }
        Ok(result)
    }

    /// Blur the matte for smooth edges.
    ///
    /// # Arguments
    ///
    /// * `matte` - Input matte
    /// * `radius` - Blur radius in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn blur(&self, matte: &AlphaMatte, radius: f32) -> CvResult<AlphaMatte> {
        if radius <= 0.0 {
            return Ok(matte.clone());
        }

        let kernel_size = (radius * 2.0).ceil() as usize * 2 + 1;
        let sigma = radius / 2.0;

        self.gaussian_blur(matte, kernel_size, sigma)
    }

    /// Apply edge feathering for smooth compositing.
    ///
    /// # Arguments
    ///
    /// * `matte` - Input matte
    /// * `amount` - Feathering amount (0.0-1.0)
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn feather(&self, matte: &AlphaMatte, amount: f32) -> CvResult<AlphaMatte> {
        if amount <= 0.0 {
            return Ok(matte.clone());
        }

        // Feather by blurring and then adjusting the curve
        let blurred = self.blur(matte, amount * 5.0)?;
        let mut result = blurred.clone();

        // Apply smooth curve to transition region
        for alpha in result.data_mut() {
            *alpha = self.feather_curve(*alpha, amount);
        }

        Ok(result)
    }

    /// Perform garbage matte operation (multiply with user-defined mask).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn garbage_matte(&self, matte: &AlphaMatte, mask: &AlphaMatte) -> CvResult<AlphaMatte> {
        matte.multiply(mask)
    }

    /// Perform one erosion pass.
    fn erode_once(&self, matte: &AlphaMatte) -> CvResult<AlphaMatte> {
        let width = matte.width() as usize;
        let height = matte.height() as usize;
        let mut new_data = vec![0.0f32; width * height];

        let radius = self.kernel_size / 2;

        for y in 0..height {
            for x in 0..width {
                let mut min_alpha = 1.0f32;

                // Find minimum in kernel
                for ky in 0..self.kernel_size {
                    for kx in 0..self.kernel_size {
                        let ny = (y + ky).wrapping_sub(radius);
                        let nx = (x + kx).wrapping_sub(radius);

                        if ny < height && nx < width {
                            let idx = ny * width + nx;
                            min_alpha = min_alpha.min(matte.data()[idx]);
                        }
                    }
                }

                new_data[y * width + x] = min_alpha;
            }
        }

        Ok(AlphaMatte::new(matte.width(), matte.height(), new_data))
    }

    /// Perform one dilation pass.
    fn dilate_once(&self, matte: &AlphaMatte) -> CvResult<AlphaMatte> {
        let width = matte.width() as usize;
        let height = matte.height() as usize;
        let mut new_data = vec![0.0f32; width * height];

        let radius = self.kernel_size / 2;

        for y in 0..height {
            for x in 0..width {
                let mut max_alpha = 0.0f32;

                // Find maximum in kernel
                for ky in 0..self.kernel_size {
                    for kx in 0..self.kernel_size {
                        let ny = (y + ky).wrapping_sub(radius);
                        let nx = (x + kx).wrapping_sub(radius);

                        if ny < height && nx < width {
                            let idx = ny * width + nx;
                            max_alpha = max_alpha.max(matte.data()[idx]);
                        }
                    }
                }

                new_data[y * width + x] = max_alpha;
            }
        }

        Ok(AlphaMatte::new(matte.width(), matte.height(), new_data))
    }

    /// Gaussian blur implementation.
    fn gaussian_blur(
        &self,
        matte: &AlphaMatte,
        kernel_size: usize,
        sigma: f32,
    ) -> CvResult<AlphaMatte> {
        let kernel = self.create_gaussian_kernel(kernel_size, sigma);

        let width = matte.width() as usize;
        let height = matte.height() as usize;
        let radius = kernel_size / 2;

        // Horizontal pass
        let mut temp = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for i in 0..kernel_size {
                    let nx = (x + i).wrapping_sub(radius);
                    if nx < width {
                        let idx = y * width + nx;
                        sum += matte.data()[idx] * kernel[i];
                        weight_sum += kernel[i];
                    }
                }

                temp[y * width + x] = sum / weight_sum;
            }
        }

        // Vertical pass
        let mut result = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for i in 0..kernel_size {
                    let ny = (y + i).wrapping_sub(radius);
                    if ny < height {
                        let idx = ny * width + x;
                        sum += temp[idx] * kernel[i];
                        weight_sum += kernel[i];
                    }
                }

                result[y * width + x] = sum / weight_sum;
            }
        }

        Ok(AlphaMatte::new(matte.width(), matte.height(), result))
    }

    /// Create a 1D Gaussian kernel.
    fn create_gaussian_kernel(&self, size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = vec![0.0f32; size];
        let radius = (size / 2) as f32;
        let two_sigma_sq = 2.0 * sigma * sigma;

        for i in 0..size {
            let x = i as f32 - radius;
            kernel[i] = (-x * x / two_sigma_sq).exp();
        }

        // Normalize
        let sum: f32 = kernel.iter().sum();
        for val in &mut kernel {
            *val /= sum;
        }

        kernel
    }

    /// Feather curve for smooth edge transitions.
    fn feather_curve(&self, alpha: f32, amount: f32) -> f32 {
        // S-curve for smooth feathering
        let t = alpha.clamp(0.0, 1.0);
        let feathered = t * t * (3.0 - 2.0 * t);

        // Blend between original and feathered
        alpha * (1.0 - amount) + feathered * amount
    }
}

impl Default for MatteRefiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Garbage matte for manual foreground/background separation.
///
/// Allows users to define regions that should definitely be kept or removed,
/// independent of the chroma key.
pub struct GarbageMatte {
    matte: AlphaMatte,
}

impl GarbageMatte {
    /// Create a new garbage matte initialized to fully opaque.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            matte: AlphaMatte::solid(width, height, 1.0),
        }
    }

    /// Set a rectangular region to transparent (garbage region).
    pub fn add_garbage_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        for gy in y..(y + height).min(self.matte.height()) {
            for gx in x..(x + width).min(self.matte.width()) {
                self.matte.set(gx, gy, 0.0);
            }
        }
    }

    /// Set a rectangular region to opaque (keep region).
    pub fn add_keep_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        for gy in y..(y + height).min(self.matte.height()) {
            for gx in x..(x + width).min(self.matte.width()) {
                self.matte.set(gx, gy, 1.0);
            }
        }
    }

    /// Add a circular garbage region.
    pub fn add_garbage_circle(&mut self, center_x: u32, center_y: u32, radius: f32) {
        let r_sq = radius * radius;
        let cx = center_x as f32;
        let cy = center_y as f32;

        for y in 0..self.matte.height() {
            for x in 0..self.matte.width() {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    self.matte.set(x, y, 0.0);
                }
            }
        }
    }

    /// Add a circular keep region.
    pub fn add_keep_circle(&mut self, center_x: u32, center_y: u32, radius: f32) {
        let r_sq = radius * radius;
        let cx = center_x as f32;
        let cy = center_y as f32;

        for y in 0..self.matte.height() {
            for x in 0..self.matte.width() {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    self.matte.set(x, y, 1.0);
                }
            }
        }
    }

    /// Blur the garbage matte for softer edges.
    ///
    /// # Errors
    ///
    /// Returns an error if blur operation fails.
    pub fn blur(&mut self, radius: f32) -> CvResult<()> {
        let refiner = MatteRefiner::new();
        self.matte = refiner.blur(&self.matte, radius)?;
        Ok(())
    }

    /// Get the internal matte.
    #[must_use]
    pub fn matte(&self) -> &AlphaMatte {
        &self.matte
    }

    /// Apply garbage matte to another matte.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn apply(&self, target: &AlphaMatte) -> CvResult<AlphaMatte> {
        target.multiply(&self.matte)
    }
}
