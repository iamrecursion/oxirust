//! Morphological operations for image preprocessing
//!
//! This module provides morphological operations like erosion, dilation,
//! opening, closing, and morphological gradient.

use crate::error::{Result, VisionError};
use image::{DynamicImage, ImageBuffer, Luma};

/// Structuring element (kernel) shape
#[derive(Debug, Clone, Copy)]
pub enum StructuringElement {
    /// Rectangular structuring element
    Rectangle(usize, usize),
    /// Elliptical/circular structuring element
    Ellipse(usize, usize),
    /// Cross structuring element
    Cross(usize),
}

/// Create a structuring element (kernel) for morphological operations
///
/// # Arguments
///
/// * `shape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing a binary kernel
#[allow(dead_code)]
fn create_structuring_element(shape: StructuringElement) -> Result<Vec<Vec<bool>>> {
    match shape {
        StructuringElement::Rectangle(width, height) => {
            if width == 0 || height == 0 {
                return Err(VisionError::InvalidParameter(
                    "Width and height must be positive".to_string(),
                ));
            }

            // Create a rectangular kernel filled with true values
            let kernel = vec![vec![true; width]; height];
            Ok(kernel)
        }
        StructuringElement::Ellipse(width, height) => {
            if width == 0 || height == 0 {
                return Err(VisionError::InvalidParameter(
                    "Width and height must be positive".to_string(),
                ));
            }

            let center_x = width as f32 / 2.0;
            let center_y = height as f32 / 2.0;
            let radius_x = center_x;
            let radius_y = center_y;

            // Create an elliptical kernel
            let mut kernel = vec![vec![false; width]; height];

            for (y, row) in kernel.iter_mut().enumerate() {
                for (x, cell) in row.iter_mut().enumerate() {
                    let dx = (x as f32 - center_x + 0.5) / radius_x;
                    let dy = (y as f32 - center_y + 0.5) / radius_y;

                    // Check if point is within the ellipse: (dx/a)^2 + (dy/b)^2 <= 1
                    if dx * dx + dy * dy <= 1.0 {
                        *cell = true;
                    }
                }
            }

            Ok(kernel)
        }
        StructuringElement::Cross(size) => {
            if size == 0 {
                return Err(VisionError::InvalidParameter(
                    "Size must be positive".to_string(),
                ));
            }

            // Ensure size is odd
            let size = if size % 2 == 0 { size + 1 } else { size };

            // Create a cross-shaped kernel
            let mut kernel = vec![vec![false; size]; size];
            let center = size / 2;

            for (y, row) in kernel.iter_mut().enumerate() {
                for (x, cell) in row.iter_mut().enumerate() {
                    if x == center || y == center {
                        *cell = true;
                    }
                }
            }

            Ok(kernel)
        }
    }
}

/// Apply erosion to an image
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the eroded image
#[allow(dead_code)]
pub fn erode(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let kernel = create_structuring_element(kernelshape)?;

    let kernel_height = kernel.len();
    let kernel_width = kernel[0].len();

    // Calculate anchor point (center of kernel)
    let anchor_x = kernel_width / 2;
    let anchor_y = kernel_height / 2;

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Initialize with maximum possible value
            let mut min_val = 255u8;

            // Apply kernel
            for (ky, kernel_row) in kernel.iter().enumerate() {
                for (kx, &is_active) in kernel_row.iter().enumerate() {
                    // Skip if kernel element is not active
                    if !is_active {
                        continue;
                    }

                    // Calculate image coordinates with kernel anchor
                    let img_x = x as isize + (kx as isize - anchor_x as isize);
                    let img_y = y as isize + (ky as isize - anchor_y as isize);

                    // Skip coordinates outside image bounds
                    if img_x < 0 || img_x >= width as isize || img_y < 0 || img_y >= height as isize
                    {
                        continue;
                    }

                    // Update minimum value
                    let pixel_val = gray.get_pixel(img_x as u32, img_y as u32)[0];
                    min_val = min_val.min(pixel_val);
                }
            }

            // Set result pixel
            result.put_pixel(x, y, Luma([min_val]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply dilation to an image
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the dilated image
#[allow(dead_code)]
pub fn dilate(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let kernel = create_structuring_element(kernelshape)?;

    let kernel_height = kernel.len();
    let kernel_width = kernel[0].len();

    // Calculate anchor point (center of kernel)
    let anchor_x = kernel_width / 2;
    let anchor_y = kernel_height / 2;

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Initialize with minimum possible value
            let mut max_val = 0u8;

            // Apply kernel
            for (ky, kernel_row) in kernel.iter().enumerate() {
                for (kx, &is_active) in kernel_row.iter().enumerate() {
                    // Skip if kernel element is not active
                    if !is_active {
                        continue;
                    }

                    // Calculate image coordinates with kernel anchor
                    let img_x = x as isize - (kx as isize - anchor_x as isize);
                    let img_y = y as isize - (ky as isize - anchor_y as isize);

                    // Skip coordinates outside image bounds
                    if img_x < 0 || img_x >= width as isize || img_y < 0 || img_y >= height as isize
                    {
                        continue;
                    }

                    // Update maximum value
                    let pixel_val = gray.get_pixel(img_x as u32, img_y as u32)[0];
                    max_val = max_val.max(pixel_val);
                }
            }

            // Set result pixel
            result.put_pixel(x, y, Luma([max_val]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply morphological opening (erosion followed by dilation)
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the opened image
#[allow(dead_code)]
pub fn opening(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let eroded = erode(img, kernelshape)?;
    dilate(&eroded, kernelshape)
}

/// Apply morphological closing (dilation followed by erosion)
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the closed image
#[allow(dead_code)]
pub fn closing(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let dilated = dilate(img, kernelshape)?;
    erode(&dilated, kernelshape)
}

/// Apply morphological gradient (difference between dilation and erosion)
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the gradient image
#[allow(dead_code)]
pub fn morphological_gradient(
    img: &DynamicImage,
    kernelshape: StructuringElement,
) -> Result<DynamicImage> {
    let dilated = dilate(img, kernelshape)?;
    let eroded = erode(img, kernelshape)?;

    let dilated_gray = dilated.to_luma8();
    let eroded_gray = eroded.to_luma8();
    let (width, height) = dilated_gray.dimensions();

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let dilated_val = dilated_gray.get_pixel(x, y)[0];
            let eroded_val = eroded_gray.get_pixel(x, y)[0];

            // Calculate gradient (dilation - erosion)
            // Handle underflow with saturating subtraction
            let gradient = dilated_val.saturating_sub(eroded_val);

            // Set result pixel
            result.put_pixel(x, y, Luma([gradient]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply top-hat transform (original - opening)
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the top-hat transformed image
#[allow(dead_code)]
pub fn top_hat(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let opened = opening(img, kernelshape)?;

    let original = img.to_luma8();
    let opened_gray = opened.to_luma8();
    let (width, height) = original.dimensions();

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let original_val = original.get_pixel(x, y)[0];
            let opened_val = opened_gray.get_pixel(x, y)[0];

            // Top-hat: original - opening
            // Handle underflow with saturating subtraction
            let top_hat_val = original_val.saturating_sub(opened_val);

            // Set result pixel
            result.put_pixel(x, y, Luma([top_hat_val]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply black-hat transform (closing - original)
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `kernelshape` - Shape of the structuring element
///
/// # Returns
///
/// * Result containing the black-hat transformed image
#[allow(dead_code)]
pub fn black_hat(img: &DynamicImage, kernelshape: StructuringElement) -> Result<DynamicImage> {
    let closed = closing(img, kernelshape)?;

    let original = img.to_luma8();
    let closed_gray = closed.to_luma8();
    let (width, height) = original.dimensions();

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let original_val = original.get_pixel(x, y)[0];
            let closed_val = closed_gray.get_pixel(x, y)[0];

            // Black-hat: closing - original
            // Handle underflow with saturating subtraction
            let black_hat_val = closed_val.saturating_sub(original_val);

            // Set result pixel
            result.put_pixel(x, y, Luma([black_hat_val]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}
