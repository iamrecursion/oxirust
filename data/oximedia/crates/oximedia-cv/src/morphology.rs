#![allow(dead_code)]
//! Mathematical morphology operations for binary and grayscale images.
//!
//! This module provides fundamental morphological operations used in
//! image processing: erosion, dilation, opening, closing, gradient,
//! top-hat, and black-hat transforms.
//!
//! # Structuring Elements
//!
//! Morphological operations use a structuring element (kernel) to probe
//! the image. Common shapes include rectangles, crosses, and ellipses.
//!
//! # Operations
//!
//! - **Erosion**: Shrinks bright regions, removes small bright noise.
//! - **Dilation**: Expands bright regions, fills small dark holes.
//! - **Opening**: Erosion followed by dilation (removes small bright spots).
//! - **Closing**: Dilation followed by erosion (fills small dark holes).
//! - **Gradient**: Dilation minus erosion (highlights edges).
//! - **Top-hat**: Original minus opening (extracts bright details).
//! - **Black-hat**: Closing minus original (extracts dark details).

/// Shape of a structuring element for morphological operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuringElementShape {
    /// Rectangular structuring element (all ones).
    Rectangle,
    /// Cross-shaped structuring element.
    Cross,
    /// Elliptical (approximate disk) structuring element.
    Ellipse,
}

/// A structuring element (kernel) for morphological operations.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    /// Width of the structuring element.
    pub width: usize,
    /// Height of the structuring element.
    pub height: usize,
    /// Mask data: true means the element is active at that position.
    pub mask: Vec<bool>,
    /// Anchor X position (center column).
    pub anchor_x: usize,
    /// Anchor Y position (center row).
    pub anchor_y: usize,
}

impl StructuringElement {
    /// Creates a rectangular structuring element of the given size.
    pub fn rectangle(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            mask: vec![true; width * height],
            anchor_x: width / 2,
            anchor_y: height / 2,
        }
    }

    /// Creates a cross-shaped structuring element.
    pub fn cross(size: usize) -> Self {
        let mut mask = vec![false; size * size];
        let center = size / 2;
        for i in 0..size {
            mask[center * size + i] = true; // horizontal
            mask[i * size + center] = true; // vertical
        }
        Self {
            width: size,
            height: size,
            mask,
            anchor_x: center,
            anchor_y: center,
        }
    }

    /// Creates an elliptical structuring element.
    #[allow(clippy::cast_precision_loss)]
    pub fn ellipse(width: usize, height: usize) -> Self {
        let mut mask = vec![false; width * height];
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let rx = cx;
        let ry = cy;
        for row in 0..height {
            for col in 0..width {
                let dx = (col as f64 + 0.5 - cx) / rx;
                let dy = (row as f64 + 0.5 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 {
                    mask[row * width + col] = true;
                }
            }
        }
        Self {
            width,
            height,
            mask,
            anchor_x: width / 2,
            anchor_y: height / 2,
        }
    }

    /// Creates a structuring element from a given shape and size.
    pub fn from_shape(shape: StructuringElementShape, width: usize, height: usize) -> Self {
        match shape {
            StructuringElementShape::Rectangle => Self::rectangle(width, height),
            StructuringElementShape::Cross => Self::cross(width.max(height)),
            StructuringElementShape::Ellipse => Self::ellipse(width, height),
        }
    }

    /// Returns the number of active (true) elements.
    pub fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Checks whether the element at (col, row) is active.
    pub fn is_active(&self, col: usize, row: usize) -> bool {
        if col < self.width && row < self.height {
            self.mask[row * self.width + col]
        } else {
            false
        }
    }
}

/// Performs grayscale erosion on a 2D image.
///
/// For each pixel, the minimum value within the structuring element
/// neighborhood is computed.
pub fn erode(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let mut output = vec![f64::MAX; width * height];
    let ax = se.anchor_x as isize;
    let ay = se.anchor_y as isize;

    for row in 0..height {
        for col in 0..width {
            let mut min_val = f64::MAX;
            for sy in 0..se.height {
                for sx in 0..se.width {
                    if !se.is_active(sx, sy) {
                        continue;
                    }
                    let img_x = col as isize + sx as isize - ax;
                    let img_y = row as isize + sy as isize - ay;
                    if img_x >= 0 && img_x < width as isize && img_y >= 0 && img_y < height as isize
                    {
                        let val = image[img_y as usize * width + img_x as usize];
                        if val < min_val {
                            min_val = val;
                        }
                    }
                }
            }
            #[allow(clippy::float_cmp)]
            if min_val == f64::MAX {
                min_val = 0.0;
            }
            output[row * width + col] = min_val;
        }
    }
    output
}

/// Performs grayscale dilation on a 2D image.
///
/// For each pixel, the maximum value within the structuring element
/// neighborhood is computed.
pub fn dilate(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let mut output = vec![f64::MIN; width * height];
    let ax = se.anchor_x as isize;
    let ay = se.anchor_y as isize;

    for row in 0..height {
        for col in 0..width {
            let mut max_val = f64::MIN;
            for sy in 0..se.height {
                for sx in 0..se.width {
                    if !se.is_active(sx, sy) {
                        continue;
                    }
                    let img_x = col as isize + sx as isize - ax;
                    let img_y = row as isize + sy as isize - ay;
                    if img_x >= 0 && img_x < width as isize && img_y >= 0 && img_y < height as isize
                    {
                        let val = image[img_y as usize * width + img_x as usize];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
            }
            #[allow(clippy::float_cmp)]
            if max_val == f64::MIN {
                max_val = 0.0;
            }
            output[row * width + col] = max_val;
        }
    }
    output
}

/// Performs morphological opening (erosion followed by dilation).
///
/// Opening removes small bright regions while preserving overall shape.
pub fn opening(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let eroded = erode(image, width, height, se);
    dilate(&eroded, width, height, se)
}

/// Performs morphological closing (dilation followed by erosion).
///
/// Closing fills small dark regions while preserving overall shape.
pub fn closing(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let dilated = dilate(image, width, height, se);
    erode(&dilated, width, height, se)
}

/// Computes the morphological gradient (dilation minus erosion).
///
/// Highlights edges in the image.
pub fn morphological_gradient(
    image: &[f64],
    width: usize,
    height: usize,
    se: &StructuringElement,
) -> Vec<f64> {
    let dilated = dilate(image, width, height, se);
    let eroded = erode(image, width, height, se);
    dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d - e)
        .collect()
}

/// Computes the top-hat transform (original minus opening).
///
/// Extracts bright details smaller than the structuring element.
pub fn top_hat(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let opened = opening(image, width, height, se);
    image
        .iter()
        .zip(opened.iter())
        .map(|(&orig, &op)| (orig - op).max(0.0))
        .collect()
}

/// Computes the black-hat transform (closing minus original).
///
/// Extracts dark details smaller than the structuring element.
pub fn black_hat(image: &[f64], width: usize, height: usize, se: &StructuringElement) -> Vec<f64> {
    let closed = closing(image, width, height, se);
    closed
        .iter()
        .zip(image.iter())
        .map(|(&cl, &orig)| (cl - orig).max(0.0))
        .collect()
}

/// Performs binary erosion on a boolean image.
///
/// A pixel is true in the output only if all active SE neighbors are true.
pub fn binary_erode(
    image: &[bool],
    width: usize,
    height: usize,
    se: &StructuringElement,
) -> Vec<bool> {
    let mut output = vec![false; width * height];
    let ax = se.anchor_x as isize;
    let ay = se.anchor_y as isize;

    for row in 0..height {
        for col in 0..width {
            let mut all_true = true;
            for sy in 0..se.height {
                for sx in 0..se.width {
                    if !se.is_active(sx, sy) {
                        continue;
                    }
                    let img_x = col as isize + sx as isize - ax;
                    let img_y = row as isize + sy as isize - ay;
                    if img_x < 0 || img_x >= width as isize || img_y < 0 || img_y >= height as isize
                    {
                        all_true = false;
                        break;
                    }
                    if !image[img_y as usize * width + img_x as usize] {
                        all_true = false;
                        break;
                    }
                }
                if !all_true {
                    break;
                }
            }
            output[row * width + col] = all_true;
        }
    }
    output
}

/// Performs binary dilation on a boolean image.
///
/// A pixel is true in the output if any active SE neighbor is true.
pub fn binary_dilate(
    image: &[bool],
    width: usize,
    height: usize,
    se: &StructuringElement,
) -> Vec<bool> {
    let mut output = vec![false; width * height];
    let ax = se.anchor_x as isize;
    let ay = se.anchor_y as isize;

    for row in 0..height {
        for col in 0..width {
            let mut any_true = false;
            for sy in 0..se.height {
                for sx in 0..se.width {
                    if !se.is_active(sx, sy) {
                        continue;
                    }
                    let img_x = col as isize + sx as isize - ax;
                    let img_y = row as isize + sy as isize - ay;
                    if img_x >= 0
                        && img_x < width as isize
                        && img_y >= 0
                        && img_y < height as isize
                        && image[img_y as usize * width + img_x as usize]
                    {
                        any_true = true;
                        break;
                    }
                }
                if any_true {
                    break;
                }
            }
            output[row * width + col] = any_true;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangle_se() {
        let se = StructuringElement::rectangle(3, 3);
        assert_eq!(se.width, 3);
        assert_eq!(se.height, 3);
        assert_eq!(se.active_count(), 9);
        assert!(se.is_active(0, 0));
        assert!(se.is_active(2, 2));
    }

    #[test]
    fn test_cross_se() {
        let se = StructuringElement::cross(3);
        assert_eq!(se.active_count(), 5);
        assert!(se.is_active(1, 0)); // top center
        assert!(se.is_active(0, 1)); // left center
        assert!(se.is_active(1, 1)); // center
        assert!(!se.is_active(0, 0)); // corner
    }

    #[test]
    fn test_ellipse_se() {
        let se = StructuringElement::ellipse(5, 5);
        assert!(se.is_active(2, 2)); // center
        assert!(se.active_count() > 0);
        assert!(se.active_count() < 25); // less than full rectangle
    }

    #[test]
    fn test_from_shape() {
        let se = StructuringElement::from_shape(StructuringElementShape::Rectangle, 3, 3);
        assert_eq!(se.active_count(), 9);
    }

    #[test]
    fn test_erode_uniform() {
        let image = vec![1.0; 9]; // 3x3 all ones
        let se = StructuringElement::rectangle(3, 3);
        let result = erode(&image, 3, 3, &se);
        assert!((result[4] - 1.0).abs() < 1e-10); // center
    }

    #[test]
    fn test_dilate_single_pixel() {
        let mut image = vec![0.0; 25]; // 5x5 zeros
        image[12] = 1.0; // center pixel
        let se = StructuringElement::rectangle(3, 3);
        let result = dilate(&image, 5, 5, &se);
        // Center and 8-neighbors should be 1.0
        assert!((result[12] - 1.0).abs() < 1e-10);
        assert!((result[7] - 1.0).abs() < 1e-10); // above center
        assert!((result[17] - 1.0).abs() < 1e-10); // below center
    }

    #[test]
    fn test_opening_removes_small_bright() {
        let mut image = vec![0.0; 25]; // 5x5 zeros
        image[12] = 1.0; // single bright pixel
        let se = StructuringElement::rectangle(3, 3);
        let result = opening(&image, 5, 5, &se);
        // The single pixel should be removed by opening
        assert!((result[12] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_closing_fills_small_dark() {
        let mut image = vec![1.0; 25]; // 5x5 ones
        image[12] = 0.0; // single dark pixel
        let se = StructuringElement::rectangle(3, 3);
        let result = closing(&image, 5, 5, &se);
        // The single dark pixel should be filled by closing
        assert!((result[12] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_morphological_gradient() {
        let image = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let se = StructuringElement::rectangle(3, 3);
        let grad = morphological_gradient(&image, 5, 5, &se);
        // Edge pixels should have non-zero gradient
        assert!(grad[6] > 0.0); // corner of bright region
    }

    #[test]
    fn test_top_hat() {
        let image = vec![0.5; 9]; // uniform 3x3
        let se = StructuringElement::rectangle(3, 3);
        let result = top_hat(&image, 3, 3, &se);
        // For a uniform image, top-hat should be zero
        for &v in &result {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_black_hat() {
        let image = vec![0.5; 9]; // uniform 3x3
        let se = StructuringElement::rectangle(3, 3);
        let result = black_hat(&image, 3, 3, &se);
        // For a uniform image, black-hat should be zero
        for &v in &result {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_binary_erode() {
        let image = vec![true, true, true, true, true, true, true, true, true];
        let se = StructuringElement::rectangle(3, 3);
        let result = binary_erode(&image, 3, 3, &se);
        // Only center should be true (all neighbors within bounds)
        assert!(result[4]);
    }

    #[test]
    fn test_binary_dilate() {
        let mut image = vec![false; 9];
        image[4] = true; // center only
        let se = StructuringElement::rectangle(3, 3);
        let result = binary_dilate(&image, 3, 3, &se);
        // All 9 pixels should be true
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn test_is_active_out_of_bounds() {
        let se = StructuringElement::rectangle(3, 3);
        assert!(!se.is_active(5, 5));
    }
}
