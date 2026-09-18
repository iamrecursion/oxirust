//! Binary and grayscale morphological operations.
//!
//! Provides erosion, dilation, opening, and closing on single-channel
//! floating-point image buffers using flat structuring elements.

#![allow(dead_code)]

/// Morphological operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphOp {
    /// Keep the minimum value within the structuring element (shrinks bright regions).
    Erode,
    /// Keep the maximum value within the structuring element (expands bright regions).
    Dilate,
    /// Erosion followed by dilation (removes small bright spots).
    Open,
    /// Dilation followed by erosion (fills small holes).
    Close,
}

impl MorphOp {
    /// Returns a human-readable name for the operation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Erode => "erode",
            Self::Dilate => "dilate",
            Self::Open => "open",
            Self::Close => "close",
        }
    }

    /// Returns true if this operation can increase pixel values.
    #[must_use]
    pub const fn can_expand(self) -> bool {
        matches!(self, Self::Dilate | Self::Close)
    }
}

/// A flat (binary) structuring element defined by its half-extents.
///
/// The element covers a rectangle of `(2*half_w + 1) × (2*half_h + 1)` pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructElement {
    /// Half-width: the element extends `half_w` pixels left and right of centre.
    pub half_w: usize,
    /// Half-height: the element extends `half_h` pixels above and below centre.
    pub half_h: usize,
}

impl StructElement {
    /// Creates a square structuring element with the given radius.
    #[must_use]
    pub const fn square(radius: usize) -> Self {
        Self {
            half_w: radius,
            half_h: radius,
        }
    }

    /// Creates a rectangular structuring element.
    #[must_use]
    pub const fn rect(half_w: usize, half_h: usize) -> Self {
        Self { half_w, half_h }
    }

    /// Returns the total number of pixels covered by the structuring element.
    #[must_use]
    pub const fn area(&self) -> usize {
        (2 * self.half_w + 1) * (2 * self.half_h + 1)
    }

    /// Returns the full width of the structuring element.
    #[must_use]
    pub const fn width(&self) -> usize {
        2 * self.half_w + 1
    }

    /// Returns the full height of the structuring element.
    #[must_use]
    pub const fn height(&self) -> usize {
        2 * self.half_h + 1
    }
}

/// Processor that applies morphological operations to single-channel f32 images.
#[derive(Debug, Clone)]
pub struct MorphProcessor {
    element: StructElement,
}

impl MorphProcessor {
    /// Creates a new processor with the given structuring element.
    #[must_use]
    pub fn new(element: StructElement) -> Self {
        Self { element }
    }

    /// Returns a reference to the structuring element.
    #[must_use]
    pub fn element(&self) -> &StructElement {
        &self.element
    }

    /// Applies `op` to `input` (row-major, `width × height`) and writes the
    /// result to `output`.
    ///
    /// # Panics
    ///
    /// Panics if slice lengths do not match `width * height`.
    pub fn apply(
        &self,
        op: MorphOp,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), width * height);
        assert_eq!(output.len(), width * height);
        match op {
            MorphOp::Erode => self.erode(input, output, width, height),
            MorphOp::Dilate => self.dilate(input, output, width, height),
            MorphOp::Open => {
                let mut tmp = vec![0.0_f32; width * height];
                self.erode(input, &mut tmp, width, height);
                self.dilate(&tmp, output, width, height);
            }
            MorphOp::Close => {
                let mut tmp = vec![0.0_f32; width * height];
                self.dilate(input, &mut tmp, width, height);
                self.erode(&tmp, output, width, height);
            }
        }
    }

    fn erode(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
        self.fold_op(input, output, width, height, f32::MAX, f32::min);
    }

    fn dilate(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
        self.fold_op(input, output, width, height, f32::MIN, f32::max);
    }

    #[allow(clippy::cast_precision_loss)]
    fn fold_op<F>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        init: f32,
        fold: F,
    ) where
        F: Fn(f32, f32) -> f32,
    {
        let hw = self.element.half_w as isize;
        let hh = self.element.half_h as isize;
        for cy in 0..height {
            for cx in 0..width {
                let mut acc = init;
                for ky in -hh..=hh {
                    for kx in -hw..=hw {
                        let sx = cx as isize + kx;
                        let sy = cy as isize + ky;
                        if sx >= 0 && sy >= 0 && (sx as usize) < width && (sy as usize) < height {
                            acc = fold(acc, input[sy as usize * width + sx as usize]);
                        }
                    }
                }
                output[cy * width + cx] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morph_op_names() {
        assert_eq!(MorphOp::Erode.name(), "erode");
        assert_eq!(MorphOp::Dilate.name(), "dilate");
        assert_eq!(MorphOp::Open.name(), "open");
        assert_eq!(MorphOp::Close.name(), "close");
    }

    #[test]
    fn morph_op_can_expand() {
        assert!(!MorphOp::Erode.can_expand());
        assert!(MorphOp::Dilate.can_expand());
        assert!(!MorphOp::Open.can_expand());
        assert!(MorphOp::Close.can_expand());
    }

    #[test]
    fn struct_element_square_area() {
        let se = StructElement::square(1);
        assert_eq!(se.area(), 9);
    }

    #[test]
    fn struct_element_rect_dimensions() {
        let se = StructElement::rect(2, 1);
        assert_eq!(se.width(), 5);
        assert_eq!(se.height(), 3);
        assert_eq!(se.area(), 15);
    }

    #[test]
    fn struct_element_zero_radius() {
        let se = StructElement::square(0);
        assert_eq!(se.area(), 1);
    }

    #[test]
    fn erode_uniform_image() {
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let input = vec![0.5_f32; 25];
        let mut output = vec![0.0_f32; 25];
        proc.apply(MorphOp::Erode, &input, &mut output, 5, 5);
        // Uniform image → erosion leaves it unchanged
        assert!(output.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn dilate_uniform_image() {
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let input = vec![0.5_f32; 25];
        let mut output = vec![0.0_f32; 25];
        proc.apply(MorphOp::Dilate, &input, &mut output, 5, 5);
        assert!(output.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn dilate_single_bright_pixel() {
        // 3×3 black image with one bright centre pixel
        let mut input = vec![0.0_f32; 9];
        input[4] = 1.0; // centre
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let mut output = vec![0.0_f32; 9];
        proc.apply(MorphOp::Dilate, &input, &mut output, 3, 3);
        // All pixels should now be 1.0 (3×3 element covers the whole image)
        assert!(output.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn erode_single_dark_pixel() {
        // 3×3 white image with one dark centre pixel
        let mut input = vec![1.0_f32; 9];
        input[4] = 0.0;
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let mut output = vec![0.0_f32; 9];
        proc.apply(MorphOp::Erode, &input, &mut output, 3, 3);
        // Interior pixel eroded to 0 due to dark centre; borders may vary
        assert_eq!(output[4], 0.0);
    }

    #[test]
    fn open_removes_small_bright_spot() {
        let mut input = vec![0.0_f32; 25];
        input[12] = 1.0; // single bright pixel in 5×5 dark image
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let mut output = vec![0.0_f32; 25];
        proc.apply(MorphOp::Open, &input, &mut output, 5, 5);
        // Opening with radius-1 element erases isolated bright pixels
        assert!(output.iter().all(|&v| v < 1e-6));
    }

    #[test]
    fn close_fills_small_hole() {
        let mut input = vec![1.0_f32; 25];
        input[12] = 0.0; // single dark pixel in bright image
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let mut output = vec![0.0_f32; 25];
        proc.apply(MorphOp::Close, &input, &mut output, 5, 5);
        // Closing fills the isolated dark pixel back to 1
        assert!((output[12] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn processor_exposes_element() {
        let se = StructElement::rect(2, 3);
        let proc = MorphProcessor::new(se.clone());
        assert_eq!(proc.element().half_w, 2);
        assert_eq!(proc.element().half_h, 3);
    }

    #[test]
    fn apply_panics_on_bad_input_len() {
        let se = StructElement::square(1);
        let proc = MorphProcessor::new(se);
        let result = std::panic::catch_unwind(|| {
            let input = vec![0.0_f32; 8]; // wrong length for 3×3
            let mut output = vec![0.0_f32; 9];
            proc.apply(MorphOp::Dilate, &input, &mut output, 3, 3);
        });
        assert!(result.is_err());
    }

    #[test]
    fn morph_op_copy() {
        let op = MorphOp::Dilate;
        let op2 = op;
        assert_eq!(op, op2);
    }

    #[test]
    fn struct_element_equality() {
        assert_eq!(StructElement::square(2), StructElement::rect(2, 2));
    }
}
