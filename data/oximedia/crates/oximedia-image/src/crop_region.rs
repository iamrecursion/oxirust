#![allow(dead_code)]
//! Crop and region-of-interest (ROI) operations for image buffers.
//!
//! Provides rectangular region definitions, cropping, padding, border operations,
//! and ROI extraction for single-channel and multi-channel image buffers.

use std::fmt;

/// A rectangular region defined by its top-left corner and dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    /// X coordinate of the top-left corner.
    pub x: usize,
    /// Y coordinate of the top-left corner.
    pub y: usize,
    /// Width of the region.
    pub width: usize,
    /// Height of the region.
    pub height: usize,
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rect(x={}, y={}, w={}, h={})",
            self.x, self.y, self.width, self.height
        )
    }
}

impl Rect {
    /// Creates a new rectangle.
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Creates a rectangle from the full image dimensions.
    pub fn from_image_size(width: usize, height: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// Returns the area of this rectangle.
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Returns true if this rectangle has zero area.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Returns the right edge x coordinate (exclusive).
    pub fn right(&self) -> usize {
        self.x + self.width
    }

    /// Returns the bottom edge y coordinate (exclusive).
    pub fn bottom(&self) -> usize {
        self.y + self.height
    }

    /// Returns the center point (integer division).
    pub fn center(&self) -> (usize, usize) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    /// Tests whether a point lies inside this rectangle.
    pub fn contains(&self, px: usize, py: usize) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Returns the intersection of two rectangles, or None if they don't overlap.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.right().min(other.right());
        let y2 = self.bottom().min(other.bottom());

        if x1 < x2 && y1 < y2 {
            Some(Self {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
            })
        } else {
            None
        }
    }

    /// Returns the bounding box that contains both rectangles.
    pub fn union(&self, other: &Self) -> Self {
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = self.right().max(other.right());
        let y2 = self.bottom().max(other.bottom());
        Self {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    /// Clips this rectangle to fit within the given image dimensions.
    pub fn clip_to_image(&self, img_width: usize, img_height: usize) -> Self {
        let img_rect = Self::from_image_size(img_width, img_height);
        self.intersect(&img_rect).unwrap_or(Self::new(0, 0, 0, 0))
    }

    /// Expands or shrinks the rectangle by `amount` pixels on each side.
    ///
    /// Negative values shrink the rectangle. The result is clamped so
    /// width/height never go below zero.
    pub fn expand(&self, amount: i32) -> Self {
        let x = (self.x as i64 - i64::from(amount)).max(0) as usize;
        let y = (self.y as i64 - i64::from(amount)).max(0) as usize;
        let r = self.right() as i64 + i64::from(amount);
        let b = self.bottom() as i64 + i64::from(amount);
        let w = (r - x as i64).max(0) as usize;
        let h = (b - y as i64).max(0) as usize;
        Self {
            x,
            y,
            width: w,
            height: h,
        }
    }
}

/// Border fill mode for padding operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BorderMode {
    /// Fill border pixels with a constant value.
    Constant(f32),
    /// Replicate the nearest edge pixel.
    Replicate,
    /// Reflect pixels across the edge.
    Reflect,
    /// Wrap around to the opposite edge.
    Wrap,
}

impl Default for BorderMode {
    fn default() -> Self {
        Self::Constant(0.0)
    }
}

/// Crops a single-channel f32 buffer to the given rectangle.
///
/// Returns a new buffer of size `rect.width * rect.height`.
pub fn crop_f32(
    buf: &[f32],
    img_width: usize,
    img_height: usize,
    rect: &Rect,
) -> Result<Vec<f32>, String> {
    let clipped = rect.clip_to_image(img_width, img_height);
    if clipped.is_empty() {
        return Ok(Vec::new());
    }
    if buf.len() != img_width * img_height {
        return Err(format!(
            "buffer size mismatch: expected {}, got {}",
            img_width * img_height,
            buf.len()
        ));
    }

    let mut out = Vec::with_capacity(clipped.area());
    for y in clipped.y..clipped.bottom() {
        let start = y * img_width + clipped.x;
        out.extend_from_slice(&buf[start..start + clipped.width]);
    }
    Ok(out)
}

/// Crops a multi-channel interleaved f32 buffer.
///
/// `num_channels` is the number of components per pixel.
pub fn crop_multichannel_f32(
    buf: &[f32],
    img_width: usize,
    img_height: usize,
    num_channels: usize,
    rect: &Rect,
) -> Result<Vec<f32>, String> {
    let clipped = rect.clip_to_image(img_width, img_height);
    if clipped.is_empty() {
        return Ok(Vec::new());
    }
    let expected = img_width * img_height * num_channels;
    if buf.len() != expected {
        return Err(format!(
            "buffer size mismatch: expected {expected}, got {}",
            buf.len()
        ));
    }

    let row_stride = img_width * num_channels;
    let crop_row_len = clipped.width * num_channels;
    let mut out = Vec::with_capacity(clipped.area() * num_channels);
    for y in clipped.y..clipped.bottom() {
        let start = y * row_stride + clipped.x * num_channels;
        out.extend_from_slice(&buf[start..start + crop_row_len]);
    }
    Ok(out)
}

/// Pads a single-channel f32 buffer on all four sides.
#[allow(clippy::cast_precision_loss)]
pub fn pad_f32(
    buf: &[f32],
    width: usize,
    height: usize,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: BorderMode,
) -> Vec<f32> {
    let new_w = width + left + right;
    let new_h = height + top + bottom;
    let mut out = vec![0.0_f32; new_w * new_h];

    for ny in 0..new_h {
        for nx in 0..new_w {
            let val = sample_with_border(buf, width, height, nx, ny, left, top, mode);
            out[ny * new_w + nx] = val;
        }
    }
    out
}

/// Samples a pixel with border handling.
fn sample_with_border(
    buf: &[f32],
    width: usize,
    height: usize,
    out_x: usize,
    out_y: usize,
    offset_x: usize,
    offset_y: usize,
    mode: BorderMode,
) -> f32 {
    let sx = out_x as i64 - offset_x as i64;
    let sy = out_y as i64 - offset_y as i64;

    if sx >= 0 && (sx as usize) < width && sy >= 0 && (sy as usize) < height {
        return buf[sy as usize * width + sx as usize];
    }

    match mode {
        BorderMode::Constant(c) => c,
        BorderMode::Replicate => {
            let cx = sx.clamp(0, width as i64 - 1) as usize;
            let cy = sy.clamp(0, height as i64 - 1) as usize;
            buf[cy * width + cx]
        }
        BorderMode::Reflect => {
            let cx = reflect_coord(sx, width);
            let cy = reflect_coord(sy, height);
            buf[cy * width + cx]
        }
        BorderMode::Wrap => {
            let cx = wrap_coord(sx, width);
            let cy = wrap_coord(sy, height);
            buf[cy * width + cx]
        }
    }
}

/// Reflects a coordinate into valid range.
fn reflect_coord(c: i64, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let s = size as i64;
    let mut v = c;
    if v < 0 {
        v = -v - 1;
    }
    let period = 2 * s;
    v %= period;
    if v < 0 {
        v += period;
    }
    if v >= s {
        v = period - v - 1;
    }
    v.clamp(0, s - 1) as usize
}

/// Wraps a coordinate using modular arithmetic.
fn wrap_coord(c: i64, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let s = size as i64;
    ((c % s + s) % s) as usize
}

/// Extracts a centered crop of the given size from the image.
///
/// If the requested crop is larger than the image, the result is clipped.
pub fn center_crop_f32(
    buf: &[f32],
    img_width: usize,
    img_height: usize,
    crop_w: usize,
    crop_h: usize,
) -> Result<Vec<f32>, String> {
    let cx = if crop_w >= img_width {
        0
    } else {
        (img_width - crop_w) / 2
    };
    let cy = if crop_h >= img_height {
        0
    } else {
        (img_height - crop_h) / 2
    };
    let rect = Rect::new(cx, cy, crop_w, crop_h);
    crop_f32(buf, img_width, img_height, &rect)
}

/// Auto-crops by removing border rows/columns where all values equal the border value.
///
/// Returns the bounding `Rect` of the non-border content.
pub fn auto_crop_bounds(
    buf: &[f32],
    width: usize,
    height: usize,
    border_value: f32,
    tolerance: f32,
) -> Rect {
    let matches_border = |v: f32| (v - border_value).abs() <= tolerance;

    let mut top = 0;
    'top_scan: for y in 0..height {
        for x in 0..width {
            if !matches_border(buf[y * width + x]) {
                break 'top_scan;
            }
        }
        top = y + 1;
    }

    if top >= height {
        return Rect::new(0, 0, 0, 0);
    }

    let mut bot = height;
    'bot_scan: for y in (0..height).rev() {
        for x in 0..width {
            if !matches_border(buf[y * width + x]) {
                break 'bot_scan;
            }
        }
        bot = y;
    }

    let mut left = width;
    for y in top..bot {
        for x in 0..left {
            if !matches_border(buf[y * width + x]) {
                left = x;
                break;
            }
        }
    }

    let mut right = 0;
    for y in top..bot {
        for x in (right..width).rev() {
            if !matches_border(buf[y * width + x]) {
                right = x + 1;
                break;
            }
        }
    }

    if left >= right || top >= bot {
        Rect::new(0, 0, 0, 0)
    } else {
        Rect::new(left, top, right - left, bot - top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_basic() {
        let r = Rect::new(10, 20, 100, 50);
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 70);
        assert_eq!(r.area(), 5000);
        assert!(!r.is_empty());
        assert_eq!(r.center(), (60, 45));
    }

    #[test]
    fn test_rect_empty() {
        let r = Rect::new(0, 0, 0, 10);
        assert!(r.is_empty());
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn test_rect_contains() {
        let r = Rect::new(5, 5, 10, 10);
        assert!(r.contains(5, 5));
        assert!(r.contains(14, 14));
        assert!(!r.contains(15, 5));
        assert!(!r.contains(4, 5));
    }

    #[test]
    fn test_rect_intersect() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        let isec = a.intersect(&b).expect("should succeed in test");
        assert_eq!(isec, Rect::new(5, 5, 5, 5));

        let c = Rect::new(20, 20, 5, 5);
        assert!(a.intersect(&c).is_none());
    }

    #[test]
    fn test_rect_union() {
        let a = Rect::new(0, 0, 5, 5);
        let b = Rect::new(3, 3, 5, 5);
        let u = a.union(&b);
        assert_eq!(u, Rect::new(0, 0, 8, 8));
    }

    #[test]
    fn test_rect_expand() {
        let r = Rect::new(10, 10, 20, 20);
        let expanded = r.expand(5);
        assert_eq!(expanded, Rect::new(5, 5, 30, 30));

        let shrunk = r.expand(-5);
        assert_eq!(shrunk, Rect::new(15, 15, 10, 10));
    }

    #[test]
    fn test_crop_f32() {
        let buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let rect = Rect::new(1, 1, 2, 2);
        let cropped = crop_f32(&buf, 4, 4, &rect).expect("should succeed in test");
        assert_eq!(cropped, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_crop_multichannel() {
        // 2x2 RGB image
        let buf = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let rect = Rect::new(1, 0, 1, 2);
        let cropped = crop_multichannel_f32(&buf, 2, 2, 3, &rect).expect("should succeed in test");
        assert_eq!(cropped, vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_center_crop() {
        let buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cropped = center_crop_f32(&buf, 4, 4, 2, 2).expect("should succeed in test");
        assert_eq!(cropped, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_pad_constant() {
        let buf = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_f32(&buf, 2, 2, 1, 1, 1, 1, BorderMode::Constant(0.0));
        assert_eq!(padded.len(), 16);
        // top-left corner should be 0
        assert_eq!(padded[0], 0.0);
        // center should be original
        assert_eq!(padded[5], 1.0);
        assert_eq!(padded[6], 2.0);
    }

    #[test]
    fn test_pad_replicate() {
        let buf = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_f32(&buf, 2, 2, 1, 0, 0, 0, BorderMode::Replicate);
        // First row should replicate the first image row
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
    }

    #[test]
    fn test_auto_crop_bounds() {
        // 4x4 with a 2x2 non-zero region in the center
        #[rustfmt::skip]
        let buf = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let bounds = auto_crop_bounds(&buf, 4, 4, 0.0, 0.001);
        assert_eq!(bounds, Rect::new(1, 1, 2, 2));
    }

    #[test]
    fn test_auto_crop_all_same() {
        let buf = vec![0.0; 16];
        let bounds = auto_crop_bounds(&buf, 4, 4, 0.0, 0.001);
        assert!(bounds.is_empty());
    }

    #[test]
    fn test_rect_display() {
        let r = Rect::new(1, 2, 3, 4);
        assert_eq!(r.to_string(), "Rect(x=1, y=2, w=3, h=4)");
    }

    #[test]
    fn test_clip_to_image() {
        let r = Rect::new(90, 90, 20, 20);
        let clipped = r.clip_to_image(100, 100);
        assert_eq!(clipped, Rect::new(90, 90, 10, 10));
    }
}
