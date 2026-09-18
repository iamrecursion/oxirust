//! Image geometric transform operations.
//!
//! Provides 2D affine/homogeneous transforms, bilinear resize, crop,
//! flip, and 90-degree rotation.

#![allow(dead_code)]

/// 3×3 homogeneous 2D transform matrix (row-major).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform2D {
    /// Row-major 3×3 homogeneous matrix.
    pub matrix: [[f32; 3]; 3],
}

impl Transform2D {
    /// Identity transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Translation transform.
    #[must_use]
    pub const fn translate(dx: f32, dy: f32) -> Self {
        Self {
            matrix: [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
        }
    }

    /// Rotation transform (clockwise, degrees).
    #[must_use]
    pub fn rotate_deg(angle: f32) -> Self {
        let r = angle.to_radians();
        let c = r.cos();
        let s = r.sin();
        Self {
            matrix: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Scale transform.
    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Compose (multiply) two transforms: `self` applied first, then `other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = &other.matrix;
        let b = &self.matrix;
        let mut m = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    m[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        Self { matrix: m }
    }

    /// Apply the transform to a 2D point `(x, y)`, returning the transformed point.
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.matrix;
        let xp = m[0][0] * x + m[0][1] * y + m[0][2];
        let yp = m[1][0] * x + m[1][1] * y + m[1][2];
        let wp = m[2][0] * x + m[2][1] * y + m[2][2];
        (xp / wp, yp / wp)
    }
}

/// Resampling filter for resize operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest-neighbour (no anti-aliasing).
    Nearest,
    /// Bilinear (2×2 kernel).
    Bilinear,
    /// Bicubic (4×4 kernel).
    Bicubic,
    /// Lanczos-3 (6×6 kernel).
    Lanczos3,
}

impl ResizeFilter {
    /// Filter support radius in source pixels.
    #[must_use]
    pub const fn support(self) -> f32 {
        match self {
            Self::Nearest => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos3 => 3.0,
        }
    }
}

/// Resize a single-channel `f32` image using bilinear interpolation.
///
/// `src` has `sw × sh` pixels; the output has `dw × dh` pixels.
#[must_use]
pub fn resize_bilinear(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (dw * dh) as usize];
    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let sx = ((dx as f32 + 0.5) * x_ratio - 0.5).clamp(0.0, (sw - 1) as f32);
            let sy = ((dy as f32 + 0.5) * y_ratio - 0.5).clamp(0.0, (sh - 1) as f32);

            let x0 = (sx.floor() as u32).min(sw - 1);
            let y0 = (sy.floor() as u32).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);

            let tx = sx - x0 as f32;
            let ty = sy - y0 as f32;

            let p00 = src[(y0 * sw + x0) as usize];
            let p10 = src[(y0 * sw + x1) as usize];
            let p01 = src[(y1 * sw + x0) as usize];
            let p11 = src[(y1 * sw + x1) as usize];

            let v = p00 * (1.0 - tx) * (1.0 - ty)
                + p10 * tx * (1.0 - ty)
                + p01 * (1.0 - tx) * ty
                + p11 * tx * ty;

            dst[(dy * dw + dx) as usize] = v;
        }
    }

    dst
}

/// Crop a rectangular region from a single-channel `f32` image.
///
/// Returns a new buffer of `cw × ch` pixels starting at `(x, y)`.
///
/// Coordinates that fall outside the source image are clamped.
#[must_use]
pub fn crop(src: &[f32], sw: u32, sh: u32, x: u32, y: u32, cw: u32, ch: u32) -> Vec<f32> {
    let x = x.min(sw.saturating_sub(1));
    let y = y.min(sh.saturating_sub(1));
    let cw = cw.min(sw - x);
    let ch = ch.min(sh - y);

    let mut dst = vec![0.0f32; (cw * ch) as usize];
    for row in 0..ch {
        for col in 0..cw {
            let src_idx = ((y + row) * sw + (x + col)) as usize;
            let dst_idx = (row * cw + col) as usize;
            dst[dst_idx] = src[src_idx];
        }
    }
    dst
}

/// Flip a single-channel `f32` image horizontally (left ↔ right).
#[must_use]
pub fn flip_horizontal(src: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            dst[(row * w + (w - 1 - col)) as usize] = src[(row * w + col) as usize];
        }
    }
    dst
}

/// Flip a single-channel `f32` image vertically (top ↔ bottom).
#[must_use]
pub fn flip_vertical(src: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            dst[((h - 1 - row) * w + col) as usize] = src[(row * w + col) as usize];
        }
    }
    dst
}

/// Rotate a single-channel `f32` image 90° clockwise.
///
/// The output dimensions are swapped: width→height, height→width.
#[must_use]
pub fn rotate_90(src: &[f32], w: u32, h: u32) -> Vec<f32> {
    // Output: new_w = h, new_h = w
    let mut dst = vec![0.0f32; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            // CW: dst[col][h-1-row] = src[row][col]
            // new image is h×w (new_w=h, new_h=w)
            let dst_idx = (col * h + (h - 1 - row)) as usize;
            dst[dst_idx] = src[(row * w + col) as usize];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_apply() {
        let t = Transform2D::identity();
        let (x, y) = t.apply(3.0, 4.0);
        assert!((x - 3.0).abs() < 1e-6);
        assert!((y - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_translate() {
        let t = Transform2D::translate(5.0, -3.0);
        let (x, y) = t.apply(1.0, 1.0);
        assert!((x - 6.0).abs() < 1e-6);
        assert!((y - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let t = Transform2D::scale(2.0, 3.0);
        let (x, y) = t.apply(4.0, 5.0);
        assert!((x - 8.0).abs() < 1e-6);
        assert!((y - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_90_deg() {
        // 90° CW: (1,0) → (0,−1)
        let t = Transform2D::rotate_deg(90.0);
        let (x, y) = t.apply(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-5, "x = {x}");
        assert!((y - 1.0).abs() < 1e-5, "y = {y}");
    }

    #[test]
    fn test_compose_translate_scale() {
        let trans = Transform2D::translate(10.0, 0.0);
        let scale = Transform2D::scale(2.0, 2.0);
        // translate then scale
        let composed = trans.compose(&scale);
        let (x, y) = composed.apply(1.0, 1.0);
        // (1+10)*2 = 22, (1)*2 = 2
        assert!((x - 22.0).abs() < 1e-5);
        assert!((y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_resize_filter_support() {
        assert_eq!(ResizeFilter::Nearest.support(), 0.5);
        assert_eq!(ResizeFilter::Bilinear.support(), 1.0);
        assert_eq!(ResizeFilter::Bicubic.support(), 2.0);
        assert_eq!(ResizeFilter::Lanczos3.support(), 3.0);
    }

    #[test]
    fn test_resize_bilinear_upscale() {
        // 2×2 image upscaled to 4×4
        let src = vec![0.0, 1.0, 2.0, 3.0];
        let dst = resize_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(dst.len(), 16);
        // Corners should still match
        assert!((dst[0] - 0.0).abs() < 0.5);
        assert!((dst[3] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_resize_bilinear_same_size() {
        let src: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let dst = resize_bilinear(&src, 3, 3, 3, 3);
        // Should be roughly the same
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1.0);
        }
    }

    #[test]
    fn test_crop_basic() {
        // 4×4 image, crop 2×2 starting at (1,1)
        let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let dst = crop(&src, 4, 4, 1, 1, 2, 2);
        assert_eq!(dst.len(), 4);
        // pixel(1,1) = 5, pixel(2,1)=6, pixel(1,2)=9, pixel(2,2)=10
        assert_eq!(dst[0], 5.0);
        assert_eq!(dst[1], 6.0);
        assert_eq!(dst[2], 9.0);
        assert_eq!(dst[3], 10.0);
    }

    #[test]
    fn test_flip_horizontal() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = flip_horizontal(&src, 3, 2);
        assert_eq!(dst, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flip_vertical() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = flip_vertical(&src, 3, 2);
        assert_eq!(dst, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rotate_90_cw() {
        // 2×3 image (w=2, h=3):
        // [1 2]
        // [3 4]
        // [5 6]
        // After 90° CW → 3×2 (new_w=3, new_h=2):
        // [5 3 1]
        // [6 4 2]
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = rotate_90(&src, 2, 3);
        assert_eq!(dst.len(), 6);
        assert_eq!(dst[0], 5.0);
        assert_eq!(dst[1], 3.0);
        assert_eq!(dst[2], 1.0);
        assert_eq!(dst[3], 6.0);
        assert_eq!(dst[4], 4.0);
        assert_eq!(dst[5], 2.0);
    }

    #[test]
    fn test_identity_compose() {
        let id = Transform2D::identity();
        let t = Transform2D::translate(3.0, 4.0);
        let composed = id.compose(&t);
        let (x, y) = composed.apply(0.0, 0.0);
        assert!((x - 3.0).abs() < 1e-6);
        assert!((y - 4.0).abs() < 1e-6);
    }
}
