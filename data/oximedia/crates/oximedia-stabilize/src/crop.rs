//! Crop and zoom management for video stabilization.
//!
//! After stabilization the frame edges may contain black borders introduced by
//! the warp transforms.  This module provides utilities for computing the
//! minimum required zoom and an optimal crop window that hides those borders.

#![allow(dead_code)]

/// A rectangular crop window described in normalised [0, 1] coordinates.
///
/// `(x, y)` is the top-left corner; `(width, height)` are the extents.
/// All values must be in `[0, 1]` and the window must fit within the frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CropWindow {
    /// Normalised x coordinate of the left edge.
    pub x: f64,
    /// Normalised y coordinate of the top edge.
    pub y: f64,
    /// Normalised width.
    pub width: f64,
    /// Normalised height.
    pub height: f64,
}

impl CropWindow {
    /// Creates a new `CropWindow`.
    #[must_use]
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            x,
            y,
            width: w,
            height: h,
        }
    }

    /// Returns a centred crop window that covers the entire frame.
    #[must_use]
    pub fn center() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }

    /// Returns a new `CropWindow` scaled by `factor` around its own centre.
    ///
    /// A factor < 1.0 zooms in (smaller window, more crop).
    /// A factor > 1.0 zooms out (larger window, potentially outside frame).
    #[must_use]
    pub fn scale_by(&self, factor: f64) -> Self {
        let cx = self.x + self.width * 0.5;
        let cy = self.y + self.height * 0.5;
        let new_w = self.width * factor;
        let new_h = self.height * factor;
        Self::new(cx - new_w * 0.5, cy - new_h * 0.5, new_w, new_h)
    }

    /// Returns the area of this crop window in normalised units.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Returns `true` if the normalised point `(px, py)` falls inside this window.
    #[must_use]
    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Clamps the window so it stays within `[0, 1] x [0, 1]`.
    #[must_use]
    pub fn clamped(&self) -> Self {
        let x = self.x.max(0.0);
        let y = self.y.max(0.0);
        let w = (self.width + self.x - x).min(1.0 - x).max(0.0);
        let h = (self.height + self.y - y).min(1.0 - y).max(0.0);
        Self::new(x, y, w, h)
    }
}

impl Default for CropWindow {
    fn default() -> Self {
        Self::center()
    }
}

/// A stabilization transform applied to a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilTransform {
    /// Horizontal translation in pixels.
    pub dx: f64,
    /// Vertical translation in pixels.
    pub dy: f64,
    /// Rotation in degrees.
    pub rotation: f64,
    /// Uniform scale factor (1.0 = no zoom).
    pub scale: f64,
}

impl StabilTransform {
    /// Creates an identity transform (no correction).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            rotation: 0.0,
            scale: 1.0,
        }
    }
}

impl Default for StabilTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Computes the minimum zoom factor required to hide black borders introduced
/// by a maximum translation and rotation.
///
/// # Arguments
///
/// * `max_translation` – The maximum pixel shift in any direction.
/// * `max_rotation_deg` – The maximum rotation in degrees.
/// * `width` – Frame width in pixels.
/// * `height` – Frame height in pixels.
///
/// The result is always >= 1.0.
#[must_use]
pub fn compute_required_zoom(
    max_translation: f64,
    max_rotation_deg: f64,
    width: u32,
    height: u32,
) -> f64 {
    if width == 0 || height == 0 {
        return 1.0;
    }
    let w = width as f64;
    let h = height as f64;

    // Translation component: fraction of frame size that is shifted.
    let tx_fraction = max_translation / w;
    let ty_fraction = max_translation / h;
    let translation_zoom = 1.0 + tx_fraction.max(ty_fraction) * 2.0;

    // Rotation component: extra zoom to cover the rotated corners.
    let theta = max_rotation_deg.to_radians().abs();
    let half_diag = (w * w + h * h).sqrt() * 0.5;
    let diagonal_zoom = if theta < 1e-10 {
        1.0
    } else {
        // The farthest corner sweeps out an arc; the required scale is the
        // ratio of the original half-diagonal to the nearest edge after rotation.
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        // Required scale so that the rotated frame covers the output frame:
        //   scale >= max(w, h) / (w * cos_t - h * sin_t) ... simplified approach
        let _ = half_diag; // suppress unused
        let scale_x = 1.0 / (cos_t - (h / w) * sin_t).abs().max(1e-10);
        let scale_y = 1.0 / (cos_t - (w / h) * sin_t).abs().max(1e-10);
        scale_x.min(scale_y).max(1.0)
    };

    translation_zoom.max(diagonal_zoom).max(1.0)
}

/// Computes the optimal crop window for a sequence of stabilization transforms,
/// so that the stabilised output avoids black borders while respecting
/// `zoom_limit` (the maximum allowed zoom factor, e.g. 1.2 = 120%).
///
/// The crop window is centred and symmetrical; it is expressed in normalised
/// `[0, 1]` coordinates.
///
/// If no transforms are provided, the full-frame window is returned.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn optimal_crop_window(
    stabilized_transforms: &[StabilTransform],
    width: u32,
    height: u32,
    zoom_limit: f64,
) -> CropWindow {
    if stabilized_transforms.is_empty() || width == 0 || height == 0 {
        return CropWindow::center();
    }

    let w = width as f64;
    let h = height as f64;

    // Find the worst-case translation and rotation across all frames.
    let max_dx = stabilized_transforms
        .iter()
        .map(|t| t.dx.abs())
        .fold(0.0_f64, f64::max);
    let max_dy = stabilized_transforms
        .iter()
        .map(|t| t.dy.abs())
        .fold(0.0_f64, f64::max);
    let max_rot = stabilized_transforms
        .iter()
        .map(|t| t.rotation.abs())
        .fold(0.0_f64, f64::max);

    let required_zoom = compute_required_zoom(max_dx.max(max_dy), max_rot, width, height);
    let zoom = required_zoom.min(zoom_limit).max(1.0);

    // The crop window shrinks by 1/zoom on each axis.
    let crop_w = 1.0 / zoom;
    let crop_h = 1.0 / zoom;

    // Additional inset from translation (normalised).
    let inset_x = (max_dx / w).min((1.0 - crop_w) * 0.5);
    let inset_y = (max_dy / h).min((1.0 - crop_h) * 0.5);

    let x = ((1.0 - crop_w) * 0.5 + inset_x).max(0.0);
    let y = ((1.0 - crop_h) * 0.5 + inset_y).max(0.0);
    let effective_w = (crop_w - 2.0 * inset_x).max(0.01);
    let effective_h = (crop_h - 2.0 * inset_y).max(0.01);

    CropWindow::new(x, y, effective_w, effective_h).clamped()
}

/// Applies a crop-and-scale operation: extracts the region described by `crop`
/// from `frame` (in RGBA or any 4-byte-per-pixel format) and writes the
/// scaled result into `dst`.
///
/// `src_w` / `src_h` are the source frame dimensions; `dst_w` / `dst_h` are
/// the destination dimensions.  Uses nearest-neighbour scaling for speed.
#[allow(clippy::too_many_arguments)]
pub fn apply_crop(
    frame: &[u8],
    src_w: u32,
    src_h: u32,
    crop: &CropWindow,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let bytes_per_pixel = 4usize;
    let mut dst = vec![0u8; (dst_w * dst_h) as usize * bytes_per_pixel];

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return dst;
    }

    let crop_x0 = (crop.x * src_w as f64).round() as u32;
    let crop_y0 = (crop.y * src_h as f64).round() as u32;
    let crop_w = ((crop.width * src_w as f64).round() as u32).min(src_w - crop_x0);
    let crop_h = ((crop.height * src_h as f64).round() as u32).min(src_h - crop_y0);

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel back to crop-region pixel.
            let sx = crop_x0 + dx * crop_w / dst_w.max(1);
            let sy = crop_y0 + dy * crop_h / dst_h.max(1);
            let sx = sx.min(src_w - 1);
            let sy = sy.min(src_h - 1);

            let src_idx = (sy * src_w + sx) as usize * bytes_per_pixel;
            let dst_idx = (dy * dst_w + dx) as usize * bytes_per_pixel;

            if src_idx + bytes_per_pixel <= frame.len() && dst_idx + bytes_per_pixel <= dst.len() {
                dst[dst_idx..dst_idx + bytes_per_pixel]
                    .copy_from_slice(&frame[src_idx..src_idx + bytes_per_pixel]);
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_window_area() {
        let w = CropWindow::new(0.0, 0.0, 0.8, 0.9);
        assert!((w.area() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_crop_window_contains() {
        let w = CropWindow::new(0.1, 0.1, 0.8, 0.8);
        assert!(w.contains(0.5, 0.5));
        assert!(!w.contains(0.05, 0.5));
        assert!(!w.contains(0.5, 0.95));
    }

    #[test]
    fn test_crop_window_center() {
        let w = CropWindow::center();
        assert!((w.x).abs() < 1e-10);
        assert!((w.y).abs() < 1e-10);
        assert!((w.width - 1.0).abs() < 1e-10);
        assert!((w.height - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crop_window_scale_by_reduces_size() {
        let w = CropWindow::new(0.0, 0.0, 1.0, 1.0);
        let scaled = w.scale_by(0.5);
        assert!((scaled.width - 0.5).abs() < 1e-10);
        assert!((scaled.height - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_crop_window_scale_by_centred() {
        let w = CropWindow::new(0.0, 0.0, 1.0, 1.0);
        let scaled = w.scale_by(0.5);
        // Centre should remain at (0.5, 0.5)
        let cx = scaled.x + scaled.width * 0.5;
        let cy = scaled.y + scaled.height * 0.5;
        assert!((cx - 0.5).abs() < 1e-10);
        assert!((cy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_required_zoom_no_motion() {
        let zoom = compute_required_zoom(0.0, 0.0, 1920, 1080);
        assert!(zoom >= 1.0);
    }

    #[test]
    fn test_compute_required_zoom_increases_with_translation() {
        let z1 = compute_required_zoom(10.0, 0.0, 1920, 1080);
        let z2 = compute_required_zoom(50.0, 0.0, 1920, 1080);
        assert!(z2 > z1);
    }

    #[test]
    fn test_compute_required_zoom_zero_dimensions() {
        let zoom = compute_required_zoom(10.0, 5.0, 0, 0);
        assert!((zoom - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_crop_window_no_transforms() {
        let w = optimal_crop_window(&[], 1920, 1080, 1.5);
        assert!((w.x).abs() < 1e-10);
        assert!((w.width - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_crop_window_identity_transforms() {
        let transforms = vec![StabilTransform::identity(); 10];
        let w = optimal_crop_window(&transforms, 1920, 1080, 1.5);
        // With no motion the window should be close to full frame.
        assert!(w.width > 0.9);
        assert!(w.height > 0.9);
    }

    #[test]
    fn test_apply_crop_output_size() {
        let src = vec![128u8; 4 * 100 * 100];
        let crop = CropWindow::new(0.1, 0.1, 0.8, 0.8);
        let dst = apply_crop(&src, 100, 100, &crop, 80, 80);
        assert_eq!(dst.len(), 4 * 80 * 80);
    }
}
