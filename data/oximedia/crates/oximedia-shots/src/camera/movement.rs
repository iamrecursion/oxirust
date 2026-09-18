//! Camera movement detection and analysis.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};
use crate::types::{CameraMovement, MovementType};

/// Camera movement detector.
pub struct MovementDetector {
    /// Threshold for movement detection.
    threshold: f32,
    /// Minimum duration for valid movement (frames).
    min_duration: usize,
}

impl MovementDetector {
    /// Create a new movement detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.1,
            min_duration: 5,
        }
    }

    /// Detect camera movements in a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or have mismatched dimensions.
    pub fn detect_movements(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<CameraMovement>> {
        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        let mut movements = Vec::new();

        // Calculate optical flow between consecutive frames
        let mut flow_vectors = Vec::new();
        for i in 1..frames.len() {
            let flow = self.calculate_optical_flow(&frames[i - 1], &frames[i])?;
            flow_vectors.push(flow);
        }

        // Analyze flow patterns to detect movements
        let mut i = 0;
        while i < flow_vectors.len() {
            if let Some((movement_type, duration, speed)) =
                self.analyze_flow_pattern(&flow_vectors[i..])
            {
                let start = i as f64 / 30.0; // Assuming 30 fps
                let end = start + (duration as f64 / 30.0);

                movements.push(CameraMovement {
                    movement_type,
                    start,
                    end,
                    confidence: 0.8,
                    speed,
                });

                i += duration;
            } else {
                i += 1;
            }
        }

        Ok(movements)
    }

    /// Calculate optical flow between two frames using proxy downscaling.
    ///
    /// Both frames are box-downsampled to a 160-pixel-wide proxy before running
    /// Lucas-Kanade gradient estimation.  The resulting motion vectors are scaled
    /// back to full-resolution coordinates before being returned, giving the same
    /// semantic meaning at a fraction of the computational cost.
    fn calculate_optical_flow(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
    ) -> ShotResult<(f32, f32)> {
        if frame1.dim() != frame2.dim() {
            return Err(ShotError::InvalidFrame(
                "Frame dimensions do not match".to_string(),
            ));
        }

        let shape = frame1.dim();
        let full_w = shape.1 as u32;
        let full_h = shape.0 as u32;

        // Convert to grayscale then build proxy frames for optical flow.
        let gray1 = self.to_grayscale(frame1);
        let gray2 = self.to_grayscale(frame2);

        let (proxy1, proxy_w, proxy_h) = box_downsample_to_proxy(&gray1, full_w, full_h);
        let (proxy2, _, _) = box_downsample_to_proxy(&gray2, full_w, full_h);

        // Build GrayImage wrappers around proxy data.
        let pg1 = GrayImageProxy::new(proxy1, proxy_h as usize, proxy_w as usize);
        let pg2 = GrayImageProxy::new(proxy2, proxy_h as usize, proxy_w as usize);

        let pw = proxy_w as usize;
        let ph = proxy_h as usize;

        let mut dx_sum = 0.0_f32;
        let mut dy_sum = 0.0_f32;
        let mut count = 0u32;

        // Sample grid points on the proxy.
        for y in (5..ph.saturating_sub(5)).step_by(5) {
            for x in (5..pw.saturating_sub(5)).step_by(5) {
                if let Some((dx, dy)) = self.compute_local_flow_gray(&pg1, &pg2, x, y) {
                    dx_sum += dx;
                    dy_sum += dy;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok((0.0, 0.0));
        }

        // Scale motion vectors back to full-resolution coordinates.
        let scale_x = full_w as f32 / proxy_w as f32;
        let scale_y = full_h as f32 / proxy_h as f32;

        Ok((
            dx_sum / count as f32 * scale_x,
            dy_sum / count as f32 * scale_y,
        ))
    }

    /// Compute local optical flow at a point (operates on [`GrayImageProxy`]).
    fn compute_local_flow_gray(
        &self,
        gray1: &GrayImageProxy,
        gray2: &GrayImageProxy,
        x: usize,
        y: usize,
    ) -> Option<(f32, f32)> {
        let window_size = 5;
        let (ph, pw) = (gray1.height, gray1.width);

        if y < window_size
            || y >= ph.saturating_sub(window_size)
            || x < window_size
            || x >= pw.saturating_sub(window_size)
        {
            return None;
        }

        // Compute image gradients (central difference).
        let ix = (f32::from(gray1.get(y, x + 1)) - f32::from(gray1.get(y, x - 1))) / 2.0;
        let iy = (f32::from(gray1.get(y + 1, x)) - f32::from(gray1.get(y - 1, x))) / 2.0;
        let it = f32::from(gray2.get(y, x)) - f32::from(gray1.get(y, x));

        // Solve for flow using least-squares (Lucas-Kanade single-point form).
        let denom = ix * ix + iy * iy;
        if denom < 1.0 {
            return None;
        }

        let vx = -(ix * it) / denom;
        let vy = -(iy * it) / denom;

        Some((vx, vy))
    }

    /// Analyze flow pattern to determine movement type.
    fn analyze_flow_pattern(&self, flows: &[(f32, f32)]) -> Option<(MovementType, usize, f32)> {
        if flows.len() < self.min_duration {
            return None;
        }

        // Calculate average flow
        let mut avg_dx = 0.0;
        let mut avg_dy = 0.0;

        for (dx, dy) in flows.iter().take(self.min_duration) {
            avg_dx += dx;
            avg_dy += dy;
        }

        avg_dx /= self.min_duration as f32;
        avg_dy /= self.min_duration as f32;

        let magnitude = (avg_dx * avg_dx + avg_dy * avg_dy).sqrt();

        if magnitude < self.threshold {
            return Some((MovementType::Static, self.min_duration, 0.0));
        }

        // Determine movement type based on flow direction
        let movement_type = if avg_dx.abs() > avg_dy.abs() * 2.0 {
            if avg_dx > 0.0 {
                MovementType::PanRight
            } else {
                MovementType::PanLeft
            }
        } else if avg_dy.abs() > avg_dx.abs() * 2.0 {
            if avg_dy > 0.0 {
                MovementType::TiltDown
            } else {
                MovementType::TiltUp
            }
        } else {
            // Check for zoom or dolly
            if self.is_zoom_pattern(flows) {
                if magnitude > 0.0 {
                    MovementType::ZoomIn
                } else {
                    MovementType::ZoomOut
                }
            } else {
                MovementType::Handheld
            }
        };

        Some((movement_type, self.min_duration, magnitude))
    }

    /// Check if flow pattern indicates zoom.
    fn is_zoom_pattern(&self, flows: &[(f32, f32)]) -> bool {
        if flows.len() < 3 {
            return false;
        }

        // Zoom creates radial flow from center
        let center_x = 0.0; // Assuming normalized coordinates
        let center_y: f32 = 0.0;

        let mut radial_consistency = 0.0;

        for (dx, dy) in flows.iter().take(self.min_duration.min(flows.len())) {
            let angle = dy.atan2(*dx);
            let expected_angle = center_y.atan2(center_x);
            let diff = (angle - expected_angle).abs();

            if diff < 0.5 {
                radial_consistency += 1.0;
            }
        }

        radial_consistency / self.min_duration as f32 > 0.6
    }

    /// Convert RGB to grayscale.
    fn to_grayscale(&self, frame: &FrameBuffer) -> GrayImage {
        let shape = frame.dim();
        let mut gray = GrayImage::zeros(shape.0, shape.1);

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
            }
        }

        gray
    }
}

impl Default for MovementDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Proxy downsampling helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Target proxy width in pixels used for optical-flow estimation.
pub const PROXY_WIDTH: u32 = 160;

/// Lightweight grayscale image wrapper used for proxy-resolution optical flow.
///
/// Avoids allocating a full [`GrayImage`] when operating on proxy frames stored
/// as flat `Vec<u8>`.
struct GrayImageProxy {
    data: Vec<u8>,
    height: usize,
    width: usize,
}

impl GrayImageProxy {
    fn new(data: Vec<u8>, height: usize, width: usize) -> Self {
        Self {
            data,
            height,
            width,
        }
    }

    #[inline]
    fn get(&self, y: usize, x: usize) -> u8 {
        self.data[y * self.width + x]
    }
}

/// Box-downsample a [`GrayImage`] to a proxy whose width is [`PROXY_WIDTH`],
/// preserving the original aspect ratio.
///
/// Returns `(pixel_data, proxy_width, proxy_height)`.  When the source is
/// already narrower than the target, the source is returned unchanged.
fn box_downsample_to_proxy(gray: &GrayImage, src_w: u32, src_h: u32) -> (Vec<u8>, u32, u32) {
    if src_w == 0 || src_h == 0 {
        return (Vec::new(), 0, 0);
    }

    // If the frame is already at or below the proxy size, return as-is.
    if src_w <= PROXY_WIDTH {
        let (gh, gw) = gray.dim();
        let mut data = Vec::with_capacity(gh * gw);
        for y in 0..gh {
            for x in 0..gw {
                data.push(gray.get(y, x));
            }
        }
        return (data, src_w, src_h);
    }

    let proxy_w = PROXY_WIDTH;
    let proxy_h = (src_h * proxy_w / src_w).max(1);

    // Block dimensions (truncated to whole pixels).
    let bx = src_w / proxy_w;
    let by = src_h / proxy_h;
    // Guard against zero block sizes.
    let bx = bx.max(1);
    let by = by.max(1);

    let mut dst = vec![0u8; (proxy_w * proxy_h) as usize];

    for py in 0..proxy_h {
        for px in 0..proxy_w {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in 0..by {
                for dx in 0..bx {
                    let sy = py * by + dy;
                    let sx = px * bx + dx;
                    if sy < src_h && sx < src_w {
                        sum += u32::from(gray.get(sy as usize, sx as usize));
                        count += 1;
                    }
                }
            }
            dst[(py * proxy_w + px) as usize] = sum.checked_div(count).map_or(0, |v| v as u8);
        }
    }

    (dst, proxy_w, proxy_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_detector_creation() {
        let detector = MovementDetector::new();
        assert!((detector.threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_movement_single_frame() {
        let detector = MovementDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_movements(&frames);
        assert!(result.is_ok());
        if let Ok(movements) = result {
            assert!(movements.is_empty());
        }
    }

    #[test]
    fn test_static_frames() {
        let detector = MovementDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3); 10];
        let result = detector.detect_movements(&frames);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proxy_flow_direction_matches_full() {
        // Two grayscale 640×360 frames: frame2 is frame1 shifted right by 8px.
        // We expect the estimated horizontal motion to be positive (rightward).
        let height = 360usize;
        let width = 640usize;
        let channels = 3usize;
        let shift = 8usize;

        // Build a frame with a diagonal brightness ramp to create detectable gradients.
        let mut frame1 = FrameBuffer::zeros(height, width, channels);
        for y in 0..height {
            for x in 0..width {
                let v = ((x + y) % 256) as u8;
                for c in 0..channels {
                    frame1.set(y, x, c, v);
                }
            }
        }
        // frame2: shift frame1 right by `shift` pixels (fill left edge with zeros).
        let mut frame2 = FrameBuffer::zeros(height, width, channels);
        for y in 0..height {
            for x in shift..width {
                for c in 0..channels {
                    let v = frame1.get(y, x - shift, c);
                    frame2.set(y, x, c, v);
                }
            }
        }

        let detector = MovementDetector::new();
        let result = detector.calculate_optical_flow(&frame1, &frame2);
        assert!(result.is_ok(), "optical flow should not error");
        let (dx, _dy) = result.expect("ok");
        // The shift is rightward so the estimated dx should be positive.
        assert!(
            dx > 0.0,
            "expected positive dx for rightward shift, got {dx}"
        );
    }

    #[test]
    fn test_box_downsample_to_proxy_aspect_ratio() {
        // A 640×360 grayscale image should downsample to 160×90.
        let gray = GrayImage::zeros(360, 640);
        let (_, pw, ph) = box_downsample_to_proxy(&gray, 640, 360);
        assert_eq!(pw, PROXY_WIDTH);
        assert_eq!(ph, 90);
    }

    #[test]
    fn test_box_downsample_preserves_small_frames() {
        // Frames already narrower than PROXY_WIDTH should pass through unchanged.
        let gray = GrayImage::zeros(90, 100);
        let (data, pw, ph) = box_downsample_to_proxy(&gray, 100, 90);
        assert_eq!(pw, 100);
        assert_eq!(ph, 90);
        assert_eq!(data.len(), 100 * 90);
    }
}
