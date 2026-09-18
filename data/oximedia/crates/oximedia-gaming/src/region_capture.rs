//! Screen region capture primitives.
//!
//! Provides [`CaptureRegion`] (a rectangular screen area), [`RegionCapture`]
//! (a capture session bound to that region), and [`CaptureFrame`] (a single
//! captured frame).  All types are purely in-memory and platform-agnostic;
//! use [`RegionCapture::capture_synthetic`] to generate test frames.

/// A rectangular region of the screen, specified in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRegion {
    /// Left edge of the region (pixels from the left of the screen).
    pub x: u32,
    /// Top edge of the region (pixels from the top of the screen).
    pub y: u32,
    /// Width of the region in pixels.
    pub width: u32,
    /// Height of the region in pixels.
    pub height: u32,
}

impl CaptureRegion {
    /// Create a new capture region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Total pixel area of the region.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` when the region has a non-zero size.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Returns `true` when the point `(px, py)` falls inside this region.
    #[must_use]
    pub fn contains_point(&self, px: u32, py: u32) -> bool {
        px >= self.x
            && px < self.x.saturating_add(self.width)
            && py >= self.y
            && py < self.y.saturating_add(self.height)
    }

    /// Returns `true` when this region overlaps `other`.
    #[must_use]
    pub fn intersects(&self, other: &CaptureRegion) -> bool {
        let self_right = self.x.saturating_add(self.width);
        let self_bottom = self.y.saturating_add(self.height);
        let other_right = other.x.saturating_add(other.width);
        let other_bottom = other.y.saturating_add(other.height);

        self.x < other_right
            && self_right > other.x
            && self.y < other_bottom
            && self_bottom > other.y
    }
}

/// A capture session bound to a screen region.
#[derive(Debug, Clone)]
pub struct RegionCapture {
    region: CaptureRegion,
    fps: f32,
}

impl RegionCapture {
    /// Create a new region capture session.
    ///
    /// # Errors
    ///
    /// Returns an error string when `fps` is outside `(0.0, 240.0]` or when
    /// `region` is invalid (zero width or height).
    pub fn new(region: CaptureRegion, fps: f32) -> Result<Self, String> {
        if fps <= 0.0 || fps > 240.0 {
            return Err(format!("fps must be in (0, 240], got {fps}"));
        }
        if !region.is_valid() {
            return Err("region must have non-zero width and height".to_string());
        }
        Ok(Self { region, fps })
    }

    /// Generate a synthetic [`CaptureFrame`] filled with `color` (RGBA).
    ///
    /// Useful for unit testing without actual screen access.
    #[must_use]
    pub fn capture_synthetic(&self, color: [u8; 4]) -> CaptureFrame {
        let pixel_count = self.region.area() as usize;
        let mut data = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            data.push(color[0]);
            data.push(color[1]);
            data.push(color[2]);
            data.push(color[3]);
        }
        CaptureFrame::new(data, 0, self.region.width, self.region.height)
    }

    /// The capture region.
    #[must_use]
    pub fn region(&self) -> &CaptureRegion {
        &self.region
    }

    /// The configured frames-per-second.
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Expected interval between frames in milliseconds.
    #[must_use]
    pub fn frame_interval_ms(&self) -> f64 {
        1000.0 / f64::from(self.fps)
    }
}

/// A single captured frame.
#[derive(Debug, Clone)]
pub struct CaptureFrame {
    /// Raw RGBA pixel data (`width * height * 4` bytes).
    pub data: Vec<u8>,
    /// Monotonic timestamp when the frame was captured, in milliseconds.
    pub timestamp_ms: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl CaptureFrame {
    /// Create a new capture frame.
    #[must_use]
    pub fn new(data: Vec<u8>, timestamp_ms: u64, width: u32, height: u32) -> Self {
        Self {
            data,
            timestamp_ms,
            width,
            height,
        }
    }

    /// Total number of pixels in the frame.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Number of bytes in the data buffer.
    #[must_use]
    pub fn byte_count(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the data buffer holds exactly 4 bytes per pixel.
    #[must_use]
    pub fn is_rgba(&self) -> bool {
        self.data.len() == self.pixel_count() * 4
    }

    /// Return the RGBA colour of the pixel at `(x, y)`, or `None` when out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y as usize * self.width as usize + x as usize) * 4;
        if idx + 3 >= self.data.len() {
            return None;
        }
        Some([
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn region(x: u32, y: u32, w: u32, h: u32) -> CaptureRegion {
        CaptureRegion::new(x, y, w, h)
    }

    // --- CaptureRegion ---

    #[test]
    fn test_region_area() {
        assert_eq!(region(0, 0, 100, 200).area(), 20_000);
    }

    #[test]
    fn test_region_area_zero_width() {
        assert_eq!(region(0, 0, 0, 200).area(), 0);
    }

    #[test]
    fn test_region_is_valid() {
        assert!(region(0, 0, 1, 1).is_valid());
        assert!(!region(0, 0, 0, 100).is_valid());
        assert!(!region(0, 0, 100, 0).is_valid());
    }

    #[test]
    fn test_contains_point_inside() {
        let r = region(10, 20, 100, 80);
        assert!(r.contains_point(10, 20));
        assert!(r.contains_point(50, 50));
        assert!(r.contains_point(109, 99));
    }

    #[test]
    fn test_contains_point_outside() {
        let r = region(10, 20, 100, 80);
        assert!(!r.contains_point(9, 20));
        assert!(!r.contains_point(110, 20));
        assert!(!r.contains_point(10, 100));
    }

    #[test]
    fn test_intersects_overlapping() {
        let a = region(0, 0, 100, 100);
        let b = region(50, 50, 100, 100);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn test_intersects_adjacent_no_overlap() {
        let a = region(0, 0, 100, 100);
        let b = region(100, 0, 100, 100);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_intersects_completely_separate() {
        let a = region(0, 0, 50, 50);
        let b = region(100, 100, 50, 50);
        assert!(!a.intersects(&b));
    }

    // --- RegionCapture ---

    #[test]
    fn test_region_capture_new_valid() {
        let rc = RegionCapture::new(region(0, 0, 1920, 1080), 60.0);
        assert!(rc.is_ok());
    }

    #[test]
    fn test_region_capture_fps_zero_errors() {
        let rc = RegionCapture::new(region(0, 0, 1920, 1080), 0.0);
        assert!(rc.is_err());
    }

    #[test]
    fn test_region_capture_fps_negative_errors() {
        let rc = RegionCapture::new(region(0, 0, 1920, 1080), -1.0);
        assert!(rc.is_err());
    }

    #[test]
    fn test_region_capture_fps_above_240_errors() {
        let rc = RegionCapture::new(region(0, 0, 1920, 1080), 241.0);
        assert!(rc.is_err());
    }

    #[test]
    fn test_region_capture_invalid_region_errors() {
        let rc = RegionCapture::new(region(0, 0, 0, 1080), 30.0);
        assert!(rc.is_err());
    }

    #[test]
    fn test_frame_interval_ms() {
        let rc = RegionCapture::new(region(0, 0, 1920, 1080), 60.0).expect("valid");
        let interval = rc.frame_interval_ms();
        assert!((interval - 1000.0 / 60.0).abs() < 1e-6);
    }

    // --- capture_synthetic ---

    #[test]
    fn test_capture_synthetic_pixel_values() {
        let rc = RegionCapture::new(region(0, 0, 4, 4), 30.0).expect("valid");
        let color = [255u8, 128, 64, 255];
        let frame = rc.capture_synthetic(color);
        let px = frame.get_pixel(2, 2).expect("pixel should exist");
        assert_eq!(px, color);
    }

    #[test]
    fn test_capture_synthetic_is_rgba() {
        let rc = RegionCapture::new(region(0, 0, 16, 9), 30.0).expect("valid");
        let frame = rc.capture_synthetic([0, 0, 0, 255]);
        assert!(frame.is_rgba());
    }

    // --- CaptureFrame ---

    #[test]
    fn test_frame_pixel_count() {
        let data = vec![0u8; 1920 * 1080 * 4];
        let frame = CaptureFrame::new(data, 0, 1920, 1080);
        assert_eq!(frame.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_frame_get_pixel_out_of_bounds() {
        let data = vec![0u8; 4 * 4 * 4];
        let frame = CaptureFrame::new(data, 0, 4, 4);
        assert!(frame.get_pixel(4, 0).is_none());
        assert!(frame.get_pixel(0, 4).is_none());
    }

    #[test]
    fn test_capture_synthetic_top_left_pixel() {
        let rc = RegionCapture::new(region(0, 0, 10, 10), 25.0).expect("valid");
        let color = [10u8, 20, 30, 40];
        let frame = rc.capture_synthetic(color);
        assert_eq!(frame.get_pixel(0, 0), Some(color));
    }

    #[test]
    fn test_capture_synthetic_bottom_right_pixel() {
        let rc = RegionCapture::new(region(0, 0, 8, 8), 25.0).expect("valid");
        let color = [1u8, 2, 3, 4];
        let frame = rc.capture_synthetic(color);
        assert_eq!(frame.get_pixel(7, 7), Some(color));
    }
}
