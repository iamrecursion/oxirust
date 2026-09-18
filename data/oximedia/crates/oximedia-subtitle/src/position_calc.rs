#![allow(dead_code)]
//! Position calculation for subtitle placement on screen.
//!
//! Computes subtitle bounding boxes, safe-area positioning,
//! collision avoidance, and multi-line layout coordinates for
//! rendering subtitles onto video frames.

/// Horizontal alignment for subtitle text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HAlign {
    /// Left-aligned.
    Left,
    /// Center-aligned.
    Center,
    /// Right-aligned.
    Right,
}

/// Vertical alignment for subtitle text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VAlign {
    /// Top of the screen.
    Top,
    /// Middle of the screen.
    Middle,
    /// Bottom of the screen.
    Bottom,
}

/// Represents a rectangular region on screen in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Left edge (x).
    pub x: f64,
    /// Top edge (y).
    pub y: f64,
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge coordinate.
    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    /// Bottom edge coordinate.
    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    /// Center X coordinate.
    pub fn center_x(&self) -> f64 {
        self.x + self.width / 2.0
    }

    /// Center Y coordinate.
    pub fn center_y(&self) -> f64 {
        self.y + self.height / 2.0
    }

    /// Area of the rectangle.
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Check if two rectangles overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Check if this rectangle fully contains another.
    pub fn contains_rect(&self, other: &Self) -> bool {
        self.x <= other.x
            && self.y <= other.y
            && self.right() >= other.right()
            && self.bottom() >= other.bottom()
    }

    /// Check if a point is inside this rectangle.
    pub fn contains_point(&self, px: f64, py: f64) -> bool {
        px >= self.x && px <= self.right() && py >= self.y && py <= self.bottom()
    }
}

/// Title-safe area margins (percentage of frame dimensions).
#[derive(Clone, Copy, Debug)]
pub struct SafeArea {
    /// Margin from left edge (0.0-0.5).
    pub left: f64,
    /// Margin from right edge (0.0-0.5).
    pub right: f64,
    /// Margin from top edge (0.0-0.5).
    pub top: f64,
    /// Margin from bottom edge (0.0-0.5).
    pub bottom: f64,
}

impl SafeArea {
    /// Standard broadcast title-safe area (10% margins).
    pub fn broadcast() -> Self {
        Self {
            left: 0.1,
            right: 0.1,
            top: 0.1,
            bottom: 0.1,
        }
    }

    /// Streaming/web safe area (5% margins).
    pub fn streaming() -> Self {
        Self {
            left: 0.05,
            right: 0.05,
            top: 0.05,
            bottom: 0.05,
        }
    }

    /// Compute the safe rectangle for a given frame size.
    pub fn to_rect(&self, frame_width: f64, frame_height: f64) -> Rect {
        let x = frame_width * self.left;
        let y = frame_height * self.top;
        let w = frame_width * (1.0 - self.left - self.right);
        let h = frame_height * (1.0 - self.top - self.bottom);
        Rect::new(x, y, w, h)
    }
}

impl Default for SafeArea {
    fn default() -> Self {
        Self::broadcast()
    }
}

/// Parameters for positioning a subtitle block.
#[derive(Clone, Debug)]
pub struct PositionParams {
    /// Frame width in pixels.
    pub frame_width: f64,
    /// Frame height in pixels.
    pub frame_height: f64,
    /// Horizontal alignment.
    pub h_align: HAlign,
    /// Vertical alignment.
    pub v_align: VAlign,
    /// Safe area margins.
    pub safe_area: SafeArea,
    /// Additional vertical offset from the aligned position (pixels).
    pub vertical_offset: f64,
    /// Additional horizontal offset from the aligned position (pixels).
    pub horizontal_offset: f64,
}

impl PositionParams {
    /// Create default positioning for a given frame size.
    pub fn new(frame_width: f64, frame_height: f64) -> Self {
        Self {
            frame_width,
            frame_height,
            h_align: HAlign::Center,
            v_align: VAlign::Bottom,
            safe_area: SafeArea::broadcast(),
            vertical_offset: 0.0,
            horizontal_offset: 0.0,
        }
    }
}

/// Compute the positioned rectangle for a subtitle text block.
///
/// `text_width` and `text_height` are the measured dimensions of the rendered text.
pub fn compute_position(params: &PositionParams, text_width: f64, text_height: f64) -> Rect {
    let safe = params
        .safe_area
        .to_rect(params.frame_width, params.frame_height);

    let x = match params.h_align {
        HAlign::Left => safe.x + params.horizontal_offset,
        HAlign::Center => safe.center_x() - text_width / 2.0 + params.horizontal_offset,
        HAlign::Right => safe.right() - text_width + params.horizontal_offset,
    };

    let y = match params.v_align {
        VAlign::Top => safe.y + params.vertical_offset,
        VAlign::Middle => safe.center_y() - text_height / 2.0 + params.vertical_offset,
        VAlign::Bottom => safe.bottom() - text_height + params.vertical_offset,
    };

    Rect::new(x, y, text_width, text_height)
}

/// Move a subtitle rectangle so it does not overlap with existing ones.
///
/// Tries to shift upward first, then downward.
pub fn avoid_collision(candidate: &Rect, existing: &[Rect], frame_height: f64) -> Rect {
    if existing.iter().all(|e| !candidate.overlaps(e)) {
        return *candidate;
    }

    // Try shifting upward
    let mut adjusted = *candidate;
    for _ in 0..50 {
        adjusted.y -= candidate.height + 2.0;
        if adjusted.y < 0.0 {
            break;
        }
        if existing.iter().all(|e| !adjusted.overlaps(e)) {
            return adjusted;
        }
    }

    // Try shifting downward from original
    adjusted = *candidate;
    for _ in 0..50 {
        adjusted.y += candidate.height + 2.0;
        if adjusted.bottom() > frame_height {
            break;
        }
        if existing.iter().all(|e| !adjusted.overlaps(e)) {
            return adjusted;
        }
    }

    // Return original as fallback
    *candidate
}

/// Compute Y positions for multi-line subtitle layout.
///
/// Returns the Y coordinate for each line, starting from `base_y`.
pub fn multi_line_y_positions(
    line_count: usize,
    line_height: f64,
    line_spacing: f64,
    base_y: f64,
) -> Vec<f64> {
    (0..line_count)
        .map(|i| base_y + i as f64 * (line_height + line_spacing))
        .collect()
}

/// Compute total height needed for multi-line subtitles.
pub fn multi_line_total_height(line_count: usize, line_height: f64, line_spacing: f64) -> f64 {
    if line_count == 0 {
        return 0.0;
    }
    line_count as f64 * line_height + (line_count as f64 - 1.0) * line_spacing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_right_bottom() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!((r.right() - 110.0).abs() < f64::EPSILON);
        assert!((r.bottom() - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rect_center() {
        let r = Rect::new(0.0, 0.0, 100.0, 80.0);
        assert!((r.center_x() - 50.0).abs() < f64::EPSILON);
        assert!((r.center_y() - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        assert!((r.area() - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rect_overlaps() {
        let a = Rect::new(0.0, 0.0, 100.0, 50.0);
        let b = Rect::new(50.0, 25.0, 100.0, 50.0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_rect_no_overlap() {
        let a = Rect::new(0.0, 0.0, 50.0, 50.0);
        let b = Rect::new(100.0, 100.0, 50.0, 50.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_rect_contains_rect() {
        let outer = Rect::new(0.0, 0.0, 200.0, 200.0);
        let inner = Rect::new(10.0, 10.0, 50.0, 50.0);
        assert!(outer.contains_rect(&inner));
        assert!(!inner.contains_rect(&outer));
    }

    #[test]
    fn test_rect_contains_point() {
        let r = Rect::new(10.0, 10.0, 100.0, 50.0);
        assert!(r.contains_point(50.0, 30.0));
        assert!(!r.contains_point(5.0, 30.0));
    }

    #[test]
    fn test_safe_area_broadcast() {
        let sa = SafeArea::broadcast();
        let r = sa.to_rect(1920.0, 1080.0);
        assert!((r.x - 192.0).abs() < 1e-10);
        assert!((r.y - 108.0).abs() < 1e-10);
        assert!((r.width - 1536.0).abs() < 1e-10);
        assert!((r.height - 864.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_area_streaming() {
        let sa = SafeArea::streaming();
        let r = sa.to_rect(1920.0, 1080.0);
        assert!((r.x - 96.0).abs() < 1e-10);
        assert!((r.y - 54.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_position_center_bottom() {
        let params = PositionParams::new(1920.0, 1080.0);
        let rect = compute_position(&params, 400.0, 60.0);
        // Center horizontally in safe area
        let safe = params.safe_area.to_rect(1920.0, 1080.0);
        let expected_x = safe.center_x() - 200.0;
        assert!((rect.x - expected_x).abs() < 1e-10);
        // Bottom of safe area
        let expected_y = safe.bottom() - 60.0;
        assert!((rect.y - expected_y).abs() < 1e-10);
    }

    #[test]
    fn test_compute_position_left_top() {
        let mut params = PositionParams::new(1920.0, 1080.0);
        params.h_align = HAlign::Left;
        params.v_align = VAlign::Top;
        let rect = compute_position(&params, 200.0, 40.0);
        let safe = params.safe_area.to_rect(1920.0, 1080.0);
        assert!((rect.x - safe.x).abs() < 1e-10);
        assert!((rect.y - safe.y).abs() < 1e-10);
    }

    #[test]
    fn test_avoid_collision_no_overlap() {
        let candidate = Rect::new(100.0, 900.0, 200.0, 40.0);
        let existing = vec![Rect::new(100.0, 100.0, 200.0, 40.0)];
        let result = avoid_collision(&candidate, &existing, 1080.0);
        assert!((result.y - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avoid_collision_with_overlap() {
        let candidate = Rect::new(100.0, 900.0, 200.0, 40.0);
        let existing = vec![Rect::new(100.0, 895.0, 200.0, 40.0)];
        let result = avoid_collision(&candidate, &existing, 1080.0);
        assert!(!result.overlaps(&existing[0]));
    }

    #[test]
    fn test_multi_line_y_positions() {
        let positions = multi_line_y_positions(3, 30.0, 5.0, 100.0);
        assert_eq!(positions.len(), 3);
        assert!((positions[0] - 100.0).abs() < f64::EPSILON);
        assert!((positions[1] - 135.0).abs() < f64::EPSILON);
        assert!((positions[2] - 170.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_line_total_height() {
        let h = multi_line_total_height(3, 30.0, 5.0);
        assert!((h - 100.0).abs() < f64::EPSILON); // 3*30 + 2*5 = 100
    }

    #[test]
    fn test_multi_line_total_height_zero_lines() {
        assert!((multi_line_total_height(0, 30.0, 5.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_position_right_middle() {
        let mut params = PositionParams::new(1920.0, 1080.0);
        params.h_align = HAlign::Right;
        params.v_align = VAlign::Middle;
        let rect = compute_position(&params, 300.0, 50.0);
        let safe = params.safe_area.to_rect(1920.0, 1080.0);
        let expected_x = safe.right() - 300.0;
        assert!((rect.x - expected_x).abs() < 1e-10);
        let expected_y = safe.center_y() - 25.0;
        assert!((rect.y - expected_y).abs() < 1e-10);
    }
}
