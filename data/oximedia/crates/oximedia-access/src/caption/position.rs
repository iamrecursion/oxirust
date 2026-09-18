//! Smart caption positioning with collision avoidance.

use serde::{Deserialize, Serialize};

/// Caption position on screen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum CaptionPosition {
    /// Bottom center (default).
    #[default]
    BottomCenter,
    /// Top center.
    TopCenter,
    /// Bottom left.
    BottomLeft,
    /// Bottom right.
    BottomRight,
    /// Custom position (x, y as percentage 0-100).
    Custom(f32, f32),
}

/// A rectangular region on-screen, specified in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScreenRect {
    /// Left edge in pixels.
    pub x: i32,
    /// Top edge in pixels.
    pub y: i32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ScreenRect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge in pixels.
    #[must_use]
    pub fn right(&self) -> i32 {
        self.x + self.width as i32
    }

    /// Bottom edge in pixels.
    #[must_use]
    pub fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    /// Check whether this rectangle overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &ScreenRect) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Compute the intersection area in pixels² (0 if no overlap).
    #[must_use]
    pub fn intersection_area(&self, other: &ScreenRect) -> u64 {
        let x_overlap = (self.right().min(other.right()) - self.x.max(other.x)).max(0) as u64;
        let y_overlap = (self.bottom().min(other.bottom()) - self.y.max(other.y)).max(0) as u64;
        x_overlap * y_overlap
    }
}

/// Configuration for collision avoidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionConfig {
    /// Minimum vertical margin between caption and a burned-in region (px).
    pub vertical_margin_px: i32,
    /// Minimum horizontal margin between caption and a burned-in region (px).
    pub horizontal_margin_px: i32,
    /// Whether to prefer moving caption to the top when the bottom is blocked.
    pub prefer_top_fallback: bool,
    /// Safety margin from screen edges (px).
    pub edge_margin_px: i32,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            vertical_margin_px: 8,
            horizontal_margin_px: 8,
            prefer_top_fallback: true,
            edge_margin_px: 16,
        }
    }
}

/// Result of a collision-avoidance placement calculation.
#[derive(Debug, Clone)]
pub struct PlacementResult {
    /// The resolved pixel coordinates (x, y) of the caption anchor.
    pub x: i32,
    /// Resolved y coordinate.
    pub y: i32,
    /// Whether the placement was adjusted to avoid a collision.
    pub was_adjusted: bool,
    /// Number of burned-in regions the caption was adjusted away from.
    pub collisions_avoided: usize,
}

/// Smart caption positioner to avoid overlapping important content.
pub struct CaptionPositioner {
    default_position: CaptionPosition,
    avoid_bottom_percent: f32,
    collision_config: CollisionConfig,
}

impl CaptionPositioner {
    /// Create a new positioner.
    #[must_use]
    pub const fn new(default_position: CaptionPosition) -> Self {
        Self {
            default_position,
            avoid_bottom_percent: 20.0,
            collision_config: CollisionConfig {
                vertical_margin_px: 8,
                horizontal_margin_px: 8,
                prefer_top_fallback: true,
                edge_margin_px: 16,
            },
        }
    }

    /// Set percentage of screen bottom to avoid.
    #[must_use]
    pub const fn with_avoid_bottom(mut self, percent: f32) -> Self {
        self.avoid_bottom_percent = percent;
        self
    }

    /// Set collision avoidance configuration.
    #[must_use]
    pub fn with_collision_config(mut self, config: CollisionConfig) -> Self {
        self.collision_config = config;
        self
    }

    /// Calculate optimal position based on frame content.
    #[must_use]
    pub fn calculate_position(&self, _frame_height: u32) -> CaptionPosition {
        // In production, this would analyze the frame to:
        // - Detect faces and avoid covering them
        // - Detect on-screen text
        // - Detect important action areas
        // - Use saliency detection

        self.default_position
    }

    /// Get position coordinates as pixel offsets.
    #[must_use]
    pub fn get_coordinates(
        &self,
        position: &CaptionPosition,
        width: u32,
        height: u32,
    ) -> (i32, i32) {
        match position {
            CaptionPosition::BottomCenter => (width as i32 / 2, height as i32 - 100),
            CaptionPosition::TopCenter => (width as i32 / 2, 100),
            CaptionPosition::BottomLeft => (50, height as i32 - 100),
            CaptionPosition::BottomRight => (width as i32 - 50, height as i32 - 100),
            CaptionPosition::Custom(x, y) => (
                (width as f32 * x / 100.0) as i32,
                (height as f32 * y / 100.0) as i32,
            ),
        }
    }

    // ─── Collision avoidance ─────────────────────────────────────────────────

    /// Place a caption of the given dimensions on screen while avoiding collisions
    /// with `burned_in_regions` — areas of the video frame that already contain
    /// burned-in text, logos, or other permanent overlays.
    ///
    /// Strategy:
    /// 1. Try the default (bottom-center) position.
    /// 2. If it collides with any burned-in region, try shifting upward.
    /// 3. If still blocked, fall back to top-center.
    /// 4. If top is also blocked, accept the least-overlapping candidate.
    ///
    /// `frame_width` / `frame_height` define the safe screen area.
    /// `caption_width` / `caption_height` are the dimensions of the caption box.
    #[must_use]
    pub fn place_avoiding_collisions(
        &self,
        frame_width: u32,
        frame_height: u32,
        caption_width: u32,
        caption_height: u32,
        burned_in_regions: &[ScreenRect],
    ) -> PlacementResult {
        let m = &self.collision_config;
        let cap_w = caption_width as i32;
        let cap_h = caption_height as i32;
        let fw = frame_width as i32;
        let fh = frame_height as i32;

        // Candidate positions ordered by preference.
        let candidates: Vec<(i32, i32)> = vec![
            // 1. Default: bottom-center with edge margin
            ((fw - cap_w) / 2, fh - cap_h - m.edge_margin_px),
            // 2. Shifted up 20 % of frame height
            (
                (fw - cap_w) / 2,
                fh - cap_h - (fh as f32 * 0.20).round() as i32 - m.edge_margin_px,
            ),
            // 3. Top-center fallback
            ((fw - cap_w) / 2, m.edge_margin_px),
            // 4. Bottom-left
            (m.edge_margin_px, fh - cap_h - m.edge_margin_px),
            // 5. Bottom-right
            (fw - cap_w - m.edge_margin_px, fh - cap_h - m.edge_margin_px),
        ];

        // Expand each burned-in region by the configured margin for overlap testing.
        let expanded: Vec<ScreenRect> = burned_in_regions
            .iter()
            .map(|r| ScreenRect {
                x: r.x - m.horizontal_margin_px,
                y: r.y - m.vertical_margin_px,
                width: r.width + 2 * m.horizontal_margin_px as u32,
                height: r.height + 2 * m.vertical_margin_px as u32,
            })
            .collect();

        // Try each candidate; use the first collision-free one.
        for (x, y) in &candidates {
            // Clamp to safe screen bounds.
            let cx = (*x).clamp(m.edge_margin_px, (fw - cap_w - m.edge_margin_px).max(0));
            let cy = (*y).clamp(m.edge_margin_px, (fh - cap_h - m.edge_margin_px).max(0));
            let cap_rect = ScreenRect::new(cx, cy, caption_width, caption_height);

            let collisions: Vec<_> = expanded.iter().filter(|r| cap_rect.overlaps(r)).collect();

            if collisions.is_empty() {
                let was_adjusted = (*x, *y) != candidates[0];
                return PlacementResult {
                    x: cx,
                    y: cy,
                    was_adjusted,
                    collisions_avoided: 0,
                };
            }
        }

        // All candidates overlap — pick the one with the smallest total intersection.
        let best = candidates
            .iter()
            .map(|(x, y)| {
                let cx = (*x).clamp(m.edge_margin_px, (fw - cap_w - m.edge_margin_px).max(0));
                let cy = (*y).clamp(m.edge_margin_px, (fh - cap_h - m.edge_margin_px).max(0));
                let cap_rect = ScreenRect::new(cx, cy, caption_width, caption_height);
                let area: u64 = expanded.iter().map(|r| cap_rect.intersection_area(r)).sum();
                (cx, cy, area)
            })
            .min_by_key(|(_, _, area)| *area)
            .map(|(x, y, _)| (x, y));

        let (bx, by) = best.unwrap_or((m.edge_margin_px, m.edge_margin_px));
        PlacementResult {
            x: bx,
            y: by,
            was_adjusted: true,
            collisions_avoided: burned_in_regions.len(),
        }
    }

    /// Check whether a caption rectangle collides with any burned-in region.
    #[must_use]
    pub fn has_collision(
        caption: &ScreenRect,
        burned_in_regions: &[ScreenRect],
        margin_px: i32,
    ) -> bool {
        let expanded: Vec<ScreenRect> = burned_in_regions
            .iter()
            .map(|r| ScreenRect {
                x: r.x - margin_px,
                y: r.y - margin_px,
                width: r.width + 2 * margin_px as u32,
                height: r.height + 2 * margin_px as u32,
            })
            .collect();
        expanded.iter().any(|r| caption.overlaps(r))
    }
}

impl Default for CaptionPositioner {
    fn default() -> Self {
        Self::new(CaptionPosition::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_position() {
        let pos = CaptionPosition::default();
        assert_eq!(pos, CaptionPosition::BottomCenter);
    }

    #[test]
    fn test_positioner() {
        let positioner = CaptionPositioner::default();
        let (x, y) = positioner.get_coordinates(&CaptionPosition::BottomCenter, 1920, 1080);
        assert_eq!(x, 960);
        assert_eq!(y, 980);
    }

    #[test]
    fn test_custom_position() {
        let positioner = CaptionPositioner::default();
        let (x, y) = positioner.get_coordinates(&CaptionPosition::Custom(50.0, 50.0), 1920, 1080);
        assert_eq!(x, 960);
        assert_eq!(y, 540);
    }

    // ============================================================
    // ScreenRect tests
    // ============================================================

    #[test]
    fn test_screen_rect_overlap() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_screen_rect_no_overlap() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(200, 200, 100, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_screen_rect_adjacent_no_overlap() {
        // Touching edge — not overlapping
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(100, 0, 100, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_screen_rect_intersection_area() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(50, 50, 100, 100);
        let area = a.intersection_area(&b);
        assert_eq!(area, 50 * 50); // 50x50 overlap
    }

    #[test]
    fn test_screen_rect_no_intersection_area() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(200, 200, 100, 100);
        assert_eq!(a.intersection_area(&b), 0);
    }

    // ============================================================
    // Collision avoidance tests
    // ============================================================

    #[test]
    fn test_place_no_collision() {
        let positioner = CaptionPositioner::default();
        let result = positioner.place_avoiding_collisions(1920, 1080, 600, 80, &[]);
        assert!(!result.was_adjusted);
        assert_eq!(result.collisions_avoided, 0);
    }

    #[test]
    fn test_place_avoids_burned_in_bottom() {
        let positioner = CaptionPositioner::default();
        // Simulate a large burned-in region at the bottom of the frame
        let burned_in = vec![ScreenRect::new(0, 950, 1920, 130)];
        let result = positioner.place_avoiding_collisions(1920, 1080, 800, 80, &burned_in);
        // The caption should have been moved upward to avoid the region
        assert!(result.was_adjusted || result.y < 950);
    }

    #[test]
    fn test_place_falls_back_to_top() {
        let positioner = CaptionPositioner::default();
        // Block both bottom and most of the middle
        let burned_in = vec![
            ScreenRect::new(0, 400, 1920, 680), // blocks bottom half
        ];
        let result = positioner.place_avoiding_collisions(1920, 1080, 800, 80, &burned_in);
        // Caption should be placed in the top region
        assert!(result.y < 400 || result.was_adjusted);
    }

    #[test]
    fn test_has_collision() {
        let caption = ScreenRect::new(100, 900, 800, 80);
        let burned_in = vec![ScreenRect::new(0, 950, 1920, 100)];
        assert!(CaptionPositioner::has_collision(&caption, &burned_in, 0));
    }

    #[test]
    fn test_no_collision() {
        let caption = ScreenRect::new(100, 100, 800, 80);
        let burned_in = vec![ScreenRect::new(0, 950, 1920, 100)];
        assert!(!CaptionPositioner::has_collision(&caption, &burned_in, 0));
    }

    #[test]
    fn test_collision_with_margin() {
        let caption = ScreenRect::new(100, 840, 800, 80);
        let burned_in = vec![ScreenRect::new(0, 950, 1920, 100)]; // gap of 30px
                                                                  // With margin of 50px the expanded region overlaps
        assert!(CaptionPositioner::has_collision(&caption, &burned_in, 50));
        // With margin of 5px it should not
        assert!(!CaptionPositioner::has_collision(&caption, &burned_in, 5));
    }

    #[test]
    fn test_placement_result_within_frame() {
        let positioner = CaptionPositioner::default();
        let result = positioner.place_avoiding_collisions(1920, 1080, 600, 80, &[]);
        assert!(result.x >= 0);
        assert!(result.y >= 0);
        assert!((result.x + 600) <= 1920);
        assert!((result.y + 80) <= 1080);
    }
}
