#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
//! Constraint-based layout engine for broadcast graphics.
//!
//! Provides a simple box-model layout system that resolves fixed and
//! flexible size constraints, enabling responsive broadcast graphics
//! without hard-coded pixel coordinates.

/// A sizing constraint for one dimension of a layout box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LayoutConstraint {
    /// Exact pixel size.
    Fixed(f32),
    /// Proportional share of the available space (weight).
    Flex(f32),
    /// Minimum and maximum pixel bounds.
    Bounded { min: f32, max: f32 },
    /// Expands to fill all remaining space after fixed boxes are placed.
    #[default]
    Fill,
}

impl LayoutConstraint {
    /// Returns `true` if this constraint can stretch to fill available space.
    pub fn is_flexible(&self) -> bool {
        matches!(self, Self::Flex(_) | Self::Fill)
    }

    /// Returns the fixed size if this is a `Fixed` constraint, otherwise `None`.
    pub fn fixed_size(&self) -> Option<f32> {
        match self {
            Self::Fixed(s) => Some(*s),
            _ => None,
        }
    }

    /// Clamps `value` to be within this constraint's bounds where applicable.
    pub fn clamp(&self, value: f32) -> f32 {
        match self {
            Self::Fixed(s) => *s,
            Self::Bounded { min, max } => value.clamp(*min, *max),
            Self::Flex(_) | Self::Fill => value.max(0.0),
        }
    }
}

// ── LayoutBox ─────────────────────────────────────────────────────────────

/// A named rectangular region with width/height constraints and a margin.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    /// Unique identifier for this box.
    pub id: String,
    /// Constraint governing the box width.
    pub width_constraint: LayoutConstraint,
    /// Constraint governing the box height.
    pub height_constraint: LayoutConstraint,
    /// Uniform margin applied to all four sides (pixels).
    pub margin: f32,
}

impl LayoutBox {
    /// Creates a new layout box with the given constraints.
    pub fn new(
        id: impl Into<String>,
        width_constraint: LayoutConstraint,
        height_constraint: LayoutConstraint,
    ) -> Self {
        Self {
            id: id.into(),
            width_constraint,
            height_constraint,
            margin: 0.0,
        }
    }

    /// Sets the margin and returns `self` for chaining.
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// Computes the resolved `(width, height)` given the available space.
    ///
    /// Flexible constraints are resolved to the full available dimension.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_size(&self, available_width: f32, available_height: f32) -> (f32, f32) {
        let w = match self.width_constraint {
            LayoutConstraint::Fixed(s) => s,
            LayoutConstraint::Flex(weight) => available_width * weight,
            LayoutConstraint::Fill => available_width,
            LayoutConstraint::Bounded { min, max } => available_width.clamp(min, max),
        };
        let h = match self.height_constraint {
            LayoutConstraint::Fixed(s) => s,
            LayoutConstraint::Flex(weight) => available_height * weight,
            LayoutConstraint::Fill => available_height,
            LayoutConstraint::Bounded { min, max } => available_height.clamp(min, max),
        };
        (w - self.margin * 2.0, h - self.margin * 2.0)
    }
}

// ── LayoutResult ──────────────────────────────────────────────────────────

/// The resolved geometry for a single box after layout.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Identifier matching the source `LayoutBox`.
    pub id: String,
    /// Left edge of the resolved box in container coordinates.
    pub x: f32,
    /// Top edge of the resolved box in container coordinates.
    pub y: f32,
    /// Resolved width of the box after margin subtraction.
    pub width: f32,
    /// Resolved height of the box after margin subtraction.
    pub height: f32,
}

impl LayoutResult {
    /// Returns the area of this layout result.
    pub fn total_area(&self) -> f32 {
        self.width * self.height
    }

    /// Returns `true` if the point `(px, py)` falls within this box.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }
}

// ── LayoutEngine ──────────────────────────────────────────────────────────

/// A simple horizontal-flow layout engine.
///
/// Boxes are placed left-to-right within the container. Flexible boxes
/// share the remaining width equally after fixed boxes are placed.
#[derive(Debug)]
pub struct LayoutEngine {
    boxes: Vec<LayoutBox>,
    container_width: f32,
    container_height: f32,
}

impl LayoutEngine {
    /// Creates a layout engine with the given container dimensions.
    pub fn new(container_width: f32, container_height: f32) -> Self {
        Self {
            boxes: Vec::new(),
            container_width,
            container_height,
        }
    }

    /// Appends a box to the layout.
    pub fn add_box(&mut self, b: LayoutBox) {
        self.boxes.push(b);
    }

    /// Returns the number of boxes registered.
    pub fn box_count(&self) -> usize {
        self.boxes.len()
    }

    /// Resolves all box geometries using a horizontal flow algorithm.
    ///
    /// Fixed-width boxes are placed first; remaining width is divided
    /// among flexible boxes proportionally to their flex weight.
    #[allow(clippy::cast_precision_loss)]
    pub fn layout_all(&self) -> Vec<LayoutResult> {
        // Phase 1: total fixed width used
        let fixed_width: f32 = self
            .boxes
            .iter()
            .filter_map(|b| b.width_constraint.fixed_size())
            .sum();

        let remaining_width = (self.container_width - fixed_width).max(0.0);

        // Phase 2: total flex weight
        let total_flex_weight: f32 = self
            .boxes
            .iter()
            .map(|b| match b.width_constraint {
                LayoutConstraint::Flex(w) => w,
                LayoutConstraint::Fill => 1.0,
                _ => 0.0,
            })
            .sum();

        let flex_unit = if total_flex_weight > 0.0 {
            remaining_width / total_flex_weight
        } else {
            0.0
        };

        // Phase 3: assign positions
        let mut x = 0.0_f32;
        let mut results = Vec::with_capacity(self.boxes.len());

        for b in &self.boxes {
            let w = match b.width_constraint {
                LayoutConstraint::Fixed(s) => s,
                LayoutConstraint::Flex(weight) => flex_unit * weight,
                LayoutConstraint::Fill => flex_unit,
                LayoutConstraint::Bounded { min, max } => remaining_width.clamp(min, max),
            };
            let h = match b.height_constraint {
                LayoutConstraint::Fixed(s) => s,
                LayoutConstraint::Flex(weight) => self.container_height * weight,
                LayoutConstraint::Fill => self.container_height,
                LayoutConstraint::Bounded { min, max } => self.container_height.clamp(min, max),
            };

            let inner_w = (w - b.margin * 2.0).max(0.0);
            let inner_h = (h - b.margin * 2.0).max(0.0);

            results.push(LayoutResult {
                id: b.id.clone(),
                x: x + b.margin,
                y: b.margin,
                width: inner_w,
                height: inner_h,
            });

            x += w;
        }

        results
    }
}

// ── SafeAreaMargins ────────────────────────────────────────────────────────

/// Configurable broadcast safe-area margins.
///
/// Broadcast standards (EBU, SMPTE) define minimum margins within which all
/// essential content must appear so that it is not cropped by consumer displays.
/// `SafeAreaMargins` expresses these margins as fractions of the full frame
/// width / height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeAreaMargins {
    /// Left margin as a fraction of frame width (0.0–0.5).
    pub left: f32,
    /// Right margin as a fraction of frame width (0.0–0.5).
    pub right: f32,
    /// Top margin as a fraction of frame height (0.0–0.5).
    pub top: f32,
    /// Bottom margin as a fraction of frame height (0.0–0.5).
    pub bottom: f32,
}

impl SafeAreaMargins {
    /// Create custom margins (all values clamped to `[0.0, 0.5]`).
    pub fn new(left: f32, right: f32, top: f32, bottom: f32) -> Self {
        Self {
            left: left.clamp(0.0, 0.5),
            right: right.clamp(0.0, 0.5),
            top: top.clamp(0.0, 0.5),
            bottom: bottom.clamp(0.0, 0.5),
        }
    }

    /// Uniform margin on all four sides.
    pub fn uniform(fraction: f32) -> Self {
        let f = fraction.clamp(0.0, 0.5);
        Self {
            left: f,
            right: f,
            top: f,
            bottom: f,
        }
    }

    /// EBU R 95 / SMPTE RP 218 **action-safe** area: 3.5% margin on each side.
    pub fn action_safe() -> Self {
        Self::uniform(0.035)
    }

    /// EBU R 95 / SMPTE RP 218 **title-safe** area: 5% margin on each side.
    pub fn title_safe() -> Self {
        Self::uniform(0.05)
    }

    /// No margins (full-frame).
    pub fn none() -> Self {
        Self::uniform(0.0)
    }

    /// Compute the inset rectangle for a frame of the given pixel dimensions.
    ///
    /// Returns `(x, y, width, height)` in pixels.
    pub fn inset_rect(&self, frame_width: f32, frame_height: f32) -> (f32, f32, f32, f32) {
        let x = self.left * frame_width;
        let y = self.top * frame_height;
        let right = frame_width - self.right * frame_width;
        let bottom = frame_height - self.bottom * frame_height;
        let w = (right - x).max(0.0);
        let h = (bottom - y).max(0.0);
        (x, y, w, h)
    }

    /// Returns `true` if the point `(px, py)` (in pixels) falls within the
    /// safe area for a frame of size `(frame_width, frame_height)`.
    pub fn contains_point(&self, px: f32, py: f32, frame_width: f32, frame_height: f32) -> bool {
        let (sx, sy, sw, sh) = self.inset_rect(frame_width, frame_height);
        px >= sx && px <= sx + sw && py >= sy && py <= sy + sh
    }
}

impl Default for SafeAreaMargins {
    fn default() -> Self {
        Self::title_safe()
    }
}

// ── LayoutEngine with SafeArea ─────────────────────────────────────────────

impl LayoutEngine {
    /// Set safe-area margins and re-constrain the container dimensions so that
    /// all layout boxes reside within the safe area.
    ///
    /// The engine's `container_width` and `container_height` are replaced with
    /// the safe-area inset dimensions, and the returned `(offset_x, offset_y)`
    /// values indicate where the safe-area origin falls within the original frame.
    pub fn apply_safe_area(
        &mut self,
        safe_area: SafeAreaMargins,
        frame_width: f32,
        frame_height: f32,
    ) -> (f32, f32) {
        let (x, y, w, h) = safe_area.inset_rect(frame_width, frame_height);
        self.container_width = w;
        self.container_height = h;
        (x, y)
    }
}

// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_is_flexible() {
        assert!(LayoutConstraint::Flex(1.0).is_flexible());
        assert!(LayoutConstraint::Fill.is_flexible());
        assert!(!LayoutConstraint::Fixed(100.0).is_flexible());
        assert!(!LayoutConstraint::Bounded {
            min: 10.0,
            max: 200.0
        }
        .is_flexible());
    }

    #[test]
    fn test_constraint_fixed_size() {
        assert_eq!(LayoutConstraint::Fixed(50.0).fixed_size(), Some(50.0));
        assert_eq!(LayoutConstraint::Fill.fixed_size(), None);
    }

    #[test]
    fn test_constraint_clamp_fixed() {
        let c = LayoutConstraint::Fixed(120.0);
        assert_eq!(c.clamp(999.0), 120.0);
    }

    #[test]
    fn test_constraint_clamp_bounded() {
        let c = LayoutConstraint::Bounded {
            min: 50.0,
            max: 200.0,
        };
        assert_eq!(c.clamp(300.0), 200.0);
        assert_eq!(c.clamp(10.0), 50.0);
        assert_eq!(c.clamp(100.0), 100.0);
    }

    #[test]
    fn test_default_constraint_is_fill() {
        assert_eq!(LayoutConstraint::default(), LayoutConstraint::Fill);
    }

    #[test]
    fn test_layout_box_compute_size_fixed() {
        let b = LayoutBox::new(
            "box",
            LayoutConstraint::Fixed(100.0),
            LayoutConstraint::Fixed(50.0),
        );
        assert_eq!(b.compute_size(800.0, 600.0), (100.0, 50.0));
    }

    #[test]
    fn test_layout_box_compute_size_fill() {
        let b = LayoutBox::new("box", LayoutConstraint::Fill, LayoutConstraint::Fill);
        assert_eq!(b.compute_size(800.0, 600.0), (800.0, 600.0));
    }

    #[test]
    fn test_layout_box_margin_reduces_size() {
        let b = LayoutBox::new(
            "box",
            LayoutConstraint::Fixed(100.0),
            LayoutConstraint::Fixed(50.0),
        )
        .with_margin(5.0);
        let (w, h) = b.compute_size(800.0, 600.0);
        assert_eq!(w, 90.0); // 100 - 2*5
        assert_eq!(h, 40.0); // 50  - 2*5
    }

    #[test]
    fn test_layout_result_total_area() {
        let r = LayoutResult {
            id: "x".into(),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        };
        assert_eq!(r.total_area(), 5000.0);
    }

    #[test]
    fn test_layout_result_contains() {
        let r = LayoutResult {
            id: "x".into(),
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 50.0,
        };
        assert!(r.contains(50.0, 30.0));
        assert!(!r.contains(5.0, 30.0));
        assert!(!r.contains(50.0, 5.0));
    }

    #[test]
    fn test_engine_single_fixed_box() {
        let mut engine = LayoutEngine::new(1920.0, 1080.0);
        engine.add_box(LayoutBox::new(
            "logo",
            LayoutConstraint::Fixed(200.0),
            LayoutConstraint::Fixed(100.0),
        ));
        let results = engine.layout_all();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].width, 200.0);
        assert_eq!(results[0].x, 0.0);
    }

    #[test]
    fn test_engine_two_flex_boxes_split_equally() {
        let mut engine = LayoutEngine::new(1000.0, 100.0);
        engine.add_box(LayoutBox::new(
            "a",
            LayoutConstraint::Flex(1.0),
            LayoutConstraint::Fill,
        ));
        engine.add_box(LayoutBox::new(
            "b",
            LayoutConstraint::Flex(1.0),
            LayoutConstraint::Fill,
        ));
        let results = engine.layout_all();
        assert_eq!(results.len(), 2);
        assert!((results[0].width - 500.0).abs() < 0.01);
        assert!((results[1].width - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_engine_box_count() {
        let mut engine = LayoutEngine::new(500.0, 500.0);
        engine.add_box(LayoutBox::new(
            "a",
            LayoutConstraint::Fill,
            LayoutConstraint::Fill,
        ));
        engine.add_box(LayoutBox::new(
            "b",
            LayoutConstraint::Fill,
            LayoutConstraint::Fill,
        ));
        assert_eq!(engine.box_count(), 2);
    }

    #[test]
    fn test_engine_fixed_plus_flex() {
        let mut engine = LayoutEngine::new(1000.0, 100.0);
        engine.add_box(LayoutBox::new(
            "sidebar",
            LayoutConstraint::Fixed(200.0),
            LayoutConstraint::Fill,
        ));
        engine.add_box(LayoutBox::new(
            "main",
            LayoutConstraint::Flex(1.0),
            LayoutConstraint::Fill,
        ));
        let results = engine.layout_all();
        assert_eq!(results[0].width, 200.0);
        assert!((results[1].width - 800.0).abs() < 0.01);
    }

    #[test]
    fn test_engine_empty_produces_empty_results() {
        let engine = LayoutEngine::new(1920.0, 1080.0);
        assert!(engine.layout_all().is_empty());
    }

    // --- SafeAreaMargins tests ---

    #[test]
    fn test_safe_area_uniform() {
        let m = SafeAreaMargins::uniform(0.1);
        assert!((m.left - 0.1).abs() < f32::EPSILON);
        assert!((m.right - 0.1).abs() < f32::EPSILON);
        assert!((m.top - 0.1).abs() < f32::EPSILON);
        assert!((m.bottom - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_safe_area_action_safe() {
        let m = SafeAreaMargins::action_safe();
        assert!((m.left - 0.035).abs() < 0.001);
    }

    #[test]
    fn test_safe_area_title_safe() {
        let m = SafeAreaMargins::title_safe();
        assert!((m.left - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_safe_area_none() {
        let m = SafeAreaMargins::none();
        assert_eq!(m.inset_rect(1920.0, 1080.0), (0.0, 0.0, 1920.0, 1080.0));
    }

    #[test]
    fn test_safe_area_inset_rect_1080p_title() {
        let m = SafeAreaMargins::title_safe();
        let (x, y, w, h) = m.inset_rect(1920.0, 1080.0);
        // x = 0.05*1920 = 96; y = 0.05*1080 = 54
        // w = 1920 - 2*96 = 1728; h = 1080 - 2*54 = 972
        assert!((x - 96.0).abs() < 0.1);
        assert!((y - 54.0).abs() < 0.1);
        assert!((w - 1728.0).abs() < 0.1);
        assert!((h - 972.0).abs() < 0.1);
    }

    #[test]
    fn test_safe_area_contains_point_center() {
        let m = SafeAreaMargins::title_safe();
        assert!(m.contains_point(960.0, 540.0, 1920.0, 1080.0));
    }

    #[test]
    fn test_safe_area_contains_point_corner() {
        let m = SafeAreaMargins::title_safe();
        // Top-left corner is outside title safe.
        assert!(!m.contains_point(0.0, 0.0, 1920.0, 1080.0));
    }

    #[test]
    fn test_safe_area_clamp_over_half() {
        let m = SafeAreaMargins::uniform(0.9); // clamped to 0.5
        assert!((m.left - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layout_engine_apply_safe_area() {
        let mut engine = LayoutEngine::new(1920.0, 1080.0);
        let (ox, oy) = engine.apply_safe_area(SafeAreaMargins::title_safe(), 1920.0, 1080.0);
        // Container should now be the safe-area inner dimensions.
        assert!((engine.container_width - 1728.0).abs() < 0.1);
        assert!((engine.container_height - 972.0).abs() < 0.1);
        // Offset should be the top-left corner of the safe area.
        assert!((ox - 96.0).abs() < 0.1);
        assert!((oy - 54.0).abs() < 0.1);
    }

    #[test]
    fn test_safe_area_custom_asymmetric() {
        let m = SafeAreaMargins::new(0.05, 0.1, 0.03, 0.07);
        let (x, y, w, h) = m.inset_rect(1000.0, 1000.0);
        assert!((x - 50.0).abs() < 0.1); // left 5%
        assert!((y - 30.0).abs() < 0.1); // top 3%
                                         // right edge at 1000 - 100 = 900; w = 900 - 50 = 850
        assert!((w - 850.0).abs() < 0.1);
        // bottom edge at 1000 - 70 = 930; h = 930 - 30 = 900
        assert!((h - 900.0).abs() < 0.1);
    }
}
