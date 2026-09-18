//! Measurement annotations: rulers, angle markers, and safe-area overlays.
//!
//! All coordinates are normalised (0.0–1.0) matching the rest of the drawing
//! subsystem.  Pixel distances are computed by multiplying by the frame's
//! actual pixel dimensions at render time.

#![allow(dead_code)]

use crate::drawing::{Point, Rectangle};
use serde::{Deserialize, Serialize};

// ── Ruler ─────────────────────────────────────────────────────────────────────

/// A ruler that measures the distance between two points in normalised space.
///
/// Optionally carries a pixel-space calibration factor so the label can show
/// real-world units (pixels, millimetres, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ruler {
    /// Start of the measurement.
    pub start: Point,
    /// End of the measurement.
    pub end: Point,
    /// Optional scale: real-world units per normalised unit.
    /// For example `1920.0` converts from normalised to pixels on a 1920-wide frame.
    pub scale: Option<f32>,
    /// Label shown next to the ruler (e.g. "240 px").
    pub label: Option<String>,
    /// Whether to show tick marks at each end.
    pub show_ticks: bool,
}

impl Ruler {
    /// Create a ruler from `start` to `end` with no scaling.
    #[must_use]
    pub fn new(start: Point, end: Point) -> Self {
        Self {
            start,
            end,
            scale: None,
            label: None,
            show_ticks: true,
        }
    }

    /// Attach a scale factor and auto-generate a label.
    ///
    /// `scale` is real-world units per normalised unit (e.g. frame pixel width).
    #[must_use]
    pub fn with_scale(mut self, scale: f32, unit: &str) -> Self {
        self.scale = Some(scale);
        let px = self.length_normalised() * scale;
        self.label = Some(format!("{px:.1} {unit}"));
        self
    }

    /// Set an explicit label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Length in normalised coordinates (0.0–√2 for the diagonal).
    #[must_use]
    pub fn length_normalised(&self) -> f32 {
        self.start.distance_to(&self.end)
    }

    /// Length in scaled units (returns `None` when no scale is set).
    #[must_use]
    pub fn length_scaled(&self) -> Option<f32> {
        self.scale.map(|s| self.length_normalised() * s)
    }

    /// Midpoint of the ruler (where the label is anchored).
    #[must_use]
    pub fn midpoint(&self) -> Point {
        Point::new(
            (self.start.x + self.end.x) * 0.5,
            (self.start.y + self.end.y) * 0.5,
        )
    }

    /// Angle of the ruler in degrees, measured counter-clockwise from the
    /// positive X axis.
    #[must_use]
    pub fn angle_degrees(&self) -> f32 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        dy.atan2(dx).to_degrees()
    }
}

// ── Angle marker ─────────────────────────────────────────────────────────────

/// An angle marker defined by three points: the vertex and two arms.
///
/// The measured angle is at `vertex` between the rays `vertex→arm_a` and
/// `vertex→arm_b`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AngleMarker {
    /// The vertex of the angle.
    pub vertex: Point,
    /// First arm endpoint.
    pub arm_a: Point,
    /// Second arm endpoint.
    pub arm_b: Point,
    /// Radius of the arc drawn between the arms (normalised units).
    pub arc_radius: f32,
    /// Whether the smaller or larger of the two possible angles is shown.
    pub show_reflex: bool,
    /// Optional label override (if `None` the angle in degrees is shown).
    pub label: Option<String>,
}

impl AngleMarker {
    /// Create a new angle marker.  `arc_radius` defaults to `0.05`.
    #[must_use]
    pub fn new(vertex: Point, arm_a: Point, arm_b: Point) -> Self {
        Self {
            vertex,
            arm_a,
            arm_b,
            arc_radius: 0.05,
            show_reflex: false,
            label: None,
        }
    }

    /// Set the arc radius.
    #[must_use]
    pub fn with_arc_radius(mut self, r: f32) -> Self {
        self.arc_radius = r.max(0.0);
        self
    }

    /// Override the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// The angle at `vertex` between the two arms, in degrees (0–360).
    ///
    /// When `show_reflex` is false the return value is clamped to 0–180.
    #[must_use]
    pub fn angle_degrees(&self) -> f32 {
        let ax = self.arm_a.x - self.vertex.x;
        let ay = self.arm_a.y - self.vertex.y;
        let bx = self.arm_b.x - self.vertex.x;
        let by = self.arm_b.y - self.vertex.y;

        let dot = ax * bx + ay * by;
        let cross = ax * by - ay * bx;

        let mut deg = cross.atan2(dot).to_degrees();
        if deg < 0.0 {
            deg += 360.0;
        }

        if !self.show_reflex && deg > 180.0 {
            deg = 360.0 - deg;
        }
        deg
    }

    /// Display label: either the override or the computed angle string.
    #[must_use]
    pub fn display_label(&self) -> String {
        self.label
            .clone()
            .unwrap_or_else(|| format!("{:.1}°", self.angle_degrees()))
    }
}

// ── Safe-area overlay ────────────────────────────────────────────────────────

/// A named broadcast/streaming safe-area rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeAreaPreset {
    /// EBU R 95 – 16:9 action safe (93.3% of frame).
    ActionSafe,
    /// EBU R 95 – 16:9 title safe (80% of frame).
    TitleSafe,
    /// Legacy 4:3 action safe (90% of frame).
    Action43,
    /// Legacy 4:3 title safe (80% of frame).
    Title43,
    /// Vertical video action safe (90% of frame height / width).
    VerticalActionSafe,
    /// Custom zone — use with `SafeAreaOverlay::custom`.
    Custom,
}

impl SafeAreaPreset {
    /// Return the inset fraction from each edge (left=right, top=bottom).
    ///
    /// For example, 4:3 title safe insets by 10 % on each side.
    #[must_use]
    pub fn inset(self) -> (f32, f32) {
        match self {
            Self::ActionSafe => (0.0335, 0.0335), // 3.35 % each side → 93.3 %
            Self::TitleSafe => (0.1, 0.1),        // 10 % each side → 80 %
            Self::Action43 => (0.05, 0.05),       // 5 % each side → 90 %
            Self::Title43 => (0.1, 0.1),          // 10 % each side → 80 %
            Self::VerticalActionSafe => (0.05, 0.05), // 5 % each side
            Self::Custom => (0.0, 0.0),           // caller sets bounds directly
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ActionSafe => "Action Safe (16:9)",
            Self::TitleSafe => "Title Safe (16:9)",
            Self::Action43 => "Action Safe (4:3)",
            Self::Title43 => "Title Safe (4:3)",
            Self::VerticalActionSafe => "Action Safe (Vertical)",
            Self::Custom => "Custom",
        }
    }
}

/// A safe-area overlay drawn on top of the media frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeAreaOverlay {
    /// Which preset this overlay represents.
    pub preset: SafeAreaPreset,
    /// Explicit bounding rectangle (derived from preset insets when created
    /// via [`SafeAreaOverlay::from_preset`]).
    pub bounds: Rectangle,
    /// RGBA colour of the overlay border (default: yellow at 80 % opacity).
    pub color: [u8; 4],
    /// Whether to fill the region outside the safe area with a semi-transparent tint.
    pub show_outside_tint: bool,
    /// Optional label.
    pub label: Option<String>,
}

impl SafeAreaOverlay {
    /// Create an overlay from a preset.  The bounds are inferred from the
    /// preset's inset fractions.
    #[must_use]
    pub fn from_preset(preset: SafeAreaPreset) -> Self {
        let (ix, iy) = preset.inset();
        let bounds = Rectangle::new(Point::new(ix, iy), Point::new(1.0 - ix, 1.0 - iy));
        Self {
            preset,
            bounds,
            color: [255, 220, 0, 204], // yellow, 80 % opacity
            show_outside_tint: false,
            label: None,
        }
    }

    /// Create a custom overlay with an explicit bounding rectangle.
    #[must_use]
    pub fn custom(bounds: Rectangle) -> Self {
        Self {
            preset: SafeAreaPreset::Custom,
            bounds,
            color: [255, 220, 0, 204],
            show_outside_tint: false,
            label: None,
        }
    }

    /// Set the border colour.
    #[must_use]
    pub fn with_color(mut self, color: [u8; 4]) -> Self {
        self.color = color;
        self
    }

    /// Enable or disable the outside-area tint.
    #[must_use]
    pub fn with_outside_tint(mut self, enabled: bool) -> Self {
        self.show_outside_tint = enabled;
        self
    }

    /// Set a label for this overlay.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Width of the safe area in normalised coordinates.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.bounds.width()
    }

    /// Height of the safe area in normalised coordinates.
    #[must_use]
    pub fn height(&self) -> f32 {
        self.bounds.height()
    }
}

// ── Measurement annotation ─────────────────────────────────────────────────

/// A measurement annotation that can be one of the supported types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeasurementAnnotation {
    /// A ruler between two points.
    Ruler(Ruler),
    /// An angle marker at a vertex.
    Angle(AngleMarker),
    /// A safe-area overlay.
    SafeArea(SafeAreaOverlay),
}

impl MeasurementAnnotation {
    /// Create a ruler annotation.
    #[must_use]
    pub fn ruler(start: Point, end: Point) -> Self {
        Self::Ruler(Ruler::new(start, end))
    }

    /// Create an angle annotation.
    #[must_use]
    pub fn angle(vertex: Point, arm_a: Point, arm_b: Point) -> Self {
        Self::Angle(AngleMarker::new(vertex, arm_a, arm_b))
    }

    /// Create a safe-area overlay annotation.
    #[must_use]
    pub fn safe_area(preset: SafeAreaPreset) -> Self {
        Self::SafeArea(SafeAreaOverlay::from_preset(preset))
    }

    /// Return a short descriptive label for this annotation.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Ruler(r) => r
                .label
                .clone()
                .unwrap_or_else(|| format!("{:.4}", r.length_normalised())),
            Self::Angle(a) => a.display_label(),
            Self::SafeArea(s) => s
                .label
                .clone()
                .unwrap_or_else(|| s.preset.name().to_string()),
        }
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1 — Ruler length in normalised coords
    #[test]
    fn test_ruler_length_normalised() {
        let r = Ruler::new(Point::new(0.0, 0.0), Point::new(0.3, 0.4));
        assert!((r.length_normalised() - 0.5).abs() < 1e-5);
    }

    // 2 — Ruler midpoint
    #[test]
    fn test_ruler_midpoint() {
        let r = Ruler::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let mid = r.midpoint();
        assert!((mid.x - 0.5).abs() < 1e-5);
        assert!((mid.y - 0.5).abs() < 1e-5);
    }

    // 3 — Ruler with scale sets label
    #[test]
    fn test_ruler_with_scale_label() {
        let r = Ruler::new(Point::new(0.0, 0.0), Point::new(0.5, 0.0)).with_scale(1920.0, "px");
        assert!(r.label.as_deref().unwrap_or("").contains("px"));
        let scaled = r.length_scaled().expect("scale set");
        assert!((scaled - 960.0).abs() < 1.0);
    }

    // 4 — Ruler horizontal angle is 0°
    #[test]
    fn test_ruler_angle_horizontal() {
        let r = Ruler::new(Point::new(0.0, 0.5), Point::new(1.0, 0.5));
        assert!(r.angle_degrees().abs() < 1e-4);
    }

    // 5 — Ruler no scale gives None
    #[test]
    fn test_ruler_no_scale_returns_none() {
        let r = Ruler::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        assert!(r.length_scaled().is_none());
    }

    // 6 — AngleMarker: right angle
    #[test]
    fn test_angle_marker_right_angle() {
        // arm_a points right, arm_b points up → 90°
        let a = AngleMarker::new(
            Point::new(0.5, 0.5),
            Point::new(1.0, 0.5), // arm_a → right
            Point::new(0.5, 0.0), // arm_b → up
        );
        assert!((a.angle_degrees() - 90.0).abs() < 0.1);
    }

    // 7 — AngleMarker display label uses override when set
    #[test]
    fn test_angle_marker_display_label_override() {
        let a = AngleMarker::new(
            Point::new(0.5, 0.5),
            Point::new(1.0, 0.5),
            Point::new(0.5, 0.0),
        )
        .with_label("90°");
        assert_eq!(a.display_label(), "90°");
    }

    // 8 — AngleMarker default label includes degree symbol
    #[test]
    fn test_angle_marker_default_label_has_degree() {
        let a = AngleMarker::new(
            Point::new(0.5, 0.5),
            Point::new(1.0, 0.5),
            Point::new(0.5, 0.0),
        );
        assert!(a.display_label().contains('°'));
    }

    // 9 — SafeAreaPreset inset
    #[test]
    fn test_safe_area_preset_inset() {
        let (ix, iy) = SafeAreaPreset::TitleSafe.inset();
        assert!((ix - 0.1).abs() < 1e-5);
        assert!((iy - 0.1).abs() < 1e-5);
    }

    // 10 — SafeAreaOverlay::from_preset bounds
    #[test]
    fn test_safe_area_overlay_title_safe_bounds() {
        let ov = SafeAreaOverlay::from_preset(SafeAreaPreset::TitleSafe);
        assert!((ov.bounds.top_left.x - 0.1).abs() < 1e-4);
        assert!((ov.bounds.bottom_right.x - 0.9).abs() < 1e-4);
        assert!((ov.width() - 0.8).abs() < 1e-4);
        assert!((ov.height() - 0.8).abs() < 1e-4);
    }

    // 11 — SafeAreaOverlay::custom accepts arbitrary bounds
    #[test]
    fn test_safe_area_overlay_custom() {
        let bounds = Rectangle::new(Point::new(0.05, 0.05), Point::new(0.95, 0.95));
        let ov = SafeAreaOverlay::custom(bounds);
        assert_eq!(ov.preset, SafeAreaPreset::Custom);
    }

    // 12 — SafeAreaPreset names are non-empty
    #[test]
    fn test_safe_area_preset_names() {
        for preset in [
            SafeAreaPreset::ActionSafe,
            SafeAreaPreset::TitleSafe,
            SafeAreaPreset::Action43,
            SafeAreaPreset::Title43,
            SafeAreaPreset::VerticalActionSafe,
            SafeAreaPreset::Custom,
        ] {
            assert!(!preset.name().is_empty());
        }
    }

    // 13 — MeasurementAnnotation::ruler label
    #[test]
    fn test_measurement_annotation_ruler_label() {
        let ann = MeasurementAnnotation::ruler(Point::new(0.0, 0.0), Point::new(1.0, 0.0));
        assert!(!ann.label().is_empty());
    }

    // 14 — MeasurementAnnotation::angle label
    #[test]
    fn test_measurement_annotation_angle_label() {
        let ann = MeasurementAnnotation::angle(
            Point::new(0.5, 0.5),
            Point::new(1.0, 0.5),
            Point::new(0.5, 0.0),
        );
        assert!(ann.label().contains('°'));
    }

    // 15 — MeasurementAnnotation::safe_area label uses preset name
    #[test]
    fn test_measurement_annotation_safe_area_label() {
        let ann = MeasurementAnnotation::safe_area(SafeAreaPreset::ActionSafe);
        assert!(ann.label().contains("Safe"));
    }
}
