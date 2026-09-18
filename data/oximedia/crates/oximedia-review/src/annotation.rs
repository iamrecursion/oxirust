//! Media annotation tools for review workflows.
//!
//! Provides drawing tools, region annotations, timestamp notes, and layered
//! annotation management for frame-accurate media review.

#![allow(dead_code)]

use std::collections::HashMap;

/// A 2-D point in normalised (0.0–1.0) coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Normalised horizontal position (0.0 = left, 1.0 = right).
    pub x: f64,
    /// Normalised vertical position (0.0 = top, 1.0 = bottom).
    pub y: f64,
}

impl Point {
    /// Create a new point.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// An axis-aligned bounding rectangle in normalised coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Top-left corner.
    pub origin: Point,
    /// Width (0.0–1.0).
    pub width: f64,
    /// Height (0.0–1.0).
    pub height: f64,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            origin: Point::new(x, y),
            width,
            height,
        }
    }

    /// Area of the rectangle.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Check whether a point is inside the rectangle.
    #[must_use]
    pub fn contains(&self, p: &Point) -> bool {
        p.x >= self.origin.x
            && p.x <= self.origin.x + self.width
            && p.y >= self.origin.y
            && p.y <= self.origin.y + self.height
    }
}

/// A colour represented as RGBA bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel (255 = fully opaque).
    pub a: u8,
}

impl Color {
    /// Create a new colour.
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque red.
    #[must_use]
    pub fn red() -> Self {
        Self::new(255, 0, 0, 255)
    }

    /// Opaque yellow.
    #[must_use]
    pub fn yellow() -> Self {
        Self::new(255, 255, 0, 255)
    }

    /// Opaque white.
    #[must_use]
    pub fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }
}

/// The kind of drawing tool used to create an annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolKind {
    /// Freehand pen stroke described by a list of control points.
    Pen {
        /// Ordered list of control points forming the stroke path.
        points: Vec<Point>,
    },
    /// Rectangular region highlight.
    Rectangle(Rect),
    /// Circular/ellipse region.
    Ellipse {
        /// Center point of the ellipse.
        center: Point,
        /// Horizontal radius of the ellipse.
        radius_x: f64,
        /// Vertical radius of the ellipse.
        radius_y: f64,
    },
    /// Arrow from one point to another.
    Arrow {
        /// Tail (start) point of the arrow.
        from: Point,
        /// Head (end) point of the arrow.
        to: Point,
    },
    /// Text label placed at a position.
    TextLabel {
        /// Position at which the label is anchored.
        position: Point,
        /// Text content of the label.
        text: String,
    },
}

/// A single drawn annotation on a frame.
#[derive(Debug, Clone)]
pub struct DrawAnnotation {
    /// Unique identifier for this annotation.
    pub id: u64,
    /// Frame number this annotation belongs to.
    pub frame: u64,
    /// Layer this annotation sits on (higher = on top).
    pub layer: u32,
    /// The drawing tool / shape.
    pub tool: ToolKind,
    /// Stroke colour.
    pub color: Color,
    /// Stroke width in pixels.
    pub stroke_width: f32,
    /// Author user ID.
    pub author: String,
    /// Timestamp in milliseconds since epoch.
    pub created_ms: u64,
    /// Whether this annotation is currently visible.
    pub visible: bool,
}

impl DrawAnnotation {
    /// Create a new draw annotation.
    #[must_use]
    pub fn new(
        id: u64,
        frame: u64,
        layer: u32,
        tool: ToolKind,
        color: Color,
        author: impl Into<String>,
        created_ms: u64,
    ) -> Self {
        Self {
            id,
            frame,
            layer,
            tool,
            color,
            stroke_width: 2.0,
            author: author.into(),
            created_ms,
            visible: true,
        }
    }
}

/// A timestamp-anchored text note on a media item.
#[derive(Debug, Clone)]
pub struct TimestampNote {
    /// Unique identifier.
    pub id: u64,
    /// Frame the note is anchored to.
    pub frame: u64,
    /// Duration in frames this note applies to (0 = single frame).
    pub duration_frames: u64,
    /// Note text.
    pub text: String,
    /// Author user ID.
    pub author: String,
    /// Creation timestamp in ms since epoch.
    pub created_ms: u64,
    /// Tag / category.
    pub tag: Option<String>,
}

impl TimestampNote {
    /// Create a new timestamp note.
    #[must_use]
    pub fn new(
        id: u64,
        frame: u64,
        text: impl Into<String>,
        author: impl Into<String>,
        created_ms: u64,
    ) -> Self {
        Self {
            id,
            frame,
            duration_frames: 0,
            text: text.into(),
            author: author.into(),
            created_ms,
            tag: None,
        }
    }

    /// Set the duration span.
    pub fn with_duration(mut self, frames: u64) -> Self {
        self.duration_frames = frames;
        self
    }

    /// Returns true if a given frame falls within this note's span.
    #[must_use]
    pub fn covers_frame(&self, frame: u64) -> bool {
        frame >= self.frame && frame <= self.frame + self.duration_frames
    }
}

/// A named annotation layer grouping draw annotations.
#[derive(Debug, Clone)]
pub struct AnnotationLayer {
    /// Layer index (0 = bottom).
    pub index: u32,
    /// Display name.
    pub name: String,
    /// Whether the layer is visible.
    pub visible: bool,
    /// Whether the layer is locked (no edits allowed).
    pub locked: bool,
}

impl AnnotationLayer {
    /// Create a new annotation layer.
    #[must_use]
    pub fn new(index: u32, name: impl Into<String>) -> Self {
        Self {
            index,
            name: name.into(),
            visible: true,
            locked: false,
        }
    }
}

/// Collection of all annotations for a single review session.
#[derive(Debug, Default)]
pub struct AnnotationCollection {
    /// Draw annotations indexed by ID.
    pub annotations: HashMap<u64, DrawAnnotation>,
    /// Timestamp notes indexed by ID.
    pub notes: HashMap<u64, TimestampNote>,
    /// Layer definitions.
    pub layers: Vec<AnnotationLayer>,
}

impl AnnotationCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a draw annotation.
    pub fn add_annotation(&mut self, annotation: DrawAnnotation) {
        self.annotations.insert(annotation.id, annotation);
    }

    /// Remove a draw annotation by ID.
    pub fn remove_annotation(&mut self, id: u64) -> bool {
        self.annotations.remove(&id).is_some()
    }

    /// Add a timestamp note.
    pub fn add_note(&mut self, note: TimestampNote) {
        self.notes.insert(note.id, note);
    }

    /// Add a layer definition.
    pub fn add_layer(&mut self, layer: AnnotationLayer) {
        self.layers.push(layer);
    }

    /// Get all annotations for a given frame, sorted by layer.
    #[must_use]
    pub fn annotations_at_frame(&self, frame: u64) -> Vec<&DrawAnnotation> {
        let mut result: Vec<&DrawAnnotation> = self
            .annotations
            .values()
            .filter(|a| a.frame == frame && a.visible)
            .collect();
        result.sort_by_key(|a| a.layer);
        result
    }

    /// Get all notes that cover a given frame.
    #[must_use]
    pub fn notes_at_frame(&self, frame: u64) -> Vec<&TimestampNote> {
        self.notes
            .values()
            .filter(|n| n.covers_frame(frame))
            .collect()
    }

    /// Count all annotations.
    #[must_use]
    pub fn annotation_count(&self) -> usize {
        self.annotations.len()
    }

    /// Count all notes.
    #[must_use]
    pub fn note_count(&self) -> usize {
        self.notes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        let d = a.distance_to(&b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_distance_same() {
        let p = Point::new(0.5, 0.5);
        assert_eq!(p.distance_to(&p), 0.0);
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0.0, 0.0, 0.5, 0.4);
        assert!((r.area() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_rect_contains_inside() {
        let r = Rect::new(0.1, 0.1, 0.5, 0.5);
        let p = Point::new(0.3, 0.3);
        assert!(r.contains(&p));
    }

    #[test]
    fn test_rect_contains_outside() {
        let r = Rect::new(0.1, 0.1, 0.5, 0.5);
        let p = Point::new(0.8, 0.8);
        assert!(!r.contains(&p));
    }

    #[test]
    fn test_color_red() {
        let c = Color::red();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_draw_annotation_creation() {
        let ann = DrawAnnotation::new(
            1,
            100,
            0,
            ToolKind::Rectangle(Rect::new(0.1, 0.1, 0.3, 0.3)),
            Color::red(),
            "alice",
            5000,
        );
        assert_eq!(ann.id, 1);
        assert_eq!(ann.frame, 100);
        assert!(ann.visible);
    }

    #[test]
    fn test_timestamp_note_covers_frame_single() {
        let note = TimestampNote::new(1, 50, "Check color", "alice", 0);
        assert!(note.covers_frame(50));
        assert!(!note.covers_frame(51));
    }

    #[test]
    fn test_timestamp_note_covers_frame_span() {
        let note = TimestampNote::new(1, 50, "Check motion", "bob", 0).with_duration(10);
        assert!(note.covers_frame(50));
        assert!(note.covers_frame(55));
        assert!(note.covers_frame(60));
        assert!(!note.covers_frame(61));
    }

    #[test]
    fn test_annotation_layer_defaults() {
        let layer = AnnotationLayer::new(0, "Base");
        assert!(layer.visible);
        assert!(!layer.locked);
    }

    #[test]
    fn test_collection_add_and_count() {
        let mut col = AnnotationCollection::new();
        let ann = DrawAnnotation::new(
            1,
            10,
            0,
            ToolKind::Arrow {
                from: Point::new(0.0, 0.0),
                to: Point::new(0.5, 0.5),
            },
            Color::yellow(),
            "alice",
            100,
        );
        col.add_annotation(ann);
        assert_eq!(col.annotation_count(), 1);
    }

    #[test]
    fn test_collection_remove_annotation() {
        let mut col = AnnotationCollection::new();
        let ann = DrawAnnotation::new(
            2,
            20,
            0,
            ToolKind::TextLabel {
                position: Point::new(0.5, 0.5),
                text: "Hello".into(),
            },
            Color::white(),
            "bob",
            200,
        );
        col.add_annotation(ann);
        assert!(col.remove_annotation(2));
        assert!(!col.remove_annotation(2));
        assert_eq!(col.annotation_count(), 0);
    }

    #[test]
    fn test_collection_annotations_at_frame_sorted_by_layer() {
        let mut col = AnnotationCollection::new();
        let ann1 = DrawAnnotation::new(
            1,
            5,
            2,
            ToolKind::Rectangle(Rect::new(0.0, 0.0, 0.1, 0.1)),
            Color::red(),
            "alice",
            0,
        );
        let ann2 = DrawAnnotation::new(
            2,
            5,
            0,
            ToolKind::Rectangle(Rect::new(0.0, 0.0, 0.2, 0.2)),
            Color::white(),
            "alice",
            0,
        );
        col.add_annotation(ann1);
        col.add_annotation(ann2);
        let at_frame = col.annotations_at_frame(5);
        assert_eq!(at_frame.len(), 2);
        assert_eq!(at_frame[0].layer, 0);
        assert_eq!(at_frame[1].layer, 2);
    }

    #[test]
    fn test_collection_notes_at_frame() {
        let mut col = AnnotationCollection::new();
        col.add_note(TimestampNote::new(1, 100, "Note A", "alice", 0).with_duration(5));
        col.add_note(TimestampNote::new(2, 200, "Note B", "bob", 0));
        let notes = col.notes_at_frame(102);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].text, "Note A");
    }

    #[test]
    fn test_collection_note_count() {
        let mut col = AnnotationCollection::new();
        col.add_note(TimestampNote::new(1, 10, "A", "x", 0));
        col.add_note(TimestampNote::new(2, 20, "B", "y", 0));
        assert_eq!(col.note_count(), 2);
    }
}
