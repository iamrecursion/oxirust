//! Collaborative annotation layer for frame-level review and markup.
//!
//! Provides geometric shapes, per-annotation metadata, and a layer container
//! for managing annotations within a collaborative review workflow.

#![allow(dead_code)]

/// Geometric shape of an annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationShape {
    /// A single point at `(x, y)` in normalised frame coordinates.
    Point(f32, f32),
    /// An axis-aligned rectangle `(x, y, width, height)`.
    Rect(f32, f32, f32, f32),
    /// An arrow from `(x1, y1)` to `(x2, y2)`.
    Arrow(f32, f32, f32, f32),
}

impl AnnotationShape {
    /// Return the geometric centre of the shape.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        match *self {
            Self::Point(x, y) => (x, y),
            Self::Rect(x, y, w, h) => (x + w / 2.0, y + h / 2.0),
            Self::Arrow(x1, y1, x2, y2) => ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        }
    }
}

/// A single annotation placed by a collaborator on an asset frame.
#[derive(Debug, Clone)]
pub struct Annotation {
    /// Unique annotation identifier within the layer.
    pub id: u64,
    /// Display name of the annotation's author.
    pub author: String,
    /// Geometric shape of the annotation.
    pub shape: AnnotationShape,
    /// RGB display colour of the annotation.
    pub color: [u8; 3],
    /// Optional text comment attached to the annotation.
    pub text: String,
    /// Wall-clock creation time in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Whether this annotation has been marked as resolved.
    pub resolved: bool,
}

impl Annotation {
    /// Mark the annotation as resolved.
    pub fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Return the age of this annotation relative to `now` (milliseconds).
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// A collection of annotations attached to a specific asset and optional frame.
pub struct AnnotationLayer {
    /// Asset identifier this layer belongs to.
    pub asset_id: String,
    /// Optional frame number; `None` means the annotation is asset-wide.
    pub frame: Option<u64>,
    /// All stored annotations.
    pub annotations: Vec<Annotation>,
    /// Counter used to assign unique ids.
    pub next_id: u64,
}

impl AnnotationLayer {
    /// Create a new, empty `AnnotationLayer`.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, frame: Option<u64>) -> Self {
        Self {
            asset_id: asset_id.into(),
            frame,
            annotations: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a new annotation and return its assigned id.
    pub fn add(
        &mut self,
        author: impl Into<String>,
        shape: AnnotationShape,
        color: [u8; 3],
        text: impl Into<String>,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.annotations.push(Annotation {
            id,
            author: author.into(),
            shape,
            color,
            text: text.into(),
            timestamp_ms: now_ms,
            resolved: false,
        });
        id
    }

    /// Mark the annotation with the given `id` as resolved.
    ///
    /// Returns `true` if the annotation was found and resolved.
    pub fn resolve(&mut self, id: u64) -> bool {
        if let Some(ann) = self.annotations.iter_mut().find(|a| a.id == id) {
            ann.resolve();
            true
        } else {
            false
        }
    }

    /// Return references to all unresolved annotations.
    #[must_use]
    pub fn unresolved(&self) -> Vec<&Annotation> {
        self.annotations.iter().filter(|a| !a.resolved).collect()
    }

    /// Return references to all annotations created by `author`.
    #[must_use]
    pub fn by_author(&self, author: &str) -> Vec<&Annotation> {
        self.annotations
            .iter()
            .filter(|a| a.author == author)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer() -> AnnotationLayer {
        AnnotationLayer::new("asset-001", Some(42))
    }

    // ---- AnnotationShape ----

    #[test]
    fn test_point_center() {
        let s = AnnotationShape::Point(0.3, 0.7);
        let (cx, cy) = s.center();
        assert!((cx - 0.3).abs() < 1e-6);
        assert!((cy - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_rect_center() {
        // rect at (0,0) with w=4, h=2 → center (2, 1)
        let s = AnnotationShape::Rect(0.0, 0.0, 4.0, 2.0);
        let (cx, cy) = s.center();
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_arrow_center() {
        let s = AnnotationShape::Arrow(0.0, 0.0, 1.0, 1.0);
        let (cx, cy) = s.center();
        assert!((cx - 0.5).abs() < 1e-6);
        assert!((cy - 0.5).abs() < 1e-6);
    }

    // ---- Annotation ----

    #[test]
    fn test_annotation_resolve() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.5, 0.5),
            [255, 0, 0],
            "note",
            1000,
        );
        layer.resolve(id);
        assert!(layer.annotations[0].resolved);
    }

    #[test]
    fn test_annotation_age_ms() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.0, 0.0),
            [0, 0, 0],
            "",
            1000,
        );
        let ann = layer
            .annotations
            .iter()
            .find(|a| a.id == id)
            .expect("collab test operation should succeed");
        assert_eq!(ann.age_ms(3000), 2000);
    }

    #[test]
    fn test_annotation_age_ms_before_creation() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.0, 0.0),
            [0, 0, 0],
            "",
            5000,
        );
        let ann = layer
            .annotations
            .iter()
            .find(|a| a.id == id)
            .expect("collab test operation should succeed");
        // now < timestamp → saturating_sub → 0
        assert_eq!(ann.age_ms(1000), 0);
    }

    // ---- AnnotationLayer ----

    #[test]
    fn test_layer_add_returns_incrementing_ids() {
        let mut layer = make_layer();
        let id1 = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        let id2 = layer.add("b", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_layer_new_is_empty() {
        let layer = make_layer();
        assert!(layer.annotations.is_empty());
        assert_eq!(layer.asset_id, "asset-001");
        assert_eq!(layer.frame, Some(42));
    }

    #[test]
    fn test_layer_resolve_returns_true_when_found() {
        let mut layer = make_layer();
        let id = layer.add("x", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        assert!(layer.resolve(id));
    }

    #[test]
    fn test_layer_resolve_returns_false_when_not_found() {
        let mut layer = make_layer();
        assert!(!layer.resolve(999));
    }

    #[test]
    fn test_layer_unresolved_filters_resolved() {
        let mut layer = make_layer();
        let id1 = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add("b", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.resolve(id1);
        let unresolved = layer.unresolved();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].author, "b");
    }

    #[test]
    fn test_layer_unresolved_all_resolved() {
        let mut layer = make_layer();
        let id = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.resolve(id);
        assert!(layer.unresolved().is_empty());
    }

    #[test]
    fn test_layer_by_author() {
        let mut layer = make_layer();
        layer.add("alice", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add("bob", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add(
            "alice",
            AnnotationShape::Rect(0.0, 0.0, 1.0, 1.0),
            [0, 0, 0],
            "",
            0,
        );
        let alice = layer.by_author("alice");
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|a| a.author == "alice"));
    }

    #[test]
    fn test_layer_by_author_missing() {
        let layer = make_layer();
        assert!(layer.by_author("nobody").is_empty());
    }

    #[test]
    fn test_annotation_color_stored() {
        let mut layer = make_layer();
        layer.add("a", AnnotationShape::Point(0.0, 0.0), [10, 20, 30], "", 0);
        assert_eq!(layer.annotations[0].color, [10, 20, 30]);
    }

    #[test]
    fn test_annotation_text_stored() {
        let mut layer = make_layer();
        layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "hello", 0);
        assert_eq!(layer.annotations[0].text, "hello");
    }

    #[test]
    fn test_layer_frame_none() {
        let layer = AnnotationLayer::new("asset-002", None);
        assert_eq!(layer.frame, None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestamp-anchored annotations
// ─────────────────────────────────────────────────────────────────────────────

/// A precise point in a media timeline expressed in microseconds from the
/// project start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MediaTimestamp {
    /// Microseconds from the beginning of the project (or clip, depending on
    /// context).
    pub micros: u64,
}

impl MediaTimestamp {
    /// Create a new media timestamp from microseconds.
    #[must_use]
    pub fn from_micros(micros: u64) -> Self {
        Self { micros }
    }

    /// Create a timestamp from seconds (truncated to whole microseconds).
    #[must_use]
    pub fn from_secs_f64(secs: f64) -> Self {
        let micros = (secs * 1_000_000.0) as u64;
        Self { micros }
    }

    /// Return the timestamp in seconds.
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.micros as f64 / 1_000_000.0
    }

    /// Return the timestamp in milliseconds (truncated).
    #[must_use]
    pub fn as_millis(self) -> u64 {
        self.micros / 1_000
    }
}

impl std::fmt::Display for MediaTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.micros / 1_000_000;
        let ms = (self.micros % 1_000_000) / 1_000;
        write!(f, "{secs}.{ms:03}s")
    }
}

/// An optional time range anchoring an annotation to a span of the media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeAnchor {
    /// Start of the annotation's temporal extent.
    pub start: MediaTimestamp,
    /// End of the annotation's temporal extent (exclusive).
    pub end: MediaTimestamp,
}

impl TimeAnchor {
    /// Create a new time anchor from start and end microseconds.
    #[must_use]
    pub fn new(start_micros: u64, end_micros: u64) -> Self {
        Self {
            start: MediaTimestamp::from_micros(start_micros),
            end: MediaTimestamp::from_micros(end_micros),
        }
    }

    /// Duration of this anchor in microseconds.
    #[must_use]
    pub fn duration_micros(self) -> u64 {
        self.end.micros.saturating_sub(self.start.micros)
    }

    /// Check whether a given timestamp falls within this anchor.
    #[must_use]
    pub fn contains(self, ts: MediaTimestamp) -> bool {
        ts >= self.start && ts < self.end
    }

    /// Check whether two time anchors overlap.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// An annotation that is anchored to a specific time range in the media
/// timeline in addition to (optionally) a spatial shape on a frame.
#[derive(Debug, Clone)]
pub struct TimestampedAnnotation {
    /// Unique identifier within the owning collection.
    pub id: u64,
    /// Display name of the annotation's author.
    pub author: String,
    /// Optional spatial shape (if the annotation is also visually placed).
    pub shape: Option<AnnotationShape>,
    /// Temporal anchor.
    pub anchor: TimeAnchor,
    /// Free-text comment.
    pub text: String,
    /// RGB display colour.
    pub color: [u8; 3],
    /// Wall-clock creation time in milliseconds since the Unix epoch.
    pub created_at_ms: u64,
    /// Whether this annotation has been resolved.
    pub resolved: bool,
    /// Optional tags for categorisation (e.g. "visual", "audio", "pacing").
    pub tags: Vec<String>,
}

impl TimestampedAnnotation {
    /// Mark the annotation as resolved.
    pub fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Check whether this annotation overlaps with `anchor`.
    #[must_use]
    pub fn overlaps_anchor(&self, other: TimeAnchor) -> bool {
        self.anchor.overlaps(other)
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
}

/// A collection of timestamp-anchored annotations for a single project or
/// clip, supporting querying by time range and author.
#[derive(Debug, Default)]
pub struct TimestampedAnnotationLayer {
    annotations: Vec<TimestampedAnnotation>,
    next_id: u64,
}

impl TimestampedAnnotationLayer {
    /// Create an empty layer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new timestamped annotation and return its id.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        author: impl Into<String>,
        anchor: TimeAnchor,
        text: impl Into<String>,
        color: [u8; 3],
        shape: Option<AnnotationShape>,
        created_at_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.annotations.push(TimestampedAnnotation {
            id,
            author: author.into(),
            shape,
            anchor,
            text: text.into(),
            color,
            created_at_ms,
            resolved: false,
            tags: Vec::new(),
        });
        id
    }

    /// Resolve an annotation by id.  Returns `true` if found.
    pub fn resolve(&mut self, id: u64) -> bool {
        if let Some(ann) = self.annotations.iter_mut().find(|a| a.id == id) {
            ann.resolve();
            true
        } else {
            false
        }
    }

    /// Query all annotations that overlap with `anchor`.
    #[must_use]
    pub fn overlapping(&self, anchor: TimeAnchor) -> Vec<&TimestampedAnnotation> {
        self.annotations
            .iter()
            .filter(|a| a.anchor.overlaps(anchor))
            .collect()
    }

    /// Query all annotations by author.
    #[must_use]
    pub fn by_author(&self, author: &str) -> Vec<&TimestampedAnnotation> {
        self.annotations
            .iter()
            .filter(|a| a.author == author)
            .collect()
    }

    /// All unresolved annotations sorted by anchor start.
    #[must_use]
    pub fn unresolved_sorted(&self) -> Vec<&TimestampedAnnotation> {
        let mut result: Vec<&TimestampedAnnotation> =
            self.annotations.iter().filter(|a| !a.resolved).collect();
        result.sort_by_key(|a| a.anchor.start);
        result
    }

    /// Total number of annotations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// True when the layer has no annotations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }
}

#[cfg(test)]
mod timestamped_tests {
    use super::*;

    fn make_anchor(start_s: f64, end_s: f64) -> TimeAnchor {
        TimeAnchor::new((start_s * 1_000_000.0) as u64, (end_s * 1_000_000.0) as u64)
    }

    #[test]
    fn test_media_timestamp_display() {
        let ts = MediaTimestamp::from_micros(5_123_456);
        assert_eq!(ts.to_string(), "5.123s");
    }

    #[test]
    fn test_media_timestamp_round_trip() {
        let ts = MediaTimestamp::from_secs_f64(3.5);
        assert!((ts.as_secs_f64() - 3.5).abs() < 1e-4);
    }

    #[test]
    fn test_time_anchor_contains() {
        let a = make_anchor(1.0, 3.0);
        assert!(a.contains(MediaTimestamp::from_secs_f64(1.0)));
        assert!(a.contains(MediaTimestamp::from_secs_f64(2.5)));
        assert!(!a.contains(MediaTimestamp::from_secs_f64(3.0))); // exclusive end
        assert!(!a.contains(MediaTimestamp::from_secs_f64(0.5)));
    }

    #[test]
    fn test_time_anchor_overlaps() {
        let a = make_anchor(0.0, 5.0);
        let b = make_anchor(3.0, 8.0);
        let c = make_anchor(5.0, 10.0);
        assert!(a.overlaps(b));
        assert!(!a.overlaps(c)); // just touching at boundary
    }

    #[test]
    fn test_time_anchor_duration() {
        let a = make_anchor(1.0, 3.5);
        assert_eq!(a.duration_micros(), 2_500_000);
    }

    #[test]
    fn test_layer_add_and_count() {
        let mut layer = TimestampedAnnotationLayer::new();
        layer.add(
            "alice",
            make_anchor(0.0, 2.0),
            "check scene",
            [255, 0, 0],
            None,
            1_000,
        );
        layer.add(
            "bob",
            make_anchor(3.0, 5.0),
            "audio issue",
            [0, 255, 0],
            None,
            2_000,
        );
        assert_eq!(layer.len(), 2);
        assert!(!layer.is_empty());
    }

    #[test]
    fn test_layer_overlapping_query() {
        let mut layer = TimestampedAnnotationLayer::new();
        layer.add("alice", make_anchor(0.0, 2.0), "a", [0, 0, 0], None, 0);
        layer.add("bob", make_anchor(5.0, 8.0), "b", [0, 0, 0], None, 0);
        layer.add("carol", make_anchor(1.5, 4.0), "c", [0, 0, 0], None, 0);

        let query = make_anchor(1.0, 3.0);
        let results = layer.overlapping(query);
        assert_eq!(results.len(), 2); // alice (0-2) and carol (1.5-4) overlap with (1-3)
    }

    #[test]
    fn test_layer_resolve() {
        let mut layer = TimestampedAnnotationLayer::new();
        let id = layer.add("x", make_anchor(0.0, 1.0), "note", [0, 0, 0], None, 0);
        assert!(layer.resolve(id));
        let unresolved = layer.unresolved_sorted();
        assert!(unresolved.is_empty());
    }

    #[test]
    fn test_layer_unresolved_sorted_by_start() {
        let mut layer = TimestampedAnnotationLayer::new();
        layer.add("a", make_anchor(5.0, 7.0), "later", [0, 0, 0], None, 0);
        layer.add("b", make_anchor(1.0, 3.0), "earlier", [0, 0, 0], None, 0);
        let sorted = layer.unresolved_sorted();
        assert_eq!(sorted.len(), 2);
        assert!(sorted[0].anchor.start < sorted[1].anchor.start);
    }

    #[test]
    fn test_annotation_add_tag() {
        let mut layer = TimestampedAnnotationLayer::new();
        let id = layer.add("a", make_anchor(0.0, 1.0), "", [0, 0, 0], None, 0);
        if let Some(ann) = layer.annotations.iter_mut().find(|a| a.id == id) {
            ann.add_tag("visual");
            ann.add_tag("visual"); // duplicate
            ann.add_tag("audio");
            assert_eq!(ann.tags.len(), 2);
        }
    }
}
