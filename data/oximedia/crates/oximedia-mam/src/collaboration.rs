//! Real-time comments, annotations, and review markers on media assets.
//!
//! This module provides collaborative review capabilities for media assets:
//! - Timestamped comments anchored to a specific frame or time range
//! - Point/region annotations with drawing metadata
//! - Review markers ("review gates") that block asset progression until resolved
//! - Resolution tracking so review threads can be closed
//! - Reaction emoji on comments

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Comment
// ---------------------------------------------------------------------------

/// A comment left by a reviewer on an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetComment {
    /// Unique comment id.
    pub id: Uuid,
    /// Asset this comment belongs to.
    pub asset_id: Uuid,
    /// User who wrote the comment.
    pub author_id: Uuid,
    /// Display name of the author.
    pub author_name: String,
    /// Comment text (Markdown supported).
    pub body: String,
    /// Optional parent comment id for threaded replies.
    pub parent_id: Option<Uuid>,
    /// Optional timecode anchor in milliseconds.
    pub timecode_ms: Option<u64>,
    /// Optional end timecode for a range comment.
    pub timecode_end_ms: Option<u64>,
    /// Whether the comment thread is resolved.
    pub resolved: bool,
    /// User who resolved the thread (if resolved).
    pub resolved_by: Option<Uuid>,
    /// When the thread was resolved.
    pub resolved_at: Option<DateTime<Utc>>,
    /// Emoji reactions: emoji → list of user ids.
    pub reactions: HashMap<String, Vec<Uuid>>,
    /// When the comment was created.
    pub created_at: DateTime<Utc>,
    /// When the comment was last edited.
    pub updated_at: DateTime<Utc>,
}

impl AssetComment {
    /// Create a new top-level comment.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        author_id: Uuid,
        author_name: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            asset_id,
            author_id,
            author_name: author_name.into(),
            body: body.into(),
            parent_id: None,
            timecode_ms: None,
            timecode_end_ms: None,
            resolved: false,
            resolved_by: None,
            resolved_at: None,
            reactions: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Builder: anchor the comment at a timecode (milliseconds).
    #[must_use]
    pub fn at_timecode(mut self, ms: u64) -> Self {
        self.timecode_ms = Some(ms);
        self
    }

    /// Builder: set a time range anchor.
    #[must_use]
    pub fn at_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.timecode_ms = Some(start_ms);
        self.timecode_end_ms = Some(end_ms);
        self
    }

    /// Builder: make this a reply to another comment.
    #[must_use]
    pub fn as_reply(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Edit the comment body.
    pub fn edit(&mut self, new_body: impl Into<String>) {
        self.body = new_body.into();
        self.updated_at = Utc::now();
    }

    /// Resolve the comment thread.
    pub fn resolve(&mut self, resolved_by: Uuid) {
        self.resolved = true;
        self.resolved_by = Some(resolved_by);
        self.resolved_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Reopen the comment thread.
    pub fn reopen(&mut self) {
        self.resolved = false;
        self.resolved_by = None;
        self.resolved_at = None;
        self.updated_at = Utc::now();
    }

    /// Add an emoji reaction from a user.
    ///
    /// Returns `true` if the reaction was newly added, `false` if already present.
    pub fn add_reaction(&mut self, emoji: impl Into<String>, user_id: Uuid) -> bool {
        let emoji = emoji.into();
        let users = self.reactions.entry(emoji).or_default();
        if users.contains(&user_id) {
            return false;
        }
        users.push(user_id);
        true
    }

    /// Remove an emoji reaction from a user.
    ///
    /// Returns `true` if the reaction was removed.
    pub fn remove_reaction(&mut self, emoji: &str, user_id: Uuid) -> bool {
        if let Some(users) = self.reactions.get_mut(emoji) {
            let before = users.len();
            users.retain(|u| *u != user_id);
            let removed = users.len() < before;
            let now_empty = users.is_empty();
            if now_empty {
                self.reactions.remove(emoji);
            }
            return removed;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Annotation
// ---------------------------------------------------------------------------

/// Shape type for a region annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationShape {
    /// A single point (x, y) in normalised 0-1 coordinates.
    Point { x: f64, y: f64 },
    /// A rectangle defined by top-left corner and dimensions (all normalised).
    Rect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    },
    /// A freehand polyline.
    Polyline { points: Vec<(f64, f64)> },
    /// An ellipse (normalised cx, cy, rx, ry).
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
    /// An arrow from one point to another.
    Arrow { from: (f64, f64), to: (f64, f64) },
}

/// A visual annotation drawn on an asset frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Unique annotation id.
    pub id: Uuid,
    /// Asset this annotation belongs to.
    pub asset_id: Uuid,
    /// Author of the annotation.
    pub author_id: Uuid,
    /// Display name.
    pub author_name: String,
    /// Shape drawn.
    pub shape: AnnotationShape,
    /// Timecode (ms) the annotation is anchored to.
    pub timecode_ms: u64,
    /// Optional label text shown next to the shape.
    pub label: Option<String>,
    /// RGBA stroke colour as hex string (e.g. `"#FF0000FF"`).
    pub colour: String,
    /// Stroke width in pixels.
    pub stroke_width: f32,
    /// Optional linked comment id.
    pub comment_id: Option<Uuid>,
    /// When the annotation was created.
    pub created_at: DateTime<Utc>,
}

impl Annotation {
    /// Create a new annotation.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        author_id: Uuid,
        author_name: impl Into<String>,
        shape: AnnotationShape,
        timecode_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id,
            author_id,
            author_name: author_name.into(),
            shape,
            timecode_ms,
            label: None,
            colour: "#FF0000FF".to_string(),
            stroke_width: 2.0,
            comment_id: None,
            created_at: Utc::now(),
        }
    }

    /// Builder: attach a label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Builder: set colour.
    #[must_use]
    pub fn with_colour(mut self, colour: impl Into<String>) -> Self {
        self.colour = colour.into();
        self
    }

    /// Builder: link to a comment.
    #[must_use]
    pub fn with_comment(mut self, comment_id: Uuid) -> Self {
        self.comment_id = Some(comment_id);
        self
    }
}

// ---------------------------------------------------------------------------
// Review marker
// ---------------------------------------------------------------------------

/// Severity of a review marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerSeverity {
    /// Informational note — does not block.
    Info,
    /// Minor issue — should be fixed before delivery.
    Minor,
    /// Major issue — must be fixed before delivery.
    Major,
    /// Blocker — asset cannot progress until resolved.
    Blocker,
}

impl MarkerSeverity {
    /// Returns `true` if this severity level prevents asset progression.
    #[must_use]
    pub const fn is_blocking(&self) -> bool {
        matches!(self, Self::Blocker)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Minor => "minor",
            Self::Major => "major",
            Self::Blocker => "blocker",
        }
    }
}

/// Status of a review marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerStatus {
    /// Marker is open and unresolved.
    Open,
    /// Marker is under active discussion.
    InProgress,
    /// Marker has been resolved.
    Resolved,
    /// Marker was dismissed as not applicable.
    Dismissed,
}

impl MarkerStatus {
    /// Returns `true` if the marker is still actionable.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        matches!(self, Self::Open | Self::InProgress)
    }
}

/// A review marker that flags a specific issue on an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewMarker {
    /// Unique marker id.
    pub id: Uuid,
    /// Asset the marker belongs to.
    pub asset_id: Uuid,
    /// User who created the marker.
    pub created_by: Uuid,
    /// Display name of creator.
    pub creator_name: String,
    /// Short title describing the issue.
    pub title: String,
    /// Detailed description.
    pub description: Option<String>,
    /// Severity level.
    pub severity: MarkerSeverity,
    /// Current status.
    pub status: MarkerStatus,
    /// Optional timecode anchor (ms).
    pub timecode_ms: Option<u64>,
    /// User assigned to resolve this marker.
    pub assignee_id: Option<Uuid>,
    /// Assignee display name.
    pub assignee_name: Option<String>,
    /// Linked comment ids.
    pub comment_ids: Vec<Uuid>,
    /// When the marker was created.
    pub created_at: DateTime<Utc>,
    /// When the marker was last updated.
    pub updated_at: DateTime<Utc>,
    /// When the marker was resolved/dismissed.
    pub closed_at: Option<DateTime<Utc>>,
}

impl ReviewMarker {
    /// Create a new review marker.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        created_by: Uuid,
        creator_name: impl Into<String>,
        title: impl Into<String>,
        severity: MarkerSeverity,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            asset_id,
            created_by,
            creator_name: creator_name.into(),
            title: title.into(),
            description: None,
            severity,
            status: MarkerStatus::Open,
            timecode_ms: None,
            assignee_id: None,
            assignee_name: None,
            comment_ids: Vec::new(),
            created_at: now,
            updated_at: now,
            closed_at: None,
        }
    }

    /// Builder: set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: anchor at timecode.
    #[must_use]
    pub fn at_timecode(mut self, ms: u64) -> Self {
        self.timecode_ms = Some(ms);
        self
    }

    /// Builder: assign to a user.
    #[must_use]
    pub fn assign_to(mut self, user_id: Uuid, name: impl Into<String>) -> Self {
        self.assignee_id = Some(user_id);
        self.assignee_name = Some(name.into());
        self
    }

    /// Transition to in-progress.
    pub fn start_progress(&mut self) {
        self.status = MarkerStatus::InProgress;
        self.updated_at = Utc::now();
    }

    /// Resolve the marker.
    pub fn resolve(&mut self) {
        self.status = MarkerStatus::Resolved;
        let now = Utc::now();
        self.updated_at = now;
        self.closed_at = Some(now);
    }

    /// Dismiss the marker.
    pub fn dismiss(&mut self) {
        self.status = MarkerStatus::Dismissed;
        let now = Utc::now();
        self.updated_at = now;
        self.closed_at = Some(now);
    }

    /// Link a comment to this marker.
    pub fn link_comment(&mut self, comment_id: Uuid) {
        if !self.comment_ids.contains(&comment_id) {
            self.comment_ids.push(comment_id);
            self.updated_at = Utc::now();
        }
    }

    /// Returns `true` if this marker is blocking asset progression.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        self.severity.is_blocking() && self.status.is_open()
    }
}

// ---------------------------------------------------------------------------
// Collaboration session
// ---------------------------------------------------------------------------

/// Aggregated collaboration state for an asset (comments + annotations + markers).
#[derive(Debug, Clone)]
pub struct CollaborationSession {
    pub asset_id: Uuid,
    pub comments: Vec<AssetComment>,
    pub annotations: Vec<Annotation>,
    pub markers: Vec<ReviewMarker>,
}

impl CollaborationSession {
    /// Create a new empty session.
    #[must_use]
    pub fn new(asset_id: Uuid) -> Self {
        Self {
            asset_id,
            comments: Vec::new(),
            annotations: Vec::new(),
            markers: Vec::new(),
        }
    }

    /// Add a comment.
    pub fn add_comment(&mut self, comment: AssetComment) {
        self.comments.push(comment);
    }

    /// Add an annotation.
    pub fn add_annotation(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    /// Add a review marker.
    pub fn add_marker(&mut self, marker: ReviewMarker) {
        self.markers.push(marker);
    }

    /// Returns all open blocking markers.
    #[must_use]
    pub fn blocking_markers(&self) -> Vec<&ReviewMarker> {
        self.markers.iter().filter(|m| m.is_blocking()).collect()
    }

    /// Returns `true` if the asset is blocked by unresolved blockers.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.markers.iter().any(|m| m.is_blocking())
    }

    /// Total unresolved comment threads.
    #[must_use]
    pub fn open_comment_count(&self) -> usize {
        self.comments
            .iter()
            .filter(|c| !c.resolved && c.parent_id.is_none())
            .count()
    }

    /// Comments anchored near a given timecode (within `tolerance_ms`).
    #[must_use]
    pub fn comments_near_timecode(&self, ms: u64, tolerance_ms: u64) -> Vec<&AssetComment> {
        self.comments
            .iter()
            .filter(|c| {
                if let Some(tc) = c.timecode_ms {
                    tc.abs_diff(ms) <= tolerance_ms
                } else {
                    false
                }
            })
            .collect()
    }

    /// Annotations at a given timecode (exact match).
    #[must_use]
    pub fn annotations_at(&self, ms: u64) -> Vec<&Annotation> {
        self.annotations
            .iter()
            .filter(|a| a.timecode_ms == ms)
            .collect()
    }

    /// Summary statistics for the session.
    #[must_use]
    pub fn summary(&self) -> CollaborationSummary {
        let open_markers = self.markers.iter().filter(|m| m.status.is_open()).count();
        let blocking = self.markers.iter().filter(|m| m.is_blocking()).count();
        CollaborationSummary {
            total_comments: self.comments.len(),
            open_threads: self.open_comment_count(),
            total_annotations: self.annotations.len(),
            total_markers: self.markers.len(),
            open_markers,
            blocking_markers: blocking,
        }
    }
}

/// Summary statistics for a collaboration session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSummary {
    pub total_comments: usize,
    pub open_threads: usize,
    pub total_annotations: usize,
    pub total_markers: usize,
    pub open_markers: usize,
    pub blocking_markers: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    // --- AssetComment ---

    #[test]
    fn test_comment_creation() {
        let aid = uid();
        let auth = uid();
        let c = AssetComment::new(aid, auth, "Alice", "Looks good!");
        assert_eq!(c.asset_id, aid);
        assert_eq!(c.author_id, auth);
        assert_eq!(c.body, "Looks good!");
        assert!(!c.resolved);
        assert!(c.parent_id.is_none());
    }

    #[test]
    fn test_comment_at_timecode() {
        let c = AssetComment::new(uid(), uid(), "Alice", "Check this frame").at_timecode(12_345);
        assert_eq!(c.timecode_ms, Some(12_345));
        assert!(c.timecode_end_ms.is_none());
    }

    #[test]
    fn test_comment_at_range() {
        let c = AssetComment::new(uid(), uid(), "Bob", "Bad audio segment").at_range(5_000, 8_000);
        assert_eq!(c.timecode_ms, Some(5_000));
        assert_eq!(c.timecode_end_ms, Some(8_000));
    }

    #[test]
    fn test_comment_reply() {
        let parent_id = uid();
        let c = AssetComment::new(uid(), uid(), "Carol", "Agreed").as_reply(parent_id);
        assert_eq!(c.parent_id, Some(parent_id));
    }

    #[test]
    fn test_comment_edit() {
        let mut c = AssetComment::new(uid(), uid(), "Dave", "Original body");
        c.edit("Updated body");
        assert_eq!(c.body, "Updated body");
    }

    #[test]
    fn test_comment_resolve_reopen() {
        let mut c = AssetComment::new(uid(), uid(), "Eve", "Issue here");
        let resolver = uid();
        c.resolve(resolver);
        assert!(c.resolved);
        assert_eq!(c.resolved_by, Some(resolver));
        assert!(c.resolved_at.is_some());

        c.reopen();
        assert!(!c.resolved);
        assert!(c.resolved_by.is_none());
        assert!(c.resolved_at.is_none());
    }

    #[test]
    fn test_comment_reactions() {
        let mut c = AssetComment::new(uid(), uid(), "Frank", "Nice shot");
        let u1 = uid();
        let u2 = uid();

        assert!(c.add_reaction("👍", u1));
        assert!(c.add_reaction("👍", u2));
        assert!(!c.add_reaction("👍", u1)); // duplicate

        assert_eq!(c.reactions["👍"].len(), 2);

        let removed = c.remove_reaction("👍", u1);
        assert!(removed);
        assert_eq!(c.reactions["👍"].len(), 1);

        // Remove last reaction — key should disappear
        c.remove_reaction("👍", u2);
        assert!(!c.reactions.contains_key("👍"));
    }

    // --- Annotation ---

    #[test]
    fn test_annotation_point() {
        let a = Annotation::new(
            uid(),
            uid(),
            "Grace",
            AnnotationShape::Point { x: 0.5, y: 0.5 },
            1000,
        );
        assert_eq!(a.timecode_ms, 1000);
        assert!(matches!(a.shape, AnnotationShape::Point { .. }));
    }

    #[test]
    fn test_annotation_rect_with_label() {
        let a = Annotation::new(
            uid(),
            uid(),
            "Henry",
            AnnotationShape::Rect {
                x: 0.1,
                y: 0.1,
                width: 0.5,
                height: 0.3,
            },
            2_000,
        )
        .with_label("Logo area")
        .with_colour("#00FF00FF");
        assert_eq!(a.label.as_deref(), Some("Logo area"));
        assert_eq!(a.colour, "#00FF00FF");
    }

    #[test]
    fn test_annotation_with_comment_link() {
        let cid = uid();
        let a = Annotation::new(
            uid(),
            uid(),
            "Iris",
            AnnotationShape::Point { x: 0.0, y: 0.0 },
            0,
        )
        .with_comment(cid);
        assert_eq!(a.comment_id, Some(cid));
    }

    // --- ReviewMarker ---

    #[test]
    fn test_marker_creation() {
        let m = ReviewMarker::new(
            uid(),
            uid(),
            "Jake",
            "Color banding visible",
            MarkerSeverity::Major,
        );
        assert_eq!(m.status, MarkerStatus::Open);
        assert!(!m.is_blocking()); // Major is not Blocker
    }

    #[test]
    fn test_marker_blocker() {
        let m = ReviewMarker::new(
            uid(),
            uid(),
            "Kim",
            "Black frame at 00:01:00",
            MarkerSeverity::Blocker,
        );
        assert!(m.is_blocking());
    }

    #[test]
    fn test_marker_resolve() {
        let mut m = ReviewMarker::new(uid(), uid(), "Leo", "Audio sync off", MarkerSeverity::Major);
        m.start_progress();
        assert_eq!(m.status, MarkerStatus::InProgress);
        m.resolve();
        assert_eq!(m.status, MarkerStatus::Resolved);
        assert!(m.closed_at.is_some());
        assert!(!m.is_blocking());
    }

    #[test]
    fn test_marker_dismiss() {
        let mut m = ReviewMarker::new(uid(), uid(), "Mia", "Not an issue", MarkerSeverity::Minor);
        m.dismiss();
        assert_eq!(m.status, MarkerStatus::Dismissed);
        assert!(!m.status.is_open());
    }

    #[test]
    fn test_marker_link_comment() {
        let mut m = ReviewMarker::new(uid(), uid(), "Ned", "Noise in audio", MarkerSeverity::Minor);
        let cid = uid();
        m.link_comment(cid);
        assert!(m.comment_ids.contains(&cid));
        // Duplicate link should not add twice
        m.link_comment(cid);
        assert_eq!(m.comment_ids.len(), 1);
    }

    #[test]
    fn test_marker_assign() {
        let assignee = uid();
        let m = ReviewMarker::new(
            uid(),
            uid(),
            "Olive",
            "Wrong aspect ratio",
            MarkerSeverity::Blocker,
        )
        .assign_to(assignee, "Penny");
        assert_eq!(m.assignee_id, Some(assignee));
        assert_eq!(m.assignee_name.as_deref(), Some("Penny"));
    }

    // --- CollaborationSession ---

    #[test]
    fn test_session_is_blocked() {
        let aid = uid();
        let mut session = CollaborationSession::new(aid);

        let blocker = ReviewMarker::new(
            uid(),
            uid(),
            "Quinn",
            "Missing audio",
            MarkerSeverity::Blocker,
        );
        session.add_marker(blocker);
        assert!(session.is_blocked());

        let mut resolved =
            ReviewMarker::new(uid(), uid(), "Ray", "Color issue", MarkerSeverity::Blocker);
        resolved.resolve();
        session.add_marker(resolved);

        // Still blocked because first marker is open
        assert!(session.is_blocked());
        assert_eq!(session.blocking_markers().len(), 1);
    }

    #[test]
    fn test_session_comments_near_timecode() {
        let aid = uid();
        let mut session = CollaborationSession::new(aid);
        session.add_comment(AssetComment::new(uid(), uid(), "Sam", "Early").at_timecode(1_000));
        session.add_comment(AssetComment::new(uid(), uid(), "Tina", "Later").at_timecode(5_000));

        let near = session.comments_near_timecode(1_100, 200);
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].timecode_ms, Some(1_000));
    }

    #[test]
    fn test_session_annotations_at() {
        let aid = uid();
        let mut session = CollaborationSession::new(aid);
        session.add_annotation(Annotation::new(
            uid(),
            uid(),
            "Uma",
            AnnotationShape::Point { x: 0.5, y: 0.5 },
            2_000,
        ));
        session.add_annotation(Annotation::new(
            uid(),
            uid(),
            "Vera",
            AnnotationShape::Point { x: 0.1, y: 0.1 },
            5_000,
        ));

        assert_eq!(session.annotations_at(2_000).len(), 1);
        assert_eq!(session.annotations_at(9_999).len(), 0);
    }

    #[test]
    fn test_session_open_comment_count() {
        let aid = uid();
        let mut session = CollaborationSession::new(aid);
        let mut c1 = AssetComment::new(uid(), uid(), "Walt", "Issue A");
        c1.resolve(uid());
        session.add_comment(c1);
        session.add_comment(AssetComment::new(uid(), uid(), "Xena", "Issue B"));
        // Reply should not count as open thread
        let reply = AssetComment::new(uid(), uid(), "Yara", "Agreed").as_reply(uid());
        session.add_comment(reply);

        assert_eq!(session.open_comment_count(), 1);
    }

    #[test]
    fn test_session_summary() {
        let aid = uid();
        let mut session = CollaborationSession::new(aid);
        session.add_comment(AssetComment::new(uid(), uid(), "Zack", "Note"));
        session.add_annotation(Annotation::new(
            uid(),
            uid(),
            "Amy",
            AnnotationShape::Point { x: 0.0, y: 0.0 },
            0,
        ));
        session.add_marker(ReviewMarker::new(
            uid(),
            uid(),
            "Ben",
            "Blocker!",
            MarkerSeverity::Blocker,
        ));

        let s = session.summary();
        assert_eq!(s.total_comments, 1);
        assert_eq!(s.total_annotations, 1);
        assert_eq!(s.total_markers, 1);
        assert_eq!(s.blocking_markers, 1);
    }

    #[test]
    fn test_severity_label() {
        assert_eq!(MarkerSeverity::Info.label(), "info");
        assert_eq!(MarkerSeverity::Minor.label(), "minor");
        assert_eq!(MarkerSeverity::Major.label(), "major");
        assert_eq!(MarkerSeverity::Blocker.label(), "blocker");
    }

    #[test]
    fn test_severity_is_blocking() {
        assert!(!MarkerSeverity::Info.is_blocking());
        assert!(!MarkerSeverity::Minor.is_blocking());
        assert!(!MarkerSeverity::Major.is_blocking());
        assert!(MarkerSeverity::Blocker.is_blocking());
    }

    #[test]
    fn test_marker_status_is_open() {
        assert!(MarkerStatus::Open.is_open());
        assert!(MarkerStatus::InProgress.is_open());
        assert!(!MarkerStatus::Resolved.is_open());
        assert!(!MarkerStatus::Dismissed.is_open());
    }
}
