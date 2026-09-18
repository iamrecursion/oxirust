//! Review report: summarises comments, annotations, and markers for a session.

use crate::{
    comment::{Comment, CommentStatus},
    drawing::annotation::Annotation,
    marker::{MarkerCategory, ReviewMarker},
    SessionId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Per-author statistics within a report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorStats {
    /// Author identifier.
    pub author: String,
    /// Total comments written.
    pub comment_count: usize,
    /// Resolved comments.
    pub resolved_count: usize,
    /// Annotations created.
    pub annotation_count: usize,
    /// Markers placed.
    pub marker_count: usize,
}

/// Summary of all review activity for one session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewReport {
    /// Session this report is for.
    pub session_id: SessionId,
    /// Human-readable title of the session.
    pub session_title: String,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,

    // --- Comment statistics ---
    /// All comments in the session.
    pub comments: Vec<Comment>,
    /// Number of open (unresolved) comments.
    pub open_comment_count: usize,
    /// Number of resolved comments.
    pub resolved_comment_count: usize,

    // --- Annotation statistics ---
    /// All annotations in the session.
    pub annotations: Vec<Annotation>,
    /// Total annotation count.
    pub annotation_count: usize,

    // --- Marker statistics ---
    /// All markers in the session.
    pub markers: Vec<ReviewMarker>,
    /// Markers flagged as issues.
    pub issue_marker_count: usize,
    /// Approval markers.
    pub approval_marker_count: usize,

    // --- Per-author breakdown ---
    /// Per-author activity statistics.
    pub author_stats: Vec<AuthorStats>,
}

impl ReviewReport {
    /// Build a `ReviewReport` from raw review data.
    #[must_use]
    pub fn build(
        session_id: SessionId,
        session_title: impl Into<String>,
        comments: Vec<Comment>,
        annotations: Vec<Annotation>,
        markers: Vec<ReviewMarker>,
    ) -> Self {
        let open_comment_count = comments
            .iter()
            .filter(|c| c.status == CommentStatus::Open)
            .count();
        let resolved_comment_count = comments
            .iter()
            .filter(|c| c.status == CommentStatus::Resolved)
            .count();

        let annotation_count = annotations.len();

        let issue_marker_count = markers
            .iter()
            .filter(|m| m.category == MarkerCategory::Issue)
            .count();
        let approval_marker_count = markers
            .iter()
            .filter(|m| m.category == MarkerCategory::Approval)
            .count();

        // Build per-author stats using a map
        let mut author_map: std::collections::HashMap<String, AuthorStats> =
            std::collections::HashMap::new();

        for comment in &comments {
            let entry = author_map
                .entry(comment.author.id.clone())
                .or_insert_with(|| AuthorStats {
                    author: comment.author.name.clone(),
                    comment_count: 0,
                    resolved_count: 0,
                    annotation_count: 0,
                    marker_count: 0,
                });
            entry.comment_count += 1;
            if comment.status == CommentStatus::Resolved {
                entry.resolved_count += 1;
            }
        }

        for annotation in &annotations {
            let entry = author_map
                .entry(annotation.drawing.author.clone())
                .or_insert_with(|| AuthorStats {
                    author: annotation.drawing.author.clone(),
                    comment_count: 0,
                    resolved_count: 0,
                    annotation_count: 0,
                    marker_count: 0,
                });
            entry.annotation_count += 1;
        }

        for marker in &markers {
            let entry = author_map
                .entry(marker.author.clone())
                .or_insert_with(|| AuthorStats {
                    author: marker.author.clone(),
                    comment_count: 0,
                    resolved_count: 0,
                    annotation_count: 0,
                    marker_count: 0,
                });
            entry.marker_count += 1;
        }

        let mut author_stats: Vec<AuthorStats> = author_map.into_values().collect();
        author_stats.sort_by(|a, b| a.author.cmp(&b.author));

        Self {
            session_id,
            session_title: session_title.into(),
            generated_at: Utc::now(),
            comments,
            open_comment_count,
            resolved_comment_count,
            annotations,
            annotation_count,
            markers,
            issue_marker_count,
            approval_marker_count,
            author_stats,
        }
    }

    /// Return the total number of comments.
    #[must_use]
    pub fn total_comments(&self) -> usize {
        self.comments.len()
    }

    /// Return the resolution rate as a fraction (0.0–1.0).
    #[must_use]
    pub fn resolution_rate(&self) -> f64 {
        if self.comments.is_empty() {
            return 1.0;
        }
        self.resolved_comment_count as f64 / self.comments.len() as f64
    }

    /// Returns true when all comments are resolved.
    #[must_use]
    pub fn is_fully_resolved(&self) -> bool {
        self.open_comment_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        comment::{CommentPriority, CommentStatus},
        drawing::{
            annotation::Annotation,
            color::{Color, StrokeStyle},
            tools::{DrawingTool, Shape},
            Circle, Drawing, Point,
        },
        marker::MarkerCategory,
        AnnotationType, CommentId, DrawingId, SessionId, User, UserRole,
    };

    fn make_comment(author_id: &str, status: CommentStatus) -> Comment {
        Comment {
            id: CommentId::new(),
            session_id: SessionId::new(),
            frame: 1,
            text: "test".to_string(),
            annotation_type: AnnotationType::General,
            author: User {
                id: author_id.to_string(),
                name: author_id.to_string(),
                email: format!("{author_id}@test.com"),
                role: UserRole::Reviewer,
            },
            status,
            priority: CommentPriority::Normal,
            parent_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            resolved_at: None,
            resolved_by: None,
        }
    }

    fn make_annotation(author: &str) -> Annotation {
        let drawing = Drawing {
            id: DrawingId::new(),
            session_id: SessionId::new(),
            frame: 10,
            tool: DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.1)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: author.to_string(),
        };
        Annotation::new(drawing)
    }

    fn make_marker(author: &str, category: MarkerCategory) -> ReviewMarker {
        ReviewMarker::new(100, Color::red(), "label", category, author)
    }

    #[test]
    fn test_report_empty() {
        let sid = SessionId::new();
        let report = ReviewReport::build(sid, "Empty Session", vec![], vec![], vec![]);
        assert_eq!(report.total_comments(), 0);
        assert_eq!(report.open_comment_count, 0);
        assert_eq!(report.resolved_comment_count, 0);
        assert!(report.is_fully_resolved());
        assert!((report.resolution_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_comment_counts() {
        let sid = SessionId::new();
        let comments = vec![
            make_comment("alice", CommentStatus::Open),
            make_comment("alice", CommentStatus::Resolved),
            make_comment("bob", CommentStatus::Open),
        ];
        let report = ReviewReport::build(sid, "Session", comments, vec![], vec![]);
        assert_eq!(report.total_comments(), 3);
        assert_eq!(report.open_comment_count, 2);
        assert_eq!(report.resolved_comment_count, 1);
        assert!(!report.is_fully_resolved());
    }

    #[test]
    fn test_report_resolution_rate() {
        let sid = SessionId::new();
        let comments = vec![
            make_comment("a", CommentStatus::Resolved),
            make_comment("b", CommentStatus::Resolved),
            make_comment("c", CommentStatus::Open),
            make_comment("d", CommentStatus::Open),
        ];
        let report = ReviewReport::build(sid, "Session", comments, vec![], vec![]);
        let rate = report.resolution_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_annotation_count() {
        let sid = SessionId::new();
        let annotations = vec![make_annotation("alice"), make_annotation("bob")];
        let report = ReviewReport::build(sid, "Session", vec![], annotations, vec![]);
        assert_eq!(report.annotation_count, 2);
    }

    #[test]
    fn test_report_marker_counts() {
        let sid = SessionId::new();
        let markers = vec![
            make_marker("alice", MarkerCategory::Issue),
            make_marker("bob", MarkerCategory::Issue),
            make_marker("alice", MarkerCategory::Approval),
        ];
        let report = ReviewReport::build(sid, "Session", vec![], vec![], markers);
        assert_eq!(report.issue_marker_count, 2);
        assert_eq!(report.approval_marker_count, 1);
    }

    #[test]
    fn test_report_author_stats() {
        let sid = SessionId::new();
        let comments = vec![
            make_comment("alice", CommentStatus::Open),
            make_comment("alice", CommentStatus::Resolved),
            make_comment("bob", CommentStatus::Open),
        ];
        let annotations = vec![make_annotation("alice")];
        let markers = vec![make_marker("bob", MarkerCategory::Info)];

        let report = ReviewReport::build(sid, "Session", comments, annotations, markers);
        // author_stats sorted alphabetically
        assert_eq!(report.author_stats.len(), 2);

        let alice = report
            .author_stats
            .iter()
            .find(|a| a.author == "alice")
            .expect("should succeed in test");
        assert_eq!(alice.comment_count, 2);
        assert_eq!(alice.resolved_count, 1);
        assert_eq!(alice.annotation_count, 1);
        assert_eq!(alice.marker_count, 0);

        let bob = report
            .author_stats
            .iter()
            .find(|a| a.author == "bob")
            .expect("should succeed in test");
        assert_eq!(bob.comment_count, 1);
        assert_eq!(bob.marker_count, 1);
    }

    #[test]
    fn test_report_fully_resolved() {
        let sid = SessionId::new();
        let comments = vec![
            make_comment("a", CommentStatus::Resolved),
            make_comment("b", CommentStatus::Resolved),
        ];
        let report = ReviewReport::build(sid, "Session", comments, vec![], vec![]);
        assert!(report.is_fully_resolved());
        assert!((report.resolution_rate() - 1.0).abs() < f64::EPSILON);
    }
}
