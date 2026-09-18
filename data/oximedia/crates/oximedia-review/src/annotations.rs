//! Media annotation system for review workflows.
//!
//! Provides frame-accurate annotations with type, authorship, and resolution tracking.

/// Type of annotation attached to a media item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AnnotationType {
    /// A general note about the media.
    Note,
    /// A question requiring a response.
    Question,
    /// Approval of the current state.
    Approval,
    /// Rejection with a required explanation.
    Rejection,
    /// A correction to be applied.
    Correction,
    /// A highlighted region of interest.
    Highlight,
}

impl AnnotationType {
    /// Returns `true` if this annotation type requires a response from the recipient.
    #[must_use]
    pub fn requires_response(&self) -> bool {
        matches!(self, Self::Question | Self::Rejection | Self::Correction)
    }

    /// Returns a CSS-style hex colour string representing this annotation type.
    #[must_use]
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Note => "#4A90D9",
            Self::Question => "#F5A623",
            Self::Approval => "#7ED321",
            Self::Rejection => "#D0021B",
            Self::Correction => "#9B59B6",
            Self::Highlight => "#F8E71C",
        }
    }
}

/// A single annotation attached to a media item at a specific timestamp.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Annotation {
    /// Unique identifier for this annotation.
    pub id: u64,
    /// Identifier of the media item this annotation belongs to.
    pub media_id: String,
    /// Position in the media (milliseconds from the beginning).
    pub timestamp_ms: u64,
    /// Duration the annotation spans in milliseconds (0 = point annotation).
    pub duration_ms: u32,
    /// Semantic type of this annotation.
    pub annotation_type: AnnotationType,
    /// Human-readable text content.
    pub text: String,
    /// Author who created this annotation.
    pub author: String,
    /// Whether this annotation has been marked as resolved.
    pub resolved: bool,
}

impl Annotation {
    /// Mark this annotation as resolved.
    pub fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Returns `true` if this annotation has exceeded its time-to-live.
    ///
    /// # Arguments
    ///
    /// * `now_ms` - Current time in milliseconds since epoch.
    /// * `ttl_ms` - Maximum allowed age in milliseconds.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms.saturating_sub(self.timestamp_ms) > ttl_ms
    }
}

/// Filter criteria for querying annotations from an `AnnotationStore`.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct AnnotationFilter {
    /// If non-empty, only annotations with one of these types match.
    pub types: Vec<AnnotationType>,
    /// If `Some`, only annotations by this author match.
    pub author: Option<String>,
    /// If `Some(true)`, only resolved annotations match; `Some(false)` = unresolved only.
    pub resolved: Option<bool>,
}

impl AnnotationFilter {
    /// Returns `true` if `ann` satisfies every criterion in this filter.
    #[must_use]
    pub fn matches(&self, ann: &Annotation) -> bool {
        if !self.types.is_empty() && !self.types.contains(&ann.annotation_type) {
            return false;
        }
        if let Some(ref author) = self.author {
            if &ann.author != author {
                return false;
            }
        }
        if let Some(resolved) = self.resolved {
            if ann.resolved != resolved {
                return false;
            }
        }
        true
    }
}

/// In-memory store for managing annotations.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct AnnotationStore {
    /// All annotations in the store.
    pub annotations: Vec<Annotation>,
    /// Next identifier to assign.
    pub next_id: u64,
}

impl AnnotationStore {
    /// Create an empty `AnnotationStore`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            annotations: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an annotation to the store, assigning it a unique ID.
    ///
    /// Returns the assigned ID.
    pub fn add(&mut self, mut ann: Annotation) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        ann.id = id;
        self.annotations.push(ann);
        id
    }

    /// Retrieve an annotation by its ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Annotation> {
        self.annotations.iter().find(|a| a.id == id)
    }

    /// Mark the annotation with the given ID as resolved.
    ///
    /// Returns `true` if the annotation was found and updated.
    pub fn resolve(&mut self, id: u64) -> bool {
        if let Some(ann) = self.annotations.iter_mut().find(|a| a.id == id) {
            ann.resolve();
            true
        } else {
            false
        }
    }

    /// Return all annotations that match the given filter.
    #[must_use]
    pub fn filter(&self, f: &AnnotationFilter) -> Vec<&Annotation> {
        self.annotations.iter().filter(|a| f.matches(a)).collect()
    }

    /// Return the number of annotations that have not yet been resolved.
    #[must_use]
    pub fn unresolved_count(&self) -> usize {
        self.annotations.iter().filter(|a| !a.resolved).count()
    }
}

// ---------------------------------------------------------------------------
// Helpers for tests
// ---------------------------------------------------------------------------

#[cfg(test)]
fn make_annotation(id: u64, ann_type: AnnotationType, author: &str, resolved: bool) -> Annotation {
    Annotation {
        id,
        media_id: "media-001".to_string(),
        timestamp_ms: 1_000,
        duration_ms: 0,
        annotation_type: ann_type,
        text: "Test annotation".to_string(),
        author: author.to_string(),
        resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AnnotationType ---

    #[test]
    fn test_requires_response_question() {
        assert!(AnnotationType::Question.requires_response());
    }

    #[test]
    fn test_requires_response_rejection() {
        assert!(AnnotationType::Rejection.requires_response());
    }

    #[test]
    fn test_requires_response_correction() {
        assert!(AnnotationType::Correction.requires_response());
    }

    #[test]
    fn test_no_response_required_for_note() {
        assert!(!AnnotationType::Note.requires_response());
    }

    #[test]
    fn test_no_response_required_for_approval() {
        assert!(!AnnotationType::Approval.requires_response());
    }

    #[test]
    fn test_no_response_required_for_highlight() {
        assert!(!AnnotationType::Highlight.requires_response());
    }

    #[test]
    fn test_color_hex_is_valid_css() {
        for t in [
            AnnotationType::Note,
            AnnotationType::Question,
            AnnotationType::Approval,
            AnnotationType::Rejection,
            AnnotationType::Correction,
            AnnotationType::Highlight,
        ] {
            let hex = t.color_hex();
            assert!(hex.starts_with('#'), "expected '#' prefix: {hex}");
            assert_eq!(hex.len(), 7, "expected 7-char hex: {hex}");
        }
    }

    // --- Annotation ---

    #[test]
    fn test_annotation_resolve() {
        let mut ann = make_annotation(1, AnnotationType::Note, "alice", false);
        assert!(!ann.resolved);
        ann.resolve();
        assert!(ann.resolved);
    }

    #[test]
    fn test_annotation_is_expired_when_past_ttl() {
        let ann = make_annotation(2, AnnotationType::Note, "bob", false);
        // timestamp_ms = 1000; now = 2000; ttl = 500 -> 1000 > 500 -> expired
        assert!(ann.is_expired(2_000, 500));
    }

    #[test]
    fn test_annotation_not_expired_within_ttl() {
        let ann = make_annotation(3, AnnotationType::Note, "carol", false);
        // timestamp_ms = 1000; now = 1200; ttl = 5000 -> 200 <= 5000 -> not expired
        assert!(!ann.is_expired(1_200, 5_000));
    }

    // --- AnnotationFilter ---

    #[test]
    fn test_filter_by_type_match() {
        let ann = make_annotation(4, AnnotationType::Question, "dave", false);
        let f = AnnotationFilter {
            types: vec![AnnotationType::Question],
            ..Default::default()
        };
        assert!(f.matches(&ann));
    }

    #[test]
    fn test_filter_by_type_no_match() {
        let ann = make_annotation(5, AnnotationType::Note, "eve", false);
        let f = AnnotationFilter {
            types: vec![AnnotationType::Approval],
            ..Default::default()
        };
        assert!(!f.matches(&ann));
    }

    #[test]
    fn test_filter_by_author_match() {
        let ann = make_annotation(6, AnnotationType::Note, "frank", false);
        let f = AnnotationFilter {
            author: Some("frank".to_string()),
            ..Default::default()
        };
        assert!(f.matches(&ann));
    }

    #[test]
    fn test_filter_by_author_no_match() {
        let ann = make_annotation(7, AnnotationType::Note, "grace", false);
        let f = AnnotationFilter {
            author: Some("heidi".to_string()),
            ..Default::default()
        };
        assert!(!f.matches(&ann));
    }

    // --- AnnotationStore ---

    #[test]
    fn test_store_add_assigns_id() {
        let mut store = AnnotationStore::new();
        let ann = make_annotation(0, AnnotationType::Note, "ivan", false);
        let id = store.add(ann);
        assert_eq!(id, 1);
        assert_eq!(store.get(1).expect("should succeed in test").id, 1);
    }

    #[test]
    fn test_store_get_nonexistent_returns_none() {
        let store = AnnotationStore::new();
        assert!(store.get(999).is_none());
    }

    #[test]
    fn test_store_resolve_existing() {
        let mut store = AnnotationStore::new();
        store.add(make_annotation(0, AnnotationType::Note, "judy", false));
        assert!(store.resolve(1));
        assert!(store.get(1).expect("should succeed in test").resolved);
    }

    #[test]
    fn test_store_resolve_nonexistent_returns_false() {
        let mut store = AnnotationStore::new();
        assert!(!store.resolve(999));
    }

    #[test]
    fn test_store_unresolved_count() {
        let mut store = AnnotationStore::new();
        store.add(make_annotation(0, AnnotationType::Note, "kate", false));
        store.add(make_annotation(0, AnnotationType::Note, "kate", true));
        store.add(make_annotation(0, AnnotationType::Note, "kate", false));
        assert_eq!(store.unresolved_count(), 2);
    }

    #[test]
    fn test_store_filter_returns_matching() {
        let mut store = AnnotationStore::new();
        store.add(make_annotation(0, AnnotationType::Question, "leo", false));
        store.add(make_annotation(0, AnnotationType::Note, "leo", false));
        let f = AnnotationFilter {
            types: vec![AnnotationType::Question],
            ..Default::default()
        };
        let results = store.filter(&f);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].annotation_type, AnnotationType::Question);
    }

    #[test]
    fn test_store_filter_resolved_only() {
        let mut store = AnnotationStore::new();
        store.add(make_annotation(0, AnnotationType::Note, "mia", false));
        store.add(make_annotation(0, AnnotationType::Note, "mia", true));
        let f = AnnotationFilter {
            resolved: Some(true),
            ..Default::default()
        };
        let results = store.filter(&f);
        assert_eq!(results.len(), 1);
        assert!(results[0].resolved);
    }
}
