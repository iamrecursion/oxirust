//! Paginated listing of comments for sessions with 1000+ comments.
//!
//! All listing functions are synchronous and operate on in-memory slices.
//! Storage-backed implementations should convert their own data to `&[Comment]`
//! (or use the cursor helpers) before calling into this module.

#![allow(dead_code)]

use crate::comment::{Comment, CommentPriority, CommentStatus};
use crate::AnnotationType;

// ── Pagination parameters ─────────────────────────────────────────────────────

/// Parameters for a single paginated comment query.
#[derive(Debug, Clone)]
pub struct PageRequest {
    /// Zero-based page index.
    pub page: usize,
    /// Maximum number of comments per page (must be ≥ 1).
    pub page_size: usize,
    /// Optional filter — only include comments matching this status.
    pub status_filter: Option<CommentStatus>,
    /// Optional filter — only include comments of this annotation type.
    pub type_filter: Option<AnnotationType>,
    /// Optional filter — only include comments at or above this priority.
    pub min_priority: Option<CommentPriority>,
    /// Optional frame range filter `[start, end]` (inclusive).
    pub frame_range: Option<(i64, i64)>,
    /// Sort order.
    pub sort: CommentSortOrder,
}

impl PageRequest {
    /// Create a page request with sensible defaults.
    ///
    /// * `page` — zero-based page index.
    /// * `page_size` — clamped to the range `[1, 10_000]`.
    #[must_use]
    pub fn new(page: usize, page_size: usize) -> Self {
        Self {
            page,
            page_size: page_size.max(1).min(10_000),
            status_filter: None,
            type_filter: None,
            min_priority: None,
            frame_range: None,
            sort: CommentSortOrder::ByFrameAsc,
        }
    }

    /// Filter by status.
    #[must_use]
    pub fn with_status(mut self, status: CommentStatus) -> Self {
        self.status_filter = Some(status);
        self
    }

    /// Filter by annotation type.
    #[must_use]
    pub fn with_type(mut self, ann_type: AnnotationType) -> Self {
        self.type_filter = Some(ann_type);
        self
    }

    /// Filter by minimum priority (inclusive).
    #[must_use]
    pub fn with_min_priority(mut self, priority: CommentPriority) -> Self {
        self.min_priority = Some(priority);
        self
    }

    /// Filter by frame range.
    #[must_use]
    pub fn with_frame_range(mut self, start: i64, end: i64) -> Self {
        self.frame_range = Some((start.min(end), start.max(end)));
        self
    }

    /// Set sort order.
    #[must_use]
    pub fn with_sort(mut self, sort: CommentSortOrder) -> Self {
        self.sort = sort;
        self
    }
}

/// Sort order for paginated comment listings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentSortOrder {
    /// Oldest first (by `created_at`).
    ByCreatedAsc,
    /// Newest first (by `created_at`).
    ByCreatedDesc,
    /// Frame number ascending.
    ByFrameAsc,
    /// Frame number descending.
    ByFrameDesc,
    /// Priority descending (Critical first).
    ByPriorityDesc,
}

// ── Paginated response ─────────────────────────────────────────────────────────

/// The result of a paginated comment query.
#[derive(Debug, Clone)]
pub struct CommentPage {
    /// The comments on this page.
    pub items: Vec<Comment>,
    /// Current page index (zero-based).
    pub page: usize,
    /// Page size used for this request.
    pub page_size: usize,
    /// Total number of comments matching the filter (across all pages).
    pub total_count: usize,
    /// Total number of pages.
    pub total_pages: usize,
    /// Whether there is a next page.
    pub has_next: bool,
    /// Whether there is a previous page.
    pub has_prev: bool,
}

impl CommentPage {
    /// Returns `true` when this is the first page.
    #[must_use]
    pub fn is_first(&self) -> bool {
        self.page == 0
    }

    /// Returns `true` when this is the last page.
    #[must_use]
    pub fn is_last(&self) -> bool {
        !self.has_next
    }

    /// Number of items on this page.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

// ── Pagination engine ─────────────────────────────────────────────────────────

/// Apply a [`PageRequest`] to a slice of [`Comment`]s and return a [`CommentPage`].
///
/// This function performs filtering, sorting, and slicing in one pass.
/// For very large in-memory datasets (>100 k comments) consider pre-sorting.
#[must_use]
pub fn paginate_comments(comments: &[Comment], req: &PageRequest) -> CommentPage {
    // 1. Filter
    let mut filtered: Vec<&Comment> = comments
        .iter()
        .filter(|c| {
            if let Some(s) = req.status_filter {
                if c.status != s {
                    return false;
                }
            }
            if let Some(t) = req.type_filter {
                if c.annotation_type != t {
                    return false;
                }
            }
            if let Some(p) = req.min_priority {
                if c.priority < p {
                    return false;
                }
            }
            if let Some((start, end)) = req.frame_range {
                if c.frame < start || c.frame > end {
                    return false;
                }
            }
            true
        })
        .collect();

    // 2. Sort
    match req.sort {
        CommentSortOrder::ByCreatedAsc => filtered.sort_by_key(|c| c.created_at),
        CommentSortOrder::ByCreatedDesc => filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
        CommentSortOrder::ByFrameAsc => filtered.sort_by_key(|c| c.frame),
        CommentSortOrder::ByFrameDesc => filtered.sort_by(|a, b| b.frame.cmp(&a.frame)),
        CommentSortOrder::ByPriorityDesc => filtered.sort_by(|a, b| b.priority.cmp(&a.priority)),
    }

    // 3. Pagination arithmetic
    let total_count = filtered.len();
    let page_size = req.page_size;
    let total_pages = if total_count == 0 {
        1
    } else {
        (total_count + page_size - 1) / page_size
    };
    let page = req.page.min(total_pages.saturating_sub(1));
    let start = page * page_size;
    let end = (start + page_size).min(total_count);

    let items: Vec<Comment> = filtered[start..end].iter().map(|&c| c.clone()).collect();
    let has_next = end < total_count;
    let has_prev = page > 0;

    CommentPage {
        items,
        page,
        page_size,
        total_count,
        total_pages,
        has_next,
        has_prev,
    }
}

/// Cursor-based pagination: return comments after the given offset.
///
/// This is a lightweight alternative to offset/limit pagination that is
/// efficient for large datasets when comments are appended only (append-only
/// log pattern).
#[must_use]
pub fn cursor_page(comments: &[Comment], after_index: usize, limit: usize) -> CommentPage {
    let req = PageRequest::new(0, limit.max(1));
    let total_count = comments.len().saturating_sub(after_index);
    let start = after_index.min(comments.len());
    let end = (start + limit).min(comments.len());
    let items = comments[start..end].to_vec();
    let has_next = end < comments.len();

    CommentPage {
        items,
        page: 0,
        page_size: req.page_size,
        total_count,
        total_pages: if total_count == 0 {
            1
        } else {
            (total_count + limit - 1) / limit
        },
        has_next,
        has_prev: after_index > 0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnnotationType, CommentId, SessionId, User, UserRole};
    use chrono::Utc;

    fn make_comment(frame: i64, priority: CommentPriority, ann: AnnotationType) -> Comment {
        Comment {
            id: CommentId::new(),
            session_id: SessionId::new(),
            frame,
            text: format!("comment at frame {frame}"),
            annotation_type: ann,
            author: User {
                id: "tester".into(),
                name: "Tester".into(),
                email: "test@test.com".into(),
                role: UserRole::Reviewer,
            },
            status: CommentStatus::Open,
            priority,
            parent_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            resolved_at: None,
            resolved_by: None,
        }
    }

    fn make_resolved_comment(frame: i64) -> Comment {
        let mut c = make_comment(frame, CommentPriority::Normal, AnnotationType::Issue);
        c.status = CommentStatus::Resolved;
        c
    }

    fn sample_set(n: usize) -> Vec<Comment> {
        (0..n as i64)
            .map(|i| make_comment(i, CommentPriority::Normal, AnnotationType::Issue))
            .collect()
    }

    // 1 — basic pagination
    #[test]
    fn test_paginate_basic() {
        let comments = sample_set(100);
        let req = PageRequest::new(0, 10);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.item_count(), 10);
        assert_eq!(page.total_count, 100);
        assert_eq!(page.total_pages, 10);
        assert!(page.has_next);
        assert!(!page.has_prev);
    }

    // 2 — last page
    #[test]
    fn test_paginate_last_page() {
        let comments = sample_set(25);
        let req = PageRequest::new(2, 10);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.item_count(), 5);
        assert!(!page.has_next);
        assert!(page.has_prev);
        assert!(page.is_last());
    }

    // 3 — empty result
    #[test]
    fn test_paginate_empty() {
        let page = paginate_comments(&[], &PageRequest::new(0, 10));
        assert_eq!(page.total_count, 0);
        assert_eq!(page.item_count(), 0);
        assert!(!page.has_next);
        assert!(!page.has_prev);
    }

    // 4 — status filter
    #[test]
    fn test_paginate_status_filter() {
        let mut comments = sample_set(10);
        comments[0].status = CommentStatus::Resolved;
        comments[1].status = CommentStatus::Resolved;
        let req = PageRequest::new(0, 20).with_status(CommentStatus::Resolved);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.total_count, 2);
    }

    // 5 — type filter
    #[test]
    fn test_paginate_type_filter() {
        let mut comments = sample_set(5);
        comments[0].annotation_type = AnnotationType::Suggestion;
        let req = PageRequest::new(0, 20).with_type(AnnotationType::Suggestion);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.total_count, 1);
    }

    // 6 — priority filter
    #[test]
    fn test_paginate_priority_filter() {
        let mut comments = sample_set(5);
        comments[0].priority = CommentPriority::Critical;
        comments[1].priority = CommentPriority::High;
        let req = PageRequest::new(0, 20).with_min_priority(CommentPriority::High);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.total_count, 2);
    }

    // 7 — frame range filter
    #[test]
    fn test_paginate_frame_range() {
        let comments = sample_set(20); // frames 0..19
        let req = PageRequest::new(0, 20).with_frame_range(5, 10);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.total_count, 6); // frames 5,6,7,8,9,10
    }

    // 8 — sort by frame desc
    #[test]
    fn test_paginate_sort_frame_desc() {
        let comments = sample_set(5);
        let req = PageRequest::new(0, 10).with_sort(CommentSortOrder::ByFrameDesc);
        let page = paginate_comments(&comments, &req);
        let frames: Vec<i64> = page.items.iter().map(|c| c.frame).collect();
        assert_eq!(frames, vec![4, 3, 2, 1, 0]);
    }

    // 9 — sort by priority desc
    #[test]
    fn test_paginate_sort_priority_desc() {
        let comments = vec![
            make_comment(0, CommentPriority::Low, AnnotationType::Issue),
            make_comment(1, CommentPriority::Critical, AnnotationType::Issue),
            make_comment(2, CommentPriority::Normal, AnnotationType::Issue),
        ];
        // Ensure deterministic ordering for the test
        let req = PageRequest::new(0, 10).with_sort(CommentSortOrder::ByPriorityDesc);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.items[0].priority, CommentPriority::Critical);
    }

    // 10 — out-of-range page is clamped
    #[test]
    fn test_paginate_page_clamped() {
        let comments = sample_set(5);
        let req = PageRequest::new(999, 10);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.item_count(), 5); // clamped to first (and only) page
    }

    // 11 — page_size of 1
    #[test]
    fn test_paginate_page_size_one() {
        let comments = sample_set(3);
        let req = PageRequest::new(1, 1);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.item_count(), 1);
        assert_eq!(page.total_pages, 3);
        assert!(page.has_next);
        assert!(page.has_prev);
    }

    // 12 — is_first and is_last
    #[test]
    fn test_paginate_is_first_last() {
        let comments = sample_set(1);
        let req = PageRequest::new(0, 10);
        let page = paginate_comments(&comments, &req);
        assert!(page.is_first());
        assert!(page.is_last());
    }

    // 13 — cursor_page basic
    #[test]
    fn test_cursor_page_basic() {
        let comments = sample_set(20);
        let page = cursor_page(&comments, 5, 5);
        assert_eq!(page.item_count(), 5);
        assert_eq!(page.items[0].frame, 5);
        assert!(page.has_next);
        assert!(page.has_prev);
    }

    // 14 — cursor_page at end
    #[test]
    fn test_cursor_page_end() {
        let comments = sample_set(10);
        let page = cursor_page(&comments, 8, 5);
        assert_eq!(page.item_count(), 2);
        assert!(!page.has_next);
    }

    // 15 — cursor_page past end returns empty
    #[test]
    fn test_cursor_page_past_end() {
        let comments = sample_set(5);
        let page = cursor_page(&comments, 10, 5);
        assert_eq!(page.item_count(), 0);
    }

    // 16 — 1000+ comments pagination
    #[test]
    fn test_paginate_large_dataset() {
        let comments = sample_set(1500);
        let req = PageRequest::new(14, 100);
        let page = paginate_comments(&comments, &req);
        assert_eq!(page.total_count, 1500);
        assert_eq!(page.total_pages, 15);
        assert_eq!(page.item_count(), 100);
        assert!(!page.has_next); // last page
    }
}
