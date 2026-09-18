//! Comment threading: status tracking, resolution, and unresolved count.

#![allow(dead_code)]

/// Lifecycle status of a single review comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentStatus {
    /// Comment is open and awaiting response or resolution.
    Open,
    /// Comment has been acknowledged but not yet actioned.
    Acknowledged,
    /// Comment has been resolved — no further action required.
    Resolved,
    /// Comment has been marked won't-fix — deliberately not actioned.
    WontFix,
    /// Comment has been dismissed as not applicable.
    Dismissed,
    /// Comment is blocked on an external dependency.
    Blocked,
}

// ── Thread-level resolution state ─────────────────────────────────────────

/// High-level resolution state of an entire `CommentThread`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadResolutionState {
    /// Thread is open: at least one comment needs action.
    Open,
    /// Thread has been resolved — all comments actioned.
    Resolved,
    /// Thread was explicitly marked won't-fix at the thread level.
    WontFix,
}

impl ThreadResolutionState {
    /// Returns `true` for closed states (Resolved or WontFix).
    #[must_use]
    pub fn is_closed(self) -> bool {
        matches!(self, Self::Resolved | Self::WontFix)
    }
}

/// A single entry in the thread audit trail.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Wall-clock timestamp in milliseconds since UNIX epoch.
    pub timestamp_ms: u64,
    /// Identifier of the user who performed the action.
    pub actor: String,
    /// What happened.
    pub action: AuditAction,
}

/// The type of action recorded in an `AuditEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    /// A new comment was added (carries the comment ID).
    CommentAdded(u64),
    /// A comment's status was changed.
    CommentStatusChanged {
        /// Comment that was changed.
        comment_id: u64,
        /// Previous status.
        from: CommentStatus,
        /// New status.
        to: CommentStatus,
    },
    /// The thread-level resolution state was changed.
    ThreadStateChanged {
        /// Previous thread state.
        from: ThreadResolutionState,
        /// New thread state.
        to: ThreadResolutionState,
        /// Optional free-text reason.
        reason: Option<String>,
    },
}

impl CommentStatus {
    /// Returns `true` if this status represents a resolved/closed comment.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved | Self::Dismissed | Self::WontFix)
    }

    /// Returns `true` if this status requires attention.
    #[must_use]
    pub fn needs_action(&self) -> bool {
        matches!(self, Self::Open | Self::Blocked)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Acknowledged => "Acknowledged",
            Self::Resolved => "Resolved",
            Self::WontFix => "Won't Fix",
            Self::Dismissed => "Dismissed",
            Self::Blocked => "Blocked",
        }
    }
}

/// A single comment in a review thread.
#[derive(Debug, Clone)]
pub struct ReviewComment {
    /// Unique comment identifier.
    pub id: u64,
    /// Author identifier.
    pub author_id: String,
    /// Text body of the comment.
    pub body: String,
    /// Current lifecycle status.
    pub status: CommentStatus,
    /// Optional reference to a parent comment (for replies).
    pub parent_id: Option<u64>,
    /// Optional frame/timecode reference (milliseconds).
    pub timecode_ms: Option<u64>,
}

impl ReviewComment {
    /// Create a new top-level comment.
    #[must_use]
    pub fn new(id: u64, author_id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id,
            author_id: author_id.into(),
            body: body.into(),
            status: CommentStatus::Open,
            parent_id: None,
            timecode_ms: None,
        }
    }

    /// Create a reply to an existing comment.
    #[must_use]
    pub fn reply(
        id: u64,
        author_id: impl Into<String>,
        body: impl Into<String>,
        parent_id: u64,
    ) -> Self {
        Self {
            id,
            author_id: author_id.into(),
            body: body.into(),
            status: CommentStatus::Open,
            parent_id: Some(parent_id),
            timecode_ms: None,
        }
    }

    /// Attach a timecode reference (milliseconds from start).
    #[must_use]
    pub fn with_timecode(mut self, ms: u64) -> Self {
        self.timecode_ms = Some(ms);
        self
    }

    /// Returns `true` if this comment has been resolved or dismissed.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.status.is_resolved()
    }

    /// Mark this comment as resolved.
    pub fn resolve(&mut self) {
        self.status = CommentStatus::Resolved;
    }

    /// Mark this comment as won't-fix (deliberately not actioned).
    pub fn mark_wont_fix(&mut self) {
        self.status = CommentStatus::WontFix;
    }

    /// Dismiss the comment as not applicable.
    pub fn dismiss(&mut self) {
        self.status = CommentStatus::Dismissed;
    }

    /// Returns `true` if this is a top-level comment (not a reply).
    #[must_use]
    pub fn is_top_level(&self) -> bool {
        self.parent_id.is_none()
    }
}

/// A thread of related comments attached to a review session.
#[derive(Debug, Clone)]
pub struct CommentThread {
    /// Thread identifier.
    pub id: u64,
    /// Short subject/title for the thread.
    pub subject: String,
    /// Ordered list of comments in this thread.
    comments: Vec<ReviewComment>,
    /// High-level resolution state of the whole thread.
    pub resolution_state: ThreadResolutionState,
    /// Chronological audit trail of all state changes.
    audit_trail: Vec<AuditEntry>,
}

impl CommentThread {
    /// Create a new empty comment thread.
    #[must_use]
    pub fn new(id: u64, subject: impl Into<String>) -> Self {
        Self {
            id,
            subject: subject.into(),
            comments: Vec::new(),
            resolution_state: ThreadResolutionState::Open,
            audit_trail: Vec::new(),
        }
    }

    /// Add a comment to the thread, recording an audit entry.
    pub fn add_comment_audited(&mut self, comment: ReviewComment, actor: &str, now_ms: u64) {
        let comment_id = comment.id;
        self.comments.push(comment);
        self.audit_trail.push(AuditEntry {
            timestamp_ms: now_ms,
            actor: actor.to_string(),
            action: AuditAction::CommentAdded(comment_id),
        });
    }

    /// Add a comment to the thread (no audit entry — kept for backward compat).
    pub fn add_comment(&mut self, comment: ReviewComment) {
        self.comments.push(comment);
    }

    /// Change a single comment's status and record an audit entry.
    ///
    /// Returns `false` if no comment with `id` was found.
    pub fn set_comment_status(
        &mut self,
        id: u64,
        new_status: CommentStatus,
        actor: &str,
        now_ms: u64,
    ) -> bool {
        if let Some(c) = self.comments.iter_mut().find(|c| c.id == id) {
            let old = c.status;
            c.status = new_status;
            self.audit_trail.push(AuditEntry {
                timestamp_ms: now_ms,
                actor: actor.to_string(),
                action: AuditAction::CommentStatusChanged {
                    comment_id: id,
                    from: old,
                    to: new_status,
                },
            });
            true
        } else {
            false
        }
    }

    /// Mark the thread as resolved, recording an audit entry.
    pub fn mark_resolved(&mut self, actor: &str, reason: Option<String>, now_ms: u64) {
        let old = self.resolution_state;
        self.resolution_state = ThreadResolutionState::Resolved;
        // Also resolve all open comments.
        for c in &mut self.comments {
            if !c.is_resolved() {
                c.status = CommentStatus::Resolved;
            }
        }
        self.audit_trail.push(AuditEntry {
            timestamp_ms: now_ms,
            actor: actor.to_string(),
            action: AuditAction::ThreadStateChanged {
                from: old,
                to: ThreadResolutionState::Resolved,
                reason,
            },
        });
    }

    /// Mark the thread as won't-fix, recording an audit entry.
    pub fn mark_wont_fix(&mut self, actor: &str, reason: Option<String>, now_ms: u64) {
        let old = self.resolution_state;
        self.resolution_state = ThreadResolutionState::WontFix;
        for c in &mut self.comments {
            if !c.is_resolved() {
                c.status = CommentStatus::WontFix;
            }
        }
        self.audit_trail.push(AuditEntry {
            timestamp_ms: now_ms,
            actor: actor.to_string(),
            action: AuditAction::ThreadStateChanged {
                from: old,
                to: ThreadResolutionState::WontFix,
                reason,
            },
        });
    }

    /// Re-open the thread (e.g., after incorrect resolution).
    pub fn reopen(&mut self, actor: &str, reason: Option<String>, now_ms: u64) {
        let old = self.resolution_state;
        self.resolution_state = ThreadResolutionState::Open;
        self.audit_trail.push(AuditEntry {
            timestamp_ms: now_ms,
            actor: actor.to_string(),
            action: AuditAction::ThreadStateChanged {
                from: old,
                to: ThreadResolutionState::Open,
                reason,
            },
        });
    }

    /// Return the full audit trail (oldest first).
    #[must_use]
    pub fn audit_trail(&self) -> &[AuditEntry] {
        &self.audit_trail
    }

    /// Resolve all comments in the thread (no audit — kept for backward compat).
    pub fn resolve(&mut self) {
        for c in &mut self.comments {
            c.resolve();
        }
    }

    /// Return a reference to all comments in the thread.
    #[must_use]
    pub fn comments(&self) -> &[ReviewComment] {
        &self.comments
    }

    /// Count of comments that are still unresolved.
    #[must_use]
    pub fn unresolved_count(&self) -> usize {
        self.comments.iter().filter(|c| !c.is_resolved()).count()
    }

    /// Total comment count.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.comments.len()
    }

    /// Returns `true` if all comments in this thread are resolved.
    #[must_use]
    pub fn is_fully_resolved(&self) -> bool {
        !self.comments.is_empty() && self.unresolved_count() == 0
    }

    /// Find a comment by ID (returns a mutable reference).
    pub fn find_mut(&mut self, id: u64) -> Option<&mut ReviewComment> {
        self.comments.iter_mut().find(|c| c.id == id)
    }
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn thread() -> CommentThread {
        CommentThread::new(1, "Color Grading Issues")
    }

    fn open_comment(id: u64) -> ReviewComment {
        ReviewComment::new(id, "user-42", "Please fix the highlights.")
    }

    // 1 — CommentStatus::is_resolved
    #[test]
    fn test_status_is_resolved() {
        assert!(CommentStatus::Resolved.is_resolved());
        assert!(CommentStatus::Dismissed.is_resolved());
        assert!(!CommentStatus::Open.is_resolved());
        assert!(!CommentStatus::Acknowledged.is_resolved());
        assert!(!CommentStatus::Blocked.is_resolved());
    }

    // 2 — CommentStatus::needs_action
    #[test]
    fn test_status_needs_action() {
        assert!(CommentStatus::Open.needs_action());
        assert!(CommentStatus::Blocked.needs_action());
        assert!(!CommentStatus::Resolved.needs_action());
        assert!(!CommentStatus::Dismissed.needs_action());
        assert!(!CommentStatus::Acknowledged.needs_action());
    }

    // 3 — CommentStatus::label
    #[test]
    fn test_status_label() {
        assert_eq!(CommentStatus::Open.label(), "Open");
        assert_eq!(CommentStatus::Resolved.label(), "Resolved");
        assert_eq!(CommentStatus::Dismissed.label(), "Dismissed");
    }

    // 4 — ReviewComment initial state
    #[test]
    fn test_new_comment_is_open() {
        let c = open_comment(1);
        assert_eq!(c.status, CommentStatus::Open);
        assert!(!c.is_resolved());
        assert!(c.is_top_level());
    }

    // 5 — ReviewComment::resolve sets resolved
    #[test]
    fn test_comment_resolve() {
        let mut c = open_comment(1);
        c.resolve();
        assert!(c.is_resolved());
        assert_eq!(c.status, CommentStatus::Resolved);
    }

    // 6 — ReviewComment::dismiss sets dismissed
    #[test]
    fn test_comment_dismiss() {
        let mut c = open_comment(1);
        c.dismiss();
        assert!(c.is_resolved());
        assert_eq!(c.status, CommentStatus::Dismissed);
    }

    // 7 — Reply has parent_id set
    #[test]
    fn test_reply_parent_id() {
        let reply = ReviewComment::reply(2, "user-7", "Agreed!", 1);
        assert_eq!(reply.parent_id, Some(1));
        assert!(!reply.is_top_level());
    }

    // 8 — with_timecode attaches ms
    #[test]
    fn test_with_timecode() {
        let c = open_comment(1).with_timecode(5000);
        assert_eq!(c.timecode_ms, Some(5000));
    }

    // 9 — thread::add_comment and total_count
    #[test]
    fn test_thread_add_and_count() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        t.add_comment(open_comment(2));
        assert_eq!(t.total_count(), 2);
    }

    // 10 — unresolved_count with mixed states
    #[test]
    fn test_unresolved_count() {
        let mut t = thread();
        let mut resolved = open_comment(1);
        resolved.resolve();
        t.add_comment(resolved);
        t.add_comment(open_comment(2));
        assert_eq!(t.unresolved_count(), 1);
    }

    // 11 — thread::resolve resolves all
    #[test]
    fn test_thread_resolve_all() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        t.add_comment(open_comment(2));
        t.resolve();
        assert_eq!(t.unresolved_count(), 0);
        assert!(t.is_fully_resolved());
    }

    // 12 — is_fully_resolved is false when thread is empty
    #[test]
    fn test_empty_thread_not_fully_resolved() {
        let t = thread();
        assert!(!t.is_fully_resolved());
    }

    // 13 — find_mut allows targeted resolution
    #[test]
    fn test_find_mut_resolve_single() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        t.add_comment(open_comment(2));
        if let Some(c) = t.find_mut(1) {
            c.resolve();
        }
        assert_eq!(t.unresolved_count(), 1);
    }

    // 14 — find_mut returns None for missing id
    #[test]
    fn test_find_mut_missing() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        assert!(t.find_mut(99).is_none());
    }

    // 15 — comments() returns slice in order
    #[test]
    fn test_comments_order() {
        let mut t = thread();
        t.add_comment(open_comment(10));
        t.add_comment(open_comment(20));
        let ids: Vec<u64> = t.comments().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![10, 20]);
    }

    // ── Resolution tracking & audit trail ────────────────────────────────────

    #[test]
    fn test_wont_fix_status_is_resolved() {
        assert!(CommentStatus::WontFix.is_resolved());
        assert_eq!(CommentStatus::WontFix.label(), "Won't Fix");
    }

    #[test]
    fn test_thread_default_state_is_open() {
        let t = thread();
        assert_eq!(t.resolution_state, ThreadResolutionState::Open);
    }

    #[test]
    fn test_mark_resolved_sets_state_and_audits() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        t.mark_resolved("alice", Some("All done".into()), 5000);
        assert_eq!(t.resolution_state, ThreadResolutionState::Resolved);
        assert!(t.is_fully_resolved());
        let trail = t.audit_trail();
        assert_eq!(trail.len(), 1);
        assert_eq!(trail[0].actor, "alice");
        assert!(matches!(
            trail[0].action,
            AuditAction::ThreadStateChanged {
                to: ThreadResolutionState::Resolved,
                ..
            }
        ));
    }

    #[test]
    fn test_mark_wont_fix_sets_state_and_audits() {
        let mut t = thread();
        t.add_comment(open_comment(1));
        t.mark_wont_fix("bob", Some("By design".into()), 6000);
        assert_eq!(t.resolution_state, ThreadResolutionState::WontFix);
        assert!(t.is_fully_resolved());
        let trail = t.audit_trail();
        assert!(matches!(
            trail[0].action,
            AuditAction::ThreadStateChanged {
                to: ThreadResolutionState::WontFix,
                ..
            }
        ));
    }

    #[test]
    fn test_reopen_thread() {
        let mut t = thread();
        t.mark_resolved("alice", None, 1000);
        t.reopen("bob", Some("Needs more work".into()), 2000);
        assert_eq!(t.resolution_state, ThreadResolutionState::Open);
        assert_eq!(t.audit_trail().len(), 2);
    }

    #[test]
    fn test_thread_state_is_closed() {
        assert!(ThreadResolutionState::Resolved.is_closed());
        assert!(ThreadResolutionState::WontFix.is_closed());
        assert!(!ThreadResolutionState::Open.is_closed());
    }

    #[test]
    fn test_add_comment_audited() {
        let mut t = thread();
        t.add_comment_audited(open_comment(1), "alice", 100);
        assert_eq!(t.total_count(), 1);
        assert_eq!(t.audit_trail().len(), 1);
        assert!(matches!(
            t.audit_trail()[0].action,
            AuditAction::CommentAdded(1)
        ));
    }

    #[test]
    fn test_set_comment_status_audits() {
        let mut t = thread();
        t.add_comment(open_comment(42));
        let changed = t.set_comment_status(42, CommentStatus::Resolved, "carol", 9000);
        assert!(changed);
        assert_eq!(t.unresolved_count(), 0);
        let trail = t.audit_trail();
        assert_eq!(trail.len(), 1);
        assert!(matches!(
            trail[0].action,
            AuditAction::CommentStatusChanged {
                comment_id: 42,
                from: CommentStatus::Open,
                to: CommentStatus::Resolved,
            }
        ));
    }

    #[test]
    fn test_set_comment_status_returns_false_for_missing() {
        let mut t = thread();
        assert!(!t.set_comment_status(999, CommentStatus::Resolved, "alice", 0));
    }

    #[test]
    fn test_comment_mark_wont_fix() {
        let mut c = open_comment(1);
        c.mark_wont_fix();
        assert!(c.is_resolved());
        assert_eq!(c.status, CommentStatus::WontFix);
    }
}
