//! Audit logging for access events.
//!
//! Provides tamper-evident audit trails for access control events in media production.
//! Events are recorded with timestamps, user identifiers, resources, and actions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// The action performed in an access event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditAction {
    /// A read/view was performed
    Read,
    /// A write/modify was performed
    Write,
    /// A delete was performed
    Delete,
    /// A login attempt
    Login,
    /// A logout
    Logout,
    /// Role assignment
    RoleAssigned,
    /// Role removal
    RoleRemoved,
    /// Permission check (access denied)
    AccessDenied,
    /// Export operation
    Export,
    /// Custom action
    Custom(String),
}

impl AuditAction {
    /// Returns the string representation of the action.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Read => "READ",
            Self::Write => "WRITE",
            Self::Delete => "DELETE",
            Self::Login => "LOGIN",
            Self::Logout => "LOGOUT",
            Self::RoleAssigned => "ROLE_ASSIGNED",
            Self::RoleRemoved => "ROLE_REMOVED",
            Self::AccessDenied => "ACCESS_DENIED",
            Self::Export => "EXPORT",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Outcome of an audited action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure(String),
}

/// A single audit log entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Sequential entry id
    pub id: u64,
    /// Unix timestamp in seconds
    pub timestamp: u64,
    /// User who performed the action
    pub user_id: String,
    /// Resource that was accessed
    pub resource: String,
    /// Action performed
    pub action: AuditAction,
    /// Outcome of the action
    pub outcome: AuditOutcome,
    /// Optional context / metadata
    pub context: Option<String>,
    /// Hash of the previous entry for tamper detection
    pub prev_hash: u64,
    /// Hash of this entry
    pub entry_hash: u64,
}

impl AuditEntry {
    /// Compute the hash of this entry's content (excluding `entry_hash`).
    fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        self.user_id.hash(&mut hasher);
        self.resource.hash(&mut hasher);
        self.action.hash(&mut hasher);
        self.prev_hash.hash(&mut hasher);
        if let Some(ctx) = &self.context {
            ctx.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Verify that this entry's hash matches its content.
    #[must_use]
    pub fn verify(&self) -> bool {
        self.entry_hash == self.compute_hash()
    }
}

/// Audit log that stores entries in a tamper-evident chain.
#[derive(Debug)]
pub struct AuditLog {
    entries: VecDeque<AuditEntry>,
    next_id: u64,
    last_hash: u64,
    max_entries: usize,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl AuditLog {
    /// Create a new audit log with a maximum number of retained entries.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            next_id: 1,
            last_hash: 0,
            max_entries,
        }
    }

    /// Record a new audit event.
    pub fn record(
        &mut self,
        user_id: impl Into<String>,
        resource: impl Into<String>,
        action: AuditAction,
        outcome: AuditOutcome,
        context: Option<String>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut entry = AuditEntry {
            id: self.next_id,
            timestamp,
            user_id: user_id.into(),
            resource: resource.into(),
            action,
            outcome,
            context,
            prev_hash: self.last_hash,
            entry_hash: 0,
        };
        entry.entry_hash = entry.compute_hash();
        self.last_hash = entry.entry_hash;
        self.next_id += 1;

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Return the number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify the integrity of the entire log chain.
    ///
    /// Returns `true` if all entry hashes are valid.
    #[must_use]
    pub fn verify_chain(&self) -> bool {
        for entry in &self.entries {
            if !entry.verify() {
                return false;
            }
        }
        true
    }

    /// Query entries by user id.
    #[must_use]
    pub fn entries_for_user(&self, user_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.user_id == user_id)
            .collect()
    }

    /// Query entries by action type.
    #[must_use]
    pub fn entries_for_action(&self, action: &AuditAction) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.action == action)
            .collect()
    }

    /// Query entries where outcome is failure.
    #[must_use]
    pub fn failed_entries(&self) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.outcome, AuditOutcome::Failure(_)))
            .collect()
    }

    /// Get all entries (newest last).
    #[must_use]
    pub fn all_entries(&self) -> Vec<&AuditEntry> {
        self.entries.iter().collect()
    }

    /// Count successful events for a user.
    #[must_use]
    pub fn success_count_for_user(&self, user_id: &str) -> usize {
        self.entries
            .iter()
            .filter(|e| e.user_id == user_id && e.outcome == AuditOutcome::Success)
            .count()
    }
}

/// Summary statistics for audit log analysis.
#[derive(Debug, Clone)]
pub struct AuditSummary {
    /// Total events recorded
    pub total_events: usize,
    /// Number of successful events
    pub success_count: usize,
    /// Number of failed events
    pub failure_count: usize,
    /// Number of access-denied events
    pub access_denied_count: usize,
}

impl AuditSummary {
    /// Compute summary statistics from an audit log.
    #[must_use]
    pub fn from_log(log: &AuditLog) -> Self {
        let total_events = log.len();
        let success_count = log
            .entries
            .iter()
            .filter(|e| e.outcome == AuditOutcome::Success)
            .count();
        let failure_count = total_events - success_count;
        let access_denied_count = log.entries_for_action(&AuditAction::AccessDenied).len();
        Self {
            total_events,
            success_count,
            failure_count,
            access_denied_count,
        }
    }

    /// Success rate as a fraction (0.0 - 1.0).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_events == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.total_events as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log() -> AuditLog {
        AuditLog::new(100)
    }

    #[test]
    fn test_new_log_is_empty() {
        let log = make_log();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_record_single_entry() {
        let mut log = make_log();
        log.record(
            "alice",
            "/media/file.mp4",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_entry_ids_increment() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        log.record("bob", "/b", AuditAction::Write, AuditOutcome::Success, None);
        let entries = log.all_entries();
        assert_eq!(entries[0].id, 1);
        assert_eq!(entries[1].id, 2);
    }

    #[test]
    fn test_verify_chain_valid() {
        let mut log = make_log();
        for i in 0..5u32 {
            log.record(
                format!("user{i}"),
                "/media",
                AuditAction::Read,
                AuditOutcome::Success,
                None,
            );
        }
        assert!(log.verify_chain());
    }

    #[test]
    fn test_verify_chain_detects_tampering() {
        let mut log = make_log();
        log.record(
            "alice",
            "/media",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        // Tamper with an entry
        if let Some(entry) = log.entries.front_mut() {
            entry.user_id = "hacker".to_string();
        }
        assert!(!log.verify_chain());
    }

    #[test]
    fn test_entries_for_user() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        log.record("bob", "/b", AuditAction::Write, AuditOutcome::Success, None);
        log.record(
            "alice",
            "/c",
            AuditAction::Delete,
            AuditOutcome::Success,
            None,
        );
        let alice_entries = log.entries_for_user("alice");
        assert_eq!(alice_entries.len(), 2);
    }

    #[test]
    fn test_entries_for_action() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Login,
            AuditOutcome::Success,
            None,
        );
        log.record(
            "bob",
            "/b",
            AuditAction::Login,
            AuditOutcome::Failure("bad password".to_string()),
            None,
        );
        log.record(
            "carol",
            "/c",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        let logins = log.entries_for_action(&AuditAction::Login);
        assert_eq!(logins.len(), 2);
    }

    #[test]
    fn test_failed_entries() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Delete,
            AuditOutcome::Success,
            None,
        );
        log.record(
            "bob",
            "/b",
            AuditAction::Write,
            AuditOutcome::Failure("permission denied".to_string()),
            None,
        );
        let failed = log.failed_entries();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].user_id, "bob");
    }

    #[test]
    fn test_max_entries_eviction() {
        let mut log = AuditLog::new(3);
        for i in 0..5u32 {
            log.record(
                format!("u{i}"),
                "/x",
                AuditAction::Read,
                AuditOutcome::Success,
                None,
            );
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_audit_summary_counts() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        log.record(
            "bob",
            "/b",
            AuditAction::Write,
            AuditOutcome::Failure("err".to_string()),
            None,
        );
        log.record(
            "carol",
            "/c",
            AuditAction::AccessDenied,
            AuditOutcome::Failure("denied".to_string()),
            None,
        );
        let summary = AuditSummary::from_log(&log);
        assert_eq!(summary.total_events, 3);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.failure_count, 2);
        assert_eq!(summary.access_denied_count, 1);
    }

    #[test]
    fn test_success_rate() {
        let mut log = make_log();
        log.record("u", "/a", AuditAction::Read, AuditOutcome::Success, None);
        log.record("u", "/b", AuditAction::Read, AuditOutcome::Success, None);
        log.record(
            "u",
            "/c",
            AuditAction::Write,
            AuditOutcome::Failure("e".to_string()),
            None,
        );
        let summary = AuditSummary::from_log(&log);
        let rate = summary.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_empty_log() {
        let log = make_log();
        let summary = AuditSummary::from_log(&log);
        assert_eq!(summary.success_rate(), 0.0);
    }

    #[test]
    fn test_success_count_for_user() {
        let mut log = make_log();
        log.record(
            "alice",
            "/a",
            AuditAction::Read,
            AuditOutcome::Success,
            None,
        );
        log.record(
            "alice",
            "/b",
            AuditAction::Write,
            AuditOutcome::Failure("e".to_string()),
            None,
        );
        log.record(
            "alice",
            "/c",
            AuditAction::Export,
            AuditOutcome::Success,
            None,
        );
        assert_eq!(log.success_count_for_user("alice"), 2);
        assert_eq!(log.success_count_for_user("bob"), 0);
    }

    #[test]
    fn test_audit_action_as_str() {
        assert_eq!(AuditAction::Read.as_str(), "READ");
        assert_eq!(AuditAction::AccessDenied.as_str(), "ACCESS_DENIED");
        assert_eq!(
            AuditAction::Custom("BULK_EXPORT".to_string()).as_str(),
            "BULK_EXPORT"
        );
    }

    #[test]
    fn test_entry_with_context() {
        let mut log = make_log();
        log.record(
            "alice",
            "/media/project.mp4",
            AuditAction::Export,
            AuditOutcome::Success,
            Some("format=H264, preset=4K".to_string()),
        );
        let entries = log.all_entries();
        assert!(entries[0].context.is_some());
        assert!(entries[0].verify());
    }
}
