#![allow(dead_code)]
//! Audit trail for proxy workflow operations.
//!
//! This module records every significant action in the proxy lifecycle —
//! creation, verification, deletion, relinking — into an append-only log.
//! Each entry is timestamped and carries structured metadata so that the
//! history of any proxy file can be reconstructed.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Action kinds
// ---------------------------------------------------------------------------

/// The kind of auditable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditAction {
    /// Proxy was generated from original media.
    Created,
    /// Proxy was verified against its checksum or original.
    Verified,
    /// Proxy was re-linked to a different original.
    Relinked,
    /// Proxy was deleted.
    Deleted,
    /// Proxy metadata was updated.
    MetadataUpdated,
    /// Proxy was accessed / downloaded.
    Accessed,
    /// Proxy failed verification.
    VerificationFailed,
    /// Proxy was archived.
    Archived,
}

impl fmt::Display for AuditAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Created => "CREATED",
            Self::Verified => "VERIFIED",
            Self::Relinked => "RELINKED",
            Self::Deleted => "DELETED",
            Self::MetadataUpdated => "METADATA_UPDATED",
            Self::Accessed => "ACCESSED",
            Self::VerificationFailed => "VERIFICATION_FAILED",
            Self::Archived => "ARCHIVED",
        };
        write!(f, "{label}")
    }
}

// ---------------------------------------------------------------------------
// Audit entry
// ---------------------------------------------------------------------------

/// A single audit log entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Timestamp of the action.
    pub timestamp: SystemTime,
    /// The action performed.
    pub action: AuditAction,
    /// Path or identifier of the proxy file.
    pub proxy_path: String,
    /// Optional path of the linked original.
    pub original_path: Option<String>,
    /// Actor who performed the action (e.g. user name or system).
    pub actor: String,
    /// Optional free-form detail message.
    pub detail: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry.
    pub fn new(
        seq: u64,
        action: AuditAction,
        proxy_path: impl Into<String>,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            timestamp: SystemTime::now(),
            action,
            proxy_path: proxy_path.into(),
            original_path: None,
            actor: actor.into(),
            detail: None,
        }
    }

    /// Attach the original file path.
    pub fn with_original(mut self, path: impl Into<String>) -> Self {
        self.original_path = Some(path.into());
        self
    }

    /// Attach a detail message.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{seq}] {action} {path} by {actor}",
            seq = self.seq,
            action = self.action,
            path = self.proxy_path,
            actor = self.actor,
        )
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// Append-only audit log for proxy operations.
#[derive(Debug, Clone)]
pub struct AuditLog {
    /// All entries in chronological order.
    entries: Vec<AuditEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Maximum entries to retain (0 = unlimited).
    max_entries: usize,
}

impl AuditLog {
    /// Create a new empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 1,
            max_entries: 0,
        }
    }

    /// Set maximum entry retention.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Record an action.
    pub fn record(
        &mut self,
        action: AuditAction,
        proxy_path: impl Into<String>,
        actor: impl Into<String>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = AuditEntry::new(seq, action, proxy_path, actor);
        self.entries.push(entry);
        self.enforce_limit();
        seq
    }

    /// Record an action with a detail message.
    pub fn record_with_detail(
        &mut self,
        action: AuditAction,
        proxy_path: impl Into<String>,
        actor: impl Into<String>,
        detail: impl Into<String>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = AuditEntry::new(seq, action, proxy_path, actor).with_detail(detail);
        self.entries.push(entry);
        self.enforce_limit();
        seq
    }

    /// Total number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by sequence number.
    pub fn get(&self, seq: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Return all entries for a given proxy path.
    pub fn history_for(&self, proxy_path: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.proxy_path == proxy_path)
            .collect()
    }

    /// Return entries matching a given action kind.
    pub fn filter_by_action(&self, action: AuditAction) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.action == action).collect()
    }

    /// Return the most recent `n` entries.
    pub fn recent(&self, n: usize) -> &[AuditEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Count entries per action kind.
    pub fn action_counts(&self) -> HashMap<AuditAction, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.action).or_insert(0) += 1;
        }
        counts
    }

    /// Return distinct actors in the log.
    pub fn actors(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for entry in &self.entries {
            if !seen.contains(&entry.actor.as_str()) {
                seen.push(&entry.actor);
            }
        }
        seen
    }

    /// Clear the entire log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    // Internal: trim to max_entries.
    fn enforce_limit(&mut self) {
        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(..excess);
        }
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(1, AuditAction::Created, "proxy.mp4", "system");
        assert_eq!(entry.seq, 1);
        assert_eq!(entry.action, AuditAction::Created);
        assert_eq!(entry.proxy_path, "proxy.mp4");
        assert_eq!(entry.actor, "system");
        assert!(entry.detail.is_none());
    }

    #[test]
    fn test_audit_entry_builder() {
        let entry = AuditEntry::new(1, AuditAction::Relinked, "p.mp4", "user1")
            .with_original("orig.mov")
            .with_detail("re-linked after conform");
        assert_eq!(entry.original_path.as_deref(), Some("orig.mov"));
        assert_eq!(entry.detail.as_deref(), Some("re-linked after conform"));
    }

    #[test]
    fn test_audit_entry_display() {
        let entry = AuditEntry::new(42, AuditAction::Deleted, "p.mp4", "admin");
        let s = format!("{entry}");
        assert!(s.contains("42"));
        assert!(s.contains("DELETED"));
        assert!(s.contains("p.mp4"));
        assert!(s.contains("admin"));
    }

    #[test]
    fn test_action_display() {
        assert_eq!(format!("{}", AuditAction::Created), "CREATED");
        assert_eq!(
            format!("{}", AuditAction::VerificationFailed),
            "VERIFICATION_FAILED"
        );
    }

    #[test]
    fn test_log_record() {
        let mut log = AuditLog::new();
        let s1 = log.record(AuditAction::Created, "a.mp4", "sys");
        let s2 = log.record(AuditAction::Verified, "a.mp4", "sys");
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_get() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "sys");
        assert!(log.get(1).is_some());
        assert!(log.get(99).is_none());
    }

    #[test]
    fn test_log_history_for() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "sys");
        log.record(AuditAction::Accessed, "b.mp4", "sys");
        log.record(AuditAction::Verified, "a.mp4", "sys");
        let hist = log.history_for("a.mp4");
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn test_log_filter_by_action() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "sys");
        log.record(AuditAction::Created, "b.mp4", "sys");
        log.record(AuditAction::Deleted, "a.mp4", "sys");
        assert_eq!(log.filter_by_action(AuditAction::Created).len(), 2);
        assert_eq!(log.filter_by_action(AuditAction::Deleted).len(), 1);
    }

    #[test]
    fn test_log_retention() {
        let mut log = AuditLog::new().with_max_entries(3);
        for i in 0..5 {
            log.record(AuditAction::Accessed, format!("{i}.mp4"), "sys");
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_log_action_counts() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "sys");
        log.record(AuditAction::Created, "b.mp4", "sys");
        log.record(AuditAction::Deleted, "a.mp4", "sys");
        let counts = log.action_counts();
        assert_eq!(counts[&AuditAction::Created], 2);
        assert_eq!(counts[&AuditAction::Deleted], 1);
    }

    #[test]
    fn test_log_actors() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "alice");
        log.record(AuditAction::Verified, "a.mp4", "bob");
        log.record(AuditAction::Deleted, "a.mp4", "alice");
        let actors = log.actors();
        assert_eq!(actors.len(), 2);
        assert!(actors.contains(&"alice"));
        assert!(actors.contains(&"bob"));
    }

    #[test]
    fn test_log_clear() {
        let mut log = AuditLog::new();
        log.record(AuditAction::Created, "a.mp4", "sys");
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_log_default() {
        let log = AuditLog::default();
        assert!(log.is_empty());
    }
}
