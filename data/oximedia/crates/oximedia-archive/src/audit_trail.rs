#![allow(dead_code)]
//! Archive audit trail tracking
//!
//! Provides comprehensive event logging for all archive operations including
//! ingestion, access, modification, deletion, and verification events. Tracks
//! who did what, when, and provides tamper-evident chain-of-custody records.

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for an audit event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AuditEventId(String);

impl AuditEventId {
    /// Create a new audit event ID from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the inner string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AuditEventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The category of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditCategory {
    /// Asset ingestion into the archive.
    Ingest,
    /// Read access to an archived asset.
    Access,
    /// Modification of an archived asset or its metadata.
    Modify,
    /// Deletion or purge of an archived asset.
    Delete,
    /// Verification / fixity check of an archived asset.
    Verify,
    /// Restore operation from archive.
    Restore,
    /// Migration between storage tiers or formats.
    Migrate,
    /// Administrative operation (config change, user management).
    Admin,
}

impl fmt::Display for AuditCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Ingest => "INGEST",
            Self::Access => "ACCESS",
            Self::Modify => "MODIFY",
            Self::Delete => "DELETE",
            Self::Verify => "VERIFY",
            Self::Restore => "RESTORE",
            Self::Migrate => "MIGRATE",
            Self::Admin => "ADMIN",
        };
        write!(f, "{s}")
    }
}

/// Outcome of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditOutcome {
    /// The operation succeeded.
    Success,
    /// The operation failed.
    Failure,
    /// The operation was denied (authorization).
    Denied,
    /// The operation is still in progress.
    Pending,
}

impl fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Success => "SUCCESS",
            Self::Failure => "FAILURE",
            Self::Denied => "DENIED",
            Self::Pending => "PENDING",
        };
        write!(f, "{s}")
    }
}

/// A single audit event entry.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Unique event identifier.
    pub id: AuditEventId,
    /// Timestamp in epoch milliseconds.
    pub timestamp_ms: u64,
    /// Category of the event.
    pub category: AuditCategory,
    /// Outcome of the event.
    pub outcome: AuditOutcome,
    /// The user or system actor that triggered the event.
    pub actor: String,
    /// The asset path or identifier acted upon.
    pub target: String,
    /// Human-readable description.
    pub description: String,
    /// Optional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl AuditEvent {
    /// Create a new audit event.
    pub fn new(
        id: AuditEventId,
        timestamp_ms: u64,
        category: AuditCategory,
        outcome: AuditOutcome,
        actor: impl Into<String>,
        target: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            timestamp_ms,
            category,
            outcome,
            actor: actor.into(),
            target: target.into(),
            description: description.into(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a key-value metadata pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Computes a simple hash chain digest for tamper evidence.
///
/// Each event digest includes the previous digest so any tampering is detectable.
#[allow(clippy::cast_precision_loss)]
fn chain_digest(previous: u64, event: &AuditEvent) -> u64 {
    let mut h = previous;
    for b in event.id.as_str().bytes() {
        h = h.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    h = h.wrapping_mul(31).wrapping_add(event.timestamp_ms);
    for b in event.actor.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    for b in event.target.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    h
}

/// In-memory audit trail ledger with tamper-evident hash chaining.
#[derive(Debug)]
pub struct AuditTrail {
    /// All recorded events.
    events: Vec<AuditEvent>,
    /// Running chain digest for tamper detection.
    chain_hash: u64,
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditTrail {
    /// Create a new empty audit trail.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            chain_hash: 0,
        }
    }

    /// Append an event to the trail and update the chain hash.
    pub fn record(&mut self, event: AuditEvent) {
        self.chain_hash = chain_digest(self.chain_hash, &event);
        self.events.push(event);
    }

    /// Return the number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the current chain hash for tamper checking.
    pub fn chain_hash(&self) -> u64 {
        self.chain_hash
    }

    /// Get a reference to all events.
    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    /// Filter events by category.
    pub fn filter_by_category(&self, category: AuditCategory) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Filter events by outcome.
    pub fn filter_by_outcome(&self, outcome: AuditOutcome) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.outcome == outcome)
            .collect()
    }

    /// Filter events by actor.
    pub fn filter_by_actor(&self, actor: &str) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.actor == actor).collect()
    }

    /// Filter events by target asset.
    pub fn filter_by_target(&self, target: &str) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.target == target).collect()
    }

    /// Filter events within a time range (inclusive).
    pub fn filter_by_time_range(&self, start_ms: u64, end_ms: u64) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Verify the chain hash by replaying all events.
    pub fn verify_chain(&self) -> bool {
        let mut h: u64 = 0;
        for event in &self.events {
            h = chain_digest(h, event);
        }
        h == self.chain_hash
    }

    /// Generate a summary count per category.
    pub fn summary_by_category(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for event in &self.events {
            *map.entry(event.category.to_string()).or_insert(0) += 1;
        }
        map
    }

    /// Clear the trail (for testing or rotation).
    pub fn clear(&mut self) {
        self.events.clear();
        self.chain_hash = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event(id: &str, cat: AuditCategory, outcome: AuditOutcome) -> AuditEvent {
        AuditEvent::new(
            AuditEventId::new(id),
            1_000_000 + id.len() as u64,
            cat,
            outcome,
            "test_user",
            "/archive/video.mxf",
            format!("Test event {id}"),
        )
    }

    #[test]
    fn test_new_trail_is_empty() {
        let trail = AuditTrail::new();
        assert!(trail.is_empty());
        assert_eq!(trail.len(), 0);
        assert_eq!(trail.chain_hash(), 0);
    }

    #[test]
    fn test_record_event() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        assert_eq!(trail.len(), 1);
        assert!(!trail.is_empty());
        assert_ne!(trail.chain_hash(), 0);
    }

    #[test]
    fn test_chain_hash_deterministic() {
        let mut t1 = AuditTrail::new();
        let mut t2 = AuditTrail::new();
        let e = sample_event("e1", AuditCategory::Access, AuditOutcome::Success);
        t1.record(e.clone());
        t2.record(e);
        assert_eq!(t1.chain_hash(), t2.chain_hash());
    }

    #[test]
    fn test_chain_verification_ok() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e2",
            AuditCategory::Access,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e3",
            AuditCategory::Delete,
            AuditOutcome::Denied,
        ));
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_filter_by_category() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e2",
            AuditCategory::Access,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e3",
            AuditCategory::Ingest,
            AuditOutcome::Failure,
        ));
        let ingests = trail.filter_by_category(AuditCategory::Ingest);
        assert_eq!(ingests.len(), 2);
    }

    #[test]
    fn test_filter_by_outcome() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e2",
            AuditCategory::Access,
            AuditOutcome::Failure,
        ));
        trail.record(sample_event(
            "e3",
            AuditCategory::Delete,
            AuditOutcome::Success,
        ));
        let failures = trail.filter_by_outcome(AuditOutcome::Failure);
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn test_filter_by_actor() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        let results = trail.filter_by_actor("test_user");
        assert_eq!(results.len(), 1);
        let empty = trail.filter_by_actor("nobody");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_filter_by_target() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        let results = trail.filter_by_target("/archive/video.mxf");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_by_time_range() {
        let mut trail = AuditTrail::new();
        // timestamps: 1_000_002, 1_000_002, 1_000_002 (id lengths are all 2)
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e2",
            AuditCategory::Access,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e3",
            AuditCategory::Delete,
            AuditOutcome::Success,
        ));
        let all = trail.filter_by_time_range(0, u64::MAX);
        assert_eq!(all.len(), 3);
        let none = trail.filter_by_time_range(0, 100);
        assert!(none.is_empty());
    }

    #[test]
    fn test_summary_by_category() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.record(sample_event(
            "e2",
            AuditCategory::Ingest,
            AuditOutcome::Failure,
        ));
        trail.record(sample_event(
            "e3",
            AuditCategory::Delete,
            AuditOutcome::Success,
        ));
        let summary = trail.summary_by_category();
        assert_eq!(summary.get("INGEST"), Some(&2));
        assert_eq!(summary.get("DELETE"), Some(&1));
    }

    #[test]
    fn test_clear_trail() {
        let mut trail = AuditTrail::new();
        trail.record(sample_event(
            "e1",
            AuditCategory::Ingest,
            AuditOutcome::Success,
        ));
        trail.clear();
        assert!(trail.is_empty());
        assert_eq!(trail.chain_hash(), 0);
    }

    #[test]
    fn test_metadata_on_event() {
        let event = sample_event("e1", AuditCategory::Ingest, AuditOutcome::Success)
            .with_metadata("checksum", "abc123")
            .with_metadata("size_bytes", "1048576");
        assert_eq!(event.metadata.get("checksum"), Some(&"abc123".to_string()));
        assert_eq!(
            event.metadata.get("size_bytes"),
            Some(&"1048576".to_string())
        );
    }

    #[test]
    fn test_event_id_display() {
        let id = AuditEventId::new("audit-001");
        assert_eq!(id.to_string(), "audit-001");
        assert_eq!(id.as_str(), "audit-001");
    }

    #[test]
    fn test_category_display() {
        assert_eq!(AuditCategory::Ingest.to_string(), "INGEST");
        assert_eq!(AuditCategory::Access.to_string(), "ACCESS");
        assert_eq!(AuditCategory::Modify.to_string(), "MODIFY");
        assert_eq!(AuditCategory::Migrate.to_string(), "MIGRATE");
        assert_eq!(AuditCategory::Admin.to_string(), "ADMIN");
    }
}
