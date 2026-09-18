#![allow(dead_code)]
//! Server audit trail logging for security and compliance.
//!
//! Records immutable, append-only audit events for user actions,
//! authentication changes, data access, and configuration modifications.
//! Supports filtering, retention policies, and summary reporting.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Category of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditCategory {
    /// User authentication (login, logout, token refresh).
    Authentication,
    /// Authorisation decisions (access granted / denied).
    Authorization,
    /// Data access (read / download).
    DataAccess,
    /// Data modification (create / update / delete).
    DataModification,
    /// Configuration changes.
    ConfigChange,
    /// Administrative actions (user management, role changes).
    Admin,
    /// System events (startup, shutdown, errors).
    System,
}

impl AuditCategory {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Authentication => "Authentication",
            Self::Authorization => "Authorization",
            Self::DataAccess => "Data Access",
            Self::DataModification => "Data Modification",
            Self::ConfigChange => "Config Change",
            Self::Admin => "Admin",
            Self::System => "System",
        }
    }

    /// Returns `true` for security-sensitive categories.
    pub fn is_security_sensitive(&self) -> bool {
        matches!(
            self,
            Self::Authentication | Self::Authorization | Self::Admin
        )
    }
}

/// Severity of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AuditSeverity {
    /// Informational – routine operations.
    Info,
    /// Warning – unusual but non-critical.
    Warning,
    /// Error – operation failed.
    Error,
    /// Critical – security incident.
    Critical,
}

impl AuditSeverity {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

/// A single audit event.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Timestamp of the event.
    pub timestamp: SystemTime,
    /// Category.
    pub category: AuditCategory,
    /// Severity.
    pub severity: AuditSeverity,
    /// User or actor identifier (empty for system events).
    pub actor: String,
    /// Action performed (e.g. "login", "delete_media").
    pub action: String,
    /// Resource affected (e.g. "media/1234").
    pub resource: String,
    /// Outcome description.
    pub outcome: String,
    /// Source IP address (if applicable).
    pub source_ip: String,
}

/// Builder for `AuditEvent`.
pub struct AuditEventBuilder {
    /// Category.
    category: AuditCategory,
    /// Severity.
    severity: AuditSeverity,
    /// Actor.
    actor: String,
    /// Action.
    action: String,
    /// Resource.
    resource: String,
    /// Outcome.
    outcome: String,
    /// Source IP.
    source_ip: String,
}

impl AuditEventBuilder {
    /// Creates a new builder with required fields.
    pub fn new(category: AuditCategory, action: impl Into<String>) -> Self {
        Self {
            category,
            severity: AuditSeverity::Info,
            actor: String::new(),
            action: action.into(),
            resource: String::new(),
            outcome: String::new(),
            source_ip: String::new(),
        }
    }

    /// Sets the severity.
    #[must_use]
    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Sets the actor.
    #[must_use]
    pub fn actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = actor.into();
        self
    }

    /// Sets the resource.
    #[must_use]
    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = resource.into();
        self
    }

    /// Sets the outcome.
    #[must_use]
    pub fn outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = outcome.into();
        self
    }

    /// Sets the source IP.
    #[must_use]
    pub fn source_ip(mut self, ip: impl Into<String>) -> Self {
        self.source_ip = ip.into();
        self
    }

    /// Builds the event (sequence number and timestamp are assigned by the trail).
    fn build(self, seq: u64) -> AuditEvent {
        AuditEvent {
            seq,
            timestamp: SystemTime::now(),
            category: self.category,
            severity: self.severity,
            actor: self.actor,
            action: self.action,
            resource: self.resource,
            outcome: self.outcome,
            source_ip: self.source_ip,
        }
    }
}

/// Configuration for the audit trail.
#[derive(Debug, Clone)]
pub struct AuditTrailConfig {
    /// Maximum events to retain in memory.
    pub max_events: usize,
    /// Minimum severity to record.
    pub min_severity: AuditSeverity,
    /// Retention period – events older than this may be purged.
    pub retention: Duration,
}

impl Default for AuditTrailConfig {
    fn default() -> Self {
        Self {
            max_events: 10_000,
            min_severity: AuditSeverity::Info,
            retention: Duration::from_secs(90 * 24 * 3600), // 90 days
        }
    }
}

/// Summary statistics for the audit trail.
#[derive(Debug, Clone)]
pub struct AuditSummary {
    /// Total events recorded.
    pub total_events: u64,
    /// Events currently in memory.
    pub current_events: usize,
    /// Number of critical events.
    pub critical_count: u64,
    /// Number of error events.
    pub error_count: u64,
    /// Number of warning events.
    pub warning_count: u64,
}

impl AuditSummary {
    /// Creates a zeroed summary.
    pub fn new() -> Self {
        Self {
            total_events: 0,
            current_events: 0,
            critical_count: 0,
            error_count: 0,
            warning_count: 0,
        }
    }
}

impl Default for AuditSummary {
    fn default() -> Self {
        Self::new()
    }
}

/// The audit trail – an append-only ring-buffer of audit events.
pub struct AuditTrail {
    /// Configuration.
    config: AuditTrailConfig,
    /// Events in order.
    events: VecDeque<AuditEvent>,
    /// Next sequence number.
    next_seq: u64,
    /// Running summary counts.
    summary: AuditSummary,
}

impl AuditTrail {
    /// Creates a new audit trail.
    pub fn new(config: AuditTrailConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            next_seq: 1,
            summary: AuditSummary::new(),
        }
    }

    /// Records an event from a builder.
    pub fn record(&mut self, builder: AuditEventBuilder) {
        if builder.severity < self.config.min_severity {
            return;
        }
        let event = builder.build(self.next_seq);
        self.next_seq += 1;

        match event.severity {
            AuditSeverity::Critical => self.summary.critical_count += 1,
            AuditSeverity::Error => self.summary.error_count += 1,
            AuditSeverity::Warning => self.summary.warning_count += 1,
            AuditSeverity::Info => {}
        }

        self.events.push_back(event);
        self.summary.total_events += 1;

        // Trim if over capacity
        while self.events.len() > self.config.max_events {
            self.events.pop_front();
        }

        self.summary.current_events = self.events.len();
    }

    /// Returns the most recent `n` events.
    pub fn recent(&self, n: usize) -> Vec<&AuditEvent> {
        self.events.iter().rev().take(n).collect()
    }

    /// Filters events by category.
    pub fn by_category(&self, category: AuditCategory) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Filters events by actor.
    pub fn by_actor(&self, actor: &str) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.actor == actor).collect()
    }

    /// Filters events by severity (at or above).
    pub fn by_min_severity(&self, min: AuditSeverity) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.severity >= min).collect()
    }

    /// Returns events matching a given action string.
    pub fn by_action(&self, action: &str) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.action == action).collect()
    }

    /// Returns the summary statistics.
    pub fn summary(&self) -> &AuditSummary {
        &self.summary
    }

    /// Number of events currently stored.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if no events are stored.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Purges events older than the retention period.
    pub fn purge_old(&mut self) -> usize {
        let cutoff = SystemTime::now()
            .checked_sub(self.config.retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let before = self.events.len();
        self.events.retain(|e| e.timestamp >= cutoff);
        let removed = before - self.events.len();
        self.summary.current_events = self.events.len();
        removed
    }

    /// Clears all events.
    pub fn clear(&mut self) {
        self.events.clear();
        self.summary.current_events = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trail() -> AuditTrail {
        AuditTrail::new(AuditTrailConfig::default())
    }

    #[test]
    fn test_record_and_len() {
        let mut t = trail();
        t.record(AuditEventBuilder::new(AuditCategory::Authentication, "login").actor("alice"));
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_recent() {
        let mut t = trail();
        for i in 0..5 {
            t.record(
                AuditEventBuilder::new(AuditCategory::DataAccess, format!("action_{i}"))
                    .actor("bob"),
            );
        }
        let recent = t.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].action, "action_4");
    }

    #[test]
    fn test_by_category() {
        let mut t = trail();
        t.record(AuditEventBuilder::new(
            AuditCategory::Authentication,
            "login",
        ));
        t.record(AuditEventBuilder::new(AuditCategory::DataAccess, "read"));
        t.record(AuditEventBuilder::new(
            AuditCategory::Authentication,
            "logout",
        ));
        let auth = t.by_category(AuditCategory::Authentication);
        assert_eq!(auth.len(), 2);
    }

    #[test]
    fn test_by_actor() {
        let mut t = trail();
        t.record(AuditEventBuilder::new(AuditCategory::Admin, "create_user").actor("admin"));
        t.record(AuditEventBuilder::new(AuditCategory::DataAccess, "read").actor("viewer"));
        let admin_events = t.by_actor("admin");
        assert_eq!(admin_events.len(), 1);
    }

    #[test]
    fn test_by_min_severity() {
        let mut t = trail();
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "startup").severity(AuditSeverity::Info),
        );
        t.record(
            AuditEventBuilder::new(AuditCategory::Authentication, "brute_force")
                .severity(AuditSeverity::Critical),
        );
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "disk_low")
                .severity(AuditSeverity::Warning),
        );
        let critical = t.by_min_severity(AuditSeverity::Critical);
        assert_eq!(critical.len(), 1);
        let warn_up = t.by_min_severity(AuditSeverity::Warning);
        assert_eq!(warn_up.len(), 2);
    }

    #[test]
    fn test_summary_counts() {
        let mut t = trail();
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "err").severity(AuditSeverity::Error),
        );
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "crit").severity(AuditSeverity::Critical),
        );
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "warn").severity(AuditSeverity::Warning),
        );
        assert_eq!(t.summary().error_count, 1);
        assert_eq!(t.summary().critical_count, 1);
        assert_eq!(t.summary().warning_count, 1);
        assert_eq!(t.summary().total_events, 3);
    }

    #[test]
    fn test_capacity_trimming() {
        let mut t = AuditTrail::new(AuditTrailConfig {
            max_events: 3,
            ..Default::default()
        });
        for i in 0..5 {
            t.record(AuditEventBuilder::new(
                AuditCategory::DataAccess,
                format!("a{i}"),
            ));
        }
        assert_eq!(t.len(), 3);
        // Oldest events should be dropped
        let all: Vec<_> = t.recent(3);
        assert_eq!(all[2].action, "a2");
    }

    #[test]
    fn test_min_severity_filter() {
        let mut t = AuditTrail::new(AuditTrailConfig {
            min_severity: AuditSeverity::Warning,
            ..Default::default()
        });
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "info").severity(AuditSeverity::Info),
        );
        t.record(
            AuditEventBuilder::new(AuditCategory::System, "warn").severity(AuditSeverity::Warning),
        );
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut t = trail();
        t.record(AuditEventBuilder::new(AuditCategory::System, "x"));
        t.clear();
        assert!(t.is_empty());
    }

    #[test]
    fn test_category_labels() {
        assert_eq!(AuditCategory::Authentication.label(), "Authentication");
        assert_eq!(AuditCategory::System.label(), "System");
    }

    #[test]
    fn test_category_security_sensitive() {
        assert!(AuditCategory::Authentication.is_security_sensitive());
        assert!(AuditCategory::Admin.is_security_sensitive());
        assert!(!AuditCategory::DataAccess.is_security_sensitive());
        assert!(!AuditCategory::System.is_security_sensitive());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AuditSeverity::Critical > AuditSeverity::Error);
        assert!(AuditSeverity::Error > AuditSeverity::Warning);
        assert!(AuditSeverity::Warning > AuditSeverity::Info);
    }

    #[test]
    fn test_by_action() {
        let mut t = trail();
        t.record(AuditEventBuilder::new(AuditCategory::Authentication, "login").actor("alice"));
        t.record(AuditEventBuilder::new(AuditCategory::Authentication, "logout").actor("alice"));
        t.record(AuditEventBuilder::new(AuditCategory::Authentication, "login").actor("bob"));
        let logins = t.by_action("login");
        assert_eq!(logins.len(), 2);
    }

    #[test]
    fn test_event_builder_full() {
        let mut t = trail();
        t.record(
            AuditEventBuilder::new(AuditCategory::DataModification, "delete_media")
                .severity(AuditSeverity::Warning)
                .actor("admin")
                .resource("media/1234")
                .outcome("success")
                .source_ip("192.168.1.10"),
        );
        let ev = &t.recent(1)[0];
        assert_eq!(ev.actor, "admin");
        assert_eq!(ev.resource, "media/1234");
        assert_eq!(ev.source_ip, "192.168.1.10");
    }
}
