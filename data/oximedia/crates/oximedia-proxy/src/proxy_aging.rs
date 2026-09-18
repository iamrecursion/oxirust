#![allow(dead_code)]
//! Age-based proxy lifecycle management and expiration.
//!
//! Manages proxy files through their lifecycle from creation to deletion,
//! applying aging policies that automatically expire, archive, or
//! regenerate proxies based on configurable rules.

use std::collections::HashMap;

/// Lifecycle stage of a proxy file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyStage {
    /// Freshly created, actively in use.
    Active,
    /// Not accessed recently but still available.
    Idle,
    /// Marked for archival or cold storage.
    Stale,
    /// Scheduled for deletion.
    Expired,
    /// Archived to cold storage.
    Archived,
    /// Deleted.
    Deleted,
}

impl ProxyStage {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active => "Active",
            Self::Idle => "Idle",
            Self::Stale => "Stale",
            Self::Expired => "Expired",
            Self::Archived => "Archived",
            Self::Deleted => "Deleted",
        }
    }

    /// Whether the proxy is still usable.
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Active | Self::Idle)
    }
}

/// Aging policy configuration.
#[derive(Debug, Clone)]
pub struct AgingPolicy {
    /// Days after last access before becoming idle.
    pub idle_after_days: u64,
    /// Days after last access before becoming stale.
    pub stale_after_days: u64,
    /// Days after last access before expiration.
    pub expire_after_days: u64,
    /// Whether to auto-archive stale proxies.
    pub auto_archive: bool,
    /// Whether to auto-delete expired proxies.
    pub auto_delete: bool,
    /// Minimum size in bytes to apply aging (skip tiny files).
    pub min_size_bytes: u64,
}

impl Default for AgingPolicy {
    fn default() -> Self {
        Self {
            idle_after_days: 7,
            stale_after_days: 30,
            expire_after_days: 90,
            auto_archive: true,
            auto_delete: false,
            min_size_bytes: 1024,
        }
    }
}

impl AgingPolicy {
    /// Create a strict policy for space-constrained environments.
    pub fn strict() -> Self {
        Self {
            idle_after_days: 3,
            stale_after_days: 14,
            expire_after_days: 30,
            auto_archive: true,
            auto_delete: true,
            min_size_bytes: 0,
        }
    }

    /// Create a relaxed policy for archival workflows.
    pub fn relaxed() -> Self {
        Self {
            idle_after_days: 30,
            stale_after_days: 180,
            expire_after_days: 365,
            auto_archive: false,
            auto_delete: false,
            min_size_bytes: 0,
        }
    }
}

/// Metadata for a managed proxy file.
#[derive(Debug, Clone)]
pub struct ProxyRecord {
    /// Proxy file identifier.
    pub id: String,
    /// File path.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Creation timestamp (days since epoch).
    pub created_day: u64,
    /// Last access timestamp (days since epoch).
    pub last_access_day: u64,
    /// Number of times accessed.
    pub access_count: u64,
    /// Current lifecycle stage.
    pub stage: ProxyStage,
}

impl ProxyRecord {
    /// Create a new proxy record.
    pub fn new(id: &str, path: &str, size_bytes: u64, created_day: u64) -> Self {
        Self {
            id: id.to_string(),
            path: path.to_string(),
            size_bytes,
            created_day,
            last_access_day: created_day,
            access_count: 0,
            stage: ProxyStage::Active,
        }
    }

    /// Record an access event.
    pub fn record_access(&mut self, day: u64) {
        self.last_access_day = day;
        self.access_count += 1;
        // Accessing a proxy reactivates it
        if self.stage.is_usable() || self.stage == ProxyStage::Stale {
            self.stage = ProxyStage::Active;
        }
    }

    /// Days since last access relative to the given day.
    pub fn days_since_access(&self, current_day: u64) -> u64 {
        current_day.saturating_sub(self.last_access_day)
    }

    /// Age in days since creation relative to the given day.
    pub fn age_days(&self, current_day: u64) -> u64 {
        current_day.saturating_sub(self.created_day)
    }
}

/// Result of an aging sweep.
#[derive(Debug, Clone)]
pub struct AgingSweepResult {
    /// Number of proxies transitioned to idle.
    pub newly_idle: usize,
    /// Number of proxies transitioned to stale.
    pub newly_stale: usize,
    /// Number of proxies transitioned to expired.
    pub newly_expired: usize,
    /// Number of proxies archived.
    pub archived: usize,
    /// Number of proxies deleted.
    pub deleted: usize,
    /// Total bytes reclaimed.
    pub bytes_reclaimed: u64,
}

impl AgingSweepResult {
    /// Total number of transitions.
    pub fn total_transitions(&self) -> usize {
        self.newly_idle + self.newly_stale + self.newly_expired + self.archived + self.deleted
    }
}

/// Manager for proxy aging lifecycle.
pub struct AgingManager {
    /// Aging policy.
    policy: AgingPolicy,
    /// Managed proxy records.
    records: HashMap<String, ProxyRecord>,
}

impl AgingManager {
    /// Create a new aging manager with the given policy.
    pub fn new(policy: AgingPolicy) -> Self {
        Self {
            policy,
            records: HashMap::new(),
        }
    }

    /// Add a proxy record.
    pub fn add_record(&mut self, record: ProxyRecord) {
        self.records.insert(record.id.clone(), record);
    }

    /// Get a record by ID.
    pub fn get_record(&self, id: &str) -> Option<&ProxyRecord> {
        self.records.get(id)
    }

    /// Record an access event for a proxy.
    pub fn record_access(&mut self, id: &str, day: u64) -> bool {
        if let Some(record) = self.records.get_mut(id) {
            record.record_access(day);
            true
        } else {
            false
        }
    }

    /// Total number of managed records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Total size of all managed proxies in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.records.values().map(|r| r.size_bytes).sum()
    }

    /// Run an aging sweep for the given current day.
    pub fn sweep(&mut self, current_day: u64) -> AgingSweepResult {
        let mut result = AgingSweepResult {
            newly_idle: 0,
            newly_stale: 0,
            newly_expired: 0,
            archived: 0,
            deleted: 0,
            bytes_reclaimed: 0,
        };

        let mut to_delete = Vec::new();

        for record in self.records.values_mut() {
            if record.size_bytes < self.policy.min_size_bytes {
                continue;
            }

            let days_inactive = record.days_since_access(current_day);
            let old_stage = record.stage;

            // Determine new stage based on inactivity
            let new_stage = if days_inactive >= self.policy.expire_after_days {
                ProxyStage::Expired
            } else if days_inactive >= self.policy.stale_after_days {
                ProxyStage::Stale
            } else if days_inactive >= self.policy.idle_after_days {
                ProxyStage::Idle
            } else {
                ProxyStage::Active
            };

            // Only advance stage, never go backwards during sweep
            if new_stage as u8 > old_stage as u8 || old_stage == ProxyStage::Active {
                match new_stage {
                    ProxyStage::Idle if old_stage == ProxyStage::Active => {
                        record.stage = ProxyStage::Idle;
                        result.newly_idle += 1;
                    }
                    ProxyStage::Stale
                        if old_stage == ProxyStage::Active || old_stage == ProxyStage::Idle =>
                    {
                        if self.policy.auto_archive {
                            record.stage = ProxyStage::Archived;
                            result.archived += 1;
                            result.bytes_reclaimed += record.size_bytes;
                        } else {
                            record.stage = ProxyStage::Stale;
                            result.newly_stale += 1;
                        }
                    }
                    ProxyStage::Expired
                        if old_stage != ProxyStage::Expired && old_stage != ProxyStage::Deleted =>
                    {
                        if self.policy.auto_delete {
                            to_delete.push(record.id.clone());
                            result.deleted += 1;
                            result.bytes_reclaimed += record.size_bytes;
                        } else {
                            record.stage = ProxyStage::Expired;
                            result.newly_expired += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        for id in &to_delete {
            self.records.remove(id);
        }

        result
    }

    /// Get all records in a specific stage.
    pub fn records_in_stage(&self, stage: ProxyStage) -> Vec<&ProxyRecord> {
        self.records.values().filter(|r| r.stage == stage).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, size: u64, created_day: u64) -> ProxyRecord {
        ProxyRecord::new(id, &format!("/proxy/{id}.mp4"), size, created_day)
    }

    #[test]
    fn test_proxy_stage_labels() {
        assert_eq!(ProxyStage::Active.label(), "Active");
        assert_eq!(ProxyStage::Expired.label(), "Expired");
        assert_eq!(ProxyStage::Deleted.label(), "Deleted");
    }

    #[test]
    fn test_proxy_stage_usable() {
        assert!(ProxyStage::Active.is_usable());
        assert!(ProxyStage::Idle.is_usable());
        assert!(!ProxyStage::Stale.is_usable());
        assert!(!ProxyStage::Expired.is_usable());
        assert!(!ProxyStage::Archived.is_usable());
    }

    #[test]
    fn test_policy_defaults() {
        let policy = AgingPolicy::default();
        assert_eq!(policy.idle_after_days, 7);
        assert_eq!(policy.stale_after_days, 30);
        assert_eq!(policy.expire_after_days, 90);
    }

    #[test]
    fn test_policy_strict() {
        let policy = AgingPolicy::strict();
        assert!(policy.auto_delete);
        assert!(policy.expire_after_days < AgingPolicy::default().expire_after_days);
    }

    #[test]
    fn test_policy_relaxed() {
        let policy = AgingPolicy::relaxed();
        assert!(!policy.auto_delete);
        assert!(policy.expire_after_days > AgingPolicy::default().expire_after_days);
    }

    #[test]
    fn test_record_access_reactivates() {
        let mut rec = make_record("a", 1000, 0);
        rec.stage = ProxyStage::Stale;
        rec.record_access(50);
        assert_eq!(rec.stage, ProxyStage::Active);
        assert_eq!(rec.last_access_day, 50);
        assert_eq!(rec.access_count, 1);
    }

    #[test]
    fn test_record_days_since_access() {
        let rec = make_record("a", 1000, 10);
        assert_eq!(rec.days_since_access(25), 15);
    }

    #[test]
    fn test_record_age_days() {
        let rec = make_record("a", 1000, 10);
        assert_eq!(rec.age_days(50), 40);
    }

    #[test]
    fn test_manager_add_and_get() {
        let mut mgr = AgingManager::new(AgingPolicy::default());
        mgr.add_record(make_record("a", 5000, 0));
        assert_eq!(mgr.record_count(), 1);
        assert!(mgr.get_record("a").is_some());
        assert!(mgr.get_record("b").is_none());
    }

    #[test]
    fn test_manager_total_size() {
        let mut mgr = AgingManager::new(AgingPolicy::default());
        mgr.add_record(make_record("a", 5000, 0));
        mgr.add_record(make_record("b", 3000, 0));
        assert_eq!(mgr.total_size_bytes(), 8000);
    }

    #[test]
    fn test_sweep_idle_transition() {
        let mut mgr = AgingManager::new(AgingPolicy::default());
        mgr.add_record(make_record("a", 5000, 0));
        // Sweep at day 10 (idle_after_days = 7)
        let result = mgr.sweep(10);
        assert_eq!(result.newly_idle, 1);
        assert_eq!(
            mgr.get_record("a").expect("should succeed in test").stage,
            ProxyStage::Idle
        );
    }

    #[test]
    fn test_sweep_auto_archive() {
        let mut policy = AgingPolicy::default();
        policy.auto_archive = true;
        let mut mgr = AgingManager::new(policy);
        mgr.add_record(make_record("a", 5000, 0));
        // Sweep at day 35 (stale_after_days = 30)
        let result = mgr.sweep(35);
        assert_eq!(result.archived, 1);
        assert_eq!(
            mgr.get_record("a").expect("should succeed in test").stage,
            ProxyStage::Archived
        );
    }

    #[test]
    fn test_sweep_auto_delete() {
        let policy = AgingPolicy::strict();
        let mut mgr = AgingManager::new(policy);
        mgr.add_record(make_record("a", 5000, 0));
        // Sweep at day 35 (strict expire_after_days = 30)
        let result = mgr.sweep(35);
        assert_eq!(result.deleted, 1);
        assert!(mgr.get_record("a").is_none());
    }

    #[test]
    fn test_sweep_skips_small_files() {
        let mut policy = AgingPolicy::default();
        policy.min_size_bytes = 10_000;
        let mut mgr = AgingManager::new(policy);
        mgr.add_record(make_record("tiny", 500, 0));
        let result = mgr.sweep(100);
        // Should not transition tiny files
        assert_eq!(result.total_transitions(), 0);
        assert_eq!(
            mgr.get_record("tiny")
                .expect("should succeed in test")
                .stage,
            ProxyStage::Active
        );
    }

    #[test]
    fn test_records_in_stage() {
        let mut mgr = AgingManager::new(AgingPolicy::default());
        mgr.add_record(make_record("a", 5000, 0));
        mgr.add_record(make_record("b", 5000, 0));
        let active = mgr.records_in_stage(ProxyStage::Active);
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_record_access_through_manager() {
        let mut mgr = AgingManager::new(AgingPolicy::default());
        mgr.add_record(make_record("a", 5000, 0));
        assert!(mgr.record_access("a", 5));
        assert!(!mgr.record_access("nonexistent", 5));
        assert_eq!(
            mgr.get_record("a")
                .expect("should succeed in test")
                .access_count,
            1
        );
    }
}
