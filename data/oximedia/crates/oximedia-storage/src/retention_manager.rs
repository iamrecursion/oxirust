#![allow(dead_code)]
//! Object retention and hold management.
//!
//! Implements retention policies, legal holds, and expiration scheduling
//! for objects in the storage layer.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RetentionMode
// ---------------------------------------------------------------------------

/// How a retention lock is enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionMode {
    /// Governance mode — privileged users can override the lock.
    Governance,
    /// Compliance mode — nobody can delete or shorten the lock.
    Compliance,
}

impl std::fmt::Display for RetentionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Governance => write!(f, "governance"),
            Self::Compliance => write!(f, "compliance"),
        }
    }
}

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// A retention policy that can be attached to objects.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Unique policy identifier.
    pub id: String,
    /// Descriptive name.
    pub name: String,
    /// Retention mode.
    pub mode: RetentionMode,
    /// Retention period in seconds.
    pub duration_secs: u64,
    /// Whether the policy is currently active.
    pub active: bool,
}

impl RetentionPolicy {
    /// Create a new retention policy.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        mode: RetentionMode,
        duration_secs: u64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            mode,
            duration_secs,
            active: true,
        }
    }

    /// Deactivate the policy.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Check whether an object created at `created_epoch` has expired
    /// relative to `now_epoch`.
    pub fn is_expired(&self, created_epoch: u64, now_epoch: u64) -> bool {
        now_epoch >= created_epoch + self.duration_secs
    }
}

// ---------------------------------------------------------------------------
// LegalHold
// ---------------------------------------------------------------------------

/// A legal hold that prevents deletion regardless of retention policy.
#[derive(Debug, Clone)]
pub struct LegalHold {
    /// Unique hold identifier.
    pub hold_id: String,
    /// Reason for the hold.
    pub reason: String,
    /// Epoch when the hold was placed.
    pub placed_epoch: u64,
    /// Whether the hold is still active.
    pub active: bool,
}

impl LegalHold {
    /// Create a new active legal hold.
    pub fn new(hold_id: impl Into<String>, reason: impl Into<String>, placed_epoch: u64) -> Self {
        Self {
            hold_id: hold_id.into(),
            reason: reason.into(),
            placed_epoch,
            active: true,
        }
    }

    /// Release the hold.
    pub fn release(&mut self) {
        self.active = false;
    }
}

// ---------------------------------------------------------------------------
// ObjectRetention
// ---------------------------------------------------------------------------

/// Retention state for a single object.
#[derive(Debug, Clone)]
pub struct ObjectRetention {
    /// Object key.
    pub key: String,
    /// Applied retention policy ID (if any).
    pub policy_id: Option<String>,
    /// Epoch when retention started.
    pub retention_start: u64,
    /// Legal holds currently applied.
    pub holds: Vec<LegalHold>,
}

impl ObjectRetention {
    /// Create retention state for an object.
    pub fn new(key: impl Into<String>, policy_id: Option<String>, retention_start: u64) -> Self {
        Self {
            key: key.into(),
            policy_id,
            retention_start,
            holds: Vec::new(),
        }
    }

    /// Add a legal hold.
    pub fn add_hold(&mut self, hold: LegalHold) {
        self.holds.push(hold);
    }

    /// Release a hold by ID.
    pub fn release_hold(&mut self, hold_id: &str) -> bool {
        for h in &mut self.holds {
            if h.hold_id == hold_id && h.active {
                h.release();
                return true;
            }
        }
        false
    }

    /// Whether any active legal hold is in place.
    pub fn has_active_hold(&self) -> bool {
        self.holds.iter().any(|h| h.active)
    }

    /// Count of active holds.
    pub fn active_hold_count(&self) -> usize {
        self.holds.iter().filter(|h| h.active).count()
    }

    /// Whether the object can be deleted given a policy and current time.
    pub fn can_delete(&self, policy: Option<&RetentionPolicy>, now_epoch: u64) -> bool {
        if self.has_active_hold() {
            return false;
        }
        match (policy, &self.policy_id) {
            (Some(p), Some(pid)) if p.id == *pid && p.active => {
                p.is_expired(self.retention_start, now_epoch)
            }
            _ => true,
        }
    }
}

// ---------------------------------------------------------------------------
// RetentionManager
// ---------------------------------------------------------------------------

/// Manages retention policies and per-object retention state.
#[derive(Debug, Default)]
pub struct RetentionManager {
    /// Registered policies.
    policies: HashMap<String, RetentionPolicy>,
    /// Per-object retention state.
    objects: HashMap<String, ObjectRetention>,
}

impl RetentionManager {
    /// Create an empty retention manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a retention policy.
    pub fn add_policy(&mut self, policy: RetentionPolicy) {
        self.policies.insert(policy.id.clone(), policy);
    }

    /// Look up a policy by ID.
    pub fn get_policy(&self, id: &str) -> Option<&RetentionPolicy> {
        self.policies.get(id)
    }

    /// Apply a policy to an object.
    pub fn apply(
        &mut self,
        key: impl Into<String>,
        policy_id: impl Into<String>,
        retention_start: u64,
    ) {
        let key = key.into();
        let pid = policy_id.into();
        self.objects.insert(
            key.clone(),
            ObjectRetention::new(key, Some(pid), retention_start),
        );
    }

    /// Place a legal hold on an object.
    pub fn place_hold(&mut self, key: &str, hold: LegalHold) -> bool {
        if let Some(obj) = self.objects.get_mut(key) {
            obj.add_hold(hold);
            true
        } else {
            false
        }
    }

    /// Release a legal hold on an object.
    pub fn release_hold(&mut self, key: &str, hold_id: &str) -> bool {
        self.objects
            .get_mut(key)
            .is_some_and(|obj| obj.release_hold(hold_id))
    }

    /// Check whether an object can be deleted at `now_epoch`.
    pub fn can_delete(&self, key: &str, now_epoch: u64) -> bool {
        match self.objects.get(key) {
            Some(obj) => {
                let policy = obj
                    .policy_id
                    .as_ref()
                    .and_then(|pid| self.policies.get(pid));
                obj.can_delete(policy, now_epoch)
            }
            None => true,
        }
    }

    /// Number of tracked objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Number of registered policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_mode_display() {
        assert_eq!(RetentionMode::Governance.to_string(), "governance");
        assert_eq!(RetentionMode::Compliance.to_string(), "compliance");
    }

    #[test]
    fn test_retention_policy_expired() {
        let p = RetentionPolicy::new("p1", "30d", RetentionMode::Compliance, 100);
        assert!(!p.is_expired(0, 50));
        assert!(p.is_expired(0, 100));
        assert!(p.is_expired(0, 200));
    }

    #[test]
    fn test_retention_policy_deactivate() {
        let mut p = RetentionPolicy::new("p1", "test", RetentionMode::Governance, 60);
        assert!(p.active);
        p.deactivate();
        assert!(!p.active);
    }

    #[test]
    fn test_legal_hold_release() {
        let mut h = LegalHold::new("h1", "litigation", 1000);
        assert!(h.active);
        h.release();
        assert!(!h.active);
    }

    #[test]
    fn test_object_retention_holds() {
        let mut obj = ObjectRetention::new("obj/1", Some("p1".into()), 0);
        assert!(!obj.has_active_hold());
        obj.add_hold(LegalHold::new("h1", "reason", 100));
        assert!(obj.has_active_hold());
        assert_eq!(obj.active_hold_count(), 1);
        assert!(obj.release_hold("h1"));
        assert!(!obj.has_active_hold());
    }

    #[test]
    fn test_object_retention_release_unknown() {
        let mut obj = ObjectRetention::new("obj/1", None, 0);
        assert!(!obj.release_hold("nonexistent"));
    }

    #[test]
    fn test_can_delete_no_policy() {
        let obj = ObjectRetention::new("obj/1", None, 0);
        assert!(obj.can_delete(None, 1000));
    }

    #[test]
    fn test_can_delete_with_hold() {
        let mut obj = ObjectRetention::new("obj/1", None, 0);
        obj.add_hold(LegalHold::new("h1", "r", 0));
        assert!(!obj.can_delete(None, 1000));
    }

    #[test]
    fn test_can_delete_before_expiry() {
        let p = RetentionPolicy::new("p1", "n", RetentionMode::Compliance, 100);
        let obj = ObjectRetention::new("obj/1", Some("p1".into()), 0);
        assert!(!obj.can_delete(Some(&p), 50));
    }

    #[test]
    fn test_can_delete_after_expiry() {
        let p = RetentionPolicy::new("p1", "n", RetentionMode::Compliance, 100);
        let obj = ObjectRetention::new("obj/1", Some("p1".into()), 0);
        assert!(obj.can_delete(Some(&p), 100));
    }

    #[test]
    fn test_manager_apply_and_delete() {
        let mut mgr = RetentionManager::new();
        mgr.add_policy(RetentionPolicy::new(
            "p1",
            "30d",
            RetentionMode::Governance,
            100,
        ));
        mgr.apply("obj/1", "p1", 0);
        assert!(!mgr.can_delete("obj/1", 50));
        assert!(mgr.can_delete("obj/1", 100));
        assert_eq!(mgr.object_count(), 1);
        assert_eq!(mgr.policy_count(), 1);
    }

    #[test]
    fn test_manager_hold() {
        let mut mgr = RetentionManager::new();
        mgr.add_policy(RetentionPolicy::new(
            "p1",
            "n",
            RetentionMode::Compliance,
            10,
        ));
        mgr.apply("obj/1", "p1", 0);
        mgr.place_hold("obj/1", LegalHold::new("h1", "r", 0));
        assert!(!mgr.can_delete("obj/1", 9999));
        mgr.release_hold("obj/1", "h1");
        assert!(mgr.can_delete("obj/1", 9999));
    }

    #[test]
    fn test_manager_unknown_key() {
        let mgr = RetentionManager::new();
        assert!(mgr.can_delete("nonexistent", 0));
    }
}
