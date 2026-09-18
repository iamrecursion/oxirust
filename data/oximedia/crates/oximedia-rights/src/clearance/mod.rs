//! Clearance tracking module – rights clearance workflow management.

#![allow(dead_code)]

pub mod footage;
pub mod management;
pub mod music;
pub mod sync;
pub mod talent;

pub use footage::FootageClearance;
pub use management::{ClearanceDatabase, ClearanceRecord};
pub use music::MusicClearance;
pub use sync::SyncRights;
pub use talent::TalentRelease;

use serde::{Deserialize, Serialize};

// ── ClearanceStatus ──────────────────────────────────────────────────────────

/// Status of a clearance request in the workflow
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClearanceStatus {
    /// Request submitted, awaiting review
    Requested,
    /// Review is actively underway
    UnderReview,
    /// Clearance has been granted
    Cleared,
    /// Clearance has been denied
    Denied,
    /// Clearance was granted but has since expired
    Expired,
}

impl ClearanceStatus {
    /// Returns `true` when the clearance allows usage (only `Cleared`)
    pub fn is_active(&self) -> bool {
        matches!(self, ClearanceStatus::Cleared)
    }

    /// Returns `true` for states from which no further transitions are expected
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ClearanceStatus::Cleared | ClearanceStatus::Denied | ClearanceStatus::Expired
        )
    }

    /// Convert to a lowercase string tag (for persistence / logging)
    pub fn as_str(&self) -> &str {
        match self {
            ClearanceStatus::Requested => "requested",
            ClearanceStatus::UnderReview => "under_review",
            ClearanceStatus::Cleared => "cleared",
            ClearanceStatus::Denied => "denied",
            ClearanceStatus::Expired => "expired",
        }
    }
}

// ── ClearanceRequest ─────────────────────────────────────────────────────────

/// A single request for rights clearance
#[derive(Debug, Clone)]
pub struct ClearanceRequest {
    /// Unique identifier for this request
    pub id: u64,
    /// Identifier for the asset requiring clearance
    pub asset_id: String,
    /// The entity holding the rights (e.g. publisher, label)
    pub rights_holder: String,
    /// The entity submitting this request
    pub requested_by: String,
    /// Intended usage (e.g. "broadcast", "streaming", "theatrical")
    pub usage_type: String,
    /// Current workflow status
    pub status: ClearanceStatus,
    /// Unix-epoch milliseconds when the request was created
    pub created_ms: u64,
}

impl ClearanceRequest {
    /// Create a new request in the `Requested` state
    pub fn new(
        id: u64,
        asset_id: &str,
        rights_holder: &str,
        requested_by: &str,
        usage_type: &str,
        created_ms: u64,
    ) -> Self {
        Self {
            id,
            asset_id: asset_id.to_string(),
            rights_holder: rights_holder.to_string(),
            requested_by: requested_by.to_string(),
            usage_type: usage_type.to_string(),
            status: ClearanceStatus::Requested,
            created_ms,
        }
    }

    /// How many milliseconds old this request is relative to `now_ms`
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_ms)
    }

    /// Transition to `Cleared`
    pub fn clear(&mut self) {
        self.status = ClearanceStatus::Cleared;
    }

    /// Transition to `Denied`
    pub fn deny(&mut self) {
        self.status = ClearanceStatus::Denied;
    }
}

// ── ClearanceWorkflow ─────────────────────────────────────────────────────────

/// In-memory clearance workflow manager
#[derive(Debug, Default)]
pub struct ClearanceWorkflow {
    /// All clearance requests tracked by this workflow
    pub requests: Vec<ClearanceRequest>,
    /// Counter used to assign unique IDs
    pub next_id: u64,
}

impl ClearanceWorkflow {
    /// Create an empty workflow
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new clearance request and return its assigned ID
    pub fn submit(
        &mut self,
        asset_id: &str,
        holder: &str,
        requester: &str,
        usage: &str,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(ClearanceRequest::new(
            id, asset_id, holder, requester, usage, now_ms,
        ));
        id
    }

    /// Approve the request with `id`; returns `true` if found and updated
    pub fn approve(&mut self, id: u64) -> bool {
        if let Some(req) = self.requests.iter_mut().find(|r| r.id == id) {
            req.clear();
            true
        } else {
            false
        }
    }

    /// Deny the request with `id`; returns `true` if found and updated
    pub fn deny(&mut self, id: u64) -> bool {
        if let Some(req) = self.requests.iter_mut().find(|r| r.id == id) {
            req.deny();
            true
        } else {
            false
        }
    }

    /// Return references to requests that are not yet in a terminal state
    pub fn pending_requests(&self) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| !r.status.is_terminal())
            .collect()
    }

    /// Returns `true` if any request for `asset_id` has status `Cleared`
    pub fn cleared_for(&self, asset_id: &str) -> bool {
        self.requests
            .iter()
            .any(|r| r.asset_id == asset_id && r.status == ClearanceStatus::Cleared)
    }
}

// ── ClearanceType (kept from original module for re-use) ─────────────────────

/// Broad category of clearance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClearanceType {
    /// Music / sync clearance
    Music,
    /// Stock-footage clearance
    Footage,
    /// Talent / model release
    Talent,
    /// Synchronisation rights
    Sync,
}

impl ClearanceType {
    /// Convert to a lowercase string tag
    pub fn as_str(&self) -> &str {
        match self {
            ClearanceType::Music => "music",
            ClearanceType::Footage => "footage",
            ClearanceType::Talent => "talent",
            ClearanceType::Sync => "sync",
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workflow() -> ClearanceWorkflow {
        ClearanceWorkflow::new()
    }

    // ── ClearanceStatus tests ────────────────────────────────────────────────

    #[test]
    fn test_status_cleared_is_active() {
        assert!(ClearanceStatus::Cleared.is_active());
    }

    #[test]
    fn test_status_non_cleared_is_not_active() {
        assert!(!ClearanceStatus::Requested.is_active());
        assert!(!ClearanceStatus::UnderReview.is_active());
        assert!(!ClearanceStatus::Denied.is_active());
        assert!(!ClearanceStatus::Expired.is_active());
    }

    #[test]
    fn test_status_terminal_variants() {
        assert!(ClearanceStatus::Cleared.is_terminal());
        assert!(ClearanceStatus::Denied.is_terminal());
        assert!(ClearanceStatus::Expired.is_terminal());
    }

    #[test]
    fn test_status_non_terminal_variants() {
        assert!(!ClearanceStatus::Requested.is_terminal());
        assert!(!ClearanceStatus::UnderReview.is_terminal());
    }

    // ── ClearanceRequest tests ───────────────────────────────────────────────

    #[test]
    fn test_request_initial_status_is_requested() {
        let req = ClearanceRequest::new(1, "asset-1", "ACME", "Alice", "broadcast", 1_000);
        assert_eq!(req.status, ClearanceStatus::Requested);
    }

    #[test]
    fn test_request_age_ms() {
        let req = ClearanceRequest::new(1, "asset-1", "ACME", "Alice", "broadcast", 1_000);
        assert_eq!(req.age_ms(5_000), 4_000);
    }

    #[test]
    fn test_request_clear_sets_cleared() {
        let mut req = ClearanceRequest::new(1, "asset-1", "ACME", "Alice", "broadcast", 1_000);
        req.clear();
        assert_eq!(req.status, ClearanceStatus::Cleared);
    }

    #[test]
    fn test_request_deny_sets_denied() {
        let mut req = ClearanceRequest::new(1, "asset-1", "ACME", "Alice", "broadcast", 1_000);
        req.deny();
        assert_eq!(req.status, ClearanceStatus::Denied);
    }

    // ── ClearanceWorkflow tests ──────────────────────────────────────────────

    #[test]
    fn test_workflow_submit_assigns_sequential_ids() {
        let mut wf = make_workflow();
        let id0 = wf.submit("a1", "Holder", "User", "streaming", 0);
        let id1 = wf.submit("a2", "Holder", "User", "streaming", 0);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_workflow_approve_returns_true() {
        let mut wf = make_workflow();
        let id = wf.submit("a1", "Holder", "User", "streaming", 0);
        assert!(wf.approve(id));
        assert!(wf.cleared_for("a1"));
    }

    #[test]
    fn test_workflow_approve_unknown_id_returns_false() {
        let mut wf = make_workflow();
        assert!(!wf.approve(999));
    }

    #[test]
    fn test_workflow_deny_returns_true() {
        let mut wf = make_workflow();
        let id = wf.submit("a1", "Holder", "User", "streaming", 0);
        assert!(wf.deny(id));
        assert!(!wf.cleared_for("a1"));
    }

    #[test]
    fn test_workflow_pending_requests_excludes_terminal() {
        let mut wf = make_workflow();
        let id0 = wf.submit("a1", "H", "U", "streaming", 0);
        wf.submit("a2", "H", "U", "streaming", 0);
        wf.approve(id0);
        // Only a2 is still pending
        let pending = wf.pending_requests();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].asset_id, "a2");
    }

    #[test]
    fn test_workflow_cleared_for_false_when_only_denied() {
        let mut wf = make_workflow();
        let id = wf.submit("a1", "H", "U", "streaming", 0);
        wf.deny(id);
        assert!(!wf.cleared_for("a1"));
    }

    #[test]
    fn test_clearance_type_as_str() {
        assert_eq!(ClearanceType::Music.as_str(), "music");
        assert_eq!(ClearanceType::Footage.as_str(), "footage");
        assert_eq!(ClearanceType::Talent.as_str(), "talent");
        assert_eq!(ClearanceType::Sync.as_str(), "sync");
    }
}
