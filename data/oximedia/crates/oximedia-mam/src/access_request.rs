//! Request/approve access workflows for media assets.
//!
//! This module implements a lightweight, self-contained access-request workflow:
//!
//! 1. A user submits an `AccessRequest` specifying the asset(s) and the access
//!    level they need.
//! 2. The system notifies the relevant approver(s).
//! 3. An approver approves, denies, or escalates the request.
//! 4. On approval, an `AccessGrant` is created with an expiry date.
//! 5. Grants can be revoked at any time.
//!
//! All state is kept in-memory in the `AccessRequestManager`; persistence is
//! the responsibility of the caller (serialize via serde).

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Access level
// ---------------------------------------------------------------------------

/// The level of access being requested or granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AccessLevel {
    /// View metadata only.
    MetadataRead,
    /// View proxy/low-res version of the asset.
    ProxyView,
    /// Download original asset.
    OriginalDownload,
    /// Edit asset metadata.
    MetadataEdit,
    /// Full administrative control over the asset.
    Admin,
}

impl AccessLevel {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::MetadataRead => "metadata_read",
            Self::ProxyView => "proxy_view",
            Self::OriginalDownload => "original_download",
            Self::MetadataEdit => "metadata_edit",
            Self::Admin => "admin",
        }
    }

    /// Returns `true` if this level includes the given level.
    #[must_use]
    pub fn includes(&self, other: &AccessLevel) -> bool {
        self >= other
    }
}

// ---------------------------------------------------------------------------
// Request status
// ---------------------------------------------------------------------------

/// Status of an access request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Awaiting review.
    Pending,
    /// Sent to a higher authority for review.
    Escalated,
    /// Request was approved.
    Approved,
    /// Request was denied.
    Denied,
    /// Requester withdrew the request.
    Withdrawn,
    /// Request expired before being acted on.
    Expired,
}

impl RequestStatus {
    /// Returns `true` when the request is still awaiting a decision.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        matches!(self, Self::Pending | Self::Escalated)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Escalated => "escalated",
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::Withdrawn => "withdrawn",
            Self::Expired => "expired",
        }
    }
}

// ---------------------------------------------------------------------------
// Access request
// ---------------------------------------------------------------------------

/// An access request submitted by a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRequest {
    pub id: Uuid,
    /// User requesting access.
    pub requester_id: Uuid,
    /// Display name of the requester.
    pub requester_name: String,
    /// Assets the user wants access to.
    pub asset_ids: Vec<Uuid>,
    /// The level of access requested.
    pub access_level: AccessLevel,
    /// Business justification.
    pub justification: String,
    /// Current status.
    pub status: RequestStatus,
    /// Who reviewed the request (if acted on).
    pub reviewed_by: Option<Uuid>,
    /// Display name of reviewer.
    pub reviewer_name: Option<String>,
    /// When the request was reviewed.
    pub reviewed_at: Option<DateTime<Utc>>,
    /// Reviewer's note (approval message or reason for denial).
    pub reviewer_note: Option<String>,
    /// Requested access duration in days (None = indefinite).
    pub duration_days: Option<u32>,
    /// When the request expires if not acted on (None = no deadline).
    pub request_deadline: Option<DateTime<Utc>>,
    /// When the request was submitted.
    pub created_at: DateTime<Utc>,
    /// When the request was last updated.
    pub updated_at: DateTime<Utc>,
}

impl AccessRequest {
    /// Create a new pending access request.
    #[must_use]
    pub fn new(
        requester_id: Uuid,
        requester_name: impl Into<String>,
        asset_ids: Vec<Uuid>,
        access_level: AccessLevel,
        justification: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            requester_id,
            requester_name: requester_name.into(),
            asset_ids,
            access_level,
            justification: justification.into(),
            status: RequestStatus::Pending,
            reviewed_by: None,
            reviewer_name: None,
            reviewed_at: None,
            reviewer_note: None,
            duration_days: None,
            request_deadline: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Builder: request access for a specific number of days.
    #[must_use]
    pub fn for_duration(mut self, days: u32) -> Self {
        self.duration_days = Some(days);
        self
    }

    /// Builder: set a deadline by which the request must be acted on.
    #[must_use]
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.request_deadline = Some(deadline);
        self
    }

    /// Approve the request.
    pub fn approve(
        &mut self,
        reviewer_id: Uuid,
        reviewer_name: impl Into<String>,
        note: Option<String>,
    ) {
        let now = Utc::now();
        self.status = RequestStatus::Approved;
        self.reviewed_by = Some(reviewer_id);
        self.reviewer_name = Some(reviewer_name.into());
        self.reviewed_at = Some(now);
        self.reviewer_note = note;
        self.updated_at = now;
    }

    /// Deny the request.
    pub fn deny(
        &mut self,
        reviewer_id: Uuid,
        reviewer_name: impl Into<String>,
        reason: impl Into<String>,
    ) {
        let now = Utc::now();
        self.status = RequestStatus::Denied;
        self.reviewed_by = Some(reviewer_id);
        self.reviewer_name = Some(reviewer_name.into());
        self.reviewed_at = Some(now);
        self.reviewer_note = Some(reason.into());
        self.updated_at = now;
    }

    /// Escalate the request to a higher authority.
    pub fn escalate(
        &mut self,
        escalated_by: Uuid,
        escalated_by_name: impl Into<String>,
        note: Option<String>,
    ) {
        let now = Utc::now();
        self.status = RequestStatus::Escalated;
        self.reviewed_by = Some(escalated_by);
        self.reviewer_name = Some(escalated_by_name.into());
        self.reviewed_at = Some(now);
        self.reviewer_note = note;
        self.updated_at = now;
    }

    /// Withdraw the request (by the requester).
    pub fn withdraw(&mut self) {
        self.status = RequestStatus::Withdrawn;
        self.updated_at = Utc::now();
    }

    /// Mark as expired (called by the manager during sweep).
    pub fn expire(&mut self) {
        self.status = RequestStatus::Expired;
        self.updated_at = Utc::now();
    }

    /// Returns `true` if the request has passed its deadline and is still open.
    #[must_use]
    pub fn is_overdue(&self, now: DateTime<Utc>) -> bool {
        self.status.is_open() && self.request_deadline.map(|d| now > d).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Access grant
// ---------------------------------------------------------------------------

/// A confirmed grant of access to an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessGrant {
    pub id: Uuid,
    /// The request that produced this grant.
    pub request_id: Uuid,
    /// User who holds the grant.
    pub grantee_id: Uuid,
    /// Display name of grantee.
    pub grantee_name: String,
    /// Asset the grant covers.
    pub asset_id: Uuid,
    /// Level of access granted.
    pub access_level: AccessLevel,
    /// When this grant expires (None = no expiry).
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether the grant has been explicitly revoked.
    pub revoked: bool,
    /// Who revoked it.
    pub revoked_by: Option<Uuid>,
    /// When it was revoked.
    pub revoked_at: Option<DateTime<Utc>>,
    /// When the grant was created.
    pub created_at: DateTime<Utc>,
}

impl AccessGrant {
    /// Create a new active grant.
    #[must_use]
    pub fn new(
        request_id: Uuid,
        grantee_id: Uuid,
        grantee_name: impl Into<String>,
        asset_id: Uuid,
        access_level: AccessLevel,
        duration_days: Option<u32>,
    ) -> Self {
        let now = Utc::now();
        let expires_at = duration_days.map(|d| now + Duration::days(i64::from(d)));
        Self {
            id: Uuid::new_v4(),
            request_id,
            grantee_id,
            grantee_name: grantee_name.into(),
            asset_id,
            access_level,
            expires_at,
            revoked: false,
            revoked_by: None,
            revoked_at: None,
            created_at: now,
        }
    }

    /// Revoke this grant.
    pub fn revoke(&mut self, revoked_by: Uuid) {
        self.revoked = true;
        self.revoked_by = Some(revoked_by);
        self.revoked_at = Some(Utc::now());
    }

    /// Returns `true` if the grant is currently active (not revoked, not expired).
    #[must_use]
    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        if self.revoked {
            return false;
        }
        self.expires_at.map(|exp| now < exp).unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Access request manager
// ---------------------------------------------------------------------------

/// Manages the full lifecycle of access requests and grants.
#[derive(Debug)]
pub struct AccessRequestManager {
    requests: HashMap<Uuid, AccessRequest>,
    grants: Vec<AccessGrant>,
}

impl AccessRequestManager {
    /// Create a new empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            grants: Vec::new(),
        }
    }

    /// Submit a new access request.
    pub fn submit(&mut self, request: AccessRequest) {
        self.requests.insert(request.id, request);
    }

    /// Get a request by id.
    #[must_use]
    pub fn get_request(&self, id: Uuid) -> Option<&AccessRequest> {
        self.requests.get(&id)
    }

    /// Get a mutable request by id.
    #[must_use]
    pub fn get_request_mut(&mut self, id: Uuid) -> Option<&mut AccessRequest> {
        self.requests.get_mut(&id)
    }

    /// Approve a request and create grants for each asset.
    ///
    /// Returns the ids of the grants created, or `None` if the request
    /// does not exist or is not in an open state.
    pub fn approve(
        &mut self,
        request_id: Uuid,
        reviewer_id: Uuid,
        reviewer_name: impl Into<String>,
        note: Option<String>,
    ) -> Option<Vec<Uuid>> {
        let reviewer_name = reviewer_name.into();
        let request = self.requests.get_mut(&request_id)?;
        if !request.status.is_open() {
            return None;
        }
        let grantee_id = request.requester_id;
        let grantee_name = request.requester_name.clone();
        let asset_ids = request.asset_ids.clone();
        let access_level = request.access_level;
        let duration_days = request.duration_days;

        request.approve(reviewer_id, reviewer_name, note);
        let req_id = request.id;

        let mut grant_ids = Vec::new();
        for asset_id in asset_ids {
            let grant = AccessGrant::new(
                req_id,
                grantee_id,
                grantee_name.clone(),
                asset_id,
                access_level,
                duration_days,
            );
            grant_ids.push(grant.id);
            self.grants.push(grant);
        }
        Some(grant_ids)
    }

    /// Deny a request.
    ///
    /// Returns `true` if the denial succeeded.
    pub fn deny(
        &mut self,
        request_id: Uuid,
        reviewer_id: Uuid,
        reviewer_name: impl Into<String>,
        reason: impl Into<String>,
    ) -> bool {
        if let Some(req) = self.requests.get_mut(&request_id) {
            if req.status.is_open() {
                req.deny(reviewer_id, reviewer_name, reason);
                return true;
            }
        }
        false
    }

    /// Revoke an existing grant.
    pub fn revoke_grant(&mut self, grant_id: Uuid, revoked_by: Uuid) -> bool {
        if let Some(grant) = self.grants.iter_mut().find(|g| g.id == grant_id) {
            grant.revoke(revoked_by);
            return true;
        }
        false
    }

    /// Get all pending/escalated requests.
    #[must_use]
    pub fn open_requests(&self) -> Vec<&AccessRequest> {
        self.requests
            .values()
            .filter(|r| r.status.is_open())
            .collect()
    }

    /// Get all requests by a specific user.
    #[must_use]
    pub fn requests_by_user(&self, user_id: Uuid) -> Vec<&AccessRequest> {
        self.requests
            .values()
            .filter(|r| r.requester_id == user_id)
            .collect()
    }

    /// Active grants for a user on a given asset.
    #[must_use]
    pub fn active_grants_for(&self, user_id: Uuid, asset_id: Uuid) -> Vec<&AccessGrant> {
        let now = Utc::now();
        self.grants
            .iter()
            .filter(|g| g.grantee_id == user_id && g.asset_id == asset_id && g.is_active(now))
            .collect()
    }

    /// Check whether a user has at least the given access level on an asset.
    #[must_use]
    pub fn has_access(&self, user_id: Uuid, asset_id: Uuid, required: AccessLevel) -> bool {
        self.active_grants_for(user_id, asset_id)
            .iter()
            .any(|g| g.access_level.includes(&required))
    }

    /// Expire all overdue open requests.
    ///
    /// Returns the number of requests expired.
    pub fn sweep_expired(&mut self) -> usize {
        let now = Utc::now();
        let mut count = 0;
        for req in self.requests.values_mut() {
            if req.is_overdue(now) {
                req.expire();
                count += 1;
            }
        }
        count
    }

    /// Total number of requests (all statuses).
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Total number of grants (all statuses).
    #[must_use]
    pub fn grant_count(&self) -> usize {
        self.grants.len()
    }
}

impl Default for AccessRequestManager {
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

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    fn make_request() -> AccessRequest {
        AccessRequest::new(
            uid(),
            "Alice",
            vec![uid(), uid()],
            AccessLevel::ProxyView,
            "Need to review footage",
        )
    }

    // --- AccessLevel ---

    #[test]
    fn test_access_level_ordering() {
        assert!(AccessLevel::Admin.includes(&AccessLevel::MetadataRead));
        assert!(AccessLevel::Admin.includes(&AccessLevel::Admin));
        assert!(!AccessLevel::MetadataRead.includes(&AccessLevel::Admin));
    }

    #[test]
    fn test_access_level_labels() {
        assert_eq!(AccessLevel::MetadataRead.label(), "metadata_read");
        assert_eq!(AccessLevel::Admin.label(), "admin");
    }

    // --- RequestStatus ---

    #[test]
    fn test_request_status_is_open() {
        assert!(RequestStatus::Pending.is_open());
        assert!(RequestStatus::Escalated.is_open());
        assert!(!RequestStatus::Approved.is_open());
        assert!(!RequestStatus::Denied.is_open());
        assert!(!RequestStatus::Withdrawn.is_open());
    }

    // --- AccessRequest ---

    #[test]
    fn test_request_creation() {
        let req = make_request();
        assert_eq!(req.status, RequestStatus::Pending);
        assert_eq!(req.asset_ids.len(), 2);
        assert!(req.reviewed_by.is_none());
    }

    #[test]
    fn test_request_approve() {
        let mut req = make_request();
        let reviewer = uid();
        req.approve(
            reviewer,
            "Manager",
            Some("Approved for project X".to_string()),
        );
        assert_eq!(req.status, RequestStatus::Approved);
        assert_eq!(req.reviewed_by, Some(reviewer));
        assert!(req.reviewed_at.is_some());
    }

    #[test]
    fn test_request_deny() {
        let mut req = make_request();
        req.deny(uid(), "Manager", "No valid justification");
        assert_eq!(req.status, RequestStatus::Denied);
        assert!(req.reviewer_note.is_some());
    }

    #[test]
    fn test_request_escalate() {
        let mut req = make_request();
        req.escalate(uid(), "Team Lead", None);
        assert_eq!(req.status, RequestStatus::Escalated);
    }

    #[test]
    fn test_request_withdraw() {
        let mut req = make_request();
        req.withdraw();
        assert_eq!(req.status, RequestStatus::Withdrawn);
        assert!(!req.status.is_open());
    }

    #[test]
    fn test_request_duration_builder() {
        let req = make_request().for_duration(30);
        assert_eq!(req.duration_days, Some(30));
    }

    #[test]
    fn test_request_is_overdue() {
        let past = Utc::now() - Duration::days(2);
        let mut req = make_request().with_deadline(past);
        assert!(req.is_overdue(Utc::now()));
        req.approve(uid(), "M", None);
        assert!(!req.is_overdue(Utc::now())); // No longer open
    }

    // --- AccessGrant ---

    #[test]
    fn test_grant_active_no_expiry() {
        let grant = AccessGrant::new(uid(), uid(), "Alice", uid(), AccessLevel::ProxyView, None);
        assert!(grant.is_active(Utc::now()));
    }

    #[test]
    fn test_grant_expired() {
        let grant = AccessGrant::new(
            uid(),
            uid(),
            "Alice",
            uid(),
            AccessLevel::ProxyView,
            Some(0), // 0 days → already expired
        );
        // expires_at = now + 0 days = now; is_active checks now < exp which will be false
        let future = Utc::now() + Duration::seconds(1);
        assert!(!grant.is_active(future));
    }

    #[test]
    fn test_grant_revoke() {
        let mut grant =
            AccessGrant::new(uid(), uid(), "Alice", uid(), AccessLevel::ProxyView, None);
        assert!(grant.is_active(Utc::now()));
        grant.revoke(uid());
        assert!(grant.revoked);
        assert!(!grant.is_active(Utc::now()));
    }

    // --- AccessRequestManager ---

    #[test]
    fn test_manager_submit_and_retrieve() {
        let mut mgr = AccessRequestManager::new();
        let req = make_request();
        let id = req.id;
        mgr.submit(req);
        assert!(mgr.get_request(id).is_some());
        assert_eq!(mgr.request_count(), 1);
    }

    #[test]
    fn test_manager_approve_creates_grants() {
        let mut mgr = AccessRequestManager::new();
        let req = make_request(); // 2 assets
        let rid = req.id;
        mgr.submit(req);
        let grant_ids = mgr.approve(rid, uid(), "Manager", None);
        assert!(grant_ids.is_some());
        assert_eq!(grant_ids.expect("approve should return grant IDs").len(), 2);
        assert_eq!(mgr.grant_count(), 2);
    }

    #[test]
    fn test_manager_deny() {
        let mut mgr = AccessRequestManager::new();
        let req = make_request();
        let rid = req.id;
        mgr.submit(req);
        assert!(mgr.deny(rid, uid(), "Manager", "Insufficient justification"));
        assert_eq!(
            mgr.get_request(rid).expect("request should exist").status,
            RequestStatus::Denied
        );
    }

    #[test]
    fn test_manager_open_requests() {
        let mut mgr = AccessRequestManager::new();
        let r1 = make_request();
        let r2 = make_request();
        let id1 = r1.id;
        mgr.submit(r1);
        mgr.submit(r2);
        assert_eq!(mgr.open_requests().len(), 2);
        mgr.deny(id1, uid(), "M", "No");
        assert_eq!(mgr.open_requests().len(), 1);
    }

    #[test]
    fn test_manager_has_access() {
        let mut mgr = AccessRequestManager::new();
        let user = uid();
        let asset = uid();
        let req = AccessRequest::new(
            user,
            "Alice",
            vec![asset],
            AccessLevel::OriginalDownload,
            "need it",
        );
        let rid = req.id;
        mgr.submit(req);
        mgr.approve(rid, uid(), "Mgr", None);

        assert!(mgr.has_access(user, asset, AccessLevel::ProxyView));
        assert!(mgr.has_access(user, asset, AccessLevel::OriginalDownload));
        assert!(!mgr.has_access(user, asset, AccessLevel::Admin));
    }

    #[test]
    fn test_manager_revoke_grant() {
        let mut mgr = AccessRequestManager::new();
        let user = uid();
        let asset = uid();
        let req = AccessRequest::new(user, "Alice", vec![asset], AccessLevel::ProxyView, "r");
        let rid = req.id;
        mgr.submit(req);
        let grant_ids = mgr
            .approve(rid, uid(), "M", None)
            .expect("approve should succeed");
        assert!(mgr.has_access(user, asset, AccessLevel::ProxyView));

        mgr.revoke_grant(grant_ids[0], uid());
        assert!(!mgr.has_access(user, asset, AccessLevel::ProxyView));
    }

    #[test]
    fn test_manager_sweep_expired() {
        let mut mgr = AccessRequestManager::new();
        let mut req = make_request();
        let past = Utc::now() - Duration::days(1);
        req.request_deadline = Some(past);
        let id = req.id;
        mgr.submit(req);

        let swept = mgr.sweep_expired();
        assert_eq!(swept, 1);
        assert_eq!(
            mgr.get_request(id).expect("request should exist").status,
            RequestStatus::Expired
        );
    }

    #[test]
    fn test_manager_approve_already_closed() {
        let mut mgr = AccessRequestManager::new();
        let req = make_request();
        let rid = req.id;
        mgr.submit(req);
        mgr.deny(rid, uid(), "M", "No");
        // Second approve attempt should return None
        assert!(mgr.approve(rid, uid(), "M", None).is_none());
    }

    #[test]
    fn test_requests_by_user() {
        let mut mgr = AccessRequestManager::new();
        let user = uid();
        let r1 = AccessRequest::new(user, "Alice", vec![uid()], AccessLevel::ProxyView, "r");
        let r2 = AccessRequest::new(uid(), "Bob", vec![uid()], AccessLevel::MetadataRead, "r");
        mgr.submit(r1);
        mgr.submit(r2);
        assert_eq!(mgr.requests_by_user(user).len(), 1);
    }
}
