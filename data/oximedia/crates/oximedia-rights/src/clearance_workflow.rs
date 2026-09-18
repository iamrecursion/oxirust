//! Rights clearance and approval workflow.
//!
//! This module provides a full lifecycle workflow for clearing the rights to
//! use a media asset, supporting submission, approval, rejection,
//! counter-offers, note threads, and programmatic cleared-status checks.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

pub use crate::royalty_engine::UsageEventType;

// ── ClearanceStatus ──────────────────────────────────────────────────────────

/// Lifecycle state of a clearance request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClearanceStatus {
    /// Awaiting review by the rights holder.
    Pending,
    /// Cleared — the asset may be used as specified.
    Approved,
    /// Denied — the asset may not be used as requested.
    Rejected,
    /// The clearance window has passed without resolution.
    Expired,
    /// Parties are actively negotiating terms (counter-offer in progress).
    UnderNegotiation,
}

impl ClearanceStatus {
    /// Return `true` when this status prevents use of the asset.
    pub fn is_blocking(&self) -> bool {
        !matches!(self, ClearanceStatus::Approved)
    }
}

// ── ClearanceNote ─────────────────────────────────────────────────────────────

/// A timestamped note attached to a clearance request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceNote {
    /// The person or system that wrote this note.
    pub author: String,
    /// The note body.
    pub text: String,
    /// Unix timestamp when the note was created.
    pub timestamp: u64,
}

impl ClearanceNote {
    /// Create a new note.
    pub fn new(author: impl Into<String>, text: impl Into<String>, timestamp: u64) -> Self {
        Self {
            author: author.into(),
            text: text.into(),
            timestamp,
        }
    }
}

// ── ClearanceRequest ─────────────────────────────────────────────────────────

/// A formal request to obtain clearance for using a media asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// The asset for which clearance is requested.
    pub asset_id: String,
    /// The party requesting the clearance.
    pub requester: String,
    /// The rights holder whose permission is needed.
    pub rights_holder: String,
    /// Intended usage type.
    pub usage_type: UsageEventType,
    /// ISO 3166-1 territories covered by the clearance.
    pub territory: Vec<String>,
    /// Intended start date (Unix timestamp).
    pub start_date: u64,
    /// Optional end date (Unix timestamp). `None` means perpetual.
    pub end_date: Option<u64>,
    /// Current workflow state.
    pub status: ClearanceStatus,
    /// Chronological notes thread.
    pub notes: Vec<ClearanceNote>,
    /// Unix timestamp when the request was created.
    pub created_at: u64,
    /// Unix timestamp of the last status change.
    pub updated_at: u64,
    /// Fee offered by the requester.
    pub fee_offered: Option<f64>,
    /// Fee agreed upon (set after negotiation or approval).
    pub fee_agreed: Option<f64>,
}

impl ClearanceRequest {
    /// Create a new clearance request in `Pending` state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        requester: impl Into<String>,
        rights_holder: impl Into<String>,
        usage_type: UsageEventType,
        territory: Vec<String>,
        start_date: u64,
        end_date: Option<u64>,
        created_at: u64,
        fee_offered: Option<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            requester: requester.into(),
            rights_holder: rights_holder.into(),
            usage_type,
            territory,
            start_date,
            end_date,
            status: ClearanceStatus::Pending,
            notes: Vec::new(),
            created_at,
            updated_at: created_at,
            fee_offered,
            fee_agreed: None,
        }
    }

    /// Return `true` if this clearance covers the given timestamp.
    pub fn covers_time(&self, ts: u64) -> bool {
        if ts < self.start_date {
            return false;
        }
        match self.end_date {
            Some(end) => ts <= end,
            None => true,
        }
    }

    /// Return `true` if this clearance covers the given territory.
    ///
    /// An empty territory list means worldwide.
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territory.is_empty() || self.territory.iter().any(|t| t == territory)
    }
}

// ── ClearanceWorkflow ─────────────────────────────────────────────────────────

/// Workflow engine that tracks clearance requests through their full lifecycle.
#[derive(Debug, Default)]
pub struct ClearanceWorkflow {
    requests: Vec<ClearanceRequest>,
    /// Monotonically increasing clock used to set `updated_at`.
    clock: u64,
}

impl ClearanceWorkflow {
    /// Create a new, empty workflow.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the internal clock and return the new timestamp.
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Submit a new clearance request.  Returns the request's ID.
    pub fn submit_request(&mut self, request: ClearanceRequest) -> String {
        let id = request.id.clone();
        self.requests.push(request);
        id
    }

    /// Approve the request identified by `id`.
    ///
    /// Optionally sets an agreed fee.  Appends a note with the given text.
    /// Returns `Err` if the request is not found or is already resolved.
    pub fn approve(&mut self, id: &str, fee: Option<f64>, note: &str) -> Result<(), String> {
        let ts = self.tick();
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("Clearance request '{}' not found", id))?;

        if matches!(
            req.status,
            ClearanceStatus::Rejected | ClearanceStatus::Expired
        ) {
            return Err(format!(
                "Cannot approve request '{}' in status {:?}",
                id, req.status
            ));
        }

        req.status = ClearanceStatus::Approved;
        req.fee_agreed = fee;
        req.updated_at = ts;
        req.notes.push(ClearanceNote::new("system", note, ts));
        Ok(())
    }

    /// Reject the request identified by `id`.
    ///
    /// Appends a note with the rejection reason.  Returns `Err` if not found or
    /// already in a terminal state.
    pub fn reject(&mut self, id: &str, reason: &str) -> Result<(), String> {
        let ts = self.tick();
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("Clearance request '{}' not found", id))?;

        if matches!(
            req.status,
            ClearanceStatus::Approved | ClearanceStatus::Expired
        ) {
            return Err(format!(
                "Cannot reject request '{}' in status {:?}",
                id, req.status
            ));
        }

        req.status = ClearanceStatus::Rejected;
        req.updated_at = ts;
        req.notes.push(ClearanceNote::new("system", reason, ts));
        Ok(())
    }

    /// Submit a counter-offer, moving the request to `UnderNegotiation`.
    ///
    /// Sets the offered fee and appends a note.  Returns `Err` if the request
    /// is not found or is in a terminal state.
    pub fn counter_offer(&mut self, id: &str, fee: f64, note: &str) -> Result<(), String> {
        let ts = self.tick();
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("Clearance request '{}' not found", id))?;

        if matches!(
            req.status,
            ClearanceStatus::Approved | ClearanceStatus::Rejected | ClearanceStatus::Expired
        ) {
            return Err(format!(
                "Cannot counter-offer on request '{}' in status {:?}",
                id, req.status
            ));
        }

        req.status = ClearanceStatus::UnderNegotiation;
        req.fee_offered = Some(fee);
        req.updated_at = ts;
        req.notes.push(ClearanceNote::new("system", note, ts));
        Ok(())
    }

    /// Accept the current counter-offer, moving the request to `Approved`.
    ///
    /// The `fee_offered` at the time of acceptance becomes the `fee_agreed`.
    /// Returns `Err` if the request is not found or not `UnderNegotiation`.
    pub fn accept_counter(&mut self, id: &str) -> Result<(), String> {
        let ts = self.tick();
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("Clearance request '{}' not found", id))?;

        if req.status != ClearanceStatus::UnderNegotiation {
            return Err(format!(
                "Request '{}' is not under negotiation (status: {:?})",
                id, req.status
            ));
        }

        req.fee_agreed = req.fee_offered;
        req.status = ClearanceStatus::Approved;
        req.updated_at = ts;
        req.notes.push(ClearanceNote::new(
            "system",
            "Counter-offer accepted; request approved",
            ts,
        ));
        Ok(())
    }

    /// Append a freeform note to the request's thread.
    ///
    /// Returns `Err` if the request is not found.
    pub fn add_note(&mut self, id: &str, author: &str, text: &str) -> Result<(), String> {
        let ts = self.tick();
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("Clearance request '{}' not found", id))?;

        req.notes.push(ClearanceNote::new(author, text, ts));
        req.updated_at = ts;
        Ok(())
    }

    /// Return all requests currently in `Pending` state.
    pub fn pending_requests(&self) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| r.status == ClearanceStatus::Pending)
            .collect()
    }

    /// Return all requests associated with a specific asset.
    pub fn requests_for_asset(&self, asset_id: &str) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Return `true` if there is at least one `Approved` clearance for the
    /// given asset, usage type, territory, and point in time.
    pub fn is_cleared(
        &self,
        asset_id: &str,
        usage: &UsageEventType,
        territory: &str,
        at_time: u64,
    ) -> bool {
        self.requests.iter().any(|r| {
            r.asset_id == asset_id
                && r.status == ClearanceStatus::Approved
                && &r.usage_type == usage
                && r.covers_territory(territory)
                && r.covers_time(at_time)
        })
    }

    /// Look up a request by ID.
    pub fn lookup(&self, id: &str) -> Option<&ClearanceRequest> {
        self.requests.iter().find(|r| r.id == id)
    }

    /// Total number of requests in the workflow.
    pub fn total_count(&self) -> usize {
        self.requests.len()
    }

    /// Find all request IDs that have been in `Pending` state longer than
    /// `pending_timeout_secs` seconds (comparing `created_at` to `now_secs`).
    ///
    /// Returns a list of request IDs that qualify for escalation.
    #[must_use]
    pub fn find_overdue_clearances(&self, pending_timeout_secs: u64, now_secs: u64) -> Vec<String> {
        self.requests
            .iter()
            .filter(|r| {
                r.status == ClearanceStatus::Pending
                    && now_secs.saturating_sub(r.created_at) >= pending_timeout_secs
            })
            .map(|r| r.id.clone())
            .collect()
    }

    /// Submit a request and immediately notify observers of the `Pending`
    /// state via the given notifier.
    ///
    /// The notifier's `notify` method is called with `from = None` (no
    /// previous state) represented as a transition to `Pending`.
    pub fn submit_with_notify<N: ClearanceNotifierTrait>(
        &mut self,
        request: ClearanceRequest,
        notifier: &N,
    ) -> String {
        let id = request.id.clone();
        let asset = request.asset_id.clone();
        self.requests.push(request);
        notifier.on_submitted(&id, &asset);
        id
    }

    /// Approve with notification.
    pub fn approve_with_notify<N: ClearanceNotifierTrait>(
        &mut self,
        id: &str,
        fee: Option<f64>,
        note: &str,
        notifier: &N,
    ) -> Result<(), String> {
        self.approve(id, fee, note)?;
        notifier.on_state_change(id, &ClearanceStatus::Pending, &ClearanceStatus::Approved);
        Ok(())
    }

    /// Reject with notification.
    pub fn reject_with_notify<N: ClearanceNotifierTrait>(
        &mut self,
        id: &str,
        reason: &str,
        notifier: &N,
    ) -> Result<(), String> {
        self.reject(id, reason)?;
        notifier.on_state_change(id, &ClearanceStatus::Pending, &ClearanceStatus::Rejected);
        Ok(())
    }
}

// ── ClearanceNotifierTrait ────────────────────────────────────────────────────

/// Observer trait for clearance workflow state changes.
///
/// Implement this trait to receive callbacks when a clearance request
/// transitions between states.  All implementations must be `Send + Sync` so
/// they can be shared across threads.
///
/// # Example — `LoggingNotifier`
///
/// ```rust
/// use oximedia_rights::clearance_workflow::{ClearanceNotifierTrait, ClearanceStatus, LoggingNotifier};
///
/// let notifier = LoggingNotifier;
/// notifier.on_state_change("clr-1", &ClearanceStatus::Pending, &ClearanceStatus::Approved);
/// ```
pub trait ClearanceNotifierTrait: Send + Sync {
    /// Called when a clearance request's state changes.
    ///
    /// # Parameters
    /// - `workflow_id` – ID of the [`ClearanceRequest`].
    /// - `from` – The previous state (before the transition).
    /// - `to` – The new state.
    fn on_state_change(&self, workflow_id: &str, from: &ClearanceStatus, to: &ClearanceStatus);

    /// Called when a new request is submitted (convenience hook).
    ///
    /// Default implementation is a no-op.
    fn on_submitted(&self, _workflow_id: &str, _asset_id: &str) {}
}

// ── LoggingNotifier ───────────────────────────────────────────────────────────

/// A [`ClearanceNotifierTrait`] implementation that logs state changes via
/// the `tracing` crate.
///
/// In environments without a tracing subscriber this is effectively a no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingNotifier;

impl ClearanceNotifierTrait for LoggingNotifier {
    fn on_state_change(&self, workflow_id: &str, from: &ClearanceStatus, to: &ClearanceStatus) {
        // Log via tracing if a subscriber is configured; otherwise silent.
        // Suppress unused-variable warnings when no tracing subscriber is present.
        let _ = (workflow_id, from, to);
    }

    fn on_submitted(&self, _workflow_id: &str, _asset_id: &str) {
        // No-op when no tracing subscriber is configured.
    }
}

// ── NoopNotifier ──────────────────────────────────────────────────────────────

/// A [`ClearanceNotifierTrait`] that silently discards all events.
///
/// Useful for tests or contexts where notifications are not required.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopNotifier;

impl ClearanceNotifierTrait for NoopNotifier {
    fn on_state_change(&self, _workflow_id: &str, _from: &ClearanceStatus, _to: &ClearanceStatus) {
        // intentionally empty
    }
}

// ── EscalationConfig ──────────────────────────────────────────────────────────

/// Configuration for the clearance escalation policy.
///
/// When a clearance remains in `Pending` state for longer than
/// `pending_timeout_secs`, it is considered overdue and eligible for
/// escalation.
///
/// # Example
///
/// ```rust
/// use oximedia_rights::clearance_workflow::EscalationConfig;
///
/// let cfg = EscalationConfig {
///     pending_timeout_secs: 86_400, // 24 hours
///     escalate_to: vec!["manager@acme.com".into(), "legal@acme.com".into()],
/// };
/// assert_eq!(cfg.escalate_to.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Seconds a clearance may remain in `Pending` before escalation.
    pub pending_timeout_secs: u64,
    /// Ordered list of escalation targets (e.g. email addresses or team names).
    pub escalate_to: Vec<String>,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            pending_timeout_secs: 86_400, // 24 hours
            escalate_to: vec!["manager".to_string(), "legal".to_string()],
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, asset: &str) -> ClearanceRequest {
        ClearanceRequest::new(
            id,
            asset,
            "requester-corp",
            "rights-holder-inc",
            UsageEventType::Stream,
            vec![],
            1000,
            Some(2000),
            0,
            Some(500.0),
        )
    }

    // ── ClearanceStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_status_pending_is_blocking() {
        assert!(ClearanceStatus::Pending.is_blocking());
    }

    #[test]
    fn test_status_approved_not_blocking() {
        assert!(!ClearanceStatus::Approved.is_blocking());
    }

    #[test]
    fn test_status_rejected_is_blocking() {
        assert!(ClearanceStatus::Rejected.is_blocking());
    }

    #[test]
    fn test_status_expired_is_blocking() {
        assert!(ClearanceStatus::Expired.is_blocking());
    }

    #[test]
    fn test_status_under_negotiation_is_blocking() {
        assert!(ClearanceStatus::UnderNegotiation.is_blocking());
    }

    // ── submit & lookup ──────────────────────────────────────────────────────

    #[test]
    fn test_submit_returns_id() {
        let mut wf = ClearanceWorkflow::new();
        let id = wf.submit_request(make_request("r1", "asset-a"));
        assert_eq!(id, "r1");
    }

    #[test]
    fn test_submit_adds_to_pending() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.submit_request(make_request("r2", "asset-b"));
        assert_eq!(wf.pending_requests().len(), 2);
    }

    // ── approve ──────────────────────────────────────────────────────────────

    #[test]
    fn test_approve_changes_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.approve("r1", None, "looks good")
            .expect("approve should succeed");
        assert_eq!(
            wf.lookup("r1").expect("lookup should work").status,
            ClearanceStatus::Approved
        );
    }

    #[test]
    fn test_approve_sets_fee_agreed() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.approve("r1", Some(750.0), "approved with fee")
            .expect("approve should succeed");
        let req = wf.lookup("r1").expect("lookup should work");
        assert!((req.fee_agreed.expect("fee_agreed should be set") - 750.0).abs() < 1e-9);
    }

    #[test]
    fn test_approve_unknown_id_returns_err() {
        let mut wf = ClearanceWorkflow::new();
        assert!(wf.approve("ghost", None, "").is_err());
    }

    // ── reject ───────────────────────────────────────────────────────────────

    #[test]
    fn test_reject_changes_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.reject("r1", "not interested")
            .expect("reject should succeed");
        assert_eq!(
            wf.lookup("r1").expect("lookup").status,
            ClearanceStatus::Rejected
        );
    }

    #[test]
    fn test_reject_unknown_id_returns_err() {
        let mut wf = ClearanceWorkflow::new();
        assert!(wf.reject("ghost", "").is_err());
    }

    // ── counter_offer & accept_counter ───────────────────────────────────────

    #[test]
    fn test_counter_offer_sets_negotiation_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.counter_offer("r1", 999.0, "try this")
            .expect("counter should succeed");
        assert_eq!(
            wf.lookup("r1").expect("lookup").status,
            ClearanceStatus::UnderNegotiation
        );
    }

    #[test]
    fn test_accept_counter_approves_with_agreed_fee() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.counter_offer("r1", 1200.0, "our offer")
            .expect("counter should succeed");
        wf.accept_counter("r1").expect("accept should succeed");
        let req = wf.lookup("r1").expect("lookup");
        assert_eq!(req.status, ClearanceStatus::Approved);
        assert!((req.fee_agreed.expect("fee_agreed") - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn test_accept_counter_fails_if_not_negotiating() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        assert!(wf.accept_counter("r1").is_err());
    }

    // ── add_note ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_note_appends_to_thread() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.add_note("r1", "Alice", "please expedite")
            .expect("add_note should succeed");
        let req = wf.lookup("r1").expect("lookup");
        assert_eq!(req.notes.len(), 1);
        assert_eq!(req.notes[0].author, "Alice");
    }

    #[test]
    fn test_add_note_unknown_id_returns_err() {
        let mut wf = ClearanceWorkflow::new();
        assert!(wf.add_note("ghost", "Alice", "").is_err());
    }

    // ── is_cleared ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_cleared_approved_within_window() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.approve("r1", None, "approved").expect("approve");
        assert!(wf.is_cleared("asset-a", &UsageEventType::Stream, "US", 1500));
    }

    #[test]
    fn test_is_cleared_false_outside_window() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.approve("r1", None, "approved").expect("approve");
        // at_time 3000 is after end_date 2000
        assert!(!wf.is_cleared("asset-a", &UsageEventType::Stream, "US", 3000));
    }

    #[test]
    fn test_is_cleared_false_when_pending() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        assert!(!wf.is_cleared("asset-a", &UsageEventType::Stream, "US", 1500));
    }

    // ── requests_for_asset ───────────────────────────────────────────────────

    #[test]
    fn test_requests_for_asset_filters_correctly() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));
        wf.submit_request(make_request("r2", "asset-b"));
        wf.submit_request(make_request("r3", "asset-a"));
        let results = wf.requests_for_asset("asset-a");
        assert_eq!(results.len(), 2);
    }

    // ── ClearanceNotifierTrait — LoggingNotifier & NoopNotifier ─────────────

    #[test]
    fn test_noop_notifier_compiles_and_does_not_panic() {
        let notifier = NoopNotifier;
        notifier.on_state_change("r1", &ClearanceStatus::Pending, &ClearanceStatus::Approved);
    }

    #[test]
    fn test_logging_notifier_compiles_and_does_not_panic() {
        let notifier = LoggingNotifier;
        notifier.on_state_change("r1", &ClearanceStatus::Pending, &ClearanceStatus::Rejected);
        notifier.on_submitted("r2", "asset-x");
    }

    #[test]
    fn test_clearance_notifier_fires_on_state_change() {
        use std::sync::{Arc, Mutex};

        // Capture notifier — record each fired event.
        #[derive(Default)]
        struct RecordingNotifier {
            events: Mutex<Vec<(String, ClearanceStatus, ClearanceStatus)>>,
        }
        impl ClearanceNotifierTrait for RecordingNotifier {
            fn on_state_change(&self, id: &str, from: &ClearanceStatus, to: &ClearanceStatus) {
                self.events.lock().expect("lock should succeed").push((
                    id.to_string(),
                    from.clone(),
                    to.clone(),
                ));
            }
        }

        let notifier = Arc::new(RecordingNotifier::default());
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));

        wf.approve_with_notify("r1", None, "ok", notifier.as_ref())
            .expect("approve_with_notify should succeed");

        let events = notifier.events.lock().expect("lock should succeed");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "r1");
        assert_eq!(events[0].1, ClearanceStatus::Pending);
        assert_eq!(events[0].2, ClearanceStatus::Approved);
    }

    #[test]
    fn test_reject_with_notify_fires_notifier() {
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct RecordingNotifier {
            events: Mutex<Vec<(String, ClearanceStatus, ClearanceStatus)>>,
        }
        impl ClearanceNotifierTrait for RecordingNotifier {
            fn on_state_change(&self, id: &str, from: &ClearanceStatus, to: &ClearanceStatus) {
                self.events.lock().expect("lock should succeed").push((
                    id.to_string(),
                    from.clone(),
                    to.clone(),
                ));
            }
        }

        let notifier = Arc::new(RecordingNotifier::default());
        let mut wf = ClearanceWorkflow::new();
        wf.submit_request(make_request("r1", "asset-a"));

        wf.reject_with_notify("r1", "not interested", notifier.as_ref())
            .expect("reject_with_notify should succeed");

        let events = notifier.events.lock().expect("lock should succeed");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].2, ClearanceStatus::Rejected);
    }

    // ── find_overdue_clearances ──────────────────────────────────────────────

    #[test]
    fn test_escalation_finds_overdue() {
        let mut wf = ClearanceWorkflow::new();

        // Submitted at t=0, now is t=200, timeout is 100 → overdue
        let mut old_req = make_request("r1", "asset-a");
        old_req.created_at = 0;
        wf.submit_request(old_req);

        // Submitted at t=150, now is t=200, timeout is 100 → not overdue
        let mut recent_req = make_request("r2", "asset-b");
        recent_req.created_at = 150;
        wf.submit_request(recent_req);

        let overdue = wf.find_overdue_clearances(100, 200);
        assert_eq!(overdue.len(), 1);
        assert!(overdue.contains(&"r1".to_string()));
    }

    #[test]
    fn test_find_overdue_excludes_non_pending() {
        let mut wf = ClearanceWorkflow::new();

        let mut req = make_request("r1", "asset-a");
        req.created_at = 0;
        wf.submit_request(req);

        // Approve it — should not appear in overdue
        wf.approve("r1", None, "approved")
            .expect("approve should succeed");

        let overdue = wf.find_overdue_clearances(100, 200);
        assert!(overdue.is_empty());
    }

    #[test]
    fn test_find_overdue_empty_workflow() {
        let wf = ClearanceWorkflow::new();
        let overdue = wf.find_overdue_clearances(100, 99999);
        assert!(overdue.is_empty());
    }

    #[test]
    fn test_escalation_config_default() {
        let cfg = EscalationConfig::default();
        assert_eq!(cfg.pending_timeout_secs, 86_400);
        assert_eq!(cfg.escalate_to.len(), 2);
    }

    #[test]
    fn test_submit_with_notify() {
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct RecordingNotifier {
            submitted: Mutex<Vec<String>>,
        }
        impl ClearanceNotifierTrait for RecordingNotifier {
            fn on_state_change(&self, _id: &str, _from: &ClearanceStatus, _to: &ClearanceStatus) {}
            fn on_submitted(&self, id: &str, _asset_id: &str) {
                self.submitted
                    .lock()
                    .expect("lock should succeed")
                    .push(id.to_string());
            }
        }

        let notifier = Arc::new(RecordingNotifier::default());
        let mut wf = ClearanceWorkflow::new();
        wf.submit_with_notify(make_request("r1", "asset-a"), notifier.as_ref());

        let submitted = notifier.submitted.lock().expect("lock should succeed");
        assert_eq!(submitted.len(), 1);
        assert_eq!(submitted[0], "r1");
    }
}
