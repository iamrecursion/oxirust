//! Collaboration notification system.
//!
//! Provides typed notification kinds, per-notification metadata, an inbox
//! that supports delivery, bulk-read, and per-recipient filtering, plus
//! webhook integrations for external notification dispatch.

#![allow(dead_code)]

use std::collections::HashMap;

/// The category of a collaboration notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationKind {
    /// A collaborator mentioned this user.
    Mention,
    /// A collaborator replied to a comment by this user.
    Reply,
    /// The status of a review item changed.
    StatusChange,
    /// Work was assigned to this user.
    Assignment,
    /// A deadline is approaching or was missed.
    Deadline,
}

impl NotificationKind {
    /// Return `true` for high-priority notifications that require immediate attention.
    #[must_use]
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::Mention | Self::Assignment | Self::Deadline)
    }
}

/// A single collaboration notification.
#[derive(Debug, Clone)]
pub struct CollabNotification {
    /// Unique identifier within the inbox.
    pub id: u64,
    /// User id of the intended recipient.
    pub recipient_id: String,
    /// User id of the sender.
    pub sender_id: String,
    /// Category of the notification.
    pub kind: NotificationKind,
    /// Human-readable notification message.
    pub message: String,
    /// Identifier of the resource this notification relates to.
    pub resource_id: String,
    /// Whether the recipient has read this notification.
    pub read: bool,
    /// Creation time in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl CollabNotification {
    /// Mark this notification as read.
    pub fn mark_read(&mut self) {
        self.read = true;
    }

    /// Return the age of this notification relative to `now` (milliseconds).
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// An inbox that stores and manages `CollabNotification`s.
#[derive(Debug, Default)]
pub struct NotificationInbox {
    /// All stored notifications in delivery order.
    pub notifications: Vec<CollabNotification>,
    /// Counter used to assign unique ids.
    pub next_id: u64,
}

impl NotificationInbox {
    /// Create an empty `NotificationInbox`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Deliver a new notification and return its assigned id.
    #[allow(clippy::too_many_arguments)]
    pub fn deliver(
        &mut self,
        recipient: impl Into<String>,
        sender: impl Into<String>,
        kind: NotificationKind,
        message: impl Into<String>,
        resource: impl Into<String>,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.notifications.push(CollabNotification {
            id,
            recipient_id: recipient.into(),
            sender_id: sender.into(),
            kind,
            message: message.into(),
            resource_id: resource.into(),
            read: false,
            timestamp_ms: now_ms,
        });
        id
    }

    /// Mark all notifications addressed to `recipient_id` as read.
    pub fn mark_all_read(&mut self, recipient_id: &str) {
        for n in &mut self.notifications {
            if n.recipient_id == recipient_id {
                n.mark_read();
            }
        }
    }

    /// Return the number of unread notifications for `recipient_id`.
    #[must_use]
    pub fn unread_count(&self, recipient_id: &str) -> usize {
        self.notifications
            .iter()
            .filter(|n| n.recipient_id == recipient_id && !n.read)
            .count()
    }

    /// Return all notifications addressed to `id` in delivery order.
    #[must_use]
    pub fn for_recipient(&self, id: &str) -> Vec<&CollabNotification> {
        self.notifications
            .iter()
            .filter(|n| n.recipient_id == id)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Webhook notification for external integrations
// ─────────────────────────────────────────────────────────────────────────────

/// The HTTP method used for a webhook call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookMethod {
    Post,
    Put,
}

impl std::fmt::Display for WebhookMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
        }
    }
}

/// A registered webhook endpoint that receives collaboration events.
#[derive(Debug, Clone)]
pub struct WebhookEndpoint {
    /// Unique identifier for this webhook.
    pub id: u64,
    /// Destination URL.
    pub url: String,
    /// HTTP method to use.
    pub method: WebhookMethod,
    /// Optional HMAC secret for signing payloads (hex-encoded).
    pub secret: Option<String>,
    /// Extra HTTP headers to include with every request.
    pub headers: HashMap<String, String>,
    /// Filter: only fire for these notification kinds.  Empty = all kinds.
    pub kinds_filter: Vec<NotificationKind>,
    /// Whether this webhook is enabled.
    pub enabled: bool,
}

impl WebhookEndpoint {
    /// Create a new enabled webhook endpoint.
    pub fn new(id: u64, url: impl Into<String>, method: WebhookMethod) -> Self {
        Self {
            id,
            url: url.into(),
            method,
            secret: None,
            headers: HashMap::new(),
            kinds_filter: Vec::new(),
            enabled: true,
        }
    }

    /// Set an HMAC signing secret.
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Add an extra header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Restrict firing to a specific set of notification kinds.
    pub fn with_kinds_filter(mut self, kinds: Vec<NotificationKind>) -> Self {
        self.kinds_filter = kinds;
        self
    }

    /// Disable this webhook.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check whether this webhook should fire for the given notification kind.
    #[must_use]
    pub fn should_fire(&self, kind: NotificationKind) -> bool {
        if !self.enabled {
            return false;
        }
        if self.kinds_filter.is_empty() {
            return true;
        }
        self.kinds_filter.contains(&kind)
    }

    /// Build the serialised JSON payload for a notification.
    ///
    /// Returns a compact JSON string with the notification fields and the
    /// webhook's own URL as metadata.
    #[must_use]
    pub fn build_payload(&self, notification: &CollabNotification) -> String {
        // Produce a minimal hand-crafted JSON payload to avoid pulling in
        // additional serialisation dependencies inside this module.
        let kind_str = format!("{:?}", notification.kind);
        format!(
            r#"{{"webhook_url":"{url}","notification_id":{id},"recipient":"{recip}","sender":"{sender}","kind":"{kind}","message":"{msg}","resource_id":"{res}","timestamp_ms":{ts},"read":{read}}}"#,
            url = self.url,
            id = notification.id,
            recip = notification.recipient_id,
            sender = notification.sender_id,
            kind = kind_str,
            msg = notification.message.replace('"', "\\\""),
            res = notification.resource_id,
            ts = notification.timestamp_ms,
            read = notification.read,
        )
    }
}

/// Delivery record for a single webhook dispatch attempt.
#[derive(Debug, Clone)]
pub struct WebhookDelivery {
    /// The webhook endpoint id.
    pub endpoint_id: u64,
    /// The notification id that triggered this delivery.
    pub notification_id: u64,
    /// The serialised payload that was (or would be) sent.
    pub payload: String,
    /// Whether the delivery was considered successful.
    pub success: bool,
    /// Optional error message if the delivery failed.
    pub error: Option<String>,
    /// Timestamp of the delivery attempt (epoch milliseconds).
    pub attempted_at_ms: u64,
}

/// Registry and dispatcher for webhook endpoints.
///
/// In a real deployment, `dispatch` would issue an HTTP request.  Here it
/// records the delivery attempt into an in-process log so that tests can
/// inspect what would have been sent without any I/O.
#[derive(Debug, Default)]
pub struct WebhookRegistry {
    endpoints: Vec<WebhookEndpoint>,
    next_id: u64,
    /// Delivery log — inspectable in tests.
    pub deliveries: Vec<WebhookDelivery>,
}

impl WebhookRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new webhook endpoint and return its assigned id.
    pub fn register(&mut self, url: impl Into<String>, method: WebhookMethod) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.endpoints.push(WebhookEndpoint::new(id, url, method));
        id
    }

    /// Register a pre-built endpoint.
    pub fn register_endpoint(&mut self, mut endpoint: WebhookEndpoint) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        endpoint.id = id;
        self.endpoints.push(endpoint);
        id
    }

    /// Remove an endpoint by id.  Returns `true` if found.
    pub fn deregister(&mut self, id: u64) -> bool {
        let before = self.endpoints.len();
        self.endpoints.retain(|e| e.id != id);
        self.endpoints.len() != before
    }

    /// Disable an endpoint without removing it.  Returns `true` if found.
    pub fn disable(&mut self, id: u64) -> bool {
        if let Some(ep) = self.endpoints.iter_mut().find(|e| e.id == id) {
            ep.disable();
            true
        } else {
            false
        }
    }

    /// Dispatch a notification to all matching endpoints.
    ///
    /// Each eligible endpoint gets a delivery record appended to
    /// `self.deliveries`.  The function returns the number of endpoints
    /// notified.
    pub fn dispatch(&mut self, notification: &CollabNotification, now_ms: u64) -> usize {
        // Collect into a temporary Vec to avoid borrowing `self.endpoints`
        // and `self.deliveries` at the same time.
        let deliveries: Vec<WebhookDelivery> = self
            .endpoints
            .iter()
            .filter(|ep| ep.should_fire(notification.kind))
            .map(|ep| {
                let payload = ep.build_payload(notification);
                WebhookDelivery {
                    endpoint_id: ep.id,
                    notification_id: notification.id,
                    payload,
                    success: true, // simulated success
                    error: None,
                    attempted_at_ms: now_ms,
                }
            })
            .collect();

        let count = deliveries.len();
        self.deliveries.extend(deliveries);
        count
    }

    /// Return all delivery records for a given endpoint.
    #[must_use]
    pub fn deliveries_for_endpoint(&self, endpoint_id: u64) -> Vec<&WebhookDelivery> {
        self.deliveries
            .iter()
            .filter(|d| d.endpoint_id == endpoint_id)
            .collect()
    }

    /// Return all delivery records for a given notification.
    #[must_use]
    pub fn deliveries_for_notification(&self, notification_id: u64) -> Vec<&WebhookDelivery> {
        self.deliveries
            .iter()
            .filter(|d| d.notification_id == notification_id)
            .collect()
    }

    /// Number of registered endpoints.
    #[must_use]
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Number of recorded delivery attempts.
    #[must_use]
    pub fn delivery_count(&self) -> usize {
        self.deliveries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deliver_one(inbox: &mut NotificationInbox, recipient: &str) -> u64 {
        inbox.deliver(
            recipient,
            "system",
            NotificationKind::Mention,
            "You were mentioned",
            "res-1",
            1_000,
        )
    }

    // ---- NotificationKind ----

    #[test]
    fn test_mention_is_urgent() {
        assert!(NotificationKind::Mention.is_urgent());
    }

    #[test]
    fn test_assignment_is_urgent() {
        assert!(NotificationKind::Assignment.is_urgent());
    }

    #[test]
    fn test_deadline_is_urgent() {
        assert!(NotificationKind::Deadline.is_urgent());
    }

    #[test]
    fn test_reply_not_urgent() {
        assert!(!NotificationKind::Reply.is_urgent());
    }

    #[test]
    fn test_status_change_not_urgent() {
        assert!(!NotificationKind::StatusChange.is_urgent());
    }

    // ---- CollabNotification ----

    #[test]
    fn test_mark_read_sets_flag() {
        let mut inbox = NotificationInbox::new();
        let id = deliver_one(&mut inbox, "alice");
        let n = inbox
            .notifications
            .iter_mut()
            .find(|n| n.id == id)
            .expect("collab test operation should succeed");
        assert!(!n.read);
        n.mark_read();
        assert!(n.read);
    }

    #[test]
    fn test_age_ms_positive() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        let n = &inbox.notifications[0];
        assert_eq!(n.age_ms(3_000), 2_000);
    }

    #[test]
    fn test_age_ms_before_creation_is_zero() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        let n = &inbox.notifications[0];
        // now < timestamp_ms → saturating_sub → 0
        assert_eq!(n.age_ms(500), 0);
    }

    // ---- NotificationInbox ----

    #[test]
    fn test_deliver_returns_incrementing_ids() {
        let mut inbox = NotificationInbox::new();
        let id1 = deliver_one(&mut inbox, "alice");
        let id2 = deliver_one(&mut inbox, "bob");
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_new_inbox_is_empty() {
        let inbox = NotificationInbox::new();
        assert!(inbox.notifications.is_empty());
    }

    #[test]
    fn test_unread_count_increments_on_deliver() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "alice");
        assert_eq!(inbox.unread_count("alice"), 2);
    }

    #[test]
    fn test_unread_count_zero_for_other_user() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        assert_eq!(inbox.unread_count("bob"), 0);
    }

    #[test]
    fn test_mark_all_read_clears_unread() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "alice");
        inbox.mark_all_read("alice");
        assert_eq!(inbox.unread_count("alice"), 0);
    }

    #[test]
    fn test_mark_all_read_only_affects_recipient() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "bob");
        inbox.mark_all_read("alice");
        assert_eq!(inbox.unread_count("alice"), 0);
        assert_eq!(inbox.unread_count("bob"), 1); // bob untouched
    }

    #[test]
    fn test_for_recipient_filters_correctly() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "bob");
        deliver_one(&mut inbox, "alice");
        let alice_notifs = inbox.for_recipient("alice");
        assert_eq!(alice_notifs.len(), 2);
        assert!(alice_notifs.iter().all(|n| n.recipient_id == "alice"));
    }

    #[test]
    fn test_for_recipient_empty_for_unknown() {
        let inbox = NotificationInbox::new();
        assert!(inbox.for_recipient("ghost").is_empty());
    }

    #[test]
    fn test_notification_fields_stored() {
        let mut inbox = NotificationInbox::new();
        inbox.deliver(
            "alice",
            "bob",
            NotificationKind::Reply,
            "Great work!",
            "res-42",
            9_999,
        );
        let n = &inbox.notifications[0];
        assert_eq!(n.recipient_id, "alice");
        assert_eq!(n.sender_id, "bob");
        assert_eq!(n.kind, NotificationKind::Reply);
        assert_eq!(n.message, "Great work!");
        assert_eq!(n.resource_id, "res-42");
        assert_eq!(n.timestamp_ms, 9_999);
        assert!(!n.read);
    }

    // ---- WebhookRegistry ----

    fn make_notification(id: u64, kind: NotificationKind) -> CollabNotification {
        CollabNotification {
            id,
            recipient_id: "alice".to_string(),
            sender_id: "system".to_string(),
            kind,
            message: "test message".to_string(),
            resource_id: "res-1".to_string(),
            read: false,
            timestamp_ms: 1_000,
        }
    }

    #[test]
    fn test_webhook_register_and_count() {
        let mut reg = WebhookRegistry::new();
        reg.register("https://example.com/hook", WebhookMethod::Post);
        reg.register("https://example.com/hook2", WebhookMethod::Put);
        assert_eq!(reg.endpoint_count(), 2);
    }

    #[test]
    fn test_webhook_deregister() {
        let mut reg = WebhookRegistry::new();
        let id = reg.register("https://example.com/hook", WebhookMethod::Post);
        assert!(reg.deregister(id));
        assert_eq!(reg.endpoint_count(), 0);
        assert!(!reg.deregister(id)); // second call returns false
    }

    #[test]
    fn test_webhook_dispatch_fires_all_matching() {
        let mut reg = WebhookRegistry::new();
        reg.register("https://a.example.com/hook", WebhookMethod::Post);
        reg.register("https://b.example.com/hook", WebhookMethod::Post);
        let notif = make_notification(0, NotificationKind::Mention);
        let count = reg.dispatch(&notif, 2_000);
        assert_eq!(count, 2);
        assert_eq!(reg.delivery_count(), 2);
    }

    #[test]
    fn test_webhook_dispatch_respects_kinds_filter() {
        let mut reg = WebhookRegistry::new();
        let ep = WebhookEndpoint::new(0, "https://example.com/hook", WebhookMethod::Post)
            .with_kinds_filter(vec![NotificationKind::Mention]);
        reg.register_endpoint(ep);

        // Mention → should fire
        let notif_mention = make_notification(0, NotificationKind::Mention);
        let count1 = reg.dispatch(&notif_mention, 1_000);
        assert_eq!(count1, 1);

        // Reply → should NOT fire (not in filter)
        let notif_reply = make_notification(1, NotificationKind::Reply);
        let count2 = reg.dispatch(&notif_reply, 2_000);
        assert_eq!(count2, 0);
    }

    #[test]
    fn test_webhook_dispatch_disabled_endpoint_skipped() {
        let mut reg = WebhookRegistry::new();
        let id = reg.register("https://example.com/hook", WebhookMethod::Post);
        reg.disable(id);
        let notif = make_notification(0, NotificationKind::Mention);
        let count = reg.dispatch(&notif, 1_000);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_webhook_build_payload_contains_fields() {
        let ep = WebhookEndpoint::new(0, "https://example.com", WebhookMethod::Post);
        let notif = make_notification(7, NotificationKind::Assignment);
        let payload = ep.build_payload(&notif);
        assert!(payload.contains("\"notification_id\":7"));
        assert!(payload.contains("alice"));
        assert!(payload.contains("Assignment"));
    }

    #[test]
    fn test_webhook_deliveries_for_endpoint() {
        let mut reg = WebhookRegistry::new();
        let id = reg.register("https://example.com/hook", WebhookMethod::Post);
        let notif = make_notification(0, NotificationKind::Deadline);
        reg.dispatch(&notif, 1_000);
        let deliveries = reg.deliveries_for_endpoint(id);
        assert_eq!(deliveries.len(), 1);
        assert_eq!(deliveries[0].endpoint_id, id);
    }

    #[test]
    fn test_webhook_secret_stored() {
        let ep = WebhookEndpoint::new(0, "https://example.com", WebhookMethod::Post)
            .with_secret("my_secret_key");
        assert_eq!(ep.secret, Some("my_secret_key".to_string()));
    }

    #[test]
    fn test_webhook_method_display() {
        assert_eq!(WebhookMethod::Post.to_string(), "POST");
        assert_eq!(WebhookMethod::Put.to_string(), "PUT");
    }
}
